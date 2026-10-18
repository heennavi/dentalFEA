"""
geometry_worker.py
FEA 기하 처리 비동기 백그라운드 연산 워커 스레드 (QThread).
- 임플란트 주변 ROI Bounding Box 산출
- 피질골/수질골 서브볼륨 마스크 표면 메쉬화 (Marching Cubes / Flying Edges + Sinc Smoothing)
- Boolean 차집합 및 인터페이스 분할 (manifold3d)
- 체적 보존 및 접촉 면적 분석 리포트 생성
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import vtk
import trimesh
from PySide6.QtCore import QThread, Signal

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.geometry.surface_extract import mask_to_surface, _vtk_polydata_to_trimesh
from dental_fea_app.geometry.boolean_ops import build_fea_geometry, FeaGeometry
from dental_fea_app.placement.implant_placement_manager import PlacedImplant
from dental_fea_app.placement.implant_generator import ImplantLibraryManager


def get_implant_world_mesh(state_mgr: StateManager) -> trimesh.Trimesh:
    """
    StateManager에 등록된 식립 임플란트(PlacedImplant 목록 또는 단일 ImplantModel)로부터
    World LPS mm 좌표계의 단일 trimesh.Trimesh 객체를 추출합니다.

    [ROI 필터링 규칙]
    ROI boundary가 설정되어 있는 경우, 해당 ROI 영역 내에 위치하거나 걸치는 임플란트만
    선별하여 기하 처리 및 Boolean 연산 기준으로 적용합니다.
    """
    mgr = getattr(state_mgr, "implant_placement_mgr", None)
    meshes: List[trimesh.Trimesh] = []
    roi_b = getattr(state_mgr, "roi_bounds", None)

    def is_mesh_in_roi(m: trimesh.Trimesh, roi: tuple) -> bool:
        if roi is None or len(roi) != 6:
            return True
        xmin, xmax, ymin, ymax, zmin, zmax = roi
        b_min, b_max = m.bounds
        return (
            b_max[0] >= xmin and b_min[0] <= xmax and
            b_max[1] >= ymin and b_min[1] <= ymax and
            b_max[2] >= zmin and b_min[2] <= zmax
        )

    if mgr and mgr.implants:
        for imp in mgr.implants.values():
            poly = imp.raw_polydata
            if poly is None and imp.spec:
                poly = ImplantLibraryManager.load_implant_polydata(imp.spec)
            if poly is None:
                continue

            trans = vtk.vtkTransform()
            trans.PostMultiply()
            half_len = imp.spec.length_mm / 2.0
            trans.Translate(0.0, 0.0, -half_len)
            trans.RotateZ(imp.orientation[0])
            trans.RotateX(imp.orientation[1])
            trans.RotateY(imp.orientation[2])
            trans.Translate(imp.position[0], imp.position[1], imp.position[2])

            tf_filter = vtk.vtkTransformPolyDataFilter()
            tf_filter.SetTransform(trans)
            tf_filter.SetInputData(poly)
            tf_filter.Update()
            tm = _vtk_polydata_to_trimesh(tf_filter.GetOutput())
            if tm and len(tm.faces) > 0:
                # ROI가 설정되어 있다면 ROI 내부/교차 임플란트만 포함
                if is_mesh_in_roi(tm, roi_b):
                    meshes.append(tm)

    elif state_mgr.implant_model:
        model = state_mgr.implant_model
        poly = model.get_transformed_polydata()
        if poly is not None:
            tm = _vtk_polydata_to_trimesh(poly)
            if tm and len(tm.faces) > 0:
                if is_mesh_in_roi(tm, roi_b):
                    meshes.append(tm)

    if not meshes:
        if roi_b is not None:
            raise ValueError("설정된 ROI 영역 내에 식립된 임플란트가 존재하지 않습니다. ROI 영역을 조정하거나 임플란트를 확인해 주십시오.")
        else:
            raise ValueError("식립된 임플란트가 없습니다. 먼저 '3. 임플란트' 탭에서 임플란트를 식립해 주십시오.")

    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


class GeometryWorker(QThread):
    """FEA 기하 생성 비동기 처리 워커"""

    # (step, total_steps, message)
    progress_updated = Signal(int, int, str)
    finished_success = Signal(object)  # FeaGeometry
    failed = Signal(str)

    def __init__(
        self,
        state_manager: StateManager,
        roi_margin_mm: float = 12.0,
        smooth_iterations: int = 15,
        smooth_passband: float = 0.10,
        target_edge_mm: Optional[float] = 1.0,
        undersize_mm: float = 0.0,
        parent: Optional[Any] = None
    ):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.roi_margin_mm = float(roi_margin_mm)
        self.smooth_iterations = int(smooth_iterations)
        self.smooth_passband = float(smooth_passband)
        self.target_edge_mm = float(target_edge_mm) if target_edge_mm and target_edge_mm > 0 else None
        self.undersize_mm = float(undersize_mm)
        self._is_aborted = False

    def abort(self):
        self._is_aborted = True

    def run(self):
        try:
            # 1. 입력 유효성 검사
            self.progress_updated.emit(1, 4, "1/4: 임플란트 Bounding Box 및 ROI 마스크 영역 계산 중...")
            if self._is_aborted:
                return

            seg_res = self.state_mgr.segmentation_result
            if seg_res is None or seg_res.cortical_mask is None or seg_res.cancellous_mask is None:
                raise ValueError("뼈 분할 결과가 없습니다. 먼저 '2. 분할' 탭에서 분할을 실행해 주십시오.")

            vtk_img = self.state_mgr.vtk_image
            if vtk_img is None:
                raise ValueError("영상 볼륨 데이터(CT)가 로드되지 않았습니다.")

            # 2. 임플란트 메쉬 및 ROI 산출
            implant_mesh = get_implant_world_mesh(self.state_mgr)
            imp_bounds = implant_mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]

            origin = self.state_mgr.origin
            spacing = self.state_mgr.spacing
            dims = self.state_mgr.dimensions  # (nx, ny, nz)

            # 3D 뷰에서 ROI를 설정한 경우(roi_bounds), 해당 구간을 우선 적용
            if self.state_mgr.roi_bounds is not None:
                roi_min_x, roi_max_x, roi_min_y, roi_max_y, roi_min_z, roi_max_z = self.state_mgr.roi_bounds
                roi_world_min = np.array([roi_min_x, roi_min_y, roi_min_z])
                roi_world_max = np.array([roi_max_x, roi_max_y, roi_max_z])
            else:
                roi_world_min = imp_bounds[0] - self.roi_margin_mm
                roi_world_max = imp_bounds[1] + self.roi_margin_mm

            # 월드 좌표 -> 복셀 인덱스 i, j, k
            imin = max(0, int(np.floor((roi_world_min[0] - origin[0]) / spacing[0])))
            imax = min(dims[0] - 1, int(np.ceil((roi_world_max[0] - origin[0]) / spacing[0])))
            jmin = max(0, int(np.floor((roi_world_min[1] - origin[1]) / spacing[1])))
            jmax = min(dims[1] - 1, int(np.ceil((roi_world_max[1] - origin[1]) / spacing[1])))
            kmin = max(0, int(np.floor((roi_world_min[2] - origin[2]) / spacing[2])))
            kmax = min(dims[2] - 1, int(np.ceil((roi_world_max[2] - origin[2]) / spacing[2])))

            roi_ijk = (imin, imax, jmin, jmax, kmin, kmax)

            # 3. 피질골 표면 메쉬 추출
            self.progress_updated.emit(2, 4, "2/4: 피질골 및 수질골 표면 메쉬 추출 및 부피보존 스무딩 중...")
            if self._is_aborted:
                return

            cortical_mesh = mask_to_surface(
                mask_data=seg_res.cortical_mask,
                reference_vtk=vtk_img,
                spacing=spacing,
                origin=origin,
                roi_ijk=roi_ijk,
                smooth_iterations=self.smooth_iterations,
                smooth_passband=self.smooth_passband,
                target_edge_mm=self.target_edge_mm,
                keep_largest_component=False
            )

            if self._is_aborted:
                return

            # 4. 수질골 표면 메쉬 추출
            cancellous_mesh = mask_to_surface(
                mask_data=seg_res.cancellous_mask,
                reference_vtk=vtk_img,
                spacing=spacing,
                origin=origin,
                roi_ijk=roi_ijk,
                smooth_iterations=self.smooth_iterations,
                smooth_passband=self.smooth_passband,
                target_edge_mm=self.target_edge_mm,
                keep_largest_component=False
            )

            # 5. Boolean 차집합 및 인터페이스 분할
            self.progress_updated.emit(3, 4, "3/4: 임플란트 식립부 Boolean 차집합 및 인터페이스 정리 중...")
            if self._is_aborted:
                return

            fea_geometry = build_fea_geometry(
                cortical_mesh=cortical_mesh,
                cancellous_mesh=cancellous_mesh,
                implant_mesh=implant_mesh,
                undersize_mm=self.undersize_mm
            )

            if self._is_aborted:
                return

            self.progress_updated.emit(4, 4, "4/4: 접촉 면적 분석 및 FEA 솔리드 구축 완료!")
            self.finished_success.emit(fea_geometry)

        except Exception as e:
            self.failed.emit(str(e))
