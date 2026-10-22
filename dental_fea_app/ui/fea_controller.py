"""
fea_controller.py
Controller bridging the FEA user interface and backend computational engines
(Gmsh mesher, bone material mapping, boundary conditions, and CalculiX runner).
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from vtk.util import numpy_support
from PySide6.QtCore import QObject, Signal

from dental_fea_app.core.state_manager import StateManager
from fea.models import (
    MeshModel,
    MaterialDef,
    MaterialAssignment,
    BoundaryConditionSet,
    TieDef,
    AnalysisConfig,
)
from fea.workers import MeshWorker
from material.bone_mapping import (
    assign_bone_materials,
    titanium_assignment,
    load_preset,
)
from fea.boundary_conditions import (
    fix_by_plane,
    fix_by_region,
    load_from_angle,
    default_load_presets,
    validate_bc,
)
from solver.calculix_writer import write_inp
from solver.runner import CalculixRunner

logger = logging.getLogger(__name__)


class FeaController(QObject):
    """
    Orchestrates FEA operations and syncs results with StateManager.

    Signals:
        meshing_progress(percent, msg): Progress update during volume meshing.
        meshing_finished(report_str): Emitted when meshing completes successfully.
        meshing_error(err_msg): Emitted when meshing encounters an error.

        material_assigned(summary_dict): Emitted when material mapping finishes.
        material_error(err_msg): Emitted on material mapping error.

        bc_validated(issues_list): Emitted with validation result strings (empty if valid).
        bc_error(err_msg): Emitted on boundary condition setup error.

        solver_started(): Emitted when ccx process starts.
        solver_progress(percent, msg): Solver progress update.
        solver_log(line): Real-time stdout/stderr log stream line.
        solver_finished(success, frd_path, msg): Emitted when solver run finishes.
        solver_error(err_msg): Emitted on solver execution error.
    """

    meshing_progress = Signal(int, str)
    meshing_finished = Signal(str)
    meshing_error = Signal(str)

    material_assigned = Signal(dict)
    material_error = Signal(str)

    bc_validated = Signal(list)
    bc_error = Signal(str)

    solver_started = Signal()
    solver_progress = Signal(int, str)
    solver_log = Signal(str)
    solver_finished = Signal(bool, str, str)
    solver_error = Signal(str)

    def __init__(self, state_manager: StateManager, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.mesh_worker: Optional[MeshWorker] = None
        self.solver_runner: Optional[CalculixRunner] = None

    # --- 1. 체적 메쉬 생성 (Meshing) ---

    def start_meshing(self, config: AnalysisConfig, contact_tol_mm: float = 0.3):
        """
        Start 3D tetrahedral meshing in a background worker thread.
        """
        geom = self.state_mgr.fea_geometry
        if geom is None:
            self.meshing_error.emit("기하 형상(FeaGeometry)이 생성되지 않았습니다. '4. 기하 처리' 탭을 먼저 실행하세요.")
            return

        if self.mesh_worker is not None and self.mesh_worker.isRunning():
            self.meshing_error.emit("이미 메쉬 생성이 진행 중입니다.")
            return

        # 메쉬 재생성 시 기존 물성, 경계조건, 해석 결과 등 하류 데이터 즉시 제거
        self.state_mgr.set_material_assignment(None)
        self.state_mgr.set_bc_set(None)
        self.state_mgr.set_last_frd_path(None)
        self.state_mgr.set_bc_fix_box(None)

        self.mesh_worker = MeshWorker(
            fea_geometry=geom,
            config=config,
            contact_tolerance_mm=contact_tol_mm,
            parent=self,
        )
        self.mesh_worker.progress.connect(self.meshing_progress.emit)
        self.mesh_worker.finished.connect(self._on_meshing_finished)
        self.mesh_worker.error.connect(self.meshing_error.emit)
        self.mesh_worker.start()

    def cancel_meshing(self):
        """Request cancellation of running mesh worker."""
        if self.mesh_worker is not None and self.mesh_worker.isRunning():
            self.mesh_worker.cancel()

    def _on_meshing_finished(self, mesh_model: MeshModel, ties: List[TieDef], q_report: str):
        # 신규 체적 메쉬 등록 시 기존 물성/경계조건/해석 결과 확실히 초기화
        self.state_mgr.set_material_assignment(None)
        self.state_mgr.set_bc_set(None)
        self.state_mgr.set_last_frd_path(None)
        self.state_mgr.set_bc_fix_box(None)

        self.state_mgr.set_mesh_model(mesh_model)
        self.state_mgr.set_ties(ties)
        self.meshing_finished.emit(q_report)

    # --- 2. 골-임플란트 물성 할당 (Material Assignment) ---

    def assign_materials(self, preset_path: Optional[str] = None, n_bins: int = 30) -> Optional[MaterialAssignment]:
        """
        Assign heterogeneous bone properties from CT HU sampling and homogeneous titanium to implant.
        """
        mesh_model = self.state_mgr.mesh_model
        if mesh_model is None:
            err = "메쉬 모델이 없습니다. 먼저 [메쉬 생성]을 완료하세요."
            self.material_error.emit(err)
            return None

        bone_part = mesh_model.get_part("bone")
        if bone_part is None:
            err = "메쉬 모델에 'bone' 파트가 존재하지 않습니다."
            self.material_error.emit(err)
            return None

        # 볼륨 배열 추출
        vtk_image = self.state_mgr.vtk_image
        if vtk_image is None:
            err = "로드된 CT 볼륨 영상이 없습니다."
            self.material_error.emit(err)
            return None

        try:
            scalars = vtk_image.GetPointData().GetScalars()
            arr = numpy_support.vtk_to_numpy(scalars)
            dims = self.state_mgr.dimensions  # (nx, ny, nz)
            vol_array = arr.reshape((dims[2], dims[1], dims[0]))  # (nz, ny, nx)

            geom_info = {
                "origin": self.state_mgr.origin,
                "spacing": self.state_mgr.spacing,
                "direction": self.state_mgr.metadata.get("direction", None),
            }

            cortical_mask = None
            if self.state_mgr.segmentation_result is not None:
                cortical_mask = self.state_mgr.segmentation_result.cortical_mask

            preset = load_preset(preset_path)
            preset["n_bins"] = max(2, int(n_bins))

            # 1. 골 조직 물성 매핑 (HU -> Density -> E)
            bone_assign = assign_bone_materials(
                part=bone_part,
                volume_array=vol_array,
                geometry_info=geom_info,
                cortical_mask_array=cortical_mask,
                preset=preset,
            )

            # 2. 임플란트 파트 물성 매핑 (Titanium)
            all_materials: List[MaterialDef] = list(bone_assign.materials)
            elem_mat_dict: Dict[str, np.ndarray] = {
                "bone": bone_assign.element_material["bone"]
            }

            implant_part = mesh_model.get_part("implant")
            if implant_part is not None:
                ti_assign = titanium_assignment(implant_part, preset)
                ti_offset = len(all_materials)
                ti_indices = ti_assign.element_material["implant"] + ti_offset
                elem_mat_dict["implant"] = ti_indices
                all_materials.extend(ti_assign.materials)

            final_assignment = MaterialAssignment(
                element_material=elem_mat_dict,
                materials=all_materials,
            )

            self.state_mgr.set_material_assignment(final_assignment)

            # 요약 통계 계산
            bone_moduli = [m.E for m in bone_assign.materials]
            bone_elem_mats = bone_assign.element_material["bone"]
            elem_E_vals = np.array([bone_assign.materials[idx].E for idx in bone_elem_mats])

            summary = {
                "num_materials": len(all_materials),
                "num_bone_bins": len(bone_assign.materials),
                "e_min": float(np.min(elem_E_vals)),
                "e_max": float(np.max(elem_E_vals)),
                "e_mean": float(np.mean(elem_E_vals)),
            }
            self.material_assigned.emit(summary)
            return final_assignment

        except Exception as e:
            logger.exception("물성 할당 실패")
            self.material_error.emit(f"물성 할당 오류: {str(e)}")
            return None

    # --- 3. 경계조건 설정 (Boundary Conditions) ---

    def setup_boundary_conditions(
        self,
        bottom_z: Optional[float] = None,
        load_preset_key: str = "axial_100N",
        load_mag: Optional[float] = None,
        load_angle: Optional[float] = None,
        load_azimuth: Optional[float] = None,
        top_margin_mm: float = 2.0,
        fix_mode: str = "z_plane",
        fix_box: Optional[Tuple[float, float, float, float, float, float]] = None,
        load_position: Optional[Tuple[float, float, float]] = None,
        load_direction: Optional[Tuple[float, float, float]] = None,
    ) -> List[str]:
        """
        Configure bottom/region fixation constraint and occlusal force load.
        """
        mesh_model = self.state_mgr.mesh_model
        if mesh_model is None:
            err = "메쉬 모델이 없습니다. 먼저 [메쉬 생성]을 완료하세요."
            self.bc_error.emit(err)
            return [err]

        bone_part = mesh_model.get_part("bone")
        if bone_part is None:
            err = "골 조직('bone') 파트가 없습니다."
            self.bc_error.emit(err)
            return [err]

        try:
            # 1. 고정면 설정 (Z-축 평면 또는 육면체 축 지정 영역)
            if fix_mode == "box" and fix_box is not None:
                x0, x1, y0, y1, z0, z1 = fix_box
                def _in_box_predicate(pts: np.ndarray) -> np.ndarray:
                    return (
                        (pts[:, 0] >= x0 - 0.5) & (pts[:, 0] <= x1 + 0.5) &
                        (pts[:, 1] >= y0 - 0.5) & (pts[:, 1] <= y1 + 0.5) &
                        (pts[:, 2] >= z0 - 0.5) & (pts[:, 2] <= z1 + 0.5)
                    )
                fix_dict = fix_by_region(
                    part=bone_part,
                    predicate=_in_box_predicate,
                    dofs=(1, 3),
                    name="FIX_MANDIBLE_REGION",
                )
            else:
                bz = bottom_z if bottom_z is not None else 0.0
                fix_dict = fix_by_plane(
                    part=bone_part,
                    plane_point=(0.0, 0.0, float(bz)),
                    plane_normal=(0.0, 0.0, 1.0),
                    side="below",
                    tol_mm=0.5,
                    dofs=(1, 3),
                    name="FIX_MANDIBLE_BASE",
                )

            # 2. 상단 하중 면/절점 선택
            presets = default_load_presets()
            p_cfg = presets.get(load_preset_key, presets.get("axial_100N", {}))

            mag = float(load_mag if load_mag is not None else p_cfg.get("magnitude_N", 100.0))
            angle = float(load_angle if load_angle is not None else p_cfg.get("tilt_deg", 0.0))
            azimuth = float(load_azimuth if load_azimuth is not None else p_cfg.get("tilt_azimuth_deg", 0.0))

            # 하중 적용 타깃 파트: 임플란트 우선, 없으면 골 상단
            target_part = mesh_model.get_part("implant")
            if target_part is None:
                target_part = bone_part

            nodes = target_part.nodes

            if load_position is not None:
                # 지정된 load_position 근처의 절점 탐색 (반경 3mm 이내 절점, 없으면 가장 가까운 5개 절점)
                pos_arr = np.array(load_position, dtype=np.float64)
                dists = np.linalg.norm(nodes - pos_arr, axis=1)
                near_mask = dists <= max(1.5, float(top_margin_mm))
                if np.any(near_mask):
                    load_node_ids = np.where(near_mask)[0].astype(np.int64)
                else:
                    load_node_ids = np.argsort(dists)[:max(1, min(5, len(nodes)))].astype(np.int64)
            else:
                max_z = float(np.max(nodes[:, 2]))
                top_threshold = max_z - max(0.5, float(top_margin_mm))
                top_node_mask = nodes[:, 2] >= top_threshold
                load_node_ids = np.where(top_node_mask)[0].astype(np.int64)

            if load_direction is not None:
                ld = np.array(load_direction, dtype=np.float64)
                norm_ld = np.linalg.norm(ld)
                if norm_ld > 1e-6:
                    ld = ld / norm_ld
                    fx, fy, fz = tuple(float(f) for f in ld * mag)
                else:
                    fx, fy, fz = load_from_angle(
                        magnitude_N=mag,
                        axis_direction=(0.0, 0.0, -1.0),
                        tilt_deg=angle,
                        tilt_azimuth_deg=azimuth,
                    )
            else:
                fx, fy, fz = load_from_angle(
                    magnitude_N=mag,
                    axis_direction=(0.0, 0.0, -1.0),
                    tilt_deg=angle,
                    tilt_azimuth_deg=azimuth,
                )


            load_dict = {
                "name": "OCCLUSAL_FORCE",
                "part": target_part.name,
                "node_ids": load_node_ids,
                "force_N": (fx, fy, fz),
            }

            bc_set = BoundaryConditionSet(
                fixed_node_sets=[fix_dict],
                loads=[load_dict],
            )
            self.state_mgr.set_bc_set(bc_set)

            # 검증 실행
            issues = validate_bc(mesh_model, bc_set)
            self.bc_validated.emit(issues)
            return issues

        except Exception as e:
            logger.exception("경계조건 설정 실패")
            self.bc_error.emit(f"경계조건 오류: {str(e)}")
            return [str(e)]

    # --- 4. CalculiX .inp 작성 및 해석 실행 ---

    def export_inp(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate and write CalculiX input deck (.inp) file.
        """
        mesh_model = self.state_mgr.mesh_model
        mat_assign = self.state_mgr.material_assignment
        bc_set = self.state_mgr.bc_set
        ties = self.state_mgr.ties

        if mesh_model is None:
            self.solver_error.emit("메쉬 모델이 생성되지 않았습니다.")
            return None
        if mat_assign is None:
            self.solver_error.emit("물성이 할당되지 않았습니다.")
            return None
        if bc_set is None:
            self.solver_error.emit("경계조건이 정의되지 않았습니다.")
            return None

        if not output_path:
            import dental_fea_app
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(dental_fea_app.__file__)))
            saved_dir = os.path.join(base_dir, "saved")
            os.makedirs(saved_dir, exist_ok=True)
            output_path = os.path.join(saved_dir, "dental_fea_job.inp")

        try:
            # write_inp expects Dict[str, MaterialAssignment]
            if isinstance(mat_assign, MaterialAssignment):
                mat_dict = {}
                for p_name in mat_assign.element_material.keys():
                    mat_dict[p_name] = MaterialAssignment(
                        element_material={p_name: mat_assign.element_material[p_name]},
                        materials=mat_assign.materials,
                    )
            elif isinstance(mat_assign, dict):
                mat_dict = mat_assign
            else:
                mat_dict = {}

            write_inp(
                path=output_path,
                model=mesh_model,
                materials=mat_dict,
                bc=bc_set,
                ties=ties or [],
                title="dental_fea",
            )
            return output_path
        except Exception as e:
            logger.exception(".inp 파일 작성 실패")
            self.solver_error.emit(f".inp 파일 저장 실패: {str(e)}")
            return None


    def start_solver(
        self,
        inp_path: Optional[str] = None,
        ccx_path: Optional[str] = None,
        n_threads: int = 4,
    ):
        """
        Run CalculiX solver asynchronously.
        """
        if inp_path is None or not os.path.exists(inp_path):
            inp_path = self.export_inp(inp_path)
            if not inp_path:
                return

        if self.solver_runner is not None and self.solver_runner.is_running:
            self.solver_error.emit("이미 해석이 실행 중입니다.")
            return

        self.solver_runner = CalculixRunner(
            ccx_path=ccx_path,
            n_threads=n_threads,
            parent=self,
        )
        self.solver_runner.started.connect(self.solver_started.emit)
        self.solver_runner.progress.connect(self.solver_progress.emit)
        self.solver_runner.log_line.connect(self.solver_log.emit)
        self.solver_runner.finished.connect(self._on_solver_finished)
        self.solver_runner.error.connect(self.solver_error.emit)

        self.solver_runner.run(inp_path)

    def cancel_solver(self):
        """Cancel running CalculiX solver process."""
        if self.solver_runner is not None:
            self.solver_runner.cancel()

    def _on_solver_finished(self, success: bool, frd_path: str, msg: str):
        if success and frd_path:
            self.state_mgr.set_last_frd_path(frd_path)
        self.solver_finished.emit(success, frd_path, msg)
