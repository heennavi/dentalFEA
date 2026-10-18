"""
surface_extract.py
피질골/수질골 마스크를 FEA에 쓸 수 있는 깨끗한 삼각형 표면 메쉬로 변환하는 순수 로직 모듈.

[원칙]
- 순수 기하 및 수학 연산 모듈로 PySide6를 일절 import하지 않습니다.
- 모든 기하 데이터는 DICOM LPS World 좌표계(mm)를 따릅니다.
- 단위: 길이(mm), 면적(mm^2), 부피(mm^3).
"""

from typing import Tuple, Dict, Any, Optional, Union
import warnings
import numpy as np
import vtk
from vtk.util import numpy_support
import trimesh

from dental_fea_app.segmentation.cortical_cancellous import mask_to_vtk_image


def mask_to_surface(
    mask_data: Union[vtk.vtkImageData, np.ndarray],
    spacing: Optional[Tuple[float, float, float]] = None,
    origin: Optional[Tuple[float, float, float]] = None,
    direction: Optional[Tuple[float, ...]] = None,
    smooth_iterations: int = 15,
    smooth_passband: float = 0.1,
    target_edge_mm: Optional[float] = None,
    roi_ijk: Optional[Union[Tuple[int, ...], list]] = None,
    keep_largest_component: bool = True,
    reference_vtk: Optional[vtk.vtkImageData] = None,
) -> trimesh.Trimesh:
    """
    3D 이진 마스크를 깨끗하고 균일한 삼각형 표면 메쉬(trimesh.Trimesh, world 좌표 mm)로 변환합니다.

    [처리 순서]
    1. vtkFlyingEdges3D로 등가 표면 추출 (iso=0.5 또는 127.5)
    2. vtkWindowedSincPolyDataFilter로 스무딩 (계단 아티팩트 제거, 부피 수축 최소화)
    3. target_edge_mm가 지정되면 vtkQuadricDecimation으로 삼각형 크기 감축
    4. 법선 정리, 중복 정점 병합, 퇴화 삼각형 제거
    5. 가장 큰 연결 성분 유지 옵션 (keep_largest_component)
    6. watertight 검사 및 필요시 구멍 메우기 시도, 실패 시 상세 경고

    Args:
        mask_data: 3D 마스크 (vtk.vtkImageData 또는 3D numpy ndarray (nz, ny, nx))
        spacing: 복셀 간격 (sx, sy, sz) in mm. numpy 마스크일 때 사용. (기본값: (1.0, 1.0, 1.0))
        origin: 볼륨 원점 (ox, oy, oz) in mm. numpy 마스크일 때 사용. (기본값: (0.0, 0.0, 0.0))
        direction: 9개 원소 방향 행렬. numpy 마스크일 때 사용.
        smooth_iterations: Windowed Sinc 스무딩 반복 횟수 (기본값 15)
        smooth_passband: Windowed Sinc 패스밴드 주파수 (기본값 0.1, 낮을수록 부드러움)
        target_edge_mm: 목표 삼각형 변 길이 (mm). 지정 시 데시메이션으로 삼각형 크기 조절.
        roi_ijk: 관심영역 Bounding Box 인덱스 (imin, imax, jmin, jmax, kmin, kmax).
                 지정 시 해당 영역만 추출하여 고속 변환 수행.
        keep_largest_component: True일 경우 가장 큰 단일 연결 덩어리만 유지 (기본값 True)
        reference_vtk: 원본 영상 vtkImageData. numpy 마스크 전달 시 geometry 일치용.

    Returns:
        trimesh.Trimesh: World 좌표계(mm) 기준의 표면 메쉬 객체
    """
    # -------------------------------------------------------------
    # 0. 입력 데이터 정규화 및 vtkImageData 준비
    # -------------------------------------------------------------
    if isinstance(mask_data, np.ndarray):
        spc = spacing if spacing is not None else (1.0, 1.0, 1.0)
        org = origin if origin is not None else (0.0, 0.0, 0.0)

        # ROI가 주어진 경우 numpy 레벨에서 조기 슬라이싱으로 메모리/연산량 최적화
        if roi_ijk is not None and len(roi_ijk) == 6:
            nz, ny, nx = mask_data.shape
            imin, imax, jmin, jmax, kmin, kmax = roi_ijk
            imin = max(0, min(nx - 1, int(imin)))
            imax = max(imin, min(nx - 1, int(imax)))
            jmin = max(0, min(ny - 1, int(jmin)))
            jmax = max(jmin, min(ny - 1, int(jmax)))
            kmin = max(0, min(nz - 1, int(kmin)))
            kmax = max(kmin, min(nz - 1, int(kmax)))

            sub_mask = mask_data[kmin : kmax + 1, jmin : jmax + 1, imin : imax + 1]
            sub_origin = (
                org[0] + imin * spc[0],
                org[1] + jmin * spc[1],
                org[2] + kmin * spc[2],
            )
            vtk_image = mask_to_vtk_image(
                sub_mask,
                spacing=spc,
                origin=sub_origin,
                direction=direction,
                foreground_value=255,
            )
        else:
            vtk_image = mask_to_vtk_image(
                mask_data,
                reference_vtk=reference_vtk,
                spacing=spc,
                origin=org,
                direction=direction,
                foreground_value=255,
            )

    elif isinstance(mask_data, vtk.vtkImageData):
        if roi_ijk is not None and len(roi_ijk) == 6:
            # vtkExtractVOI 필터로 서브 볼륨 슬라이싱
            dims = mask_data.GetDimensions()
            imin, imax, jmin, jmax, kmin, kmax = roi_ijk
            imin = max(0, min(dims[0] - 1, int(imin)))
            imax = max(imin, min(dims[0] - 1, int(imax)))
            jmin = max(0, min(dims[1] - 1, int(jmin)))
            jmax = max(jmin, min(dims[1] - 1, int(jmax)))
            kmin = max(0, min(dims[2] - 1, int(kmin)))
            kmax = max(kmin, min(dims[2] - 1, int(kmax)))

            extract_voi = vtk.vtkExtractVOI()
            extract_voi.SetInputData(mask_data)
            extract_voi.SetVOI(imin, imax, jmin, jmax, kmin, kmax)
            extract_voi.Update()
            vtk_image = extract_voi.GetOutput()
        else:
            vtk_image = mask_data
    else:
        raise TypeError(f"지원하지 않는 마스크 데이터 형식입니다: {type(mask_data)}")

    # 스칼라 데이터 검사
    scalars = vtk_image.GetPointData().GetScalars()
    if scalars is None or scalars.GetNumberOfTuples() == 0:
        return trimesh.Trimesh()

    scalar_range = vtk_image.GetScalarRange()
    max_val = scalar_range[1]
    if max_val <= 0:
        # 전경 마스크가 전혀 없음
        return trimesh.Trimesh()

    iso_value = 127.5 if max_val > 1.0 else 0.5

    # -------------------------------------------------------------
    # 1. vtkFlyingEdges3D 표면 추출
    # -------------------------------------------------------------
    flying_edges = vtk.vtkFlyingEdges3D()
    flying_edges.SetInputData(vtk_image)
    flying_edges.SetValue(0, iso_value)
    flying_edges.ComputeNormalsOn()
    flying_edges.ComputeGradientsOff()
    flying_edges.ComputeScalarsOff()
    flying_edges.Update()

    raw_poly = flying_edges.GetOutput()
    if raw_poly.GetNumberOfPoints() == 0 or raw_poly.GetNumberOfPolys() == 0:
        return trimesh.Trimesh()

    # 스무딩 전 부피 측정 (부피 변화율 감시용)
    vol_pre = _compute_polydata_volume(raw_poly)

    # -------------------------------------------------------------
    # 2. vtkWindowedSincPolyDataFilter 스무딩
    # -------------------------------------------------------------
    if smooth_iterations > 0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(raw_poly)
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetPassBand(smooth_passband)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.Update()
        smoothed_poly = smoother.GetOutput()
    else:
        smoothed_poly = raw_poly

    # 스무딩 후 부피 변화율 검증 및 경고 (3% 초과 시)
    vol_post = _compute_polydata_volume(smoothed_poly)
    if vol_pre > 1e-3 and vol_post > 1e-3:
        vol_change_ratio = abs(vol_post - vol_pre) / vol_pre
        if vol_change_ratio > 0.03:
            warnings.warn(
                f"스무딩 후 부피가 원본 대비 {vol_change_ratio * 100:.2f}% 변했습니다. "
                f"(기준치: 3% 초과, 원본: {vol_pre:.2f} mm^3 -> 스무딩: {vol_post:.2f} mm^3)",
                UserWarning,
                stacklevel=2,
            )

    current_poly = smoothed_poly

    # -------------------------------------------------------------
    # 3. target_edge_mm 삼각형 크기 맞춤 (Decimation)
    # -------------------------------------------------------------
    if target_edge_mm is not None and target_edge_mm > 0:
        current_poly = _decimate_to_target_edge(current_poly, target_edge_mm)

    # -------------------------------------------------------------
    # 4. 법선 정리, 중복 정점 병합, 퇴화 삼각형 제거
    # -------------------------------------------------------------
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(current_poly)
    cleaner.PointMergingOn()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.Update()
    cleaned_poly = cleaner.GetOutput()

    # -------------------------------------------------------------
    # 5. 가장 큰 연결 성분 유지 (keep_largest_component)
    # -------------------------------------------------------------
    if keep_largest_component and cleaned_poly.GetNumberOfPolys() > 0:
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(cleaned_poly)
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        cleaned_poly = conn.GetOutput()

    # 법선 벡터 일관성(Consistency) 및 바깥 방향(AutoOrient) 재정리
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(cleaned_poly)
    normals_filter.ConsistencyOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.SplittingOff()
    normals_filter.Update()
    final_poly = normals_filter.GetOutput()

    # -------------------------------------------------------------
    # 6. vtkPolyData -> trimesh.Trimesh 변환 및 Watertight 검사/복구
    # -------------------------------------------------------------
    mesh = _vtk_polydata_to_trimesh(final_poly)

    if len(mesh.faces) == 0:
        return mesh

    # 법선 방향 바로잡기 (부피가 음수이면 외향 법선으로 뒤집기)
    if mesh.volume < 0:
        mesh.invert()

    # Watertight 검사 및 구멍 메우기 시도
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_inversion(mesh)
        except Exception as e:
            warnings.warn(f"메쉬 구멍 메우기 중 예외 발생: {e}", UserWarning, stacklevel=2)

        if not mesh.is_watertight:
            warnings.warn(
                "메쉬가 watertight(닫힌 매니폴드) 상태가 아니며 구멍 메우기에 실패했습니다. "
                "FEA 체적 메싱(Volume meshing) 시 경계 오류가 발생할 수 있습니다.",
                UserWarning,
                stacklevel=2,
            )

    return mesh


def mesh_report(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    삼각형 표면 메쉬에 대한 형상 및 품질 요약 보고서를 생성합니다.

    Args:
        mesh: trimesh.Trimesh 객체

    Returns:
        dict:
            - n_vertices (int): 정점 수
            - n_faces (int): 삼각형 면 수
            - volume (float): 체적 (mm^3)
            - surface_area (float): 표면적 (mm^2)
            - is_watertight (bool): 방수/닫힌 매니폴드 여부
            - min_edge_length (float): 최소 삼각형 변 길이 (mm)
            - mean_edge_length (float): 평균 삼각형 변 길이 (mm)
            - max_edge_length (float): 최대 삼각형 변 길이 (mm)
            - is_self_intersecting (bool): 메쉬 자기교차 여부
    """
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return {
            "n_vertices": 0,
            "n_faces": 0,
            "volume": 0.0,
            "surface_area": 0.0,
            "is_watertight": False,
            "min_edge_length": 0.0,
            "mean_edge_length": 0.0,
            "max_edge_length": 0.0,
            "is_self_intersecting": False,
        }

    edge_lengths = mesh.edges_unique_length
    min_edge = float(np.min(edge_lengths)) if len(edge_lengths) > 0 else 0.0
    mean_edge = float(np.mean(edge_lengths)) if len(edge_lengths) > 0 else 0.0
    max_edge = float(np.max(edge_lengths)) if len(edge_lengths) > 0 else 0.0

    # 체적 계산 (닫힌 메쉬일 때는 정확한 divergence 부피, 아니면 양수화)
    vol = float(abs(mesh.volume)) if mesh.is_volume else (
        float(mesh.convex_hull.volume) if mesh.is_watertight else 0.0
    )
    area = float(mesh.area)
    watertight = bool(mesh.is_watertight)

    # 자기교차(Self-intersection) 검사 (trimesh 지원 범위 내)
    self_intersecting = False
    try:
        if hasattr(mesh, "is_self_intersecting"):
            self_intersecting = bool(mesh.is_self_intersecting)
    except Exception:
        self_intersecting = False

    return {
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "volume": vol,
        "surface_area": area,
        "is_watertight": watertight,
        "min_edge_length": min_edge,
        "mean_edge_length": mean_edge,
        "max_edge_length": max_edge,
        "is_self_intersecting": self_intersecting,
    }


def _compute_polydata_volume(poly: vtk.vtkPolyData) -> float:
    """vtkMassProperties를 이용하여 vtkPolyData의 닫힌 부피(mm^3)를 계산합니다."""
    if poly.GetNumberOfPoints() < 4 or poly.GetNumberOfPolys() < 4:
        return 0.0
    try:
        mass = vtk.vtkMassProperties()
        mass.SetInputData(poly)
        mass.Update()
        return float(abs(mass.GetVolume()))
    except Exception:
        return 0.0


def _get_polydata_polys_array(poly: vtk.vtkPolyData) -> np.ndarray:
    """vtkPolyData의 Polys 셀 배열을 VTK 9.0+ 호환 방식으로 추출합니다."""
    cell_arr = poly.GetPolys()
    if cell_arr is None or cell_arr.GetNumberOfCells() == 0:
        return np.array([], dtype=int)

    if hasattr(cell_arr, "ExportLegacyFormat"):
        legacy_arr = vtk.vtkIdTypeArray()
        cell_arr.ExportLegacyFormat(legacy_arr)
        return numpy_support.vtk_to_numpy(legacy_arr)
    else:
        # Fallback
        return numpy_support.vtk_to_numpy(cell_arr.GetData())


def _decimate_to_target_edge(poly: vtk.vtkPolyData, target_edge_mm: float) -> vtk.vtkPolyData:
    """
    현재 메쉬의 평균 변 길이를 측정하여 target_edge_mm에 부합하도록
    vtkQuadricDecimation을 통해 감축합니다.
    """
    if poly.GetNumberOfPolys() < 10:
        return poly

    pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    polys_raw = _get_polydata_polys_array(poly)

    if polys_raw.size % 4 == 0 and polys_raw[0] == 3:
        faces = polys_raw.reshape(-1, 4)[:, 1:]
    else:
        return poly

    # 표본 에지 길이 샘플링 (계산 속도 최적화)
    n_sample = min(len(faces), 1000)
    sample_faces = faces[:n_sample]
    p0 = pts[sample_faces[:, 0]]
    p1 = pts[sample_faces[:, 1]]
    p2 = pts[sample_faces[:, 2]]

    d01 = np.linalg.norm(p1 - p0, axis=1)
    d12 = np.linalg.norm(p2 - p1, axis=1)
    d20 = np.linalg.norm(p0 - p2, axis=1)
    cur_avg_edge = float(np.mean([d01, d12, d20]))

    if cur_avg_edge <= 0 or cur_avg_edge >= target_edge_mm:
        # 이미 목표 변 길이 이상이거나 에지 측정이 불가하면 감축하지 않음
        return poly

    target_ratio = (cur_avg_edge / target_edge_mm) ** 2
    reduction = 1.0 - target_ratio
    reduction = max(0.05, min(0.95, reduction))

    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(poly)
    decimator.SetTargetReduction(reduction)
    decimator.VolumePreservationOn()
    decimator.Update()
    return decimator.GetOutput()


def _vtk_polydata_to_trimesh(poly: vtk.vtkPolyData) -> trimesh.Trimesh:
    """vtkPolyData를 trimesh.Trimesh 객체로 안전하게 변환합니다."""
    n_pts = poly.GetNumberOfPoints()
    n_polys = poly.GetNumberOfPolys()

    if n_pts == 0 or n_polys == 0:
        return trimesh.Trimesh()

    pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    polys_raw = _get_polydata_polys_array(poly)

    if polys_raw.size == n_polys * 4 and polys_raw[0] == 3:
        faces = polys_raw.reshape(-1, 4)[:, 1:]
    else:
        faces_list = []
        idx = 0
        total = len(polys_raw)
        while idx < total:
            count = polys_raw[idx]
            idx += 1
            if count == 3:
                faces_list.append((polys_raw[idx], polys_raw[idx + 1], polys_raw[idx + 2]))
            elif count > 3:
                # 다각형 팬(fan) 분할
                for k in range(1, count - 1):
                    faces_list.append((polys_raw[idx], polys_raw[idx + k], polys_raw[idx + k + 1]))
            idx += count
        faces = np.array(faces_list, dtype=int)

    return trimesh.Trimesh(vertices=pts, faces=faces, process=True)
