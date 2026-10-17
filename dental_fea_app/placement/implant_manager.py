"""
implant_manager.py
임플란트 STL 로드, 3D 위치 및 자세(Transform 4x4 matrix) 관리, 충돌/적합도 분석 순수 로직 모듈.

[설계 원칙]
- 순수 로직 모듈: PySide6를 일절 import하지 않습니다.
- 상태 변경 알림: Callable 콜백 옵저버 리스트를 사용하여 PySide6 신호 없이 UI/상태 관리자와 연동합니다.
- 좌표계 및 물리 단위계:
  * 단위: 길이 mm, 각도 deg, 밀도 HU.
  * 기하 좌표계: DICOM LPS World 좌표계(mm).
  * 로컬 좌표계 규칙: 임플란트 장축 = 로컬 +Z 방향, 아래쪽(식립 방향 끝, apex) = 로컬 -Z 방향.
    (apex_local = [0, 0, z_min], top_local = [0, 0, z_max])
"""

import os
import math
import warnings
from typing import Tuple, List, Callable, Optional, Dict, Any, Union
import numpy as np
import scipy.spatial.transform
import scipy.ndimage
import vtk
from vtk.util import numpy_support


class ImplantModel:
    """
    임플란트 3D 형상(STL) 로드 및 4x4 변환 행렬 기반 3D 위치/자세를 관리하는 모델 클래스.
    """

    def __init__(self, stl_path: Optional[str] = None):
        # 원본 로컬 vtkPolyData 및 원본 정점 백업
        self._raw_polydata: Optional[vtk.vtkPolyData] = None
        self._stl_path: Optional[str] = None

        # 4x4 World <- Local 아핀 변환 행렬 (기본 항등행렬)
        self._transform_matrix: np.ndarray = np.eye(4, dtype=np.float64)

        # 상태 변경 관찰자(콜백 함수 목록: callback(self))
        self._observers: List[Callable[["ImplantModel"], None]] = []

        if stl_path is not None:
            self.load_stl(stl_path)

    @property
    def stl_path(self) -> Optional[str]:
        return self._stl_path

    @property
    def raw_polydata(self) -> Optional[vtk.vtkPolyData]:
        return self._raw_polydata

    @property
    def transform_matrix(self) -> np.ndarray:
        """4x4 변환 행렬 (World <- Local) 사본 반환"""
        return self._transform_matrix.copy()

    @transform_matrix.setter
    def transform_matrix(self, mat: np.ndarray):
        mat = np.asarray(mat, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"변환 행렬은 4x4 형태여야 합니다: {mat.shape}")
        self._transform_matrix = mat.copy()
        self._notify_observers()

    @property
    def position(self) -> np.ndarray:
        """World 좌표계 상의 원점 위치 (x, y, z) [mm]"""
        return self._transform_matrix[:3, 3].copy()

    @position.setter
    def position(self, pos: Union[Tuple[float, float, float], np.ndarray, list]):
        pos = np.asarray(pos, dtype=np.float64).flatten()
        if len(pos) != 3:
            raise ValueError("위치 좌표는 3차원 (x, y, z) 형태여야 합니다.")
        self._transform_matrix[:3, 3] = pos
        self._notify_observers()

    def add_observer(self, callback: Callable[["ImplantModel"], None]) -> None:
        """모델 변환 또는 형상 변경 시 호출될 옵저버 콜백 등록"""
        if callback not in self._observers:
            self._observers.append(callback)

    def remove_observer(self, callback: Callable[["ImplantModel"], None]) -> None:
        """옵저버 콜백 등록 해제"""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self) -> None:
        for cb in self._observers:
            try:
                cb(self)
            except Exception as e:
                warnings.warn(f"ImplantModel 옵저버 호출 중 오류 발생: {e}")

    # -------------------------------------------------------------------------
    # STL 파일 로드 및 유효성 검증
    # -------------------------------------------------------------------------
    def load_stl(self, path: str) -> Dict[str, Any]:
        """
        STL 파일을 로드하고 bounding box 크기 유효성을 검증합니다.

        Args:
            path: STL 파일 경로

        Returns:
            info_dict: {"path": str, "dimensions": (dx, dy, dz), "warning": Optional[str]}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL 파일을 찾을 수 없습니다: {path}")

        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()

        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError(f"유효한 폴리곤 데이터를 읽을 수 없습니다: {path}")

        # 원본 데이터 보관
        self._raw_polydata = vtk.vtkPolyData()
        self._raw_polydata.DeepCopy(poly)
        self._stl_path = path

        # 바운딩 박스 크기 검사 (단위 mm 가정)
        bounds = self._raw_polydata.GetBounds()  # xmin, xmax, ymin, ymax, zmin, zmax
        dx = float(bounds[1] - bounds[0])
        dy = float(bounds[3] - bounds[2])
        dz = float(bounds[5] - bounds[4])
        dims = (dx, dy, dz)
        max_dim = max(dims)
        min_dim = min(dims)

        warning_msg = None
        if max_dim > 100.0 or min_dim < 0.1:
            warning_msg = (
                f"임플란트 치수가 비정상적입니다 (최대 치수 {max_dim:.2f} mm, 최소 {min_dim:.2f} mm). "
                "STL 파일의 단위계(m, cm, um 등) 오류 가능성이 높습니다."
            )
            warnings.warn(warning_msg, UserWarning)

        # 포즈 초기화
        self._transform_matrix = np.eye(4, dtype=np.float64)
        self._notify_observers()

        return {
            "path": path,
            "dimensions": dims,
            "bounds": bounds,
            "warning": warning_msg,
        }

    def set_raw_polydata(self, poly: vtk.vtkPolyData) -> None:
        """테스트 및 프로그래밍 방식으로 직접 vtkPolyData를 주입할 때 사용"""
        self._raw_polydata = vtk.vtkPolyData()
        self._raw_polydata.DeepCopy(poly)
        self._stl_path = None
        self._transform_matrix = np.eye(4, dtype=np.float64)
        self._notify_observers()

    # -------------------------------------------------------------------------
    # 로컬 좌표계 및 기하 정보
    # -------------------------------------------------------------------------
    def get_local_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """로컬 좌표계 상의 Bounding box (xmin, xmax, ymin, ymax, zmin, zmax)"""
        if self._raw_polydata is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return self._raw_polydata.GetBounds()

    def get_local_apex(self) -> np.ndarray:
        """
        로컬 좌표계 상에서 임플란트 식립 끝(Apex, -Z 방향 최하단) 점의 중심 좌표 반환.
        규칙: 장축 = 로컬 +Z, Apex = 로컬 -Z
        """
        if self._raw_polydata is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        bounds = self._raw_polydata.GetBounds()
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz_min = bounds[4]  # 로컬 -Z 최하단
        return np.array([cx, cy, cz_min], dtype=np.float64)

    def get_local_top(self) -> np.ndarray:
        """
        로컬 좌표계 상에서 임플란트 상단(Platform/Top, +Z 방향 최상단) 점의 중심 좌표 반환.
        """
        if self._raw_polydata is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        bounds = self._raw_polydata.GetBounds()
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz_max = bounds[5]  # 로컬 +Z 최상단
        return np.array([cx, cy, cz_max], dtype=np.float64)

    def set_local_axis(self, direction: Union[Tuple[float, float, float], np.ndarray, list]) -> None:
        """
        임플란트 STL의 현재 장축 방향이 +Z가 아닐 때,
        주어진 방향 벡터(direction)가 로컬 +Z가 되도록 내부 raw_polydata 정점을 회전 보정합니다.
        """
        if self._raw_polydata is None:
            raise ValueError("로드된 형상 데이터가 없습니다.")

        v_from = np.asarray(direction, dtype=np.float64).flatten()
        norm = np.linalg.norm(v_from)
        if norm < 1e-8:
            raise ValueError("방향 벡터의 크기가 0입니다.")
        v_from = v_from / norm

        v_to = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # v_from을 v_to로 정렬하는 회전 행렬 계산
        dot = np.dot(v_from, v_to)
        if np.isclose(dot, 1.0):
            R = np.eye(3)
        elif np.isclose(dot, -1.0):
            # 180도 반전 (임의의 수직축으로 뒤집기)
            axis = np.array([1.0, 0.0, 0.0]) if abs(v_from[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(v_from, axis)
            axis = axis / np.linalg.norm(axis)
            R = scipy.spatial.transform.Rotation.from_rotvec(np.pi * axis).as_matrix()
        else:
            axis = np.cross(v_from, v_to)
            axis = axis / np.linalg.norm(axis)
            angle = math.acos(np.clip(dot, -1.0, 1.0))
            R = scipy.spatial.transform.Rotation.from_rotvec(angle * axis).as_matrix()

        # PolyData 정점 회전 적용
        points = self._raw_polydata.GetPoints()
        pts_np = numpy_support.vtk_to_numpy(points.GetData())
        rotated_pts = pts_np @ R.T

        # 새로운 점 데이터로 갱신
        new_vtk_pts = numpy_support.numpy_to_vtk(rotated_pts, deep=True, array_type=vtk.VTK_FLOAT)
        new_points = vtk.vtkPoints()
        new_points.SetData(new_vtk_pts)
        self._raw_polydata.SetPoints(new_points)

        self._notify_observers()

    # -------------------------------------------------------------------------
    # 포즈 및 변환(Transform) 제어
    # -------------------------------------------------------------------------
    def reset_pose(self) -> None:
        """변환 행렬을 항등행렬(원점 위치, 회전 없음)로 리셋합니다."""
        self._transform_matrix = np.eye(4, dtype=np.float64)
        self._notify_observers()

    def translate(self, dx: float, dy: float, dz: float) -> None:
        """World 좌표계 기준으로 평행 이동"""
        self._transform_matrix[0, 3] += float(dx)
        self._transform_matrix[1, 3] += float(dy)
        self._transform_matrix[2, 3] += float(dz)
        self._notify_observers()

    def rotate_about_axis(self, angle_deg: float, axis_world: Union[Tuple[float, float, float], np.ndarray, list] = (0.0, 0.0, 1.0)) -> None:
        """
        World 좌표계의 임의 축에 대해 현재 위치(또는 원점)를 기준으로 회전합니다.
        (기본: 임플란트의 현재 위치 중심 회전)
        """
        axis = np.asarray(axis_world, dtype=np.float64).flatten()
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            raise ValueError("회전축 벡터의 크기가 0입니다.")
        axis = axis / norm

        rot = scipy.spatial.transform.Rotation.from_rotvec(np.radians(angle_deg) * axis)
        R_delta = rot.as_matrix()

        # 임플란트 현재 월드 위치를 중심점으로 회전 수행
        pos = self.position
        # 1. 위치를 원점으로 이동 -> 2. 회전 -> 3. 위치 복원
        current_R = self._transform_matrix[:3, :3]
        new_R = R_delta @ current_R

        self._transform_matrix[:3, :3] = new_R
        self._transform_matrix[:3, 3] = pos
        self._notify_observers()

    def get_euler_angles(self, seq: str = "zyx") -> Tuple[float, float, float]:
        """
        현재 회전 성분을 오일러 각(도 단위)으로 반환합니다.
        기본 시퀀스: 'zyx' (yaw: Z, pitch: Y, roll: X)
        """
        R = self._transform_matrix[:3, :3]
        rot = scipy.spatial.transform.Rotation.from_matrix(R)
        angles = rot.as_euler(seq, degrees=True)
        return float(angles[0]), float(angles[1]), float(angles[2])

    def set_euler_angles(self, angle1: float, angle2: float, angle3: float, seq: str = "zyx") -> None:
        """
        오일러 각(도 단위)으로 회전 성분을 설정합니다. (위치는 유지)
        기본 시퀀스: 'zyx' (yaw, pitch, roll)
        """
        rot = scipy.spatial.transform.Rotation.from_euler(seq, [angle1, angle2, angle3], degrees=True)
        R = rot.as_matrix()
        self._transform_matrix[:3, :3] = R
        self._notify_observers()

    # 편의용 yaw, pitch, roll 속성
    @property
    def yaw_deg(self) -> float:
        return self.get_euler_angles("zyx")[0]

    @property
    def pitch_deg(self) -> float:
        return self.get_euler_angles("zyx")[1]

    @property
    def roll_deg(self) -> float:
        return self.get_euler_angles("zyx")[2]

    def set_yaw_pitch_roll(self, yaw: float, pitch: float, roll: float) -> None:
        self.set_euler_angles(yaw, pitch, roll, seq="zyx")

    def place_at(
        self,
        point_world: Union[Tuple[float, float, float], np.ndarray, list],
        axis_world: Union[Tuple[float, float, float], np.ndarray, list],
    ) -> None:
        """
        임플란트의 장축(+Z)이 axis_world 방향을 향하고,
        식립 끝점(Apex, 로컬 -Z)이 point_world에 정확히 위치하도록 4x4 행렬을 배치합니다.

        Args:
            point_world: 식립 지점 (Apex가 위치할 World 좌표, mm)
            axis_world: 식립 방향 (Apex에서 Top으로 향하는 장축 벡터, World)
        """
        pt_w = np.asarray(point_world, dtype=np.float64).flatten()
        ax_w = np.asarray(axis_world, dtype=np.float64).flatten()

        norm = np.linalg.norm(ax_w)
        if norm < 1e-8:
            raise ValueError("axis_world 방향 벡터의 크기가 0입니다.")
        z_axis_w = ax_w / norm

        # 로컬 +Z 벡터 ([0, 0, 1])를 z_axis_w로 매핑하는 회전 행렬 R 계산
        z_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        dot = np.dot(z_local, z_axis_w)

        if np.isclose(dot, 1.0):
            R = np.eye(3)
        elif np.isclose(dot, -1.0):
            # 180도 반전 (+Z -> -Z)
            R = scipy.spatial.transform.Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
        else:
            axis = np.cross(z_local, z_axis_w)
            axis = axis / np.linalg.norm(axis)
            angle = math.acos(np.clip(dot, -1.0, 1.0))
            R = scipy.spatial.transform.Rotation.from_rotvec(angle * axis).as_matrix()

        # 로컬 Apex 좌표 (보통 [0, 0, z_min])
        apex_local = self.get_local_apex()

        # World 좌표계 상에서 Apex가 point_world에 도달하기 위한 원점 이동 t 계산:
        # Apex_world = R @ apex_local + t = point_world
        # => t = point_world - R @ apex_local
        t = pt_w - (R @ apex_local)

        # 4x4 행렬 구성
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        self._transform_matrix = T
        self._notify_observers()

    def get_world_apex(self) -> np.ndarray:
        """현재 포즈에서 World 좌표계 상의 Apex 좌표 반환"""
        apex_local = self.get_local_apex()
        apex_h = np.append(apex_local, 1.0)
        return (self._transform_matrix @ apex_h)[:3]

    def get_world_top(self) -> np.ndarray:
        """현재 포즈에서 World 좌표계 상의 Top 좌표 반환"""
        top_local = self.get_local_top()
        top_h = np.append(top_local, 1.0)
        return (self._transform_matrix @ top_h)[:3]

    def get_world_axis(self) -> np.ndarray:
        """현재 포즈에서 World 좌표계 상의 장축 방향 단위 벡터 (+Z 축) 반환"""
        return self._transform_matrix[:3, 2].copy()

    # -------------------------------------------------------------------------
    # World 변환 PolyData 생성
    # -------------------------------------------------------------------------
    def get_transformed_polydata(self) -> vtk.vtkPolyData:
        """
        현재 4x4 변환 행렬이 적용된 World 좌표계의 vtkPolyData를 생성하여 반환합니다.
        """
        if self._raw_polydata is None:
            return vtk.vtkPolyData()

        vtk_mat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_mat.SetElement(r, c, float(self._transform_matrix[r, c]))

        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_mat)

        transform_filter = getattr(vtk, "vtkTransformFilter", vtk.vtkTransformPolyDataFilter)()
        transform_filter.SetInputData(self._raw_polydata)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        poly_out = vtk.vtkPolyData()
        poly_out.DeepCopy(transform_filter.GetOutput())
        return poly_out


# -----------------------------------------------------------------------------
# 충돌 및 적합도(Fit) 확인 유틸리티 함수
# -----------------------------------------------------------------------------
def sample_hu_along_axis(
    volume: Union[vtk.vtkImageData, np.ndarray],
    pose: Union[ImplantModel, np.ndarray],
    spacing: Optional[Tuple[float, float, float]] = None,
    origin: Optional[Tuple[float, float, float]] = None,
    num_samples: int = 50,
    length_mm: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    임플란트 장축을 따라 CT 볼륨의 HU 프로파일(거리 mm 및 HU 값)을 샘플링합니다.

    Args:
        volume: CT 볼륨 (vtk.vtkImageData 또는 3D ndarray (nz, ny, nx))
        pose: ImplantModel 인스턴스 또는 4x4 변환 행렬
        spacing: 복셀 간격 (mm), volume이 ndarray인 경우 필수 (sx, sy, sz)
        origin: 볼륨 원점 (mm), volume이 ndarray인 경우 필수 (ox, oy, oz)
        num_samples: 샘플링 점 개수 (기본 50)
        length_mm: 샘플링할 축 길이 (None이면 임플란트 모델 전체 길이)

    Returns:
        (distances_mm, hu_values):
            distances_mm: Apex(0.0 mm)로부터의 축 방향 거리 배열
            hu_values: 각 지점에서의 HU 밀도 값 배열
    """
    if isinstance(pose, ImplantModel):
        apex_w = pose.get_world_apex()
        axis_w = pose.get_world_axis()
        if length_mm is None:
            top_w = pose.get_world_top()
            length_mm = float(np.linalg.norm(top_w - apex_w))
    elif isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        apex_w = pose[:3, 3]
        axis_w = pose[:3, 2]
        if length_mm is None:
            length_mm = 10.0  # 기본 10mm
    else:
        raise TypeError("pose는 ImplantModel 또는 4x4 ndarray여야 합니다.")

    if length_mm <= 0.0:
        length_mm = 10.0

    # 1. 볼륨 정보 추출
    if isinstance(volume, vtk.vtkImageData):
        spc = volume.GetSpacing()
        spacing = (float(spc[0]), float(spc[1]), float(spc[2]))
        org = volume.GetOrigin()
        origin = (float(org[0]), float(org[1]), float(org[2]))
        nx, ny, nz = volume.GetDimensions()
        scalars = volume.GetPointData().GetScalars()
        if scalars is None:
            raise ValueError("vtkImageData에 스칼라 데이터가 없습니다.")
        hu_3d = numpy_support.vtk_to_numpy(scalars).reshape((nz, ny, nx))
    elif isinstance(volume, np.ndarray):
        if spacing is None or origin is None:
            raise ValueError("numpy volume 입력 시 spacing과 origin을 반드시 제공해야 합니다.")
        hu_3d = volume
        nz, ny, nx = hu_3d.shape
    else:
        raise TypeError(f"지원하지 않는 볼륨 데이터 타입입니다: {type(volume)}")

    # 2. 축 샘플링 좌표 생성
    distances_mm = np.linspace(0.0, length_mm, num_samples)
    sample_points_world = apex_w + distances_mm[:, np.newaxis] * axis_w

    # 3. World (x, y, z) -> Voxel (i, j, k) 변환
    # i = (x - ox) / sx, j = (y - oy) / sy, k = (z - oz) / sz
    # hu_3d 인덱싱은 (k, j, i)
    ox, oy, oz = origin
    sx, sy, sz = spacing

    vox_i = (sample_points_world[:, 0] - ox) / sx
    vox_j = (sample_points_world[:, 1] - oy) / sy
    vox_k = (sample_points_world[:, 2] - oz) / sz

    # scipy map_coordinates 사용 (선형 보간)
    # coordinates: shape (3, num_samples) in order (z, y, x) = (k, j, i)
    coords = np.vstack([vox_k, vox_j, vox_i])
    hu_values = scipy.ndimage.map_coordinates(
        hu_3d,
        coords,
        order=1,
        mode="constant",
        cval=-1000.0,  # 볼륨 외부 공기 HU
    )

    return distances_mm, hu_values


def bone_overlap_ratio(
    implant_polydata: vtk.vtkPolyData,
    bone_mask: Union[np.ndarray, vtk.vtkImageData],
    mask_geometry: Optional[Dict[str, Any]] = None,
) -> float:
    """
    임플란트 표면 샘플 점 중 뼈 마스크(피질골+수질골) 내부 복셀에 포함되는 비율(0.0 ~ 1.0)을 계산합니다.

    Args:
        implant_polydata: World 좌표계 상의 임플란트 메쉬 (get_transformed_polydata 결과)
        bone_mask: 3D bool 마스크 배열 (shape: (nz, ny, nx)) 또는 vtkImageData
        mask_geometry: bone_mask가 ndarray인 경우 geometry 정보 dict:
            {"origin": (ox, oy, oz), "spacing": (sx, sy, sz)}

    Returns:
        overlap_ratio: 0.0 ~ 1.0 사이의 뼈 중첩 비율
    """
    if implant_polydata is None or implant_polydata.GetNumberOfPoints() == 0:
        return 0.0

    # 1. 뼈 마스크 및 지오메트리 추출
    if isinstance(bone_mask, vtk.vtkImageData):
        spc = bone_mask.GetSpacing()
        spacing = (float(spc[0]), float(spc[1]), float(spc[2]))
        org = bone_mask.GetOrigin()
        origin = (float(org[0]), float(org[1]), float(org[2]))
        nx, ny, nz = bone_mask.GetDimensions()
        scalars = bone_mask.GetPointData().GetScalars()
        if scalars is None:
            return 0.0
        mask_3d = (numpy_support.vtk_to_numpy(scalars).reshape((nz, ny, nx)) > 0.5)
    elif isinstance(bone_mask, np.ndarray):
        if mask_geometry is None:
            raise ValueError("bone_mask가 ndarray일 경우 mask_geometry(origin, spacing)가 필수입니다.")
        origin = mask_geometry["origin"]
        spacing = mask_geometry["spacing"]
        mask_3d = bone_mask.astype(bool)
        nz, ny, nx = mask_3d.shape
    else:
        raise TypeError(f"지원하지 않는 마스크 타입입니다: {type(bone_mask)}")

    # 2. 임플란트 표면 점 추출 (World 좌표)
    vtk_pts = implant_polydata.GetPoints()
    if vtk_pts is None or vtk_pts.GetNumberOfPoints() == 0:
        return 0.0

    pts_w = numpy_support.vtk_to_numpy(vtk_pts.GetData())
    num_pts = len(pts_w)
    if num_pts == 0:
        return 0.0

    # 3. World -> Voxel Index 정수 변환 (반올림)
    ox, oy, oz = origin
    sx, sy, sz = spacing

    idx_x = np.round((pts_w[:, 0] - ox) / sx).astype(np.int64)
    idx_y = np.round((pts_w[:, 1] - oy) / sy).astype(np.int64)
    idx_z = np.round((pts_w[:, 2] - oz) / sz).astype(np.int64)

    # 볼륨 내부 인덱스 필터링
    valid = (
        (idx_x >= 0) & (idx_x < nx) &
        (idx_y >= 0) & (idx_y < ny) &
        (idx_z >= 0) & (idx_z < nz)
    )

    if not np.any(valid):
        return 0.0

    # 4. 마스크 내부 포함 여부 확인
    # mask_3d shape: (nz, ny, nx) -> [z, y, x]
    in_bone = np.zeros(num_pts, dtype=bool)
    in_bone[valid] = mask_3d[idx_z[valid], idx_y[valid], idx_x[valid]]

    overlap_count = np.sum(in_bone)
    return float(overlap_count / num_pts)
