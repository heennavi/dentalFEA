"""
cortical_cancellous.py
CT/CBCT 볼륨에서 뼈 전체, 피질골(Cortical bone), 수질골(Cancellous bone)을 분할하는 순수 로직 모듈.

[원칙]
- 순수 로직 모듈로 PySide6를 일절 import하지 않습니다.
- 모든 기하 데이터는 DICOM LPS World 좌표계(mm) 및 동일 geometry를 따릅니다.
- 단위: 길이(mm), 영상 밀도(Hounsfield Unit, HU).
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import scipy.ndimage
import vtk
from vtk.util import numpy_support


@dataclass
class SegmentationResult:
    """
    뼈 및 피질골/수질골 분할 결과 데이터클래스.

    Attributes:
        bone_mask: 전체 뼈 영역 3D 이진 마스크 (dtype: bool, shape: (nz, ny, nx))
        cortical_mask: 피질골 3D 이진 마스크 (dtype: bool, shape: (nz, ny, nx))
        cancellous_mask: 수질골 3D 이진 마스크 (dtype: bool, shape: (nz, ny, nx))
        params: 분할에 사용된 매개변수 딕셔너리
    """
    bone_mask: np.ndarray
    cortical_mask: np.ndarray
    cancellous_mask: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 불변성 검증: cortical과 cancellous는 상호 배타적이고 합집합은 bone_mask와 일치
        if self.bone_mask is not None and self.cortical_mask is not None and self.cancellous_mask is not None:
            if not np.array_equal(self.cortical_mask | self.cancellous_mask, self.bone_mask):
                raise ValueError("분할 불변성 위반: cortical_mask와 cancellous_mask의 합집합이 bone_mask와 일치하지 않습니다.")
            if np.any(self.cortical_mask & self.cancellous_mask):
                raise ValueError("분할 불변성 위반: cortical_mask와 cancellous_mask가 겹치는 영역이 존재합니다.")


def extract_bone(
    hu: np.ndarray,
    spacing: Tuple[float, float, float],
    bone_threshold: float = 200.0,
    keep_largest: bool = True,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    CT HU 볼륨에서 뼈 전체 영역(하악골/치아 등)을 이진화하여 3D 마스크를 추출합니다.

    Args:
        hu: 3D 넘파이 배열 (단위: HU, shape: (nz, ny, nx))
        spacing: 복셀 간격 (단위: mm, (sx, sy, sz))
        bone_threshold: 뼈 하한 임계값 (단위: HU, 기본값 200.0)
        keep_largest: True일 경우 배경을 제외한 가장 큰 단일 연결 덩어리만 유지 (기본값 True)
        fill_holes: True일 경우 Axial 슬라이스별 뼈 내부 빈 공간 메우기 수행 (기본값 True)

    Returns:
        bone_mask: 3D bool 마스크 (shape: (nz, ny, nx))
    """
    if hu.ndim != 3:
        raise ValueError(f"입력 배열은 3차원이어야 합니다. 현재 차원: {hu.ndim}")

    # 1. 임계값(Threshold) 기반 초기 뼈 영역 이진화
    mask = (hu >= bone_threshold)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    # 2. 가장 큰 연결 성분만 유지 (소형 노이즈 및 고립 구조물 제거)
    if keep_largest:
        labeled, num_features = scipy.ndimage.label(mask)
        if num_features > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0  # 배경 제외
            if np.max(counts) > 0:
                largest_label = int(np.argmax(counts))
                mask = (labeled == largest_label)

    # 3. 슬라이스별(Axial z축) 구멍 메우기 (치조골/골수강 내부 고립 빈공간 보존)
    if fill_holes:
        nz = mask.shape[0]
        filled_mask = np.empty_like(mask)
        for z in range(nz):
            if np.any(mask[z]):
                filled_mask[z] = scipy.ndimage.binary_fill_holes(mask[z])
            else:
                filled_mask[z] = False
        mask = filled_mask

    # 4. 가벼운 형태학적 닫힘(Morphological Closing)으로 미세 틈 메우기
    struct = scipy.ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    mask = scipy.ndimage.binary_closing(mask, structure=struct, iterations=1)

    return mask.astype(bool)


def split_cortical_cancellous(
    bone_mask: np.ndarray,
    hu: np.ndarray,
    spacing: Tuple[float, float, float],
    method: str = "distance",
    cortical_thickness_mm: float = 1.5,
    cortical_hu: float = 700.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    추출된 뼈 마스크를 외곽 피질골(Cortical bone)과 내부 수질골(Cancellous bone)로 분리합니다.

    [분리 원칙]
    - cortical_mask | cancellous_mask == bone_mask
    - cortical_mask & cancellous_mask == 0 (완전 상호 배타적)

    Args:
        bone_mask: 3D 뼈 영역 마스크 (dtype: bool, shape: (nz, ny, nx))
        hu: 3D CT HU 배열 (단위: HU, shape: (nz, ny, nx))
        spacing: 복셀 간격 (단위: mm, (sx, sy, sz))
        method: 분할 방식 ("distance" 또는 "threshold", 기본값 "distance")
        cortical_thickness_mm: 피질골 목표 두께 (단위: mm, 기본값 1.5mm)
        cortical_hu: threshold 방식 시 피질골 기준 HU (단위: HU, 기본값 700.0)

    Returns:
        (cortical_mask, cancellous_mask): 각 bool 마스크 튜플
    """
    if bone_mask.shape != hu.shape:
        raise ValueError(f"bone_mask와 hu의 크기가 다릅니다: {bone_mask.shape} vs {hu.shape}")

    if not np.any(bone_mask):
        empty = np.zeros_like(bone_mask, dtype=bool)
        return empty, empty.copy()

    # hu 배열의 축 순서가 (z, y, x)이므로 sampling은 (sz, sy, sx)로 매핑
    sampling = (float(spacing[2]), float(spacing[1]), float(spacing[0]))

    if method == "distance":
        # EDT 성능 최적화: 전체 볼륨 대신 bone_mask가 존재하는 3D 바운딩 박스 영역만 잘라서 계산
        pts = np.argwhere(bone_mask)
        z0, y0, x0 = pts.min(axis=0)
        z1, y1, x1 = pts.max(axis=0) + 1

        sub_bone = bone_mask[z0:z1, y0:y1, x0:x1]
        sub_edt = scipy.ndimage.distance_transform_edt(sub_bone, sampling=sampling)

        sub_cortical = sub_bone & (sub_edt <= cortical_thickness_mm)
        sub_cancellous = sub_bone & (~sub_cortical)

        cortical = np.zeros_like(bone_mask, dtype=bool)
        cancellous = np.zeros_like(bone_mask, dtype=bool)
        cortical[z0:z1, y0:y1, x0:x1] = sub_cortical
        cancellous[z0:z1, y0:y1, x0:x1] = sub_cancellous

    elif method == "threshold":
        # 뼈 내부에서 고밀도(HU >= cortical_hu) 부위를 피질골 후보로 지정
        cortical_candidate = bone_mask & (hu >= cortical_hu)

        # 작은 고립 잡음 성분 필터링 (너무 작은 성분은 수질골로 귀속)
        labeled, num_features = scipy.ndimage.label(cortical_candidate)
        if num_features > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            # 최소 유효 성분 크기 (약 15 복셀 미만 노이즈 제거)
            min_voxels = 15
            valid_labels = np.where(counts >= min_voxels)[0]
            cortical = np.isin(labeled, valid_labels)
        else:
            cortical = cortical_candidate

        # 형태학적 보정 (미세 구멍 메우기)
        struct = scipy.ndimage.generate_binary_structure(3, 1)
        cortical = scipy.ndimage.binary_closing(cortical, structure=struct, iterations=1)
        cortical = bone_mask & cortical
        cancellous = bone_mask & (~cortical)

    else:
        raise ValueError(f"지원하지 않는 분할 방식입니다: {method} (가능한 값: 'distance', 'threshold')")

    return cortical.astype(bool), cancellous.astype(bool)


def run_segmentation(
    volume_data: Union[vtk.vtkImageData, np.ndarray],
    spacing: Optional[Tuple[float, float, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    roi_ijk: Optional[Union[Tuple[int, ...], list]] = None,
) -> SegmentationResult:
    """
    CT 볼륨(또는 넘파이 배열)에 대해 뼈, 피질골, 수질골 분할 파이프라인을 일괄 실행합니다.

    Args:
        volume_data: 입력 CT 데이터 (vtk.vtkImageData 또는 3D ndarray (nz, ny, nx))
        spacing: 복셀 간격 (단위: mm, (sx, sy, sz)). volume_data가 vtkImageData인 경우 자동 추출.
        params: 분할 매개변수 딕셔너리
            - bone_threshold (float, default: 200.0)
            - keep_largest (bool, default: True)
            - fill_holes (bool, default: True)
            - method (str, default: "distance")
            - cortical_thickness_mm (float, default: 1.5)
            - cortical_hu (float, default: 700.0)
        roi_ijk: 관심영역 Bounding Box 인덱스.
            (imin, imax, jmin, jmax, kmin, kmax) (x, y, z 6개 튜플)
            지정된 경우 해당 하위 영역만 고속 분할 후 전체 볼륨 마스크로 복원.

    Returns:
        SegmentationResult (bone_mask, cortical_mask, cancellous_mask, params)
    """
    # 1. 입력 데이터 및 Spacing 정규화
    if isinstance(volume_data, vtk.vtkImageData):
        nx, ny, nz = volume_data.GetDimensions()
        spc = volume_data.GetSpacing()
        spacing = (float(spc[0]), float(spc[1]), float(spc[2]))
        scalars = volume_data.GetPointData().GetScalars()
        if scalars is None:
            raise ValueError("입력 vtkImageData에 스칼라 데이터가 존재하지 않습니다.")
        hu_1d = numpy_support.vtk_to_numpy(scalars)
        hu = hu_1d.reshape((nz, ny, nx))
    elif isinstance(volume_data, np.ndarray):
        hu = volume_data
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
    else:
        raise TypeError(f"지원하지 않는 볼륨 데이터 형식입니다: {type(volume_data)}")

    # 2. 파라미터 기본값 병합
    default_params: Dict[str, Any] = {
        "bone_threshold": 200.0,
        "keep_largest": True,
        "fill_holes": True,
        "method": "distance",
        "cortical_thickness_mm": 1.5,
        "cortical_hu": 700.0,
    }
    merged_params = dict(default_params)
    if params:
        merged_params.update(params)

    nz, ny, nx = hu.shape

    # 3. ROI (관심 영역) 슬라이싱 처리
    if roi_ijk is not None and len(roi_ijk) == 6:
        # roi_ijk: (imin, imax, jmin, jmax, kmin, kmax) in (x, y, z)
        x_min, x_max, y_min, y_max, z_min, z_max = roi_ijk
        # 유효 범위 클램핑
        x_min = max(0, min(nx - 1, int(x_min)))
        x_max = max(x_min, min(nx - 1, int(x_max)))
        y_min = max(0, min(ny - 1, int(y_min)))
        y_max = max(y_min, min(ny - 1, int(y_max)))
        z_min = max(0, min(nz - 1, int(z_min)))
        z_max = max(z_min, min(nz - 1, int(z_max)))

        # 슬라이스 추출
        sub_hu = hu[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]

        # 서브 볼륨 분할
        sub_bone = extract_bone(
            sub_hu,
            spacing=spacing,
            bone_threshold=merged_params["bone_threshold"],
            keep_largest=merged_params["keep_largest"],
            fill_holes=merged_params["fill_holes"],
        )
        sub_cortical, sub_cancellous = split_cortical_cancellous(
            sub_bone,
            sub_hu,
            spacing=spacing,
            method=merged_params["method"],
            cortical_thickness_mm=merged_params["cortical_thickness_mm"],
            cortical_hu=merged_params["cortical_hu"],
        )

        # 전체 볼륨 크기의 빈 마스크에 서브 마스크 복원
        bone_mask = np.zeros((nz, ny, nx), dtype=bool)
        cortical_mask = np.zeros((nz, ny, nx), dtype=bool)
        cancellous_mask = np.zeros((nz, ny, nx), dtype=bool)

        bone_mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = sub_bone
        cortical_mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = sub_cortical
        cancellous_mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = sub_cancellous

        merged_params["roi_ijk"] = (x_min, x_max, y_min, y_max, z_min, z_max)

    else:
        # 전체 볼륨 분할 수행
        bone_mask = extract_bone(
            hu,
            spacing=spacing,
            bone_threshold=merged_params["bone_threshold"],
            keep_largest=merged_params["keep_largest"],
            fill_holes=merged_params["fill_holes"],
        )
        cortical_mask, cancellous_mask = split_cortical_cancellous(
            bone_mask,
            hu,
            spacing=spacing,
            method=merged_params["method"],
            cortical_thickness_mm=merged_params["cortical_thickness_mm"],
            cortical_hu=merged_params["cortical_hu"],
        )

    return SegmentationResult(
        bone_mask=bone_mask,
        cortical_mask=cortical_mask,
        cancellous_mask=cancellous_mask,
        params=merged_params,
    )


def mask_to_vtk_image(
    mask: np.ndarray,
    reference_vtk: Optional[vtk.vtkImageData] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Optional[Tuple[float, ...]] = None,
    foreground_value: int = 255,
) -> vtk.vtkImageData:
    """
    3D 이진 마스크(bool ndarray, (nz, ny, nx))를 원본 영상과 동일한 geometry를 갖는
    vtkImageData (uint8 스칼라)로 변환합니다.

    Args:
        mask: 3차원 bool/uint8 배열 (shape: (nz, ny, nx))
        reference_vtk: 원본 볼륨 vtkImageData. 제공될 경우 origin/spacing/direction 자동 일치.
        spacing: 복셀 간격 (mm, (sx, sy, sz)) - reference_vtk 미제공 시 사용
        origin: 볼륨 원점 (mm, (ox, oy, oz)) - reference_vtk 미제공 시 사용
        direction: 9개 원소 방향 행렬 - reference_vtk 미제공 시 사용
        foreground_value: True 영역에 할당할 uint8 값 (기본값 255)

    Returns:
        vtk.vtkImageData (단일 컴포넌트 VTK_UNSIGNED_CHAR)
    """
    if mask.ndim != 3:
        raise ValueError(f"마스크는 3차원 배열이어야 합니다: ndim={mask.ndim}")

    nz, ny, nx = mask.shape

    # 0 또는 foreground_value(255)의 uint8 배열 생성
    if mask.dtype == bool:
        uint8_arr = np.where(mask, np.uint8(foreground_value), np.uint8(0))
    else:
        uint8_arr = mask.astype(np.uint8)

    if not uint8_arr.flags.c_contiguous:
        uint8_arr = np.ascontiguousarray(uint8_arr)

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(nx, ny, nz)

    if reference_vtk is not None:
        ref_dims = reference_vtk.GetDimensions()
        if (nx, ny, nz) != ref_dims:
            raise ValueError(f"마스크 차원 (nx={nx}, ny={ny}, nz={nz})과 참조 볼륨 차원 {ref_dims}이 일치하지 않습니다.")
        vtk_image.SetSpacing(reference_vtk.GetSpacing())
        vtk_image.SetOrigin(reference_vtk.GetOrigin())
        if hasattr(reference_vtk, "GetDirectionMatrix"):
            vtk_image.SetDirectionMatrix(reference_vtk.GetDirectionMatrix())
    else:
        vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
        vtk_image.SetOrigin(origin[0], origin[1], origin[2])
        if direction is not None and len(direction) == 9:
            mat = vtk.vtkMatrix3x3()
            for r in range(3):
                for c in range(3):
                    mat.SetElement(r, c, direction[r * 3 + c])
            vtk_image.SetDirectionMatrix(mat)

    vtk_scalars = numpy_support.numpy_to_vtk(
        num_array=uint8_arr.ravel(),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    vtk_scalars.SetName("Mask_Scalars")
    vtk_image.GetPointData().SetScalars(vtk_scalars)

    return vtk_image


def vtk_image_to_mask(vtk_image: vtk.vtkImageData, threshold: float = 0.5) -> np.ndarray:
    """
    vtkImageData로부터 3차원 bool 마스크 배열((nz, ny, nx))을 추출합니다.

    Args:
        vtk_image: VTK ImageData
        threshold: 마스크 판별 기준치 (기본값 0.5)

    Returns:
        mask: bool 3D 넘파이 배열 (shape: (nz, ny, nx))
    """
    nx, ny, nz = vtk_image.GetDimensions()
    scalars = vtk_image.GetPointData().GetScalars()
    if scalars is None:
        raise ValueError("입력 vtkImageData에 스칼라 데이터가 없습니다.")

    arr_1d = numpy_support.vtk_to_numpy(scalars)
    arr_3d = arr_1d.reshape((nz, ny, nx))
    return (arr_3d > threshold)
