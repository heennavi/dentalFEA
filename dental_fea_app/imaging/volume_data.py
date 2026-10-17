"""
volume_data.py
SimpleITK Image <-> VTK vtkImageData 상호 변환 및 영상 메타데이터 정규화 유틸리티.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


def itk_to_vtk_image(itk_image: sitk.Image, reorient_to_lps: bool = True) -> Tuple[vtk.vtkImageData, Dict[str, Any]]:
    """
    SimpleITK Image 객체를 VTK vtkImageData로 변환합니다.
    Spacing, Origin, Direction 매트릭스를 정합하여 월드 좌표계를 보존합니다.

    Args:
        itk_image: SimpleITK 3D Image
        reorient_to_lps: True일 경우 표준 LPS (Left, Posterior, Superior) 좌표계로 정규화

    Returns:
        (vtk.vtkImageData, metadata_dict)
    """
    if reorient_to_lps:
        try:
            itk_image = sitk.DICOMOrient(itk_image, 'LPS')
        except Exception:
            # DICOMOrient 실패 시 원본 방향 유지
            pass

    # ITK 넘파이 배열 추출: shape는 (Z, Y, X)
    nda = sitk.GetArrayFromImage(itk_image)

    # 부동소수점 또는 16비트 정수형 정규화
    if nda.dtype not in (np.int16, np.int32, np.float32, np.float64):
        # 8비트 부호없는 정수 등이 들어온 경우 안전하게 int16으로 변환
        nda = nda.astype(np.int16)

    # C-contiguous 보장
    if not nda.flags.c_contiguous:
        nda = np.ascontiguousarray(nda)

    # VTK ImageData 생성
    vtk_image = vtk.vtkImageData()
    nz, ny, nx = nda.shape
    vtk_image.SetDimensions(nx, ny, nz)

    spacing = itk_image.GetSpacing()
    origin = itk_image.GetOrigin()
    direction = itk_image.GetDirection()

    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(origin[0], origin[1], origin[2])

    # VTK 스칼라 배열 연결
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=nda.ravel(),
        deep=True,
        array_type=numpy_support.get_vtk_array_type(nda.dtype)
    )
    vtk_array.SetName("HU_Scalars")
    vtk_image.GetPointData().SetScalars(vtk_array)

    metadata = {
        "dimensions": (nx, ny, nz),
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "scalar_min": float(np.min(nda)),
        "scalar_max": float(np.max(nda)),
        "scalar_mean": float(np.mean(nda)),
        "dtype": str(nda.dtype)
    }

    return vtk_image, metadata


def compute_auto_window_level(scalar_range: Tuple[float, float]) -> Tuple[float, float]:
    """
    스칼라 최소/최대 범위로부터 적절한 기본 Window / Level을 계산합니다.
    """
    s_min, s_max = scalar_range
    window = max(100.0, s_max - s_min)
    level = 0.5 * (s_max + s_min)
    return window, level
