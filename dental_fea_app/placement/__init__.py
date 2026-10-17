"""
dental_fea_app.placement
임플란트 STL 로드, 3D 위치/자세(Transform) 행렬 연산 및 식립 관리 순수 로직 패키지.
"""

from dental_fea_app.placement.implant_manager import (
    ImplantModel,
    sample_hu_along_axis,
    bone_overlap_ratio,
)

__all__ = [
    "ImplantModel",
    "sample_hu_along_axis",
    "bone_overlap_ratio",
]
