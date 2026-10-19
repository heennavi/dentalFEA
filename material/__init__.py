"""
material package
물성 정의 및 CT HU 기반 골 물성 매핑 패키지.
"""

from .bone_mapping import (
    load_preset,
    evaluate_formula,
    world_to_index,
    sample_hu,
    element_centroids,
    assign_bone_materials,
    titanium_assignment,
)

__all__ = [
    "load_preset",
    "evaluate_formula",
    "world_to_index",
    "sample_hu",
    "element_centroids",
    "assign_bone_materials",
    "titanium_assignment",
]
