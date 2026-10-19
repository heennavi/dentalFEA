"""
fea package
치과 임플란트 유한요소해석(FEA) 파이프라인 패키지.
"""

from .models import (
    MeshPart,
    MeshModel,
    MaterialDef,
    MaterialAssignment,
    BoundaryConditionSet,
    TieDef,
    AnalysisConfig,
)
from .boundary_conditions import (
    fix_by_plane,
    fix_by_region,
    load_on_region,
    load_from_angle,
    default_load_presets,
    validate_bc,
)

__all__ = [
    "MeshPart",
    "MeshModel",
    "MaterialDef",
    "MaterialAssignment",
    "BoundaryConditionSet",
    "TieDef",
    "AnalysisConfig",
    "fix_by_plane",
    "fix_by_region",
    "load_on_region",
    "load_from_angle",
    "default_load_presets",
    "validate_bc",
]
