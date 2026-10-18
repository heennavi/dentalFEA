"""
geometry/boolean_ops.py (루트 호환용)
"""

from dental_fea_app.geometry.boolean_ops import (
    FeaGeometry,
    ensure_watertight,
    filter_small_components,
    calculate_contact_area,
    create_implant_drilling_space,
    build_fea_geometry,
    export_stl,
)

__all__ = [
    "FeaGeometry",
    "ensure_watertight",
    "filter_small_components",
    "calculate_contact_area",
    "create_implant_drilling_space",
    "build_fea_geometry",
    "export_stl",
]
