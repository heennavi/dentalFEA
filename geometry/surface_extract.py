"""
geometry/surface_extract.py (루트 호환용)
"""

from dental_fea_app.geometry.surface_extract import (
    mask_to_surface,
    mesh_report,
    _compute_polydata_volume,
    _decimate_to_target_edge,
    _vtk_polydata_to_trimesh,
)

__all__ = ["mask_to_surface", "mesh_report"]
