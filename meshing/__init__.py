"""
meshing package
3D 사면체 메쉬 생성 및 처리 패키지.
"""

from .gmsh_mesher import (
    surface_to_stl,
    compute_tetrahedron_volumes,
    fix_orientation,
    gmsh_tet10_to_ccx,
    mesh_closed_surface,
    build_part_mesh,
)
from .quality_checker import (
    tet_quality,
    quality_report_text,
    assert_mesh_ok,
)
from .surface_extractor import (
    CALCULIX_TET_FACES,
    boundary_faces,
    faces_in_region,
    plane_below,
    plane_above,
    sphere,
    cylinder,
    contact_pair_from_parts,
)

__all__ = [
    "surface_to_stl",
    "compute_tetrahedron_volumes",
    "fix_orientation",
    "gmsh_tet10_to_ccx",
    "mesh_closed_surface",
    "build_part_mesh",
    "tet_quality",
    "quality_report_text",
    "assert_mesh_ok",
    "CALCULIX_TET_FACES",
    "boundary_faces",
    "faces_in_region",
    "plane_below",
    "plane_above",
    "sphere",
    "cylinder",
    "contact_pair_from_parts",
]
