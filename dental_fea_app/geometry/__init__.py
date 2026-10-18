"""
geometry package
치과 FEA 공통 기하 및 좌표계 변환, 표면 메쉬 추출 순수 로직 패키지.
"""

from .surface_extract import mask_to_surface, mesh_report
from .boolean_ops import FeaGeometry, build_fea_geometry, export_stl, ensure_watertight, calculate_contact_area

__all__ = [
    "mask_to_surface",
    "mesh_report",
    "FeaGeometry",
    "build_fea_geometry",
    "export_stl",
    "ensure_watertight",
    "calculate_contact_area",
]
