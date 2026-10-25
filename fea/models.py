"""
models.py
Data contracts and core models for the FEA pipeline.

[Units]
- Length: mm
- Force: N
- Stress / Modulus: MPa (N/mm^2)
- Density: tonne / mm^3
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np


@dataclass
class MeshPart:
    """
    Representation of a single meshed solid part.

    Attributes:
        name: Name of the part ("bone", "implant", "abutment", "prosthesis").
        nodes: (N, 3) float64 array of vertex coordinates in world space (mm, LPS).
        elements: (M, 4 or 10) int64 array of element nodal connectivity (0-based, global or local).
        element_type: Solid element type identifier ("C3D4" or "C3D10").
        surface_tags: Dictionary mapping surface tag name to element or node index array.
    """
    name: str                                  # "bone", "implant", "abutment", "prosthesis"
    nodes: np.ndarray                          # (N,3) float64, mm, 월드좌표
    elements: np.ndarray                       # (M,4 또는 10) int64, 0-based, 전역 절점 인덱스
    element_type: str                          # "C3D4" 또는 "C3D10"
    surface_tags: Dict[str, np.ndarray] = field(default_factory=dict)  # 이름 -> (K,) 요소/절점 인덱스 (3-2에서 채움)


@dataclass
class MeshModel:
    """
    Collection of mesh parts forming the complete FEA assembly.

    Provides helper methods to query global node offsets, total nodes, and total elements.
    """
    parts: List[MeshPart]

    def global_node_offset(self, part_name: str) -> int:
        """
        Calculate the 0-based global node index offset for the specified part.

        Args:
            part_name: Name of the part to locate.

        Returns:
            int: Starting global node index offset.

        Raises:
            KeyError: If part_name is not found in parts.
        """
        offset = 0
        found = False
        for p in self.parts:
            if p.name == part_name:
                found = True
                break
            offset += len(p.nodes)

        if not found:
            raise KeyError(f"Part '{part_name}' not found in MeshModel.")
        return offset

    def total_nodes(self) -> int:
        """Calculate total number of nodes across all mesh parts."""
        return sum(len(p.nodes) for p in self.parts)

    def total_elements(self) -> int:
        """Calculate total number of elements across all mesh parts."""
        return sum(len(p.elements) for p in self.parts)

    def get_part(self, part_name: str) -> Optional[MeshPart]:
        """Find and return a part by its name, or None if not found."""
        for p in self.parts:
            if p.name == part_name:
                return p
        return None


@dataclass
class MaterialDef:
    """
    Linear elastic isotropic material property definition.

    Attributes:
        name: Material name identifier (e.g. "cortical_bone", "titanium").
        E: Young's modulus (MPa).
        nu: Poisson's ratio (dimensionless).
        density: Mass density (tonne / mm^3), optional for static analysis.
    """
    name: str
    E: float                                   # MPa
    nu: float
    density: Optional[float] = None            # tonne/mm^3, 정적해석에서는 생략 가능


@dataclass
class MaterialAssignment:
    """
    Mapping of elements to defined materials across mesh parts.

    Attributes:
        element_material: Mapping from part name to 1D array of material indices (length M).
        materials: List of MaterialDef instances referenced by index.
    """
    element_material: Dict[str, np.ndarray]    # part 이름 -> (M,) int, materials 리스트의 인덱스
    materials: List[MaterialDef]


@dataclass
class BoundaryConditionSet:
    """
    Collection of boundary constraints and applied external forces.

    Attributes:
        fixed_node_sets: List of constraint dicts:
            {"name": str, "part": str, "node_ids": np.ndarray, "dofs": (1, 3)}
        loads: List of force load dicts:
            {"name": str, "part": str, "node_ids": np.ndarray, "force_N": (fx, fy, fz)}
            (Total resultant force is equally distributed over the node set)
    """
    fixed_node_sets: List[dict]                # {"name": str, "part": str, "node_ids": np.ndarray, "dofs": (1,3)}
    loads: List[dict]                          # {"name": str, "part": str, "node_ids": np.ndarray, "force_N": (fx,fy,fz) 합력}


@dataclass
class TieDef:
    """
    CalculiX surface-to-surface or node-to-surface *TIE constraint definition.

    Attributes:
        name: Unique tie constraint identifier.
        master_part: Name of the master part.
        slave_part: Name of the slave part.
        master_faces: (K, 2) int array of [element_idx, face_id(1~4)].
        slave_nodes: (L,) int array of slave node indices (local to slave part).
        position_tolerance_mm: Maximum search distance in mm.
    """
    name: str
    master_part: str
    slave_part: str
    master_faces: np.ndarray                   # (K,2) [element_idx, face_id(1~4)]
    slave_nodes: np.ndarray                    # (L,) 절점 인덱스(해당 part 로컬)
    position_tolerance_mm: float


@dataclass
class AnalysisConfig:
    """
    Configuration parameters for FEA meshing and solver execution.

    Supports JSON serialization via to_dict() and from_dict().
    """
    mesh_size_far_mm: float
    mesh_size_near_mm: float
    element_type: str
    ccx_path: Optional[str]
    n_threads: int
    mesh_algorithm_3d: str = "auto"       # "auto", "delaunay", "netgen", "hxt"
    optimize_netgen: bool = True          # Netgen 3D 볼륨 최적화기 활성화 여부
    mesh_size_abutment_mm: Optional[float] = None  # Abutment target mesh edge size (mm)
    mesh_size_crown_mm: Optional[float] = None     # Crown target mesh edge size (mm)
    mesh_size_screw_mm: Optional[float] = None     # Screw target mesh edge size (mm)
    screw_contact_tol_mm: float = 0.20             # Thread clearance tolerance for screw-implant contact (mm)
    preload_enabled: bool = True                   # Whether abutment screw preload is applied
    preload_force_n: float = 200.0                 # Abutment screw tightening preload force (N)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        return {
            "mesh_size_far_mm": float(self.mesh_size_far_mm),
            "mesh_size_near_mm": float(self.mesh_size_near_mm),
            "element_type": str(self.element_type),
            "ccx_path": self.ccx_path,
            "n_threads": int(self.n_threads),
            "mesh_algorithm_3d": str(self.mesh_algorithm_3d),
            "optimize_netgen": bool(self.optimize_netgen),
            "mesh_size_abutment_mm": float(self.mesh_size_abutment_mm) if self.mesh_size_abutment_mm is not None else None,
            "mesh_size_crown_mm": float(self.mesh_size_crown_mm) if self.mesh_size_crown_mm is not None else None,
            "mesh_size_screw_mm": float(self.mesh_size_screw_mm) if self.mesh_size_screw_mm is not None else None,
            "screw_contact_tol_mm": float(self.screw_contact_tol_mm),
            "preload_enabled": bool(self.preload_enabled),
            "preload_force_n": float(self.preload_force_n),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        """Instantiate AnalysisConfig from dictionary."""
        abut_sz = data.get("mesh_size_abutment_mm")
        crown_sz = data.get("mesh_size_crown_mm")
        screw_sz = data.get("mesh_size_screw_mm")
        return cls(
            mesh_size_far_mm=float(data.get("mesh_size_far_mm", 2.0)),
            mesh_size_near_mm=float(data.get("mesh_size_near_mm", 0.40)),
            element_type=str(data.get("element_type", "C3D10")),
            ccx_path=data.get("ccx_path"),
            n_threads=int(data.get("n_threads", 8)),
            mesh_algorithm_3d=str(data.get("mesh_algorithm_3d", "auto")),
            optimize_netgen=bool(data.get("optimize_netgen", True)),
            mesh_size_abutment_mm=float(abut_sz) if abut_sz is not None else None,
            mesh_size_crown_mm=float(crown_sz) if crown_sz is not None else None,
            mesh_size_screw_mm=float(screw_sz) if screw_sz is not None else None,
            screw_contact_tol_mm=float(data.get("screw_contact_tol_mm", 0.20)),
            preload_enabled=bool(data.get("preload_enabled", True)),
            preload_force_n=float(data.get("preload_force_n", 200.0)),
        )
