"""
boundary_conditions.py
Boundary condition and occlusal load definitions, spatial region selectors, and validity checks.

[Units]
- Length: mm
- Force: N
- Angles: degrees
"""

import logging
from typing import Sequence, Optional, Dict, Any, Callable, List, Tuple
import numpy as np

from fea.models import MeshPart, MeshModel, BoundaryConditionSet

logger = logging.getLogger(__name__)


def fix_by_plane(
    part: MeshPart,
    plane_point: Sequence[float] | np.ndarray,
    plane_normal: Sequence[float] | np.ndarray,
    side: str = "below",
    tol_mm: float = 0.01,
    dofs: Tuple[int, int] = (1, 3),
    name: str = "FIX_BASE"
) -> Dict[str, Any]:
    """
    Select and fix nodes lying on, below, or above a given geometric cutting plane.

    Used primarily in dental FEA to constrain resected base or lateral boundaries of mandibular/maxillary bone blocks.

    Plane equation: dot(p - plane_point, n) = d
    - "below": d <= tol_mm
    - "above": d >= -tol_mm
    - "on": abs(d) <= tol_mm

    Args:
        part: Target MeshPart instance.
        plane_point: Reference point on the plane (x0, y0, z0) in mm.
        plane_normal: Plane normal vector (nx, ny, nz).
        side: Selection side ("below", "above", or "on"). Default: "below".
        tol_mm: Distance tolerance threshold in mm.
        dofs: Constrained degrees of freedom tuple (e.g. (1, 3) for UX, UY, UZ).
        name: Name identifier for the fixed node set.

    Returns:
        dict: Single entry formatted for BoundaryConditionSet.fixed_node_sets:
              {"name": name, "part": part.name, "node_ids": np.ndarray, "dofs": dofs}
    """
    p0 = np.asarray(plane_point, dtype=np.float64)
    n = np.asarray(plane_normal, dtype=np.float64)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        raise ValueError("Plane normal vector cannot be zero.")
    n = n / norm_n

    # Signed distance from plane for all nodes
    dists = np.dot(part.nodes - p0, n)

    side_lower = side.lower()
    if side_lower == "below":
        mask = dists <= tol_mm
    elif side_lower == "above":
        mask = dists >= -tol_mm
    elif side_lower in ("on", "coplanar"):
        mask = np.abs(dists) <= tol_mm
    else:
        raise ValueError(f"Invalid side parameter: '{side}'. Allowed: 'below', 'above', 'on'")

    node_ids = np.where(mask)[0].astype(np.int64)

    return {
        "name": name,
        "part": part.name,
        "node_ids": node_ids,
        "dofs": dofs,
    }


def fix_by_region(
    part: MeshPart,
    predicate: Callable[[np.ndarray], np.ndarray],
    dofs: Tuple[int, int] = (1, 3),
    name: str = "FIX_REGION"
) -> Dict[str, Any]:
    """
    Select and fix nodes satisfying an arbitrary geometric predicate function.

    Args:
        part: Target MeshPart instance.
        predicate: Callable taking (N, 3) nodal coordinates and returning (N,) bool mask.
        dofs: Constrained degrees of freedom tuple (default: (1, 3)).
        name: Name identifier for the fixed node set.

    Returns:
        dict: Single entry formatted for BoundaryConditionSet.fixed_node_sets.
    """
    mask = predicate(part.nodes)
    node_ids = np.where(mask)[0].astype(np.int64)

    return {
        "name": name,
        "part": part.name,
        "node_ids": node_ids,
        "dofs": dofs,
    }


def load_on_region(
    part: MeshPart,
    predicate_or_node_ids: Callable[[np.ndarray], np.ndarray] | Sequence[int] | np.ndarray,
    force_vector_N: Sequence[float] | np.ndarray,
    name: str = "LOAD_OCCLUSAL"
) -> Dict[str, Any]:
    """
    Apply resultant force vector force_vector_N distributed over target nodes.

    [Load Distribution Strategy]
    The total resultant force vector (Fx, Fy, Fz) is uniformly partitioned among all
    selected nodes: f_node = force_vector_N / len(node_ids).
    (Note: Area-weighted traction distribution over surface elements is a planned extension point).

    Args:
        part: Target MeshPart instance (e.g. prosthesis, abutment, or implant).
        predicate_or_node_ids: Either a predicate function returning bool mask on nodes,
                               or an explicit sequence of local node indices.
        force_vector_N: Resultant force vector (fx, fy, fz) in N.
        name: Load set identifier.

    Returns:
        dict: Single entry formatted for BoundaryConditionSet.loads:
              {"name": name, "part": part.name, "node_ids": np.ndarray, "force_N": (fx, fy, fz)}
    """
    if callable(predicate_or_node_ids):
        mask = predicate_or_node_ids(part.nodes)
        node_ids = np.where(mask)[0].astype(np.int64)
    else:
        node_ids = np.asarray(predicate_or_node_ids, dtype=np.int64)

    f_vec = tuple(float(f) for f in force_vector_N)
    if len(f_vec) != 3:
        raise ValueError(f"Force vector must have exactly 3 components (Fx, Fy, Fz), got {len(f_vec)}.")

    return {
        "name": name,
        "part": part.name,
        "node_ids": node_ids,
        "force_N": f_vec,
    }


def load_from_angle(
    magnitude_N: float,
    axis_direction: Sequence[float] | np.ndarray,
    tilt_deg: float = 0.0,
    tilt_azimuth_deg: float = 0.0
) -> np.ndarray:
    """
    Calculate 3D force vector inclined at tilt_deg relative to the tooth/implant long axis.

    Preserves exact force magnitude: ||F|| = magnitude_N.

    Args:
        magnitude_N: Total force magnitude in Newtons (N).
        axis_direction: 3D vector defining the long axis of the tooth/implant
                        (directed towards root/apex in occlusal compression, e.g. [0, 0, -1]).
        tilt_deg: Inclination angle in degrees away from axis_direction (0 = pure axial load).
        tilt_azimuth_deg: Azimuthal orientation angle in degrees around the axis (buccal/lingual/mesial/distal).

    Returns:
        (3,) float64 numpy array of force components (Fx, Fy, Fz) in N.
    """
    axis = np.asarray(axis_direction, dtype=np.float64)
    norm_a = np.linalg.norm(axis)
    if norm_a < 1e-12:
        raise ValueError("axis_direction vector cannot be a zero vector.")
    a = axis / norm_a

    # Construct orthogonal coordinate frame (a, u, v)
    if abs(a[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    u = np.cross(a, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(a, u)

    theta = np.deg2rad(tilt_deg)
    phi = np.deg2rad(tilt_azimuth_deg)

    # Unit direction vector tilted by theta and rotated by phi
    d = np.cos(theta) * a + np.sin(theta) * (np.cos(phi) * u + np.sin(phi) * v)
    d = d / np.linalg.norm(d)

    force_vec = float(magnitude_N) * d
    return force_vec


def default_load_presets() -> Dict[str, Dict[str, Any]]:
    """
    Return editable default occlusal loading presets for dental implant FEA.

    Note:
        The loading conditions below represent typical clinical and ISO 14801 test values.
        Actual magnitudes and angles must be selected by the user according to the specific
        study objective (e.g. masticatory maximum, bruxism, or fatigue test standards).
    """
    return {
        "axial_100N": {
            "name": "수직 교합 하중 (100N)",
            "magnitude_N": 100.0,
            "tilt_deg": 0.0,
            "tilt_azimuth_deg": 0.0,
            "description": "치아 장축 방향 수직 압축 하중 100 N (전치부/소구치 표준)",
        },
        "oblique_30deg_100N": {
            "name": "경사 교합 하중 (30°, 100N)",
            "magnitude_N": 100.0,
            "tilt_deg": 30.0,
            "tilt_azimuth_deg": 0.0,
            "description": "치아 장축 대비 30도 경사 하중 100 N (ISO 14801 피로 시험 표준 기준)",
        },
        "axial_150N": {
            "name": "구치부 수직 하중 (150N)",
            "magnitude_N": 150.0,
            "tilt_deg": 0.0,
            "tilt_azimuth_deg": 0.0,
            "description": "대구치 수직 저작 압축 하중 150 N",
        },
        "oblique_15deg_200N": {
            "name": "구치부 경사 하중 (15°, 200N)",
            "magnitude_N": 200.0,
            "tilt_deg": 15.0,
            "tilt_azimuth_deg": 45.0,
            "description": "대구치 측방 교합 하중 200 N (15도 협설측 경사)",
        },
    }


def validate_bc(model: MeshModel, bc: BoundaryConditionSet) -> List[str]:
    """
    Validate boundary conditions and external loads before submitting to solver.

    Checks:
        1. Whether any fixed boundary condition nodes exist (warns against rigid body motion).
        2. Whether sufficient DOFs are constrained (at least 3 non-collinear nodes or planar constraints).
        3. Whether load sets reference valid parts and contain at least 1 node (error).
        4. Whether load force vector magnitude is non-zero (error).
        5. Whether referenced parts exist in the mesh model (error).

    Args:
        model: MeshModel containing all meshed parts.
        bc: BoundaryConditionSet containing fixed sets and loads.

    Returns:
        List[str]: List of Korean error/warning message strings. Returns empty list if valid.
    """
    issues: List[str] = []
    part_names = {p.name for p in model.parts}

    # 1. 고정 절점 검사
    total_fixed_nodes = 0
    all_fixed_node_coords: List[np.ndarray] = []

    for item in bc.fixed_node_sets:
        part_name = item.get("part", "")
        if part_name not in part_names:
            issues.append(f"오류: 고정 경계조건 '{item.get('name', 'FIX')}'이 지정한 파트 '{part_name}'가 모델에 존재하지 않습니다.")
            continue

        p = model.get_part(part_name)
        node_ids = item.get("node_ids", np.array([], dtype=np.int64))
        if len(node_ids) == 0:
            issues.append(f"경고: 고정 경계조건 '{item.get('name', 'FIX')}'에 할당된 절점이 0개입니다.")
            continue

        total_fixed_nodes += len(node_ids)
        if p is not None:
            all_fixed_node_coords.append(p.nodes[node_ids])

    if total_fixed_nodes == 0:
        issues.append("경고: 고정된 경계조건 절점(Fixed Nodes)이 0개입니다. 강체 운동(Rigid Body Motion)으로 솔버가 실패합니다.")
    elif total_fixed_nodes < 3:
        issues.append(f"경고: 고정된 절점 수({total_fixed_nodes}개)가 3개 미만으로 강체 회전을 방지하기에 불충분할 수 있습니다.")
    else:
        # Check collinearity if few fixed points
        if total_fixed_nodes <= 10 and all_fixed_node_coords:
            pts = np.vstack(all_fixed_node_coords)
            # SVD of centered points
            centered = pts - np.mean(pts, axis=0)
            _, s, _ = np.linalg.svd(centered)
            if len(s) >= 2 and s[1] < 1e-4 * s[0]:
                issues.append("경고: 고정 절점들이 거의 일직선상에 위치하여 축 회전 강체 운동이 발생할 위험이 있습니다.")

    # 2. 하중 조건 검사
    if not bc.loads:
        issues.append("오류: 정의된 하중 조건(Loads)이 존재하지 않습니다.")

    for item in bc.loads:
        load_name = item.get("name", "LOAD")
        part_name = item.get("part", "")
        if part_name not in part_names:
            issues.append(f"오류: 하중 '{load_name}'이 지정한 파트 '{part_name}'가 모델에 존재하지 않습니다.")
            continue

        node_ids = item.get("node_ids", np.array([], dtype=np.int64))
        if len(node_ids) == 0:
            issues.append(f"오류: 하중 '{load_name}'이 적용될 절점이 0개입니다.")

        f_vec = np.asarray(item.get("force_N", (0.0, 0.0, 0.0)), dtype=np.float64)
        mag = np.linalg.norm(f_vec)
        if mag < 1e-6:
            issues.append(f"오류: 하중 '{load_name}'의 합력 크기(Force magnitude)가 0 N입니다.")

    return issues
