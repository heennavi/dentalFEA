"""
surface_extractor.py
Boundary surface extraction, geometric region filtering, and contact pair detection for FEA models.

[CalculiX Tetrahedron Face Conventions]
- S1: Nodes 1-2-3 (0-based: 0, 1, 2)
- S2: Nodes 1-4-2 (0-based: 0, 3, 1)
- S3: Nodes 2-4-3 (0-based: 1, 3, 2)
- S4: Nodes 3-4-1 (0-based: 2, 3, 0)
"""

from typing import Tuple, Callable, Optional, Dict, List
import numpy as np
from scipy.spatial import cKDTree

from fea.models import MeshPart, TieDef

# CalculiX 1-based tetrahedron face connectivity (0-based indices) and opposite apex index
CALCULIX_TET_FACES: Dict[int, Tuple[Tuple[int, int, int], int]] = {
    1: ((0, 1, 2), 3),  # S1: 1-2-3, opposite apex: 4 (idx 3)
    2: ((0, 3, 1), 2),  # S2: 1-4-2, opposite apex: 3 (idx 2)
    3: ((1, 3, 2), 0),  # S3: 2-4-3, opposite apex: 1 (idx 0)
    4: ((2, 3, 0), 1),  # S4: 3-4-1, opposite apex: 2 (idx 1)
}


def boundary_faces(
    elements: np.ndarray,
    nodes: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract boundary faces from a tetrahedral mesh (faces belonging to exactly one element).

    Follows CalculiX solid tetrahedron face numbering conventions (S1 to S4).
    Ensures that the returned (K, 3) vertex indices form outward-pointing normal vectors.

    Args:
        elements: (M, 4 or 10) int64 array of tetrahedral element nodal connectivity.
        nodes: Optional (N, 3) float64 array of node coordinates. If provided, outward
               normal directions are verified and guaranteed with exact nodal coordinates.

    Returns:
        faces_nodes: (K, 3) int64 array of triangular face vertex indices (outward-oriented).
        face_info: (K, 2) int64 array of [element_idx, face_id (1~4)].
    """
    n_elems = len(elements)
    if n_elems == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    # Use first 4 corner vertices for linear or quadratic elements
    elems4 = elements[:, :4]

    # Preallocate list for all 4 faces of each element
    # face_canonical_key -> (count, element_idx, face_id, face_nodes)
    face_map: Dict[Tuple[int, int, int], List[Any]] = {}

    for elem_idx in range(n_elems):
        el = elems4[elem_idx]
        for face_id, (local_idx, opp_idx) in CALCULIX_TET_FACES.items():
            fnodes = [int(el[local_idx[0]]), int(el[local_idx[1]]), int(el[local_idx[2]])]
            key = tuple(sorted(fnodes))
            opp_node = int(el[opp_idx])

            if key not in face_map:
                face_map[key] = [1, elem_idx, face_id, fnodes, opp_node]
            else:
                face_map[key][0] += 1

    # Filter boundary faces (count == 1)
    boundary_list = [val for val in face_map.values() if val[0] == 1]
    n_boundary = len(boundary_list)

    if n_boundary == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    faces_nodes = np.empty((n_boundary, 3), dtype=np.int64)
    face_info = np.empty((n_boundary, 2), dtype=np.int64)

    for i, (_, elem_idx, face_id, fnodes, opp_node) in enumerate(boundary_list):
        face_info[i] = [elem_idx, face_id]

        if nodes is not None:
            # Determine outward orientation using coordinate geometry
            p0 = nodes[fnodes[0]]
            p1 = nodes[fnodes[1]]
            p2 = nodes[fnodes[2]]
            p_opp = nodes[opp_node]

            normal = np.cross(p1 - p0, p2 - p0)
            f_center = (p0 + p1 + p2) / 3.0
            outward_vec = f_center - p_opp

            if np.dot(normal, outward_vec) < 0.0:
                # Reverse winding order to point outward
                faces_nodes[i] = [fnodes[0], fnodes[2], fnodes[1]]
            else:
                faces_nodes[i] = fnodes
        else:
            # Standard positive-volume tetrahedral outward convention:
            # Face vertices local_idx of S1~S4 with V>0 require swapping vertex 1 and 2 to point outward
            faces_nodes[i] = [fnodes[0], fnodes[2], fnodes[1]]

    return faces_nodes, face_info


# --- Geometric Region Predicate Factories ---

def plane_below(point: Tuple[float, float, float] | np.ndarray, normal: Tuple[float, float, float] | np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predicate testing if centroids lie on or below a plane (dot(p - p0, n) <= 0)."""
    p0 = np.asarray(point, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n), 1e-12)
    return lambda centroids: np.dot(centroids - p0, n) <= 0.0


def plane_above(point: Tuple[float, float, float] | np.ndarray, normal: Tuple[float, float, float] | np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predicate testing if centroids lie on or above a plane (dot(p - p0, n) >= 0)."""
    p0 = np.asarray(point, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n), 1e-12)
    return lambda centroids: np.dot(centroids - p0, n) >= 0.0


def sphere(center: Tuple[float, float, float] | np.ndarray, radius: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predicate testing if centroids lie inside a sphere (dist(p, c) <= r)."""
    c = np.asarray(center, dtype=np.float64)
    r2 = float(radius) ** 2
    return lambda centroids: np.sum((centroids - c) ** 2, axis=1) <= r2


def cylinder(
    p1: Tuple[float, float, float] | np.ndarray,
    p2: Tuple[float, float, float] | np.ndarray,
    radius: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predicate testing if centroids lie inside a finite cylinder segment."""
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    ab = b - a
    len_ab2 = np.dot(ab, ab)
    r2 = float(radius) ** 2

    def _test(centroids: np.ndarray) -> np.ndarray:
        ap = centroids - a
        t = np.dot(ap, ab) / max(len_ab2, 1e-12)
        in_segment = (t >= 0.0) & (t <= 1.0)
        proj = a + t[:, np.newaxis] * ab
        dist2 = np.sum((centroids - proj) ** 2, axis=1)
        return in_segment & (dist2 <= r2)

    return _test


def faces_in_region(
    nodes: np.ndarray,
    faces: np.ndarray,
    predicate: Callable[[np.ndarray], np.ndarray],
    return_indices: bool = True
) -> np.ndarray:
    """
    Select triangular faces whose centroids satisfy the given geometric predicate.

    Args:
        nodes: (N, 3) float64 array of node coordinates.
        faces: (K, 3) int64 array of face vertex indices.
        predicate: Callable taking (K, 3) centroid coordinates and returning (K,) bool array.
        return_indices: If True, returns (S,) int64 array of indices into faces.
                        If False, returns (S, 3) int64 array of selected faces.

    Returns:
        np.ndarray: Selected face indices or selected face vertex connectivity.
    """
    if len(faces) == 0:
        return np.array([], dtype=np.int64) if return_indices else np.empty((0, 3), dtype=np.int64)

    v0 = nodes[faces[:, 0]]
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0

    mask = predicate(centroids)
    selected_indices = np.where(mask)[0]

    if return_indices:
        return selected_indices
    return faces[selected_indices]


def contact_pair_from_parts(
    part_a: MeshPart,
    part_b: MeshPart,
    tol_mm: float,
    master: str = "auto",
    name: str = "tie_contact"
) -> TieDef:
    """
    Detect contact interface between two mesh parts within distance tolerance tol_mm.

    Identifies master boundary faces (element-wise) and slave boundary nodes using k-d tree proximity search.

    Master Selection Strategy:
        - "auto": Assigns the part with fewer elements (coarser mesh) as master,
                  and the more refined part as slave for superior contact kinematics.
        - "a": Forces part_a as master, part_b as slave.
        - "b": Forces part_b as master, part_a as slave.

    Args:
        part_a: First MeshPart instance.
        part_b: Second MeshPart instance.
        tol_mm: Distance tolerance threshold in mm.
        master: Master assignment mode ("auto", "a", or "b").
        name: Name of the contact tie definition.

    Returns:
        TieDef: CalculiX tie constraint definition with master faces and slave node ids.
    """
    if master == "auto":
        is_master_a = (len(part_a.elements) <= len(part_b.elements))
    elif master.lower() == "a":
        is_master_a = True
    elif master.lower() == "b":
        is_master_a = False
    else:
        raise ValueError(f"Invalid master option: '{master}'. Allowed: 'auto', 'a', 'b'")

    m_part = part_a if is_master_a else part_b
    s_part = part_b if is_master_a else part_a

    # 1. Master 경계면 추출
    m_faces_nodes, m_faces_info = boundary_faces(m_part.elements, nodes=m_part.nodes)

    # 2. Slave 경계 절점 추출
    s_faces_nodes, _ = boundary_faces(s_part.elements, nodes=s_part.nodes)
    if len(m_faces_nodes) == 0 or len(s_faces_nodes) == 0:
        return TieDef(
            name=name,
            master_part=m_part.name,
            slave_part=s_part.name,
            master_faces=np.empty((0, 2), dtype=np.int64),
            slave_nodes=np.empty((0,), dtype=np.int64),
            position_tolerance_mm=tol_mm,
        )

    s_boundary_node_ids = np.unique(s_faces_nodes)
    s_boundary_coords = s_part.nodes[s_boundary_node_ids]

    # 3. Master 경계면 중심점 계산
    v0 = m_part.nodes[m_faces_nodes[:, 0]]
    v1 = m_part.nodes[m_faces_nodes[:, 1]]
    v2 = m_part.nodes[m_faces_nodes[:, 2]]
    m_centroids = (v0 + v1 + v2) / 3.0

    # 4. cKDTree 근접 거리 질의: slave boundary nodes -> master face centroids
    tree = cKDTree(m_centroids)
    dists, nearest_m_face_idx = tree.query(s_boundary_coords, distance_upper_bound=tol_mm)

    valid_mask = dists <= tol_mm
    matched_slave_nodes = s_boundary_node_ids[valid_mask]
    matched_master_face_idx = np.unique(nearest_m_face_idx[valid_mask])

    if len(matched_slave_nodes) == 0:
        return TieDef(
            name=name,
            master_part=m_part.name,
            slave_part=s_part.name,
            master_faces=np.empty((0, 2), dtype=np.int64),
            slave_nodes=np.empty((0,), dtype=np.int64),
            position_tolerance_mm=tol_mm,
        )

    master_faces = m_faces_info[matched_master_face_idx]

    return TieDef(
        name=name,
        master_part=m_part.name,
        slave_part=s_part.name,
        master_faces=master_faces,
        slave_nodes=matched_slave_nodes,
        position_tolerance_mm=tol_mm,
    )
