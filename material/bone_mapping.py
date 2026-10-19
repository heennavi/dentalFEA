"""
bone_mapping.py
Heterogeneous bone material assignment and HU-density-elasticity empirical mapping.

[Units]
- Length: mm
- Elastic Modulus: MPa (N/mm^2)
- Density: g/cm^3 or tonne/mm^3
- Video Density: Hounsfield Unit (HU)
"""

import os
import json
import logging
from typing import Sequence, Optional, Dict, Any, Callable
import numpy as np
from scipy.ndimage import map_coordinates

from fea.models import MeshPart, MaterialDef, MaterialAssignment

logger = logging.getLogger(__name__)

DEFAULT_PRESET_PATH = os.path.join(os.path.dirname(__file__), "presets.json")


def load_preset(path: Optional[str] = None) -> Dict[str, Any]:
    """Load empirical material mapping presets from JSON."""
    preset_file = path or DEFAULT_PRESET_PATH
    with open(preset_file, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Formula Evaluation Engine ---

def _eval_linear(cfg: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    a = float(cfg.get("a", 0.0))
    b = float(cfg.get("b", 1.0))
    return a + b * x


def _eval_power(cfg: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    c = float(cfg.get("c", 1.0))
    d = float(cfg.get("d", 1.0))
    # Prevent negative values before raising to power
    non_neg_x = np.maximum(x, 0.0)
    return c * (non_neg_x ** d)


FORMULA_REGISTRY: Dict[str, Callable[[Dict[str, Any], np.ndarray], np.ndarray]] = {
    "linear": _eval_linear,
    "power": _eval_power,
}


def evaluate_formula(formula_cfg: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """
    Evaluate empirical conversion formula (linear, power, etc.).

    Args:
        formula_cfg: Dictionary containing "type" and model coefficients.
        x: Input numpy array.

    Returns:
        np.ndarray: Converted output array.
    """
    f_type = formula_cfg.get("type", "linear").lower()
    if f_type not in FORMULA_REGISTRY:
        raise ValueError(f"Unsupported formula type: '{f_type}'. Available: {list(FORMULA_REGISTRY.keys())}")
    return FORMULA_REGISTRY[f_type](formula_cfg, x)


# --- Coordinate Transformation & Sampling ---

def world_to_index(
    points_xyz: np.ndarray,
    origin: Sequence[float],
    spacing: Sequence[float],
    direction: Optional[Sequence[float] | np.ndarray] = None
) -> np.ndarray:
    """
    Convert continuous world coordinates (mm, LPS) to continuous image voxel indices.

    Accounts for origin, spacing, and 3x3 direction cosine matrix:
        p_index = (1 / spacing) * (D^-1 * (p_world - origin))

    Args:
        points_xyz: (N, 3) float64 array of coordinates in mm.
        origin: (3,) sequence of volume origin in mm.
        spacing: (3,) sequence of voxel spacing in mm.
        direction: Optional 9-sequence or (3, 3) direction cosine matrix.

    Returns:
        (N, 3) float64 array of continuous voxel indices [i, j, k] corresponding to (x, y, z).
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    orig = np.asarray(origin, dtype=np.float64)
    spc = np.asarray(spacing, dtype=np.float64)

    if direction is None:
        d_inv = np.eye(3, dtype=np.float64)
    else:
        d_mat = np.asarray(direction, dtype=np.float64)
        if d_mat.size == 9:
            d_mat = d_mat.reshape((3, 3))
        # For orthogonal direction matrices, inverse equals transpose
        try:
            d_inv = np.linalg.inv(d_mat)
        except np.linalg.LinAlgError:
            d_inv = d_mat.T

    shifted = pts - orig
    rotated = shifted @ d_inv.T
    idx_xyz = rotated / spc
    return idx_xyz


def sample_hu(
    volume_array: np.ndarray,
    points_xyz: np.ndarray,
    origin: Sequence[float],
    spacing: Sequence[float],
    direction: Optional[Sequence[float] | np.ndarray] = None,
    outside_hu: float = -1000.0,
) -> np.ndarray:
    """
    Sample continuous CT HU values at 3D world coordinates using trilinear interpolation.

    Note on array axis order:
        volume_array has shape (nz, ny, nx).
        Continuous indices [i, j, k] map to (x, y, z) axes, hence coordinates passed
        to map_coordinates must be in (z, y, x) = (k, j, i) order.

    Args:
        volume_array: (nz, ny, nx) 3D numpy array of HU scalars.
        points_xyz: (N, 3) float64 array of sample coordinates in mm.
        origin: Volume origin (ox, oy, oz).
        spacing: Voxel spacing (sx, sy, sz).
        direction: Direction matrix.
        outside_hu: Scalar value assigned to points outside volume bounds (default: -1000.0 HU).

    Returns:
        (N,) float64 array of interpolated HU values.
    """
    idx_xyz = world_to_index(points_xyz, origin, spacing, direction)
    nz, ny, nx = volume_array.shape

    # Check bounds: i in [0, nx-1], j in [0, ny-1], k in [0, nz-1]
    i = idx_xyz[:, 0]
    j = idx_xyz[:, 1]
    k = idx_xyz[:, 2]

    out_of_bounds = (i < 0.0) | (i > nx - 1) | (j < 0.0) | (j > ny - 1) | (k < 0.0) | (k > nz - 1)
    n_out = int(np.sum(out_of_bounds))
    if n_out > 0:
        logger.warning(
            f"sample_hu: {n_out}/{len(points_xyz)} points fall outside the CT volume bounds; "
            f"assigned outside_hu={outside_hu}."
        )

    # map_coordinates expects coordinates in axis order: (axis 0: Z, axis 1: Y, axis 2: X)
    coords = [k, j, i]
    sampled = map_coordinates(
        volume_array,
        coords,
        order=1,
        mode="constant",
        cval=outside_hu,
        prefilter=False
    )

    if n_out > 0:
        sampled[out_of_bounds] = outside_hu

    return sampled


def element_centroids(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Compute 3D centroid coordinates of tetrahedral solid elements.

    Evaluates the mean of the first 4 corner vertices.

    Args:
        nodes: (N, 3) float64 array of nodal coordinates.
        elements: (M, 4 or 10) int64 element connectivity array.

    Returns:
        (M, 3) float64 array of centroid coordinates in mm.
    """
    v0 = nodes[elements[:, 0]]
    v1 = nodes[elements[:, 1]]
    v2 = nodes[elements[:, 2]]
    v3 = nodes[elements[:, 3]]
    return (v0 + v1 + v2 + v3) / 4.0


# --- Bone & Titanium Material Assignment ---

def assign_bone_materials(
    part: MeshPart,
    volume_array: np.ndarray,
    geometry_info: Dict[str, Any],
    cortical_mask_array: Optional[np.ndarray] = None,
    preset: Optional[Dict[str, Any]] = None,
) -> MaterialAssignment:
    """
    Assign heterogeneous bone material properties based on CT HU sampling and binning.

    Workflow:
        1. Compute element centroids.
        2. Sample HU values from volume_array.
        3. If cortical_mask_array is provided, mark elements whose centroid is inside cortical bone
           as fixed cortical material (cortical_fixed).
        4. For remaining elements, convert HU -> density -> Young's modulus E.
        5. Clip E to [E_min, E_max].
        6. Quantize E into n_bins discrete bins to avoid creating an intractable number of materials.
        7. Assemble MaterialAssignment instance.

    Args:
        part: Bone MeshPart instance.
        volume_array: (nz, ny, nx) CT HU numpy array.
        geometry_info: Dict containing "origin", "spacing", and optionally "direction".
        cortical_mask_array: Optional (nz, ny, nx) bool mask of cortical bone.
        preset: Configuration dictionary (or None to load default presets.json).

    Returns:
        MaterialAssignment: Mapping of elements to defined MaterialDef instances.
    """
    cfg = preset or load_preset()

    origin = geometry_info.get("origin", (0.0, 0.0, 0.0))
    spacing = geometry_info.get("spacing", (1.0, 1.0, 1.0))
    direction = geometry_info.get("direction", None)

    # 1. Centroids & HU Sampling
    centroids = element_centroids(part.nodes, part.elements)
    hu_vals = sample_hu(volume_array, centroids, origin, spacing, direction)

    n_elems = len(part.elements)
    is_cortical = np.zeros(n_elems, dtype=bool)

    # 2. Cortical Mask Evaluation
    if cortical_mask_array is not None:
        idx_xyz = world_to_index(centroids, origin, spacing, direction)
        coords = [idx_xyz[:, 2], idx_xyz[:, 1], idx_xyz[:, 0]]
        mask_sampled = map_coordinates(
            cortical_mask_array.astype(np.float32),
            coords,
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False
        )
        is_cortical = (mask_sampled > 0.5)

    # 3. HU -> Density -> Modulus (for cancellous / non-cortical elements)
    densities = evaluate_formula(cfg["density_from_hu"], hu_vals)
    moduli = evaluate_formula(cfg["modulus_from_density"], densities)

    # 4. Clipping
    e_min = float(cfg.get("E_min", 10.0))
    e_max = float(cfg.get("E_max", 20000.0))
    moduli_clipped = np.clip(moduli, e_min, e_max)

    # 5. Binning / Quantization
    n_bins = int(cfg.get("n_bins", 30))
    scale_type = cfg.get("binning_scale", "linear").lower()

    if scale_type == "log":
        bin_edges = np.geomspace(e_min, e_max, n_bins + 1)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
    else:
        bin_edges = np.linspace(e_min, e_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Arithmetic mean

    # Digitize into 0-based bin indices [0, n_bins - 1]
    raw_bin_ids = np.digitize(moduli_clipped, bin_edges[1:-1])

    # Find unique active bins to minimize material count in CalculiX input deck
    active_bins = np.unique(raw_bin_ids[~is_cortical]) if np.any(~is_cortical) else np.array([], dtype=int)

    materials: list[MaterialDef] = []
    bin_to_mat_idx: Dict[int, int] = {}

    poisson_bone = float(cfg.get("poisson_bone", 0.3))

    for b_id in active_bins:
        mat_idx = len(materials)
        bin_to_mat_idx[int(b_id)] = mat_idx
        mat_name = f"BONE_{b_id:03d}"
        materials.append(MaterialDef(
            name=mat_name,
            E=float(bin_centers[b_id]),
            nu=poisson_bone
        ))

    elem_material_indices = np.empty(n_elems, dtype=np.int32)
    for b_id in active_bins:
        match_mask = (~is_cortical) & (raw_bin_ids == b_id)
        elem_material_indices[match_mask] = bin_to_mat_idx[int(b_id)]

    # 6. Cortical Fixed Material
    if np.any(is_cortical):
        cortical_cfg = cfg.get("cortical_fixed", {"E": 13700.0, "nu": 0.3})
        cortical_mat_idx = len(materials)
        materials.append(MaterialDef(
            name="CORTICAL",
            E=float(cortical_cfg["E"]),
            nu=float(cortical_cfg["nu"])
        ))
        elem_material_indices[is_cortical] = cortical_mat_idx

    return MaterialAssignment(
        element_material={part.name: elem_material_indices},
        materials=materials
    )


def titanium_assignment(
    part: MeshPart,
    preset: Optional[Dict[str, Any]] = None
) -> MaterialAssignment:
    """
    Assign homogeneous titanium alloy material to implant/abutment parts.

    Args:
        part: MeshPart instance.
        preset: Configuration dictionary.

    Returns:
        MaterialAssignment: Homogeneous assignment referencing a single TITANIUM MaterialDef.
    """
    cfg = preset or load_preset()
    ti_cfg = cfg.get("titanium", {"E": 110000.0, "nu": 0.34, "density": 4.5e-9})

    mat = MaterialDef(
        name="TITANIUM",
        E=float(ti_cfg["E"]),
        nu=float(ti_cfg["nu"]),
        density=float(ti_cfg.get("density", 4.5e-9))
    )

    n_elems = len(part.elements)
    return MaterialAssignment(
        element_material={part.name: np.zeros(n_elems, dtype=np.int32)},
        materials=[mat]
    )
