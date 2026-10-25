"""
io_frd.py
CalculiX .frd result file I/O utilities for fatigue analysis.
Extracts 6-component stress tensors [SXX, SYY, SZZ, SXY, SYZ, SZX] across cyclic load steps.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from fea.results import FrdBlock, FrdData
from postprocess.frd_parser import parse_frd



def _reorder_stress_columns(
    block: FrdBlock,
) -> np.ndarray:
    """Ensure stress tensor array columns follow standard [SXX, SYY, SZZ, SXY, SYZ, SZX] order."""
    expected_order = ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"]
    vals = block.values

    if vals.shape[1] < 6:
        raise ValueError(
            f"STRESS block requires at least 6 components, got {vals.shape[1]} (components: {block.component_names})"
        )

    # Check if component_names match standard order
    names_clean = [c.strip().upper() for c in block.component_names[:6]]
    if names_clean == expected_order:
        return vals[:, :6]

    # Map column indices if names are present
    col_indices = []
    for comp in expected_order:
        if comp in names_clean:
            col_indices.append(names_clean.index(comp))
        else:
            # Fallback to positional columns 0..5
            return vals[:, :6]

    return vals[:, col_indices]


def _align_nodal_data(
    node_ids_1: np.ndarray,
    vals_1: np.ndarray,
    node_ids_2: np.ndarray,
    vals_2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two nodal arrays to a common node sequence."""
    if np.array_equal(node_ids_1, node_ids_2):
        return node_ids_1, vals_1, vals_2

    # Find common node IDs
    common_nodes = np.intersect1d(node_ids_1, node_ids_2)
    if len(common_nodes) == 0:
        raise ValueError("No common node IDs found between the two result sets.")

    # Sort and index match
    idx1 = np.searchsorted(node_ids_1, common_nodes, sorter=np.argsort(node_ids_1))
    idx2 = np.searchsorted(node_ids_2, common_nodes, sorter=np.argsort(node_ids_2))

    s1_sorted = np.argsort(node_ids_1)
    s2_sorted = np.argsort(node_ids_2)

    return common_nodes, vals_1[s1_sorted[idx1]], vals_2[s2_sorted[idx2]]


def _resolve_frd_data(
    source: Union[str, Path, FrdData],
) -> FrdData:
    """Load FrdData from filepath or return directly if already parsed."""
    if isinstance(source, FrdData):
        return source
    p = Path(source)
    if not p.is_file():
        raise FileNotFoundError(f"CalculiX .frd file not found: {p}")
    return parse_frd(str(p))



def extract_stress_from_frd(
    frd_source: Union[str, Path, FrdData],
    which: Union[str, int] = "last",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract nodal IDs and 6-component stress tensor from a CalculiX .frd file.

    Args:
        frd_source: Path to .frd file or pre-parsed FrdData instance.
        which: 'last' for final increment, 'first' for first, or integer index.

    Returns:
        Tuple of:
            node_ids: (N,) int64 array of 1-based global node IDs.
            stress6: (N, 6) float64 array of [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.
    """
    data = _resolve_frd_data(frd_source)
    block = data.get_block("STRESS", which=which)
    if block is None:
        raise KeyError(f"No STRESS block found in .frd results for which='{which}'")

    stress6 = _reorder_stress_columns(block)
    return block.node_ids, stress6


def extract_cyclic_stresses_from_frd(
    frd_source: Union[str, Path, FrdData],
    step_max: Union[int, str] = "last",
    step_min: Union[int, str] = "first",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract maximum and minimum cyclic stress tensors from a multi-increment/step .frd file.

    Args:
        frd_source: Path to .frd file or pre-parsed FrdData instance.
        step_max: Identifier for maximum load step/increment ('last' or int step).
        step_min: Identifier for minimum load step/increment ('first' or int step).

    Returns:
        Tuple of:
            node_ids: (N,) aligned 1-based node IDs.
            stress_max: (N, 6) stress tensor at maximum load step.
            stress_min: (N, 6) stress tensor at minimum load step.
    """
    data = _resolve_frd_data(frd_source)

    node_ids_max, s_max = extract_stress_from_frd(data, which=step_max)
    node_ids_min, s_min = extract_stress_from_frd(data, which=step_min)

    node_ids, aligned_max, aligned_min = _align_nodal_data(
        node_ids_max, s_max, node_ids_min, s_min
    )
    return node_ids, aligned_max, aligned_min


def extract_cyclic_stresses_from_two_frds(
    frd_max_source: Union[str, Path, FrdData],
    frd_min_source: Union[str, Path, FrdData],
    which_max: Union[int, str] = "last",
    which_min: Union[int, str] = "last",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and align cyclic stress tensors from two separate .frd files.

    Args:
        frd_max_source: .frd file for the maximum load case.
        frd_min_source: .frd file for the minimum load case.
        which_max: Step identifier for maximum load case.
        which_min: Step identifier for minimum load case.

    Returns:
        Tuple of:
            node_ids: (N,) aligned 1-based node IDs.
            stress_max: (N, 6) stress tensor at maximum load step.
            stress_min: (N, 6) stress tensor at minimum load step.
    """
    node_ids_max, s_max = extract_stress_from_frd(frd_max_source, which=which_max)
    node_ids_min, s_min = extract_stress_from_frd(frd_min_source, which=which_min)

    node_ids, aligned_max, aligned_min = _align_nodal_data(
        node_ids_max, s_max, node_ids_min, s_min
    )
    return node_ids, aligned_max, aligned_min


def extract_cyclic_stresses_with_ratio(
    frd_source: Union[str, Path, FrdData],
    stress_ratio_R: float = 0.1,
    which: Union[int, str] = "last",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract maximum stress tensor and derive minimum cyclic stress via constant stress ratio R.

    Formula:
        stress_min = stress_max * stress_ratio_R

    Args:
        frd_source: Path to .frd file or pre-parsed FrdData instance.
        stress_ratio_R: Cycle stress ratio R = sigma_min / sigma_max (typically 0.1 for dental fatigue).
        which: Step identifier in .frd file.

    Returns:
        Tuple of:
            node_ids: (N,) 1-based node IDs.
            stress_max: (N, 6) stress tensor at peak load.
            stress_min: (N, 6) derived stress tensor at trough load.
    """
    node_ids, s_max = extract_stress_from_frd(frd_source, which=which)
    s_min = s_max * float(stress_ratio_R)
    return node_ids, s_max, s_min
