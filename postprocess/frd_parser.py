"""
frd_parser.py
High-performance parser for CalculiX ASCII .frd result files.

Parses nodal coordinates and result blocks (displacements, stresses, strains,
reaction forces) using fixed-width slicing and vectorized numpy conversion.
"""

import os
import re
import logging
from typing import List, Tuple, Optional
import numpy as np

from fea.results import FrdBlock, FrdData

logger = logging.getLogger(__name__)

# Fortran compact exponent pattern (e.g. 1.23456-100 -> 1.23456E-100)
_RE_COMPACT_EXP = re.compile(r"([0-9])([+-][0-9]{2,3})$")


def _safe_float(s: str) -> float:
    """Safely parse a single float string chunk, converting invalid/inf/overflow (>1e30) values to NaN."""
    s = s.strip()
    if not s:
        return np.nan
    try:
        val = float(s.replace("D", "E").replace("d", "e"))
        if np.isinf(val) or abs(val) > 1e30:
            return np.nan
        return val
    except ValueError:
        # Check compact Fortran notation like "1.23456-100"
        m = _RE_COMPACT_EXP.search(s)
        if m:
            s_fix = s[: m.start(2)] + "E" + m.group(2)
            try:
                val = float(s_fix)
                if np.isinf(val) or abs(val) > 1e30:
                    return np.nan
                return val
            except ValueError:
                pass
        return np.nan


def _safe_float_col(str_list: List[str]) -> np.ndarray:
    """
    Vectorized float array conversion with fallback to element-wise safe parsing.

    Args:
        str_list: List of fixed-width string representations of numbers.

    Returns:
        np.ndarray: 1D float64 array with inf/errors/overflows (>1e30) replaced by NaN.
    """
    try:
        arr = np.asarray(str_list, dtype=np.float64)
        bad_mask = np.isinf(arr) | (np.abs(arr) > 1e30)
        if np.any(bad_mask):
            arr = arr.copy()
            arr[bad_mask] = np.nan
        return arr
    except (ValueError, TypeError):
        # Fallback for irregular exponent formats or corrupted fields
        out = np.empty(len(str_list), dtype=np.float64)
        for idx, s in enumerate(str_list):
            out[idx] = _safe_float(s)
        return out


def _safe_int_array(str_list: List[str]) -> np.ndarray:
    """Parse a list of integer strings into a 1D int64 array."""
    try:
        return np.asarray(str_list, dtype=np.int64)
    except (ValueError, TypeError):
        out = np.zeros(len(str_list), dtype=np.int64)
        for idx, s in enumerate(str_list):
            s_clean = s.strip()
            out[idx] = int(s_clean) if s_clean else 0
        return out


def _parse_node_block(node_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse node coordinate block into node IDs and LPS coordinate array."""
    if not node_lines:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

    node_ids = _safe_int_array([l[3:13] for l in node_lines])
    x = _safe_float_col([l[13:25] for l in node_lines])
    y = _safe_float_col([l[25:37] for l in node_lines])
    z = _safe_float_col([l[37:49] for l in node_lines])
    coords = np.column_stack([x, y, z])
    return node_ids, coords


def _parse_result_block(
    name: str,
    time: float,
    step_idx: int,
    components: List[str],
    data_lines: List[str],
    cont_lines: List[str],
) -> FrdBlock:
    """
    Construct FrdBlock from extracted result records with dynamic component resolution.

    Args:
        name: Name of result block (e.g. 'DISP', 'STRESS').
        time: Step time.
        step_idx: Block sequence index.
        components: Declared component names from -5 cards.
        data_lines: Primary records starting with ' -1'.
        cont_lines: Continuation records starting with ' -2'.

    Returns:
        FrdBlock: Parsed result block object.
    """
    if not data_lines:
        return FrdBlock(
            name=name,
            time=time,
            step=step_idx,
            component_names=[],
            node_ids=np.zeros(0, dtype=np.int64),
            values=np.zeros((0, 0), dtype=np.float64),
        )

    # 1. Parse node IDs (cols 4-13)
    node_ids = _safe_int_array([l[3:13] for l in data_lines])

    # 2. Determine actual component count from the first line
    l0 = data_lines[0]
    n_first = 0
    for i in range(6):
        start = 13 + 12 * i
        end = start + 12
        if len(l0) >= start and l0[start:end].strip():
            n_first += 1
        else:
            break

    # Check continuation line if present
    has_cont = len(cont_lines) == len(data_lines)
    n_cont = 0
    if has_cont and len(cont_lines) > 0:
        c0 = cont_lines[0]
        for i in range(6):
            start = 13 + 12 * i
            end = start + 12
            if len(c0) >= start and c0[start:end].strip():
                n_cont += 1
            else:
                break

    actual_n_comp = n_first + n_cont

    # 3. Reconcile component names (handling virtual 'ALL' component in DISP/FORC)
    comp_names = [c for c in components if c]
    has_all = any(c.strip().upper() == "ALL" for c in comp_names)
    if has_all and len(comp_names) == actual_n_comp + 1:
        comp_names = [c for c in comp_names if c.strip().upper() != "ALL"]
    elif len(comp_names) > actual_n_comp:
        comp_names = comp_names[:actual_n_comp]
    elif len(comp_names) < actual_n_comp:
        comp_names.extend([f"C{i + 1}" for i in range(len(comp_names), actual_n_comp)])

    # 4. Extract values column-by-column using fast vectorized slicing
    cols: List[np.ndarray] = []
    for i in range(n_first):
        start = 13 + 12 * i
        end = start + 12
        col_strs = [l[start:end] for l in data_lines]
        cols.append(_safe_float_col(col_strs))

    if n_cont > 0 and has_cont:
        for i in range(n_cont):
            start = 13 + 12 * i
            end = start + 12
            col_strs = [l[start:end] for l in cont_lines]
            cols.append(_safe_float_col(col_strs))

    if cols:
        values = np.column_stack(cols)
    else:
        values = np.zeros((len(node_ids), 0), dtype=np.float64)

    return FrdBlock(
        name=name,
        time=time,
        step=step_idx,
        component_names=comp_names,
        node_ids=node_ids,
        values=values,
    )


def parse_frd_lines(lines: List[str]) -> FrdData:
    """
    Parse a list of raw .frd file lines into FrdData structure.

    Args:
        lines: List of line strings from .frd.

    Returns:
        FrdData: Parsed node coordinates and result blocks.
    """
    node_lines: List[str] = []
    blocks: List[FrdBlock] = []

    in_nodes = False
    in_results = False
    in_elements = False

    current_time = 0.0
    current_name = ""
    current_components: List[str] = []
    current_data_lines: List[str] = []
    current_cont_lines: List[str] = []
    step_counter = 0

    for line in lines:
        if not line:
            continue

        prefix_3 = line[:3]
        prefix_6 = line[:6]
        prefix_7 = line[:7]

        # 1. Node block start
        if prefix_6 == "    2C" or line.startswith("2C"):
            in_nodes = True
            in_results = False
            in_elements = False
            node_lines = []
            continue

        # 2. Element block start (skip content)
        if prefix_6 == "    3C" or line.startswith("3C"):
            in_elements = True
            in_nodes = False
            in_results = False
            continue

        # 3. Result block start (100C or 100CL)
        if prefix_6 == "  100C" or prefix_7 == "  100CL" or line.strip().startswith("100C"):
            in_results = True
            in_nodes = False
            in_elements = False
            # Parse step time from 3rd token: '  100CL  101 1.000000000 ...'
            parts = line.split()
            if len(parts) >= 3:
                current_time = _safe_float(parts[2])
                if np.isnan(current_time):
                    current_time = 0.0
            else:
                current_time = 0.0
            current_name = ""
            current_components = []
            current_data_lines = []
            current_cont_lines = []
            continue

        # 4. Block terminator (-3)
        if prefix_3 == " -3":
            if in_nodes:
                in_nodes = False
            elif in_results:
                in_results = False
                block = _parse_result_block(
                    name=current_name,
                    time=current_time,
                    step_idx=step_counter,
                    components=current_components,
                    data_lines=current_data_lines,
                    cont_lines=current_cont_lines,
                )
                blocks.append(block)
                step_counter += 1
                current_data_lines = []
                current_cont_lines = []
                current_components = []
            elif in_elements:
                in_elements = False
            continue

        # 5. Node data lines
        if in_nodes:
            if prefix_3 == " -1":
                node_lines.append(line)
            continue

        # 6. Element data lines (skip)
        if in_elements:
            continue

        # 7. Result metadata and data lines
        if in_results:
            if prefix_3 == " -4":
                # Header card: ' -4  DISP        4    1'
                raw_name = line[5:15].strip()
                if not raw_name:
                    parts = line.split()
                    raw_name = parts[1] if len(parts) > 1 else "UNKNOWN"
                current_name = raw_name
            elif prefix_3 == " -5":
                # Component name card: ' -5  D1          1    2    1    0'
                comp_name = line[5:15].strip()
                if not comp_name:
                    parts = line.split()
                    comp_name = parts[1] if len(parts) > 1 else ""
                current_components.append(comp_name)
            elif prefix_3 == " -1":
                current_data_lines.append(line)
            elif prefix_3 == " -2":
                current_cont_lines.append(line)
            continue

    node_ids, node_coords = _parse_node_block(node_lines)

    if len(blocks) == 0:
        logger.warning("CalculiX .frd 파일에 결과 블록이 존재하지 않습니다 (해석 실패 또는 빈 결과).")

    return FrdData(node_ids=node_ids, node_coords=node_coords, blocks=blocks)


def parse_frd_text(text: str) -> FrdData:
    """
    Parse CalculiX .frd content from a string buffer.

    Args:
        text: Complete text content of an ASCII .frd file.

    Returns:
        FrdData: Parsed FrdData container.

    Raises:
        ValueError: If text is empty or contains only whitespace.
    """
    if not text or not text.strip():
        raise ValueError("결과 데이터가 비어 있습니다.")

    lines = text.splitlines()
    return parse_frd_lines(lines)


def parse_frd(path: str) -> FrdData:
    """
    Parse CalculiX .frd result file from local filesystem.

    Args:
        path: Path to .frd result file.

    Returns:
        FrdData: Parsed FrdData container with node coordinates and result blocks.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CalculiX 결과 파일을 찾을 수 없습니다: {path}")

    if os.path.getsize(path) == 0:
        raise ValueError(f"결과 파일이 비어 있습니다: {path}")

    with open(path, "r", encoding="ascii", errors="replace") as f:
        lines = f.readlines()

    return parse_frd_lines(lines)
