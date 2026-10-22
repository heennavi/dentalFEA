"""
result_reader.py
CalculiX ASCII .frd result file parser for nodal displacements and stresses.

Units:
    - Displacement: mm
    - Stress: MPa
"""

import os
import re
from typing import Dict, Any, Optional
import numpy as np


def _parse_float_field(chunk: str) -> float:
    """Parse a fixed-width float string chunk safely, handling exponent notation."""
    chunk = chunk.strip()
    if not chunk:
        return 0.0
    return float(chunk)


def read_frd_displacements(frd_path: str) -> Dict[int, np.ndarray]:
    """
    Read nodal displacements from CalculiX .frd file.

    Args:
        frd_path: Path to the .frd result file.

    Returns:
        Dict[int, np.ndarray]: Mapping from 1-based global node ID to displacement vector
                               [dx, dy, dz] in mm.
    """
    if not os.path.isfile(frd_path):
        raise FileNotFoundError(f"CalculiX result file not found: {frd_path}")

    displacements: Dict[int, np.ndarray] = {}
    in_disp_block = False

    with open(frd_path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue

            # Detect displacement block header
            if "-4  DISP" in line_str:
                in_disp_block = True
                continue

            if in_disp_block:
                # End of current dataset block is indicated by ' -3' or another header
                if line_str.startswith(" -3"):
                    in_disp_block = False
                    continue
                if line_str.startswith(" -4") or line_str.startswith("    1P"):
                    in_disp_block = False
                    continue

                # Header cards inside block (e.g., -5 D1, -5 D2, etc.)
                if line_str.startswith(" -5"):
                    continue

                # Data record: starts with ' -1'
                if line_str.startswith(" -1"):
                    try:
                        # Fixed width format:
                        # cols 1-3: ' -1'
                        # cols 4-13: node ID (10 chars)
                        # cols 14-25: D1 (12 chars)
                        # cols 26-37: D2 (12 chars)
                        # cols 38-49: D3 (12 chars)
                        nid = int(line_str[3:13])
                        dx = _parse_float_field(line_str[13:25])
                        dy = _parse_float_field(line_str[25:37])
                        dz = _parse_float_field(line_str[37:49])
                        displacements[nid] = np.array([dx, dy, dz], dtype=np.float64)
                    except (ValueError, IndexError):
                        # Fallback parsing with regex if column positions vary
                        tokens = line_str[3:].split()
                        if len(tokens) >= 4:
                            nid = int(tokens[0])
                            dx = float(tokens[1])
                            dy = float(tokens[2])
                            dz = float(tokens[3])
                            displacements[nid] = np.array([dx, dy, dz], dtype=np.float64)

    return displacements


def read_frd_stresses(frd_path: str) -> Dict[int, np.ndarray]:
    """
    Read nodal stresses from CalculiX .frd file.

    Args:
        frd_path: Path to the .frd result file.

    Returns:
        Dict[int, np.ndarray]: Mapping from 1-based global node ID to stress tensor components
                               [sxx, syy, szz, sxy, syz, szx] in MPa.
    """
    if not os.path.isfile(frd_path):
        raise FileNotFoundError(f"CalculiX result file not found: {frd_path}")

    stresses: Dict[int, np.ndarray] = {}
    in_stress_block = False

    with open(frd_path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue

            # Detect stress block header
            if "-4  STRESS" in line_str:
                in_stress_block = True
                continue

            if in_stress_block:
                if line_str.startswith(" -3"):
                    in_stress_block = False
                    continue
                if line_str.startswith(" -4") or line_str.startswith("    1P"):
                    in_stress_block = False
                    continue
                if line_str.startswith(" -5"):
                    continue

                if line_str.startswith(" -1"):
                    try:
                        # Fixed width format:
                        # cols 4-13: node ID
                        # cols 14-25: SXX
                        # cols 26-37: SYY
                        # cols 38-49: SZZ
                        # cols 50-61: SXY
                        # cols 62-73: SYZ
                        # cols 74-85: SZX
                        nid = int(line_str[3:13])
                        sxx = _parse_float_field(line_str[13:25])
                        syy = _parse_float_field(line_str[25:37])
                        szz = _parse_float_field(line_str[37:49])
                        sxy = _parse_float_field(line_str[49:61])
                        syz = _parse_float_field(line_str[61:73])
                        szx = _parse_float_field(line_str[73:85])
                        stresses[nid] = np.array([sxx, syy, szz, sxy, syz, szx], dtype=np.float64)
                    except (ValueError, IndexError):
                        tokens = line_str[3:].split()
                        if len(tokens) >= 7:
                            nid = int(tokens[0])
                            vals = [float(x) for x in tokens[1:7]]
                            stresses[nid] = np.array(vals, dtype=np.float64)

    return stresses


def read_frd_results(frd_path: str) -> Dict[str, Any]:
    """
    Read displacements and stresses from CalculiX .frd file.

    Args:
        frd_path: Path to the .frd result file.

    Returns:
        Dict[str, Any]: Dictionary containing 'displacements' and 'stresses' dictionaries.
    """
    return {
        "displacements": read_frd_displacements(frd_path),
        "stresses": read_frd_stresses(frd_path),
    }
