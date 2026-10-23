"""
postprocess package
FEA post-processing and result parsing utilities for dental FEA simulations.
"""

from .result_reader import (
    read_frd_displacements,
    read_frd_stresses,
    read_frd_results,
)
from .frd_parser import (
    parse_frd,
    parse_frd_text,
)

__all__ = [
    "read_frd_displacements",
    "read_frd_stresses",
    "read_frd_results",
    "parse_frd",
    "parse_frd_text",
]
