"""
postprocess package
FEA post-processing and result parsing utilities for dental FEA simulations.
"""

from .result_reader import (
    read_frd_displacements,
    read_frd_stresses,
    read_frd_results,
)

__all__ = [
    "read_frd_displacements",
    "read_frd_stresses",
    "read_frd_results",
]
