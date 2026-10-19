"""
solver package
CalculiX input generation and solver execution pipeline.
"""

from solver.calculix_writer import write_inp, GlobalIndexMapper
from solver.runner import CalculixRunner

__all__ = ["write_inp", "GlobalIndexMapper", "CalculixRunner"]

