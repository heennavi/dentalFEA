"""
results.py
Data contracts and core models for CalculiX FEA simulation results.

[Units]
- Length / Displacement (u): mm
- Force / Reaction Force (rf): N
- Stress: MPa (N/mm^2)
- Strain: Dimensionless
- Time: s
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import numpy as np

from fea.models import MeshModel


@dataclass
class FrdBlock:
    """
    Representation of a single result block in CalculiX .frd file.

    Attributes:
        name: Name of result block (e.g. 'DISP', 'STRESS', 'TOSTRAIN', 'FORC').
        time: Step time value for the increment.
        step: 0-based block sequence index.
        component_names: Column header names (e.g. ['SXX', 'SYY', 'SZZ', 'SXY', 'SYZ', 'SZX']).
        node_ids: (N,) int64 array of 1-based global node numbers from .frd.
        values: (N, ncomp) float64 array of nodal data.
    """
    name: str                    # "DISP", "STRESS", "TOSTRAIN", "FORC" 등 .frd 결과 블록 이름
    time: float                  # 해당 increment의 step time
    step: int                    # 0부터 시작하는 블록 순서 번호
    component_names: list[str]   # 예: ["SXX","SYY","SZZ","SXY","SYZ","SZX"]
    node_ids: np.ndarray         # (N,) int64, .frd 절점 번호(= .inp 전역 1-based)
    values: np.ndarray           # (N, ncomp) float64


@dataclass
class FrdData:
    """
    In-memory representation of raw parsed .frd file data.

    Attributes:
        node_ids: (N,) int64 array of 1-based global node IDs.
        node_coords: (N, 3) float64 array of undeformed nodal coordinates in mm (LPS).
        blocks: List of FrdBlock result sets parsed from file.
    """
    node_ids: np.ndarray         # (N,) int64
    node_coords: np.ndarray      # (N,3) float64, 비변형 좌표
    blocks: list[FrdBlock] = field(default_factory=list)

    def get_block(self, name: str, which: Union[str, int] = "last") -> Optional[FrdBlock]:
        """
        Retrieve a specific FrdBlock by its block name.

        Args:
            name: Block identifier (e.g. 'DISP', 'STRESS', 'TOSTRAIN', 'FORC').
            which: 'last' for final increment, 'first' for first, or integer index/step.

        Returns:
            FrdBlock instance if found, or None.
        """
        target_name = name.strip().upper()
        matching = [b for b in self.blocks if b.name.strip().upper() == target_name]
        if not matching:
            return None

        if isinstance(which, str):
            mode = which.strip().lower()
            if mode == "last":
                return matching[-1]
            elif mode == "first":
                return matching[0]
            else:
                try:
                    idx = int(mode)
                    return matching[idx]
                except (ValueError, IndexError):
                    return None
        elif isinstance(which, int):
            # Try index within matching blocks first
            if -len(matching) <= which < len(matching):
                return matching[which]
            # Fallback: check matching by FrdBlock.step
            for b in matching:
                if b.step == which:
                    return b
            return None
        return None

    def times(self, name: Optional[str] = None) -> list[float]:
        """
        Return list of step time values.

        Args:
            name: Optional block name filter. If None, returns times for all blocks.

        Returns:
            list[float]: Step times in seconds.
        """
        if name is not None:
            target_name = name.strip().upper()
            return [b.time for b in self.blocks if b.name.strip().upper() == target_name]
        return [b.time for b in self.blocks]


@dataclass
class ResultSet:
    """
    Unified, aligned FEA result field aligned to MeshModel global 0-based node ordering.

    All arrays are indexed in the global 0-based node order formed by concatenating
    mesh parts (model.parts[0], model.parts[1], ...). Nodes absent in the .frd
    are filled with NaN.

    Attributes:
        time: Simulation step time (s).
        u: (Ntot, 3) float64 displacement vector (mm).
        u_mag: (Ntot,) float64 displacement magnitude (mm).
        stress6: (Ntot, 6) float64 Cauchy stress components (SXX, SYY, SZZ, SXY, SYZ, SZX) in MPa.
        von_mises: (Ntot,) float64 von Mises equivalent stress in MPa.
        s_max_principal: (Ntot,) float64 maximum principal stress in MPa (tensile positive).
        s_min_principal: (Ntot,) float64 minimum principal stress in MPa (compressive negative).
        strain6: (Ntot, 6) float64 total strain components (EXX, EYY, EZZ, EXY, EYZ, EZX).
        rf: (Ntot, 3) float64 reaction force vector (N).
        source_frd: Path to the source .frd file.
    """
    time: float
    u: np.ndarray | None = None                # (Ntot,3) mm
    u_mag: np.ndarray | None = None            # (Ntot,)
    stress6: np.ndarray | None = None   # (Ntot,6) MPa, 순서 SXX SYY SZZ SXY SYZ SZX
    von_mises: np.ndarray | None = None # (Ntot,)
    s_max_principal: np.ndarray | None = None   # (Ntot,)  인장 방향이 양수
    s_min_principal: np.ndarray | None = None   # (Ntot,)  압축 방향이 음수
    strain6: np.ndarray | None = None   # (Ntot,6)
    rf: np.ndarray | None = None        # (Ntot,3) 반력 N
    source_frd: str = ""              # 원본 .frd 경로

    def slice_for_part(self, model: MeshModel, part_name: str) -> slice:
        """
        Return the 0-based index slice corresponding to a specific MeshPart in this ResultSet.

        Args:
            model: MeshModel containing the assembly definition.
            part_name: Name of the part to slice.

        Returns:
            slice: Python slice object [offset : offset + n_nodes].

        Raises:
            KeyError: If part_name is not present in model.
        """
        offset = model.global_node_offset(part_name)
        part = model.get_part(part_name)
        if part is None:
            raise KeyError(f"Part '{part_name}' not found in MeshModel.")
        return slice(offset, offset + len(part.nodes))

    def summary(self) -> dict:
        """
        Generate a summary dictionary of key scalar metrics.

        Returns:
            dict: Summary metrics containing maximum displacement, von Mises stress,
                  principal stresses, total reaction force, and source metadata.
        """
        res: Dict[str, Any] = {
            "time": float(self.time),
            "source_frd": str(self.source_frd),
            "num_nodes": int(len(self.u_mag)),
        }

        # Displacement metrics
        if self.u_mag is not None and len(self.u_mag) > 0:
            valid_u = self.u_mag[~np.isnan(self.u_mag)]
            res["max_u_mag_mm"] = float(np.max(valid_u)) if len(valid_u) > 0 else 0.0
        else:
            res["max_u_mag_mm"] = 0.0

        # Von Mises stress metrics
        if self.von_mises is not None and len(self.von_mises) > 0:
            valid_vm = self.von_mises[~np.isnan(self.von_mises)]
            res["max_von_mises_MPa"] = float(np.max(valid_vm)) if len(valid_vm) > 0 else None
            res["mean_von_mises_MPa"] = float(np.mean(valid_vm)) if len(valid_vm) > 0 else None
        else:
            res["max_von_mises_MPa"] = None
            res["mean_von_mises_MPa"] = None

        # Principal stress metrics
        if self.s_max_principal is not None and len(self.s_max_principal) > 0:
            valid_smax = self.s_max_principal[~np.isnan(self.s_max_principal)]
            res["max_principal_stress_MPa"] = float(np.max(valid_smax)) if len(valid_smax) > 0 else None
        else:
            res["max_principal_stress_MPa"] = None

        if self.s_min_principal is not None and len(self.s_min_principal) > 0:
            valid_smin = self.s_min_principal[~np.isnan(self.s_min_principal)]
            res["min_principal_stress_MPa"] = float(np.min(valid_smin)) if len(valid_smin) > 0 else None
        else:
            res["min_principal_stress_MPa"] = None

        # Reaction forces (Resultant total force)
        if self.rf is not None and len(self.rf) > 0:
            valid_mask = ~np.isnan(self.rf).any(axis=1)
            if np.any(valid_mask):
                total_rf = np.sum(self.rf[valid_mask], axis=0)
                res["total_rf_N"] = [float(x) for x in total_rf]
                res["total_rf_mag_N"] = float(np.linalg.norm(total_rf))
            else:
                res["total_rf_N"] = [0.0, 0.0, 0.0]
                res["total_rf_mag_N"] = 0.0
        else:
            res["total_rf_N"] = None
            res["total_rf_mag_N"] = None

        return res
