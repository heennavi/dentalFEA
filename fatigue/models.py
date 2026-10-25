"""
models.py
Fatigue analysis data structures and enumerations.
All units conform to mm-N-MPa (stress/modulus in MPa, load in N, length in mm, density in tonne/mm^3).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import yaml


class MeanStressCorrectionMethod(str, Enum):
    """Mean stress correction theory."""
    GOODMAN = "goodman"
    GERBER = "gerber"
    SODERBERG = "soderberg"
    NONE = "none"


class StressMeasureType(str, Enum):
    """Stress scalar measure used for multiaxial fatigue evaluation."""
    SIGNED_VON_MISES = "signed_von_mises"
    VON_MISES = "von_mises"
    MAX_PRINCIPAL = "max_principal"
    SINES = "sines"


@dataclass
class FatigueMaterial:
    """Material mechanical and fatigue properties.
    
    Units:
        E: MPa
        nu: dimensionless
        density: tonne/mm^3
        yield_strength_Sy: MPa
        tensile_strength_Sut: MPa
        fatigue_limit_Se: MPa (fully reversed endurance limit at Ne cycles, R = -1)
        endurance_cycles_Ne: cycles (typically 1.0e7)
        basquin_sigma_f_prime: MPa (fatigue strength coefficient)
        basquin_b: dimensionless (fatigue strength exponent, typically -0.05 to -0.12)
    """
    name: str
    standard: str = ""
    category: str = "metal"
    E: float = 110000.0
    nu: float = 0.34
    density: float = 4.43e-09
    yield_strength_Sy: float = 795.0
    tensile_strength_Sut: float = 860.0
    fatigue_limit_Se: float = 510.0
    endurance_cycles_Ne: float = 1.0e+07
    stress_ratio_basis: float = -1.0
    basquin_sigma_f_prime: Optional[float] = None
    basquin_b: Optional[float] = None
    sn_cycles: Optional[np.ndarray] = None
    sn_stress_amplitudes: Optional[np.ndarray] = None
    sources: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], material_id: str) -> "FatigueMaterial":
        """Load FatigueMaterial instance from materials_fatigue.yaml."""
        p = Path(yaml_path)
        if not p.is_file():
            raise FileNotFoundError(f"Fatigue materials config not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        materials_dict = data.get("materials", {})
        if material_id not in materials_dict:
            available = list(materials_dict.keys())
            raise KeyError(f"Material '{material_id}' not found in {p}. Available: {available}")

        m = materials_dict[material_id]
        el = m.get("elastic", {})
        st = m.get("static_strength", {})
        fa = m.get("fatigue", {})
        ba = fa.get("basquin", {})
        sn = fa.get("sn_curve", {})

        sources_list = [s.get("citation", "") for s in m.get("sources", []) if isinstance(s, dict)]

        sn_c = np.array(sn.get("cycles", []), dtype=np.float64) if "cycles" in sn else None
        sn_s = np.array(sn.get("stress_amplitude_MPa", []), dtype=np.float64) if "stress_amplitude_MPa" in sn else None

        return cls(
            name=m.get("name", material_id),
            standard=m.get("standard", ""),
            category=m.get("category", "metal"),
            E=float(el.get("E", 110000.0)),
            nu=float(el.get("nu", 0.34)),
            density=float(el.get("density", 4.43e-09)),
            yield_strength_Sy=float(st.get("yield_strength_Sy", 795.0)),
            tensile_strength_Sut=float(st.get("tensile_strength_Sut", 860.0)),
            fatigue_limit_Se=float(fa.get("fatigue_limit_Se", 510.0)),
            endurance_cycles_Ne=float(fa.get("endurance_cycles_Ne", 1.0e+07)),
            stress_ratio_basis=float(fa.get("stress_ratio_basis", -1.0)),
            basquin_sigma_f_prime=float(ba["sigma_f_prime"]) if "sigma_f_prime" in ba else None,
            basquin_b=float(ba["b"]) if "b" in ba else None,
            sn_cycles=sn_c,
            sn_stress_amplitudes=sn_s,
            sources=sources_list,
        )


@dataclass
class FatigueLoadCase:
    """Fatigue loading conditions and correction factors.
    
    Attributes:
        name: Identifier for the load case.
        target_cycles: Number of design cycles (e.g. 5.0e6 for ISO 14801).
        stress_ratio_R: Cycle stress ratio R = sigma_min / sigma_max (typically 0.1 for dental implants).
        surface_factor_ka: Surface finish correction factor (1.0 for polished/ground, ~0.85 for SLA).
        notch_factor_kf: Fatigue notch factor (1.0 for smooth, ~2.0-2.2 for thread roots).
        mean_stress_method: Correction formula (Goodman, Gerber, Soderberg).
        stress_measure: Scalar equivalent stress measure.
    """
    name: str = "ISO 14801 Dynamic Fatigue"
    target_cycles: float = 5.0e+06
    stress_ratio_R: float = 0.1
    surface_factor_ka: float = 1.0
    notch_factor_kf: float = 1.0
    mean_stress_method: MeanStressCorrectionMethod = MeanStressCorrectionMethod.GOODMAN
    stress_measure: StressMeasureType = StressMeasureType.SIGNED_VON_MISES


@dataclass
class FatigueResult:
    """Nodal fatigue calculation results.
    
    Attributes:
        node_indices: 0-based node indices.
        sigma_max: Maximum cyclic stress (MPa).
        sigma_min: Minimum cyclic stress (MPa).
        sigma_a: Alternating stress amplitude (MPa), (sigma_max - sigma_min) / 2.
        sigma_m: Mean stress (MPa), (sigma_max + sigma_min) / 2.
        sigma_ar: Equivalent fully reversed stress amplitude (MPa) after mean stress correction.
        safety_factor: Fatigue safety factor against fatigue limit Se.
        life_cycles: Predicted cycles to failure (N_f). np.inf for infinite life.
        damage: Cumulative damage fraction D = target_cycles / life_cycles.
    """
    node_indices: np.ndarray
    sigma_max: np.ndarray
    sigma_min: np.ndarray
    sigma_a: np.ndarray
    sigma_m: np.ndarray
    sigma_ar: np.ndarray
    safety_factor: np.ndarray
    life_cycles: np.ndarray
    damage: np.ndarray

    @property
    def min_safety_factor(self) -> float:
        """Return minimum safety factor across all nodes."""
        if len(self.safety_factor) == 0:
            return 0.0
        return float(np.min(self.safety_factor))

    @property
    def hotspot_node_index(self) -> int:
        """Return node index with minimum safety factor."""
        if len(self.safety_factor) == 0:
            return -1
        return int(self.node_indices[np.argmin(self.safety_factor)])
