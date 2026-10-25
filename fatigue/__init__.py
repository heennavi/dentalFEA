"""
fatigue package
치과 임플란트 및 구조물의 고주기 피로(High-Cycle Fatigue) 수명 및 안전계수 평가 모듈.
"""

from .models import (
    FatigueMaterial,
    MeanStressCorrectionMethod,
    StressMeasureType,
    FatigueLoadCase,
    FatigueResult,
)
from .stress_measures import (
    calculate_von_mises,
    calculate_principal_stresses,
    calculate_max_principal_stress,
    calculate_signed_von_mises,
    calculate_hydrostatic_stress,
)
from .mean_stress import (
    correct_mean_stress_goodman,
    correct_mean_stress_gerber,
    correct_mean_stress_soderberg,
    calculate_fatigue_safety_factor,
)
from .sn_curve import (
    fit_basquin_from_two_points,
    calculate_basquin_stress,
    calculate_basquin_life,
    interpolate_sn_life,
    calculate_miner_damage,
)

__all__ = [
    # models
    "FatigueMaterial",
    "MeanStressCorrectionMethod",
    "StressMeasureType",
    "FatigueLoadCase",
    "FatigueResult",
    # stress_measures
    "calculate_von_mises",
    "calculate_principal_stresses",
    "calculate_max_principal_stress",
    "calculate_signed_von_mises",
    "calculate_hydrostatic_stress",
    # mean_stress
    "correct_mean_stress_goodman",
    "correct_mean_stress_gerber",
    "correct_mean_stress_soderberg",
    "calculate_fatigue_safety_factor",
    # sn_curve
    "fit_basquin_from_two_points",
    "calculate_basquin_stress",
    "calculate_basquin_life",
    "interpolate_sn_life",
    "calculate_miner_damage",
]

