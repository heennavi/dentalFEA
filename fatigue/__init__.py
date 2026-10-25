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
from .evaluate import (
    evaluate_fatigue,
)
from .io_frd import (
    extract_stress_from_frd,
    extract_cyclic_stresses_from_frd,
    extract_cyclic_stresses_from_two_frds,
    extract_cyclic_stresses_with_ratio,
)

from .cantilever_benchmark import (
    CantileverTheory,
    CantileverBenchmarkResult,
    calculate_cantilever_theory,
    build_cantilever_model_and_bc,
    run_cantilever_benchmark,
)
from .iso14801 import (
    ISO14801LoadMode,
    ISO14801Config,
    run_iso14801_pipeline,
)
from .notch_benchmark import (
    NotchSpec,
    NotchTheory,
    NotchFEAResult,
    NotchBenchmarkResult,
    calculate_notch_theory,
    calculate_notch_sensitivity,
    build_notched_plate_mesh,
    build_notch_model_and_bc,
    run_notch_benchmark,
)
from .preload_contact import (
    PreloadContactConfig,
    PreloadContactResult,
    SimplifiedModelResult,
    ModelComparisonReport,
    build_benchmark_assembly,
    write_preload_contact_deck,
    run_preload_contact_analysis,
    run_simplified_model_analysis,
    compare_simplified_vs_detailed,
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
    # evaluate
    "evaluate_fatigue",
    # io_frd
    "extract_stress_from_frd",
    "extract_cyclic_stresses_from_frd",
    "extract_cyclic_stresses_from_two_frds",
    "extract_cyclic_stresses_with_ratio",
    # cantilever_benchmark
    "CantileverTheory",
    "CantileverBenchmarkResult",
    "calculate_cantilever_theory",
    "build_cantilever_model_and_bc",
    "run_cantilever_benchmark",
    # iso14801
    "ISO14801LoadMode",
    "ISO14801Config",
    "run_iso14801_pipeline",
    # notch_benchmark
    "NotchSpec",
    "NotchTheory",
    "NotchFEAResult",
    "NotchBenchmarkResult",
    "calculate_notch_theory",
    "calculate_notch_sensitivity",
    "build_notched_plate_mesh",
    "build_notch_model_and_bc",
    "run_notch_benchmark",
    # preload_contact
    "PreloadContactConfig",
    "PreloadContactResult",
    "SimplifiedModelResult",
    "ModelComparisonReport",
    "build_benchmark_assembly",
    "write_preload_contact_deck",
    "run_preload_contact_analysis",
    "run_simplified_model_analysis",
    "compare_simplified_vs_detailed",
]



