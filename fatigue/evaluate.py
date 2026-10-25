"""
evaluate.py
Nodal array-based fatigue evaluation pipeline and orchestrator.
Computes cyclic scalar stresses, mean stress correction, fatigue safety factor,
life cycles, and cumulative damage for FEA meshes using pure vectorized NumPy.
"""

from typing import Optional, Union
import numpy as np

from .models import (
    FatigueMaterial,
    FatigueLoadCase,
    FatigueResult,
    MeanStressCorrectionMethod,
    StressMeasureType,
)
from .stress_measures import (
    calculate_von_mises,
    calculate_signed_von_mises,
    calculate_max_principal_stress,
    calculate_hydrostatic_stress,
    _ensure_2d_stress,
)
from .mean_stress import (
    correct_mean_stress_goodman,
    correct_mean_stress_gerber,
    correct_mean_stress_soderberg,
    calculate_fatigue_safety_factor,
)
from .sn_curve import (
    calculate_basquin_life,
    interpolate_sn_life,
    calculate_miner_damage,
    fit_basquin_from_two_points,
)


def _compute_scalar_stress(
    stress: np.ndarray,
    measure_type: StressMeasureType,
) -> np.ndarray:
    """Compute scalar equivalent stress for a (N, 6) stress tensor array."""
    if measure_type == StressMeasureType.SIGNED_VON_MISES:
        return calculate_signed_von_mises(stress)
    elif measure_type == StressMeasureType.VON_MISES:
        return calculate_von_mises(stress)
    elif measure_type == StressMeasureType.MAX_PRINCIPAL:
        return calculate_max_principal_stress(stress)
    elif measure_type == StressMeasureType.SINES:
        # Sines measure fallback: signed von Mises
        return calculate_signed_von_mises(stress)
    else:
        raise ValueError(f"Unsupported stress measure type: {measure_type}")


def evaluate_fatigue(
    stress_max: np.ndarray,
    stress_min: Optional[np.ndarray] = None,
    material: Optional[FatigueMaterial] = None,
    load_case: Optional[FatigueLoadCase] = None,
    node_indices: Optional[np.ndarray] = None,
) -> FatigueResult:
    """Evaluate high-cycle fatigue safety factor, life, and damage for nodal stress arrays.

    All operations are fully vectorized in NumPy for optimal performance on large meshes.

    Args:
        stress_max: (N, 6) or (6,) stress tensor array at maximum cyclic load [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.
        stress_min: Optional (N, 6) or (6,) stress tensor array at minimum cyclic load.
                    If None, derived automatically from stress_max * load_case.stress_ratio_R.
        material: FatigueMaterial instance containing static and fatigue properties.
        load_case: FatigueLoadCase defining cycle target, stress ratio, correction factors, and theories.
                   Defaults to ISO 14801 standard load case if omitted.
        node_indices: Optional (N,) 0-based node index array. Defaults to np.arange(N).

    Returns:
        FatigueResult containing nodal sigma_max, sigma_min, sigma_a, sigma_m, sigma_ar,
        safety_factor, life_cycles, and damage.
    """
    if material is None:
        raise ValueError("FatigueMaterial instance must be provided for fatigue evaluation.")

    if load_case is None:
        load_case = FatigueLoadCase()

    s_max_2d = _ensure_2d_stress(stress_max)
    num_nodes = s_max_2d.shape[0]

    if stress_min is None:
        s_min_2d = s_max_2d * load_case.stress_ratio_R
    else:
        s_min_2d = _ensure_2d_stress(stress_min)
        if s_min_2d.shape[0] != num_nodes:
            raise ValueError(
                f"stress_max ({num_nodes} nodes) and stress_min ({s_min_2d.shape[0]} nodes) node counts must match."
            )

    if node_indices is None:
        node_indices_arr = np.arange(num_nodes, dtype=np.int64)
    else:
        node_indices_arr = np.asarray(node_indices, dtype=np.int64)
        if len(node_indices_arr) != num_nodes:
            raise ValueError(
                f"node_indices length ({len(node_indices_arr)}) does not match node count ({num_nodes})."
            )

    # 1. Convert 6-component stress tensors to scalar equivalent stresses
    scalar_1 = _compute_scalar_stress(s_max_2d, load_case.stress_measure)
    scalar_2 = _compute_scalar_stress(s_min_2d, load_case.stress_measure)

    # Order algebraically so sigma_max >= sigma_min
    sigma_max = np.maximum(scalar_1, scalar_2)
    sigma_min = np.minimum(scalar_1, scalar_2)

    # 2. Alternating amplitude and mean stress
    sigma_a = 0.5 * (sigma_max - sigma_min)
    sigma_m = 0.5 * (sigma_max + sigma_min)

    # 3. Apply surface finish factor (ka) and fatigue notch factor (kf)
    # Standard practice: modify stress amplitude sigma_a_eff = sigma_a * (kf / ka)
    ka = load_case.surface_factor_ka
    kf = load_case.notch_factor_kf
    if ka <= 0.0 or kf <= 0.0:
        raise ValueError(f"surface_factor_ka ({ka}) and notch_factor_kf ({kf}) must be strictly positive.")

    factor_ak = kf / ka
    sigma_a_eff = sigma_a * factor_ak

    # 4. Mean stress correction -> Equivalent fully reversed stress amplitude (sigma_ar)
    method = load_case.mean_stress_method
    if method == MeanStressCorrectionMethod.GOODMAN:
        sigma_ar = correct_mean_stress_goodman(
            sigma_a_eff, sigma_m, Sut=material.tensile_strength_Sut
        )
    elif method == MeanStressCorrectionMethod.GERBER:
        sigma_ar = correct_mean_stress_gerber(
            sigma_a_eff, sigma_m, Sut=material.tensile_strength_Sut
        )
    elif method == MeanStressCorrectionMethod.SODERBERG:
        sigma_ar = correct_mean_stress_soderberg(
            sigma_a_eff, sigma_m, Sy=material.yield_strength_Sy
        )
    elif method == MeanStressCorrectionMethod.NONE:
        sigma_ar = np.copy(sigma_a_eff)
    else:
        raise ValueError(f"Unsupported mean stress correction method: {method}")

    # 5. Fatigue safety factor (against fatigue endurance limit Se)
    safety_factor = calculate_fatigue_safety_factor(
        sigma_a=sigma_a_eff,
        sigma_m=sigma_m,
        Se=material.fatigue_limit_Se,
        Sut=material.tensile_strength_Sut,
        Sy=material.yield_strength_Sy,
        method=method.value if isinstance(method, MeanStressCorrectionMethod) else str(method),
    )

    # 6. Fatigue life (N_f cycles to failure)
    # Priority:
    #   A) Explicit Basquin parameters (sigma_f_prime, b)
    #   B) Tabular S-N curve points (sn_cycles, sn_stress_amplitudes)
    #   C) Two-point Basquin estimation from (1e3, 0.9*Sut) and (Ne, Se)
    if material.basquin_sigma_f_prime is not None and material.basquin_b is not None:
        life_cycles = calculate_basquin_life(
            stress_amplitude=sigma_ar,
            sigma_f_prime=material.basquin_sigma_f_prime,
            b=material.basquin_b,
            endurance_limit_Se=material.fatigue_limit_Se,
        )
    elif material.sn_cycles is not None and material.sn_stress_amplitudes is not None:
        life_cycles = interpolate_sn_life(
            stress_amplitude=sigma_ar,
            sn_cycles=material.sn_cycles,
            sn_stresses=material.sn_stress_amplitudes,
            endurance_limit_Se=material.fatigue_limit_Se,
        )
    else:
        # Fallback two-point estimate
        s1 = 0.9 * material.tensile_strength_Sut
        n1 = 1.0e+03
        s2 = material.fatigue_limit_Se
        n2 = material.endurance_cycles_Ne
        sigma_f_p, b_exp = fit_basquin_from_two_points(n1, s1, n2, s2)
        life_cycles = calculate_basquin_life(
            stress_amplitude=sigma_ar,
            sigma_f_prime=sigma_f_p,
            b=b_exp,
            endurance_limit_Se=material.fatigue_limit_Se,
        )

    # 7. Palmgren-Miner cumulative damage fraction (D = target_cycles / life_cycles)
    damage = calculate_miner_damage(
        applied_cycles=load_case.target_cycles,
        life_cycles=life_cycles,
    )

    return FatigueResult(
        node_indices=node_indices_arr,
        sigma_max=np.asarray(sigma_max, dtype=np.float64),
        sigma_min=np.asarray(sigma_min, dtype=np.float64),
        sigma_a=np.asarray(sigma_a, dtype=np.float64),
        sigma_m=np.asarray(sigma_m, dtype=np.float64),
        sigma_ar=np.asarray(sigma_ar, dtype=np.float64),
        safety_factor=np.asarray(safety_factor, dtype=np.float64),
        life_cycles=np.asarray(life_cycles, dtype=np.float64),
        damage=np.asarray(damage, dtype=np.float64),
    )
