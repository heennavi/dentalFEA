"""
sn_curve.py
Stress-Life (S-N) curve evaluations, Basquin power-law relationships, and Palmgren-Miner damage.
All calculations are in MPa and cycles. Fully vectorized using NumPy.
"""

from typing import Optional, Tuple, Union
import numpy as np


def fit_basquin_from_two_points(
    N1: float,
    S1: float,
    N2: float,
    S2: float
) -> Tuple[float, float]:
    """Fit Basquin power-law parameters (sigma_f_prime, b) passing exactly through two reference points.
    
    Formula:
        sigma_a = sigma_f_prime * (2 * N_f)^b
        
        b = log10(S1 / S2) / log10((2 * N1) / (2 * N2)) = log10(S1 / S2) / log10(N1 / N2)
        sigma_f_prime = S1 / (2 * N1)^b
        
    Args:
        N1: Cycles at point 1 (> 0).
        S1: Stress amplitude at point 1 in MPa (> 0).
        N2: Cycles at point 2 (> 0, N2 != N1).
        S2: Stress amplitude at point 2 in MPa (> 0).
        
    Returns:
        Tuple of (sigma_f_prime [MPa], b [dimensionless]).
    """
    if N1 <= 0 or N2 <= 0 or S1 <= 0 or S2 <= 0:
        raise ValueError("Cycles and stress amplitudes must be strictly positive.")
    if N1 == N2:
        raise ValueError("N1 and N2 cannot be identical.")

    b = float(np.log10(S1 / S2) / np.log10(N1 / N2))
    sigma_f_prime = float(S1 / ((2.0 * N1) ** b))
    return sigma_f_prime, b


def calculate_basquin_stress(
    cycles: Union[float, np.ndarray],
    sigma_f_prime: float,
    b: float,
    endurance_limit_Se: Optional[float] = None
) -> Union[float, np.ndarray]:
    """Calculate stress amplitude sigma_a for a given number of cycles using Basquin's equation.
    
    Formula:
        sigma_a = sigma_f_prime * (2 * cycles)^b
        
    Args:
        cycles: Number of fatigue cycles (> 0).
        sigma_f_prime: Fatigue strength coefficient in MPa.
        b: Fatigue strength exponent (typically negative).
        endurance_limit_Se: Optional lower bound cutoff for stress amplitude.
        
    Returns:
        sigma_a: Stress amplitude in MPa.
    """
    is_scalar = np.isscalar(cycles)
    c = np.asarray(cycles, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        sa = sigma_f_prime * ((2.0 * c) ** b)
        if endurance_limit_Se is not None:
            sa = np.maximum(sa, endurance_limit_Se)

    return float(sa) if is_scalar else sa


def calculate_basquin_life(
    stress_amplitude: Union[float, np.ndarray],
    sigma_f_prime: float,
    b: float,
    endurance_limit_Se: Optional[float] = None
) -> Union[float, np.ndarray]:
    """Calculate fatigue life (cycles to failure N_f) for a given stress amplitude using Basquin's equation.
    
    Formula:
        N_f = 0.5 * (sigma_a / sigma_f_prime)^(1 / b)
        
    Args:
        stress_amplitude: Stress amplitude sigma_a in MPa.
        sigma_f_prime: Fatigue strength coefficient in MPa.
        b: Fatigue strength exponent (negative).
        endurance_limit_Se: Fatigue endurance limit in MPa. If stress_amplitude <= Se, life is infinite (np.inf).
        
    Returns:
        N_f: Cycles to failure. np.inf if stress_amplitude <= Se or stress_amplitude <= 0.
    """
    is_scalar = np.isscalar(stress_amplitude)
    sa = np.asarray(stress_amplitude, dtype=np.float64)

    if b >= 0:
        raise ValueError(f"Basquin exponent b must be negative, got {b}")
    if sigma_f_prime <= 0:
        raise ValueError(f"sigma_f_prime must be positive, got {sigma_f_prime}")

    inv_b = 1.0 / b

    with np.errstate(divide="ignore", invalid="ignore"):
        # Base calculation
        # If sa <= 0 -> infinite life
        # If endurance_limit_Se is specified and sa <= endurance_limit_Se -> infinite life
        ratio = np.maximum(sa / sigma_f_prime, 1e-15)
        raw_life = 0.5 * (ratio ** inv_b)

        if endurance_limit_Se is not None:
            life = np.where(sa < endurance_limit_Se, np.inf, raw_life)
        else:
            life = np.where(sa <= 0.0, np.inf, raw_life)

    return float(life) if is_scalar else life


def interpolate_sn_life(
    stress_amplitude: Union[float, np.ndarray],
    sn_cycles: np.ndarray,
    sn_stresses: np.ndarray,
    endurance_limit_Se: Optional[float] = None
) -> Union[float, np.ndarray]:
    """Estimate fatigue life from tabular S-N curve points via piecewise log-log linear interpolation.
    
    Interpolates in (log10(S), log10(N)) space.
    
    Args:
        stress_amplitude: Query stress amplitude in MPa.
        sn_cycles: 1D array of cycle counts (sorted ascending).
        sn_stresses: 1D array of corresponding stress amplitudes in MPa (sorted descending).
        endurance_limit_Se: Optional endurance limit. If sa < Se, returns np.inf.
        
    Returns:
        N_f: Interpolated fatigue life.
    """
    is_scalar = np.isscalar(stress_amplitude)
    sa = np.asarray(stress_amplitude, dtype=np.float64)

    c = np.asarray(sn_cycles, dtype=np.float64)
    s = np.asarray(sn_stresses, dtype=np.float64)

    if len(c) != len(s) or len(c) < 2:
        raise ValueError("sn_cycles and sn_stresses must have the same length >= 2")

    # Invert order if stresses are ascending
    if s[0] < s[-1]:
        c = c[::-1]
        s = s[::-1]

    log_s = np.log10(s)
    log_c = np.log10(c)

    # For np.interp, xp must be monotonically increasing.
    # log_s is descending, so we flip it.
    xp = log_s[::-1]
    fp = log_c[::-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        query_log_sa = np.log10(np.maximum(sa, 1e-12))
        log_life = np.interp(query_log_sa, xp, fp)
        life = 10.0 ** log_life

        if endurance_limit_Se is not None:
            life = np.where(sa < endurance_limit_Se, np.inf, life)
        else:
            # Below minimum table stress -> infinite life
            min_table_s = np.min(s)
            life = np.where(sa < min_table_s, np.inf, life)

    return float(life) if is_scalar else life


def calculate_miner_damage(
    applied_cycles: Union[float, np.ndarray],
    life_cycles: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate cumulative fatigue damage fraction D using the Palmgren-Miner linear damage rule.
    
    Formula:
        D = sum(n_i / N_i)
        
    Args:
        applied_cycles: Number of applied cycles n_i (>= 0).
        life_cycles: Fatigue life N_i (>= 0, np.inf indicates infinite life).
        
    Returns:
        damage: Damage fraction D_i = n_i / N_i (returns 0.0 if N_i is infinite).
    """
    is_scalar = np.isscalar(applied_cycles) and np.isscalar(life_cycles)
    n = np.asarray(applied_cycles, dtype=np.float64)
    N = np.asarray(life_cycles, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.isinf(N) | (N <= 0.0), 0.0, n / N)

    return float(d) if is_scalar else d
