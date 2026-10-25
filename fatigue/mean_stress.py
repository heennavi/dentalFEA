"""
mean_stress.py
Mean stress correction theories (Goodman, Gerber, Soderberg) and safety factor evaluation.
All computations are in MPa and fully vectorized with NumPy.
"""

from typing import Optional, Union
import numpy as np


def correct_mean_stress_goodman(
    sigma_a: Union[float, np.ndarray],
    sigma_m: Union[float, np.ndarray],
    Sut: float,
    clamp_compressive: bool = True
) -> Union[float, np.ndarray]:
    """Calculate equivalent fully reversed stress amplitude (R = -1) using Goodman's rule.
    
    Formula:
        sigma_ar = sigma_a / (1.0 - sigma_m / Sut)
        
    Args:
        sigma_a: Alternating stress amplitude in MPa (>= 0).
        sigma_m: Mean stress in MPa.
        Sut: Ultimate tensile strength in MPa (> 0).
        clamp_compressive: If True, compressive mean stress (sigma_m < 0) is clamped to 0.0
                          (conservative standard practice in high-cycle fatigue).
                          
    Returns:
        sigma_ar: Equivalent stress amplitude in MPa.
                  Returns 0.0 if sigma_a == 0.
                  Returns np.inf if sigma_m >= Sut.
    """
    is_scalar = np.isscalar(sigma_a) and np.isscalar(sigma_m)
    sa = np.asarray(sigma_a, dtype=np.float64)
    sm = np.asarray(sigma_m, dtype=np.float64)

    if Sut <= 0:
        raise ValueError(f"Sut must be positive, got {Sut}")

    # Clamping negative mean stresses
    if clamp_compressive:
        sm_eff = np.maximum(sm, 0.0)
    else:
        sm_eff = sm

    denom = 1.0 - (sm_eff / Sut)

    # Edge cases:
    # 1. sigma_a == 0 -> sigma_ar = 0.0
    # 2. denom <= 0 (sm >= Sut) -> sigma_ar = np.inf
    # 3. normal -> sa / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        ar = np.where(sa == 0.0, 0.0, np.where(denom <= 0.0, np.inf, sa / denom))

    return float(ar) if is_scalar else ar


def correct_mean_stress_gerber(
    sigma_a: Union[float, np.ndarray],
    sigma_m: Union[float, np.ndarray],
    Sut: float,
    clamp_compressive: bool = True
) -> Union[float, np.ndarray]:
    """Calculate equivalent fully reversed stress amplitude (R = -1) using Gerber's parabolic rule.
    
    Formula:
        sigma_ar = sigma_a / (1.0 - (sigma_m / Sut)^2)
    """
    is_scalar = np.isscalar(sigma_a) and np.isscalar(sigma_m)
    sa = np.asarray(sigma_a, dtype=np.float64)
    sm = np.asarray(sigma_m, dtype=np.float64)

    if Sut <= 0:
        raise ValueError(f"Sut must be positive, got {Sut}")

    if clamp_compressive:
        sm_eff = np.maximum(sm, 0.0)
    else:
        sm_eff = sm

    denom = 1.0 - (sm_eff / Sut) ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        ar = np.where(sa == 0.0, 0.0, np.where(denom <= 0.0, np.inf, sa / denom))

    return float(ar) if is_scalar else ar


def correct_mean_stress_soderberg(
    sigma_a: Union[float, np.ndarray],
    sigma_m: Union[float, np.ndarray],
    Sy: float,
    clamp_compressive: bool = True
) -> Union[float, np.ndarray]:
    """Calculate equivalent fully reversed stress amplitude (R = -1) using Soderberg's yield-based rule.
    
    Formula:
        sigma_ar = sigma_a / (1.0 - sigma_m / Sy)
    """
    is_scalar = np.isscalar(sigma_a) and np.isscalar(sigma_m)
    sa = np.asarray(sigma_a, dtype=np.float64)
    sm = np.asarray(sigma_m, dtype=np.float64)

    if Sy <= 0:
        raise ValueError(f"Sy must be positive, got {Sy}")

    if clamp_compressive:
        sm_eff = np.maximum(sm, 0.0)
    else:
        sm_eff = sm

    denom = 1.0 - (sm_eff / Sy)

    with np.errstate(divide="ignore", invalid="ignore"):
        ar = np.where(sa == 0.0, 0.0, np.where(denom <= 0.0, np.inf, sa / denom))

    return float(ar) if is_scalar else ar


def calculate_fatigue_safety_factor(
    sigma_a: Union[float, np.ndarray],
    sigma_m: Union[float, np.ndarray],
    Se: float,
    Sut: float,
    Sy: Optional[float] = None,
    method: str = "goodman",
    clamp_compressive: bool = True
) -> Union[float, np.ndarray]:
    """Calculate fatigue factor of safety against fatigue endurance limit Se.
    
    Formulas:
        - Goodman:
            1 / SF = sigma_a / Se + sigma_m / Sut
            SF = Se / (sigma_a + (Se / Sut) * sigma_m)
        - Gerber:
            (SF * sigma_a / Se) + (SF * sigma_m / Sut)^2 = 1
        - Soderberg:
            1 / SF = sigma_a / Se + sigma_m / Sy
            
    Args:
        sigma_a: Stress amplitude in MPa.
        sigma_m: Mean stress in MPa.
        Se: Fatigue endurance limit in MPa (at Ne cycles, R = -1).
        Sut: Ultimate tensile strength in MPa.
        Sy: Yield strength in MPa (required for Soderberg).
        method: 'goodman', 'gerber', 'soderberg', or 'none'.
        clamp_compressive: Clamp negative mean stress to 0.
        
    Returns:
        SF: Safety factor.
            Returns np.inf if sigma_a == 0 and sigma_m < Sut.
            Returns 0.0 if sigma_m >= Sut.
    """
    is_scalar = np.isscalar(sigma_a) and np.isscalar(sigma_m)
    sa = np.asarray(sigma_a, dtype=np.float64)
    sm = np.asarray(sigma_m, dtype=np.float64)

    if clamp_compressive:
        sm_eff = np.maximum(sm, 0.0)
    else:
        sm_eff = sm

    method_lower = method.lower()

    if method_lower == "goodman":
        # Goodman formula: 1 / SF = sa / Se + sm / Sut
        denom = (sa / Se) + (sm_eff / Sut)
        with np.errstate(divide="ignore", invalid="ignore"):
            sf = np.where(
                sm_eff >= Sut,
                0.0,
                np.where(denom <= 0.0, np.inf, 1.0 / denom)
            )
    elif method_lower == "gerber":
        # Gerber quadratic: A * SF^2 + B * SF - 1 = 0
        # A = (sm_eff / Sut)^2, B = sa / Se
        A = (sm_eff / Sut) ** 2
        B = sa / Se
        # When A == 0 (sm_eff == 0), SF = 1 / B = Se / sa
        with np.errstate(divide="ignore", invalid="ignore"):
            # Discriminant: B^2 + 4A
            disc = B ** 2 + 4.0 * A
            sf_quad = (-B + np.sqrt(disc)) / (2.0 * np.maximum(A, 1e-15))
            sf_zero_m = np.where(B > 0, 1.0 / B, np.inf)
            sf = np.where(
                sm_eff >= Sut,
                0.0,
                np.where(A < 1e-12, sf_zero_m, sf_quad)
            )
    elif method_lower == "soderberg":
        if Sy is None or Sy <= 0:
            raise ValueError(f"Sy must be provided and positive for Soderberg method, got {Sy}")
        denom = (sa / Se) + (sm_eff / Sy)
        with np.errstate(divide="ignore", invalid="ignore"):
            sf = np.where(
                sm_eff >= Sy,
                0.0,
                np.where(denom <= 0.0, np.inf, 1.0 / denom)
            )
    elif method_lower == "none":
        with np.errstate(divide="ignore", invalid="ignore"):
            sf = np.where(sa <= 0.0, np.inf, Se / sa)
    else:
        raise ValueError(f"Unknown mean stress method: '{method}'")

    return float(sf) if is_scalar else sf
