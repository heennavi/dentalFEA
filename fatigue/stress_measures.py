"""
stress_measures.py
Scalar equivalent stress measures and principal stress calculations.
Stress components format: [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.
All computations are pure NumPy and compatible with 1D (6,) or 2D (N, 6) arrays.
"""

from typing import Tuple, Union
import numpy as np


def _ensure_2d_stress(stress: np.ndarray) -> np.ndarray:
    """Ensure stress array has shape (N, 6) with float64 dtype."""
    arr = np.asarray(stress, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 6:
            raise ValueError(f"1D stress array must have length 6 (got {arr.shape[0]})")
        return arr[np.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[1] != 6:
            raise ValueError(f"2D stress array must have 6 columns (got {arr.shape[1]})")
        return arr
    else:
        raise ValueError(f"Stress array must be 1D or 2D (got {arr.ndim}D)")


def calculate_von_mises(stress: np.ndarray) -> np.ndarray:
    """Calculate equivalent von Mises stress in MPa.
    
    Formula:
        sigma_vm = sqrt(0.5 * ((sxx - syy)^2 + (syy - szz)^2 + (szz - sxx)^2 + 6*(sxy^2 + syz^2 + szx^2)))
    
    Args:
        stress: (N, 6) or (6,) array with [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.
        
    Returns:
        (N,) array of non-negative von Mises stress values in MPa.
    """
    s = _ensure_2d_stress(stress)
    sxx, syy, szz = s[:, 0], s[:, 1], s[:, 2]
    sxy, syz, szx = s[:, 3], s[:, 4], s[:, 5]

    diff_term = (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
    shear_term = 6.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    vm = np.sqrt(np.maximum(0.5 * (diff_term + shear_term), 0.0))
    return vm


def calculate_hydrostatic_stress(stress: np.ndarray) -> np.ndarray:
    """Calculate hydrostatic stress (first invariant / 3) in MPa.
    
    Formula:
        sigma_h = (sxx + syy + szz) / 3.0
    
    Args:
        stress: (N, 6) or (6,) array with [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.
        
    Returns:
        (N,) array of hydrostatic stress values in MPa.
    """
    s = _ensure_2d_stress(stress)
    return (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0


def calculate_principal_stresses(stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate 3 principal stresses (sigma_1 >= sigma_2 >= sigma_3) in MPa.
    
    Uses vectorized symmetric eigenvalue decomposition (np.linalg.eigvalsh).
    
    Args:
        stress: (N, 6) or (6,) array with [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.
        
    Returns:
        Tuple of (sigma_1, sigma_2, sigma_3), each of shape (N,) in MPa.
    """
    s = _ensure_2d_stress(stress)
    n = s.shape[0]

    # Construct symmetric 3x3 stress tensor for each node
    tensors = np.zeros((n, 3, 3), dtype=np.float64)
    tensors[:, 0, 0] = s[:, 0]  # SXX
    tensors[:, 1, 1] = s[:, 1]  # SYY
    tensors[:, 2, 2] = s[:, 2]  # SZZ
    tensors[:, 0, 1] = tensors[:, 1, 0] = s[:, 3]  # SXY
    tensors[:, 1, 2] = tensors[:, 2, 1] = s[:, 4]  # SYZ
    tensors[:, 0, 2] = tensors[:, 2, 0] = s[:, 5]  # SZX

    # eigvalsh returns eigenvalues in ascending order: s_min, s_mid, s_max
    eigvals = np.linalg.eigvalsh(tensors)
    s3 = eigvals[:, 0]
    s2 = eigvals[:, 1]
    s1 = eigvals[:, 2]

    return s1, s2, s3


def calculate_max_principal_stress(stress: np.ndarray) -> np.ndarray:
    """Calculate maximum principal stress sigma_1 in MPa."""
    s1, _, _ = calculate_principal_stresses(stress)
    return s1


def calculate_signed_von_mises(
    stress: np.ndarray,
    sign_criterion: str = "max_abs_principal"
) -> np.ndarray:
    """Calculate signed von Mises stress in MPa.
    
    The magnitude is standard von Mises stress.
    The sign is determined according to the chosen criterion:
      - 'max_abs_principal' (default): Sign follows the principal stress with the largest absolute value.
            sign = sign(sigma_1) if |sigma_1| >= |sigma_3| else sign(sigma_3).
            Ensures pure tension (+sigma) and pure compression (-sigma) preserve correct sign.
      - 'first_principal': Sign directly follows sign(sigma_1).
      - 'hydrostatic': Sign follows sign(sxx + syy + szz).
    
    Args:
        stress: (N, 6) or (6,) array with [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.
        sign_criterion: Rule for sign determination ('max_abs_principal', 'first_principal', 'hydrostatic').
        
    Returns:
        (N,) array of signed von Mises stress in MPa.
    """
    vm = calculate_von_mises(stress)
    s1, _, s3 = calculate_principal_stresses(stress)

    if sign_criterion == "max_abs_principal":
        # Check which principal stress has larger magnitude
        abs_s1 = np.abs(s1)
        abs_s3 = np.abs(s3)
        sign = np.where(abs_s1 >= abs_s3, np.sign(s1), np.sign(s3))
        # Zero threshold handling
        sign = np.where(sign == 0.0, 1.0, sign)
    elif sign_criterion == "first_principal":
        sign = np.sign(s1)
        sign = np.where(sign == 0.0, 1.0, sign)
    elif sign_criterion == "hydrostatic":
        sh = calculate_hydrostatic_stress(stress)
        sign = np.sign(sh)
        sign = np.where(sign == 0.0, 1.0, sign)
    else:
        raise ValueError(f"Unknown sign criterion: '{sign_criterion}'")

    return sign * vm
