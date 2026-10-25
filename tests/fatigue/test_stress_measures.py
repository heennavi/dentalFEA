"""
test_stress_measures.py
Tests for scalar stress measure calculation (von Mises, signed von Mises, principal stresses).
"""

import numpy as np
import pytest
from fatigue.stress_measures import (
    calculate_von_mises,
    calculate_principal_stresses,
    calculate_signed_von_mises,
    calculate_max_principal_stress,
    calculate_hydrostatic_stress,
)


def test_uniaxial_tension_signed_von_mises():
    """합격 기준 1: 단축 인장 상태에서 signed von Mises = 인가 응력 (오차 < 1e-9)."""
    applied_stresses = [1.0, 50.0, 100.0, 350.5, 950.0]
    for sigma in applied_stresses:
        # 단축 X 인장: [SXX, SYY, SZZ, SXY, SYZ, SZX]
        stress_x = np.array([[sigma, 0.0, 0.0, 0.0, 0.0, 0.0]])
        svm_x = calculate_signed_von_mises(stress_x)
        assert np.isclose(svm_x[0], sigma, atol=1e-9, rtol=1e-9), f"Failed for X tension: {svm_x[0]} != {sigma}"

        # 단축 Y 인장
        stress_y = np.array([[0.0, sigma, 0.0, 0.0, 0.0, 0.0]])
        svm_y = calculate_signed_von_mises(stress_y)
        assert np.isclose(svm_y[0], sigma, atol=1e-9, rtol=1e-9), f"Failed for Y tension: {svm_y[0]} != {sigma}"

        # 단축 Z 인장
        stress_z = np.array([[0.0, 0.0, sigma, 0.0, 0.0, 0.0]])
        svm_z = calculate_signed_von_mises(stress_z)
        assert np.isclose(svm_z[0], sigma, atol=1e-9, rtol=1e-9), f"Failed for Z tension: {svm_z[0]} != {sigma}"


def test_uniaxial_compression_signed_von_mises():
    """단축 압축 상태에서 signed von Mises = 음의 인가 응력."""
    applied_stresses = [-10.0, -100.0, -500.0]
    for sigma in applied_stresses:
        stress_x = np.array([[sigma, 0.0, 0.0, 0.0, 0.0, 0.0]])
        svm_x = calculate_signed_von_mises(stress_x)
        assert np.isclose(svm_x[0], sigma, atol=1e-9, rtol=1e-9)


def test_pure_shear_stress_measures():
    """순수 전단 응력 상태에서 von Mises = sqrt(3) * tau_xy."""
    tau = 100.0
    stress = np.array([[0.0, 0.0, 0.0, tau, 0.0, 0.0]])
    expected_vm = np.sqrt(3.0) * tau

    vm = calculate_von_mises(stress)
    assert np.isclose(vm[0], expected_vm, atol=1e-9)

    s1, s2, s3 = calculate_principal_stresses(stress)
    assert np.isclose(s1[0], tau, atol=1e-9)
    assert np.isclose(s2[0], 0.0, atol=1e-9)
    assert np.isclose(s3[0], -tau, atol=1e-9)


def test_hydrostatic_stress():
    """정수압 상태에서 von Mises = 0, Hydrostatic = p."""
    p = 250.0
    stress = np.array([[p, p, p, 0.0, 0.0, 0.0]])
    vm = calculate_von_mises(stress)
    assert np.isclose(vm[0], 0.0, atol=1e-9)

    sh = calculate_hydrostatic_stress(stress)
    assert np.isclose(sh[0], p, atol=1e-9)


def test_multidimensional_array_shape_handling():
    """배열 형상이 1D (6,) 이거나 2D (N, 6) 일 때 모두 올바르게 처리되는지 검증."""
    stress_1d = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    svm_1d = calculate_signed_von_mises(stress_1d)
    assert svm_1d.ndim == 1 and len(svm_1d) == 1
    assert np.isclose(svm_1d[0], 100.0, atol=1e-9)

    stress_2d = np.array([
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 50.0, 0.0, 0.0]
    ])
    svm_2d = calculate_signed_von_mises(stress_2d)
    assert svm_2d.shape == (3,)
    assert np.isclose(svm_2d[0], 100.0, atol=1e-9)
    assert np.isclose(svm_2d[1], -100.0, atol=1e-9)
