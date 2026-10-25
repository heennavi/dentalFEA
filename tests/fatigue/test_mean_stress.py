"""
test_mean_stress.py
Unit tests for mean stress corrections (Goodman, Gerber, Soderberg) and edge cases.
"""

import numpy as np
import pytest
from fatigue.mean_stress import (
    correct_mean_stress_goodman,
    correct_mean_stress_gerber,
    correct_mean_stress_soderberg,
    calculate_fatigue_safety_factor,
)


def test_goodman_boundary_safety_factor_exactly_one():
    """합격 기준 2: Goodman: sigma_a = sigma_e * (1 - sigma_m / UTS) 경계에서 안전계수 정확히 1.0."""
    Se = 510.0     # MPa (Ti-6Al-4V ELI fatigue limit)
    Sut = 860.0    # MPa (UTS)

    test_mean_stresses = [0.0, 100.0, 250.0, 430.0, 600.0]
    for sm in test_mean_stresses:
        # 경계선 상의 응력 진폭
        sa = Se * (1.0 - sm / Sut)

        # 등가 완전반전 응력진폭
        sigma_ar = correct_mean_stress_goodman(sa, sm, Sut)
        assert np.isclose(sigma_ar, Se, atol=1e-7, rtol=1e-7)

        # 안전계수 계산
        sf = calculate_fatigue_safety_factor(sa, sm, Se, Sut, method="goodman")
        assert np.isclose(sf, 1.0, atol=1e-7, rtol=1e-7), f"Expected 1.0, got {sf} for sm={sm}, sa={sa}"


def test_fully_reversed_r_minus_one_identity():
    """합격 기준 3: R = -1 (sigma_m = 0)에서 보정 전후 결과 동일."""
    Se = 510.0
    Sut = 860.0
    Sy = 795.0

    amplitudes = [50.0, 100.0, 255.0, 510.0, 600.0]
    for sa in amplitudes:
        sm = 0.0
        # Goodman
        ar_goodman = correct_mean_stress_goodman(sa, sm, Sut)
        assert np.isclose(ar_goodman, sa, atol=1e-9)

        # Gerber
        ar_gerber = correct_mean_stress_gerber(sa, sm, Sut)
        assert np.isclose(ar_gerber, sa, atol=1e-9)

        # Soderberg
        ar_soderberg = correct_mean_stress_soderberg(sa, sm, Sy)
        assert np.isclose(ar_soderberg, sa, atol=1e-9)

        # Safety factor
        sf = calculate_fatigue_safety_factor(sa, sm, Se, Sut, method="goodman")
        assert np.isclose(sf, Se / sa, atol=1e-9)


def test_edge_cases_no_exceptions():
    """합격 기준 5: 경계 케이스 (sigma_m >= UTS, sigma_a = 0, 음의 평균응력)에서 예외 없이 정의된 동작."""
    Se = 510.0
    Sut = 860.0

    # 1. sigma_a = 0 (정적 하중만 인가된 경우: Goodman 선도 상 SF = Sut / sm)
    sa_zero = 0.0
    sm_nonzero = 300.0
    ar_zero = correct_mean_stress_goodman(sa_zero, sm_nonzero, Sut)
    assert np.isclose(ar_zero, 0.0, atol=1e-9)
    sf_zero = calculate_fatigue_safety_factor(sa_zero, sm_nonzero, Se, Sut, method="goodman")
    assert np.isclose(sf_zero, Sut / sm_nonzero, atol=1e-7)

    # sa_zero = 0, sm = 0 (완전 무하중 상태)
    sf_unloaded = calculate_fatigue_safety_factor(0.0, 0.0, Se, Sut, method="goodman")
    assert np.isinf(sf_unloaded)

    # 2. sigma_m >= UTS (평균응력이 인장강도를 초과한 경우)
    sa_pos = 100.0
    sm_over_uts = Sut + 50.0
    ar_over = correct_mean_stress_goodman(sa_pos, sm_over_uts, Sut)
    assert np.isinf(ar_over)
    sf_over = calculate_fatigue_safety_factor(sa_pos, sm_over_uts, Se, Sut, method="goodman")
    assert sf_over == 0.0

    # sigma_m == UTS 인 경우 (경계 특이점)
    ar_exact_uts = correct_mean_stress_goodman(sa_pos, Sut, Sut)
    assert np.isinf(ar_exact_uts)
    sf_exact_uts = calculate_fatigue_safety_factor(sa_pos, Sut, Se, Sut, method="goodman")
    assert sf_exact_uts == 0.0

    # 3. 음의 평균응력 (압축 평균응력, sigma_m < 0)
    # 보수적 접근: 압축 평균응력은 피로 한계를 떨어뜨리지 않으므로 sm = 0으로 취급 (sigma_ar = sigma_a)
    sm_compressive = -200.0
    ar_comp = correct_mean_stress_goodman(sa_pos, sm_compressive, Sut)
    assert np.isclose(ar_comp, sa_pos, atol=1e-9)
    sf_comp = calculate_fatigue_safety_factor(sa_pos, sm_compressive, Se, Sut, method="goodman")
    assert np.isclose(sf_comp, Se / sa_pos, atol=1e-9)


def test_vectorized_inputs():
    """배열(N,) 입력 시 올바른 벡터화 연산 검증."""
    Se = 510.0
    Sut = 860.0

    sa_arr = np.array([0.0, 100.0, 255.0, 510.0])
    sm_arr = np.array([200.0, 0.0, -100.0, Sut + 10.0])

    ar_arr = correct_mean_stress_goodman(sa_arr, sm_arr, Sut)
    assert len(ar_arr) == 4
    assert np.isclose(ar_arr[0], 0.0)
    assert np.isclose(ar_arr[1], 100.0)
    assert np.isclose(ar_arr[2], 255.0)  # negative sm clamped
    assert np.isinf(ar_arr[3])

    sf_arr = calculate_fatigue_safety_factor(sa_arr, sm_arr, Se, Sut, method="goodman")
    assert len(sf_arr) == 4
    assert np.isclose(sf_arr[0], Sut / 200.0)
    assert np.isclose(sf_arr[1], Se / 100.0)
    assert np.isclose(sf_arr[2], Se / 255.0)
    assert sf_arr[3] == 0.0
