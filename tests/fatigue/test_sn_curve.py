"""
test_sn_curve.py
Unit tests for S-N curve evaluations, Basquin relationships, interpolation, and Miner damage.
"""

import numpy as np
import pytest
from fatigue.sn_curve import (
    calculate_basquin_stress,
    calculate_basquin_life,
    fit_basquin_from_two_points,
    interpolate_sn_life,
    calculate_miner_damage,
)


def test_basquin_passes_two_reference_points_exactly():
    """합격 기준 4: Basquin 곡선이 두 기준점을 정확히 통과."""
    # 두 기준점: (N1, S1) = (1.0e4, 720.0), (N2, S2) = (1.0e7, 510.0)
    N1, S1 = 1.0e4, 720.0
    N2, S2 = 1.0e7, 510.0

    sigma_f_prime, b = fit_basquin_from_two_points(N1, S1, N2, S2)

    # 1. 수명 N1, N2에서 응력 계산 시 정확히 S1, S2가 나와야 함 (오차 < 1e-9)
    pred_S1 = calculate_basquin_stress(N1, sigma_f_prime, b)
    pred_S2 = calculate_basquin_stress(N2, sigma_f_prime, b)
    assert np.isclose(pred_S1, S1, atol=1e-9, rtol=1e-9), f"Expected {S1}, got {pred_S1}"
    assert np.isclose(pred_S2, S2, atol=1e-9, rtol=1e-9), f"Expected {S2}, got {pred_S2}"

    # 2. 응력 S1, S2에서 수명 계산 시 정확히 N1, N2가 나와야 함 (오차 < 1e-7)
    pred_N1 = calculate_basquin_life(S1, sigma_f_prime, b)
    pred_N2 = calculate_basquin_life(S2, sigma_f_prime, b)
    assert np.isclose(pred_N1, N1, rtol=1e-7), f"Expected {N1}, got {pred_N1}"
    assert np.isclose(pred_N2, N2, rtol=1e-7), f"Expected {N2}, got {pred_N2}"


def test_basquin_endurance_limit_infinite_life():
    """피로한계 이하의 응력에서 무한 수명(np.inf) 반환 검증."""
    N1, S1 = 1.0e4, 720.0
    N2, S2 = 1.0e7, 510.0
    sigma_f_prime, b = fit_basquin_from_two_points(N1, S1, N2, S2)

    Se = 510.0

    # Se 미만의 응력
    life_below = calculate_basquin_life(500.0, sigma_f_prime, b, endurance_limit_Se=Se)
    assert np.isinf(life_below)

    life_zero = calculate_basquin_life(0.0, sigma_f_prime, b, endurance_limit_Se=Se)
    assert np.isinf(life_zero)

    # Se 초과의 응력
    life_above = calculate_basquin_life(600.0, sigma_f_prime, b, endurance_limit_Se=Se)
    assert np.isfinite(life_above) and life_above > 0


def test_sn_curve_table_interpolation():
    """표 형태 S-N 데이터의 Log-Log 선형 보간 검증."""
    cycles = np.array([1.0e4, 1.0e5, 1.0e6, 1.0e7])
    stresses = np.array([720.0, 620.0, 540.0, 510.0])

    # 표의 정확한 데이터 점을 쿼리했을 때 원래 사이클 수와 정확히 일치해야 함
    for c, s in zip(cycles, stresses):
        pred_life = interpolate_sn_life(s, cycles, stresses, endurance_limit_Se=510.0)
        assert np.isclose(pred_life, c, rtol=1e-7)

    # 중간 지점 (기하 평균)
    mid_s = np.sqrt(720.0 * 620.0)
    pred_mid_life = interpolate_sn_life(mid_s, cycles, stresses)
    expected_mid_life = np.sqrt(1.0e4 * 1.0e5)
    assert np.isclose(pred_mid_life, expected_mid_life, rtol=1e-4)


def test_miner_damage_calculation():
    """Palmgren-Miner 누적 손상률 계산 검증."""
    applied = np.array([5.0e5, 1.0e6, 5.0e6])
    lives = np.array([1.0e6, 2.0e6, np.inf])

    # D = 5e5/1e6 + 1e6/2e6 + 5e6/inf = 0.5 + 0.5 + 0 = 1.0
    damages = calculate_miner_damage(applied, lives)
    assert np.isclose(damages[0], 0.5)
    assert np.isclose(damages[1], 0.5)
    assert np.isclose(damages[2], 0.0)
    assert np.isclose(np.sum(damages), 1.0)
