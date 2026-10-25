"""
test_evaluate.py
Unit and validation tests for nodal fatigue evaluation pipeline.
Verifies hand-calculated analytical solutions and benchmark performance (100k nodes < 1s).
"""

import time
import numpy as np
import pytest

from fatigue.models import (
    FatigueMaterial,
    FatigueLoadCase,
    FatigueResult,
    MeanStressCorrectionMethod,
    StressMeasureType,
)
from fatigue.evaluate import evaluate_fatigue


@pytest.fixture
def ti6al4v_material():
    """Standard Ti-6Al-4V ELI fatigue material."""
    return FatigueMaterial(
        name="Ti-6Al-4V ELI",
        E=110000.0,
        nu=0.34,
        yield_strength_Sy=795.0,
        tensile_strength_Sut=860.0,
        fatigue_limit_Se=510.0,
        endurance_cycles_Ne=1.0e+07,
        basquin_sigma_f_prime=1440.0,
        basquin_b=-0.095,
    )


def test_hand_calculation_uniaxial_below_endurance(ti6al4v_material):
    """Verify analytical solution for cyclic uniaxial stress below endurance limit.
    
    Given:
        sigma_xx_max = 300.0 MPa, sigma_xx_min = 30.0 MPa (R = 0.1)
        Ti-6Al-4V: Sut = 860.0 MPa, Se = 510.0 MPa
    Hand Calculation:
        sigma_max = 300.0 MPa
        sigma_min = 30.0 MPa
        sigma_a = (300.0 - 30.0) / 2 = 135.0 MPa
        sigma_m = (300.0 + 30.0) / 2 = 165.0 MPa
        Goodman equivalent amplitude:
            sigma_ar = 135.0 / (1.0 - 165.0 / 860.0) = 135.0 / (695.0 / 860.0) = 167.0503597 MPa
        Fatigue safety factor (Goodman):
            1 / SF = (135.0 / 510.0) + (165.0 / 860.0) = 0.26470588 + 0.19186047 = 0.45656635
            SF = 1 / 0.45656635 = 2.19026214
        Since sigma_ar (167.05) < Se (510.0):
            life_cycles = inf
            damage = 0.0
    """
    s_max = np.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_min = np.array([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    load_case = FatigueLoadCase(
        target_cycles=5.0e+06,
        surface_factor_ka=1.0,
        notch_factor_kf=1.0,
        mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
        stress_measure=StressMeasureType.SIGNED_VON_MISES,
    )

    result = evaluate_fatigue(s_max, s_min, material=ti6al4v_material, load_case=load_case)

    assert result.sigma_max[0] == pytest.approx(300.0, rel=1e-5)
    assert result.sigma_min[0] == pytest.approx(30.0, rel=1e-5)
    assert result.sigma_a[0] == pytest.approx(135.0, rel=1e-5)
    assert result.sigma_m[0] == pytest.approx(165.0, rel=1e-5)

    expected_sigma_ar = 135.0 / (1.0 - 165.0 / 860.0)
    assert result.sigma_ar[0] == pytest.approx(expected_sigma_ar, rel=1e-5)

    expected_sf = 1.0 / ((135.0 / 510.0) + (165.0 / 860.0))
    assert result.safety_factor[0] == pytest.approx(expected_sf, rel=1e-5)
    assert np.isinf(result.life_cycles[0])
    assert result.damage[0] == pytest.approx(0.0)


def test_hand_calculation_finite_life(ti6al4v_material):
    """Verify analytical solution for cyclic stress exceeding endurance limit (finite life).
    
    Given:
        sigma_xx_max = 700.0 MPa, sigma_xx_min = 70.0 MPa (R = 0.1)
    Hand Calculation:
        sigma_a = 315.0 MPa
        sigma_m = 385.0 MPa
        sigma_ar = 315.0 / (1.0 - 385.0 / 860.0) = 315.0 / 0.55232558 = 570.3158 MPa
        Since sigma_ar (570.32) > Se (510.0):
            Basquin life: N_f = 0.5 * (570.3158 / 1440.0)^(1 / -0.095)
            ratio = 570.315789 / 1440.0 = 0.39605263
            inv_b = -10.52631579
            N_f = 0.5 * (0.39605263)^(-10.52631579) = 0.5 * 18074.83 = 9037.41 cycles
            damage = 5.0e6 / 9037.41 = 553.25
        Fatigue safety factor:
            1 / SF = (315.0 / 510.0) + (385.0 / 860.0) = 0.617647 + 0.447674 = 1.065321
            SF = 0.93868
    """
    s_max = np.array([[700.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_min = np.array([[70.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    load_case = FatigueLoadCase(
        target_cycles=5.0e+06,
        surface_factor_ka=1.0,
        notch_factor_kf=1.0,
        mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
    )

    result = evaluate_fatigue(s_max, s_min, material=ti6al4v_material, load_case=load_case)

    expected_sigma_ar = 315.0 / (1.0 - 385.0 / 860.0)
    assert result.sigma_ar[0] == pytest.approx(expected_sigma_ar, rel=1e-5)

    expected_sf = 1.0 / ((315.0 / 510.0) + (385.0 / 860.0))
    assert result.safety_factor[0] == pytest.approx(expected_sf, rel=1e-5)

    expected_life = 0.5 * ((expected_sigma_ar / 1440.0) ** (1.0 / -0.095))
    assert result.life_cycles[0] == pytest.approx(expected_life, rel=1e-4)

    expected_damage = 5.0e+06 / expected_life
    assert result.damage[0] == pytest.approx(expected_damage, rel=1e-4)


def test_notch_and_surface_factors(ti6al4v_material):
    """Verify modification of effective stress amplitude via ka and kf."""
    s_max = np.array([[400.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_min = np.array([[40.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    # Baseline: ka = 1.0, kf = 1.0
    lc_base = FatigueLoadCase(surface_factor_ka=1.0, notch_factor_kf=1.0)
    res_base = evaluate_fatigue(s_max, s_min, material=ti6al4v_material, load_case=lc_base)

    # Notched & rough surface: ka = 0.8, kf = 2.0 -> amplification = 2.0 / 0.8 = 2.5
    lc_notched = FatigueLoadCase(surface_factor_ka=0.8, notch_factor_kf=2.0)
    res_notched = evaluate_fatigue(s_max, s_min, material=ti6al4v_material, load_case=lc_notched)

    # Base: sigma_a = 180, sigma_m = 220
    # Notched: sigma_a_eff = 180 * 2.5 = 450
    # Goodman:
    # 1/SF_base = 180 / 510 + 220 / 860 = 0.35294 + 0.25581 = 0.60875 -> SF = 1.6427
    # 1/SF_notched = 450 / 510 + 220 / 860 = 0.88235 + 0.25581 = 1.13816 -> SF = 0.8786
    assert res_base.safety_factor[0] == pytest.approx(1.0 / (180.0 / 510.0 + 220.0 / 860.0), rel=1e-5)
    assert res_notched.safety_factor[0] == pytest.approx(1.0 / (450.0 / 510.0 + 220.0 / 860.0), rel=1e-5)
    assert res_notched.safety_factor[0] < res_base.safety_factor[0]


def test_automatic_stress_ratio_derivation(ti6al4v_material):
    """Verify automatic derivation of stress_min when None is supplied."""
    s_max = np.array([[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    load_case = FatigueLoadCase(stress_ratio_R=0.1)

    result = evaluate_fatigue(s_max, stress_min=None, material=ti6al4v_material, load_case=load_case)

    assert result.sigma_min[0] == pytest.approx(30.0, rel=1e-5)
    assert result.sigma_max[0] == pytest.approx(300.0, rel=1e-5)
    assert result.sigma_a[0] == pytest.approx(135.0, rel=1e-5)


def test_pure_shear_cyclic_loading(ti6al4v_material):
    """Verify pure shear stress tensor cycling."""
    # SXY cycles between 200 and 20 MPa.
    # von Mises for pure shear tau: sqrt(3) * tau
    s_max = np.array([[0.0, 0.0, 0.0, 200.0, 0.0, 0.0]])
    s_min = np.array([[0.0, 0.0, 0.0, 20.0, 0.0, 0.0]])

    load_case = FatigueLoadCase(
        mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
        stress_measure=StressMeasureType.VON_MISES,
    )
    res = evaluate_fatigue(s_max, s_min, material=ti6al4v_material, load_case=load_case)

    vm_max = np.sqrt(3.0) * 200.0
    vm_min = np.sqrt(3.0) * 20.0
    assert res.sigma_max[0] == pytest.approx(vm_max, rel=1e-5)
    assert res.sigma_min[0] == pytest.approx(vm_min, rel=1e-5)


def test_gerber_and_soderberg_evaluation(ti6al4v_material):
    """Verify Gerber and Soderberg mean stress theories in evaluate_fatigue."""
    s_max = np.array([[400.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    s_min = np.array([[40.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    lc_gerber = FatigueLoadCase(mean_stress_method=MeanStressCorrectionMethod.GERBER)
    res_gerber = evaluate_fatigue(s_max, s_min, material=ti6al4v_material, load_case=lc_gerber)

    lc_soder = FatigueLoadCase(mean_stress_method=MeanStressCorrectionMethod.SODERBERG)
    res_soder = evaluate_fatigue(s_max, s_min, material=ti6al4v_material, load_case=lc_soder)

    # Soderberg uses Sy (795) and is the most conservative: SF_soderberg < SF_goodman < SF_gerber
    lc_goodman = FatigueLoadCase(mean_stress_method=MeanStressCorrectionMethod.GOODMAN)
    res_goodman = evaluate_fatigue(s_max, s_min, material=ti6al4v_material, load_case=lc_goodman)

    assert res_soder.safety_factor[0] < res_goodman.safety_factor[0]
    assert res_goodman.safety_factor[0] < res_gerber.safety_factor[0]


def test_performance_100k_nodes_under_1s(ti6al4v_material):
    """Acceptance criterion: 100,000 nodes processed in under 1.0 second."""
    num_nodes = 100000
    rng = np.random.default_rng(12345)

    # Generate synthetic 100k nodes with non-trivial 6-component stress tensors
    stress_max = rng.uniform(-400.0, 400.0, size=(num_nodes, 6))
    stress_min = stress_max * 0.1

    load_case = FatigueLoadCase(
        target_cycles=5.0e+06,
        surface_factor_ka=0.9,
        notch_factor_kf=1.5,
        mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
        stress_measure=StressMeasureType.SIGNED_VON_MISES,
    )

    t0 = time.perf_counter()
    result = evaluate_fatigue(
        stress_max=stress_max,
        stress_min=stress_min,
        material=ti6al4v_material,
        load_case=load_case,
    )
    t_elapsed = time.perf_counter() - t0

    print(f"\n[BENCHMARK] 100k nodes fatigue evaluation time: {t_elapsed:.4f} seconds")

    # Assert performance requirement (< 1.0 s)
    assert t_elapsed < 1.0, f"Processing 100k nodes took {t_elapsed:.4f}s (exceeded 1.0s limit)"

    # Verify result shapes and attributes
    assert len(result.node_indices) == num_nodes
    assert len(result.safety_factor) == num_nodes
    assert len(result.life_cycles) == num_nodes
    assert len(result.damage) == num_nodes
    assert result.min_safety_factor >= 0.0
    assert 0 <= result.hotspot_node_index < num_nodes


def test_input_validation_errors(ti6al4v_material):
    """Verify input validation handles invalid shapes and arguments."""
    with pytest.raises(ValueError, match="FatigueMaterial instance must be provided"):
        evaluate_fatigue(np.zeros((10, 6)), material=None)

    with pytest.raises(ValueError, match="node counts must match"):
        evaluate_fatigue(np.zeros((10, 6)), np.zeros((8, 6)), material=ti6al4v_material)

    with pytest.raises(ValueError, match="node_indices length"):
        evaluate_fatigue(np.zeros((10, 6)), material=ti6al4v_material, node_indices=np.arange(5))
