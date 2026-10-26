"""
test_regression_freeze.py
Phase 5 Step 5.8 Regression Test and Freeze:
Verifies that Step 5.3 (Cantilever Benchmark) and Step 5.6 (ISO 14801 Benchmark)
analytical theory, CalculiX simulation results, and clinical validation rules
strictly adhere to the frozen baseline golden standards without regression.
"""

import os
import json
import math
import shutil
import pytest
import numpy as np

from fea.models import MeshModel, MeshPart
from fatigue.cantilever_benchmark import (
    calculate_cantilever_theory,
    run_cantilever_benchmark,
    CantileverTheory,
)
from fatigue.iso14801 import (
    ISO14801Config,
    ISO14801SetupSpec,
    validate_iso14801_setup,
    identify_fatigue_hotspot,
    calculate_predicted_fatigue_limit_load,
    compare_with_literature_benchmark,
    run_iso14801_batch_case,
)
from fatigue.models import (
    FatigueMaterial,
    FatigueResult,
    MeanStressCorrectionMethod,
    StressMeasureType,
)

CCX_PATH = shutil.which("ccx") or "/usr/bin/ccx"
HAS_CCX = os.path.isfile(CCX_PATH) and os.access(CCX_PATH, os.X_OK)

FIXTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixtures"))
BASELINE_5_3_PATH = os.path.join(FIXTURES_DIR, "fatigue_benchmark_5_3_baseline.json")
BASELINE_5_6_PATH = os.path.join(FIXTURES_DIR, "fatigue_iso14801_5_6_baseline.json")


class TestStep53CantileverRegressionFreeze:
    """Step 5.3 Cantilever Beam analytical theory and FEA benchmark freeze tests."""

    def test_01_cantilever_analytical_theory_matches_frozen_baseline(self):
        """Verify exact match of closed-form beam theory formulas with frozen 5.3 baseline."""
        with open(BASELINE_5_3_PATH, "r", encoding="utf-8") as f:
            base = json.load(f)

        geom = base["geometry"]
        loading = base["loading"]
        mat_dict = base["material"]
        theory_gold = base["analytical_theory"]

        mat = FatigueMaterial(
            name=mat_dict["name"],
            yield_strength_Sy=mat_dict["yield_strength_Sy_MPa"],
            tensile_strength_Sut=mat_dict["tensile_strength_Sut_MPa"],
            fatigue_limit_Se=mat_dict["fatigue_limit_Se_MPa"],
        )

        theory = calculate_cantilever_theory(
            length_L=geom["length_L_mm"],
            width_b=geom["width_b_mm"],
            height_h=geom["height_h_mm"],
            force_max=loading["force_max_N"],
            force_min=loading["force_min_N"],
            material=mat,
        )

        # 단면 2차 모멘트 및 단면계수 불변성 검증
        assert math.isclose(theory.moment_of_inertia_I, geom["moment_of_inertia_I_mm4"], rel_tol=1e-7)
        assert math.isclose(theory.section_modulus_Z, geom["section_modulus_Z_mm3"], rel_tol=1e-7)

        # 굽힘 모멘트 및 이론 응력 불변성 검증
        assert math.isclose(theory.moment_max, loading["moment_max_Nmm"], rel_tol=1e-7)
        assert math.isclose(theory.moment_min, loading["moment_min_Nmm"], rel_tol=1e-7)
        assert math.isclose(theory.sigma_max, theory_gold["sigma_max_MPa"], rel_tol=1e-7)
        assert math.isclose(theory.sigma_min, theory_gold["sigma_min_MPa"], rel_tol=1e-7)
        assert math.isclose(theory.sigma_a, theory_gold["sigma_a_MPa"], rel_tol=1e-7)
        assert math.isclose(theory.sigma_m, theory_gold["sigma_m_MPa"], rel_tol=1e-7)

        # 평균응력 보정 안전계수(Goodman, Soderberg, Gerber) 불변성 검증
        fs_gold = theory_gold["safety_factors"]
        assert math.isclose(theory.fs_goodman, fs_gold["goodman"], rel_tol=1e-6)
        assert math.isclose(theory.fs_soderberg, fs_gold["soderberg"], rel_tol=1e-6)
        assert math.isclose(theory.fs_gerber, fs_gold["gerber"], rel_tol=1e-6)

    @pytest.mark.skipif(not HAS_CCX, reason="CalculiX ccx binary not found on system.")
    def test_02_cantilever_fea_simulation_matches_frozen_baseline(self):
        """Verify CalculiX FEA cantilever benchmark meets frozen accuracy bounds (<= 3.0%)."""
        with open(BASELINE_5_3_PATH, "r", encoding="utf-8") as f:
            base = json.load(f)

        geom = base["geometry"]
        loading = base["loading"]
        fea_gold = base["fea_calculix_validation"]

        res = run_cantilever_benchmark(
            length_L=geom["length_L_mm"],
            width_b=geom["width_b_mm"],
            height_h=geom["height_h_mm"],
            force_max=loading["force_max_N"],
            force_min=loading["force_min_N"],
            mesh_size=3.0,
            ccx_path=CCX_PATH,
            mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
            tolerance_pct=fea_gold["tolerance_pct"],
        )

        assert res.passed is True
        assert res.passed_stress is True
        assert res.passed_safety_factor is True

        # 기준 오차 한계(3.0%) 및 평균 실측치 검증
        assert res.max_stress_error_pct <= fea_gold["tolerance_pct"]
        assert res.max_fs_error_pct <= fea_gold["tolerance_pct"]
        assert res.mean_stress_error_pct < 1.0
        assert res.mean_fs_error_pct < 1.0

        # 기준 절점 수 검증
        assert res.num_surface_nodes_evaluated == fea_gold["surface_nodes_evaluated"]


class TestStep56ISO14801RegressionFreeze:
    """Step 5.6 ISO 14801 Benchmark criteria and clinical validation freeze tests."""

    def test_01_iso14801_geometric_spec_matches_frozen_baseline(self):
        """Verify ISO 14801 geometric specification tolerances adhere to baseline standards."""
        with open(BASELINE_5_6_PATH, "r", encoding="utf-8") as f:
            base = json.load(f)

        geo_spec = base["geometric_specifications"]
        spec = ISO14801SetupSpec()

        assert spec.nominal_angle_deg == geo_spec["nominal_angle_deg"]
        assert spec.angle_tolerance_deg == geo_spec["angle_tolerance_deg"]
        assert spec.nominal_lever_arm_mm == geo_spec["nominal_lever_arm_mm"]
        assert spec.lever_arm_tolerance_mm == geo_spec["lever_arm_tolerance_mm"]
        assert spec.nominal_resorption_depth_mm == geo_spec["nominal_resorption_depth_mm"]
        assert spec.resorption_depth_tolerance_mm == geo_spec["resorption_depth_tolerance_mm"]
        assert spec.stress_ratio_R == geo_spec["stress_ratio_R"]
        assert spec.stress_ratio_tolerance == geo_spec["stress_ratio_tolerance"]

    def test_02_iso14801_clinical_hotspot_rules_match_frozen_baseline(self):
        """Verify hotspot classification logic against clinical baseline criteria."""
        with open(BASELINE_5_6_PATH, "r", encoding="utf-8") as f:
            base = json.load(f)

        hotspot_rules = base["clinical_hotspot_criteria"]
        assert "screw" in hotspot_rules["valid_regions"]
        assert "fixture_neck" in hotspot_rules["valid_regions"]

        # 3파트 더미 모델 생성
        model = MeshModel(parts=[
            MeshPart(name="screw", nodes=np.array([[0.0, 0.0, 5.0]]), elements=np.array([[0]]), element_type="C3D4"),
            MeshPart(name="fixture", nodes=np.array([[1.5, 0.0, 2.0]]), elements=np.array([[0]]), element_type="C3D4"),
        ])

        # 1. 나사 핫스팟 평가
        part_res_screw = {
            "screw": FatigueResult(
                node_indices=np.array([0]),
                sigma_max=np.array([350.0]),
                sigma_min=np.array([35.0]),
                sigma_a=np.array([200.0]),
                sigma_m=np.array([150.0]),
                sigma_ar=np.array([200.0]),
                safety_factor=np.array([1.1]),
                life_cycles=np.array([1e7]),
                damage=np.array([0.1]),
            ),
            "fixture": FatigueResult(
                node_indices=np.array([0]),
                sigma_max=np.array([80.0]),
                sigma_min=np.array([8.0]),
                sigma_a=np.array([50.0]),
                sigma_m=np.array([30.0]),
                sigma_ar=np.array([50.0]),
                safety_factor=np.array([2.5]),
                life_cycles=np.array([1e7]),
                damage=np.array([0.01]),
            ),
        }
        hotspot_screw = identify_fatigue_hotspot(model, part_res_screw)
        assert hotspot_screw.is_clinically_valid is True
        assert "지대주 나사" in hotspot_screw.hotspot_region_name

        # 2. 고정체 목부 핫스팟 평가 (z <= 3.0mm)
        part_res_fixture = {
            "screw": FatigueResult(
                node_indices=np.array([0]),
                sigma_max=np.array([60.0]),
                sigma_min=np.array([6.0]),
                sigma_a=np.array([40.0]),
                sigma_m=np.array([20.0]),
                sigma_ar=np.array([40.0]),
                safety_factor=np.array([3.0]),
                life_cycles=np.array([1e7]),
                damage=np.array([0.01]),
            ),
            "fixture": FatigueResult(
                node_indices=np.array([0]),
                sigma_max=np.array([300.0]),
                sigma_min=np.array([30.0]),
                sigma_a=np.array([180.0]),
                sigma_m=np.array([120.0]),
                sigma_ar=np.array([180.0]),
                safety_factor=np.array([1.15]),
                life_cycles=np.array([1e7]),
                damage=np.array([0.08]),
            ),
        }
        hotspot_neck = identify_fatigue_hotspot(model, part_res_fixture)
        assert hotspot_neck.is_clinically_valid is True
        assert "고정체 목부" in hotspot_neck.hotspot_region_name

    def test_03_literature_benchmark_values_and_tolerances_match_baseline(self):
        """Verify Gehrke 2020 literature benchmark value (316 N) and error classifications."""
        with open(BASELINE_5_6_PATH, "r", encoding="utf-8") as f:
            base = json.load(f)

        lit_gold = base["literature_benchmarks"]["gehrke_2020_conical"]
        target_f = lit_gold["literature_fatigue_limit_N"]

        # 양호 판정 (오차 5% 이내 -> GOOD)
        comp_good = compare_with_literature_benchmark(
            predicted_fatigue_limit_N=target_f * 1.05,
            benchmark_id="gehrke_2020_conical",
        )
        assert comp_good.status == "GOOD"
        assert comp_good.relative_error_pct < lit_gold["tolerance_pct"]

        # 재검토 권장 판정 (오차 50% -> RECHECK_NEEDED)
        comp_recheck = compare_with_literature_benchmark(
            predicted_fatigue_limit_N=target_f * 1.50,
            benchmark_id="gehrke_2020_conical",
        )
        assert comp_recheck.status == "RECHECK_NEEDED"
        assert comp_recheck.relative_error_pct > lit_gold["tolerance_pct"]

    def test_04_cross_validation_policy_unverified_fallback(self):
        """Verify unverified status reporting when literature benchmark is absent."""
        with open(BASELINE_5_6_PATH, "r", encoding="utf-8") as f:
            base = json.load(f)

        policy = base["cross_validation_policy"]

        comp_unverified = compare_with_literature_benchmark(
            predicted_fatigue_limit_N=250.0,
            benchmark_id=None,
        )
        assert comp_unverified.status == policy["unverified_status"]
        assert policy["unverified_display_text"] in comp_unverified.status_display_text
