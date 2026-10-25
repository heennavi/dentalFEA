"""
test_iso14801_validation.py
Step 5.6 Acceptance Criteria Validation: ISO 14801 integrated fatigue test pipeline.

Acceptance criteria:
1. Loading point position and angle match ISO 14801 specifications (automated script verification).
2. Critical fatigue hotspot is identified at clinically/experimentally observed locations (screw or fixture neck).
3. Predicted fatigue limit load (safety factor = 1.0) is calculated and compared with literature benchmarks.
   - Error within tens of percent -> Good / acceptable.
   - Large discrepancy -> recommend re-checking contact, preload, materials.
4. Honest limitation: When published empirical data is unavailable for the implant system,
   status is explicitly reported as "교차 검증 미실시" (Cross-validation not performed).
"""

import math
import os
import shutil
import pytest
import numpy as np

from fea.models import MeshModel, MeshPart
from fatigue.models import (
    FatigueMaterial,
    FatigueResult,
    MeanStressCorrectionMethod,
    StressMeasureType,
)
from fatigue.iso14801 import (
    ISO14801Config,
    ISO14801LoadMode,
    ISO14801SetupSpec,
    ISO14801ValidationReport,
    FatigueHotspot,
    LiteratureBenchmarkComparison,
    ISO14801BatchResult,
    validate_iso14801_setup,
    build_iso14801_benchmark_assembly,
    identify_fatigue_hotspot,
    calculate_predicted_fatigue_limit_load,
    compare_with_literature_benchmark,
    run_iso14801_batch_case,
)

CCX_PATH = shutil.which("ccx") or "/usr/bin/ccx"
HAS_CCX = os.path.isfile(CCX_PATH) and os.access(CCX_PATH, os.X_OK)


class TestISO14801GeometricValidation:
    """Acceptance Criterion 1: Automated script verification of load application point and angle."""

    def test_01_standard_setup_passes_verification(self):
        """Standard ISO 14801 dimensions (30 deg, 11 mm lever arm, 3 mm resorption, R=0.1) pass."""
        clamping_z = 0.0
        crest_z = 3.0
        loading_pt = np.array([0.0, 0.0, 11.0])
        # 30 deg tilted compression force: Fx = -250*sin(30) = -125 N, Fz = -250*cos(30) = -216.506 N
        f_max = np.array([-125.0, 0.0, -216.50635])
        f_min = f_max * 0.1

        report = validate_iso14801_setup(
            clamping_z=clamping_z,
            crest_z=crest_z,
            loading_point=loading_pt,
            force_vector_max=f_max,
            force_vector_min=f_min,
        )

        assert report.passed is True
        assert report.passed_angle is True
        assert math.isclose(report.measured_angle_deg, 30.0, abs_tol=0.1)
        assert report.passed_lever_arm is True
        assert math.isclose(report.measured_lever_arm_mm, 11.0, abs_tol=0.01)
        assert report.passed_resorption is True
        assert math.isclose(report.measured_resorption_depth_mm, 3.0, abs_tol=0.01)
        assert report.passed_stress_ratio is True
        assert math.isclose(report.measured_stress_ratio_R, 0.1, abs_tol=0.005)

    def test_02_deviant_angle_fails_verification(self):
        """Angle out of 30 +/- 2 deg tolerance fails verification with detailed error."""
        clamping_z = 0.0
        crest_z = 3.0
        loading_pt = np.array([0.0, 0.0, 11.0])
        # 40 deg angle: outside 28~32 deg tolerance
        angle_rad = math.radians(40.0)
        f_max = np.array([-250.0 * math.sin(angle_rad), 0.0, -250.0 * math.cos(angle_rad)])

        report = validate_iso14801_setup(
            clamping_z=clamping_z,
            crest_z=crest_z,
            loading_point=loading_pt,
            force_vector_max=f_max,
        )

        assert report.passed is False
        assert report.passed_angle is False
        assert report.angle_error_deg > 5.0
        assert report.passed_lever_arm is True
        assert report.passed_resorption is True

    def test_03_deviant_lever_arm_fails_verification(self):
        """Lever arm distance outside 11.0 +/- 0.5 mm tolerance fails verification."""
        clamping_z = 0.0
        crest_z = 3.0
        loading_pt = np.array([0.0, 0.0, 13.5])  # 13.5 mm lever arm
        f_max = np.array([-125.0, 0.0, -216.50635])

        report = validate_iso14801_setup(
            clamping_z=clamping_z,
            crest_z=crest_z,
            loading_point=loading_pt,
            force_vector_max=f_max,
        )

        assert report.passed is False
        assert report.passed_lever_arm is False
        assert report.lever_arm_error_mm >= 2.0
        assert report.passed_angle is True

    def test_04_deviant_resorption_depth_fails_verification(self):
        """Bone resorption exposure outside 3.0 +/- 0.2 mm fails verification."""
        clamping_z = 0.0
        crest_z = 1.5  # Only 1.5 mm exposure
        loading_pt = np.array([0.0, 0.0, 11.0])
        f_max = np.array([-125.0, 0.0, -216.50635])

        report = validate_iso14801_setup(
            clamping_z=clamping_z,
            crest_z=crest_z,
            loading_point=loading_pt,
            force_vector_max=f_max,
        )

        assert report.passed is False
        assert report.passed_resorption is False
        assert report.resorption_depth_error_mm >= 1.0


class TestISO14801HotspotIdentification:
    """Acceptance Criterion 2: Hotspot identified at screw or fixture neck."""

    def test_01_hotspot_in_abutment_screw(self):
        """Hotspot occurring in abutment screw is classified as clinically valid failure mode."""
        model, meta = build_iso14801_benchmark_assembly()

        # Dummy fatigue results where screw has the lowest safety factor (0.85)
        part_results = {
            "fixture": FatigueResult(
                node_indices=np.arange(len(model.get_part("fixture").nodes)),
                sigma_max=np.full(len(model.get_part("fixture").nodes), 150.0),
                sigma_min=np.full(len(model.get_part("fixture").nodes), 15.0),
                sigma_a=np.full(len(model.get_part("fixture").nodes), 67.5),
                sigma_m=np.full(len(model.get_part("fixture").nodes), 82.5),
                sigma_ar=np.full(len(model.get_part("fixture").nodes), 67.5),
                safety_factor=np.full(len(model.get_part("fixture").nodes), 2.5),
                life_cycles=np.full(len(model.get_part("fixture").nodes), 1e7),
                damage=np.zeros(len(model.get_part("fixture").nodes)),
            ),
            "abutment": FatigueResult(
                node_indices=np.arange(len(model.get_part("abutment").nodes)),
                sigma_max=np.full(len(model.get_part("abutment").nodes), 220.0),
                sigma_min=np.full(len(model.get_part("abutment").nodes), 22.0),
                sigma_a=np.full(len(model.get_part("abutment").nodes), 99.0),
                sigma_m=np.full(len(model.get_part("abutment").nodes), 121.0),
                sigma_ar=np.full(len(model.get_part("abutment").nodes), 99.0),
                safety_factor=np.full(len(model.get_part("abutment").nodes), 1.8),
                life_cycles=np.full(len(model.get_part("abutment").nodes), 1e7),
                damage=np.zeros(len(model.get_part("abutment").nodes)),
            ),
            "screw": FatigueResult(
                node_indices=np.arange(len(model.get_part("screw").nodes)),
                sigma_max=np.full(len(model.get_part("screw").nodes), 480.0),
                sigma_min=np.full(len(model.get_part("screw").nodes), 48.0),
                sigma_a=np.full(len(model.get_part("screw").nodes), 216.0),
                sigma_m=np.full(len(model.get_part("screw").nodes), 264.0),
                sigma_ar=np.full(len(model.get_part("screw").nodes), 216.0),
                safety_factor=np.full(len(model.get_part("screw").nodes), 0.85),
                life_cycles=np.full(len(model.get_part("screw").nodes), 2.5e5),
                damage=np.full(len(model.get_part("screw").nodes), 20.0),
            ),
        }

        hotspot = identify_fatigue_hotspot(model, part_results, clamping_z=0.0, crest_z=3.0)

        assert hotspot.part_name == "screw"
        assert hotspot.min_safety_factor == 0.85
        assert hotspot.max_stress_MPa == 480.0
        assert hotspot.is_clinically_valid is True
        assert "나사" in hotspot.hotspot_region_name or "Screw" in hotspot.hotspot_region_name

    def test_02_hotspot_in_fixture_neck(self):
        """Hotspot occurring in fixture collar/neck (z in [0.0, 3.5]) is classified as clinically valid."""
        model, meta = build_iso14801_benchmark_assembly()

        # Place critical stress node in fixture near clamping plane (z = 1.5)
        fix_part = model.get_part("fixture")
        fs_array = np.full(len(fix_part.nodes), 3.0)
        # Set node with z = 1.5 as critical
        neck_nodes = np.where(np.abs(fix_part.nodes[:, 2] - 1.5) < 1e-4)[0]
        fs_array[neck_nodes[0]] = 0.92

        part_results = {
            "fixture": FatigueResult(
                node_indices=np.arange(len(fix_part.nodes)),
                sigma_max=np.full(len(fix_part.nodes), 450.0),
                sigma_min=np.full(len(fix_part.nodes), 45.0),
                sigma_a=np.full(len(fix_part.nodes), 202.5),
                sigma_m=np.full(len(fix_part.nodes), 247.5),
                sigma_ar=np.full(len(fix_part.nodes), 202.5),
                safety_factor=fs_array,
                life_cycles=np.full(len(fix_part.nodes), 4.0e5),
                damage=np.full(len(fix_part.nodes), 12.5),
            ),
            "abutment": FatigueResult(
                node_indices=np.arange(len(model.get_part("abutment").nodes)),
                sigma_max=np.full(len(model.get_part("abutment").nodes), 200.0),
                sigma_min=np.full(len(model.get_part("abutment").nodes), 20.0),
                sigma_a=np.full(len(model.get_part("abutment").nodes), 90.0),
                sigma_m=np.full(len(model.get_part("abutment").nodes), 110.0),
                sigma_ar=np.full(len(model.get_part("abutment").nodes), 90.0),
                safety_factor=np.full(len(model.get_part("abutment").nodes), 2.0),
                life_cycles=np.full(len(model.get_part("abutment").nodes), 1e7),
                damage=np.zeros(len(model.get_part("abutment").nodes)),
            ),
        }

        hotspot = identify_fatigue_hotspot(model, part_results, clamping_z=0.0, crest_z=3.0)

        assert hotspot.part_name == "fixture"
        assert hotspot.min_safety_factor == 0.92
        assert hotspot.is_clinically_valid is True
        assert "목부" in hotspot.hotspot_region_name or "Neck" in hotspot.hotspot_region_name


class TestPredictedFatigueLimitAndLiterature:
    """Acceptance Criterion 3 & 4: Predicted fatigue limit load (FS=1.0) and literature comparison."""

    def test_01_predicted_fatigue_limit_linear_scaling(self):
        """Verify F_limit calculation where FS = 1.0."""
        # Applied 250 N with FS = 1.25 -> F_limit = 250 * 1.25 = 312.5 N
        f_limit = calculate_predicted_fatigue_limit_load(250.0, 1.25, MeanStressCorrectionMethod.GOODMAN)
        assert math.isclose(f_limit, 312.5, rel_tol=1e-5)

        # Applied 300 N with FS = 0.80 -> F_limit = 300 * 0.80 = 240.0 N
        f_limit_sub = calculate_predicted_fatigue_limit_load(300.0, 0.80, MeanStressCorrectionMethod.GOODMAN)
        assert math.isclose(f_limit_sub, 240.0, rel_tol=1e-5)

    def test_02_literature_comparison_good_match(self):
        """When literature data matches within tens of percent (<= 35%), status is GOOD."""
        # Gehrke 2020 Conical: 316.0 N
        # FEA prediction: 305.0 N (error: 3.48%)
        comp = compare_with_literature_benchmark(
            predicted_fatigue_limit_N=305.0,
            benchmark_id="gehrke_2020_conical",
        )

        assert comp.status == "GOOD"
        assert comp.relative_error_pct is not None
        assert comp.relative_error_pct < 10.0
        assert "양호" in comp.status_display_text
        assert math.isclose(comp.literature_fatigue_limit_N, 316.0, rel_tol=1e-5)

    def test_03_literature_comparison_discrepancy_recheck(self):
        """When error exceeds 35%, status is RECHECK_NEEDED recommending contact/preload review."""
        # FEA prediction: 120.0 N vs 316.0 N (error: 62%)
        comp = compare_with_literature_benchmark(
            predicted_fatigue_limit_N=120.0,
            benchmark_id="gehrke_2020_conical",
            tolerance_pct=35.0,
        )

        assert comp.status == "RECHECK_NEEDED"
        assert comp.relative_error_pct > 35.0
        assert "재점검" in comp.status_display_text
        assert "선하중" in comp.details or "접촉" in comp.details

    def test_04_honest_limitation_unverified_when_no_data(self):
        """[Honest Limitation] When benchmark_id is None or unknown, status is UNVERIFIED ('교차 검증 미실시')."""
        # Case A: benchmark_id is None
        comp_none = compare_with_literature_benchmark(
            predicted_fatigue_limit_N=290.0,
            benchmark_id=None,
        )
        assert comp_none.status == "UNVERIFIED"
        assert "교차 검증 미실시" in comp_none.status_display_text
        assert comp_none.relative_error_pct is None

        # Case B: unknown/unregistered implant system
        comp_unknown = compare_with_literature_benchmark(
            predicted_fatigue_limit_N=290.0,
            benchmark_id="custom_proprietary_implant_xyz",
        )
        assert comp_unknown.status == "UNVERIFIED"
        assert "교차 검증 미실시" in comp_unknown.status_display_text
        assert comp_unknown.relative_error_pct is None


@pytest.mark.skipif(not HAS_CCX, reason="CalculiX solver binary (ccx) not available on this system.")
class TestISO14801BatchEndToEndCalculix:
    """End-to-end integration test of Step 5.6 ISO 14801 batch fatigue pipeline with CalculiX."""

    def test_01_full_batch_case_execution(self):
        """Execute full CalculiX pipeline on standard ISO 14801 assembly and verify all criteria."""
        cfg = ISO14801Config(
            load_max_N=250.0,
            stress_ratio_R=0.1,
            angle_deg=30.0,
            mode=ISO14801LoadMode.DUAL_SIMULATION,
        )

        batch_res = run_iso14801_batch_case(
            config=cfg,
            benchmark_id="gehrke_2020_conical",
            ccx_path=CCX_PATH,
        )

        # 1. Verification Criterion: Setup dimensions
        assert batch_res.validation_report.passed is True
        assert batch_res.validation_report.passed_angle is True
        assert batch_res.validation_report.passed_lever_arm is True
        assert batch_res.validation_report.passed_resorption is True
        assert batch_res.validation_report.passed_stress_ratio is True

        # 2. Verification Criterion: Hotspot location (screw or fixture neck)
        assert batch_res.hotspot.is_clinically_valid is True
        assert batch_res.hotspot.part_name in ("screw", "fixture", "abutment")

        # 3. Verification Criterion: Predicted fatigue limit load
        assert batch_res.predicted_fatigue_limit_N > 0.0
        # Check against literature
        assert batch_res.benchmark_comparison.status in ("GOOD", "RECHECK_NEEDED")

        # Overall pass
        assert batch_res.passed is True

    def test_02_batch_case_with_unverified_implant(self):
        """Batch case with unverified implant correctly reports '교차 검증 미실시'."""
        cfg = ISO14801Config(
            load_max_N=250.0,
            stress_ratio_R=0.1,
            angle_deg=30.0,
            mode=ISO14801LoadMode.STRESS_RATIO_R,
        )

        batch_res = run_iso14801_batch_case(
            config=cfg,
            benchmark_id=None,  # No published benchmark
            ccx_path=CCX_PATH,
        )

        assert batch_res.validation_report.passed is True
        assert batch_res.hotspot.is_clinically_valid is True
        assert batch_res.benchmark_comparison.status == "UNVERIFIED"
        assert "교차 검증 미실시" in batch_res.benchmark_comparison.status_display_text
        assert batch_res.passed is True


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    from PySide6.QtWidgets import QApplication
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestISO14801UIPanelIntegration:
    """Acceptance Criterion 4 (UI): Honest limitation reporting on ResultPanel."""

    def test_ui_shows_unverified_status_when_no_literature_data(self, qapp):
        """ResultPanel explicitly displays '교차 검증 미실시 (공개 자료 부재)' when unverified."""
        from dental_fea_app.core.state_manager import StateManager
        from ui.result_panel import ResultPanel

        state_mgr = StateManager()
        panel = ResultPanel(state_mgr)

        unverified_comp = LiteratureBenchmarkComparison(
            benchmark_id=None,
            implant_system_name="미지정",
            literature_reference="N/A",
            literature_fatigue_limit_N=None,
            predicted_fatigue_limit_N=275.0,
            relative_error_pct=None,
            status="UNVERIFIED",
            status_display_text="교차 검증 미실시 (공개 자료 부재)",
            details="동일 임플란트 계열의 공인된 ISO 14801 시험 자료가 없어 교차 검증을 실시하지 않았습니다.",
        )

        dummy_batch_res = ISO14801BatchResult(
            validation_report=ISO14801ValidationReport(
                passed=True,
                measured_angle_deg=30.0,
                angle_error_deg=0.0,
                passed_angle=True,
                measured_lever_arm_mm=11.0,
                lever_arm_error_mm=0.0,
                passed_lever_arm=True,
                measured_resorption_depth_mm=3.0,
                resorption_depth_error_mm=0.0,
                passed_resorption=True,
                measured_stress_ratio_R=0.1,
                passed_stress_ratio=True,
            ),
            hotspot=FatigueHotspot(
                part_name="screw",
                node_id=10,
                coordinates=np.array([0.0, 0.0, 4.0]),
                min_safety_factor=1.1,
                max_stress_MPa=420.0,
                is_clinically_valid=True,
                hotspot_region_name="지대주 나사 (Abutment Screw)",
                clinical_notes="피로 파절 호발 부위 일치",
            ),
            predicted_fatigue_limit_N=275.0,
            benchmark_comparison=unverified_comp,
            part_results={},
            overall_min_safety_factor=1.1,
            passed=True,
        )

        panel.update_fatigue_iso14801_summary(dummy_batch_res)

        # Verify UI labels
        assert panel.group_fatigue.isHidden() is False
        assert "통과" in panel.lbl_iso_geom_status.text()
        assert "지대주 나사" in panel.lbl_iso_hotspot.text()
        assert "275.0 N" in panel.lbl_iso_f_limit.text()
        assert "교차 검증 미실시" in panel.lbl_iso_cross_validation.text()

    def test_ui_shows_good_status_when_literature_matches(self, qapp):
        """ResultPanel displays '양호 (문헌 오차 X.X%)' when literature comparison matches within tolerance."""
        from dental_fea_app.core.state_manager import StateManager
        from ui.result_panel import ResultPanel

        state_mgr = StateManager()
        panel = ResultPanel(state_mgr)

        good_comp = LiteratureBenchmarkComparison(
            benchmark_id="gehrke_2020_conical",
            implant_system_name="Ti-6Al-4V Conical",
            literature_reference="Gehrke 2020",
            literature_fatigue_limit_N=316.0,
            predicted_fatigue_limit_N=305.0,
            relative_error_pct=3.48,
            status="GOOD",
            status_display_text="양호 (문헌 오차 3.5%)",
            details="수십 % 이내 양호 판정 기준 만족",
        )

        dummy_batch_res = ISO14801BatchResult(
            validation_report=ISO14801ValidationReport(
                passed=True,
                measured_angle_deg=30.0,
                angle_error_deg=0.0,
                passed_angle=True,
                measured_lever_arm_mm=11.0,
                lever_arm_error_mm=0.0,
                passed_lever_arm=True,
                measured_resorption_depth_mm=3.0,
                resorption_depth_error_mm=0.0,
                passed_resorption=True,
                measured_stress_ratio_R=0.1,
                passed_stress_ratio=True,
            ),
            hotspot=FatigueHotspot(
                part_name="fixture",
                node_id=25,
                coordinates=np.array([2.0, 0.0, 1.5]),
                min_safety_factor=1.22,
                max_stress_MPa=390.0,
                is_clinically_valid=True,
                hotspot_region_name="고정체 목부 (Fixture Neck / Collar)",
                clinical_notes="목부 파절 호발 부위 일치",
            ),
            predicted_fatigue_limit_N=305.0,
            benchmark_comparison=good_comp,
            part_results={},
            overall_min_safety_factor=1.22,
            passed=True,
        )

        panel.update_fatigue_iso14801_summary(dummy_batch_res)

        assert panel.group_fatigue.isHidden() is False
        assert "고정체 목부" in panel.lbl_iso_hotspot.text()
        assert "305.0 N" in panel.lbl_iso_f_limit.text()
        assert "양호" in panel.lbl_iso_cross_validation.text()

