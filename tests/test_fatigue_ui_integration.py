"""
test_fatigue_ui_integration.py
Acceptance and integration tests for Step 5.7: Fatigue UI, Post-processing, Visualizer,
S-N/Haigh diagrams, Report Generator, and Step 5.3 & 5.6 numerical parity validation.
"""

import math
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import shutil
import pytest
import numpy as np
from PySide6.QtWidgets import QApplication

from dental_fea_app.core.state_manager import StateManager
from fea.models import MeshModel, MeshPart
from fea.results import ResultSet
from fatigue.models import (
    FatigueMaterial,
    FatigueLoadCase,
    FatigueResult,
    MeanStressCorrectionMethod,
    StressMeasureType,
)
from fatigue.ui_bridge import (
    evaluate_fatigue_on_result_set,
    run_benchmark_case_5_3,
    run_benchmark_case_5_6,
)
from fatigue.cantilever_benchmark import calculate_cantilever_theory
from fatigue.iso14801 import (
    validate_iso14801_setup,
    calculate_predicted_fatigue_limit_load,
    compare_with_literature_benchmark,
    identify_fatigue_hotspot,
)
from fatigue.report_generator import (
    generate_fatigue_report_html,
    generate_fatigue_report_text,
    DISCLAIMER_TEXT,
)
from postprocess.vtk_builder import build_part_grid, attach_results
from ui.result_view_controller import FIELD_DISPLAY_CONFIG
from dental_fea_app.ui.fatigue_panel import FatiguePanel
from dental_fea_app.ui.fatigue_diagrams import FatigueDiagramWidget


@pytest.fixture(scope="session")
def qapp():
    """Ensure a single QApplication instance for Qt GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def create_dummy_mesh_and_result_set(n_nodes: int = 100):
    """Create lightweight synthetic MeshModel and ResultSet for fatigue testing."""
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    nodes[:, 0] = np.linspace(0.0, 30.0, n_nodes)
    # Simple single-element placeholder or dummy elements
    elements = np.zeros((1, 4), dtype=np.int64)

    part = MeshPart(name="fixture", nodes=nodes, elements=elements, element_type="C3D4")
    model = MeshModel(parts=[part])

    # Uniaxial tensile stress in Z direction (e.g. 500 MPa)
    stress6 = np.zeros((n_nodes, 6), dtype=np.float64)
    stress6[:, 2] = np.linspace(100.0, 500.0, n_nodes)  # SZZ
    vm = stress6[:, 2]

    rs = ResultSet(
        time=1.0,
        u=np.zeros((n_nodes, 3), dtype=np.float64),
        u_mag=np.zeros(n_nodes, dtype=np.float64),
        stress6=stress6,
        von_mises=vm,
        s_max_principal=stress6[:, 2],
        s_min_principal=np.zeros(n_nodes, dtype=np.float64),
    )
    return model, rs


class TestFatigueUIIntegration:
    """Step 5.7 Acceptance Tests for Fatigue UI and Postprocessing Integration."""

    def test_01_banner_always_visible_and_disclaimer_content(self, qapp):
        """
        Verify that the clinical disclaimer and simulation assumptions are ALWAYS
        displayed in FatiguePanel banner and updated upon parameter changes.
        """
        sm = StateManager()
        panel = FatiguePanel(sm)
        panel.show()

        # 1. 필수 고지 문구 상시 표시 확인 (숨김 상태가 아님)
        assert not panel.banner_frame.isHidden()
        assert not panel.lbl_disclaimer.isHidden()
        assert "임상 판단 보조용" in panel.lbl_disclaimer.text()
        assert "시험 대체 불가" in panel.lbl_disclaimer.text()

        # 2. 가정 라벨 표시 확인 (재료, 척도, 보정, R, N)
        assumption_text = panel.lbl_assumptions.text()
        assert "가정:" in assumption_text
        assert "재료=" in assumption_text
        assert "척도=" in assumption_text
        assert "보정=" in assumption_text
        assert "R=0.10" in assumption_text
        assert "N=5.0e+06" in assumption_text

        # 3. R비 및 보정식 변경 시 배너가 실시간 갱신되는지 확인
        panel.spin_r.setValue(0.25)
        panel.combo_method.setCurrentIndex(1)  # Gerber
        updated_text = panel.lbl_assumptions.text()
        assert "R=0.25" in updated_text
        assert "gerber" in updated_text.lower()

    def test_02_field_display_config_includes_all_fatigue_fields(self):
        """
        Verify that FIELD_DISPLAY_CONFIG contains all 5 required fatigue visualization fields.
        """
        required_fields = [
            "fatigue_fs",
            "fatigue_life",
            "fatigue_damage",
            "fatigue_sigma_a",
            "fatigue_sigma_m",
        ]
        for f in required_fields:
            assert f in FIELD_DISPLAY_CONFIG, f"Field '{f}' missing from FIELD_DISPLAY_CONFIG"
            cfg = FIELD_DISPLAY_CONFIG[f]
            assert "title" in cfg
            assert "unit" in cfg
            assert "default_cmap" in cfg
            assert cfg["is_cell"] is False

    def test_03_vtk_builder_attaches_fatigue_scalar_arrays(self):
        """
        Verify that attach_results attaches all fatigue point arrays to vtkUnstructuredGrid.
        """
        model, rs = create_dummy_mesh_and_result_set(50)
        part = model.get_part("fixture")
        grid = build_part_grid(part)

        # Attach dummy fatigue arrays to ResultSet
        n = len(part.nodes)
        rs.fatigue_fs = np.full(n, 1.45, dtype=np.float64)
        rs.fatigue_life = np.full(n, 1.0e7, dtype=np.float64)
        rs.fatigue_damage = np.full(n, 0.5, dtype=np.float64)
        rs.fatigue_sigma_a = np.full(n, 225.0, dtype=np.float64)
        rs.fatigue_sigma_m = np.full(n, 275.0, dtype=np.float64)

        attach_results(grid, rs, model, part_name="fixture")

        pd = grid.GetPointData()
        assert pd.GetArray("fatigue_fs") is not None
        assert pd.GetArray("fatigue_life") is not None
        assert pd.GetArray("fatigue_damage") is not None
        assert pd.GetArray("fatigue_sigma_a") is not None
        assert pd.GetArray("fatigue_sigma_m") is not None

        # Verify numerical values
        fs_vtk = pd.GetArray("fatigue_fs")
        assert math.isclose(fs_vtk.GetValue(0), 1.45, rel_tol=1e-5)

    def test_04_evaluate_fatigue_bridge_on_result_set(self):
        """
        Verify evaluate_fatigue_on_result_set evaluates nodal fatigue and attaches arrays.
        """
        model, rs = create_dummy_mesh_and_result_set(50)
        mat = FatigueMaterial(
            name="Ti-6Al-4V ELI",
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
        )
        load_case = FatigueLoadCase(
            target_cycles=5.0e6,
            stress_ratio_R=0.1,
            mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
            stress_measure=StressMeasureType.SIGNED_VON_MISES,
        )

        updated_rs, hotspot = evaluate_fatigue_on_result_set(
            rs=rs,
            model=model,
            material=mat,
            load_case=load_case,
        )

        assert updated_rs.fatigue_fs is not None
        assert len(updated_rs.fatigue_fs) == 50
        assert updated_rs.fatigue_life is not None
        assert updated_rs.fatigue_damage is not None
        assert updated_rs.fatigue_sigma_a is not None
        assert updated_rs.fatigue_sigma_m is not None

        # Hotspot validation
        assert hotspot["min_safety_factor"] > 0.0
        assert hotspot["node_id"] == 49  # Node with highest stress (500 MPa)
        assert hotspot["part_name"] == "fixture"
        assert "임플란트 고정체 목부" in hotspot["hotspot_region_name"] or "Fixture" in hotspot["hotspot_region_name"]

    def test_05_step_5_3_cantilever_benchmark_numerical_parity(self):
        """
        [Step 5.7 Acceptance Criterion: Step 5.3 Parity]
        Verify exact numerical agreement with Step 5.3 analytical beam theory hand calculation.
        """
        mat = FatigueMaterial(
            name="Ti-6Al-4V",
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
        )

        # 1. Closed-form analytical beam theory
        theory = calculate_cantilever_theory(
            length_L=100.0,
            width_b=10.0,
            height_h=10.0,
            force_max=1000.0,
            force_min=100.0,
            material=mat,
        )

        # Expected analytical hand calculations
        assert math.isclose(theory.sigma_max, 600.0, rel_tol=1e-6)
        assert math.isclose(theory.sigma_min, 60.0, rel_tol=1e-6)
        assert math.isclose(theory.sigma_a, 270.0, rel_tol=1e-6)
        assert math.isclose(theory.sigma_m, 330.0, rel_tol=1e-6)

        # Goodman FS = 1 / (270/510 + 330/860) = 1.095131
        expected_fs_goodman = 1.0 / (270.0 / 510.0 + 330.0 / 860.0)
        assert math.isclose(theory.fs_goodman, expected_fs_goodman, rel_tol=1e-6)

        # Soderberg FS = 1 / (270/510 + 330/795) = 1.05875
        expected_fs_soderberg = 1.0 / (270.0 / 510.0 + 330.0 / 795.0)
        assert math.isclose(theory.fs_soderberg, expected_fs_soderberg, rel_tol=1e-6)

        # Gerber FS
        a_g = (330.0 / 860.0) ** 2
        b_g = 270.0 / 510.0
        expected_fs_gerber = (-b_g + math.sqrt(b_g ** 2 + 4.0 * a_g)) / (2.0 * a_g)
        assert math.isclose(theory.fs_gerber, expected_fs_gerber, rel_tol=1e-6)

        # If CalculiX is available, check FEA execution numerical parity
        ccx_path = shutil.which("ccx") or "/usr/bin/ccx"
        if os.path.isfile(ccx_path) and os.access(ccx_path, os.X_OK):
            bm_res = run_benchmark_case_5_3(mesh_size=2.5, ccx_path=ccx_path)
            assert bm_res.passed is True
            assert math.isclose(bm_res.theory.sigma_max, 600.0, rel_tol=1e-6)
            assert math.isclose(bm_res.theory.fs_goodman, 1.095131, rel_tol=1e-4)
            assert bm_res.max_stress_error_pct <= 3.0
            assert bm_res.max_fs_error_pct <= 3.0

    def test_06_step_5_6_iso14801_benchmark_numerical_parity(self):
        """
        [Step 5.7 Acceptance Criterion: Step 5.6 Parity]
        Verify exact numerical agreement with Step 5.6 ISO 14801 specification & verification rules.
        """
        # 1. Geometric setup validation
        clamping_z = 0.0
        crest_z = 3.0
        loading_pt = np.array([0.0, 0.0, 11.0])
        f_max = np.array([-125.0, 0.0, -216.50635])  # 250N at 30 deg
        f_min = f_max * 0.1

        rep = validate_iso14801_setup(
            clamping_z=clamping_z,
            crest_z=crest_z,
            loading_point=loading_pt,
            force_vector_max=f_max,
            force_vector_min=f_min,
        )
        assert rep.passed is True
        assert math.isclose(rep.measured_angle_deg, 30.0, abs_tol=0.1)
        assert math.isclose(rep.measured_lever_arm_mm, 11.0, abs_tol=0.01)
        assert math.isclose(rep.measured_resorption_depth_mm, 3.0, abs_tol=0.01)

        # 2. Predicted fatigue limit load calculation parity
        f_limit_1 = calculate_predicted_fatigue_limit_load(250.0, 1.25, MeanStressCorrectionMethod.GOODMAN)
        assert math.isclose(f_limit_1, 312.5, rel_tol=1e-6)

        f_limit_2 = calculate_predicted_fatigue_limit_load(300.0, 0.80, MeanStressCorrectionMethod.GOODMAN)
        assert math.isclose(f_limit_2, 240.0, rel_tol=1e-6)

        # 3. Literature benchmark comparison parity
        comp_good = compare_with_literature_benchmark(305.0, "gehrke_2020_conical")
        assert comp_good.status == "GOOD"
        assert math.isclose(comp_good.literature_fatigue_limit_N, 316.0, rel_tol=1e-5)
        assert comp_good.relative_error_pct < 10.0

        # 4. Honest unverified status
        comp_unverified = compare_with_literature_benchmark(280.0, None)
        assert comp_unverified.status == "UNVERIFIED"
        assert "교차 검증 미실시" in comp_unverified.status_display_text

    def test_07_diagram_widget_renders_without_error(self, qapp):
        """
        Verify that FatigueDiagramWidget updates S-N and Haigh plots seamlessly.
        """
        w = FatigueDiagramWidget()
        mat = FatigueMaterial(
            name="Ti-6Al-4V ELI",
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
        )
        load_case = FatigueLoadCase(
            target_cycles=5.0e6,
            stress_ratio_R=0.1,
            mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
        )

        w.update_diagrams(
            material=mat,
            load_case=load_case,
            sigma_a_hotspot=200.0,
            sigma_m_hotspot=240.0,
            life_hotspot=5.5e5,
        )

        # Check that canvas was drawn
        assert w.ax_sn.lines is not None
        assert len(w.ax_sn.lines) > 0
        assert len(w.ax_haigh.lines) > 0

    def test_08_fatigue_report_generator_content(self):
        """
        Verify that HTML and Text reports include mandatory disclaimer and assumptions.
        """
        model, rs = create_dummy_mesh_and_result_set(20)
        mat = FatigueMaterial(
            name="Ti-6Al-4V ELI",
            standard="ASTM F136",
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
            sources=["ASTM F136-13", "MMPDS-01"],
        )
        load_case = FatigueLoadCase(
            target_cycles=5.0e6,
            stress_ratio_R=0.1,
            mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
            stress_measure=StressMeasureType.SIGNED_VON_MISES,
        )
        rs.fatigue_fs = np.full(20, 1.25)

        hotspot_info = {
            "part_name": "fixture",
            "node_id": 10,
            "coordinates": [0.0, 5.0, 10.0],
            "min_safety_factor": 1.25,
            "max_stress_MPa": 320.0,
            "hotspot_region_name": "임플란트 고정체 목부",
        }

        # 1. HTML Report
        html = generate_fatigue_report_html(
            rs=rs,
            model=model,
            material=mat,
            load_case=load_case,
            hotspot_info=hotspot_info,
        )
        assert DISCLAIMER_TEXT in html
        assert "임상 판단 보조용, 시험 대체 불가" in html
        assert "ASTM F136" in html
        assert "Ti-6Al-4V ELI" in html
        assert "1.2500" in html
        assert "임플란트 고정체 목부" in html
        assert "GOODMAN" in html

        # 2. Text Report
        txt = generate_fatigue_report_text(
            rs=rs,
            model=model,
            material=mat,
            load_case=load_case,
            hotspot_info=hotspot_info,
        )
        assert DISCLAIMER_TEXT in txt
        assert "Min Fatigue Safety Factor (FS): 1.2500" in txt
        assert "Hotspot Region: 임플란트 고정체 목부" in txt

    def test_09_hotspot_marker_toggle_and_clear(self, qapp):
        """
        Verify that hotspot marker button toggles between display and clear,
        emitting the corresponding signals and updating button text.
        """
        sm = StateManager()
        model, rs = create_dummy_mesh_and_result_set(30)
        sm.set_mesh_model(model)
        sm.set_result_set(rs)

        panel = FatiguePanel(sm)
        panel._last_hotspot = {
            "part_name": "fixture",
            "node_id": 5,
            "coordinates": [1.0, 2.0, 3.0],
            "min_safety_factor": 1.45,
            "max_stress_MPa": 200.0,
            "hotspot_region_name": "Fixture Neck",
        }
        panel.btn_mark_hotspot.setEnabled(True)

        marked_signals = []
        clear_signals = []
        panel.hotspot_marked_requested.connect(lambda d: marked_signals.append(d))
        panel.hotspot_clear_requested.connect(lambda: clear_signals.append(True))

        # 1. First click: Display marker
        panel._on_mark_hotspot_clicked()
        assert len(marked_signals) == 1
        assert marked_signals[0]["node_id"] == 5
        assert len(clear_signals) == 0
        assert panel._hotspot_marked is True
        assert "해제" in panel.btn_mark_hotspot.text()

        # 2. Second click: Clear marker
        panel._on_mark_hotspot_clicked()
        assert len(marked_signals) == 1
        assert len(clear_signals) == 1
        assert panel._hotspot_marked is False
        assert "표시" in panel.btn_mark_hotspot.text()

        # 3. Model stale triggers automatic clear
        panel._on_mark_hotspot_clicked()
        assert panel._hotspot_marked is True
        panel._on_fea_stale_changed(True)
        assert panel._hotspot_marked is False
        assert len(clear_signals) == 2
        assert panel.btn_mark_hotspot.isEnabled() is False

    def test_10_fatigue_field_button_auto_eval_and_dispatch(self, qapp):
        """
        Verify that clicking a fatigue field button automatically executes evaluation
        if fatigue fields are not yet calculated, then emits field_changed_requested.
        """
        sm = StateManager()
        model, rs = create_dummy_mesh_and_result_set(50)
        sm.set_mesh_model(model)
        sm.set_result_set(rs)

        panel = FatiguePanel(sm)
        assert sm.result_set.fatigue_fs is None

        dispatched_fields = []
        panel.field_changed_requested.connect(lambda f: dispatched_fields.append(f))

        # Request field when not evaluated yet -> auto-evaluates and dispatches
        panel._request_fatigue_field("fatigue_fs")

        assert sm.result_set.fatigue_fs is not None
        assert len(sm.result_set.fatigue_fs) == 50
        assert "fatigue_fs" in dispatched_fields

    def test_11_all_5_fatigue_fields_dispatch(self, qapp):
        """
        Verify that all 5 fatigue scalar buttons emit correct field names.
        """
        sm = StateManager()
        model, rs = create_dummy_mesh_and_result_set(30)
        sm.set_mesh_model(model)
        sm.set_result_set(rs)

        panel = FatiguePanel(sm)
        # Pre-evaluate
        panel.run_fatigue_evaluation(blocking=True)

        emitted = []
        panel.field_changed_requested.connect(lambda f: emitted.append(f))

        panel._request_fatigue_field("fatigue_fs")
        panel._request_fatigue_field("fatigue_life")
        panel._request_fatigue_field("fatigue_damage")
        panel._request_fatigue_field("fatigue_sigma_a")
        panel._request_fatigue_field("fatigue_sigma_m")

        assert emitted[-5:] == [
            "fatigue_fs",
            "fatigue_life",
            "fatigue_damage",
            "fatigue_sigma_a",
            "fatigue_sigma_m",
        ]

