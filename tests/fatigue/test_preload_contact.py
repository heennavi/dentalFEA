"""
test_preload_contact.py
Comprehensive automated test suite for Step 5.5: Abutment screw preload (*PRE-TENSION SECTION)
and contact analysis (*CONTACT PAIR).
"""

import os
import pytest
import numpy as np

from fatigue.preload_contact import (
    PreloadContactConfig,
    PreloadContactResult,
    SimplifiedModelResult,
    ModelComparisonReport,
    build_benchmark_assembly,
    write_preload_contact_deck,
    run_preload_contact_analysis,
    run_simplified_model_analysis,
    compare_simplified_vs_detailed,
)


def test_assembly_mesh_generation():
    """Verify 3D dental assembly mesh generation and metadata."""
    model, meta = build_benchmark_assembly(
        implant_radius=2.0,
        implant_height=6.0,
        abutment_radius=2.0,
        abutment_height=6.0,
        screw_radius=0.6,
        screw_length=8.0,
    )

    assert len(model.parts) == 3
    assert set(p.name for p in model.parts) == {"implant", "abutment", "screw"}

    imp = model.get_part("implant")
    abt = model.get_part("abutment")
    sc = model.get_part("screw")

    assert imp is not None and abt is not None and sc is not None
    assert imp.element_type == "C3D8"
    assert abt.element_type == "C3D8"
    assert sc.element_type == "C3D8"

    assert len(imp.nodes) == 36
    assert len(abt.nodes) == 36
    assert len(sc.nodes) == 16

    assert len(imp.elements) == 12
    assert len(abt.elements) == 12
    assert len(sc.elements) == 3

    assert meta["interface_z"] == 6.0
    assert meta["ref_node_id"] == 9999


def test_deck_generation(tmp_path):
    """Verify CalculiX .inp deck contains required PRE-TENSION and CONTACT keywords."""
    _, meta = build_benchmark_assembly()
    config = PreloadContactConfig(
        preload_force_N=200.0,
        friction_coeff=0.15,
        penalty_stiffness=250000.0,
        contact_type="MORTAR",
    )
    inp_path = str(tmp_path / "test_deck.inp")
    write_preload_contact_deck(inp_path, meta, config)

    assert os.path.exists(inp_path)
    with open(inp_path, "r", encoding="ascii") as f:
        content = f.read()

    # Pre-tension verification
    assert "*PRE-TENSION SECTION, SURFACE=SURF_PRELOAD, NODE=9999" in content
    assert "9999, 1, 200.000000" in content
    assert "*BOUNDARY, FIXED" in content
    assert "9999, 1, 1" in content

    # Contact verification
    assert "*SURFACE, NAME=SURF_IMP_TOP, TYPE=ELEMENT" in content
    assert "*SURFACE, NAME=SURF_ABUT_BOT, TYPE=ELEMENT" in content
    assert "*CONTACT PAIR, INTERACTION=INT_PROP, TYPE=SURFACE TO SURFACE, MORTAR" in content
    assert "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR" in content
    assert "250000.0" in content
    assert "*FRICTION" in content
    assert "0.1500" in content


def test_pretension_force_accuracy(tmp_path):
    """Verify that screw axial preload is within +/- 5% of target (Acceptance Criterion 1)."""
    target_force = 200.0
    config = PreloadContactConfig(
        preload_force_N=target_force,
        tolerance_pct=5.0,
        friction_coeff=0.0,
    )
    res = run_preload_contact_analysis(config, workdir=str(tmp_path / "pretension_acc"))

    assert res.step1_converged, "Step 1 (Preload tightening) failed to converge."
    assert res.preload_passed, f"Preload error {res.preload_error_pct:.2f}% exceeds 5%."
    assert abs(res.measured_preload_N - target_force) / target_force <= 0.05
    assert res.preload_error_pct <= 5.0


def test_contact_convergence_and_gap_opening(tmp_path):
    """Verify contact convergence and physical gap opening under service load (Acceptance Criterion 2)."""
    config = PreloadContactConfig(
        preload_force_N=200.0,
        external_force_N=100.0,
        external_angle_deg=30.0,
        external_azimuth_deg=0.0, # Load directed towards +x
        friction_coeff=0.0,
    )
    res = run_preload_contact_analysis(config, workdir=str(tmp_path / "gap_test"))

    assert res.step1_converged, "Step 1 failed to converge."
    assert res.step2_converged, "Step 2 failed to converge."

    # In Step 2 with +x transverse load:
    # Tensile side (x < 0) opens (gap > 0 mm)
    # Compressive side (x > 0) penetrates slightly / tight contact (gap < 0 mm)
    assert res.gap_opening_occurred, f"Tensile gap did not open (gap = {res.tensile_gap_mm * 1e3:.3f} um)."
    assert res.tensile_gap_mm > 0.0
    assert res.compressive_gap_mm < 0.0
    assert res.tensile_gap_mm > res.compressive_gap_mm

    # Check bolt load fraction: external bending load should increase bolt tension in Step 2
    assert res.step2_bolt_force_N > res.measured_preload_N
    assert res.bolt_load_fraction_pct > 0.0


def test_frictional_contact_convergence(tmp_path):
    """Verify convergence with progressive friction (mu = 0.15) (Acceptance Criterion 2)."""
    config = PreloadContactConfig(
        preload_force_N=200.0,
        external_force_N=100.0,
        external_angle_deg=30.0,
        friction_coeff=0.15,
    )
    res = run_preload_contact_analysis(config, workdir=str(tmp_path / "friction_test"))

    assert res.step1_converged, "Step 1 with friction failed to converge."
    assert res.step2_converged, "Step 2 with friction failed to converge."
    assert res.preload_passed
    assert res.gap_opening_occurred


def test_simplified_vs_detailed_comparison(tmp_path):
    """Verify quantitative differences between simplified and detailed models (Acceptance Criterion 3)."""
    config = PreloadContactConfig(
        preload_force_N=200.0,
        external_force_N=100.0,
        external_angle_deg=30.0,
    )
    report = compare_simplified_vs_detailed(config, workdir=str(tmp_path / "compare_test"))

    assert report.detailed.step2_converged
    assert report.simplified.converged

    # Simplified model has 0.0 gap due to bonded/TIE interface
    assert report.simplified.tensile_gap_mm == 0.0
    # Detailed model opens under bending moment
    assert report.gap_opening_diff_um > 0.0

    # Detailed model captures screw stress which does not exist in simplified model
    assert report.detailed.max_von_mises_screw_MPa > 0.0
    assert len(report.clinical_risk_summary) > 50
    assert "마이크로 갭" in report.clinical_risk_summary
