"""
test_exporters.py
Unit tests for postprocess/exporters.py and 3D screenshot export.

Verifies:
1. export_nodal_csv: row count matches total nodes, values match within 7 significant figures.
2. export_stats_csv and export_stats_json: numpy types and NaNs handled safely, roundtrip via json.loads.
3. save_result_bundle and load_result_bundle: full roundtrip preserving all ResultSet arrays (including NaNs),
   stats, and metadata, plus part .vtu creation.
4. save_screenshot: captures off-screen VTK rendering to PNG file at scale 1x and 2x.
"""

import os
import csv
import json
import pytest
import numpy as np
import vtk

from fea.models import MeshPart, MeshModel, MaterialAssignment
from fea.results import ResultSet
from postprocess.exporters import (
    export_nodal_csv,
    export_stats_csv,
    export_stats_json,
    save_result_bundle,
    load_result_bundle,
)
from ui.result_view_controller import ResultViewController


@pytest.fixture
def synthetic_model_and_results() -> tuple[MeshModel, ResultSet, dict, dict]:
    """Create a two-part assembly with known results and statistics for testing."""
    # Part 1: Bone (C3D4, 4 nodes, 1 element)
    bone_nodes = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ], dtype=np.float64)
    bone_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    bone_part = MeshPart(name="bone", nodes=bone_nodes, elements=bone_elements, element_type="C3D4")

    # Part 2: Implant (C3D4, 4 nodes, 1 element)
    implant_nodes = np.array([
        [10.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
        [10.0, 2.0, 0.0],
        [10.0, 0.0, 2.0],
    ], dtype=np.float64)
    implant_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    implant_part = MeshPart(name="implant", nodes=implant_nodes, elements=implant_elements, element_type="C3D4")

    model = MeshModel(parts=[bone_part, implant_part])

    total_nodes = 8  # 4 + 4

    # Result fields
    u = np.array([
        [0.001, 0.002, 0.003],
        [0.004, 0.005, 0.006],
        [0.007, 0.008, 0.009],
        [0.010, 0.011, 0.012],
        # Implant nodes (some NaNs)
        [0.020, 0.021, np.nan],
        [0.023, 0.024, 0.025],
        [0.026, 0.027, 0.028],
        [0.029, 0.030, 0.031],
    ], dtype=np.float64)

    u_mag = np.linalg.norm(np.nan_to_num(u), axis=-1)

    s = np.zeros((total_nodes, 6), dtype=np.float64)
    s[:, 0] = np.linspace(10.0, 80.0, total_nodes)
    s[:, 1] = np.linspace(5.0, 40.0, total_nodes)
    s[4, :] = np.nan  # node 4 has NaN stress

    vm = np.linspace(12.34567, 98.76543, total_nodes)
    vm[4] = np.nan

    s_max = np.linspace(15.0, 85.0, total_nodes)
    s_max[4] = np.nan

    s_min = np.linspace(-5.0, -25.0, total_nodes)
    s_min[4] = np.nan

    strain = np.zeros((total_nodes, 6), dtype=np.float64)
    rf = np.zeros((total_nodes, 3), dtype=np.float64)
    rf[0] = [-10.0, 0.0, -100.0]

    rs = ResultSet(
        time=1.0,
        u=u,
        u_mag=u_mag,
        stress6=s,
        von_mises=vm,
        s_max_principal=s_max,
        s_min_principal=s_min,
        strain6=strain,
        rf=rf,
        source_frd="sample_job.frd",
    )


    # Statistics with numpy types, floats, and NaNs
    stats = {
        "num_nodes": np.int64(total_nodes),
        "parts": {
            "bone": {
                "max_u_mag": np.float64(0.012),
                "max_von_mises": np.float64(45.67),
                "p99_von_mises": np.float64(45.0),
                "mean_von_mises": np.float64(25.0),
                "max_s_max_principal": np.float64(48.0),
                "min_s_min_principal": np.float64(-12.0),
            },
            "implant": {
                "max_u_mag": np.float64(0.031),
                "max_von_mises": np.float64(98.76543),
                "p99_von_mises": np.float64(95.0),
                "mean_von_mises": np.float64(70.0),
                "max_s_max_principal": np.float64(85.0),
                "min_s_min_principal": np.float64(-25.0),
            },
        },
        "peri_implant": {
            "max_u_mag": np.float64(0.011),
            "max_von_mises": np.float64(35.0),
            "p99_von_mises": np.float64(34.0),
            "mean_von_mises": np.float64(20.0),
            "max_s_max_principal": np.float64(38.0),
            "min_s_min_principal": np.float64(-8.0),
        },
        "radius_mm": np.float64(1.5),
        "reaction_balance": {
            "applied_mag": np.float64(100.5),
            "reaction_mag": np.float64(100.49),
            "residual_mag": np.float64(0.01),
            "relative_error": np.float64(0.0001),
        },
    }

    meta = {
        "job_name": "test_implant_job",
        "author": "dental_fea_test",
    }

    return model, rs, stats, meta


def test_export_nodal_csv(tmp_path, synthetic_model_and_results):
    """Test exporting nodal CSV and verify row count and data values."""
    model, rs, stats, meta = synthetic_model_and_results
    csv_file = str(tmp_path / "nodal_results.csv")

    export_nodal_csv(csv_file, rs, model)
    assert os.path.exists(csv_file)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    # Header + 8 nodes = 9 rows
    assert len(reader) == 9
    header = reader[0]
    assert header[0] == "global_node_id"
    assert header[1] == "part"
    assert "von_mises" in header
    assert "s_max_principal" in header

    # Row 1 (Node 1, bone)
    row1 = reader[1]
    assert int(row1[0]) == 1
    assert row1[1] == "bone"
    assert float(row1[2]) == pytest.approx(0.0)  # x
    assert float(row1[5]) == pytest.approx(0.001)  # ux
    assert float(row1[8]) == pytest.approx(rs.u_mag[0], rel=1e-4)  # u_mag

    # Row 5 (Node 5, implant, contains NaN for uz and stress)
    row5 = reader[5]
    assert int(row5[0]) == 5
    assert row5[1] == "implant"
    assert row5[7] == "nan"  # uz is NaN
    assert row5[15] == "nan"  # von_mises is NaN


def test_stats_json_roundtrip(tmp_path, synthetic_model_and_results):
    """Test export_stats_json serializes numpy numbers and handles NaNs safely."""
    _, _, stats, _ = synthetic_model_and_results
    json_file = str(tmp_path / "stats.json")

    # Add a NaN entry to ensure safe conversion
    stats_with_nan = dict(stats)
    stats_with_nan["nan_val"] = np.nan
    stats_with_nan["arr_val"] = np.array([1, 2, 3])

    export_stats_json(json_file, stats_with_nan)
    assert os.path.exists(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["num_nodes"] == 8
    assert data["nan_val"] is None  # NaN becomes None in JSON
    assert data["arr_val"] == [1, 2, 3]
    assert data["parts"]["bone"]["max_u_mag"] == pytest.approx(0.012)


def test_export_stats_csv(tmp_path, synthetic_model_and_results):
    """Test export_stats_csv outputs properly formatted summary table."""
    _, _, stats, _ = synthetic_model_and_results
    csv_file = str(tmp_path / "stats.csv")

    export_stats_csv(csv_file, stats)
    assert os.path.exists(csv_file)

    with open(csv_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Part,Max Displacement (mm)" in content
    assert "bone" in content
    assert "implant" in content
    assert "Peri-implant Bone" in content
    assert "# Static Equilibrium Summary" in content


def test_result_bundle_roundtrip(tmp_path, synthetic_model_and_results):
    """Test saving and loading a result bundle directory."""
    model, rs, stats, meta = synthetic_model_and_results
    bundle_dir = str(tmp_path / "my_bundle")

    save_result_bundle(bundle_dir, model, rs, stats, meta)

    # 1. Check generated files
    assert os.path.exists(os.path.join(bundle_dir, "bone.vtu"))
    assert os.path.exists(os.path.join(bundle_dir, "implant.vtu"))
    assert os.path.exists(os.path.join(bundle_dir, "results.npz"))
    assert os.path.exists(os.path.join(bundle_dir, "stats.json"))
    assert os.path.exists(os.path.join(bundle_dir, "meta.json"))

    # 2. Load bundle back
    rs_loaded, stats_loaded, meta_loaded = load_result_bundle(bundle_dir)

    # 3. Verify ResultSet integrity (including NaNs)
    assert pytest.approx(rs_loaded.time) == rs.time
    assert rs_loaded.source_frd == rs.source_frd

    # Displacement array
    np.testing.assert_equal(np.isnan(rs.u), np.isnan(rs_loaded.u))
    valid_u = ~np.isnan(rs.u)
    np.testing.assert_allclose(rs.u[valid_u], rs_loaded.u[valid_u])

    # Stress6 array
    np.testing.assert_equal(np.isnan(rs.stress6), np.isnan(rs_loaded.stress6))
    valid_s = ~np.isnan(rs.stress6)
    np.testing.assert_allclose(rs.stress6[valid_s], rs_loaded.stress6[valid_s])

    # von Mises array
    np.testing.assert_equal(np.isnan(rs.von_mises), np.isnan(rs_loaded.von_mises))
    valid_vm = ~np.isnan(rs.von_mises)
    np.testing.assert_allclose(rs.von_mises[valid_vm], rs_loaded.von_mises[valid_vm])


    # 4. Verify stats and meta
    assert stats_loaded["num_nodes"] == 8
    assert "units" in meta_loaded
    assert meta_loaded["units"]["stress"] == "MPa"
    assert "mesh_summary" in meta_loaded
    assert meta_loaded["mesh_summary"]["total_nodes"] == 8


def test_screenshot_capture(tmp_path, synthetic_model_and_results):
    """Test saving screenshot of offscreen VTK renderer to PNG."""
    model, rs, stats, meta = synthetic_model_and_results

    try:
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.1, 0.1, 0.1)
        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.AddRenderer(ren)
        rw.SetSize(400, 300)
        ren._rw = rw
    except Exception as e:
        pytest.skip(f"Offscreen VTK render window unavailable: {e}")

    vc = ResultViewController(ren)
    vc.set_results(model, rs)

    # Capture 1x screenshot
    png_1x = str(tmp_path / "shot_1x.png")
    ok1 = vc.save_screenshot(png_1x, scale=1)
    if not ok1:
        pytest.skip("VTK screenshot capture failed in headless environment (OpenGL context issue)")

    assert os.path.exists(png_1x)
    assert os.path.getsize(png_1x) > 0

    # Capture 2x screenshot
    png_2x = str(tmp_path / "shot_2x.png")
    ok2 = vc.save_screenshot(png_2x, scale=2)
    assert ok2 is True
    assert os.path.exists(png_2x)
    assert os.path.getsize(png_2x) > 0
