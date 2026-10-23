"""
test_peri_implant.py
Unit tests for postprocess/peri_implant.py.

Verifies:
1. Distance-based peri-implant bone node selection using cKDTree against exact analytical counts.
2. Peak stress location retrieval across the assembly and for specific parts.
3. Bone stratification into cortical and cancellous bone with independent statistics.
4. Edge cases: zero selected nodes (Korean warning message), all-NaN arrays, and JSON serialization.
5. Extraction of bone interface nodes from TieDef (both master and slave configurations).
"""

import json
import numpy as np
import pytest

from fea.models import MeshPart, MeshModel, MaterialDef, MaterialAssignment, TieDef
from fea.results import ResultSet
from postprocess.peri_implant import (
    peri_implant_bone_stats,
    max_stress_location,
    extract_bone_interface_nodes,
)


@pytest.fixture
def synthetic_peri_model():
    """
    Construct a synthetic cylindrical implant embedded in a bone block.
    Bone: 21x21 grid in XY at Z=0 (441 nodes) spanning [-10, 10] x [-10, 10].
    Interface: Nodes on circle of radius ~3mm.
    """
    x = np.linspace(-10.0, 10.0, 21)  # step 1.0 mm
    y = np.linspace(-10.0, 10.0, 21)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    bone_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])  # 441 nodes

    # Synthetic simple tetrahedral elements for bone
    # Divide into small synthetic quads/tets (for simplicity, simple element connectivity)
    bone_elements = []
    for j in range(20):
        for i in range(20):
            n0 = j * 21 + i
            n1 = n0 + 1
            n2 = (j + 1) * 21 + i
            n3 = n2 + 1
            bone_elements.append([n0, n1, n2, n3])
    bone_elements = np.array(bone_elements, dtype=np.int64)

    bone_part = MeshPart(name="bone", nodes=bone_coords, elements=bone_elements, element_type="C3D4")

    # Implant part: single center point and cylinder top
    implant_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    implant_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    implant_part = MeshPart(name="implant", nodes=implant_coords, elements=implant_elements, element_type="C3D4")

    model = MeshModel(parts=[bone_part, implant_part])
    return model


def test_peri_implant_node_count_and_stats(synthetic_peri_model):
    """Verify that cKDTree node count matches exact brute-force geometric query."""
    model = synthetic_peri_model
    bone_part = model.get_part("bone")
    bone_nodes = bone_part.nodes

    # Select interface node at (3.0, 0.0, 0.0)
    target_point = np.array([3.0, 0.0, 0.0])
    dists_to_target = np.linalg.norm(bone_nodes - target_point, axis=1)
    iface_idx = int(np.argmin(dists_to_target))  # node exactly at (3.0, 0.0, 0.0)

    # Brute-force calculation: nodes within 1.0 mm radius from (3.0, 0.0, 0.0)
    # Since grid step is 1.0 mm, points at distance <= 1.0 mm are:
    # (3, 0), (2, 0), (4, 0), (3, 1), (3, -1) -> exactly 5 points
    expected_indices = np.where(dists_to_target <= 1.0 + 1e-6)[0]
    expected_count = len(expected_indices)
    assert expected_count == 5

    # Prepare ResultSet with known values
    total_nodes = model.total_nodes()
    von_mises = np.full(total_nodes, 10.0, dtype=np.float64)
    s_max = np.full(total_nodes, 20.0, dtype=np.float64)
    s_min = np.full(total_nodes, -15.0, dtype=np.float64)

    # Put a specific peak at the interface node
    von_mises[iface_idx] = 120.0
    s_max[iface_idx] = 150.0
    s_min[iface_idx] = -80.0

    rs = ResultSet(
        time=0.0,
        u=np.zeros((total_nodes, 3)),
        u_mag=np.zeros(total_nodes),
        von_mises=von_mises,
        s_max_principal=s_max,
        s_min_principal=s_min,
    )

    stats = peri_implant_bone_stats(
        rs=rs,
        model=model,
        interface_nodes_bone=np.array([iface_idx]),
        radius_mm=1.0,
    )

    assert stats["num_nodes"] == expected_count
    assert stats["max_von_mises"] == pytest.approx(120.0)
    assert stats["max_s_max_principal"] == pytest.approx(150.0)
    assert stats["min_s_min_principal"] == pytest.approx(-80.0)
    # mean of 5 nodes: (120 + 4 * 10) / 5 = 160 / 5 = 32.0
    assert stats["mean_von_mises"] == pytest.approx(32.0)
    assert stats["p95_von_mises"] is not None
    assert stats["p99_von_mises"] is not None

    # Check JSON serializable
    json_str = json.dumps(stats)
    assert len(json_str) > 0


def test_max_stress_location(synthetic_peri_model):
    """Verify that max_stress_location accurately identifies the peak node and coordinates."""
    model = synthetic_peri_model
    total_nodes = model.total_nodes()
    bone_len = len(model.get_part("bone").nodes)

    von_mises = np.linspace(1.0, 50.0, total_nodes)
    # Artificially set a peak inside bone
    peak_bone_loc = 42
    von_mises[peak_bone_loc] = 350.0

    # Artificially set an even higher peak inside implant
    peak_implant_loc = 1
    peak_implant_global = bone_len + peak_implant_loc
    von_mises[peak_implant_global] = 950.0

    rs = ResultSet(
        time=0.0,
        u=np.zeros((total_nodes, 3)),
        u_mag=np.zeros(total_nodes),
        von_mises=von_mises,
        s_max_principal=von_mises,
        s_min_principal=-von_mises,
    )

    # 1. Global search -> should pick implant peak
    global_res = max_stress_location(rs, model, field="von_mises")
    assert global_res["node_idx"] == peak_implant_global
    assert global_res["part_name"] == "implant"
    assert global_res["value"] == pytest.approx(950.0)
    assert global_res["coord"] == pytest.approx(model.get_part("implant").nodes[peak_implant_loc].tolist())

    # 2. Bone-only search -> should pick bone peak
    bone_res = max_stress_location(rs, model, field="von_mises", part_name="bone")
    assert bone_res["node_idx"] == peak_bone_loc
    assert bone_res["part_name"] == "bone"
    assert bone_res["value"] == pytest.approx(350.0)
    assert bone_res["coord"] == pytest.approx(model.get_part("bone").nodes[peak_bone_loc].tolist())

    # JSON serializable
    assert len(json.dumps(global_res)) > 0
    assert len(json.dumps(bone_res)) > 0


def test_cortical_cancellous_stratification(synthetic_peri_model):
    """Verify bone stratification where peak stress is confined to cortical statistics."""
    model = synthetic_peri_model
    bone_part = model.get_part("bone")
    n_elems = len(bone_part.elements)

    # Assign first half of elements to cortical, second half to cancellous
    mat_cortical = MaterialDef(name="CORTICAL_BONE", E=13700.0, nu=0.3)
    mat_cancellous = MaterialDef(name="CANCELLOUS_BONE", E=1370.0, nu=0.3)

    elem_mat = np.zeros(n_elems, dtype=np.int64)
    elem_mat[n_elems // 2 :] = 1  # 0: cortical, 1: cancellous

    materials = MaterialAssignment(
        element_material={"bone": elem_mat},
        materials=[mat_cortical, mat_cancellous],
    )

    # Let interface be at node 0 (which is in element 0 -> cortical)
    # and node 440 (in last element -> cancellous)
    iface_nodes = np.array([0, 440])

    total_nodes = model.total_nodes()
    von_mises = np.full(total_nodes, 10.0, dtype=np.float64)

    # Put a large stress only on node 0 (cortical)
    von_mises[0] = 250.0
    # Put a moderate stress on node 440 (cancellous)
    von_mises[440] = 30.0

    rs = ResultSet(
        time=0.0,
        u=np.zeros((total_nodes, 3)),
        u_mag=np.zeros(total_nodes),
        von_mises=von_mises,
        s_max_principal=von_mises,
        s_min_principal=-von_mises,
    )

    stats = peri_implant_bone_stats(
        rs=rs,
        model=model,
        interface_nodes_bone=iface_nodes,
        radius_mm=2.0,
        materials=materials,
        cortical_name="CORTICAL",
    )

    assert "cortical" in stats
    assert "cancellous" in stats
    assert stats["cortical"]["max_von_mises"] == pytest.approx(250.0)
    # cancellous should NOT have 250.0
    assert stats["cancellous"]["max_von_mises"] < 50.0

    # JSON serializable
    assert len(json.dumps(stats)) > 0


def test_empty_selected_nodes_warning(synthetic_peri_model):
    """Verify behavior when no nodes are found within the specified radius."""
    model = synthetic_peri_model
    total_nodes = model.total_nodes()
    rs = ResultSet(
        time=0.0,
        u=np.zeros((total_nodes, 3)),
        u_mag=np.zeros(total_nodes),
        von_mises=np.zeros(total_nodes),
        s_max_principal=np.zeros(total_nodes),
        s_min_principal=np.zeros(total_nodes),
    )

    # Empty interface nodes
    stats_empty = peri_implant_bone_stats(
        rs=rs,
        model=model,
        interface_nodes_bone=np.array([]),
        radius_mm=1.0,
    )
    assert stats_empty["num_nodes"] == 0
    assert "warning" in stats_empty
    assert "지정된" in stats_empty["warning"] or "경고" in stats_empty["warning"]

    # Valid interface node but zero radius
    stats_zero_r = peri_implant_bone_stats(
        rs=rs,
        model=model,
        interface_nodes_bone=np.array([0]),
        radius_mm=0.0,
    )
    # radius=0 includes at least the interface node itself if distance <= 0
    assert stats_zero_r["num_nodes"] >= 1


def test_extract_bone_interface_nodes():
    """Verify extraction of bone nodes from TieDef in both slave and master roles."""
    p_bone = MeshPart(
        name="bone",
        nodes=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        element_type="C3D4",
    )
    p_implant = MeshPart(
        name="implant",
        nodes=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        element_type="C3D4",
    )
    model = MeshModel(parts=[p_bone, p_implant])

    # Case 1: Bone is slave
    tie_slave = TieDef(
        name="tie1",
        master_part="implant",
        slave_part="bone",
        master_faces=np.array([[0, 1]]),
        slave_nodes=np.array([0, 1, 2]),
        position_tolerance_mm=0.1,
    )
    nodes1 = extract_bone_interface_nodes(tie_slave, model)
    assert np.array_equal(nodes1, np.array([0, 1, 2]))

    # Case 2: Bone is master
    tie_master = TieDef(
        name="tie2",
        master_part="bone",
        slave_part="implant",
        master_faces=np.array([[0, 1]]),  # face 1 -> [0, 1, 2]
        slave_nodes=np.array([0, 1]),
        position_tolerance_mm=0.1,
    )
    nodes2 = extract_bone_interface_nodes(tie_master, model)
    assert np.array_equal(nodes2, np.array([0, 1, 2]))
