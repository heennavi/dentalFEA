"""
test_io_frd.py
Unit tests for CalculiX .frd fatigue stress extraction module.
"""

from pathlib import Path
import numpy as np
import pytest

from fea.results import FrdBlock, FrdData
from fatigue.io_frd import (
    extract_stress_from_frd,
    extract_cyclic_stresses_from_frd,
    extract_cyclic_stresses_from_two_frds,
    extract_cyclic_stresses_with_ratio,
    _align_nodal_data,
)


@pytest.fixture
def cube_frd_path():
    """Path to existing cube_compression.frd test fixture."""
    p = Path(__file__).parent.parent / "fixtures" / "cube_compression.frd"
    return p


@pytest.fixture
def synthetic_multistep_frd():
    """Create in-memory synthetic FrdData with two stress steps."""
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    coords = np.zeros((4, 3), dtype=np.float64)

    # Step 1 (Min load, time 0.1)
    s_min = np.array([
        [10.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [20.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        [30.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        [40.0, 0.0, 0.0, 4.0, 0.0, 0.0],
    ], dtype=np.float64)

    b1 = FrdBlock(
        name="STRESS",
        time=0.1,
        step=0,
        component_names=["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"],
        node_ids=node_ids,
        values=s_min,
    )

    # Step 2 (Max load, time 1.0)
    s_max = np.array([
        [100.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        [200.0, 0.0, 0.0, 20.0, 0.0, 0.0],
        [300.0, 0.0, 0.0, 30.0, 0.0, 0.0],
        [400.0, 0.0, 0.0, 40.0, 0.0, 0.0],
    ], dtype=np.float64)

    b2 = FrdBlock(
        name="STRESS",
        time=1.0,
        step=1,
        component_names=["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"],
        node_ids=node_ids,
        values=s_max,
    )

    return FrdData(
        node_ids=node_ids,
        node_coords=coords,
        blocks=[b1, b2],
    )


def test_extract_stress_from_real_fixture(cube_frd_path):
    """Verify stress extraction on existing CalculiX cube_compression.frd."""
    if not cube_frd_path.is_file():
        pytest.skip(f"Fixture not found: {cube_frd_path}")

    node_ids, stress6 = extract_stress_from_frd(cube_frd_path, which="last")

    assert len(node_ids) > 0
    assert stress6.ndim == 2
    assert stress6.shape[1] == 6
    assert stress6.shape[0] == len(node_ids)


def test_extract_cyclic_stresses_from_frd(synthetic_multistep_frd):
    """Verify extracting two steps from a single multistep FrdData."""
    node_ids, s_max, s_min = extract_cyclic_stresses_from_frd(
        synthetic_multistep_frd,
        step_max="last",
        step_min="first",
    )

    assert np.array_equal(node_ids, [1, 2, 3, 4])
    assert s_max[0, 0] == pytest.approx(100.0)
    assert s_max[3, 0] == pytest.approx(400.0)
    assert s_min[0, 0] == pytest.approx(10.0)
    assert s_min[3, 0] == pytest.approx(40.0)


def test_extract_cyclic_stresses_with_ratio(synthetic_multistep_frd):
    """Verify extracting max step and scaling by stress_ratio_R."""
    node_ids, s_max, s_min = extract_cyclic_stresses_with_ratio(
        synthetic_multistep_frd,
        stress_ratio_R=0.25,
        which="last",
    )

    assert np.array_equal(node_ids, [1, 2, 3, 4])
    assert s_max[1, 0] == pytest.approx(200.0)
    assert s_min[1, 0] == pytest.approx(50.0)  # 200 * 0.25


def test_extract_cyclic_stresses_from_two_frds(synthetic_multistep_frd):
    """Verify extracting and combining two independent FrdData instances."""
    data_max = FrdData(
        node_ids=synthetic_multistep_frd.node_ids,
        node_coords=synthetic_multistep_frd.node_coords,
        blocks=[synthetic_multistep_frd.blocks[1]],
    )
    data_min = FrdData(
        node_ids=synthetic_multistep_frd.node_ids,
        node_coords=synthetic_multistep_frd.node_coords,
        blocks=[synthetic_multistep_frd.blocks[0]],
    )

    node_ids, s_max, s_min = extract_cyclic_stresses_from_two_frds(data_max, data_min)

    assert np.array_equal(node_ids, [1, 2, 3, 4])
    assert s_max[2, 0] == pytest.approx(300.0)
    assert s_min[2, 0] == pytest.approx(30.0)


def test_align_nodal_data_scrambled_and_subset():
    """Verify _align_nodal_data handles mismatched node ordering and subsets."""
    nodes1 = np.array([30, 10, 20], dtype=np.int64)
    vals1 = np.array([[3.0], [1.0], [2.0]], dtype=np.float64)

    nodes2 = np.array([20, 40, 10], dtype=np.int64)
    vals2 = np.array([[20.0], [40.0], [10.0]], dtype=np.float64)

    # Common nodes are [10, 20]
    c_nodes, c_vals1, c_vals2 = _align_nodal_data(nodes1, vals1, nodes2, vals2)

    assert np.array_equal(c_nodes, [10, 20])
    assert c_vals1[0, 0] == pytest.approx(1.0)
    assert c_vals1[1, 0] == pytest.approx(2.0)
    assert c_vals2[0, 0] == pytest.approx(10.0)
    assert c_vals2[1, 0] == pytest.approx(20.0)


def test_error_handling_missing_stress(synthetic_multistep_frd):
    """Verify KeyError when STRESS block is missing."""
    empty_data = FrdData(
        node_ids=synthetic_multistep_frd.node_ids,
        node_coords=synthetic_multistep_frd.node_coords,
        blocks=[],
    )
    with pytest.raises(KeyError, match="No STRESS block found"):
        extract_stress_from_frd(empty_data)

    with pytest.raises(FileNotFoundError):
        extract_stress_from_frd("/non/existent/file.frd")
