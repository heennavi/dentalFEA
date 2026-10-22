"""
test_results_models.py
Unit tests for FEA result dataclasses: FrdBlock, FrdData, and ResultSet.
"""

import pytest
import numpy as np

from fea.models import MeshModel, MeshPart
from fea.results import FrdBlock, FrdData, ResultSet


class TestResultsModels:
    """Test suite for result dataclasses and helper methods."""

    def test_frd_block_creation(self):
        """Verify initialization of FrdBlock attributes."""
        node_ids = np.array([1, 2, 3], dtype=np.int64)
        vals = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64)
        block = FrdBlock(
            name="DISP",
            time=1.0,
            step=0,
            component_names=["D1", "D2", "D3"],
            node_ids=node_ids,
            values=vals,
        )

        assert block.name == "DISP"
        assert block.time == 1.0
        assert block.step == 0
        assert block.component_names == ["D1", "D2", "D3"]
        assert len(block.node_ids) == 3
        assert block.values.shape == (3, 3)

    def test_frd_data_get_block_and_times(self):
        """Verify FrdData get_block retrieval and times helpers."""
        node_ids = np.array([1, 2], dtype=np.int64)
        coords = np.zeros((2, 3), dtype=np.float64)

        b1 = FrdBlock(
            name="DISP", time=0.5, step=0, component_names=["D1", "D2", "D3"],
            node_ids=node_ids, values=np.ones((2, 3))
        )
        b2 = FrdBlock(
            name="DISP", time=1.0, step=1, component_names=["D1", "D2", "D3"],
            node_ids=node_ids, values=np.ones((2, 3)) * 2.0
        )
        b3 = FrdBlock(
            name="STRESS", time=1.0, step=1, component_names=["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"],
            node_ids=node_ids, values=np.ones((2, 6)) * 10.0
        )

        frd_data = FrdData(
            node_ids=node_ids,
            node_coords=coords,
            blocks=[b1, b2, b3],
        )

        # get_block 'last'
        disp_last = frd_data.get_block("DISP", which="last")
        assert disp_last is not None
        assert disp_last.time == 1.0
        assert disp_last.step == 1
        assert np.allclose(disp_last.values, 2.0)

        # get_block 'first'
        disp_first = frd_data.get_block("DISP", which="first")
        assert disp_first is not None
        assert disp_first.time == 0.5
        assert disp_first.step == 0

        # get_block integer index
        assert frd_data.get_block("DISP", which=0) == b1
        assert frd_data.get_block("DISP", which=1) == b2
        assert frd_data.get_block("STRESS", which=0) == b3

        # Non-existing block
        assert frd_data.get_block("NON_EXISTING") is None

        # times() helper
        assert frd_data.times("DISP") == [0.5, 1.0]
        assert frd_data.times("STRESS") == [1.0]
        assert frd_data.times() == [0.5, 1.0, 1.0]

    def test_result_set_slice_for_part(self):
        """Verify slice_for_part calculation using MeshModel offsets."""
        bone_nodes = np.zeros((100, 3), dtype=np.float64)
        bone_elems = np.zeros((50, 4), dtype=np.int64)
        bone_part = MeshPart(name="bone", nodes=bone_nodes, elements=bone_elems, element_type="C3D4")

        implant_nodes = np.zeros((30, 3), dtype=np.float64)
        implant_elems = np.zeros((15, 4), dtype=np.int64)
        implant_part = MeshPart(name="implant", nodes=implant_nodes, elements=implant_elems, element_type="C3D4")

        model = MeshModel(parts=[bone_part, implant_part])
        ntot = model.total_nodes()
        assert ntot == 130

        result_set = ResultSet(
            time=1.0,
            u=np.zeros((ntot, 3), dtype=np.float64),
            u_mag=np.zeros(ntot, dtype=np.float64),
            stress6=np.zeros((ntot, 6), dtype=np.float64),
            von_mises=np.zeros(ntot, dtype=np.float64),
            s_max_principal=np.zeros(ntot, dtype=np.float64),
            s_min_principal=np.zeros(ntot, dtype=np.float64),
            strain6=np.zeros((ntot, 6), dtype=np.float64),
            rf=np.zeros((ntot, 3), dtype=np.float64),
            source_frd="test.frd",
        )

        bone_slice = result_set.slice_for_part(model, "bone")
        assert bone_slice == slice(0, 100)

        implant_slice = result_set.slice_for_part(model, "implant")
        assert implant_slice == slice(100, 130)

        with pytest.raises(KeyError):
            result_set.slice_for_part(model, "non_existing_part")

    def test_result_set_summary(self):
        """Verify summary dictionary outputs correct metrics and handles NaNs."""
        ntot = 4
        u = np.array([[0, 0, 0], [0, 0, 0.05], [np.nan, np.nan, np.nan], [0, 0, 0.02]])
        u_mag = np.array([0.0, 0.05, np.nan, 0.02])
        von_mises = np.array([10.0, 50.0, np.nan, 25.0])
        s_max = np.array([5.0, 40.0, np.nan, 20.0])
        s_min = np.array([-15.0, -5.0, np.nan, -30.0])
        rf = np.array([[0, 0, -20.0], [0, 0, -30.0], [np.nan, np.nan, np.nan], [0, 0, -50.0]])

        result_set = ResultSet(
            time=1.0,
            u=u,
            u_mag=u_mag,
            stress6=None,
            von_mises=von_mises,
            s_max_principal=s_max,
            s_min_principal=s_min,
            strain6=None,
            rf=rf,
            source_frd="test_summary.frd",
        )

        summary = result_set.summary()
        assert summary["time"] == 1.0
        assert summary["source_frd"] == "test_summary.frd"
        assert summary["num_nodes"] == 4
        assert pytest.approx(summary["max_u_mag_mm"]) == 0.05
        assert pytest.approx(summary["max_von_mises_MPa"]) == 50.0
        assert pytest.approx(summary["max_principal_stress_MPa"]) == 40.0
        assert pytest.approx(summary["min_principal_stress_MPa"]) == -30.0
        assert pytest.approx(summary["total_rf_N"]) == [0.0, 0.0, -100.0]
        assert pytest.approx(summary["total_rf_mag_N"]) == 100.0
