"""
test_result_fields.py
Unit tests for tensor transformations, stress calculations, ResultSet conversion,
part statistics, and reaction balance.
"""

import os
import math
import pytest
import numpy as np

from fea.models import MeshModel, MeshPart, BoundaryConditionSet
from fea.results import FrdBlock, FrdData, ResultSet
from postprocess.result_fields import (
    principal_stresses,
    von_mises,
    build_result_set,
    part_statistics,
    reaction_balance,
)
from postprocess.frd_parser import parse_frd


class TestTensorCalculations:
    """Tests for tensor stress calculation functions (principal stresses and von Mises)."""

    def test_uniaxial_stress(self):
        """Verify uniaxial tension (sxx=100, rest 0): vm == 100, principal == (100, 0, 0)."""
        stress = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        vm = von_mises(stress)
        assert pytest.approx(vm[0]) == 100.0

        ps = principal_stresses(stress)
        np.testing.assert_allclose(ps[0], [100.0, 0.0, 0.0], atol=1e-6)

    def test_pure_shear_stress(self):
        """Verify pure shear (sxy=50): vm == 50*sqrt(3), principal == (50, 0, -50)."""
        stress = np.array([[0.0, 0.0, 0.0, 50.0, 0.0, 0.0]])
        vm = von_mises(stress)
        expected_vm = 50.0 * math.sqrt(3.0)
        assert pytest.approx(vm[0]) == expected_vm

        ps = principal_stresses(stress)
        np.testing.assert_allclose(ps[0], [50.0, 0.0, -50.0], atol=1e-6)

    def test_hydrostatic_stress(self):
        """Verify hydrostatic compression (sxx=syy=szz=-10): vm == 0, principal == (-10, -10, -10)."""
        stress = np.array([[-10.0, -10.0, -10.0, 0.0, 0.0, 0.0]])
        vm = von_mises(stress)
        assert pytest.approx(vm[0]) == 0.0

        ps = principal_stresses(stress)
        np.testing.assert_allclose(ps[0], [-10.0, -10.0, -10.0], atol=1e-6)

    def test_random_tensor_batch_matches_single_calls(self):
        """Verify that batch vectorized principal_stresses matches individual eigvalsh calls for 1000 tensors."""
        np.random.seed(42)
        n = 1000
        # Random 6-components [SXX, SYY, SZZ, SXY, SYZ, SZX]
        stress6 = np.random.uniform(-100.0, 100.0, size=(n, 6))

        batch_result = principal_stresses(stress6)
        assert batch_result.shape == (n, 3)

        for i in range(n):
            s = stress6[i]
            # [ SXX  SXY  SZX ]
            # [ SXY  SYY  SYZ ]
            # [ SZX  SYZ  SZZ ]
            mat = np.array([
                [s[0], s[3], s[5]],
                [s[3], s[1], s[4]],
                [s[5], s[4], s[2]],
            ])
            expected = np.linalg.eigvalsh(mat)[::-1]
            np.testing.assert_allclose(batch_result[i], expected, atol=1e-9)

    def test_nan_handling_in_stresses(self):
        """Verify that rows with NaN preserve NaN in von Mises and principal stresses."""
        stress = np.array([
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, np.nan, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        vm = von_mises(stress)
        assert pytest.approx(vm[0]) == 100.0
        assert np.isnan(vm[1])
        assert np.isnan(vm[2])
        assert pytest.approx(vm[3]) == 0.0

        ps = principal_stresses(stress)
        np.testing.assert_allclose(ps[0], [100.0, 0.0, 0.0], atol=1e-6)
        assert np.all(np.isnan(ps[1]))
        assert np.all(np.isnan(ps[2]))
        np.testing.assert_allclose(ps[3], [0.0, 0.0, 0.0], atol=1e-6)


class TestBuildResultSet:
    """Tests for build_result_set conversion and validation."""

    def _create_synthetic_model_and_frd(self):
        """Helper to create a 2-part model (bone: 4 nodes, implant: 3 nodes)."""
        bone_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        bone_elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        bone_part = MeshPart(name="bone", nodes=bone_nodes, elements=bone_elems, element_type="C3D4")

        implant_nodes = np.array([
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ], dtype=np.float64)
        implant_elems = np.array([[0, 1, 2, 0]], dtype=np.int64)
        implant_part = MeshPart(name="implant", nodes=implant_nodes, elements=implant_elems, element_type="C3D4")

        model = MeshModel(parts=[bone_part, implant_part])

        # Global node IDs: 1..4 (bone), 5..7 (implant)
        all_node_ids = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        all_coords = np.vstack([bone_nodes, implant_nodes])

        return model, all_node_ids, all_coords

    def test_synthetic_two_part_assembly_alignment(self):
        """Verify global index alignment and part separation without boundary mixing."""
        model, all_node_ids, all_coords = self._create_synthetic_model_and_frd()

        # Bone displacement: 0.1, implant displacement: 0.5
        disp_vals = np.array([
            [0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=np.float64)

        # Bone stress: uniaxial 10 MPa; implant stress: uniaxial 500 MPa
        stress_vals = np.array([
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)

        disp_block = FrdBlock(
            name="DISP", time=1.0, step=0, component_names=["D1", "D2", "D3"],
            node_ids=all_node_ids, values=disp_vals,
        )
        stress_block = FrdBlock(
            name="STRESS", time=1.0, step=1, component_names=["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"],
            node_ids=all_node_ids, values=stress_vals,
        )

        frd = FrdData(node_ids=all_node_ids, node_coords=all_coords, blocks=[disp_block, stress_block])

        rs = build_result_set(frd, model)

        assert rs.time == 1.0
        assert rs.u.shape == (7, 3)
        assert rs.von_mises.shape == (7,)

        # Check bone slice (0..4)
        bone_slice = rs.slice_for_part(model, "bone")
        np.testing.assert_allclose(rs.u_mag[bone_slice], 0.1)
        np.testing.assert_allclose(rs.von_mises[bone_slice], 10.0)

        # Check implant slice (4..7)
        implant_slice = rs.slice_for_part(model, "implant")
        np.testing.assert_allclose(rs.u_mag[implant_slice], 0.5)
        np.testing.assert_allclose(rs.von_mises[implant_slice], 500.0)

    def test_node_count_mismatch_raises_value_error(self):
        """Verify that node count mismatch raises ValueError with Korean message."""
        model, all_node_ids, all_coords = self._create_synthetic_model_and_frd()

        # Omit 1 node from frd
        bad_node_ids = all_node_ids[:-1]
        bad_coords = all_coords[:-1]

        frd = FrdData(node_ids=bad_node_ids, node_coords=bad_coords, blocks=[])

        with pytest.raises(ValueError, match="일치하지 않습니다"):
            build_result_set(frd, model)

    def test_coordinate_mismatch_raises_value_error(self):
        """Verify that coordinate mismatch (> 1.5e-3 mm) raises ValueError with Korean message."""
        model, all_node_ids, all_coords = self._create_synthetic_model_and_frd()

        # CalculiX 12.5f 반올림 수준의 미세 오차(0.0005 mm)는 허용됨을 검증
        slight_coords = all_coords.copy()
        slight_coords[0, 0] += 0.0005
        frd_slight = FrdData(node_ids=all_node_ids, node_coords=slight_coords, blocks=[])
        rs_slight = build_result_set(frd_slight, model)
        assert rs_slight is not None

        # Perturb one coordinate by 0.01 mm (> 1.5e-3 mm) -> raises
        perturbed_coords = all_coords.copy()
        perturbed_coords[0, 0] += 0.01

        frd = FrdData(node_ids=all_node_ids, node_coords=perturbed_coords, blocks=[])

        with pytest.raises(ValueError, match="허용 오차 1.5e-3 mm"):
            build_result_set(frd, model)

    def test_partial_nodes_in_block_populates_nan(self):
        """Verify that nodes absent in a result block receive NaN values in ResultSet."""
        model, all_node_ids, all_coords = self._create_synthetic_model_and_frd()

        # DISP block has only nodes 1, 2, 5 (missing 3, 4, 6, 7)
        partial_node_ids = np.array([1, 2, 5], dtype=np.int64)
        partial_vals = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=np.float64)

        disp_block = FrdBlock(
            name="DISP", time=0.5, step=0, component_names=["D1", "D2", "D3"],
            node_ids=partial_node_ids, values=partial_vals,
        )

        frd = FrdData(node_ids=all_node_ids, node_coords=all_coords, blocks=[disp_block])
        rs = build_result_set(frd, model)

        # 0-based indices 0, 1, 4 should be populated
        assert rs.u[0, 0] == 1.0
        assert rs.u[1, 0] == 2.0
        assert rs.u[4, 0] == 5.0

        # Indices 2, 3, 5, 6 should be NaN
        assert np.isnan(rs.u[2, 0])
        assert np.isnan(rs.u[3, 0])
        assert np.isnan(rs.u[5, 0])
        assert np.isnan(rs.u[6, 0])

        assert np.isnan(rs.u_mag[2])
        assert np.isnan(rs.u_mag[3])

        # STRESS was not provided -> should be None
        assert rs.stress6 is None
        assert rs.von_mises is None
        assert rs.s_max_principal is None
        assert rs.s_min_principal is None
        assert rs.rf is None


class TestPartStatistics:
    """Tests for part_statistics calculation and NaN filtering."""

    def test_part_statistics_ignores_nan(self):
        """Verify part_statistics excludes NaN and extracts correct percentiles and peak coordinates."""
        part_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ], dtype=np.float64)
        part = MeshPart(name="bone", nodes=part_nodes, elements=np.zeros((1, 4), dtype=np.int64), element_type="C3D4")
        model = MeshModel(parts=[part])

        # 5 nodes with one NaN at index 2
        u_mag = np.array([0.01, 0.05, np.nan, 0.02, 0.03], dtype=np.float64)
        von_mises_vals = np.array([10.0, 50.0, np.nan, 30.0, 20.0], dtype=np.float64)
        s_max_vals = np.array([5.0, 40.0, np.nan, 25.0, 15.0], dtype=np.float64)
        s_min_vals = np.array([-15.0, -5.0, np.nan, -30.0, -10.0], dtype=np.float64)

        rs = ResultSet(
            time=1.0,
            u=np.zeros((5, 3)),
            u_mag=u_mag,
            stress6=None,
            von_mises=von_mises_vals,
            s_max_principal=s_max_vals,
            s_min_principal=s_min_vals,
            strain6=None,
            rf=None,
            source_frd="dummy.frd",
        )

        stats = part_statistics(rs, model, "bone")

        assert stats["part_name"] == "bone"
        assert stats["num_nodes"] == 5

        # u_mag max at node index 1 (coord: [1.0, 0.0, 0.0])
        assert pytest.approx(stats["max_u_mag"]) == 0.05
        assert stats["max_u_mag_node_idx"] == 1
        assert stats["max_u_mag_node_coord"] == [1.0, 0.0, 0.0]

        # von_mises max at node index 1
        assert pytest.approx(stats["max_von_mises"]) == 50.0
        assert stats["max_von_mises_node_idx"] == 1
        assert stats["max_von_mises_node_coord"] == [1.0, 0.0, 0.0]

        # Mean of valid [10, 50, 30, 20] is 27.5
        assert pytest.approx(stats["mean_von_mises"]) == 27.5
        assert stats["p95_von_mises"] is not None
        assert stats["p99_von_mises"] is not None

        # s_max max at node index 1
        assert pytest.approx(stats["max_s_max_principal"]) == 40.0
        assert stats["max_s_max_node_idx"] == 1

        # s_min minimum at node index 3 (-30.0)
        assert pytest.approx(stats["min_s_min_principal"]) == -30.0
        assert stats["min_s_min_node_idx"] == 3
        assert stats["min_s_min_node_coord"] == [3.0, 0.0, 0.0]


class TestReactionBalance:
    """Tests for reaction_balance equilibrium calculation."""

    def test_reaction_balance_equilibrium(self):
        """Verify reaction_balance correctly sums reaction forces and applied loads."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        part = MeshPart(name="cube", nodes=nodes, elements=np.zeros((1, 4), dtype=np.int64), element_type="C3D4")
        model = MeshModel(parts=[part])

        # Applied load: -100 N in Z on node 3
        bc = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX", "part": "cube", "node_ids": np.array([0, 1, 2]), "dofs": (1, 3)}],
            loads=[{"name": "LOAD", "part": "cube", "node_ids": np.array([3]), "force_N": (0.0, 0.0, -100.0)}],
        )

        # Equilibrium reactions at fixed nodes (sum = +100 N in Z)
        rf = np.array([
            [0.0, 0.0, 30.0],
            [0.0, 0.0, 30.0],
            [0.0, 0.0, 40.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)

        rs = ResultSet(
            time=1.0,
            u=np.zeros((4, 3)),
            u_mag=np.zeros(4),
            stress6=None,
            von_mises=None,
            s_max_principal=None,
            s_min_principal=None,
            strain6=None,
            rf=rf,
            source_frd="",
        )

        bal = reaction_balance(rs, bc, model)
        assert bal is not None
        np.testing.assert_allclose(bal["reaction_force_sum"], [0.0, 0.0, 100.0], atol=1e-6)
        np.testing.assert_allclose(bal["applied_force_sum"], [0.0, 0.0, -100.0], atol=1e-6)
        assert pytest.approx(bal["reaction_mag"]) == 100.0
        assert pytest.approx(bal["applied_mag"]) == 100.0
        assert pytest.approx(bal["residual_mag"], abs=1e-6) == 0.0
        assert pytest.approx(bal["relative_error"], abs=1e-6) == 0.0

    def test_reaction_balance_none_when_rf_missing(self):
        """Verify reaction_balance returns None if rs.rf is None."""
        part = MeshPart(name="p", nodes=np.zeros((2, 3)), elements=np.zeros((1, 4), dtype=np.int64), element_type="C3D4")
        model = MeshModel(parts=[part])
        bc = BoundaryConditionSet(fixed_node_sets=[], loads=[])
        rs = ResultSet(
            time=1.0, u=np.zeros((2, 3)), u_mag=np.zeros(2), stress6=None,
            von_mises=None, s_max_principal=None, s_min_principal=None,
            strain6=None, rf=None, source_frd="",
        )
        assert reaction_balance(rs, bc, model) is None


class TestFixtureIntegration:
    """End-to-end integration test with real CalculiX cube_compression fixture."""

    def test_cube_compression_fixture_build_result_set(self):
        """Verify parsing cube_compression.frd and building complete ResultSet."""
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        frd_path = os.path.join(fixtures_dir, "cube_compression.frd")
        if not os.path.isfile(frd_path):
            pytest.skip(f"Fixture not found: {frd_path}")

        frd = parse_frd(frd_path)
        # Create corresponding MeshModel matching the 8 nodes of the cube
        part = MeshPart(
            name="cube",
            nodes=frd.node_coords.copy(),
            elements=np.zeros((5, 4), dtype=np.int64),
            element_type="C3D4",
        )
        model = MeshModel(parts=[part])

        rs = build_result_set(frd, model)

        assert rs.u is not None
        assert rs.u.shape == (8, 3)
        assert rs.u_mag is not None
        assert rs.u_mag.shape == (8,)

        assert rs.stress6 is not None
        assert rs.stress6.shape == (8, 6)
        assert rs.von_mises is not None
        assert rs.von_mises.shape == (8,)
        assert rs.s_max_principal is not None
        assert rs.s_min_principal is not None

        assert rs.strain6 is not None
        assert rs.rf is not None
        assert rs.rf.shape == (8, 3)

        # In uniaxial compression (z-axis), s_min_principal should be negative
        valid_smin = rs.s_min_principal[~np.isnan(rs.s_min_principal)]
        assert np.all(valid_smin <= 1e-4)

        # Check part statistics
        stats = part_statistics(rs, model, "cube")
        assert stats["num_nodes"] == 8
        assert stats["max_von_mises"] > 0.0
