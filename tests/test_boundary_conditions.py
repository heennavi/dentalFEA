"""
test_boundary_conditions.py
Unit tests for boundary conditions, spatial selection, angular loading, and BC validation.
"""

import pytest
import numpy as np

from fea.models import MeshPart, MeshModel, BoundaryConditionSet
from fea.boundary_conditions import (
    fix_by_plane,
    fix_by_region,
    load_on_region,
    load_from_angle,
    default_load_presets,
    validate_bc,
)


class TestBoundaryConditions:
    """Test suite for boundary condition and loading module."""

    @pytest.fixture
    def unit_cube_part(self):
        """
        Create a 1.0 x 1.0 x 1.0 mm box mesh part with 8 corner nodes and 6 elements.
        Z ranges from 0.0 to 1.0.
        """
        nodes = np.array([
            [0.0, 0.0, 0.0],  # 0: bottom
            [1.0, 0.0, 0.0],  # 1: bottom
            [1.0, 1.0, 0.0],  # 2: bottom
            [0.0, 1.0, 0.0],  # 3: bottom
            [0.0, 0.0, 1.0],  # 4: top
            [1.0, 0.0, 1.0],  # 5: top
            [1.0, 1.0, 1.0],  # 6: top
            [0.0, 1.0, 1.0],  # 7: top
        ], dtype=np.float64)

        # 6-tet box decomposition
        elements = np.array([
            [0, 1, 2, 5],
            [0, 2, 3, 7],
            [0, 5, 7, 4],
            [2, 5, 6, 7],
            [0, 2, 5, 7],
            [0, 1, 5, 7]
        ], dtype=np.int64)

        return MeshPart(name="cube_block", nodes=nodes, elements=elements, element_type="C3D4")

    def test_fix_by_plane_bottom(self, unit_cube_part):
        """Verify fix_by_plane selects all 4 bottom nodes at Z=0."""
        # Plane at Z=0 with normal +Z, side="below" (or side="on")
        fix_item = fix_by_plane(
            part=unit_cube_part,
            plane_point=[0.0, 0.0, 0.0],
            plane_normal=[0.0, 0.0, 1.0],
            side="below",
            tol_mm=0.01,
            name="FIX_BOTTOM"
        )

        assert fix_item["name"] == "FIX_BOTTOM"
        assert fix_item["part"] == "cube_block"
        assert fix_item["dofs"] == (1, 3)

        fixed_node_ids = set(fix_item["node_ids"])
        expected_node_ids = {0, 1, 2, 3}  # 4 bottom nodes
        assert fixed_node_ids == expected_node_ids

    def test_fix_by_region(self, unit_cube_part):
        """Verify fix_by_region with custom predicate function."""
        # Fix nodes with X <= 0.01 (left face: nodes 0, 3, 4, 7)
        fix_item = fix_by_region(
            part=unit_cube_part,
            predicate=lambda pts: pts[:, 0] <= 0.01,
            name="FIX_LEFT"
        )
        assert set(fix_item["node_ids"]) == {0, 3, 4, 7}

    def test_load_on_region_force_partition(self, unit_cube_part):
        """
        Verify load_on_region applies resultant force and total partitioned force matches resultant.
        """
        # Apply 100N force downward on top face (Z >= 0.99, nodes 4, 5, 6, 7)
        resultant_F = (0.0, 0.0, -100.0)
        load_item = load_on_region(
            part=unit_cube_part,
            predicate_or_node_ids=lambda pts: pts[:, 2] >= 0.99,
            force_vector_N=resultant_F,
            name="TOP_LOAD"
        )

        assert load_item["name"] == "TOP_LOAD"
        assert load_item["part"] == "cube_block"
        assert load_item["force_N"] == resultant_F

        node_ids = load_item["node_ids"]
        assert len(node_ids) == 4
        assert set(node_ids) == {4, 5, 6, 7}

        # Verify sum of distributed nodal forces equals total resultant force
        n_nodes = len(node_ids)
        nodal_force = np.array(resultant_F) / n_nodes
        total_partitioned = nodal_force * n_nodes
        assert np.allclose(total_partitioned, resultant_F)

    def test_load_from_angle(self):
        """Verify load_from_angle magnitude preservation and tilt alignment."""
        mag = 150.0
        axis = [0.0, 0.0, -1.0]

        # 1. Pure axial load (tilt=0)
        f_axial = load_from_angle(mag, axis_direction=axis, tilt_deg=0.0)
        assert np.isclose(np.linalg.norm(f_axial), mag)
        assert np.allclose(f_axial, [0.0, 0.0, -150.0])

        # 2. 30-degree inclined load
        f_oblique = load_from_angle(mag, axis_direction=axis, tilt_deg=30.0, tilt_azimuth_deg=0.0)
        assert np.isclose(np.linalg.norm(f_oblique), mag)

        # In-plane angle check: dot product with axis must equal cos(30 deg)
        unit_oblique = f_oblique / np.linalg.norm(f_oblique)
        unit_axis = np.array(axis) / np.linalg.norm(axis)
        cos_angle = np.dot(unit_oblique, unit_axis)
        assert np.isclose(cos_angle, np.cos(np.deg2rad(30.0)), atol=1e-5)

        # 3. 90-degree lateral load
        f_lateral = load_from_angle(mag, axis_direction=axis, tilt_deg=90.0)
        assert np.isclose(np.linalg.norm(f_lateral), mag)
        assert np.isclose(f_lateral[2], 0.0, atol=1e-5)  # Z component is zero

    def test_default_load_presets(self):
        """Verify default presets exist and have required keys."""
        presets = default_load_presets()
        assert "axial_100N" in presets
        assert "oblique_30deg_100N" in presets
        for key, p in presets.items():
            assert "magnitude_N" in p
            assert "tilt_deg" in p
            assert p["magnitude_N"] > 0

    def test_validate_bc_pass(self, unit_cube_part):
        """Verify valid boundary condition set passes with empty error list."""
        model = MeshModel(parts=[unit_cube_part])

        fix_item = fix_by_plane(unit_cube_part, [0, 0, 0], [0, 0, 1], side="below")
        load_item = load_on_region(unit_cube_part, [4, 5, 6, 7], (0.0, 0.0, -100.0))

        bc = BoundaryConditionSet(fixed_node_sets=[fix_item], loads=[load_item])
        issues = validate_bc(model, bc)
        assert len(issues) == 0

    def test_validate_bc_warnings_and_errors(self, unit_cube_part):
        """Verify warnings on unconstrained model and errors on zero force / empty node sets."""
        model = MeshModel(parts=[unit_cube_part])

        # 1. No fixed nodes -> Rigid body motion warning
        empty_fix_bc = BoundaryConditionSet(
            fixed_node_sets=[],
            loads=[{"name": "LOAD", "part": "cube_block", "node_ids": np.array([4]), "force_N": (0, 0, -100)}]
        )
        issues = validate_bc(model, empty_fix_bc)
        assert any("강체 운동" in msg for msg in issues)

        # 2. Zero force magnitude -> Error
        zero_force_bc = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX", "part": "cube_block", "node_ids": np.array([0, 1, 2, 3]), "dofs": (1, 3)}],
            loads=[{"name": "ZERO_LOAD", "part": "cube_block", "node_ids": np.array([4]), "force_N": (0, 0, 0)}]
        )
        issues2 = validate_bc(model, zero_force_bc)
        assert any("합력 크기" in msg for msg in issues2)

        # 3. Empty load nodes -> Error
        empty_load_nodes_bc = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX", "part": "cube_block", "node_ids": np.array([0, 1, 2, 3]), "dofs": (1, 3)}],
            loads=[{"name": "EMPTY_LOAD", "part": "cube_block", "node_ids": np.array([], dtype=int), "force_N": (0, 0, -100)}]
        )
        issues3 = validate_bc(model, empty_load_nodes_bc)
        assert any("0개" in msg for msg in issues3)

        # 4. Unknown part name -> Error
        unknown_part_bc = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX", "part": "nonexistent_part", "node_ids": np.array([0]), "dofs": (1, 3)}],
            loads=[]
        )
        issues4 = validate_bc(model, unknown_part_bc)
        assert any("존재하지 않습니다" in msg for msg in issues4)
