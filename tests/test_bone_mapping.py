"""
test_bone_mapping.py
Unit tests for CT HU sampling, world-to-index transformations, and heterogeneous bone material assignment.
"""

import pytest
import numpy as np

from fea.models import MeshPart, MaterialAssignment
from material.bone_mapping import (
    load_preset,
    evaluate_formula,
    world_to_index,
    sample_hu,
    element_centroids,
    assign_bone_materials,
    titanium_assignment,
)


class TestBoneMapping:
    """Test suite for material mapping module."""

    @pytest.fixture
    def linear_volume_setup(self):
        """
        Create a synthetic CT volume where HU linearly increases along X axis from 0 to 1000.
        Dimensions: nz=20, ny=20, nx=100 (shape: (20, 20, 100))
        Spacing: (0.5, 0.5, 0.5) mm
        Origin: (10.0, 20.0, 30.0) mm
        Direction: 30-degree rotation around Z axis
        """
        nx, ny, nz = 100, 20, 20
        # Linear slope along X axis: HU(i) = 1000 * i / (nx - 1)
        x_line = np.linspace(0.0, 1000.0, nx, dtype=np.float32)
        vol = np.tile(x_line, (nz, ny, 1))

        origin = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        spacing = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        theta = np.deg2rad(30.0)
        c, s = np.cos(theta), np.sin(theta)
        direction = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        return vol, origin, spacing, direction

    def test_evaluate_formula(self):
        """Verify linear and power formula evaluations."""
        # Linear: y = 10 + 2 * x
        cfg_lin = {"type": "linear", "a": 10.0, "b": 2.0}
        x = np.array([0.0, 5.0, 10.0])
        assert np.allclose(evaluate_formula(cfg_lin, x), [10.0, 20.0, 30.0])

        # Power: y = 2000 * x^1.5
        cfg_pow = {"type": "power", "c": 2000.0, "d": 1.5}
        x_pow = np.array([0.0, 1.0, 4.0])
        assert np.allclose(evaluate_formula(cfg_pow, x_pow), [0.0, 2000.0, 16000.0])

        # Unsupported formula raises ValueError
        with pytest.raises(ValueError):
            evaluate_formula({"type": "unsupported"}, x)

    def test_world_to_index_and_sample_hu_with_rotation(self, linear_volume_setup):
        """
        Verify world_to_index and sample_hu accuracy within 1 HU
        on rotated non-zero origin volume.
        """
        vol, origin, spacing, direction = linear_volume_setup

        # Query points at specific fractional voxel indices: i in [10.5, 50.2, 85.0], j=10.0, k=10.0
        target_indices = np.array([
            [10.5, 10.0, 10.0],
            [50.2, 10.0, 10.0],
            [85.0, 10.0, 10.0],
        ], dtype=np.float64)

        # Compute exact analytic world coordinates: world = origin + D @ (spacing * idx)
        world_pts = origin + (direction @ (spacing * target_indices).T).T

        # 1. world_to_index verification
        calc_indices = world_to_index(world_pts, origin, spacing, direction)
        assert np.allclose(calc_indices, target_indices, atol=1e-5)

        # 2. sample_hu verification (Analytical expected HU = 1000 * i / 99)
        expected_hu = 1000.0 * target_indices[:, 0] / 99.0
        sampled_hu = sample_hu(vol, world_pts, origin, spacing, direction)

        # Sampling accuracy within 1.0 HU
        assert np.allclose(sampled_hu, expected_hu, atol=1.0)

    def test_sample_hu_out_of_bounds(self, linear_volume_setup):
        """Verify points outside the volume bounds return outside_hu (-1000.0) without error."""
        vol, origin, spacing, direction = linear_volume_setup

        # Far outside points
        outside_pts = np.array([
            [-500.0, -500.0, -500.0],
            [9999.0, 9999.0, 9999.0],
        ], dtype=np.float64)

        sampled = sample_hu(vol, outside_pts, origin, spacing, direction, outside_hu=-1000.0)
        assert len(sampled) == 2
        assert np.all(sampled == -1000.0)

    def test_bone_material_assignment_quantization_error(self, linear_volume_setup):
        """
        Verify material assignment quantization:
        - Total material count <= n_bins
        - Quantization error for each element's modulus is strictly within the bin width
        """
        vol, origin, spacing, direction = linear_volume_setup
        nx, ny, nz = 100, 20, 20

        # Construct a synthetic mesh of 50 tetrahedral elements along the X axis
        # Points along the volume interior
        n_elements = 50
        x_fracs = np.linspace(5.0, 90.0, n_elements)
        i_coords = np.column_stack([x_fracs, np.full(n_elements, 10.0), np.full(n_elements, 10.0)])
        c_world = origin + (direction @ (spacing * i_coords).T).T

        # Create small tets around each centroid
        nodes = []
        elements = []
        d = 0.1  # small offset
        for idx, c in enumerate(c_world):
            v0 = c + np.array([0.0, 0.0, 0.0])
            v1 = c + np.array([d, 0.0, 0.0])
            v2 = c + np.array([0.0, d, 0.0])
            v3 = c + np.array([0.0, 0.0, d])
            base_idx = len(nodes)
            nodes.extend([v0, v1, v2, v3])
            elements.append([base_idx, base_idx + 1, base_idx + 2, base_idx + 3])

        nodes = np.array(nodes, dtype=np.float64)
        elements = np.array(elements, dtype=np.int64)
        bone_part = MeshPart(name="mandible", nodes=nodes, elements=elements, element_type="C3D4")

        preset = load_preset()
        preset["n_bins"] = 25
        preset["binning_scale"] = "linear"

        geom_info = {"origin": origin, "spacing": spacing, "direction": direction}
        mat_assign = assign_bone_materials(
            part=bone_part,
            volume_array=vol,
            geometry_info=geom_info,
            cortical_mask_array=None,
            preset=preset
        )

        assert isinstance(mat_assign, MaterialAssignment)
        assert "mandible" in mat_assign.element_material
        elem_indices = mat_assign.element_material["mandible"]

        # Number of materials generated must not exceed n_bins
        n_unique_materials = len(mat_assign.materials)
        assert n_unique_materials <= preset["n_bins"]

        # Calculate theoretical continuous E
        centroids = element_centroids(nodes, elements)
        hu_sampled = sample_hu(vol, centroids, origin, spacing, direction)
        densities = evaluate_formula(preset["density_from_hu"], hu_sampled)
        e_continuous = evaluate_formula(preset["modulus_from_density"], densities)
        e_clipped = np.clip(e_continuous, preset["E_min"], preset["E_max"])

        # Assigned discrete E
        e_assigned = np.array([mat_assign.materials[idx].E for idx in elem_indices])

        # Bin width
        bin_width = (preset["E_max"] - preset["E_min"]) / preset["n_bins"]
        e_errors = np.abs(e_clipped - e_assigned)

        # Quantization error must be within bin width (actually <= bin_width / 2)
        assert np.all(e_errors <= bin_width + 1e-4)

    def test_cortical_mask_fixed_material(self, linear_volume_setup):
        """Verify cortical mask elements receive fixed CORTICAL material."""
        vol, origin, spacing, direction = linear_volume_setup
        nz, ny, nx = vol.shape

        # Create a cortical mask where x < 30 has cortical bone
        cortical_mask = np.zeros(vol.shape, dtype=bool)
        cortical_mask[:, :, :30] = True

        # Two elements: one inside cortical, one in cancellous
        target_indices = np.array([
            [15.0, 10.0, 10.0],  # In cortical (x < 30)
            [70.0, 10.0, 10.0],  # In cancellous (x >= 30)
        ])
        c_world = origin + (direction @ (spacing * target_indices).T).T

        nodes = []
        elements = []
        d = 0.1
        for c in c_world:
            b_idx = len(nodes)
            nodes.extend([c, c + [d, 0, 0], c + [0, d, 0], c + [0, 0, d]])
            elements.append([b_idx, b_idx + 1, b_idx + 2, b_idx + 3])

        nodes = np.array(nodes, dtype=np.float64)
        elements = np.array(elements, dtype=np.int64)
        part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type="C3D4")

        geom_info = {"origin": origin, "spacing": spacing, "direction": direction}
        preset = load_preset()
        mat_assign = assign_bone_materials(
            part=part,
            volume_array=vol,
            geometry_info=geom_info,
            cortical_mask_array=cortical_mask,
            preset=preset
        )

        elem_indices = mat_assign.element_material["bone"]
        # Element 0 is cortical
        mat0 = mat_assign.materials[elem_indices[0]]
        assert mat0.name == "CORTICAL"
        assert mat0.E == preset["cortical_fixed"]["E"]

        # Element 1 is cancellous BONE_xxx
        mat1 = mat_assign.materials[elem_indices[1]]
        assert mat1.name.startswith("BONE_")

    def test_titanium_assignment(self):
        """Verify titanium assignment creates a single homogeneous material."""
        nodes = np.zeros((4, 3), dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        implant_part = MeshPart(name="implant", nodes=nodes, elements=elements, element_type="C3D4")

        ti_assign = titanium_assignment(implant_part)
        assert len(ti_assign.materials) == 1
        mat = ti_assign.materials[0]
        assert mat.name == "TITANIUM"
        assert mat.E == 110000.0
        assert mat.nu == 0.34
        assert mat.density == 4.5e-9
        assert np.all(ti_assign.element_material["implant"] == 0)
