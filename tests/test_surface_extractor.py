"""
test_surface_extractor.py
Unit tests for boundary face extraction, outward normal verification, region predicates, and contact pair detection.
"""

import os
import tempfile
import pytest
import numpy as np
import trimesh

from meshing.gmsh_mesher import mesh_closed_surface, build_part_mesh
from meshing.surface_extractor import (
    boundary_faces,
    faces_in_region,
    plane_below,
    plane_above,
    sphere,
    cylinder,
    contact_pair_from_parts,
)
from fea.models import MeshPart, TieDef


class TestSurfaceExtractor:
    """Test suite for boundary surface extraction and contact pair detection."""

    @pytest.fixture
    def unit_cube_mesh(self):
        """Create a 1.0 x 1.0 x 1.0 mm box mesh (centered at [0.5, 0.5, 0.5])."""
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        box.apply_translation([0.5, 0.5, 0.5])  # Coordinates in [0, 1]^3

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = os.path.join(tmpdir, "unit_cube.stl")
            box.export(stl_path)
            nodes, elements = mesh_closed_surface(stl_path, size_far=0.5, size_near=0.4, element_order=1)

        return nodes, elements

    def test_boundary_faces_outward_normals(self, unit_cube_mesh):
        """
        Verify that extracted boundary faces match the surface count,
        and all face normal vectors point outward from the cube center.
        """
        nodes, elements = unit_cube_mesh
        cube_center = np.array([0.5, 0.5, 0.5])

        faces_nodes, face_info = boundary_faces(elements, nodes=nodes)

        assert len(faces_nodes) > 0
        assert len(faces_nodes) == len(face_info)

        # Check outward normal direction for every boundary face
        v0 = nodes[faces_nodes[:, 0]]
        v1 = nodes[faces_nodes[:, 1]]
        v2 = nodes[faces_nodes[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        face_centroids = (v0 + v1 + v2) / 3.0

        # Vector from cube center to face centroid
        outward_vectors = face_centroids - cube_center

        dots = np.sum(normals * outward_vectors, axis=1)

        # Every face normal must point outwards (dot > 0)
        assert np.all(dots > 0.0), f"Found {np.sum(dots <= 0.0)} inward or degenerate faces out of {len(dots)}"

        # CalculiX face IDs must be strictly between 1 and 4
        assert np.all(face_info[:, 1] >= 1) and np.all(face_info[:, 1] <= 4)

    def test_faces_in_region_predicates(self, unit_cube_mesh):
        """Verify region filtering with plane_below, plane_above, sphere, and cylinder."""
        nodes, elements = unit_cube_mesh
        faces_nodes, _ = boundary_faces(elements, nodes=nodes)

        # 1. Plane below: Z <= 0.1 (Bottom face)
        bottom_indices = faces_in_region(nodes, faces_nodes, plane_below([0.0, 0.0, 0.1], [0.0, 0.0, 1.0]))
        assert len(bottom_indices) > 0
        bottom_centroids = np.mean(nodes[faces_nodes[bottom_indices]], axis=1)
        assert np.all(bottom_centroids[:, 2] <= 0.1)

        # 2. Sphere predicate: around corner [0, 0, 0] with radius 0.4
        corner_indices = faces_in_region(nodes, faces_nodes, sphere([0.0, 0.0, 0.0], 0.4))
        assert len(corner_indices) > 0
        corner_centroids = np.mean(nodes[faces_nodes[corner_indices]], axis=1)
        assert np.all(np.linalg.norm(corner_centroids, axis=1) <= 0.4)

        # 3. Cylinder predicate: vertical axis along Z at [0.5, 0.5] with radius 0.35
        cyl_indices = faces_in_region(nodes, faces_nodes, cylinder([0.5, 0.5, 0.0], [0.5, 0.5, 1.0], 0.35))
        assert len(cyl_indices) > 0

    def test_contact_pair_from_parts_touching(self):
        """Verify contact pair detection on two adjacent touching cubes."""
        # Cube 1: X in [0, 2], Y in [0, 2], Z in [0, 2]
        cube_a = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        cube_a.apply_translation([1.0, 1.0, 1.0])

        # Cube 2: X in [2, 4], Y in [0, 2], Z in [0, 2] (Shares interface at X=2.0)
        cube_b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        cube_b.apply_translation([3.0, 1.0, 1.0])

        part_a = build_part_mesh("part_a", cube_a, size_far=1.0, size_near=0.8, order=1)
        part_b = build_part_mesh("part_b", cube_b, size_far=0.8, size_near=0.6, order=1)

        # Interface is at X = 2.0
        tie = contact_pair_from_parts(part_a, part_b, tol_mm=0.1, master="auto")

        assert isinstance(tie, TieDef)
        assert tie.master_part in ("part_a", "part_b")
        assert len(tie.slave_nodes) > 0
        assert len(tie.master_faces) > 0

        # Verify that detected slave nodes are located exactly on the interface (X ~ 2.0)
        slave_part = part_a if tie.slave_part == "part_a" else part_b
        slave_coords = slave_part.nodes[tie.slave_nodes]
        assert np.allclose(slave_coords[:, 0], 2.0, atol=0.1)

    def test_contact_pair_from_parts_separated(self):
        """Verify that separated parts result in empty contact pair."""
        # Cube 1: X in [0, 2]
        cube_a = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        cube_a.apply_translation([1.0, 1.0, 1.0])

        # Cube 2: X in [10, 12] (Separated by 8mm)
        cube_b = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        cube_b.apply_translation([11.0, 1.0, 1.0])

        part_a = build_part_mesh("part_a", cube_a, size_far=1.2, size_near=1.0, order=1)
        part_b = build_part_mesh("part_b", cube_b, size_far=1.2, size_near=1.0, order=1)

        tie = contact_pair_from_parts(part_a, part_b, tol_mm=0.2, master="auto")

        assert isinstance(tie, TieDef)
        assert len(tie.slave_nodes) == 0
        assert len(tie.master_faces) == 0
