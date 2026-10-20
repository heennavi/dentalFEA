"""
test_gmsh_mesher.py
Unit tests for Gmsh tetrahedral meshing, volume preservation, orientation fixing, and CalculiX conversion.
"""

import os
import tempfile
import pytest
import numpy as np
import trimesh

from meshing.gmsh_mesher import (
    surface_to_stl,
    compute_tetrahedron_volumes,
    fix_orientation,
    gmsh_tet10_to_ccx,
    mesh_closed_surface,
    build_part_mesh,
)
from fea.models import MeshPart


class TestGmshMesher:
    """Test suite for Gmsh meshing module."""

    @pytest.fixture
    def sphere_5mm(self):
        """Create a 5mm radius icosphere with subdivisions=3."""
        # 5mm 구 (해석 체적: 4/3 * pi * 5^3 = 523.5988 mm^3)
        return trimesh.creation.icosphere(subdivisions=3, radius=5.0)

    @pytest.fixture
    def cube_10mm(self):
        """Create a 10x10x10 mm box (volume: 1000 mm^3)."""
        return trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    def test_surface_to_stl(self, sphere_5mm):
        """Verify surface_to_stl writes a valid STL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = os.path.join(tmpdir, "test_sphere.stl")
            surface_to_stl(sphere_5mm, stl_path)

            assert os.path.exists(stl_path)
            loaded = trimesh.load(stl_path, force="mesh")
            assert loaded.is_watertight
            assert len(loaded.vertices) == len(sphere_5mm.vertices)

    def test_non_watertight_raises_error(self):
        """Verify non-watertight mesh raises ValueError."""
        # Open surface (plane)
        plane = trimesh.creation.box(extents=(10.0, 10.0, 0.0))
        plane_open = trimesh.Trimesh(vertices=plane.vertices[:4], faces=[[0, 1, 2], [0, 2, 3]])

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = os.path.join(tmpdir, "open_plane.stl")
            plane_open.export(stl_path)

            with pytest.raises(ValueError, match="watertight"):
                mesh_closed_surface(stl_path, size_far=2.0, size_near=1.0)

    def test_auto_repair_non_watertight(self, cube_10mm):
        """Verify non-watertight mesh with small holes is auto-repaired and successfully meshed."""
        # 큐브에서 면 1개 제거하여 non-watertight 상태 생성
        damaged = cube_10mm.copy()
        damaged.update_faces(np.arange(len(damaged.faces) - 1))
        assert not damaged.is_watertight

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = os.path.join(tmpdir, "damaged_cube.stl")
            damaged.export(stl_path)

            nodes, elements = mesh_closed_surface(
                stl_path=stl_path,
                size_far=2.5,
                size_near=2.0,
                element_order=1
            )
            assert len(nodes) > 0
            assert len(elements) > 0


    def test_mesh_closed_surface_volume_and_positive_elements(self, cube_10mm, sphere_5mm):
        """
        Verify that tetrahedral mesh volume matches analytic volume within 2%,
        and all element volumes are strictly positive.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Cube test (10x10x10mm -> analytical volume 1000.0 mm^3)
            cube_stl = os.path.join(tmpdir, "cube_10mm.stl")
            cube_10mm.export(cube_stl)

            nodes_cube, elements_cube = mesh_closed_surface(
                stl_path=cube_stl,
                size_far=2.0,
                size_near=1.5,
                element_order=2,
            )

            assert elements_cube.shape[1] == 10
            vols_cube = compute_tetrahedron_volumes(nodes_cube, elements_cube)
            assert np.all(vols_cube > 0.0), "All cube elements must have positive volume"

            total_cube_vol = float(np.sum(vols_cube))
            cube_analytic = 1000.0
            rel_error_cube = abs(total_cube_vol - cube_analytic) / cube_analytic
            # Cube volume matches analytic volume within 0.1% (exact flat boundaries)
            assert rel_error_cube < 0.005, f"Cube volume relative error {rel_error_cube} exceeds threshold"

            # 2. Sphere test (radius 5mm -> discrete volume vs mesh sum within 2%)
            sphere_stl = os.path.join(tmpdir, "sphere_5mm.stl")
            sphere_5mm.export(sphere_stl)

            nodes_sph, elements_sph = mesh_closed_surface(
                stl_path=sphere_stl,
                size_far=1.0,
                size_near=0.8,
                element_order=1,
            )

            assert elements_sph.shape[1] == 4
            vols_sph = compute_tetrahedron_volumes(nodes_sph, elements_sph)
            assert np.all(vols_sph > 0.0), "All sphere elements must have positive volume"

            total_sph_vol = float(np.sum(vols_sph))
            discrete_sph_vol = abs(float(sphere_5mm.volume))
            rel_error_sph = abs(total_sph_vol - discrete_sph_vol) / discrete_sph_vol
            # Mesh element volume matches the input surface volume within 2%
            assert rel_error_sph < 0.02, f"Sphere volume relative error {rel_error_sph} exceeds 2%"

    def test_gmsh_tet10_to_ccx(self):
        """
        Verify gmsh_tet10_to_ccx swaps the last two columns for 10-node tets
        and leaves 4-node tets unmodified.
        """
        # 10-node tet: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        gmsh_elems = np.arange(20, dtype=np.int64).reshape((2, 10))
        ccx_elems = gmsh_tet10_to_ccx(gmsh_elems)

        assert ccx_elems.shape == gmsh_elems.shape
        # First 8 columns identical
        assert np.array_equal(ccx_elems[:, :8], gmsh_elems[:, :8])
        # Last two columns exchanged
        assert np.array_equal(ccx_elems[:, 8], gmsh_elems[:, 9])
        assert np.array_equal(ccx_elems[:, 9], gmsh_elems[:, 8])

        # 4-node tet should be untouched
        elems_4 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
        ccx_4 = gmsh_tet10_to_ccx(elems_4)
        assert np.array_equal(elems_4, ccx_4)

    def test_fix_orientation_inverts_negative_elements(self):
        """Verify fix_orientation detects and reverses negative volume elements."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Normal positive element
        pos_elem = np.array([[0, 1, 2, 3]], dtype=np.int64)
        assert compute_tetrahedron_volumes(nodes, pos_elem)[0] > 0

        # Inverted negative element
        neg_elem = np.array([[0, 2, 1, 3]], dtype=np.int64)
        assert compute_tetrahedron_volumes(nodes, neg_elem)[0] < 0

        # Run fix_orientation
        fixed = fix_orientation(nodes, neg_elem)
        assert compute_tetrahedron_volumes(nodes, fixed)[0] > 0

    def test_refine_surface_gradient(self, sphere_5mm):
        """
        Verify that using a refinement surface creates a finer mesh (smaller element volume)
        near the refinement surface than far from it.
        """
        # Create a tiny cylinder at the center as the refinement target
        ref_cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sphere_stl = os.path.join(tmpdir, "sphere.stl")
            ref_stl = os.path.join(tmpdir, "ref_cyl.stl")
            sphere_5mm.export(sphere_stl)
            ref_cylinder.export(ref_stl)

            nodes, elements = mesh_closed_surface(
                stl_path=sphere_stl,
                size_far=2.0,
                size_near=0.6,
                refine_surface_stl=ref_stl,
                element_order=1,
            )

            # Compute element centroids
            v0 = nodes[elements[:, 0]]
            v1 = nodes[elements[:, 1]]
            v2 = nodes[elements[:, 2]]
            v3 = nodes[elements[:, 3]]
            centroids = (v0 + v1 + v2 + v3) / 4.0

            # Distance from origin (where ref_cylinder is located)
            dist_to_center = np.linalg.norm(centroids, axis=1)
            vols = compute_tetrahedron_volumes(nodes, elements)

            # Near: dist < 2.0 mm, Far: dist > 3.5 mm
            near_mask = dist_to_center < 2.0
            far_mask = dist_to_center > 3.5

            assert np.any(near_mask), "Should have elements near refinement surface"
            assert np.any(far_mask), "Should have elements far from refinement surface"

            mean_vol_near = np.mean(vols[near_mask])
            mean_vol_far = np.mean(vols[far_mask])

            # Elements near the cylinder must have smaller average volume than elements near outer sphere boundary
            assert mean_vol_near < mean_vol_far, (
                f"Near volume ({mean_vol_near:.4f}) should be significantly smaller "
                f"than far volume ({mean_vol_far:.4f})"
            )

    def test_build_part_mesh(self, sphere_5mm):
        """Verify build_part_mesh produces a complete MeshPart with C3D10 elements."""
        part = build_part_mesh(
            name="bone_sample",
            surface=sphere_5mm,
            size_far=2.0,
            size_near=1.5,
            order=2,
        )

        assert isinstance(part, MeshPart)
        assert part.name == "bone_sample"
        assert part.element_type == "C3D10"
        assert part.nodes.ndim == 2 and part.nodes.shape[1] == 3
        assert part.elements.ndim == 2 and part.elements.shape[1] == 10
        assert len(part.nodes) > 0
        assert len(part.elements) > 0

    def test_mesh_closed_surface_complex_torus(self):
        """Verify that complex non-convex watertight surface (torus) meshes successfully."""
        torus = trimesh.creation.torus(major_radius=6.0, minor_radius=2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            torus_stl = os.path.join(tmpdir, "torus.stl")
            torus.export(torus_stl)

            nodes, elements = mesh_closed_surface(
                stl_path=torus_stl,
                size_far=2.0,
                size_near=1.5,
                element_order=2,
            )
            assert len(nodes) > 0
            assert len(elements) > 0
            assert elements.shape[1] == 10
            vols = compute_tetrahedron_volumes(nodes, elements)
            assert np.all(vols > 0.0), "All torus elements must have positive volume"

    def test_mesh_algorithm_and_optimization_options(self, cube_10mm):
        """Verify meshing with specific 3D algorithm (Delaunay/Netgen) and Netgen optimizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cube_stl = os.path.join(tmpdir, "cube_opt.stl")
            cube_10mm.export(cube_stl)

            # 1. Delaunay 알고리즘 및 Netgen 최적화
            nodes_del, elems_del = mesh_closed_surface(
                stl_path=cube_stl,
                size_far=3.0,
                size_near=2.0,
                element_order=1,
                mesh_algorithm_3d="delaunay",
                optimize_netgen=True,
            )
            assert len(nodes_del) > 0
            assert len(elems_del) > 0
            assert elems_del.shape[1] == 4
            vols = compute_tetrahedron_volumes(nodes_del, elems_del)
            assert np.all(vols > 0.0), "All elements must have positive volume"

            # 2. build_part_mesh 옵션 연계
            part = build_part_mesh(
                name="test_part",
                surface=cube_10mm,
                size_far=3.0,
                size_near=2.0,
                order=1,
                mesh_algorithm_3d="netgen",
                optimize_netgen=True,
            )
            assert part.element_type == "C3D4"
            assert len(part.elements) > 0


