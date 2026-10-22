"""
test_pipeline_validation.py
Comprehensive quantitative validation of the FEA pipeline against analytical solutions and contact formulation.

[Validation Benchmarks]
1. Uniaxial Compression of Homogeneous Cube:
   - 10 x 10 x 10 mm, E = 1000 MPa, nu = 0.3, Total load = 100 N
   - Analytical: Stress = 1.0 MPa, Tip Displacement = 0.010 mm
   - Criterion: Mean tip displacement and mean vertical stress within 5% of theory.
2. Cantilever Beam Bending:
   - b = 2 mm, h = 2 mm, L = 20 mm, E = 1000 MPa, nu = 0.3, Tip Load P = 1.0 N
   - Analytical (Euler-Bernoulli): delta = PL^3 / (3EI) = 2.0 mm
   - Criterion: C3D10 quadratic tetrahedral mesh deflection within 10% of theory.
               (C3D4 linear tetrahedral mesh recorded to illustrate shear locking stiffness).
3. Contact TIE Formulation Validation:
   - Two 10 x 10 x 10 mm blocks connected via CalculiX *TIE constraint vs single 10 x 10 x 20 mm block.
   - Criterion: Top displacement difference between TIE assembly and monolithic block within 3%.
4. Minimum Result Reader Unit Test:
   - Verifies robust parsing of displacements and stresses from CalculiX .frd outputs.
"""

import os
import shutil
import tempfile
import subprocess
import pytest
import numpy as np
import trimesh

from fea.models import MeshModel, MeshPart, MaterialAssignment, MaterialDef, BoundaryConditionSet, TieDef
from meshing.gmsh_mesher import build_part_mesh
from meshing.surface_extractor import contact_pair_from_parts
from solver.calculix_writer import write_inp
from postprocess.result_reader import read_frd_displacements, read_frd_stresses, read_frd_results


CCX_PATH = shutil.which("ccx") or "/usr/bin/ccx"
HAS_CCX = os.path.isfile(CCX_PATH) and os.access(CCX_PATH, os.X_OK)


@pytest.mark.skipif(not HAS_CCX, reason="CalculiX solver binary (ccx) not available on this system.")
class TestPipelineValidation:
    """Quantitative validation suite verifying numerical accuracy of dental FEA pipeline."""

    def test_01_cube_compression_analytical(self):
        """
        Benchmark 1: Uniaxial compression of homogeneous cube.

        Dimensions: 10 x 10 x 10 mm
        Material: E = 1000.0 MPa, nu = 0.3
        Boundary: Bottom face (z=0) roller constrained (u_z = 0) with isostatic 3-2-1 in-plane
                  restraints to permit unrestrained Poisson expansion while preventing rigid body motions.
        Load: Top face (z=10) resultant compressive load of 100.0 N (F_z = -100.0 N).

        Analytical Solution:
            - Vertical Stress: sigma_zz = F / A = -100.0 / 100.0 = -1.0 MPa
            - Top Deflection: delta_z = sigma_zz * L / E = -1.0 * 10.0 / 1000.0 = -0.010 mm
        Tolerance: Mean displacement and mean vertical stress within 5% of theory.
        """
        box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        box.apply_translation([5.0, 5.0, 5.0])  # x in [0, 10], y in [0, 10], z in [0, 10]

        part = build_part_mesh("cube", box, size_far=1.5, size_near=1.5, order=2)
        model = MeshModel(parts=[part])

        z_bot = np.where(np.abs(part.nodes[:, 2] - 0.0) < 1e-4)[0]
        z_top = np.where(np.abs(part.nodes[:, 2] - 10.0) < 1e-4)[0]

        # Isostatic 3-2-1 kinematic boundary conditions to avoid Poisson constraint distortion:
        # 1. All bottom nodes: u_z = 0 (DOF 3)
        # 2. Centroid-most bottom node: u_x = 0, u_y = 0 (DOFs 1, 2)
        # 3. Maximum-x bottom node: u_y = 0 (DOF 2) to eliminate z-axis rotation
        dists_center = np.linalg.norm(part.nodes[z_bot, :2] - [5.0, 5.0], axis=1)
        center_idx = z_bot[np.argmin(dists_center)]
        rot_idx = z_bot[np.argmax(part.nodes[z_bot, 0])]

        bc = BoundaryConditionSet(
            fixed_node_sets=[
                {"name": "FIX_Z_BOT", "part": "cube", "node_ids": z_bot, "dofs": (3, 3)},
                {"name": "FIX_XY_CENTER", "part": "cube", "node_ids": np.array([center_idx]), "dofs": (1, 2)},
                {"name": "FIX_Y_ROT", "part": "cube", "node_ids": np.array([rot_idx]), "dofs": (2, 2)},
            ],
            loads=[
                {"name": "LOAD_COMPRESSION", "part": "cube", "node_ids": z_top, "force_N": (0.0, 0.0, -100.0)}
            ]
        )

        mat = MaterialDef(name="MAT_BONE_HOMOGENEOUS", E=1000.0, nu=0.3)
        mat_assign = MaterialAssignment(
            element_material={"cube": np.zeros(len(part.elements), dtype=np.int64)},
            materials=[mat]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            job_name = "cube_compression"
            inp_path = os.path.join(tmpdir, f"{job_name}.inp")
            write_inp(
                path=inp_path,
                model=model,
                materials={"cube": mat_assign},
                bc=bc,
                ties=[],
                title="Validation Benchmark 1: Uniaxial Cube Compression",
                apply_ccx_ordering=False,
            )

            res = subprocess.run([CCX_PATH, job_name], cwd=tmpdir, capture_output=True, text=True)
            assert res.returncode == 0, f"CalculiX solver failed: {res.stdout}\n{res.stderr}"

            frd_path = os.path.join(tmpdir, f"{job_name}.frd")
            assert os.path.isfile(frd_path), "Expected .frd file was not produced."

            results = read_frd_results(frd_path)
            disp_dict = results["displacements"]
            stress_dict = results["stresses"]

            # Top displacement evaluation
            top_dz = [disp_dict[int(nid) + 1][2] for nid in z_top]
            mean_top_dz = float(np.mean(top_dz))

            # Stress SZZ evaluation (across mid-height region away from end-effects and across volume)
            mid_node_mask = (part.nodes[:, 2] > 3.0) & (part.nodes[:, 2] < 7.0)
            mid_nids = np.where(mid_node_mask)[0]
            mid_szz = [stress_dict[int(nid) + 1][2] for nid in mid_nids if (int(nid) + 1) in stress_dict]
            mean_mid_szz = float(np.mean(mid_szz))

            all_szz = [s[2] for s in stress_dict.values()]
            mean_all_szz = float(np.mean(all_szz))

            err_disp_pct = abs(mean_top_dz - (-0.010)) / 0.010 * 100.0
            err_stress_pct = abs(mean_mid_szz - (-1.0)) / 1.0 * 100.0

            print(f"\n[Cube Compression Results]")
            print(f"  Theoretical delta_z = -0.010000 mm | Computed delta_z = {mean_top_dz:.6f} mm | Error = {err_disp_pct:.2f}%")
            print(f"  Theoretical sigma_zz = -1.000000 MPa | Computed mid sigma_zz = {mean_mid_szz:.6f} MPa | Error = {err_stress_pct:.2f}%")
            print(f"  Mean volume-wide sigma_zz = {mean_all_szz:.6f} MPa (Error: {abs(mean_all_szz - (-1.0)) * 100:.2f}%)")

            assert err_disp_pct <= 5.0, f"Top displacement error ({err_disp_pct:.2f}%) exceeds 5.0% threshold."
            assert err_stress_pct <= 5.0, f"Midplane vertical stress error ({err_stress_pct:.2f}%) exceeds 5.0% threshold."

    def test_02_cantilever_beam_bending(self):
        """
        Benchmark 2: Cantilever beam tip bending deflection.

        Dimensions: Width b = 2.0 mm, Height h = 2.0 mm, Length L = 20.0 mm
        Material: E = 1000.0 MPa, nu = 0.3
        Boundary: Fixed root at x = 0 (u_x = u_y = u_z = 0)
        Load: Transverse tip load P = 1.0 N applied at x = 20.0 in -z direction.

        Analytical Solution (Euler-Bernoulli beam theory):
            - Second moment of area: I = b * h^3 / 12 = 2.0 * 8.0 / 12 = 4/3 mm^4 = 1.333333 mm^4
            - Theoretical tip deflection: delta = P * L^3 / (3 * E * I)
                                                = 1.0 * 8000.0 / (3 * 1000.0 * 4/3) = 2.0000 mm
        Tolerance: C3D10 quadratic tetrahedra tip deflection within 10% of theory.
                   (C3D4 linear tetrahedra evaluated for comparison).
        """
        beam_box = trimesh.creation.box(extents=(20.0, 2.0, 2.0))
        beam_box.apply_translation([10.0, 0.0, 0.0])  # x in [0, 20], y in [-1, 1], z in [-1, 1]

        def _run_beam(order: int, name: str) -> float:
            part = build_part_mesh(name, beam_box, size_far=0.8, size_near=0.8, order=order)
            model = MeshModel(parts=[part])

            x_root = np.where(np.abs(part.nodes[:, 0] - 0.0) < 1e-4)[0]
            x_tip = np.where(np.abs(part.nodes[:, 0] - 20.0) < 1e-4)[0]

            bc = BoundaryConditionSet(
                fixed_node_sets=[{"name": "FIX_ROOT", "part": name, "node_ids": x_root, "dofs": (1, 3)}],
                loads=[{"name": "LOAD_TIP", "part": name, "node_ids": x_tip, "force_N": (0.0, 0.0, -1.0)}]
            )

            mat = MaterialDef(name="MAT_BEAM", E=1000.0, nu=0.3)
            mat_assign = MaterialAssignment(
                element_material={name: np.zeros(len(part.elements), dtype=np.int64)},
                materials=[mat]
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                job_name = f"beam_{name}"
                inp_path = os.path.join(tmpdir, f"{job_name}.inp")
                write_inp(inp_path, model, {name: mat_assign}, bc, ties=[], apply_ccx_ordering=False)
                res = subprocess.run([CCX_PATH, job_name], cwd=tmpdir, capture_output=True, text=True)
                assert res.returncode == 0, f"CalculiX failed: {res.stdout}\n{res.stderr}"

                frd_path = os.path.join(tmpdir, f"{job_name}.frd")
                res_frd = read_frd_results(frd_path)
                tip_dz = [res_frd["displacements"][int(nid) + 1][2] for nid in x_tip]
                return float(np.mean(tip_dz))

        # C3D10 quadratic tetrahedral mesh
        tip_c3d10 = _run_beam(order=2, name="c3d10")
        err_c3d10_pct = abs(abs(tip_c3d10) - 2.0) / 2.0 * 100.0

        # C3D4 linear tetrahedral mesh
        tip_c3d4 = _run_beam(order=1, name="c3d4")
        err_c3d4_pct = abs(abs(tip_c3d4) - 2.0) / 2.0 * 100.0

        print(f"\n[Cantilever Beam Bending Results]")
        print(f"  Theoretical Euler-Bernoulli deflection = -2.0000 mm")
        print(f"  C3D10 tip deflection = {tip_c3d10:.4f} mm | Error = {err_c3d10_pct:.2f}%")
        print(f"  C3D4  tip deflection = {tip_c3d4:.4f} mm | Error = {err_c3d4_pct:.2f}% (stiff shear locking)")

        assert err_c3d10_pct <= 10.0, f"C3D10 deflection error ({err_c3d10_pct:.2f}%) exceeds 10.0% threshold."
        # Confirm that C3D4 is noticeably stiffer due to shear locking
        assert err_c3d4_pct > err_c3d10_pct

    def test_03_tie_contact_validation(self):
        """
        Benchmark 3: Multi-part assembly tied with CalculiX *TIE vs Monolithic block.

        Dimensions:
            - Monolithic Block: 10 x 10 x 20 mm (z in [0, 20])
            - Assembly: Part A (10 x 10 x 10 mm, z in [0, 10]) + Part B (10 x 10 x 10 mm, z in [10, 20])
        Boundary: Bottom face (z=0) clamped (u_x = u_y = u_z = 0)
        Load: Compressive resultant load of 100 N applied on top face (z=20).
        Constraint: Contact interface at z = 10 tied using *TIE (position_tolerance_mm = 0.2 mm).

        Criterion: Top displacement difference between TIE model and monolithic model within 3.0%.
        """
        mesh_sz = 1.5

        # 1. Monolithic reference block
        single_box = trimesh.creation.box(extents=(10.0, 10.0, 20.0))
        single_box.apply_translation([5.0, 5.0, 10.0])  # z in [0, 20]

        part_single = build_part_mesh("single", single_box, size_far=mesh_sz, size_near=mesh_sz, order=2)
        model_single = MeshModel(parts=[part_single])

        z_bot_s = np.where(np.abs(part_single.nodes[:, 2] - 0.0) < 1e-4)[0]
        z_top_s = np.where(np.abs(part_single.nodes[:, 2] - 20.0) < 1e-4)[0]

        bc_single = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX_BASE", "part": "single", "node_ids": z_bot_s, "dofs": (1, 3)}],
            loads=[{"name": "LOAD_TOP", "part": "single", "node_ids": z_top_s, "force_N": (0.0, 0.0, -100.0)}]
        )
        mat = MaterialDef(name="MAT_COMMON", E=1000.0, nu=0.3)
        assign_s = MaterialAssignment(
            element_material={"single": np.zeros(len(part_single.elements), dtype=np.int64)},
            materials=[mat]
        )

        with tempfile.TemporaryDirectory() as tmp_s:
            job_name = "single_block"
            inp_s = os.path.join(tmp_s, f"{job_name}.inp")
            write_inp(inp_s, model_single, {"single": assign_s}, bc_single, ties=[], apply_ccx_ordering=False)
            res_s = subprocess.run([CCX_PATH, job_name], cwd=tmp_s, capture_output=True, text=True)
            assert res_s.returncode == 0
            frd_s = read_frd_results(os.path.join(tmp_s, f"{job_name}.frd"))
            top_dz_single = float(np.mean([frd_s["displacements"][int(n) + 1][2] for n in z_top_s]))

        # 2. Two-block assembly tied with contact_pair_from_parts
        box_a = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        box_a.apply_translation([5.0, 5.0, 5.0])  # z in [0, 10]
        box_b = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        box_b.apply_translation([5.0, 5.0, 15.0])  # z in [10, 20]

        part_a = build_part_mesh("part_a", box_a, size_far=mesh_sz, size_near=mesh_sz, order=2)
        part_b = build_part_mesh("part_b", box_b, size_far=mesh_sz, size_near=mesh_sz, order=2)

        tie = contact_pair_from_parts(part_a, part_b, tol_mm=0.2, master="a", name="tie_ab")
        assert len(tie.slave_nodes) > 0, "No contact slave nodes detected at touching interface."
        assert len(tie.master_faces) > 0, "No contact master faces detected at touching interface."

        model_tie = MeshModel(parts=[part_a, part_b])
        z_bot_a = np.where(np.abs(part_a.nodes[:, 2] - 0.0) < 1e-4)[0]
        z_top_b = np.where(np.abs(part_b.nodes[:, 2] - 20.0) < 1e-4)[0]

        bc_tie = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX_BASE", "part": "part_a", "node_ids": z_bot_a, "dofs": (1, 3)}],
            loads=[{"name": "LOAD_TOP", "part": "part_b", "node_ids": z_top_b, "force_N": (0.0, 0.0, -100.0)}]
        )
        assign_a = MaterialAssignment(
            element_material={"part_a": np.zeros(len(part_a.elements), dtype=np.int64)},
            materials=[mat]
        )
        assign_b = MaterialAssignment(
            element_material={"part_b": np.zeros(len(part_b.elements), dtype=np.int64)},
            materials=[mat]
        )

        with tempfile.TemporaryDirectory() as tmp_t:
            job_name = "tied_assembly"
            inp_t = os.path.join(tmp_t, f"{job_name}.inp")
            write_inp(
                inp_t,
                model_tie,
                {"part_a": assign_a, "part_b": assign_b},
                bc_tie,
                ties=[tie],
                apply_ccx_ordering=False
            )
            res_t = subprocess.run([CCX_PATH, job_name], cwd=tmp_t, capture_output=True, text=True)
            assert res_t.returncode == 0, f"TIE solve failed: {res_t.stdout}\n{res_t.stderr}"

            frd_t = read_frd_results(os.path.join(tmp_t, f"{job_name}.frd"))
            b_offset = len(part_a.nodes)
            top_dz_tie = float(np.mean([frd_t["displacements"][b_offset + int(n) + 1][2] for n in z_top_b]))

        diff_pct = abs(top_dz_tie - top_dz_single) / abs(top_dz_single) * 100.0

        print(f"\n[TIE Constraint Validation Results]")
        print(f"  Monolithic block top dz = {top_dz_single:.6f} mm")
        print(f"  TIE assembly top dz     = {top_dz_tie:.6f} mm")
        print(f"  Displacement difference  = {diff_pct:.2f}% (TIE master faces: {len(tie.master_faces)}, slave nodes: {len(tie.slave_nodes)})")

        assert diff_pct <= 3.0, f"TIE displacement difference ({diff_pct:.2f}%) exceeds 3.0% threshold."

    def test_04_frd_reader_structure(self):
        """Verify result reader dictionary structure and parser functionality."""
        sample_frd_content = """    1C
    1Udental_fea
    1PSTEP                         1           1           1
  100CL  101 1.000000000           2                     0    1           1
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -1         1 1.00000E-03 2.00000E-03-3.00000E-03
 -1         2 0.00000E+00 0.00000E+00-5.00000E-04
 -3
    1PSTEP                         2           1           1
  100CL  101 1.000000000           2                     0    1           1
 -4  STRESS      6    1
 -5  SXX         1    4    1    1
 -5  SYY         1    4    2    2
 -5  SZZ         1    4    3    3
 -5  SXY         1    4    1    2
 -5  SYZ         1    4    2    3
 -5  SZX         1    4    3    1
 -1         1-1.00000E+00 2.00000E+00-3.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2-5.00000E-01 5.00000E-01-1.50000E+00 1.00000E-01 2.00000E-01 3.00000E-01
 -3
"""
        with tempfile.NamedTemporaryFile("w", suffix=".frd", delete=False) as f:
            f.write(sample_frd_content)
            temp_path = f.name

        try:
            res = read_frd_results(temp_path)
            disp = res["displacements"]
            stress = res["stresses"]

            assert 1 in disp and 2 in disp
            assert np.allclose(disp[1], [0.001, 0.002, -0.003])
            assert np.allclose(disp[2], [0.0, 0.0, -0.0005])

            assert 1 in stress and 2 in stress
            assert np.allclose(stress[1], [-1.0, 2.0, -3.0, 0.0, 0.0, 0.0])
            assert np.allclose(stress[2], [-0.5, 0.5, -1.5, 0.1, 0.2, 0.3])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
