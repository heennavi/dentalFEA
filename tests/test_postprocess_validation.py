"""
test_postprocess_validation.py
Comprehensive quantitative validation of the Phase 4 post-processing pipeline against analytical solutions.

[Validation Benchmarks]
1. Uniaxial Compression of Homogeneous Cube:
   - 10 x 10 x 10 mm, E = 1000 MPa, nu = 0.3, Total load = 100 N
   - Post-process pipeline: parse_frd -> build_result_set -> part_statistics
   - Criterion: Mean top displacement within 5% of theoretical -0.010 mm.
                Mean vertical stress (SZZ) within 5% of -1.0 MPa.
                Mean von Mises stress ~ 1.0 MPa.
2. Static Reaction Equilibrium:
   - reaction_balance relative error within 0.5% (total reaction vs applied force).
3. Cantilever Beam Bending:
   - 2 x 2 x 20 mm, E = 1000 MPa, nu = 0.3, Tip transverse load P = 1.0 N
   - Analytical (Euler-Bernoulli): delta = PL^3 / (3EI) = 2.0 mm
   - Criterion: C3D10 tip deflection within 10% of theory via build_result_set.
4. Single-Element Pure Tension (C3D10):
   - 10-node quadratic tetrahedron under uniform uniaxial tensile strain field.
   - Criterion: All nodal SZZ and von Mises stresses uniformly equal to theoretical 1.0 MPa.
5. Parser Performance Benchmark:
   - Measures parsing time and throughput on result files.
"""

import os
import shutil
import tempfile
import subprocess
import time
import pytest
import numpy as np
import trimesh

from fea.models import MeshModel, MeshPart, MaterialDef, MaterialAssignment, BoundaryConditionSet
from meshing.gmsh_mesher import build_part_mesh, gmsh_tet10_to_ccx
from solver.calculix_writer import write_inp
from postprocess.frd_parser import parse_frd
from postprocess.result_fields import build_result_set, part_statistics, reaction_balance

CCX_PATH = shutil.which("ccx") or "/usr/bin/ccx"
HAS_CCX = os.path.isfile(CCX_PATH) and os.access(CCX_PATH, os.X_OK)


@pytest.mark.skipif(not HAS_CCX, reason="CalculiX solver binary (ccx) not available on this system.")
class TestPostprocessValidation:
    """Quantitative validation of the post-processing pipeline against analytical benchmarks."""

    def test_01_cube_compression_and_reaction_balance(self):
        """
        Benchmark 1 & 2: Uniaxial compression of homogeneous cube & force equilibrium.

        Dimensions: 10 x 10 x 10 mm
        Material: E = 1000.0 MPa, nu = 0.3
        Boundary: Bottom face (z=0) roller constrained (u_z = 0) with isostatic in-plane restraints.
        Load: Top face (z=10) compressive load of 100.0 N (F_z = -100.0 N).

        Analytical Solution:
            - Vertical Stress: sigma_zz = F / A = -100.0 / 100.0 = -1.0 MPa
            - Top Deflection: delta_z = sigma_zz * L / E = -1.0 * 10.0 / 1000.0 = -0.010 mm
            - Reaction Force: R_z = +100.0 N (Balance residual < 0.5%)
        """
        box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        box.apply_translation([5.0, 5.0, 5.0])  # x in [0, 10], y in [0, 10], z in [0, 10]

        part = build_part_mesh("cube", box, size_far=1.5, size_near=1.5, order=2)
        model = MeshModel(parts=[part])

        z_bot = np.where(np.abs(part.nodes[:, 2] - 0.0) < 1e-4)[0]
        z_top = np.where(np.abs(part.nodes[:, 2] - 10.0) < 1e-4)[0]

        # Isostatic 3-2-1 kinematic boundary conditions
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
                title="Validation Benchmark 1: Cube Compression Postprocess Validation",
                apply_ccx_ordering=False,
            )

            res = subprocess.run([CCX_PATH, job_name], cwd=tmpdir, capture_output=True, text=True)
            assert res.returncode == 0, f"CalculiX solver failed: {res.stdout}\n{res.stderr}"

            frd_path = os.path.join(tmpdir, f"{job_name}.frd")
            assert os.path.isfile(frd_path)

            # --- Post-process pipeline ---
            t0 = time.perf_counter()
            frd_data = parse_frd(frd_path)
            t_parse = time.perf_counter() - t0

            t0 = time.perf_counter()
            rs = build_result_set(frd_data, model)
            t_build = time.perf_counter() - t0

            # 1) Top displacement evaluation
            top_dz = rs.u[z_top, 2]
            mean_top_dz = float(np.mean(top_dz))
            err_disp_pct = abs(mean_top_dz - (-0.010)) / 0.010 * 100.0

            # 2) SZZ stress evaluation (mid-height region [3, 7] mm away from boundary singularities)
            mid_mask = (part.nodes[:, 2] > 3.0) & (part.nodes[:, 2] < 7.0)
            mid_szz = rs.stress6[mid_mask, 2]
            mean_mid_szz = float(np.mean(mid_szz))
            err_szz_pct = abs(mean_mid_szz - (-1.0)) / 1.0 * 100.0

            # 3) von Mises evaluation
            mid_vm = rs.von_mises[mid_mask]
            mean_mid_vm = float(np.mean(mid_vm))
            err_vm_pct = abs(mean_mid_vm - 1.0) / 1.0 * 100.0

            # 4) Part statistics
            stats = part_statistics(rs, model, "cube")
            assert stats["num_nodes"] == len(part.nodes)
            assert stats["part_name"] == "cube"

            # 5) Reaction force balance
            bal = reaction_balance(rs, bc, model)
            assert bal is not None, "Reaction balance calculation returned None."
            rel_error_pct = bal["relative_error"] * 100.0

            print(f"\n[Cube Compression Postprocess Validation]")
            print(f"  Parse time: {t_parse*1000:.2f} ms, Build time: {t_build*1000:.2f} ms (Nodes: {len(part.nodes)})")
            print(f"  Theory delta_z = -0.010000 mm | Computed delta_z = {mean_top_dz:.6f} mm | Error = {err_disp_pct:.2f}%")
            print(f"  Theory sigma_zz = -1.000000 MPa | Computed mid sigma_zz = {mean_mid_szz:.6f} MPa | Error = {err_szz_pct:.2f}%")
            print(f"  Theory von_mises = 1.000000 MPa | Computed mid von_mises = {mean_mid_vm:.6f} MPa | Error = {err_vm_pct:.2f}%")
            print(f"  Reaction Balance: Applied={bal['applied_mag']:.4f} N, Reaction={bal['reaction_mag']:.4f} N, Rel Error={rel_error_pct:.4f}%")

            assert err_disp_pct <= 5.0, f"Displacement error ({err_disp_pct:.2f}%) exceeds 5.0% threshold."
            assert err_szz_pct <= 5.0, f"Midplane vertical stress error ({err_szz_pct:.2f}%) exceeds 5.0% threshold."
            assert err_vm_pct <= 5.0, f"Midplane von Mises stress error ({err_vm_pct:.2f}%) exceeds 5.0% threshold."
            assert rel_error_pct <= 0.5, f"Reaction balance relative error ({rel_error_pct:.4f}%) exceeds 0.5% threshold."

    def test_02_cantilever_beam_bending(self):
        """
        Benchmark 3: Cantilever beam tip bending deflection via post-processing pipeline.

        Dimensions: Width b = 2.0 mm, Height h = 2.0 mm, Length L = 20.0 mm
        Material: E = 1000.0 MPa, nu = 0.3
        Boundary: Fixed root at x = 0 (u_x = u_y = u_z = 0)
        Load: Transverse tip load P = 1.0 N applied at x = 20.0 in -z direction.

        Analytical Solution (Euler-Bernoulli beam theory):
            - Second moment of area: I = b * h^3 / 12 = 2.0 * 8.0 / 12 = 4/3 mm^4
            - Theoretical tip deflection: delta = P * L^3 / (3 * E * I)
                                                = 1.0 * 8000.0 / (3 * 1000.0 * 4/3) = 2.0000 mm
        Tolerance: C3D10 quadratic tetrahedra tip deflection within 10% of theory.
        """
        beam_box = trimesh.creation.box(extents=(20.0, 2.0, 2.0))
        beam_box.apply_translation([10.0, 0.0, 0.0])  # x in [0, 20], y in [-1, 1], z in [-1, 1]

        part = build_part_mesh("beam", beam_box, size_far=0.8, size_near=0.8, order=2)
        model = MeshModel(parts=[part])

        x_root = np.where(np.abs(part.nodes[:, 0] - 0.0) < 1e-4)[0]
        x_tip = np.where(np.abs(part.nodes[:, 0] - 20.0) < 1e-4)[0]

        bc = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX_ROOT", "part": "beam", "node_ids": x_root, "dofs": (1, 3)}],
            loads=[{"name": "LOAD_TIP", "part": "beam", "node_ids": x_tip, "force_N": (0.0, 0.0, -1.0)}]
        )

        mat = MaterialDef(name="MAT_BEAM", E=1000.0, nu=0.3)
        mat_assign = MaterialAssignment(
            element_material={"beam": np.zeros(len(part.elements), dtype=np.int64)},
            materials=[mat]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            job_name = "beam_bending"
            inp_path = os.path.join(tmpdir, f"{job_name}.inp")
            write_inp(inp_path, model, {"beam": mat_assign}, bc, ties=[], apply_ccx_ordering=False)
            res = subprocess.run([CCX_PATH, job_name], cwd=tmpdir, capture_output=True, text=True)
            assert res.returncode == 0, f"CalculiX solver failed: {res.stdout}\n{res.stderr}"

            frd_path = os.path.join(tmpdir, f"{job_name}.frd")
            frd_data = parse_frd(frd_path)
            rs = build_result_set(frd_data, model)

            tip_dz = rs.u[x_tip, 2]
            computed_tip_dz = float(np.mean(tip_dz))
            err_tip_pct = abs(abs(computed_tip_dz) - 2.0) / 2.0 * 100.0

            print(f"\n[Cantilever Beam Postprocess Validation]")
            print(f"  Theory Euler-Bernoulli deflection = -2.0000 mm")
            print(f"  Postprocess computed deflection = {computed_tip_dz:.4f} mm | Error = {err_tip_pct:.2f}%")

            assert err_tip_pct <= 10.0, f"C3D10 deflection error ({err_tip_pct:.2f}%) exceeds 10.0% threshold."

    def test_03_single_element_pure_tension(self):
        """
        Benchmark 4: Pure uniform tension on a single C3D10 element.

        Verifies that:
        1. CalculiX 10-node tetrahedral connectivity orientation (C3D10) operates without Jacobian error.
        2. ASCII .frd fixed-width parser and build_result_set correctly align nodal stresses without permutation error.
        3. A linear displacement field (axial strain eps_zz = 0.001, nu = 0.0) yields an exact uniform
           stress field sigma_zz = 1.0 MPa and von_mises = 1.0 MPa across all 10 nodes.
        """
        v0 = [0.0, 0.0, 0.0]
        v1 = [10.0, 0.0, 0.0]
        v2 = [0.0, 10.0, 0.0]
        v3 = [0.0, 0.0, 10.0]

        nodes = np.array([
            v0,                   # 0: corner 1
            v1,                   # 1: corner 2
            v2,                   # 2: corner 3
            v3,                   # 3: corner 4
            [5.0, 0.0, 0.0],      # 4: mid 1-2
            [5.0, 5.0, 0.0],      # 5: mid 2-3
            [0.0, 5.0, 0.0],      # 6: mid 3-1
            [0.0, 0.0, 5.0],      # 7: mid 1-4
            [5.0, 0.0, 5.0],      # 8: mid 2-4
            [0.0, 5.0, 5.0],      # 9: mid 3-4
        ], dtype=np.float64)

        ccx_elem = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)

        part = MeshPart(name="single_tet", nodes=nodes, elements=ccx_elem, element_type="C3D10")
        model = MeshModel(parts=[part])

        mat = MaterialDef(name="MAT_UNIAXIAL", E=1000.0, nu=0.0)
        mat_assign = MaterialAssignment(
            element_material={"single_tet": np.zeros(1, dtype=np.int64)},
            materials=[mat]
        )
        bc = BoundaryConditionSet(fixed_node_sets=[], loads=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            job_name = "pure_tension_c3d10"
            inp_path = os.path.join(tmpdir, f"{job_name}.inp")
            write_inp(inp_path, model, {"single_tet": mat_assign}, bc, ties=[], apply_ccx_ordering=False)

            bc_text = """*BOUNDARY
1, 1, 3, 0.0
2, 2, 3, 0.0
3, 1, 3, 0.0
4, 1, 2, 0.0
4, 3, 3, 0.010
5, 3, 3, 0.0
6, 3, 3, 0.0
7, 3, 3, 0.0
8, 3, 3, 0.005
9, 3, 3, 0.005
10, 3, 3, 0.005
"""
            with open(inp_path, "r") as f:
                content = f.read()
            content = content.replace("*STEP\n*STATIC", f"*STEP\n*STATIC\n{bc_text}")
            with open(inp_path, "w") as f:
                f.write(content)

            res = subprocess.run([CCX_PATH, job_name], cwd=tmpdir, capture_output=True, text=True)
            assert res.returncode == 0, f"CalculiX failed: {res.stdout}\n{res.stderr}"

            frd_path = os.path.join(tmpdir, f"{job_name}.frd")
            frd_data = parse_frd(frd_path)
            rs = build_result_set(frd_data, model)

            szz_vals = rs.stress6[:, 2]
            vm_vals = rs.von_mises

            print(f"\n[Single C3D10 Pure Tension Validation]")
            print(f"  All 10 nodal SZZ stresses: {szz_vals}")
            print(f"  All 10 nodal von Mises stresses: {vm_vals}")

            assert np.allclose(szz_vals, 1.0, atol=1e-3, rtol=1e-3), (
                f"Nodal SZZ values deviate from 1.0 MPa: {szz_vals}"
            )
            assert np.allclose(vm_vals, 1.0, atol=1e-3, rtol=1e-3), (
                f"Nodal von Mises values deviate from 1.0 MPa: {vm_vals}"
            )
            for c in [0, 1, 3, 4, 5]:
                assert np.allclose(rs.stress6[:, c], 0.0, atol=1e-3), (
                    f"Component {c} stress should be zero but got {rs.stress6[:, c]}"
                )

    def test_04_parser_performance_benchmark(self):
        """
        Benchmark 5: Measure fixed-width ASCII .frd parsing speed and memory throughput.
        """
        n_nodes = 5000
        lines = [
            "    1C",
            "    1Ubenchmark",
            "    2C",
        ]
        rng = np.random.RandomState(42)
        coords = rng.uniform(-10.0, 10.0, size=(n_nodes, 3))
        for i in range(n_nodes):
            x, y, z = coords[i]
            lines.append(f" -1{i+1:10d}{x:12.5f}{y:12.5f}{z:12.5f}")
        lines.append(" -3")

        # DISP block
        lines.extend([
            "    1PSTEP                         1           1           1",
            "  100CL  101 1.000000000           2                     0    1           1",
            " -4  DISP        4    1",
            " -5  D1          1    2    1    0",
            " -5  D2          1    2    2    0",
            " -5  D3          1    2    3    0",
            " -5  ALL         1    2    0    0",
        ])
        disps = rng.normal(0.0, 0.01, size=(n_nodes, 3))
        for i in range(n_nodes):
            ux, uy, uz = disps[i]
            lines.append(f" -1{i+1:10d}{ux:12.5E}{uy:12.5E}{uz:12.5E}")
        lines.append(" -3")

        # STRESS block
        lines.extend([
            "    1PSTEP                         2           1           1",
            "  100CL  101 1.000000000           2                     0    1           1",
            " -4  STRESS      6    1",
            " -5  SXX         1    4    1    1",
            " -5  SYY         1    4    2    2",
            " -5  SZZ         1    4    3    3",
            " -5  SXY         1    4    1    2",
            " -5  SYZ         1    4    2    3",
            " -5  SZX         1    4    3    1",
        ])
        stresses = rng.uniform(-50.0, 50.0, size=(n_nodes, 6))
        for i in range(n_nodes):
            s = stresses[i]
            lines.append(f" -1{i+1:10d}{s[0]:12.5E}{s[1]:12.5E}{s[2]:12.5E}{s[3]:12.5E}{s[4]:12.5E}{s[5]:12.5E}")
        lines.append(" -3")

        with tempfile.NamedTemporaryFile("w", suffix=".frd", delete=False) as f:
            f.write("\n".join(lines) + "\n")
            frd_file = f.name

        try:
            t0 = time.perf_counter()
            frd_data = parse_frd(frd_file)
            elapsed = time.perf_counter() - t0

            nodes_per_sec = n_nodes / elapsed if elapsed > 0 else float("inf")
            print(f"\n[Parser Benchmark Results]")
            print(f"  Parsed {n_nodes:,} nodes with 2 result blocks in {elapsed*1000:.2f} ms")
            print(f"  Throughput: {nodes_per_sec:,.0f} nodes/sec")

            assert len(frd_data.node_ids) == n_nodes
            assert len(frd_data.blocks) == 2
            assert elapsed < 1.0, f"Parser took too long: {elapsed:.2f} s for 5000 nodes"
        finally:
            if os.path.exists(frd_file):
                os.remove(frd_file)
