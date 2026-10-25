"""
test_cantilever_validation.py
Step 5.3 Verification Benchmark: Cantilever beam analytical FEA validation with CalculiX.
Validates that root surface stresses and fatigue safety factors match classical beam theory
(sigma = M*c/I) and analytical hand calculations within +/- 3.0%.
"""

import math
import os
import shutil
import pytest
import numpy as np

from fatigue.cantilever_benchmark import (
    calculate_cantilever_theory,
    build_cantilever_model_and_bc,
    run_cantilever_benchmark,
    CantileverTheory,
    CantileverBenchmarkResult,
)
from fatigue.iso14801 import (
    ISO14801Config,
    ISO14801LoadMode,
    run_iso14801_pipeline,
)
from fatigue.models import (
    FatigueMaterial,
    MeanStressCorrectionMethod,
    StressMeasureType,
)

CCX_PATH = shutil.which("ccx") or "/usr/bin/ccx"
HAS_CCX = os.path.isfile(CCX_PATH) and os.access(CCX_PATH, os.X_OK)


class TestCantileverTheoryUnit:
    """Unit tests for analytical beam theory hand calculations."""

    def test_cantilever_analytical_formulas(self):
        """Verify exact closed-form calculations for cantilever beam bending."""
        # L = 100 mm, b = 10 mm, h = 10 mm, F_max = 1000 N, F_min = 100 N
        mat = FatigueMaterial(
            name="Ti-6Al-4V",
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
        )

        theory = calculate_cantilever_theory(
            length_L=100.0,
            width_b=10.0,
            height_h=10.0,
            force_max=1000.0,
            force_min=100.0,
            material=mat,
        )

        # I = 10 * 10^3 / 12 = 833.3333 mm^4
        assert math.isclose(theory.moment_of_inertia_I, 10000.0 / 12.0, rel_tol=1e-6)
        # Z = I / (h/2) = 166.6667 mm^3
        assert math.isclose(theory.section_modulus_Z, 1000.0 / 6.0, rel_tol=1e-6)
        # M_max = 1000 * 100 = 100,000 N*mm
        assert math.isclose(theory.moment_max, 100000.0, rel_tol=1e-6)
        # M_min = 100 * 100 = 10,000 N*mm
        assert math.isclose(theory.moment_min, 10000.0, rel_tol=1e-6)
        # sigma_max = 100000 / (1000/6) = 600.0 MPa
        assert math.isclose(theory.sigma_max, 600.0, rel_tol=1e-6)
        # sigma_min = 10000 / (1000/6) = 60.0 MPa
        assert math.isclose(theory.sigma_min, 60.0, rel_tol=1e-6)
        # sigma_a = (600 - 60)/2 = 270.0 MPa
        assert math.isclose(theory.sigma_a, 270.0, rel_tol=1e-6)
        # sigma_m = (600 + 60)/2 = 330.0 MPa
        assert math.isclose(theory.sigma_m, 330.0, rel_tol=1e-6)

        # Goodman: 1 / FS = 270 / 510 + 330 / 860 = 0.5294118 + 0.3837209 = 0.9131327 -> FS = 1.095131
        expected_fs_goodman = 1.0 / (270.0 / 510.0 + 330.0 / 860.0)
        assert math.isclose(theory.fs_goodman, expected_fs_goodman, rel_tol=1e-5)

        # Soderberg: 1 / FS = 270 / 510 + 330 / 795 = 0.5294118 + 0.4150943 = 0.944506 -> FS = 1.05875
        expected_fs_soderberg = 1.0 / (270.0 / 510.0 + 330.0 / 795.0)
        assert math.isclose(theory.fs_soderberg, expected_fs_soderberg, rel_tol=1e-5)

        # Gerber: (FS * 270/510) + (FS * 330/860)^2 = 1
        # a = (330/860)^2, b = 270/510
        a_g = (330.0 / 860.0) ** 2
        b_g = 270.0 / 510.0
        expected_fs_gerber = (-b_g + math.sqrt(b_g ** 2 + 4.0 * a_g)) / (2.0 * a_g)
        assert math.isclose(theory.fs_gerber, expected_fs_gerber, rel_tol=1e-5)

    def test_cantilever_model_and_bc_setup(self):
        """Verify cantilever mesh generation and Poisson-free boundary conditions."""
        model, mat_assign, bc_max, bc_min = build_cantilever_model_and_bc(
            length_L=100.0,
            width_b=10.0,
            height_h=10.0,
            mesh_size=3.0,
            force_max=1000.0,
            force_min=100.0,
        )

        assert len(model.parts) == 1
        part = model.parts[0]
        assert part.name == "beam"
        assert part.element_type == "C3D10"
        assert len(part.nodes) > 1000
        assert len(part.elements) > 500

        # Boundary condition sets
        assert len(bc_max.fixed_node_sets) == 3
        names = [f["name"] for f in bc_max.fixed_node_sets]
        assert "FIX_X_PLANE" in names
        assert "FIX_YZ_CENTER" in names
        assert "FIX_Z_ROLL" in names

        # Tip load
        assert len(bc_max.loads) == 1
        assert bc_max.loads[0]["force_N"] == (0.0, 0.0, -1000.0)
        assert bc_min.loads[0]["force_N"] == (0.0, 0.0, -100.0)


@pytest.mark.skipif(not HAS_CCX, reason="CalculiX solver binary (ccx) not available on this system.")
class TestCantileverCalculixValidation:
    """Step 5.3 Acceptance Criteria Validation: Full CalculiX FEA & Fatigue Pipeline."""

    def test_01_cantilever_beam_fea_and_fatigue_within_3_percent(self):
        """
        [Step 5.3 Core Acceptance Criteria]
        - CalculiX FEA runs on cantilever beam with F_max = 1000 N, F_min = 100 N.
        - .frd cyclic stresses extracted and evaluated with evaluate_fatigue.
        - Root surface stress matches sigma = M*c/I (600.0 MPa) within +/- 3.0%.
        - Fatigue safety factor matches hand calculation (1.0951) within +/- 3.0%.
        """
        result = run_cantilever_benchmark(
            length_L=100.0,
            width_b=10.0,
            height_h=10.0,
            force_max=1000.0,
            force_min=100.0,
            mesh_size=2.5,
            tolerance_pct=3.0,
            ccx_path=CCX_PATH,
        )

        # Print summary for test log visibility
        print("\n" + "=" * 65)
        print(" [Step 5.3 Cantilever Beam FEA Fatigue Benchmark Results]")
        print("=" * 65)
        print(f" Analytical sigma_max (theory): {result.theory.sigma_max:.2f} MPa")
        print(f" FEA root sigma_max (mean):     {result.fea_sigma_max_mean:.2f} MPa")
        print(f" Stress Error (mean):           {result.mean_stress_error_pct:.2f}%")
        print(f" Stress Error (max):            {result.max_stress_error_pct:.2f}% (Threshold: <= 3.0%)")
        print("-" * 65)
        print(f" Analytical FS Goodman (theory):{result.theory.fs_goodman:.4f}")
        print(f" FEA root FS Goodman (mean):    {result.fea_fs_goodman_mean:.4f}")
        print(f" FS Error (mean):               {result.mean_fs_error_pct:.2f}%")
        print(f" FS Error (max):                {result.max_fs_error_pct:.2f}% (Threshold: <= 3.0%)")
        print("-" * 65)
        print(f" Surface Nodes Evaluated:       {result.num_surface_nodes_evaluated}")
        print(f" Stress Criterion Passed:       {result.passed_stress}")
        print(f" Safety Factor Criterion Passed:{result.passed_safety_factor}")
        print(f" Overall Benchmark Status:      {'PASS' if result.passed else 'FAIL'}")
        print("=" * 65)

        # Acceptance Criteria Assertions
        assert result.passed_stress is True, (
            f"Root surface stress max error ({result.max_stress_error_pct:.2f}%) exceeds 3.0% threshold."
        )
        assert result.passed_safety_factor is True, (
            f"Root surface safety factor max error ({result.max_fs_error_pct:.2f}%) exceeds 3.0% threshold."
        )
        assert result.passed is True

        # Mean errors are typically well under 1.0%
        assert result.mean_stress_error_pct < 2.0
        assert result.mean_fs_error_pct < 2.0

    def test_02_cantilever_soderberg_and_gerber_validation(self):
        """
        Verify that cantilever benchmark passes with Soderberg and Gerber theories within +/- 3.0%.
        """
        # 1. Soderberg
        res_soderberg = run_cantilever_benchmark(
            length_L=100.0,
            width_b=10.0,
            height_h=10.0,
            force_max=1000.0,
            force_min=100.0,
            mesh_size=2.5,
            mean_stress_method=MeanStressCorrectionMethod.SODERBERG,
            tolerance_pct=3.0,
            ccx_path=CCX_PATH,
        )
        assert res_soderberg.passed is True
        assert res_soderberg.max_fs_error_pct <= 3.0
        assert math.isclose(res_soderberg.theory_fs, res_soderberg.theory.fs_soderberg, rel_tol=1e-6)

        # 2. Gerber
        res_gerber = run_cantilever_benchmark(
            length_L=100.0,
            width_b=10.0,
            height_h=10.0,
            force_max=1000.0,
            force_min=100.0,
            mesh_size=2.5,
            mean_stress_method=MeanStressCorrectionMethod.GERBER,
            tolerance_pct=3.0,
            ccx_path=CCX_PATH,
        )
        assert res_gerber.passed is True
        assert res_gerber.max_fs_error_pct <= 3.0
        assert math.isclose(res_gerber.theory_fs, res_gerber.theory.fs_gerber, rel_tol=1e-6)


class TestISO14801Pipeline:
    """Unit and integration tests for ISO 14801 dynamic load case configuration & pipeline."""

    def test_iso14801_config_and_vectors(self):
        """Verify ISO 14801 standard angle decomposition and load case generation."""
        cfg = ISO14801Config(
            load_max_N=250.0,
            stress_ratio_R=0.1,
            angle_deg=30.0,
            mean_stress_method=MeanStressCorrectionMethod.GOODMAN,
        )

        assert cfg.load_min_N == 25.0
        assert math.isclose(cfg.angle_rad, math.pi / 6.0, rel_tol=1e-6)

        # 30 deg force vector along axial (+Z) and transverse (+X)
        # Fx = -250 * sin(30) = -125.0 N
        # Fz = -250 * cos(30) = -216.50635 N
        f_vec = cfg.calculate_force_vector(250.0, axial_axis=(0, 0, 1), transverse_axis=(1, 0, 0))
        assert math.isclose(f_vec[0], -125.0, rel_tol=1e-5)
        assert math.isclose(f_vec[1], 0.0, abs_tol=1e-5)
        assert math.isclose(f_vec[2], -250.0 * math.cos(math.radians(30)), rel_tol=1e-5)

        # Vector norm matches 250 N
        assert math.isclose(np.linalg.norm(f_vec), 250.0, rel_tol=1e-5)

        fatigue_case = cfg.to_fatigue_load_case()
        assert fatigue_case.stress_ratio_R == 0.1
        assert fatigue_case.target_cycles == 5.0e6
        assert fatigue_case.mean_stress_method == MeanStressCorrectionMethod.GOODMAN

    @pytest.mark.skipif(not HAS_CCX, reason="CalculiX solver binary (ccx) not available on this system.")
    def test_iso14801_pipeline_dual_and_ratio_execution(self):
        """
        Verify end-to-end execution of run_iso14801_pipeline using CalculiX.
        Tests both DUAL_SIMULATION and STRESS_RATIO_R modes on a structural test part.
        """
        # Create a small cylindrical or box model
        from fea.models import MeshModel, MaterialAssignment, MaterialDef
        from meshing.gmsh_mesher import build_part_mesh
        import trimesh

        box = trimesh.creation.box(extents=(30.0, 6.0, 6.0))
        box.apply_translation([15.0, 0.0, 0.0])  # along X

        part = build_part_mesh("implant", box, size_far=2.0, size_near=1.5, order=2)
        model = MeshModel(parts=[part])

        # Clamping at x = 0 (embedding level)
        x_clamp = np.where(np.abs(part.nodes[:, 0] - 0.0) < 1e-4)[0]
        # Loading at tip x = 30
        x_load = np.where(np.abs(part.nodes[:, 0] - 30.0) < 1e-4)[0]

        fixed_sets = [{"name": "FIX_EMBED", "part": "implant", "node_ids": x_clamp, "dofs": (1, 3)}]

        fatigue_mat = FatigueMaterial(
            name="Ti-6Al-4V",
            E=110000.0,
            nu=0.34,
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
        )

        materials = {
            "implant": MaterialAssignment(
                element_material={"implant": np.zeros(len(part.elements), dtype=np.int64)},
                materials=[MaterialDef(name="Ti-6Al-4V", E=110000.0, nu=0.34)],
            )
        }

        # 1. Test DUAL_SIMULATION mode
        cfg_dual = ISO14801Config(
            load_max_N=100.0,
            stress_ratio_R=0.1,
            angle_deg=30.0,
            mode=ISO14801LoadMode.DUAL_SIMULATION,
        )

        res_dual = run_iso14801_pipeline(
            model=model,
            materials=materials,
            target_part_name="implant",
            fatigue_material=fatigue_mat,
            config=cfg_dual,
            fixed_node_sets=fixed_sets,
            loading_node_ids=x_load,
            axial_axis=(1.0, 0.0, 0.0),
            transverse_axis=(0.0, 0.0, 1.0),
            ccx_path=CCX_PATH,
        )

        assert len(res_dual.node_indices) == len(part.nodes)
        assert res_dual.min_safety_factor > 0.0
        assert np.all(res_dual.sigma_max >= res_dual.sigma_min)

        # 2. Test STRESS_RATIO_R mode
        cfg_ratio = ISO14801Config(
            load_max_N=100.0,
            stress_ratio_R=0.1,
            angle_deg=30.0,
            mode=ISO14801LoadMode.STRESS_RATIO_R,
        )

        res_ratio = run_iso14801_pipeline(
            model=model,
            materials=materials,
            target_part_name="implant",
            fatigue_material=fatigue_mat,
            config=cfg_ratio,
            fixed_node_sets=fixed_sets,
            loading_node_ids=x_load,
            axial_axis=(1.0, 0.0, 0.0),
            transverse_axis=(0.0, 0.0, 1.0),
            ccx_path=CCX_PATH,
        )

        assert len(res_ratio.node_indices) == len(part.nodes)
        # Linear elasticity: dual simulation and ratio derivation should be practically identical
        np.testing.assert_allclose(res_dual.safety_factor, res_ratio.safety_factor, rtol=1e-2)
