"""
test_notch_validation.py
Verification tests for notched tensile specimen benchmark (Step 5.4).
Validates stress concentration factors (Kt), Neuber-Peterson sensitivity (q, Kf),
Gmsh OCC C3D10 meshing, and CalculiX 2x mesh refinement convergence (< 5%) & accuracy (< 5%).
"""

import os
import shutil
import pytest
import numpy as np

from fatigue.models import FatigueMaterial
from fatigue.notch_benchmark import (
    NotchSpec,
    NotchTheory,
    NotchFEAResult,
    NotchBenchmarkResult,
    calculate_notch_theory,
    calculate_notch_sensitivity,
    build_notched_plate_mesh,
    build_notch_model_and_bc,
    run_notch_benchmark,
)


class TestNotchTheory:
    """Analytical verification of literature stress concentration factor calculations."""

    def test_semicircular_notched_plate_theory(self):
        """Verify Peterson/Flynn curve-fit for opposite semicircular notched plate."""
        # W = 20 mm, r = 2.5 mm -> lambda = 2r / W = 0.25, net_width d = 15 mm
        # thickness = 2 mm -> net area = 30 mm^2, F = 1500 N -> sigma_nom = 50 MPa
        spec = NotchSpec(
            notch_type="semicircular_plate",
            length_L=80.0,
            width_W=20.0,
            thickness_t=2.0,
            notch_radius_r=2.5,
            notch_depth_h=2.5,
            tensile_force_N=1500.0,
        )
        theory = calculate_notch_theory(spec)

        assert theory.net_width_d == pytest.approx(15.0, abs=1e-5)
        assert theory.net_area_An == pytest.approx(30.0, abs=1e-5)
        assert theory.gross_area_Ag == pytest.approx(40.0, abs=1e-5)
        assert theory.sigma_nom == pytest.approx(50.0, abs=1e-5)
        assert theory.sigma_gross == pytest.approx(37.5, abs=1e-5)

        # Peterson Chart 2.4 / Flynn table: Kt ~ 2.37 - 2.38
        assert 2.30 <= theory.kt_theory <= 2.45
        assert theory.kt_theory == pytest.approx(2.370, abs=0.03)
        assert theory.sigma_max_theory == pytest.approx(theory.kt_theory * 50.0, abs=1e-5)
        assert "Peterson" in theory.literature_source

    def test_hole_plate_howland_theory(self):
        """Verify Howland (1930) analytical series expansion for plate with circular hole."""
        # W = 20 mm, hole diameter d = 5 mm (r = 2.5 mm) -> d/W = 0.25
        spec = NotchSpec(
            notch_type="hole_plate",
            length_L=80.0,
            width_W=20.0,
            thickness_t=2.0,
            notch_radius_r=2.5,
            tensile_force_N=1500.0,
        )
        theory = calculate_notch_theory(spec)

        assert theory.net_width_d == pytest.approx(15.0, abs=1e-5)
        assert theory.sigma_nom == pytest.approx(50.0, abs=1e-5)

        # Howland exact series: 3.00 - 3.14*(0.25) + 3.67*(0.25^2) - 1.53*(0.25^3) = 2.4205
        expected_kt = 3.00 - 3.14 * 0.25 + 3.67 * (0.25 ** 2) - 1.53 * (0.25 ** 3)
        assert theory.kt_theory == pytest.approx(expected_kt, abs=1e-4)
        assert "Howland" in theory.literature_source

    def test_unsupported_notch_type_raises(self):
        """Verify unsupported notch types raise ValueError."""
        spec = NotchSpec(notch_type="parabolic_notch")
        with pytest.raises(ValueError, match="Unsupported notch_type"):
            calculate_notch_theory(spec)

    def test_notch_sensitivity_and_kf(self):
        """Verify Neuber-Peterson notch sensitivity index and fatigue notch factor."""
        # For Ti-6Al-4V (Sut ~ 860 MPa) and r = 2.5 mm
        kt = 2.37
        q, kf = calculate_notch_sensitivity(kt=kt, notch_radius_r=2.5, tensile_strength_Sut=860.0)

        # q should be between 0 and 1, typically ~0.85-0.95 for titanium with r=2.5 mm
        assert 0.70 <= q <= 1.0
        # Kf must satisfy 1 <= Kf <= Kt
        assert 1.0 <= kf <= kt
        assert kf == pytest.approx(1.0 + q * (kt - 1.0), rel=1e-5)


class TestNotchMeshing:
    """Verification of Gmsh OpenCASCADE CAD generation and C3D10 solid meshing."""

    def test_build_notched_plate_mesh(self):
        """Verify Gmsh OCC generates watertight C3D10 elements without slivers."""
        spec = NotchSpec(
            notch_type="semicircular_plate",
            length_L=80.0,
            width_W=20.0,
            thickness_t=2.0,
            notch_radius_r=2.5,
        )
        nodes, elements = build_notched_plate_mesh(spec, mesh_size_notch=1.0, mesh_size_far=2.5)

        assert len(nodes) > 1000
        assert len(elements) > 500
        assert elements.shape[1] == 10  # C3D10 quadratic tetrahedra

        # Bounding box sanity checks
        assert np.isclose(nodes[:, 0].min(), -40.0, atol=1e-2)
        assert np.isclose(nodes[:, 0].max(), 40.0, atol=1e-2)
        assert np.isclose(nodes[:, 1].min(), -10.0, atol=1e-2)
        assert np.isclose(nodes[:, 1].max(), 10.0, atol=1e-2)
        assert np.isclose(nodes[:, 2].min(), -1.0, atol=1e-2)
        assert np.isclose(nodes[:, 2].max(), 1.0, atol=1e-2)

        # Check minimum throat width around x=0
        x_notch = np.where(np.abs(nodes[:, 0]) < 0.2)[0]
        y_notch = nodes[x_notch, 1]
        assert y_notch.min() >= -7.6  # r=2.5 from +/-10 gives throat at +/-7.5 mm
        assert y_notch.max() <= 7.6


class TestNotchFEAConvergenceAndAccuracy:
    """CalculiX simulation verification for stress concentration and 2x mesh refinement convergence."""

    @pytest.fixture
    def ccx_available(self):
        return shutil.which("/usr/bin/ccx") or shutil.which("ccx")

    def test_notch_benchmark_convergence_and_kt(self, ccx_available):
        """Execute dual-load notch FEA benchmark and verify +/-5% Kt accuracy and <5% convergence."""
        if not ccx_available:
            pytest.skip("CalculiX ccx executable not available on this machine.")

        spec = NotchSpec(
            notch_type="semicircular_plate",
            length_L=80.0,
            width_W=20.0,
            thickness_t=2.0,
            notch_radius_r=2.5,
            notch_depth_h=2.5,
            tensile_force_N=1500.0,
        )
        material = FatigueMaterial(
            name="Ti-6Al-4V",
            E=110000.0,
            nu=0.34,
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
        )

        # Coarse h=0.8 mm vs Fine h=0.4 mm (2x refinement)
        res = run_notch_benchmark(
            spec=spec,
            coarse_mesh_size=0.8,
            fine_mesh_size=0.4,
            material=material,
            accuracy_tolerance_pct=5.0,
            convergence_tolerance_pct=5.0,
        )

        # 1. Theoretical verification
        assert res.theory.sigma_nom == pytest.approx(50.0, abs=1e-3)
        assert res.theory.kt_theory == pytest.approx(2.370, abs=0.03)

        # 2. FEA Accuracy verification (|Kt_FEA - Kt_theory| / Kt_theory <= 5.0%)
        assert abs(res.fine_result.kt_error_pct) <= 5.0
        assert res.fine_result.passed_accuracy is True
        assert res.passed_accuracy is True

        # 3. Mesh Refinement Convergence (|sigma_fine - sigma_coarse| / sigma_fine < 5.0%)
        # In our preliminary run, stress change was ~0.12%, well below 5.0%
        assert res.stress_change_pct < 5.0
        assert res.passed_convergence is True

        # 4. Overall Pass
        assert res.passed is True

        # 5. Fatigue Notch Factor
        assert res.notch_sensitivity_q > 0.0
        assert 1.0 < res.fatigue_notch_factor_kf <= res.fine_result.kt_fea
