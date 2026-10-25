"""
cantilever_benchmark.py
Analytical benchmark and CalculiX validation pipeline for cantilever beam bending.
Validates FEA stress and high-cycle fatigue evaluation against classical beam theory (sigma = M*c/I).
"""

import os
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import trimesh

from fea.models import MeshModel, MeshPart, MaterialDef, MaterialAssignment, BoundaryConditionSet
from meshing.gmsh_mesher import build_part_mesh
from solver.calculix_writer import write_inp
from .models import (
    FatigueMaterial,
    FatigueLoadCase,
    FatigueResult,
    MeanStressCorrectionMethod,
    StressMeasureType,
)
from .evaluate import evaluate_fatigue
from .io_frd import extract_cyclic_stresses_from_two_frds


@dataclass
class CantileverTheory:
    """Theoretical analytical quantities for a rectangular cantilever beam under tip point load."""
    length_L: float
    width_b: float
    height_h: float
    force_max: float
    force_min: float
    stress_ratio_R: float
    moment_of_inertia_I: float
    section_modulus_Z: float
    moment_max: float
    moment_min: float
    sigma_max: float
    sigma_min: float
    sigma_a: float
    sigma_m: float
    fs_goodman: float
    fs_soderberg: float
    fs_gerber: float


@dataclass
class CantileverBenchmarkResult:
    """Quantitative results from the cantilever beam FEA fatigue validation benchmark."""
    theory: CantileverTheory
    passed_stress: bool
    passed_safety_factor: bool
    max_stress_error_pct: float
    mean_stress_error_pct: float
    max_fs_error_pct: float
    mean_fs_error_pct: float
    fea_sigma_max_mean: float
    fea_sigma_min_mean: float
    fea_fs_goodman_mean: float
    num_surface_nodes_evaluated: int
    mean_stress_method: MeanStressCorrectionMethod = MeanStressCorrectionMethod.GOODMAN
    fea_fs_mean: float = 0.0
    theory_fs: float = 0.0
    tolerance_pct: float = 3.0

    @property
    def passed(self) -> bool:
        """Overall benchmark pass status against acceptance criteria."""
        return self.passed_stress and self.passed_safety_factor


def calculate_cantilever_theory(
    length_L: float = 100.0,
    width_b: float = 10.0,
    height_h: float = 10.0,
    force_max: float = 1000.0,
    force_min: float = 100.0,
    material: Optional[FatigueMaterial] = None,
) -> CantileverTheory:
    """Calculate exact analytical beam bending stresses and fatigue safety factors.

    Formulas:
        I = b * h^3 / 12
        Z = I / (h / 2) = b * h^2 / 6
        M_max = force_max * L
        M_min = force_min * L
        sigma_max = M_max / Z
        sigma_min = M_min / Z
        sigma_a = (sigma_max - sigma_min) / 2
        sigma_m = (sigma_max + sigma_min) / 2
        1 / FS_goodman = sigma_a / Se + sigma_m / Sut
        1 / FS_soderberg = sigma_a / Se + sigma_m / Sy
        FS_gerber: quadratic root of (FS*sigma_a/Se) + (FS*sigma_m/Sut)^2 = 1

    Args:
        length_L: Beam span in mm.
        width_b: Beam cross-section width (along Y) in mm.
        height_h: Beam cross-section height (along Z) in mm.
        force_max: Magnitude of maximum transverse tip load in N.
        force_min: Magnitude of minimum transverse tip load in N.
        material: FatigueMaterial instance (defaults to Ti-6Al-4V if None).

    Returns:
        CantileverTheory dataclass with all analytical reference values.
    """
    if material is None:
        material = FatigueMaterial(
            name="Ti-6Al-4V",
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
        )

    f_max = float(abs(force_max))
    f_min = float(abs(force_min))
    if f_max < f_min:
        f_max, f_min = f_min, f_max

    R = f_min / f_max if f_max > 0.0 else 0.0

    I = width_b * (height_h ** 3) / 12.0
    c = height_h / 2.0
    Z = I / c

    M_max = f_max * length_L
    M_min = f_min * length_L

    sigma_max = M_max / Z
    sigma_min = M_min / Z
    sigma_a = 0.5 * (sigma_max - sigma_min)
    sigma_m = 0.5 * (sigma_max + sigma_min)

    Se = material.fatigue_limit_Se
    Sut = material.tensile_strength_Sut
    Sy = material.yield_strength_Sy

    # Goodman
    denom_goodman = (sigma_a / Se) + (sigma_m / Sut)
    fs_goodman = 1.0 / denom_goodman if denom_goodman > 0.0 else float("inf")

    # Soderberg
    denom_soderberg = (sigma_a / Se) + (sigma_m / Sy)
    fs_soderberg = 1.0 / denom_soderberg if denom_soderberg > 0.0 else float("inf")

    # Gerber: a * FS^2 + b * FS - 1 = 0
    a_g = (sigma_m / Sut) ** 2
    b_g = sigma_a / Se
    if a_g > 1e-12:
        disc = b_g ** 2 + 4.0 * a_g
        fs_gerber = (-b_g + np.sqrt(disc)) / (2.0 * a_g)
    else:
        fs_gerber = 1.0 / b_g if b_g > 0.0 else float("inf")

    return CantileverTheory(
        length_L=length_L,
        width_b=width_b,
        height_h=height_h,
        force_max=f_max,
        force_min=f_min,
        stress_ratio_R=R,
        moment_of_inertia_I=I,
        section_modulus_Z=Z,
        moment_max=M_max,
        moment_min=M_min,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        sigma_a=sigma_a,
        sigma_m=sigma_m,
        fs_goodman=float(fs_goodman),
        fs_soderberg=float(fs_soderberg),
        fs_gerber=float(fs_gerber),
    )


def build_cantilever_model_and_bc(
    length_L: float = 100.0,
    width_b: float = 10.0,
    height_h: float = 10.0,
    mesh_size: float = 2.5,
    force_max: float = 1000.0,
    force_min: float = 100.0,
    material: Optional[FatigueMaterial] = None,
) -> Tuple[MeshModel, MaterialAssignment, BoundaryConditionSet, BoundaryConditionSet]:
    """Create 3D tetrahedral MeshModel and boundary conditions for cantilever beam bending.

    Applies Poisson-free kinematic fixed constraints:
    - Planar constraint (ux = 0) across all nodes at x = 0.
    - Rigid-body translations (uy = 0, uz = 0) at the cross-section centroid (0, 0, 0).
    - Rigid-body roll around X (uz = 0) at an offset edge node (0, y_max, 0).
    This enables pure 1D bending stress states without Poisson clamping singularity.

    Args:
        length_L: Beam span in mm (along X: [0, L]).
        width_b: Beam width in mm (along Y: [-b/2, b/2]).
        height_h: Beam height in mm (along Z: [-h/2, h/2]).
        mesh_size: Target tetrahedral element edge length in mm.
        force_max: Magnitude of maximum transverse tip load in N.
        force_min: Magnitude of minimum transverse tip load in N.
        material: Material properties.

    Returns:
        Tuple of (model, material_assignment, bc_max, bc_min).
    """
    if material is None:
        material = FatigueMaterial(name="Ti-6Al-4V")

    box = trimesh.creation.box(extents=(length_L, width_b, height_h))
    box.apply_translation([length_L / 2.0, 0.0, 0.0])

    part = build_part_mesh("beam", box, size_far=mesh_size, size_near=mesh_size, order=2)
    model = MeshModel(parts=[part])

    tol = 1e-4
    x_root = np.where(np.abs(part.nodes[:, 0] - 0.0) < tol)[0]
    x_tip = np.where(np.abs(part.nodes[:, 0] - length_L) < tol)[0]

    # Kinematic isostatic constraints at x = 0
    dists_origin = np.linalg.norm(part.nodes[x_root], axis=1)
    center_idx = x_root[np.argmin(dists_origin)]

    # Select roll restraint strictly on the neutral axis (z ~ 0) to prevent interference with bending
    z_dev = np.abs(part.nodes[x_root, 2])
    neutral_mask = z_dev < 1.0
    neutral_x_root = x_root[neutral_mask]
    if len(neutral_x_root) > 0:
        roll_idx = neutral_x_root[np.argmax(np.abs(part.nodes[neutral_x_root, 1]))]
    else:
        roll_idx = x_root[np.argmax(np.abs(part.nodes[x_root, 1]))]

    bc_fixed = [
        {"name": "FIX_X_PLANE", "part": "beam", "node_ids": x_root, "dofs": (1, 1)},
        {"name": "FIX_YZ_CENTER", "part": "beam", "node_ids": np.array([center_idx]), "dofs": (2, 3)},
        {"name": "FIX_Z_ROLL", "part": "beam", "node_ids": np.array([roll_idx]), "dofs": (3, 3)},
    ]

    f_max_z = -float(abs(force_max))
    f_min_z = -float(abs(force_min))

    bc_max = BoundaryConditionSet(
        fixed_node_sets=bc_fixed,
        loads=[{"name": "LOAD_MAX", "part": "beam", "node_ids": x_tip, "force_N": (0.0, 0.0, f_max_z)}],
    )
    bc_min = BoundaryConditionSet(
        fixed_node_sets=bc_fixed,
        loads=[{"name": "LOAD_MIN", "part": "beam", "node_ids": x_tip, "force_N": (0.0, 0.0, f_min_z)}],
    )

    mat_def = MaterialDef(name=material.name, E=material.E, nu=material.nu)
    mat_assign = MaterialAssignment(
        element_material={"beam": np.zeros(len(part.elements), dtype=np.int64)},
        materials=[mat_def],
    )

    return model, mat_assign, bc_max, bc_min


def run_cantilever_benchmark(
    length_L: float = 100.0,
    width_b: float = 10.0,
    height_h: float = 10.0,
    force_max: float = 1000.0,
    force_min: float = 100.0,
    mesh_size: float = 2.5,
    material: Optional[FatigueMaterial] = None,
    mean_stress_method: MeanStressCorrectionMethod = MeanStressCorrectionMethod.GOODMAN,
    ccx_path: Optional[str] = None,
    tolerance_pct: float = 3.0,
) -> CantileverBenchmarkResult:
    """Execute complete 5.3 verification benchmark: Solve 2 load cases with CalculiX and evaluate fatigue.

    Verifies acceptance criteria:
    - Fixed root surface stress within +/- tolerance_pct (default 3.0%) of sigma = M*c/I
    - Fatigue safety factor within +/- tolerance_pct (default 3.0%) of analytical hand calculation

    Args:
        length_L: Beam span in mm.
        width_b: Beam width in mm.
        height_h: Beam height in mm.
        force_max: Peak cyclic load in N.
        force_min: Trough cyclic load in N.
        mesh_size: Mesh size in mm.
        material: FatigueMaterial definition.
        mean_stress_method: Mean stress correction method (Goodman, Soderberg, Gerber).
        ccx_path: Path to CalculiX ccx binary (auto-discovered if None).
        tolerance_pct: Acceptance tolerance threshold percentage (default 3.0%).

    Returns:
        CantileverBenchmarkResult containing comparison metrics and pass/fail status.
    """
    if ccx_path is None:
        ccx_path = shutil.which("ccx") or "/usr/bin/ccx"

    if not (os.path.isfile(ccx_path) and os.access(ccx_path, os.X_OK)):
        raise RuntimeError(f"CalculiX solver binary (ccx) not found or not executable at: {ccx_path}")

    if material is None:
        material = FatigueMaterial(
            name="Ti-6Al-4V",
            E=110000.0,
            nu=0.34,
            yield_strength_Sy=795.0,
            tensile_strength_Sut=860.0,
            fatigue_limit_Se=510.0,
            endurance_cycles_Ne=1.0e7,
        )

    # 1. Theoretical analytical values
    theory = calculate_cantilever_theory(
        length_L=length_L,
        width_b=width_b,
        height_h=height_h,
        force_max=force_max,
        force_min=force_min,
        material=material,
    )

    # 2. Build model and boundary conditions
    model, mat_assign, bc_max, bc_min = build_cantilever_model_and_bc(
        length_L=length_L,
        width_b=width_b,
        height_h=height_h,
        mesh_size=mesh_size,
        force_max=force_max,
        force_min=force_min,
        material=material,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run F_max simulation
        job_max = "cantilever_max"
        inp_max = os.path.join(tmpdir, f"{job_max}.inp")
        write_inp(inp_max, model, {"beam": mat_assign}, bc_max, ties=[], apply_ccx_ordering=False)
        res_max = subprocess.run([ccx_path, job_max], cwd=tmpdir, capture_output=True, text=True)
        if res_max.returncode != 0:
            raise RuntimeError(f"CalculiX F_max simulation failed:\n{res_max.stdout}\n{res_max.stderr}")

        # Run F_min simulation
        job_min = "cantilever_min"
        inp_min = os.path.join(tmpdir, f"{job_min}.inp")
        write_inp(inp_min, model, {"beam": mat_assign}, bc_min, ties=[], apply_ccx_ordering=False)
        res_min = subprocess.run([ccx_path, job_min], cwd=tmpdir, capture_output=True, text=True)
        if res_min.returncode != 0:
            raise RuntimeError(f"CalculiX F_min simulation failed:\n{res_min.stdout}\n{res_min.stderr}")

        frd_max = os.path.join(tmpdir, f"{job_max}.frd")
        frd_min = os.path.join(tmpdir, f"{job_min}.frd")

        # 3. Extract cyclic stresses from both .frd files
        node_ids, s_max, s_min = extract_cyclic_stresses_from_two_frds(frd_max, frd_min)

        # 4. Perform high-cycle fatigue evaluation
        load_case = FatigueLoadCase(
            name="Cantilever Validation",
            target_cycles=5.0e6,
            stress_ratio_R=theory.stress_ratio_R,
            mean_stress_method=mean_stress_method,
            stress_measure=StressMeasureType.SIGNED_VON_MISES,
        )
        fatigue_res = evaluate_fatigue(
            stress_max=s_max,
            stress_min=s_min,
            material=material,
            load_case=load_case,
        )

        part = model.parts[0]
        node_0based = node_ids - 1
        coords = part.nodes[node_0based]

        # 5. Extract tension surface nodes at fixed root (x = 0, z = +h/2)
        c_top = height_h / 2.0
        root_surface_mask = (np.abs(coords[:, 0] - 0.0) < 1e-4) & (np.abs(coords[:, 2] - c_top) < 1e-4)
        surface_indices = np.where(root_surface_mask)[0]

        if len(surface_indices) == 0:
            raise RuntimeError("No nodes found on fixed root tension surface (x=0, z=h/2).")

        fea_s_max = fatigue_res.sigma_max[surface_indices]
        fea_s_min = fatigue_res.sigma_min[surface_indices]
        fea_fs = fatigue_res.safety_factor[surface_indices]

        # Select corresponding theoretical safety factor
        if mean_stress_method == MeanStressCorrectionMethod.GOODMAN:
            theory_fs = theory.fs_goodman
        elif mean_stress_method == MeanStressCorrectionMethod.SODERBERG:
            theory_fs = theory.fs_soderberg
        elif mean_stress_method == MeanStressCorrectionMethod.GERBER:
            theory_fs = theory.fs_gerber
        else:
            theory_fs = theory.fs_goodman

        # Error metrics relative to analytical theory
        stress_errors = np.abs(fea_s_max - theory.sigma_max) / theory.sigma_max * 100.0
        fs_errors = np.abs(fea_fs - theory_fs) / theory_fs * 100.0

        max_stress_err = float(np.max(stress_errors))
        mean_stress_err = float(np.mean(stress_errors))
        max_fs_err = float(np.max(fs_errors))
        mean_fs_err = float(np.mean(fs_errors))

        passed_stress = bool(max_stress_err <= tolerance_pct)
        passed_fs = bool(max_fs_err <= tolerance_pct)

        return CantileverBenchmarkResult(
            theory=theory,
            passed_stress=passed_stress,
            passed_safety_factor=passed_fs,
            max_stress_error_pct=max_stress_err,
            mean_stress_error_pct=mean_stress_err,
            max_fs_error_pct=max_fs_err,
            mean_fs_error_pct=mean_fs_err,
            fea_sigma_max_mean=float(np.mean(fea_s_max)),
            fea_sigma_min_mean=float(np.mean(fea_s_min)),
            fea_fs_goodman_mean=float(np.mean(fea_fs)),
            num_surface_nodes_evaluated=len(surface_indices),
            mean_stress_method=mean_stress_method,
            fea_fs_mean=float(np.mean(fea_fs)),
            theory_fs=theory_fs,
            tolerance_pct=tolerance_pct,
        )
