"""
iso14801.py
ISO 14801 dynamic fatigue testing setup and CalculiX simulation pipeline integration.
Provides configuration data models and execution wrappers for ISO 14801 dental implant fatigue tests.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import os
import shutil
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from fea.models import MeshModel, MaterialAssignment, BoundaryConditionSet
from solver.calculix_writer import write_inp
from .models import (
    FatigueMaterial,
    FatigueLoadCase,
    FatigueResult,
    MeanStressCorrectionMethod,
    StressMeasureType,
)
from .evaluate import evaluate_fatigue
from .io_frd import (
    extract_cyclic_stresses_from_frd,
    extract_cyclic_stresses_from_two_frds,
    extract_cyclic_stresses_with_ratio,
)


class ISO14801LoadMode(str, Enum):
    """Simulation workflow mode for ISO 14801 cyclic loading."""
    DUAL_SIMULATION = "dual_simulation"    # Solve F_max and F_min in separate CalculiX runs
    STRESS_RATIO_R = "stress_ratio_r"      # Solve F_max once and scale trough stress by R = 0.1


@dataclass
class ISO14801Config:
    """ISO 14801 dynamic fatigue testing standard configuration.

    Parameters conform to ISO 14801:2016 (Dentistry - Implants - Dynamic loading test):
    - Loading angle: 30 +/- 2 degrees relative to implant long axis.
    - Bone resorption / embedding level: 3.0 mm below nominal bone crest.
    - Lever arm distance: Typically 11.0 mm from clamping plane to loading center.
    - Cycle frequency: <= 15 Hz in air, <= 2 Hz in aqueous/saline environment.
    - Target test cycles: 5.0e6 cycles (or 2.0e6 in saline).
    - Stress ratio: R = 0.1 (F_min = 0.1 * F_max).
    """
    load_max_N: float = 250.0
    stress_ratio_R: float = 0.1
    angle_deg: float = 30.0
    bone_resorption_depth_mm: float = 3.0
    lever_arm_mm: float = 11.0
    target_cycles: float = 5.0e+06
    frequency_hz: float = 15.0
    surface_factor_ka: float = 1.0
    notch_factor_kf: float = 1.0
    mean_stress_method: MeanStressCorrectionMethod = MeanStressCorrectionMethod.GOODMAN
    stress_measure: StressMeasureType = StressMeasureType.SIGNED_VON_MISES
    mode: ISO14801LoadMode = ISO14801LoadMode.DUAL_SIMULATION

    @property
    def load_min_N(self) -> float:
        """Trough cyclic compressive load magnitude in N."""
        return self.load_max_N * self.stress_ratio_R

    @property
    def angle_rad(self) -> float:
        """Loading angle in radians."""
        return math.radians(self.angle_deg)

    def calculate_force_vector(
        self,
        magnitude_N: float,
        axial_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        transverse_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Compute 3D force vector for the specified load magnitude at 30 deg angle.

        Args:
            magnitude_N: Load magnitude in N.
            axial_axis: Unit direction along implant long axis (default +Z).
            transverse_axis: Unit direction perpendicular to long axis (default +X).

        Returns:
            (3,) np.ndarray force vector [Fx, Fy, Fz] in N.
        """
        ax = np.asarray(axial_axis, dtype=np.float64)
        ax = ax / np.linalg.norm(ax)

        tr = np.asarray(transverse_axis, dtype=np.float64)
        # Gram-Schmidt orthogonalize transverse axis against axial axis
        tr = tr - np.dot(tr, ax) * ax
        tr_norm = np.linalg.norm(tr)
        if tr_norm < 1e-12:
            tr = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            tr = tr - np.dot(tr, ax) * ax
            tr = tr / np.linalg.norm(tr)
        else:
            tr = tr / tr_norm

        theta = self.angle_rad
        # F = -magnitude * (cos(theta)*axial + sin(theta)*transverse)
        vec = -magnitude_N * (math.cos(theta) * ax + math.sin(theta) * tr)
        return vec

    def to_fatigue_load_case(self) -> FatigueLoadCase:
        """Convert ISO 14801 configuration to FatigueLoadCase."""
        return FatigueLoadCase(
            name=f"ISO 14801 ({self.load_max_N:.1f}N, {self.angle_deg:.0f}deg, R={self.stress_ratio_R})",
            target_cycles=self.target_cycles,
            stress_ratio_R=self.stress_ratio_R,
            surface_factor_ka=self.surface_factor_ka,
            notch_factor_kf=self.notch_factor_kf,
            mean_stress_method=self.mean_stress_method,
            stress_measure=self.stress_measure,
        )


def run_iso14801_pipeline(
    model: MeshModel,
    materials: Dict[str, MaterialAssignment],
    target_part_name: str,
    fatigue_material: FatigueMaterial,
    config: ISO14801Config,
    fixed_node_sets: List[Dict[str, Any]],
    loading_node_ids: np.ndarray,
    axial_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    transverse_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    ccx_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> FatigueResult:
    """Execute complete ISO 14801 dynamic fatigue simulation pipeline with CalculiX.

    Orchestrates:
    1. Formulates 3D force vectors for F_max and F_min at 30 deg inclination.
    2. Runs CalculiX solver for F_max (and optionally F_min depending on config.mode).
    3. Extracts cyclic stress fields from generated .frd files.
    4. Evaluates nodal fatigue safety factors, life cycles, and damage for target_part.

    Args:
        model: Meshed assembly MeshModel.
        materials: Material assignment mapping.
        target_part_name: Part name to evaluate fatigue on (e.g. "implant" or "abutment").
        fatigue_material: FatigueMaterial instance for target part.
        config: ISO14801Config configuration parameters.
        fixed_node_sets: Fixed constraint boundary conditions.
        loading_node_ids: Node indices on loading cap where force is applied.
        axial_axis: Vector along implant long axis.
        transverse_axis: Vector along transverse bending direction.
        ccx_path: Optional ccx binary path.
        output_dir: Optional working directory for .inp and .frd files.

    Returns:
        FatigueResult containing nodal fatigue evaluation results.
    """
    if ccx_path is None:
        ccx_path = shutil.which("ccx") or "/usr/bin/ccx"

    if not (os.path.isfile(ccx_path) and os.access(ccx_path, os.X_OK)):
        raise RuntimeError(f"CalculiX solver binary (ccx) not executable: {ccx_path}")

    # Compute 3D force vectors
    f_max_vec = config.calculate_force_vector(config.load_max_N, axial_axis, transverse_axis)
    f_min_vec = config.calculate_force_vector(config.load_min_N, axial_axis, transverse_axis)

    # Find target part in model
    target_part = None
    for p in model.parts:
        if p.name == target_part_name:
            target_part = p
            break
    if target_part is None:
        raise KeyError(f"Target part '{target_part_name}' not found in MeshModel.")

    # Determine loading part name
    # Default to target_part_name if loading_node_ids are specified
    bc_max = BoundaryConditionSet(
        fixed_node_sets=fixed_node_sets,
        loads=[{
            "name": "LOAD_ISO14801_MAX",
            "part": target_part_name,
            "node_ids": loading_node_ids,
            "force_N": tuple(f_max_vec),
        }],
    )

    work_dir = output_dir or tempfile.mkdtemp(prefix="iso14801_fea_")
    cleanup_needed = output_dir is None

    try:
        # Solve F_max
        job_max = "iso14801_max"
        inp_max = os.path.join(work_dir, f"{job_max}.inp")
        write_inp(inp_max, model, materials, bc_max, ties=[], apply_ccx_ordering=False)

        res_max = subprocess.run([ccx_path, job_max], cwd=work_dir, capture_output=True, text=True)
        if res_max.returncode != 0:
            raise RuntimeError(f"CalculiX F_max simulation failed:\n{res_max.stdout}\n{res_max.stderr}")

        frd_max = os.path.join(work_dir, f"{job_max}.frd")

        if config.mode == ISO14801LoadMode.DUAL_SIMULATION:
            # Solve F_min
            job_min = "iso14801_min"
            inp_min = os.path.join(work_dir, f"{job_min}.inp")
            bc_min = BoundaryConditionSet(
                fixed_node_sets=fixed_node_sets,
                loads=[{
                    "name": "LOAD_ISO14801_MIN",
                    "part": target_part_name,
                    "node_ids": loading_node_ids,
                    "force_N": tuple(f_min_vec),
                }],
            )
            write_inp(inp_min, model, materials, bc_min, ties=[], apply_ccx_ordering=False)
            res_min = subprocess.run([ccx_path, job_min], cwd=work_dir, capture_output=True, text=True)
            if res_min.returncode != 0:
                raise RuntimeError(f"CalculiX F_min simulation failed:\n{res_min.stdout}\n{res_min.stderr}")

            frd_min = os.path.join(work_dir, f"{job_min}.frd")
            node_ids, s_max, s_min = extract_cyclic_stresses_from_two_frds(frd_max, frd_min)
        else:
            # Scaled with stress ratio R
            node_ids, s_max, s_min = extract_cyclic_stresses_with_ratio(
                frd_max, stress_ratio_R=config.stress_ratio_R
            )

        # Evaluate fatigue
        fatigue_load_case = config.to_fatigue_load_case()
        fatigue_result = evaluate_fatigue(
            stress_max=s_max,
            stress_min=s_min,
            material=fatigue_material,
            load_case=fatigue_load_case,
        )

        return fatigue_result

    finally:
        if cleanup_needed and os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
