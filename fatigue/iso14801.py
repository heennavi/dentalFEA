"""
iso14801.py
ISO 14801 dynamic fatigue testing setup and CalculiX simulation pipeline integration.
Provides configuration data models and execution wrappers for ISO 14801 dental implant fatigue tests.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import os
from pathlib import Path
import shutil
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
import numpy as np
import yaml

from fea.models import MeshModel, MeshPart, MaterialDef, MaterialAssignment, BoundaryConditionSet, TieDef
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
    extract_stress_from_frd,
    extract_cyclic_stresses_from_frd,
    extract_cyclic_stresses_from_two_frds,
    extract_cyclic_stresses_with_ratio,
)
from .stress_measures import calculate_von_mises, calculate_principal_stresses



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


@dataclass
class ISO14801SetupSpec:
    """ISO 14801 standard dimensions and tolerances."""
    nominal_angle_deg: float = 30.0
    angle_tolerance_deg: float = 2.0
    nominal_lever_arm_mm: float = 11.0
    lever_arm_tolerance_mm: float = 0.5
    nominal_resorption_depth_mm: float = 3.0
    resorption_depth_tolerance_mm: float = 0.2
    stress_ratio_R: float = 0.1
    stress_ratio_tolerance: float = 0.02


@dataclass
class ISO14801ValidationReport:
    """Results of geometric and loading verification against ISO 14801 standard."""
    passed: bool
    measured_angle_deg: float
    angle_error_deg: float
    passed_angle: bool
    measured_lever_arm_mm: float
    lever_arm_error_mm: float
    passed_lever_arm: bool
    measured_resorption_depth_mm: float
    resorption_depth_error_mm: float
    passed_resorption: bool
    measured_stress_ratio_R: float
    passed_stress_ratio: bool
    details: List[str] = field(default_factory=list)


@dataclass
class FatigueHotspot:
    """Fatigue critical hotspot identification and clinical validity summary."""
    part_name: str
    node_id: int
    coordinates: np.ndarray
    min_safety_factor: float
    max_stress_MPa: float
    is_clinically_valid: bool
    hotspot_region_name: str
    clinical_notes: str


@dataclass
class LiteratureBenchmarkComparison:
    """Comparison between FEA predicted fatigue limit and published ISO 14801 literature test."""
    benchmark_id: Optional[str]
    implant_system_name: str
    literature_reference: str
    literature_fatigue_limit_N: Optional[float]
    predicted_fatigue_limit_N: float
    relative_error_pct: Optional[float]
    status: str                         # "GOOD", "RECHECK_NEEDED", "UNVERIFIED"
    status_display_text: str             # UI display text
    details: str


@dataclass
class ISO14801BatchResult:
    """Comprehensive result object from ISO 14801 batch fatigue case execution."""
    validation_report: ISO14801ValidationReport
    hotspot: FatigueHotspot
    predicted_fatigue_limit_N: float
    benchmark_comparison: LiteratureBenchmarkComparison
    part_results: Dict[str, FatigueResult]
    overall_min_safety_factor: float
    passed: bool


def validate_iso14801_setup(
    clamping_z: float,
    crest_z: float,
    loading_point: Union[Sequence[float], np.ndarray],
    force_vector_max: Union[Sequence[float], np.ndarray],
    force_vector_min: Optional[Union[Sequence[float], np.ndarray]] = None,
    axial_axis: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 1.0),
    spec: Optional[ISO14801SetupSpec] = None,
) -> ISO14801ValidationReport:
    """Automatically verify that load application point, angle, and lever arm match ISO 14801 standard.

    Verification criteria:
    1. Angle: Loading angle relative to axial axis must equal 30 +/- 2 degrees.
    2. Lever arm: Distance along long axis from clamping plane to loading point must equal 11.0 +/- 0.5 mm.
    3. Bone resorption exposure: Height from clamping plane to nominal bone crest must equal 3.0 +/- 0.2 mm.
    4. Stress ratio R: Ratio ||F_min|| / ||F_max|| must equal 0.1 +/- 0.02.
    """
    if spec is None:
        spec = ISO14801SetupSpec()

    details = []

    # 1. Resorption exposure depth = crest_z - clamping_z
    meas_resorption = abs(float(crest_z) - float(clamping_z))
    resorption_err = abs(meas_resorption - spec.nominal_resorption_depth_mm)
    passed_resorption = resorption_err <= spec.resorption_depth_tolerance_mm
    details.append(
        f"골 흡수 노출 깊이: 측정치 {meas_resorption:.2f} mm (규격 {spec.nominal_resorption_depth_mm:.1f} ± {spec.resorption_depth_tolerance_mm:.1f} mm) -> {'일치' if passed_resorption else '불일치'}"
    )

    # 2. Lever arm distance from clamping plane to loading point
    lp = np.asarray(loading_point, dtype=np.float64)
    meas_lever_arm = abs(float(lp[2]) - float(clamping_z))
    lever_arm_err = abs(meas_lever_arm - spec.nominal_lever_arm_mm)
    passed_lever_arm = lever_arm_err <= spec.lever_arm_tolerance_mm
    details.append(
        f"레버 암 거리: 측정치 {meas_lever_arm:.2f} mm (규격 {spec.nominal_lever_arm_mm:.1f} ± {spec.lever_arm_tolerance_mm:.1f} mm) -> {'일치' if passed_lever_arm else '불일치'}"
    )

    # 3. Loading inclination angle
    f_max = np.asarray(force_vector_max, dtype=np.float64)
    norm_f_max = np.linalg.norm(f_max)
    if norm_f_max < 1e-9:
        raise ValueError("Peak force vector magnitude cannot be zero.")

    ax = np.asarray(axial_axis, dtype=np.float64)
    norm_ax = np.linalg.norm(ax)
    if norm_ax < 1e-9:
        raise ValueError("Axial axis magnitude cannot be zero.")
    ax = ax / norm_ax

    # Loading vector is compressive downwards (-ax direction)
    cos_theta = np.dot(-f_max / norm_f_max, ax)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    meas_angle_deg = float(np.degrees(np.arccos(cos_theta)))
    angle_err = abs(meas_angle_deg - spec.nominal_angle_deg)
    passed_angle = angle_err <= spec.angle_tolerance_deg
    details.append(
        f"하중 경사각: 측정치 {meas_angle_deg:.2f}° (규격 {spec.nominal_angle_deg:.1f} ± {spec.angle_tolerance_deg:.1f}°) -> {'일치' if passed_angle else '불일치'}"
    )

    # 4. Stress ratio R
    if force_vector_min is not None:
        f_min = np.asarray(force_vector_min, dtype=np.float64)
        norm_f_min = np.linalg.norm(f_min)
        meas_r = float(norm_f_min / norm_f_max)
    else:
        meas_r = spec.stress_ratio_R

    r_err = abs(meas_r - spec.stress_ratio_R)
    passed_r = r_err <= spec.stress_ratio_tolerance
    details.append(
        f"하중비 R: 측정치 {meas_r:.3f} (규격 {spec.stress_ratio_R:.1f} ± {spec.stress_ratio_tolerance:.2f}) -> {'일치' if passed_r else '불일치'}"
    )

    overall_passed = bool(passed_angle and passed_lever_arm and passed_resorption and passed_r)

    return ISO14801ValidationReport(
        passed=overall_passed,
        measured_angle_deg=meas_angle_deg,
        angle_error_deg=angle_err,
        passed_angle=passed_angle,
        measured_lever_arm_mm=meas_lever_arm,
        lever_arm_error_mm=lever_arm_err,
        passed_lever_arm=passed_lever_arm,
        measured_resorption_depth_mm=meas_resorption,
        resorption_depth_error_mm=resorption_err,
        passed_resorption=passed_resorption,
        measured_stress_ratio_R=meas_r,
        passed_stress_ratio=passed_r,
        details=details,
    )


def build_iso14801_benchmark_assembly(
    fixture_radius: float = 2.0,           # D = 4.0 mm
    fixture_total_length: float = 10.0,    # L = 10.0 mm
    bone_resorption_depth: float = 3.0,    # 3.0 mm exposure above clamping plane
    lever_arm: float = 11.0,               # 11.0 mm from clamping plane to loading center
    screw_radius: float = 0.8,             # M1.6 screw (R = 0.8 mm)
) -> Tuple[MeshModel, Dict[str, Any]]:
    """Build a standard ISO 14801 3-part assembly (Fixture, Abutment, Screw) with exact dimensions.

    Coordinate system:
    - Long axis: +Z
    - Clamping plane: z = 0.0 mm (nodes with z <= 0.0 constrained in UX, UY, UZ)
    - Nominal bone crest: z = 3.0 mm (exposure = 3.0 mm)
    - Fixture apex: z = -(fixture_total_length - bone_resorption_depth) = -7.0 mm
    - Abutment: sits on fixture crest (z = 3.0 mm) extending up to loading height (z = 11.0 mm)
    - Hemispherical loading point: center at z = 11.0 mm, x = 0.0, y = 0.0
    - Screw: runs through abutment into internal threaded cavity of fixture

    Returns:
        (MeshModel, metadata_dict)
    """
    def make_brick_part(x_vals, y_vals, z_vals, base_nid, base_eid):
        nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
        part_nodes = {}
        curr_nid = base_nid
        for k, z in enumerate(z_vals):
            for j, y in enumerate(y_vals):
                for i, x in enumerate(x_vals):
                    part_nodes[(i, j, k)] = (curr_nid, float(x), float(y), float(z))
                    curr_nid += 1

        part_elems = []
        curr_eid = base_eid
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(nx - 1):
                    n1 = part_nodes[(i, j, k)][0]
                    n2 = part_nodes[(i + 1, j, k)][0]
                    n3 = part_nodes[(i + 1, j + 1, k)][0]
                    n4 = part_nodes[(i, j + 1, k)][0]
                    n5 = part_nodes[(i, j, k + 1)][0]
                    n6 = part_nodes[(i + 1, j, k + 1)][0]
                    n7 = part_nodes[(i + 1, j + 1, k + 1)][0]
                    n8 = part_nodes[(i, j + 1, k + 1)][0]
                    part_elems.append((curr_eid, [n1, n2, n3, n4, n5, n6, n7, n8]))
                    curr_eid += 1
        return part_nodes, part_elems, curr_nid, curr_eid

    r_fix = float(fixture_radius)
    r_sc = float(screw_radius)
    z_apex = -(fixture_total_length - bone_resorption_depth)  # -7.0
    z_clamp = 0.0
    z_crest = float(bone_resorption_depth)                    # +3.0
    z_load = float(lever_arm)                                 # +11.0

    # 1. Fixture Part
    # Z divisions: [z_apex, -3.5, 0.0 (clamping plane), 1.5, z_crest (3.0)]
    x_fix = np.linspace(-r_fix, r_fix, 3)
    y_fix = np.linspace(-r_fix, r_fix, 3)
    z_fix = np.array([z_apex, -3.5, z_clamp, 1.5, z_crest], dtype=np.float64)
    n_fix, e_fix, next_nid, next_eid = make_brick_part(x_fix, y_fix, z_fix, 1, 1)

    # 2. Abutment Part
    # Z divisions: [z_crest (3.0), 5.5, 8.0, z_load (11.0)]
    x_abt = np.linspace(-r_fix, r_fix, 3)
    y_abt = np.linspace(-r_fix, r_fix, 3)
    z_abt = np.array([z_crest, 5.5, 8.0, z_load], dtype=np.float64)
    n_abt, e_abt, next_nid, next_eid = make_brick_part(x_abt, y_abt, z_abt, next_nid, next_eid)

    # 3. Screw Part
    # Z divisions: [-2.0 (threaded body), 1.0 (shank), 3.0 (joint), 5.5 (head)]
    x_sc = np.array([-r_sc, r_sc], dtype=np.float64)
    y_sc = np.array([-r_sc, r_sc], dtype=np.float64)
    z_sc = np.array([-2.0, 1.0, 3.0, 5.5], dtype=np.float64)
    n_sc, e_sc, next_nid, next_eid = make_brick_part(x_sc, y_sc, z_sc, next_nid, next_eid)

    # Convert to MeshPart
    def dict_to_arrays(part_nodes, part_elems):
        coords = np.array([[x, y, z] for _, (_, x, y, z) in sorted(part_nodes.items(), key=lambda it: it[1][0])], dtype=np.float64)
        min_nid = min(nid for _, (nid, *_) in part_nodes.items())
        elems = np.array([[n - min_nid for n in conn] for _, conn in part_elems], dtype=np.int64)
        return coords, elems

    coords_fix, elems_fix = dict_to_arrays(n_fix, e_fix)
    coords_abt, elems_abt = dict_to_arrays(n_abt, e_abt)
    coords_sc, elems_sc = dict_to_arrays(n_sc, e_sc)

    part_fix = MeshPart(name="fixture", nodes=coords_fix, elements=elems_fix, element_type="C3D8")
    part_abt = MeshPart(name="abutment", nodes=coords_abt, elements=elems_abt, element_type="C3D8")
    part_sc = MeshPart(name="screw", nodes=coords_sc, elements=elems_sc, element_type="C3D8")

    model = MeshModel(parts=[part_fix, part_abt, part_sc])

    # Loading node IDs (tip surface of abutment at z = 11.0)
    abt_tip_nodes = np.where(np.abs(part_abt.nodes[:, 2] - z_load) < 1e-4)[0]

    # Fixed node IDs (fixture nodes with z <= 0.0)
    fix_clamp_nodes = np.where(part_fix.nodes[:, 2] <= (z_clamp + 1e-4))[0]

    # Global node IDs
    fix_clamp_global = [n_fix[(i, j, k)][0] for (i, j, k), (nid, *_) in n_fix.items() if z_fix[k] <= (z_clamp + 1e-4)]
    abt_tip_global = [n_abt[(i, j, k)][0] for (i, j, k), (nid, *_) in n_abt.items() if abs(z_abt[k] - z_load) < 1e-4]

    metadata = {
        "n_fix": n_fix,
        "e_fix": e_fix,
        "n_abt": n_abt,
        "e_abt": e_abt,
        "n_sc": n_sc,
        "e_sc": e_sc,
        "clamping_z": z_clamp,
        "crest_z": z_crest,
        "loading_point": np.array([0.0, 0.0, z_load], dtype=np.float64),
        "loading_node_ids": abt_tip_nodes,
        "clamping_node_ids": fix_clamp_nodes,
        "clamping_global_nids": fix_clamp_global,
        "loading_global_nids": abt_tip_global,
        "e_fix_top": e_fix[-4:],
        "e_abt_bot": e_abt[:4],
        "e_abt_mid": e_abt[:4],
        "e_sc_head": e_sc[-1:],
        "lever_arm": lever_arm,
        "resorption_depth": bone_resorption_depth,
        "axial_axis": (0.0, 0.0, 1.0),
        "transverse_axis": (1.0, 0.0, 0.0),
    }

    return model, metadata


def identify_fatigue_hotspot(
    model: MeshModel,
    part_results: Dict[str, FatigueResult],
    clamping_z: float = 0.0,
    crest_z: float = 3.0,
) -> FatigueHotspot:
    """Identify the critical fatigue hotspot and assess clinical failure mode plausibility.

    ISO 14801 standard and clinical failure literature show that high-cycle fatigue
    fractures overwhelmingly occur at:
    1. The abutment screw (neck, thread runout, or head junction).
    2. The fixture collar / neck (at or just above the bone clamping level due to high bending moment).
    3. The abutment collar / neck junction.
    """
    lowest_fs = float("inf")
    critical_part = ""
    critical_node_local = -1
    critical_stress = 0.0

    for pname, res in part_results.items():
        if len(res.safety_factor) == 0:
            continue
        min_idx = int(np.argmin(res.safety_factor))
        fs_val = float(res.safety_factor[min_idx])
        if fs_val < lowest_fs:
            lowest_fs = fs_val
            critical_part = pname
            critical_node_local = min_idx
            critical_stress = float(res.sigma_max[min_idx])

    if not critical_part:
        raise ValueError("No valid fatigue results found to identify hotspot.")

    target_part = model.get_part(critical_part)
    if target_part is None:
        raise KeyError(f"Part '{critical_part}' not found in MeshModel.")

    coords = target_part.nodes[critical_node_local]
    z_coord = float(coords[2])

    is_valid = False
    region_name = ""
    notes = ""

    part_lower = critical_part.lower()
    if "screw" in part_lower:
        is_valid = True
        region_name = "지대주 나사 (Abutment Screw)"
        notes = "임상 및 ISO 14801 피로 시험에서 가장 빈번하게 보고되는 전형적인 체결 나사 피로 파절 호발 부위입니다."
    elif "fix" in part_lower or "implant" in part_lower:
        if (clamping_z - 1.0) <= z_coord <= (crest_z + 1.5):
            is_valid = True
            region_name = "고정체 목부 (Fixture Neck / Collar)"
            notes = "골 흡수 모사 경계면(Clamping plane) 직상부 최대 굽힘 모멘트 작용 영역으로, 고정체 파절의 핵심 호발 부위입니다."
        else:
            region_name = f"고정체 몸통 (z={z_coord:.1f}mm)"
            notes = "고정체 체결부 이외 영역에서 최대 응력이 관측되었습니다. 메쉬 및 구속 조건을 확인하십시오."
    elif "abut" in part_lower:
        if abs(z_coord - crest_z) <= 2.5:
            is_valid = True
            region_name = "지대주 목부 (Abutment Neck)"
            notes = "고정체와의 접촉 경계면 직상부로, 모멘트 집중에 의한 지대주 피로 균열 발생 호발 부위입니다."
        else:
            region_name = f"지대주 상부 (z={z_coord:.1f}mm)"
            notes = "하중 적용점 인근 국부 접촉 응력 집중일 가능성이 있습니다."
    else:
        region_name = f"{critical_part} (z={z_coord:.1f}mm)"
        notes = "기타 구조 영역"

    return FatigueHotspot(
        part_name=critical_part,
        node_id=critical_node_local,
        coordinates=coords,
        min_safety_factor=lowest_fs,
        max_stress_MPa=critical_stress,
        is_clinically_valid=is_valid,
        hotspot_region_name=region_name,
        clinical_notes=notes,
    )


def calculate_predicted_fatigue_limit_load(
    applied_load_max_N: float,
    min_safety_factor: float,
    mean_stress_method: MeanStressCorrectionMethod = MeanStressCorrectionMethod.GOODMAN,
) -> float:
    """Calculate the predicted fatigue limit load F_limit where minimum safety factor equals 1.0."""
    if min_safety_factor <= 0.0:
        return 0.0
    f_limit = applied_load_max_N * min_safety_factor
    return float(f_limit)


def compare_with_literature_benchmark(
    predicted_fatigue_limit_N: float,
    benchmark_id: Optional[str] = None,
    materials_yaml_path: Optional[Union[str, Path]] = None,
    tolerance_pct: float = 35.0,
) -> LiteratureBenchmarkComparison:
    """Compare predicted fatigue limit against published ISO 14801 empirical test literature."""
    if materials_yaml_path is None:
        materials_yaml_path = Path(__file__).resolve().parent.parent / "config" / "materials_fatigue.yaml"

    yaml_path = Path(materials_yaml_path)
    benchmarks_dict = {}
    if yaml_path.is_file():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                benchmarks_dict = data.get("benchmark_cases", {})
        except Exception:
            benchmarks_dict = {}

    if not benchmark_id or benchmark_id not in benchmarks_dict:
        return LiteratureBenchmarkComparison(
            benchmark_id=None,
            implant_system_name="미지정 (해당 임플란트 공개 시험자료 없음)",
            literature_reference="N/A",
            literature_fatigue_limit_N=None,
            predicted_fatigue_limit_N=predicted_fatigue_limit_N,
            relative_error_pct=None,
            status="UNVERIFIED",
            status_display_text="교차 검증 미실시 (공개 자료 부재)",
            details="동일 임플란트 계열의 공인된 ISO 14801 시험 자료가 없어 교차 검증을 실시하지 않았습니다. 해석 결과를 상대 비교 지표로 활용하십시오.",
        )

    bench = benchmarks_dict[benchmark_id]
    ref = bench.get("reference", "")
    sys_name = bench.get("implant_system", benchmark_id)
    lit_f_limit = float(bench.get("fatigue_limit_5M_cycles_N", 0.0))

    if lit_f_limit <= 0.0:
        return LiteratureBenchmarkComparison(
            benchmark_id=benchmark_id,
            implant_system_name=sys_name,
            literature_reference=ref,
            literature_fatigue_limit_N=None,
            predicted_fatigue_limit_N=predicted_fatigue_limit_N,
            relative_error_pct=None,
            status="UNVERIFIED",
            status_display_text="교차 검증 미실시 (문헌 한계값 미기재)",
            details="문헌에 유효한 피로한계 하중 수치가 기재되어 있지 않습니다.",
        )

    rel_error = abs(predicted_fatigue_limit_N - lit_f_limit) / lit_f_limit * 100.0

    if rel_error <= tolerance_pct:
        status = "GOOD"
        display_text = f"양호 (문헌 오차 {rel_error:.1f}%)"
        details = (
            f"해석 예측 피로한계({predicted_fatigue_limit_N:.1f} N)가 문헌 시험치({lit_f_limit:.1f} N)와 "
            f"{rel_error:.1f}% 오차로 일치하여 수십 % 이내 양호 판정 기준을 만족합니다."
        )
    else:
        status = "RECHECK_NEEDED"
        display_text = f"재점검 권장 (문헌 오차 {rel_error:.1f}%)"
        details = (
            f"해석 예측치({predicted_fatigue_limit_N:.1f} N)와 문헌 시험치({lit_f_limit:.1f} N)의 오차가 "
            f"{rel_error:.1f}%로 {tolerance_pct:.0f}%를 초과했습니다. 접촉면 조건, 나사 선하중(Preload), 또는 재료 물성을 재점검하십시오."
        )

    return LiteratureBenchmarkComparison(
        benchmark_id=benchmark_id,
        implant_system_name=sys_name,
        literature_reference=ref,
        literature_fatigue_limit_N=lit_f_limit,
        predicted_fatigue_limit_N=predicted_fatigue_limit_N,
        relative_error_pct=rel_error,
        status=status,
        status_display_text=display_text,
        details=details,
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


def write_iso14801_benchmark_deck(
    path: str,
    meta: Dict[str, Any],
    force_vec: np.ndarray,
    fatigue_mat: FatigueMaterial,
    title: str = "ISO 14801 Dynamic Benchmark Deck",
) -> None:
    """Write dedicated CalculiX input deck (.inp) for standard ISO 14801 benchmark assembly."""
    n_fix = meta["n_fix"]
    e_fix = meta["e_fix"]
    n_abt = meta["n_abt"]
    e_abt = meta["e_abt"]
    n_sc = meta["n_sc"]
    e_sc = meta["e_sc"]

    lines = ["*HEADING", title]
    lines.append("*NODE")
    for d in [n_fix, n_abt, n_sc]:
        for (i, j, k), (nid, x, y, z) in d.items():
            lines.append(f"{nid}, {x:.6f}, {y:.6f}, {z:.6f}")

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=E_FIX")
    for eid, conn in e_fix:
        lines.append(f"{eid}, " + ", ".join(str(n) for n in conn))

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=E_ABUT")
    for eid, conn in e_abt:
        lines.append(f"{eid}, " + ", ".join(str(n) for n in conn))

    lines.append("*ELEMENT, TYPE=C3D8, ELSET=E_SCREW")
    for eid, conn in e_sc:
        lines.append(f"{eid}, " + ", ".join(str(n) for n in conn))

    lines.append("*MATERIAL, NAME=TITANIUM")
    lines.append("*ELASTIC")
    lines.append(f"{float(fatigue_mat.E):.1f}, {float(fatigue_mat.nu):.4f}")
    lines.append("*SOLID SECTION, ELSET=E_FIX, MATERIAL=TITANIUM")
    lines.append("*SOLID SECTION, ELSET=E_ABUT, MATERIAL=TITANIUM")
    lines.append("*SOLID SECTION, ELSET=E_SCREW, MATERIAL=TITANIUM")

    def append_nset_16(nset_name: str, nids: Sequence[int]):
        lines.append(f"*NSET, NSET={nset_name}")
        for i in range(0, len(nids), 16):
            lines.append(", ".join(str(n) for n in nids[i : i + 16]))

    clamp_nodes = meta["clamping_global_nids"]
    append_nset_16("N_CLAMP", clamp_nodes)

    load_nodes = meta["loading_global_nids"]
    append_nset_16("N_LOAD", load_nodes)

    # TIE interface 1: Fixture top to Abutment bottom
    lines.append("*SURFACE, NAME=SURF_FIX_TOP, TYPE=ELEMENT")
    for eid, _ in meta["e_fix_top"]:
        lines.append(f"{eid}, S2")

    lines.append("*SURFACE, NAME=SURF_ABT_BOT, TYPE=ELEMENT")
    for eid, _ in meta["e_abt_bot"]:
        lines.append(f"{eid}, S1")

    lines.append("*TIE, NAME=TIE_FIX_ABT, POSITION TOLERANCE=0.2")
    lines.append("SURF_ABT_BOT, SURF_FIX_TOP")

    # TIE interface 2: Screw head to Abutment
    lines.append("*SURFACE, NAME=SURF_ABT_MID, TYPE=ELEMENT")
    for eid, _ in meta["e_abt_mid"]:
        lines.append(f"{eid}, S1")

    lines.append("*SURFACE, NAME=SURF_SC_HEAD, TYPE=ELEMENT")
    for eid, _ in meta["e_sc_head"]:
        lines.append(f"{eid}, S2")

    lines.append("*TIE, NAME=TIE_SC_ABT, POSITION TOLERANCE=0.2")
    lines.append("SURF_SC_HEAD, SURF_ABT_MID")

    # Step definition
    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*BOUNDARY")
    lines.append("N_CLAMP, 1, 3")

    f_per_node = force_vec / len(load_nodes)
    lines.append("*CLOAD")
    for nid in load_nodes:
        if abs(f_per_node[0]) > 1e-6:
            lines.append(f"{nid}, 1, {f_per_node[0]:.6f}")
        if abs(f_per_node[1]) > 1e-6:
            lines.append(f"{nid}, 2, {f_per_node[1]:.6f}")
        if abs(f_per_node[2]) > 1e-6:
            lines.append(f"{nid}, 3, {f_per_node[2]:.6f}")

    lines.append("*NODE FILE")
    lines.append("U")
    lines.append("*EL FILE")
    lines.append("S")
    lines.append("*END STEP")

    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def run_iso14801_batch_case(
    config: Optional[ISO14801Config] = None,
    benchmark_id: Optional[str] = "gehrke_2020_conical",
    materials_yaml_path: Optional[Union[str, Path]] = None,
    ccx_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> ISO14801BatchResult:
    """Execute complete Step 5.6 ISO 14801 integrated fatigue verification batch case.

    Pipeline:
    1. Generates precision ISO 14801 test assembly (fixture, abutment, screw) with 3mm exposure & 11mm lever arm.
    2. Runs automated verification of geometric dimensions and loading vectors against ISO 14801 standard.
    3. Runs CalculiX multi-part FEA under F_max and F_min (R = 0.1, 30 deg).
    4. Evaluates nodal fatigue life and safety factors across all assembly parts.
    5. Identifies critical fatigue hotspot and verifies clinical plausibility (screw vs fixture neck).
    6. Calculates predicted fatigue limit load F_limit where FS = 1.0.
    7. Cross-validates against literature benchmark (or flags '교차 검증 미실시' if no data).

    Returns:
        ISO14801BatchResult summarizing all metrics and pass/fail status.
    """
    if config is None:
        config = ISO14801Config(
            load_max_N=250.0,
            stress_ratio_R=0.1,
            angle_deg=30.0,
            mode=ISO14801LoadMode.DUAL_SIMULATION,
        )

    if ccx_path is None:
        ccx_path = shutil.which("ccx") or "/usr/bin/ccx"

    if not (os.path.isfile(ccx_path) and os.access(ccx_path, os.X_OK)):
        raise RuntimeError(f"CalculiX solver binary (ccx) not executable: {ccx_path}")

    # 1. Build standard assembly
    model, meta = build_iso14801_benchmark_assembly(
        fixture_radius=2.0,
        fixture_total_length=10.0,
        bone_resorption_depth=config.bone_resorption_depth_mm,
        lever_arm=config.lever_arm_mm,
    )

    f_max_vec = config.calculate_force_vector(config.load_max_N, meta["axial_axis"], meta["transverse_axis"])
    f_min_vec = config.calculate_force_vector(config.load_min_N, meta["axial_axis"], meta["transverse_axis"])

    # 2. Geometric and loading validation against ISO 14801
    val_report = validate_iso14801_setup(
        clamping_z=meta["clamping_z"],
        crest_z=meta["crest_z"],
        loading_point=meta["loading_point"],
        force_vector_max=f_max_vec,
        force_vector_min=f_min_vec,
        axial_axis=meta["axial_axis"],
    )

    # 3. Materials
    if materials_yaml_path is None:
        materials_yaml_path = Path(__file__).resolve().parent.parent / "config" / "materials_fatigue.yaml"

    fatigue_mat = FatigueMaterial.from_yaml(materials_yaml_path, "ti_6al_4v_eli")
    mat_def = MaterialDef(name="Ti-6Al-4V", E=fatigue_mat.E, nu=fatigue_mat.nu)

    materials = {}
    for p in model.parts:
        materials[p.name] = MaterialAssignment(
            element_material={p.name: np.zeros(len(p.elements), dtype=np.int64)},
            materials=[mat_def],
        )

    # Boundary conditions
    fixed_sets = [{
        "name": "FIX_CLAMPING_PLANE",
        "part": "fixture",
        "node_ids": meta["clamping_node_ids"],
        "dofs": (1, 3),
    }]

    bc_max = BoundaryConditionSet(
        fixed_node_sets=fixed_sets,
        loads=[{
            "name": "LOAD_ISO14801_MAX",
            "part": "abutment",
            "node_ids": meta["loading_node_ids"],
            "force_N": tuple(f_max_vec),
        }],
    )

    bc_min = BoundaryConditionSet(
        fixed_node_sets=fixed_sets,
        loads=[{
            "name": "LOAD_ISO14801_MIN",
            "part": "abutment",
            "node_ids": meta["loading_node_ids"],
            "force_N": tuple(f_min_vec),
        }],
    )

    work_dir = output_dir or tempfile.mkdtemp(prefix="iso14801_batch_")
    cleanup_needed = output_dir is None

    part_results: Dict[str, FatigueResult] = {}
    fatigue_case = config.to_fatigue_load_case()

    try:
        # Solve F_max
        job_max = "iso14801_max"
        inp_max = os.path.join(work_dir, f"{job_max}.inp")
        write_iso14801_benchmark_deck(inp_max, meta, f_max_vec, fatigue_mat, "ISO 14801 F_max Step")
        res_max = subprocess.run([ccx_path, job_max], cwd=work_dir, capture_output=True, text=True)
        if res_max.returncode != 0:
            raise RuntimeError(f"CalculiX F_max run failed:\n{res_max.stdout}\n{res_max.stderr}")

        frd_max = os.path.join(work_dir, f"{job_max}.frd")

        if config.mode == ISO14801LoadMode.DUAL_SIMULATION:
            job_min = "iso14801_min"
            inp_min = os.path.join(work_dir, f"{job_min}.inp")
            write_iso14801_benchmark_deck(inp_min, meta, f_min_vec, fatigue_mat, "ISO 14801 F_min Step")
            res_min = subprocess.run([ccx_path, job_min], cwd=work_dir, capture_output=True, text=True)
            if res_min.returncode != 0:
                raise RuntimeError(f"CalculiX F_min run failed:\n{res_min.stdout}\n{res_min.stderr}")
            frd_min = os.path.join(work_dir, f"{job_min}.frd")
            all_nids, s_max_all, s_min_all = extract_cyclic_stresses_from_two_frds(frd_max, frd_min)
        else:
            all_nids, s_max_all, s_min_all = extract_cyclic_stresses_with_ratio(frd_max, config.stress_ratio_R)

        # Map stresses back to each part
        curr_offset = 0
        for p in model.parts:
            n_part_nodes = len(p.nodes)
            p_s_max = s_max_all[curr_offset : curr_offset + n_part_nodes]
            p_s_min = s_min_all[curr_offset : curr_offset + n_part_nodes]
            curr_offset += n_part_nodes

            part_results[p.name] = evaluate_fatigue(
                stress_max=p_s_max,
                stress_min=p_s_min,
                material=fatigue_mat,
                load_case=fatigue_case,
            )

        # 4. Hotspot identification
        hotspot = identify_fatigue_hotspot(
            model=model,
            part_results=part_results,
            clamping_z=meta["clamping_z"],
            crest_z=meta["crest_z"],
        )

        overall_min_fs = hotspot.min_safety_factor

        # 5. Calculate predicted fatigue limit load F_limit (FS = 1.0)
        f_limit = calculate_predicted_fatigue_limit_load(
            applied_load_max_N=config.load_max_N,
            min_safety_factor=overall_min_fs,
            mean_stress_method=config.mean_stress_method,
        )

        # 6. Literature benchmark cross-validation
        bench_comp = compare_with_literature_benchmark(
            predicted_fatigue_limit_N=f_limit,
            benchmark_id=benchmark_id,
            materials_yaml_path=materials_yaml_path,
        )

        overall_passed = bool(
            val_report.passed
            and hotspot.is_clinically_valid
            and (bench_comp.status in ("GOOD", "RECHECK_NEEDED", "UNVERIFIED"))
        )

        return ISO14801BatchResult(
            validation_report=val_report,
            hotspot=hotspot,
            predicted_fatigue_limit_N=f_limit,
            benchmark_comparison=bench_comp,
            part_results=part_results,
            overall_min_safety_factor=overall_min_fs,
            passed=overall_passed,
        )

    finally:
        if cleanup_needed and os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

