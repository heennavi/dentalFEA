"""
ui_bridge.py
Bridge module connecting ResultSet data structures with the fatigue evaluation engine
and executing Step 5.3 and Step 5.6 validation benchmarks for UI execution.
Pure computational module with NO PySide6 or rendering dependencies.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

from fea.models import MeshModel
from fea.results import ResultSet
from .models import (
    FatigueMaterial,
    FatigueLoadCase,
    FatigueResult,
    MeanStressCorrectionMethod,
    StressMeasureType,
)
from .evaluate import evaluate_fatigue
from .cantilever_benchmark import run_cantilever_benchmark, CantileverBenchmarkResult
from .iso14801 import (
    ISO14801Config,
    ISO14801LoadMode,
    ISO14801BatchResult,
    run_iso14801_batch_case,
    identify_fatigue_hotspot,
)


def evaluate_fatigue_on_result_set(
    rs: ResultSet,
    model: MeshModel,
    material: FatigueMaterial,
    load_case: Optional[FatigueLoadCase] = None,
    target_part_name: Optional[str] = None,
) -> Tuple[ResultSet, Dict[str, Any]]:
    """
    Execute nodal fatigue evaluation across a ResultSet and attach fatigue scalar fields.

    Args:
        rs: ResultSet containing stress6 Cauchy tensors.
        model: MeshModel containing part geometry.
        material: FatigueMaterial with mechanical and endurance parameters.
        load_case: FatigueLoadCase loading parameters (R, cycles, correction method, stress measure).
        target_part_name: Optional part name. If provided, evaluates primarily for this part,
                          or evaluates across all parts if None or 'ALL'.

    Returns:
        Tuple of (updated ResultSet with fatigue arrays attached, hotspot summary dictionary).
    """
    if rs.stress6 is None:
        raise ValueError("ResultSet does not contain stress tensor ('stress6') data.")

    if load_case is None:
        load_case = FatigueLoadCase()

    n_total = len(rs.stress6)
    fs_arr = np.full(n_total, np.nan, dtype=np.float64)
    life_arr = np.full(n_total, np.nan, dtype=np.float64)
    dmg_arr = np.full(n_total, np.nan, dtype=np.float64)
    sa_arr = np.full(n_total, np.nan, dtype=np.float64)
    sm_arr = np.full(n_total, np.nan, dtype=np.float64)

    # 파트 범위 결정
    parts_to_eval = []
    if target_part_name and target_part_name.upper() not in ("ALL", "전체", "전체 모델"):
        p = model.get_part(target_part_name)
        if p is not None:
            parts_to_eval.append(p)
    if not parts_to_eval:
        parts_to_eval = model.parts

    best_min_fs = float("inf")
    hotspot_global_idx = -1
    hotspot_part_name = ""
    hotspot_local_idx = -1
    hotspot_coord = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    part_fatigue_results: Dict[str, FatigueResult] = {}

    for part in parts_to_eval:
        p_slice = rs.slice_for_part(model, part.name)
        part_stress = rs.stress6[p_slice]

        # NaN이 아닌 유효 절점 확인
        valid_mask = ~np.isnan(part_stress).any(axis=1)
        if not np.any(valid_mask):
            continue

        fatigue_res = evaluate_fatigue(
            stress_max=part_stress,
            material=material,
            load_case=load_case,
            node_indices=np.arange(len(part_stress)),
        )
        part_fatigue_results[part.name] = fatigue_res

        fs_arr[p_slice] = fatigue_res.safety_factor
        life_arr[p_slice] = fatigue_res.life_cycles
        dmg_arr[p_slice] = fatigue_res.damage
        sa_arr[p_slice] = fatigue_res.sigma_a
        sm_arr[p_slice] = fatigue_res.sigma_m

        # 파트 내 최저 안전계수 노드 탐색
        part_min_fs = fatigue_res.min_safety_factor
        if part_min_fs < best_min_fs:
            best_min_fs = part_min_fs
            local_idx = fatigue_res.hotspot_node_index
            hotspot_local_idx = local_idx
            hotspot_part_name = part.name
            offset = model.global_node_offset(part.name)
            hotspot_global_idx = offset + local_idx
            if 0 <= local_idx < len(part.nodes):
                hotspot_coord = np.asarray(part.nodes[local_idx], dtype=np.float64)

    # 핫스팟 메타데이터 생성
    hotspot_region = "General"
    is_clinically_valid = True
    if hotspot_part_name:
        p_lower = hotspot_part_name.lower()
        if "screw" in p_lower:
            hotspot_region = "지대주 나사산 (Abutment Screw Thread)"
        elif "fixture" in p_lower or "implant" in p_lower:
            hotspot_region = "임플란트 고정체 목부 (Fixture Neck)"
        elif "abutment" in p_lower:
            hotspot_region = "지대주 마진/결합부 (Abutment Joint)"
        elif "bone" in p_lower or "mandible" in p_lower:
            hotspot_region = "치조골 경계부 (Crestal Bone)"
            is_clinically_valid = False

    max_stress_at_hotspot = 0.0
    if hotspot_global_idx >= 0 and rs.von_mises is not None:
        vm = rs.von_mises[hotspot_global_idx]
        max_stress_at_hotspot = float(vm) if not np.isnan(vm) else 0.0

    hotspot_info: Dict[str, Any] = {
        "part_name": hotspot_part_name,
        "node_id": int(hotspot_global_idx),
        "local_node_id": int(hotspot_local_idx),
        "coordinates": [float(c) for c in hotspot_coord],
        "min_safety_factor": float(best_min_fs) if best_min_fs != float("inf") else 0.0,
        "max_stress_MPa": max_stress_at_hotspot,
        "hotspot_region_name": hotspot_region,
        "is_clinically_valid": is_clinically_valid,
        "applied_cycles": float(load_case.target_cycles),
        "stress_ratio_R": float(load_case.stress_ratio_R),
        "mean_stress_method": str(load_case.mean_stress_method.value if hasattr(load_case.mean_stress_method, "value") else load_case.mean_stress_method),
        "stress_measure": str(load_case.stress_measure.value if hasattr(load_case.stress_measure, "value") else load_case.stress_measure),
        "material_name": str(material.name),
        "material_standard": str(material.standard),
        "fatigue_limit_Se": float(material.fatigue_limit_Se),
    }

    # ResultSet에 부착
    rs.fatigue_fs = fs_arr
    rs.fatigue_life = life_arr
    rs.fatigue_damage = dmg_arr
    rs.fatigue_sigma_a = sa_arr
    rs.fatigue_sigma_m = sm_arr
    rs.fatigue_hotspot_info = hotspot_info

    return rs, hotspot_info


def run_benchmark_case_5_3(
    mesh_size: float = 2.5,
    ccx_path: Optional[str] = None,
) -> CantileverBenchmarkResult:
    """
    Run Step 5.3 analytical & FEA cantilever benchmark validation.

    Acceptance criteria:
        - sigma_max matches M*c/I = 600.0 MPa within +/- 3.0%
        - Goodman safety factor matches 1.0951 within +/- 3.0%
    """
    return run_cantilever_benchmark(
        length_L=100.0,
        width_b=10.0,
        height_h=10.0,
        force_max=1000.0,
        force_min=100.0,
        mesh_size=mesh_size,
        tolerance_pct=3.0,
        ccx_path=ccx_path,
    )


def run_benchmark_case_5_6(
    ccx_path: Optional[str] = None,
) -> ISO14801BatchResult:
    """
    Run Step 5.6 ISO 14801 integrated fatigue test pipeline benchmark.

    Acceptance criteria:
        - 30 deg loading, 11 mm lever arm, 3 mm resorption validated.
        - Hotspot identified at clinically valid location (fixture neck or screw).
        - Predicted fatigue limit load calculated.
        - Cross-validation reported honestly.
    """
    cfg = ISO14801Config(
        load_max_N=250.0,
        stress_ratio_R=0.1,
        angle_deg=30.0,
        mode=ISO14801LoadMode.STRESS_RATIO_R,
    )
    return run_iso14801_batch_case(
        config=cfg,
        benchmark_id="gehrke_2020_conical",
        mesh_size=1.5,
        ccx_path=ccx_path,
    )
