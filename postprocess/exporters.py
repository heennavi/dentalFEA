"""
exporters.py
Headless export and storage functions for FEA simulation results.

Supports exporting nodal data to CSV, summary statistics to CSV and JSON,
and saving/loading complete result bundles (.vtu, results.npz, stats.json, meta.json).

[Units]
- Length / Displacement: mm
- Stress: MPa
- Strain: Dimensionless
- Force: N
"""

import os
import json
import csv
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

from fea.models import MeshModel
from fea.results import ResultSet
from postprocess.vtk_builder import build_part_grid, attach_results, write_vtu


def _fmt_num(val: Union[float, int, np.floating, np.integer]) -> str:
    """Format numerical value with 7 significant figures and '.' decimal."""
    if val is None or np.isnan(val):
        return "nan"
    return f"{float(val):.7g}"


def _to_serializable(val: Any) -> Any:
    """Recursively convert numpy types, NaN, and Inf to JSON-compliant Python types."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    elif isinstance(val, np.ndarray):
        return [_to_serializable(x) for x in val.tolist()]
    elif isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_to_serializable(x) for x in val]
    return val


def export_nodal_csv(path: str, rs: ResultSet, model: MeshModel) -> None:
    """
    Export all nodal results across all parts to a CSV file.

    Columns:
        global_node_id, part, x, y, z, ux, uy, uz, u_mag, sxx, syy, szz, sxy, syz, szx,
        von_mises, s_max_principal, s_min_principal

    Coordinates are the undeformed coordinates. Output encoded in UTF-8 with '.' decimal
    and 7 significant figures for numerical values. Writes part by part to conserve memory.

    Args:
        path: Target CSV file path.
        rs: ResultSet containing displacement, stress, and principal values.
        model: MeshModel containing part geometry and node coordinates.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    header = [
        "global_node_id", "part", "x", "y", "z",
        "ux", "uy", "uz", "u_mag",
        "sxx", "syy", "szz", "sxy", "syz", "szx",
        "von_mises", "s_max_principal", "s_min_principal"
    ]

    # 각 필드 존재 여부 확인 및 추출 (stress6, strain6 속성 우선 지원)
    u = rs.u
    u_mag = rs.u_mag
    s = getattr(rs, "stress6", None)
    if s is None:
        s = getattr(rs, "s", None)
    vm = rs.von_mises
    s_max = rs.s_max_principal
    s_min = rs.s_min_principal

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for part in model.parts:
            offset = model.global_node_offset(part.name)
            nodes = part.nodes
            n_part = len(nodes)

            for loc_idx in range(n_part):
                g_0idx = offset + loc_idx
                g_id = g_0idx + 1  # 1-based global node ID

                x, y, z = nodes[loc_idx]

                ux = u[g_0idx, 0] if u is not None and g_0idx < len(u) else np.nan
                uy = u[g_0idx, 1] if u is not None and g_0idx < len(u) else np.nan
                uz = u[g_0idx, 2] if u is not None and g_0idx < len(u) else np.nan
                um = u_mag[g_0idx] if u_mag is not None and g_0idx < len(u_mag) else np.nan

                if s is not None and g_0idx < len(s):
                    sxx, syy, szz = s[g_0idx, 0], s[g_0idx, 1], s[g_0idx, 2]
                    sxy, syz, szx = s[g_0idx, 3], s[g_0idx, 4], s[g_0idx, 5]
                else:
                    sxx = syy = szz = sxy = syz = szx = np.nan

                vm_val = vm[g_0idx] if vm is not None and g_0idx < len(vm) else np.nan
                s_max_val = s_max[g_0idx] if s_max is not None and g_0idx < len(s_max) else np.nan
                s_min_val = s_min[g_0idx] if s_min is not None and g_0idx < len(s_min) else np.nan

                row = [
                    g_id,
                    part.name,
                    _fmt_num(x),
                    _fmt_num(y),
                    _fmt_num(z),
                    _fmt_num(ux),
                    _fmt_num(uy),
                    _fmt_num(uz),
                    _fmt_num(um),
                    _fmt_num(sxx),
                    _fmt_num(syy),
                    _fmt_num(szz),
                    _fmt_num(sxy),
                    _fmt_num(syz),
                    _fmt_num(szx),
                    _fmt_num(vm_val),
                    _fmt_num(s_max_val),
                    _fmt_num(s_min_val),
                ]
                writer.writerow(row)


def export_stats_csv(path: str, stats: dict) -> None:
    """
    Export simulation summary statistics table to CSV.

    Corresponds to the 4-6 UI statistics table, covering max displacement,
    max von Mises, 99th percentile von Mises, mean von Mises, max tensile,
    and max compressive stress for each part and peri-implant bone.

    Args:
        path: Target CSV file path.
        stats: Summary statistics dictionary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    header = [
        "Part",
        "Max Displacement (mm)",
        "Max von Mises (MPa)",
        "99% von Mises (MPa)",
        "Mean von Mises (MPa)",
        "Max Tensile Stress (MPa)",
        "Max Compressive Stress (MPa)",
    ]

    part_stats = stats.get("parts", {})
    peri_stats = stats.get("peri_implant", None)
    radius = stats.get("radius_mm", 1.0)

    rows = []
    for part_name, st in part_stats.items():
        rows.append((part_name, st))

    if peri_stats is not None:
        label = f"Peri-implant Bone (Radius {radius:.1f} mm)"
        rows.append((label, peri_stats))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for name, st in rows:
            u_max = st.get("max_u_mag")
            vm_max = st.get("max_von_mises")
            vm_p99 = st.get("p99_von_mises")
            vm_mean = st.get("mean_von_mises")
            s_max = st.get("max_s_max_principal")
            s_min = st.get("min_s_min_principal")

            row = [
                name,
                _fmt_num(u_max),
                _fmt_num(vm_max),
                _fmt_num(vm_p99),
                _fmt_num(vm_mean),
                _fmt_num(s_max),
                _fmt_num(s_min),
            ]
            writer.writerow(row)

        # 힘 평형 점검 결과가 있으면 추가 요약 기록
        eq = stats.get("reaction_balance")
        if eq is not None:
            writer.writerow([])
            writer.writerow(["# Static Equilibrium Summary"])
            writer.writerow(["Applied Force Magnitude (N)", _fmt_num(eq.get("applied_mag"))])
            writer.writerow(["Reaction Force Magnitude (N)", _fmt_num(eq.get("reaction_mag"))])
            writer.writerow(["Residual Force Magnitude (N)", _fmt_num(eq.get("residual_mag"))])
            rel_err = eq.get("relative_error")
            rel_pct = rel_err * 100.0 if rel_err is not None else np.nan
            writer.writerow(["Relative Error (%)", _fmt_num(rel_pct)])


def export_stats_json(path: str, stats: dict) -> None:
    """
    Export statistics dictionary to JSON with safe numpy and NaN conversion.

    Args:
        path: Target JSON file path.
        stats: Summary statistics dictionary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serializable_stats = _to_serializable(stats)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)


def save_result_bundle(
    dir_path: str,
    model: MeshModel,
    rs: ResultSet,
    stats: dict,
    meta: Optional[dict] = None,
) -> None:
    """
    Save all FEA simulation assets into a standalone directory bundle.

    Directory contents:
    - <part.name>.vtu: VTK XML UnstructuredGrid for each part with result fields attached.
    - results.npz: Compressed numpy archive containing ResultSet tensor arrays.
    - stats.json: Summary statistics.
    - meta.json: Metadata including units, timestamp, source paths, mesh/load summary.

    Args:
        dir_path: Destination bundle directory.
        model: MeshModel of assembly.
        rs: ResultSet to store.
        stats: Summary statistics dictionary.
        meta: Optional metadata dict. Will be enriched with default units and timestamps.
    """
    os.makedirs(dir_path, exist_ok=True)

    # 1. 파트별 .vtu 저장 (결과 필드 부착)
    for part in model.parts:
        offset = model.global_node_offset(part.name)
        grid = build_part_grid(part, node_offset=offset, linear_only=False)
        attach_results(grid, rs, model, part.name, deform_scale=0.0)
        vtu_path = os.path.join(dir_path, f"{part.name}.vtu")
        write_vtu(grid, vtu_path)

    # 2. results.npz 저장
    arrays_to_save: Dict[str, np.ndarray] = {}

    def _add_arr(name: str, arr: Optional[np.ndarray]):
        if arr is not None:
            arrays_to_save[name] = np.asarray(arr)

    _add_arr("u", rs.u)
    _add_arr("u_mag", rs.u_mag)
    s_arr = getattr(rs, "stress6", None) if getattr(rs, "stress6", None) is not None else getattr(rs, "s", None)
    _add_arr("stress6", s_arr)
    strain_arr = getattr(rs, "strain6", None) if getattr(rs, "strain6", None) is not None else getattr(rs, "strain", None)
    _add_arr("strain6", strain_arr)
    _add_arr("rf", rs.rf)
    _add_arr("von_mises", rs.von_mises)
    _add_arr("s_max_principal", rs.s_max_principal)
    _add_arr("s_min_principal", rs.s_min_principal)
    _add_arr("equiv_strain", getattr(rs, "equiv_strain", None))

    arrays_to_save["time"] = np.array([float(getattr(rs, "time", 0.0))], dtype=np.float64)
    arrays_to_save["source_frd"] = np.array([getattr(rs, "source_frd", "") or ""], dtype=object)

    npz_path = os.path.join(dir_path, "results.npz")
    np.savez_compressed(npz_path, **arrays_to_save)

    # 3. stats.json 저장
    stats_path = os.path.join(dir_path, "stats.json")
    export_stats_json(stats_path, stats)

    # 4. meta.json 저장
    meta_dict = dict(meta or {})
    if "units" not in meta_dict:
        meta_dict["units"] = {
            "length": "mm",
            "force": "N",
            "stress": "MPa",
            "strain": "dimensionless",
            "modulus": "MPa",
        }
    if "created_at" not in meta_dict:
        meta_dict["created_at"] = datetime.now().isoformat()

    # 원본 frd 경로 상대경로화 시도
    src_frd = meta_dict.get("source_frd") or getattr(rs, "source_frd", "") or ""
    if src_frd:
        try:
            rel_src = os.path.relpath(src_frd, dir_path)
            meta_dict["source_frd_rel"] = rel_src
        except ValueError:
            meta_dict["source_frd_rel"] = src_frd
        meta_dict["source_frd"] = src_frd

    # 모델 요약 정보 보강
    if "mesh_summary" not in meta_dict:
        meta_dict["mesh_summary"] = {
            "num_parts": len(model.parts),
            "parts": [
                {
                    "name": p.name,
                    "num_nodes": len(p.nodes),
                    "num_elements": len(p.elements),
                    "element_type": p.element_type,
                }
                for p in model.parts
            ],
            "total_nodes": model.total_nodes(),
            "total_elements": model.total_elements(),
        }


    meta_path = os.path.join(dir_path, "meta.json")
    export_stats_json(meta_path, meta_dict)


def load_result_bundle(dir_path: str) -> Tuple[ResultSet, dict, dict]:
    """
    Restore ResultSet, summary statistics, and metadata from a result bundle directory.

    Note:
        MeshModel consistency validation (e.g. matching total node counts)
        must be performed by the caller against the active simulation model.

    Args:
        dir_path: Path to existing bundle directory.

    Returns:
        Tuple[ResultSet, dict, dict]: (restored_result_set, stats_dict, meta_dict)

    Raises:
        FileNotFoundError: If results.npz, stats.json, or meta.json is missing.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Result bundle directory not found: {dir_path}")

    npz_path = os.path.join(dir_path, "results.npz")
    stats_path = os.path.join(dir_path, "stats.json")
    meta_path = os.path.join(dir_path, "meta.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing results.npz in bundle: {npz_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing stats.json in bundle: {stats_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json in bundle: {meta_path}")

    # 1. stats.json 읽기
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # 2. meta.json 읽기
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 3. results.npz 읽어 ResultSet 복원
    with np.load(npz_path, allow_pickle=True) as data:
        def _get_arr(name: str) -> Optional[np.ndarray]:
            if name in data:
                return data[name]
            return None

        u = _get_arr("u")
        u_mag = _get_arr("u_mag")
        stress6 = _get_arr("stress6")
        if stress6 is None:
            stress6 = _get_arr("s")
        strain6 = _get_arr("strain6")
        if strain6 is None:
            strain6 = _get_arr("strain")
        rf = _get_arr("rf")
        vm = _get_arr("von_mises")
        s_max = _get_arr("s_max_principal")
        s_min = _get_arr("s_min_principal")

        time_val = float(data["time"][0]) if "time" in data and len(data["time"]) > 0 else 0.0

        src_frd = ""
        if "source_frd" in data and len(data["source_frd"]) > 0:
            src_frd = str(data["source_frd"][0])

        rs = ResultSet(
            time=time_val,
            u=u,
            u_mag=u_mag,
            stress6=stress6,
            von_mises=vm,
            s_max_principal=s_max,
            s_min_principal=s_min,
            strain6=strain6,
            rf=rf,
            source_frd=src_frd,
        )

    return rs, stats, meta
