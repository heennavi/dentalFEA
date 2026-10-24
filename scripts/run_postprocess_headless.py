#!/usr/bin/env python3
"""
run_postprocess_headless.py
Complete headless post-processing script for dental FEA simulation results.

Flow:
1. Locate and parse CalculiX ASCII .frd result file
2. Restore or reconstruct MeshModel (from --model bundle, or by parsing .inp in workdir)
3. Convert raw results to aligned ResultSet (displacements, stresses, principal stresses, reaction forces)
4. Compute part-wise summary statistics (max/mean/P95/P99)
5. Calculate static reaction force balance equilibrium
6. Evaluate peri-implant bone stress distribution within specified radius
7. Export comprehensive artifacts:
   - nodal_results.csv (all nodes, undeformed LPS coords, 18 fields)
   - summary_statistics.csv & summary_statistics.json
   - Full result bundle (.vtu per part, results.npz, stats.json, meta.json)

Usage:
    python scripts/run_postprocess_headless.py --workdir /path/to/job_dir --output /path/to/output [--radius 1.0]
"""

import os
import sys
import time
import argparse
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fea.models import MeshModel, MeshPart, BoundaryConditionSet
from fea.results import ResultSet
from postprocess.frd_parser import parse_frd
from postprocess.result_fields import build_result_set, part_statistics, reaction_balance
from postprocess.peri_implant import peri_implant_stats
from postprocess.exporters import (
    export_nodal_csv,
    export_stats_csv,
    export_stats_json,
    save_result_bundle,
    load_result_bundle,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HeadlessPostprocess")


def load_mesh_from_inp(inp_path: str) -> Tuple[MeshModel, BoundaryConditionSet]:
    """
    Reconstruct MeshModel and basic BoundaryConditionSet directly from a CalculiX .inp deck.

    Provides a standalone recovery mechanism when a dedicated serialized MeshModel file
    is not supplied. Reads *NODE, *ELEMENT, *NSET, *BOUNDARY, and *CLOAD cards.

    Args:
        inp_path: Path to CalculiX .inp file.

    Returns:
        Tuple[MeshModel, BoundaryConditionSet]: Reconstructed model and boundary conditions.
    """
    if not os.path.isfile(inp_path):
        raise FileNotFoundError(f"CalculiX input deck not found: {inp_path}")

    node_dict: Dict[int, list] = {}  # 1-based node_id -> [x, y, z]
    part_elements: Dict[str, list] = {}  # part_name -> list of element node lists (1-based)
    part_elem_types: Dict[str, str] = {}
    node_sets: Dict[str, list] = {}

    current_mode = None
    current_part = "part"
    current_nset = ""
    current_elem_type = "C3D10"

    bc_fixed: list = []
    bc_loads: list = []

    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    idx = 0
    n_lines = len(lines)
    while idx < n_lines:
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("**"):
            continue

        if line.startswith("*"):
            upper = line.upper()
            if upper.startswith("*NODE"):
                current_mode = "NODE"
            elif upper.startswith("*ELEMENT"):
                current_mode = "ELEMENT"
                # Extract TYPE and ELSET
                tokens = [t.strip() for t in line.split(",")]
                etype = "C3D10"
                elset = "part"
                for tok in tokens:
                    if tok.upper().startswith("TYPE="):
                        etype = tok.split("=")[1].strip()
                    elif tok.upper().startswith("ELSET="):
                        raw_set = tok.split("=")[1].strip()
                        # Extract part name from EALL_<NAME> or ELSET_<NAME>
                        if raw_set.upper().startswith("EALL_"):
                            elset = raw_set[5:].lower()
                        else:
                            elset = raw_set.lower()
                current_part = elset
                current_elem_type = etype
                if current_part not in part_elements:
                    part_elements[current_part] = []
                    part_elem_types[current_part] = current_elem_type
            elif upper.startswith("*NSET"):
                current_mode = "NSET"
                tokens = [t.strip() for t in line.split(",")]
                nset_name = "SET"
                for tok in tokens:
                    if tok.upper().startswith("NSET="):
                        nset_name = tok.split("=")[1].strip().upper()
                current_nset = nset_name
                if current_nset not in node_sets:
                    node_sets[current_nset] = []
            elif upper.startswith("*BOUNDARY"):
                current_mode = "BOUNDARY"
            elif upper.startswith("*CLOAD"):
                current_mode = "CLOAD"
            else:
                current_mode = None
            continue

        if current_mode == "NODE":
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 4:
                nid = int(parts[0])
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                node_dict[nid] = coords

        elif current_mode == "ELEMENT":
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if current_elem_type == "C3D10":
                # Check if continued on next line (line ends with comma or has 9 items)
                if len(parts) == 9:
                    # Next line contains remaining 2 nodes
                    if idx < n_lines:
                        next_parts = [p.strip() for p in lines[idx].split(",") if p.strip()]
                        idx += 1
                        parts.extend(next_parts)
                if len(parts) >= 11:
                    conn = [int(p) for p in parts[1:11]]
                    part_elements[current_part].append(conn)
            elif current_elem_type == "C3D4":
                if len(parts) >= 5:
                    conn = [int(p) for p in parts[1:5]]
                    part_elements[current_part].append(conn)

        elif current_mode == "NSET":
            items = [int(p.strip()) for p in line.split(",") if p.strip().isdigit()]
            node_sets[current_nset].extend(items)

        elif current_mode == "BOUNDARY":
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            if len(tokens) >= 3:
                target = tokens[0]
                dof_start = int(tokens[1])
                dof_end = int(tokens[2])
                if target in node_sets:
                    bc_fixed.append({
                        "name": target,
                        "part": "",
                        "node_ids": np.array(node_sets[target], dtype=np.int64),
                        "dofs": (dof_start, dof_end)
                    })
                elif target.isdigit():
                    bc_fixed.append({
                        "name": f"NODE_{target}",
                        "part": "",
                        "node_ids": np.array([int(target)], dtype=np.int64),
                        "dofs": (dof_start, dof_end)
                    })

        elif current_mode == "CLOAD":
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            if len(tokens) >= 3:
                nid = int(tokens[0])
                dof = int(tokens[1])
                val = float(tokens[2])
                f_vec = [0.0, 0.0, 0.0]
                if 1 <= dof <= 3:
                    f_vec[dof - 1] = val
                bc_loads.append({
                    "name": f"LOAD_NODE_{nid}",
                    "part": "",
                    "node_ids": np.array([nid], dtype=np.int64),
                    "force_N": tuple(f_vec)
                })

    if not node_dict:
        raise ValueError(f"No nodal coordinates found in {inp_path}")

    # Sort node coordinates by 1-based node_id
    sorted_nids = sorted(node_dict.keys())

    # Reconstruct MeshPart instances
    parts_list = []
    gnid_to_part_local: Dict[int, Tuple[str, int]] = {}

    if part_elements:
        for pname, elem_conns in part_elements.items():
            if not elem_conns:
                continue
            used_nids = sorted(set(n for conn in elem_conns for n in conn))
            local_nid_map = {nid: i for i, nid in enumerate(used_nids)}
            for nid, l_idx in local_nid_map.items():
                gnid_to_part_local[nid] = (pname, l_idx)

            part_nodes = np.array([node_dict[nid] for nid in used_nids], dtype=np.float64)
            part_elems = np.array([[local_nid_map[n] for n in conn] for conn in elem_conns], dtype=np.int64)
            etype = part_elem_types.get(pname, "C3D10")
            parts_list.append(MeshPart(name=pname, nodes=part_nodes, elements=part_elems, element_type=etype))
    else:
        # Fallback to single part
        all_nodes = np.array([node_dict[nid] for nid in sorted_nids], dtype=np.float64)
        for i, nid in enumerate(sorted_nids):
            gnid_to_part_local[nid] = ("model", i)
        parts_list.append(MeshPart(name="model", nodes=all_nodes, elements=np.zeros((0, 4), dtype=np.int64), element_type="C3D4"))

    # Map raw BC global node IDs to part-specific local indices
    mapped_fixed = []
    default_pname = parts_list[0].name if parts_list else "model"
    for item in bc_fixed:
        raw_nids = item.get("node_ids", [])
        # Group by part
        part_groups: Dict[str, list] = {}
        for gnid in raw_nids:
            pname, l_idx = gnid_to_part_local.get(int(gnid), (default_pname, int(gnid) - 1))
            if pname not in part_groups:
                part_groups[pname] = []
            part_groups[pname].append(l_idx)
        for pname, local_list in part_groups.items():
            mapped_fixed.append({
                "name": item.get("name", "FIX"),
                "part": pname,
                "node_ids": np.array(local_list, dtype=np.int64),
                "dofs": item.get("dofs", (1, 3))
            })

    mapped_loads = []
    for item in bc_loads:
        raw_nids = item.get("node_ids", [])
        part_groups: Dict[str, list] = {}
        for gnid in raw_nids:
            pname, l_idx = gnid_to_part_local.get(int(gnid), (default_pname, int(gnid) - 1))
            if pname not in part_groups:
                part_groups[pname] = []
            part_groups[pname].append(l_idx)
        for pname, local_list in part_groups.items():
            mapped_loads.append({
                "name": item.get("name", "LOAD"),
                "part": pname,
                "node_ids": np.array(local_list, dtype=np.int64),
                "force_N": item.get("force_N", (0.0, 0.0, 0.0))
            })

    model = MeshModel(parts=parts_list)
    bc_set = BoundaryConditionSet(fixed_node_sets=mapped_fixed, loads=mapped_loads)
    return model, bc_set


def run_postprocess(
    workdir: str,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    radius_mm: float = 1.0,
) -> Dict[str, Any]:
    """
    Execute end-to-end headless post-processing pipeline.

    Args:
        workdir: Directory containing .inp and .frd files.
        output_dir: Directory to save generated post-processing artifacts.
        model_path: Optional path to MeshModel bundle or serialized file.
        radius_mm: Peri-implant bone sampling radius in mm (default: 1.0).

    Returns:
        Dict[str, Any]: Overall post-processing summary and execution metrics.
    """
    total_start = time.perf_counter()
    workdir = os.path.abspath(workdir)
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"Specified workdir does not exist: {workdir}")

    if output_dir is None:
        output_dir = os.path.join(workdir, "postprocess")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("==================================================================")
    logger.info(" Dental FEA Headless Post-Processing Pipeline")
    logger.info(f" Workdir: {workdir}")
    logger.info(f" Output:  {output_dir}")
    logger.info(f" Radius:  {radius_mm:.1f} mm")
    logger.info("==================================================================")

    # 1. Locate .frd file
    frd_candidates = [f for f in os.listdir(workdir) if f.endswith(".frd")]
    if not frd_candidates:
        raise FileNotFoundError(f"No .frd result file found in workdir: {workdir}")
    frd_file = os.path.join(workdir, sorted(frd_candidates)[0])

    # 2. Locate .inp file
    inp_candidates = [f for f in os.listdir(workdir) if f.endswith(".inp")]
    inp_file = os.path.join(workdir, sorted(inp_candidates)[0]) if inp_candidates else None

    # -------------------------------------------------------------
    # Step 1: MeshModel Recovery
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 1/6] Reconstructing MeshModel geometry...")
    model: Optional[MeshModel] = None
    bc: Optional[BoundaryConditionSet] = None

    if model_path and os.path.isdir(model_path):
        # Attempt loading from result bundle if existing
        try:
            _, stats, meta = load_result_bundle(model_path)
            logger.info(f"  Loaded model metadata from bundle: {model_path}")
        except Exception as e:
            logger.warning(f"  Could not load from bundle ({e}), falling back to .inp deck.")

    if model is None and inp_file:
        model, bc = load_mesh_from_inp(inp_file)
        logger.info(
            f"  Reconstructed model from .inp deck: Parts={len(model.parts)}, "
            f"Total Nodes={model.total_nodes():,}, Total Elements={model.total_elements():,}"
        )
    elif model is None:
        raise RuntimeError(
            "Could not reconstruct MeshModel. Neither a valid model bundle nor a .inp deck was found. "
            "Proposal: Phase 3 did not include a standalone MeshModel serialization format. "
            "It is recommended to introduce save_mesh_model_npz / load_mesh_model_npz in fea/model_io.py."
        )

    t_mesh = time.perf_counter() - t0
    logger.info(f"  MeshModel ready (Elapsed: {t_mesh:.3f}s)")

    # -------------------------------------------------------------
    # Step 2: Parse CalculiX .frd File
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info(f"[Step 2/6] Parsing CalculiX result file: {os.path.basename(frd_file)}...")
    frd_data = parse_frd(frd_file)
    t_parse = time.perf_counter() - t0
    logger.info(
        f"  Parsed {len(frd_data.node_ids):,} nodes and {len(frd_data.blocks)} result blocks "
        f"(Elapsed: {t_parse:.3f}s, Throughput: {len(frd_data.node_ids)/t_parse:,.0f} nodes/s)"
    )

    # -------------------------------------------------------------
    # Step 3: Build Aligned ResultSet
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 3/6] Aligning fields and computing derived stresses...")
    rs = build_result_set(frd_data, model, step="last")
    t_build = time.perf_counter() - t0
    logger.info(
        f"  ResultSet built successfully: u={rs.u is not None}, "
        f"stress6={rs.stress6 is not None}, rf={rs.rf is not None} (Elapsed: {t_build:.3f}s)"
    )

    # -------------------------------------------------------------
    # Step 4: Part Statistics & Force Balance
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 4/6] Evaluating part statistics and reaction force balance...")
    all_part_stats = {}
    for part in model.parts:
        pstats = part_statistics(rs, model, part.name)
        all_part_stats[part.name] = pstats
        logger.info(
            f"  Part '{part.name}': Max Displacement={pstats['max_u_mag']:.4f} mm, "
            f"Max von Mises={pstats['max_von_mises']:.2f} MPa, P99={pstats['p99_von_mises']:.2f} MPa"
        )

    bal = None
    if bc is not None and rs.rf is not None:
        bal = reaction_balance(rs, bc, model)
        if bal:
            rel_err_pct = bal["relative_error"] * 100.0
            logger.info(
                f"  Force Equilibrium: Applied={bal['applied_mag']:.2f} N, "
                f"Reaction={bal['reaction_mag']:.2f} N, Relative Error={rel_err_pct:.4f}%"
            )

    # -------------------------------------------------------------
    # Step 5: Peri-Implant Bone Stress Evaluation
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info(f"[Step 5/6] Analyzing peri-implant bone stress within {radius_mm:.1f} mm radius...")
    peri_stats = peri_implant_stats(
        rs=rs,
        model=model,
        radius_mm=radius_mm,
    )
    t_peri = time.perf_counter() - t0
    if peri_stats.get("num_nodes", 0) > 0:
        logger.info(
            f"  Peri-implant bone ({peri_stats['num_nodes']:,} nodes): "
            f"Mean von Mises={peri_stats['mean_von_mises']:.2f} MPa, "
            f"P95={peri_stats['p95_von_mises']:.2f} MPa, "
            f"P99={peri_stats['p99_von_mises']:.2f} MPa (Elapsed: {t_peri:.3f}s)"
        )
    else:
        logger.info(f"  Peri-implant analysis skipped (single part or no implant interface detected).")

    # -------------------------------------------------------------
    # Step 6: Export Results & Artifacts
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 6/6] Exporting CSV, JSON, and simulation result bundle...")

    # 1. Nodal CSV
    csv_nodal = os.path.join(output_dir, "nodal_results.csv")
    export_nodal_csv(csv_nodal, rs, model)

    # 2. Summary stats CSV & JSON
    combined_stats = {
        "parts": all_part_stats,
        "reaction_balance": bal,
        "peri_implant": peri_stats,
        "total_nodes": model.total_nodes(),
        "total_elements": model.total_elements(),
    }
    stats_csv = os.path.join(output_dir, "summary_statistics.csv")
    stats_json = os.path.join(output_dir, "summary_statistics.json")
    export_stats_csv(stats_csv, combined_stats)
    export_stats_json(stats_json, combined_stats)

    # 3. Complete bundle (.vtu, results.npz, meta.json, stats.json)
    bundle_dir = os.path.join(output_dir, "result_bundle")
    meta = {
        "source_frd": frd_file,
        "source_inp": inp_file,
        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "peri_implant_radius_mm": radius_mm,
    }
    save_result_bundle(bundle_dir, model, rs, combined_stats, meta)
    t_export = time.perf_counter() - t0
    logger.info(f"  Artifacts saved to {output_dir} (Elapsed: {t_export:.3f}s)")

    total_time = time.perf_counter() - total_start
    logger.info("==================================================================")
    logger.info(f" Post-processing Finished Successfully in {total_time:.2f}s")
    logger.info("==================================================================")

    return {
        "total_time_s": total_time,
        "parts": all_part_stats,
        "reaction_balance": bal,
        "peri_implant": peri_stats,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Headless Post-Processing Pipeline for Dental FEA Simulation Results"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        help="Path to folder containing .inp and .frd files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output directory (default: <workdir>/postprocess)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional path to MeshModel bundle or model file"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Peri-implant bone evaluation radius in mm (default: 1.0)"
    )

    args = parser.parse_args()
    try:
        run_postprocess(
            workdir=args.workdir,
            output_dir=args.output,
            model_path=args.model,
            radius_mm=args.radius,
        )
    except Exception as e:
        logger.error(f"Post-processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
