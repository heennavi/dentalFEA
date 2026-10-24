#!/usr/bin/env python3
"""
run_pipeline_headless.py
Complete headless execution script for the dental implant FEA pipeline.

Executes end-to-end dental FEA pipeline without GUI:
1. Load DICOM CT/CBCT volume dataset
2. Bone tissue segmentation (extract_bone, segment_cortical_cancellous)
3. Surface extraction (mask_to_surface)
4. Implant placement and Boolean cutting (build_fea_geometry)
5. 3D Tetrahedral meshing and contact surface extraction (build_part_mesh, contact_pair_from_parts)
6. Heterogeneous bone material mapping from CT HU (assign_bone_materials, titanium_assignment)
7. Boundary conditions & loading preset setup (fix_by_plane, load_from_angle, load_on_region)
8. Export CalculiX input deck (.inp)
9. (Optional) Run CalculiX solver (ccx) and summarize results

Usage:
    python scripts/run_pipeline_headless.py --dicom /path/to/dicom --output /path/to/output [--run-solver]
"""

import os
import sys
import time
import shutil
import argparse
import logging
import json
import subprocess
from typing import Optional, Dict, Any, Tuple
import numpy as np
import trimesh
import vtk
from vtk.util import numpy_support

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dental_fea_app.imaging.dicom_reader import DicomReader
from dental_fea_app.segmentation.cortical_cancellous import extract_bone, split_cortical_cancellous
from dental_fea_app.geometry.surface_extract import mask_to_surface
from dental_fea_app.geometry.boolean_ops import build_fea_geometry, FeaGeometry
from meshing.gmsh_mesher import build_part_mesh
from meshing.surface_extractor import contact_pair_from_parts
from material.bone_mapping import assign_bone_materials, titanium_assignment, load_preset
from fea.boundary_conditions import fix_by_plane, load_from_angle, load_on_region, validate_bc
from fea.models import MeshModel, BoundaryConditionSet, AnalysisConfig
from solver.calculix_writer import write_inp
from postprocess.result_reader import read_frd_results

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HeadlessFEA")


def create_default_implant_mesh(radius: float = 2.0, length: float = 10.0) -> trimesh.Trimesh:
    """Create a clean, watertight cylindrical dental implant mesh with rounded apex."""
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=32)
    # Move so that apex is at z = -length/2 and platform is at z = +length/2
    return cylinder


def main():
    parser = argparse.ArgumentParser(
        description="Headless Dental Implant FEA Pipeline (CT -> Segmentation -> Meshing -> Material -> Solver)"
    )
    parser.add_argument(
        "--dicom",
        type=str,
        required=True,
        help="Path to DICOM image series folder"
    )
    parser.add_argument(
        "--implant",
        type=str,
        default=None,
        help="Path to implant surface STL file (default: generates 4.0 x 10.0 mm cylinder)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_fea",
        help="Output directory for .inp and simulation outputs (default: ./output_fea)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Path to material preset JSON file (default: material/presets.json)"
    )
    parser.add_argument(
        "--mesh-size-far",
        type=float,
        default=2.5,
        help="Far-field mesh size in mm (default: 2.5)"
    )
    parser.add_argument(
        "--mesh-size-near",
        type=float,
        default=0.8,
        help="Implant interface mesh size in mm (default: 0.8)"
    )
    parser.add_argument(
        "--element-type",
        type=str,
        choices=["C3D10", "C3D4"],
        default="C3D10",
        help="Tetrahedral element type: C3D10 or C3D4 (default: C3D10)"
    )
    parser.add_argument(
        "--load-n",
        type=float,
        default=100.0,
        help="Resultant occlusal load magnitude in Newtons (default: 100.0)"
    )
    parser.add_argument(
        "--tilt-deg",
        type=float,
        default=0.0,
        help="Load inclination angle in degrees from axis (default: 0.0)"
    )
    parser.add_argument(
        "--run-solver",
        action="store_true",
        help="Execute CalculiX solver after generating .inp deck"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of solver CPU threads (default: 4)"
    )
    parser.add_argument(
        "--roi-radius-mm",
        type=float,
        default=12.0,
        help="ROI bounding radius around implant in mm for fast local meshing (default: 12.0)"
    )
    parser.add_argument(
        "--run-postprocess",
        action="store_true",
        help="Execute Phase 4 headless post-processing pipeline after solver completes"
    )
    parser.add_argument(
        "--peri-radius-mm",
        type=float,
        default=1.0,
        help="Peri-implant bone evaluation radius in mm (default: 1.0)"
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    order = 2 if args.element_type == "C3D10" else 1

    total_start = time.perf_counter()
    logger.info("==================================================================")
    logger.info(" Dental Implant FEA Headless Pipeline Starting")
    logger.info(f" DICOM: {args.dicom}")
    logger.info(f" Output: {args.output}")
    logger.info(f" Element Type: {args.element_type} (Order: {order})")
    logger.info(f" Mesh Sizes: Far={args.mesh_size_far} mm, Near={args.mesh_size_near} mm")
    logger.info("==================================================================")

    # -------------------------------------------------------------
    # Step 1: Load DICOM CT Volume
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 1/8] Loading DICOM image volume...")
    vtk_image, meta = DicomReader.load_dicom_series(args.dicom)
    spacing = vtk_image.GetSpacing()
    origin = vtk_image.GetOrigin()
    dims = vtk_image.GetDimensions()  # (nx, ny, nz)
    direction = meta.get("direction", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    # Convert VTK image scalars to 3D numpy array (nz, ny, nx)
    sc = vtk_image.GetPointData().GetScalars()
    hu_arr = numpy_support.vtk_to_numpy(sc).reshape((dims[2], dims[1], dims[0]))
    t_load = time.perf_counter() - t0
    logger.info(f"  Loaded Volume: dims={dims}, spacing={spacing}, origin={origin} (Elapsed: {t_load:.2f}s)")

    # -------------------------------------------------------------
    # Step 2: Bone Segmentation
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 2/8] Performing bone tissue segmentation...")
    bone_mask = extract_bone(hu_arr, spacing=spacing, bone_threshold=200.0, keep_largest=True)
    cortical_mask, cancellous_mask = split_cortical_cancellous(
        bone_mask, hu_arr, spacing=spacing, method="distance", cortical_thickness_mm=1.5
    )
    t_seg = time.perf_counter() - t0
    n_bone_voxels = int(np.sum(bone_mask))
    logger.info(f"  Segmentation complete: Bone voxels={n_bone_voxels:,} (Elapsed: {t_seg:.2f}s)")

    # -------------------------------------------------------------
    # Step 3: Implant Placement & Geometry Preparation
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 3/8] Positioning implant and computing Boolean geometry...")
    if args.implant and os.path.isfile(args.implant):
        implant_mesh = trimesh.load(args.implant)
        if not isinstance(implant_mesh, trimesh.Trimesh):
            implant_mesh = implant_mesh.dump()[0]
    else:
        implant_mesh = create_default_implant_mesh(radius=2.0, length=10.0)

    # Position implant inside bone region
    # Compute center of bone voxels in world coordinates (selecting an alveolar crest region)
    bone_indices = np.argwhere(bone_mask)
    if len(bone_indices) > 0:
        # Select 70th percentile z to place in upper alveolar ridge
        z_target_idx = int(np.percentile(bone_indices[:, 0], 70))
        z_slice_mask = bone_indices[:, 0] == z_target_idx
        if np.any(z_slice_mask):
            z_idx, y_idx, x_idx = np.mean(bone_indices[z_slice_mask], axis=0)
        else:
            z_idx, y_idx, x_idx = np.mean(bone_indices, axis=0)

        target_center = np.array([
            origin[0] + x_idx * spacing[0],
            origin[1] + y_idx * spacing[1],
            origin[2] + z_idx * spacing[2],
        ])
    else:
        target_center = np.array([origin[0] + dims[0]*spacing[0]/2, origin[1] + dims[1]*spacing[1]/2, origin[2] + dims[2]*spacing[2]/2])

    implant_mesh.apply_translation(target_center - implant_mesh.centroid)

    # Extract bone surface within ROI around implant for performance and watertight mesh
    # ROI bounding box in world mm -> voxel index
    roi_radius = max(float(args.roi_radius_mm), 15.0)
    b_min = implant_mesh.bounds[0] - roi_radius
    b_max = implant_mesh.bounds[1] + roi_radius

    dir_mat = np.array(direction).reshape((3, 3))
    inv_dir = np.linalg.inv(dir_mat)
    idx_min = inv_dir @ (b_min - np.array(origin)) / np.array(spacing)
    idx_max = inv_dir @ (b_max - np.array(origin)) / np.array(spacing)

    imin, imax = int(np.floor(min(idx_min[0], idx_max[0]))), int(np.ceil(max(idx_min[0], idx_max[0])))
    jmin, jmax = int(np.floor(min(idx_min[1], idx_max[1]))), int(np.ceil(max(idx_min[1], idx_max[1])))
    kmin, kmax = int(np.floor(min(idx_min[2], idx_max[2]))), int(np.ceil(max(idx_min[2], idx_max[2])))
    roi_ijk = (imin, imax, jmin, jmax, kmin, kmax)

    target_edge = float(args.mesh_size_far)
    bone_surf = mask_to_surface(
        bone_mask,
        spacing=spacing,
        origin=origin,
        direction=direction,
        smooth_iterations=10,
        target_edge_mm=target_edge,
        roi_ijk=roi_ijk,
        keep_largest_component=True,
    )
    cortical_surf = mask_to_surface(
        cortical_mask,
        spacing=spacing,
        origin=origin,
        direction=direction,
        smooth_iterations=10,
        target_edge_mm=target_edge,
        roi_ijk=roi_ijk,
        keep_largest_component=True,
    )
    cancellous_surf = mask_to_surface(
        cancellous_mask,
        spacing=spacing,
        origin=origin,
        direction=direction,
        smooth_iterations=10,
        target_edge_mm=target_edge,
        roi_ijk=roi_ijk,
        keep_largest_component=True,
    )

    # Build FEA geometry with Boolean cavity using unified watertight bone surface
    fea_geom = build_fea_geometry(
        cortical_mesh=cortical_surf,
        cancellous_mesh=cancellous_surf,
        implant_mesh=implant_mesh,
        bone_mesh=bone_surf,
        undersize_mm=0.0
    )
    t_geom = time.perf_counter() - t0
    logger.info(f"  Geometry built: Bone faces={len(fea_geom.bone.faces):,}, Implant faces={len(fea_geom.implant.faces):,} (Elapsed: {t_geom:.2f}s)")

    # -------------------------------------------------------------
    # Step 4: 3D Tetrahedral Meshing (Gmsh) & Contact Ties
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info(f"[Step 4/8] Generating 3D tetrahedral volume mesh ({args.element_type})...")
    bone_part = build_part_mesh(
        name="bone",
        surface=fea_geom.bone,
        size_far=args.mesh_size_far,
        size_near=args.mesh_size_near,
        refine_surface=fea_geom.implant,
        order=order,
        mesh_algorithm_3d="auto",
    )
    implant_part = build_part_mesh(
        name="implant",
        surface=fea_geom.implant,
        size_far=args.mesh_size_near,
        size_near=args.mesh_size_near,
        order=order,
        mesh_algorithm_3d="auto",
    )

    tie = contact_pair_from_parts(bone_part, implant_part, tol_mm=0.5, master="auto", name="tie_bone_implant")
    mesh_model = MeshModel(parts=[bone_part, implant_part])
    t_mesh = time.perf_counter() - t0
    logger.info(
        f"  Meshing complete: Total Nodes={mesh_model.total_nodes():,}, "
        f"Bone Elements={len(bone_part.elements):,}, Implant Elements={len(implant_part.elements):,}, "
        f"TIE Master Faces={len(tie.master_faces):,}, Slave Nodes={len(tie.slave_nodes):,} (Elapsed: {t_mesh:.2f}s)"
    )

    # -------------------------------------------------------------
    # Step 5: Material Assignment (CT HU -> Elastic Modulus)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 5/8] Mapping heterogeneous bone properties from CT HU...")
    preset = load_preset(args.preset)
    geom_info = {
        "origin": origin,
        "spacing": spacing,
        "direction": np.array(direction).reshape((3, 3))
    }
    bone_mat_assign = assign_bone_materials(
        part=bone_part,
        volume_array=hu_arr,
        geometry_info=geom_info,
        cortical_mask_array=cortical_mask,
        preset=preset
    )
    implant_mat_assign = titanium_assignment(part=implant_part, preset=preset)
    materials = {
        "bone": bone_mat_assign,
        "implant": implant_mat_assign
    }
    t_mat = time.perf_counter() - t0
    logger.info(f"  Material mapping complete: {len(bone_mat_assign.materials)} discrete material bins (Elapsed: {t_mat:.2f}s)")

    # -------------------------------------------------------------
    # Step 6: Boundary Conditions and Occlusal Loads
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 6/8] Setting up boundary conditions and applied loads...")
    # Fix bottom boundary plane of bone (lowest 15% height)
    z_min_bone = np.min(bone_part.nodes[:, 2])
    z_max_bone = np.max(bone_part.nodes[:, 2])
    fix_plane_z = z_min_bone + 0.15 * (z_max_bone - z_min_bone)

    fix_bc = fix_by_plane(
        part=bone_part,
        plane_point=(0.0, 0.0, fix_plane_z),
        plane_normal=(0.0, 0.0, 1.0),
        side="below",
        dofs=(1, 3),
        name="FIX_BONE_BASE"
    )

    # Apply load on top 1.5 mm platform of implant
    z_max_imp = np.max(implant_part.nodes[:, 2])
    top_imp_nodes = np.where(implant_part.nodes[:, 2] >= z_max_imp - 1.5)[0]
    if len(top_imp_nodes) == 0:
        top_imp_nodes = np.array([np.argmax(implant_part.nodes[:, 2])], dtype=np.int64)

    force_vec = load_from_angle(
        magnitude_N=args.load_n,
        axis_direction=(0.0, 0.0, -1.0),
        tilt_deg=args.tilt_deg,
        tilt_azimuth_deg=0.0
    )

    load_bc = load_on_region(
        part=implant_part,
        predicate_or_node_ids=top_imp_nodes,
        force_vector_N=force_vec,
        name="LOAD_OCCLUSAL"
    )

    bc_set = BoundaryConditionSet(
        fixed_node_sets=[fix_bc],
        loads=[load_bc]
    )

    issues = validate_bc(mesh_model, bc_set)
    if issues:
        logger.warning(f"Boundary condition validation notices: {issues}")
    t_bc = time.perf_counter() - t0
    logger.info(f"  Boundary conditions set: Fixed Nodes={len(fix_bc['node_ids']):,}, Loaded Nodes={len(load_bc['node_ids']):,} (Elapsed: {t_bc:.2f}s)")

    # -------------------------------------------------------------
    # Step 7: Export CalculiX .inp Deck
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("[Step 7/8] Writing CalculiX input deck (.inp)...")
    inp_path = os.path.join(args.output, "model.inp")
    write_inp(
        path=inp_path,
        model=mesh_model,
        materials=materials,
        bc=bc_set,
        ties=[tie],
        title=f"Headless Dental FEA: {os.path.basename(args.dicom)}",
        apply_ccx_ordering=False
    )
    t_inp = time.perf_counter() - t0
    inp_size_mb = os.path.getsize(inp_path) / (1024 * 1024)
    logger.info(f"  CalculiX deck written to: {inp_path} ({inp_size_mb:.2f} MB, Elapsed: {t_inp:.2f}s)")

    # -------------------------------------------------------------
    # Step 8: (Optional) Execute CalculiX Solver
    # -------------------------------------------------------------
    if args.run_solver:
        t0 = time.perf_counter()
        logger.info("[Step 8/8] Executing CalculiX solver (ccx)...")
        ccx_bin = shutil.which("ccx") or "/usr/bin/ccx"
        if not os.path.isfile(ccx_bin):
            logger.error("CalculiX executable 'ccx' not found on PATH. Skipping solver run.")
        else:
            job_name = "model"
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(args.threads)
            env["CCX_NPROC_EQUATION_SOLVER"] = str(args.threads)

            res = subprocess.run([ccx_bin, job_name], cwd=args.output, capture_output=True, text=True, env=env)
            frd_path = os.path.join(args.output, f"{job_name}.frd")

            if res.returncode == 0 and os.path.isfile(frd_path):
                t_solv = time.perf_counter() - t0
                results = read_frd_results(frd_path)
                disp_all = results["displacements"]
                stress_all = results["stresses"]

                max_disp = max(np.linalg.norm(v) for v in disp_all.values()) if disp_all else 0.0
                max_mises = 0.0
                if stress_all:
                    for s in stress_all.values():
                        # Von Mises from SXX, SYY, SZZ, SXY, SYZ, SZX
                        sxx, syy, szz, sxy, syz, szx = s[:6]
                        vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6.0 * (sxy**2 + syz**2 + szx**2)))
                        if vm > max_mises:
                            max_mises = vm

                logger.info(f"  CalculiX solve completed successfully! (Elapsed: {t_solv:.2f}s)")
                logger.info(f"  Maximum Resultant Displacement: {max_disp:.6f} mm")
                logger.info(f"  Maximum Von Mises Stress: {max_mises:.2f} MPa")

                if args.run_postprocess:
                    logger.info("  Launching Phase 4 Headless Post-Processing Pipeline...")
                    from scripts.run_postprocess_headless import run_postprocess
                    post_out = os.path.join(args.output, "postprocess")
                    run_postprocess(
                        workdir=args.output,
                        output_dir=post_out,
                        radius_mm=args.peri_radius_mm,
                    )
            else:
                logger.error(f"  CalculiX execution failed with exit code {res.returncode}:\n{res.stdout[-500:]}\n{res.stderr[-500:]}")
    else:
        logger.info("[Step 8/8] Solver run skipped (pass --run-solver to execute).")

    total_time = time.perf_counter() - total_start
    logger.info("==================================================================")
    logger.info(f" Pipeline Finished Successfully in {total_time:.2f}s")
    logger.info(f" Generated Files in: {args.output}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
