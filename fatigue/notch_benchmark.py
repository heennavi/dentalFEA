"""
notch_benchmark.py
Analytical benchmark and CalculiX FEA validation pipeline for notched members.
Validates stress concentration factor (Kt) and mesh sensitivity (2x refinement convergence)
against classical literature (Peterson's Stress Concentration Factors, Howland 1930, Ling 1947).
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gmsh

from fea.models import (
    MeshModel,
    MeshPart,
    MaterialDef,
    MaterialAssignment,
    BoundaryConditionSet,
)
from meshing.gmsh_mesher import gmsh_tet10_to_ccx, fix_orientation
from solver.calculix_writer import write_inp
from .models import FatigueMaterial
from .io_frd import extract_stress_from_frd
from .stress_measures import calculate_principal_stresses, calculate_von_mises


@dataclass
class NotchSpec:
    """Geometric and loading specifications for a notched tensile specimen."""
    notch_type: str = "semicircular_plate"  # "semicircular_plate" or "hole_plate"
    length_L: float = 80.0                  # Total length (mm) along X
    width_W: float = 20.0                   # Gross width (mm) along Y
    thickness_t: float = 2.0                # Thickness (mm) along Z
    notch_radius_r: float = 2.5             # Notch root radius (mm)
    notch_depth_h: float = 2.5              # Notch depth (mm) (h = r for semicircular)
    tensile_force_N: float = 1500.0         # Axial tensile force (N)


@dataclass
class NotchTheory:
    """Theoretical analytical quantities and literature stress concentration factor."""
    spec: NotchSpec
    net_width_d: float                      # Minimum width across notch roots (mm)
    gross_area_Ag: float                    # Gross cross-sectional area (mm^2)
    net_area_An: float                      # Net cross-sectional area (mm^2)
    sigma_gross: float                      # Gross nominal stress (MPa)
    sigma_nom: float                        # Net-section nominal stress (MPa)
    kt_theory: float                        # Theoretical stress concentration factor (sigma_max / sigma_nom)
    sigma_max_theory: float                 # Theoretical maximum peak stress (MPa)
    literature_source: str                  # Citation / source reference


@dataclass
class NotchFEAResult:
    """FEA simulation result for a specific mesh resolution level."""
    mesh_level: str                         # "coarse" or "fine"
    mesh_size_notch: float                  # Target element size at notch root (mm)
    num_nodes: int                          # Total number of nodes
    num_elements: int                       # Total number of C3D10 tetrahedral elements
    sigma_max_fea: float                    # Maximum principal / tensile stress from FEA (MPa)
    sigma_vm_max_fea: float                 # Maximum von Mises equivalent stress from FEA (MPa)
    kt_fea: float                           # FEA stress concentration factor (sigma_max_fea / sigma_nom)
    kt_error_pct: float                     # Relative error vs literature Kt (%)
    passed_accuracy: bool                   # True if |kt_error_pct| <= tolerance_pct


@dataclass
class NotchBenchmarkResult:
    """Complete quantitative evaluation of stress concentration and mesh convergence."""
    theory: NotchTheory
    coarse_result: NotchFEAResult
    fine_result: NotchFEAResult
    stress_change_pct: float                # |sigma_fine - sigma_coarse| / sigma_fine * 100 (%)
    passed_convergence: bool                # True if stress_change_pct < convergence_tolerance_pct
    passed_accuracy: bool                   # True if fine_result.passed_accuracy is True
    accuracy_tolerance_pct: float = 5.0     # Acceptance threshold for Kt error (+/- 5.0%)
    convergence_tolerance_pct: float = 5.0  # Acceptance threshold for 2x mesh refinement (< 5.0%)
    notch_sensitivity_q: float = 0.0        # Neuber/Peterson notch sensitivity index q in [0, 1]
    fatigue_notch_factor_kf: float = 0.0    # Fatigue notch factor Kf = 1 + q * (Kt - 1)

    @property
    def passed(self) -> bool:
        """Overall pass status requiring both accuracy and mesh convergence criteria."""
        return self.passed_accuracy and self.passed_convergence


def calculate_notch_theory(spec: NotchSpec) -> NotchTheory:
    """Calculate theoretical nominal stress and literature stress concentration factor Kt.

    Supported geometries:
    1. 'semicircular_plate': Flat bar with opposite symmetrical semicircular notches under tension.
       Literature: Peterson's Stress Concentration Factors (Pilkey 3rd ed, Chart 2.4, Kikukawa 1962, Flynn 1966).
       Polynomial curve-fit for net-section Kt as a function of lambda = 2r / W:
       Kt = 3.060 - 3.765*lambda + 4.849*lambda^2 - 3.306*lambda^3

    2. 'hole_plate': Flat plate with a central circular hole under tension.
       Literature: Howland (1930) / Peterson Eq. 3.1:
       Kt = 3.00 - 3.14*(d/W) + 3.67*(d/W)^2 - 1.53*(d/W)^3

    Args:
        spec: Notched tensile specimen specification.

    Returns:
        NotchTheory instance containing analytical nominal and peak stresses and Kt.
    """
    W = float(spec.width_W)
    t = float(spec.thickness_t)
    r = float(spec.notch_radius_r)
    h = float(spec.notch_depth_h)
    F = float(spec.tensile_force_N)

    gross_area = W * t

    if spec.notch_type == "hole_plate":
        d_hole = 2.0 * r
        net_width = W - d_hole
        net_area = net_width * t
        sigma_gross = F / gross_area
        sigma_nom = F / net_area

        # Howland (1930) analytical series expansion
        d_W = d_hole / W
        if d_W >= 0.7:
            raise ValueError(f"Hole diameter ratio d/W={d_W:.3f} exceeds valid analytical limit 0.7")
        kt_theory = 3.00 - 3.14 * d_W + 3.67 * (d_W ** 2) - 1.53 * (d_W ** 3)
        source = "Howland (1930) / Peterson Eq. 3.1"

    elif spec.notch_type == "semicircular_plate":
        # Opposite semicircular notches
        d_net = W - 2.0 * h
        net_width = d_net
        net_area = net_width * t
        sigma_gross = F / gross_area
        sigma_nom = F / net_area

        # Peterson / Flynn & Roll (1966) / Kikukawa (1962) curve fit
        lam = (2.0 * r) / W
        if lam > 0.6:
            raise ValueError(f"Notch ratio 2r/W={lam:.3f} exceeds valid analytical limit 0.6")

        coeffs = [-3.306367, 4.849308, -3.764832, 3.059757]
        kt_theory = float(np.polyval(coeffs, lam))
        source = "Peterson's Stress Concentration Factors (Pilkey Chart 2.4 / Flynn & Roll 1966)"

    else:
        raise ValueError(f"Unsupported notch_type: '{spec.notch_type}'. Must be 'semicircular_plate' or 'hole_plate'.")

    sigma_max_theory = kt_theory * sigma_nom

    return NotchTheory(
        spec=spec,
        net_width_d=net_width,
        gross_area_Ag=gross_area,
        net_area_An=net_area,
        sigma_gross=sigma_gross,
        sigma_nom=sigma_nom,
        kt_theory=float(kt_theory),
        sigma_max_theory=float(sigma_max_theory),
        literature_source=source,
    )


def calculate_notch_sensitivity(
    kt: float,
    notch_radius_r: float,
    tensile_strength_Sut: float = 860.0,
) -> Tuple[float, float]:
    """Calculate Neuber-Peterson notch sensitivity index q and fatigue notch factor Kf.

    Formulas:
        Peterson's material constant a (mm):
        a = 0.0254 * (2070.0 / Sut_MPa)^1.8  (for high-strength steels / titanium)
        q = 1 / (1 + a / r)
        Kf = 1 + q * (Kt - 1)

    Args:
        kt: Theoretical elastic stress concentration factor.
        notch_radius_r: Notch root radius in mm.
        tensile_strength_Sut: Ultimate tensile strength in MPa (default: Ti-6Al-4V ~860 MPa).

    Returns:
        Tuple of (q, Kf).
    """
    if notch_radius_r <= 0.0:
        return 0.0, 1.0

    a_param = 0.0254 * ((2070.0 / max(tensile_strength_Sut, 100.0)) ** 1.8)
    q = 1.0 / (1.0 + (a_param / notch_radius_r))
    q = float(np.clip(q, 0.0, 1.0))
    kf = 1.0 + q * (kt - 1.0)
    return float(q), float(kf)


def build_notched_plate_mesh(
    spec: NotchSpec,
    mesh_size_notch: float = 0.8,
    mesh_size_far: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 3D C3D10 quadratic tetrahedral mesh for notched plate via Gmsh OpenCASCADE.

    Uses OpenCASCADE boolean geometry to preserve exact analytic boundary curvature
    without faceted STL discretization errors. Applies localized refinement around the notch.

    Args:
        spec: Geometric specimen specifications.
        mesh_size_notch: Target element edge length at the notch root (mm).
        mesh_size_far: Target element edge length in the unnotched bulk region (mm).

    Returns:
        Tuple of (nodes, elements) where:
            nodes: (N, 3) float64 array of nodal coordinates in mm.
            elements: (M, 10) int64 array of C3D10 elements in CalculiX node ordering.
    """
    L = float(spec.length_L)
    W = float(spec.width_W)
    t = float(spec.thickness_t)
    r = float(spec.notch_radius_r)

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize(interruptible=False)

    try:
        gmsh.model.add(f"notch_{spec.notch_type}")
        occ = gmsh.model.occ

        # Base rectangular slab centered at origin: X in [-L/2, L/2], Y in [-W/2, W/2], Z in [-t/2, t/2]
        box = occ.addBox(-L / 2.0, -W / 2.0, -t / 2.0, L, W, t)

        if spec.notch_type == "hole_plate":
            # Central circular cylinder cutting along Z
            cutter = occ.addCylinder(0.0, 0.0, -t, 0.0, 0.0, 2.0 * t, r)
            occ.cut([(3, box)], [(3, cutter)])
        else:
            # Semicircular notches at Y = +W/2 and Y = -W/2
            c1 = occ.addCylinder(0.0, W / 2.0, -t, 0.0, 0.0, 2.0 * t, r)
            c2 = occ.addCylinder(0.0, -W / 2.0, -t, 0.0, 0.0, 2.0 * t, r)
            occ.cut([(3, box)], [(3, c1), (3, c2)])

        occ.synchronize()

        # Localized mesh refinement box enclosing the notch region (|x| <= r * 2.5)
        f_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f_box, "VIn", float(mesh_size_notch))
        gmsh.model.mesh.field.setNumber(f_box, "VOut", float(mesh_size_far))
        gmsh.model.mesh.field.setNumber(f_box, "XMin", -max(r * 2.5, 4.0))
        gmsh.model.mesh.field.setNumber(f_box, "XMax", max(r * 2.5, 4.0))
        gmsh.model.mesh.field.setNumber(f_box, "YMin", -W * 1.5)
        gmsh.model.mesh.field.setNumber(f_box, "YMax", W * 1.5)
        gmsh.model.mesh.field.setNumber(f_box, "ZMin", -t * 1.5)
        gmsh.model.mesh.field.setNumber(f_box, "ZMax", t * 1.5)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_box)

        # Gmsh meshing configuration
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 1)    # 2D MeshAdapt
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 3D Delaunay
        gmsh.option.setNumber("Mesh.ElementOrder", 2) # C3D10 10-node tetrahedra
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

        gmsh.model.mesh.generate(3)

        # Extract nodes and connectivity
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.ascontiguousarray(node_coords.reshape((-1, 3)), dtype=np.float64)
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
        if 11 not in elem_types:
            raise RuntimeError("Gmsh failed to generate 10-node tetrahedral elements (type 11).")

        target_idx = list(elem_types).index(11)
        raw_conn = elem_node_tags[target_idx]
        mapped_conn = np.array([tag_to_idx[int(t)] for t in raw_conn], dtype=np.int64).reshape((-1, 10))

        # Ensure positive Jacobian orientation and convert Gmsh tet10 ordering to CalculiX C3D10
        elements = fix_orientation(nodes, mapped_conn)
        elements = gmsh_tet10_to_ccx(elements)

    finally:
        if gmsh.isInitialized():
            gmsh.finalize()

    return nodes, elements


def build_notch_model_and_bc(
    spec: NotchSpec,
    mesh_size_notch: float = 0.8,
    mesh_size_far: float = 2.0,
    material: Optional[FatigueMaterial] = None,
) -> Tuple[MeshModel, MaterialAssignment, BoundaryConditionSet]:
    """Build MeshModel and boundary conditions for uniaxial tension of a notched plate.

    Boundary conditions:
    - Left face (x = -L/2): Fixed in X (dof 1).
    - Left face centroid: Fixed in Y and Z (dofs 2, 3) to prevent rigid body translation.
    - Left face edge: Fixed in Z (dof 3) to prevent rigid body roll around X.
    - Right face (x = +L/2): Uniform tensile force F applied along +X.

    Args:
        spec: Geometric specimen specifications.
        mesh_size_notch: Local mesh size at notch root.
        mesh_size_far: Far-field bulk mesh size.
        material: FatigueMaterial instance (default: Ti-6Al-4V).

    Returns:
        Tuple of (model, material_assignment, boundary_condition_set).
    """
    if material is None:
        material = FatigueMaterial(name="Ti-6Al-4V")

    nodes, elements = build_notched_plate_mesh(
        spec,
        mesh_size_notch=mesh_size_notch,
        mesh_size_far=mesh_size_far,
    )

    part = MeshPart(name="notched_plate", nodes=nodes, elements=elements, element_type="C3D10")
    model = MeshModel(parts=[part])

    L = float(spec.length_L)
    tol = 1e-3
    x_left = np.where(np.abs(nodes[:, 0] - (-L / 2.0)) < tol)[0]
    x_right = np.where(np.abs(nodes[:, 0] - (L / 2.0)) < tol)[0]

    if len(x_left) == 0 or len(x_right) == 0:
        raise RuntimeError("Failed to detect boundary nodes on left or right tensile faces.")

    # Kinematic rigid-body restraints on left face
    center_idx = x_left[np.argmin(np.linalg.norm(nodes[x_left, 1:], axis=1))]
    edge_idx = x_left[np.argmax(np.abs(nodes[x_left, 1]))]

    bc_fixed = [
        {"name": "FIX_X_LEFT", "part": "notched_plate", "node_ids": x_left, "dofs": (1, 1)},
        {"name": "FIX_YZ_CENTER", "part": "notched_plate", "node_ids": np.array([center_idx]), "dofs": (2, 3)},
        {"name": "FIX_Z_ROLL", "part": "notched_plate", "node_ids": np.array([edge_idx]), "dofs": (3, 3)},
    ]

    loads = [
        {
            "name": "TENSION_RIGHT",
            "part": "notched_plate",
            "node_ids": x_right,
            "force_N": (float(spec.tensile_force_N), 0.0, 0.0),
        }
    ]

    bc = BoundaryConditionSet(fixed_node_sets=bc_fixed, loads=loads)

    mat_def = MaterialDef(name=material.name, E=material.E, nu=material.nu)
    mat_assign = MaterialAssignment(
        element_material={"notched_plate": np.zeros(len(elements), dtype=np.int64)},
        materials=[mat_def],
    )

    return model, mat_assign, bc


def _solve_notch_fea(
    model: MeshModel,
    mat_assign: MaterialAssignment,
    bc: BoundaryConditionSet,
    spec: NotchSpec,
    mesh_level: str,
    mesh_size_notch: float,
    theory: NotchTheory,
    ccx_path: str = "/usr/bin/ccx",
    tolerance_pct: float = 5.0,
) -> NotchFEAResult:
    """Execute CalculiX simulation, parse .frd, and extract notch peak stress and Kt."""
    part = model.parts[0]
    nodes = part.nodes

    with tempfile.TemporaryDirectory() as tmpdir:
        job_name = f"notch_{mesh_level}"
        inp_path = os.path.join(tmpdir, f"{job_name}.inp")
        frd_path = os.path.join(tmpdir, f"{job_name}.frd")

        write_inp(
            inp_path,
            model,
            {"notched_plate": mat_assign},
            bc,
            ties=[],
            apply_ccx_ordering=False,
        )

        res = subprocess.run([ccx_path, job_name], cwd=tmpdir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"CalculiX notch simulation failed ({job_name}):\n{res.stdout}\n{res.stderr}")

        node_ids, stress6 = extract_stress_from_frd(frd_path, which="last")
        p1, p2, p3 = calculate_principal_stresses(stress6)
        vm = calculate_von_mises(stress6)

        # Filter nodes strictly located within the notch concentration vicinity (|x| <= r * 2.0)
        r = float(spec.notch_radius_r)
        notch_mask = np.abs(nodes[:, 0]) <= max(r * 2.0, 3.0)

        s1_notch = p1[notch_mask]
        vm_notch = vm[notch_mask]

        sigma_max_fea = float(np.max(s1_notch))
        sigma_vm_max_fea = float(np.max(vm_notch))

        kt_fea = sigma_max_fea / theory.sigma_nom
        kt_error_pct = (kt_fea - theory.kt_theory) / theory.kt_theory * 100.0
        passed_accuracy = abs(kt_error_pct) <= tolerance_pct

        return NotchFEAResult(
            mesh_level=mesh_level,
            mesh_size_notch=mesh_size_notch,
            num_nodes=len(nodes),
            num_elements=len(part.elements),
            sigma_max_fea=sigma_max_fea,
            sigma_vm_max_fea=sigma_vm_max_fea,
            kt_fea=float(kt_fea),
            kt_error_pct=float(kt_error_pct),
            passed_accuracy=passed_accuracy,
        )


def run_notch_benchmark(
    spec: Optional[NotchSpec] = None,
    coarse_mesh_size: float = 0.8,
    fine_mesh_size: float = 0.4,
    material: Optional[FatigueMaterial] = None,
    ccx_path: str = "/usr/bin/ccx",
    accuracy_tolerance_pct: float = 5.0,
    convergence_tolerance_pct: float = 5.0,
) -> NotchBenchmarkResult:
    """Run dual-mesh notch FEA benchmark and evaluate accuracy vs literature and 2x refinement convergence.

    Criteria for pass:
    1. Kt accuracy: |Kt_FEA - Kt_literature| / Kt_literature <= +/- 5.0%
    2. Mesh sensitivity: |sigma_max(fine) - sigma_max(coarse)| / sigma_max(fine) < 5.0% (for h -> h/2)

    Args:
        spec: Geometric specimen specifications (default: semicircular notched plate).
        coarse_mesh_size: Notch element size for coarse mesh (default: 0.8 mm).
        fine_mesh_size: Notch element size for 2x refined mesh (default: 0.4 mm).
        material: Fatigue material definition.
        ccx_path: Path to CalculiX ccx executable.
        accuracy_tolerance_pct: Acceptance threshold for Kt error (default: 5.0%).
        convergence_tolerance_pct: Acceptance threshold for mesh refinement (default: 5.0%).

    Returns:
        NotchBenchmarkResult instance containing all theoretical and FEA metrics.
    """
    if spec is None:
        spec = NotchSpec()

    if material is None:
        material = FatigueMaterial(name="Ti-6Al-4V")

    # 1. Calculate literature theory
    theory = calculate_notch_theory(spec)

    # 2. Build and solve coarse model (h)
    model_c, mat_c, bc_c = build_notch_model_and_bc(
        spec,
        mesh_size_notch=coarse_mesh_size,
        mesh_size_far=2.0,
        material=material,
    )
    coarse_res = _solve_notch_fea(
        model=model_c,
        mat_assign=mat_c,
        bc=bc_c,
        spec=spec,
        mesh_level="coarse",
        mesh_size_notch=coarse_mesh_size,
        theory=theory,
        ccx_path=ccx_path,
        tolerance_pct=accuracy_tolerance_pct,
    )

    # 3. Build and solve 2x refined fine model (h/2)
    model_f, mat_f, bc_f = build_notch_model_and_bc(
        spec,
        mesh_size_notch=fine_mesh_size,
        mesh_size_far=2.0,
        material=material,
    )
    fine_res = _solve_notch_fea(
        model=model_f,
        mat_assign=mat_f,
        bc=bc_f,
        spec=spec,
        mesh_level="fine",
        mesh_size_notch=fine_mesh_size,
        theory=theory,
        ccx_path=ccx_path,
        tolerance_pct=accuracy_tolerance_pct,
    )

    # 4. Mesh refinement convergence evaluation
    stress_change_pct = (
        abs(fine_res.sigma_max_fea - coarse_res.sigma_max_fea) / max(fine_res.sigma_max_fea, 1e-6) * 100.0
    )
    passed_convergence = bool(stress_change_pct < convergence_tolerance_pct)
    passed_accuracy = bool(fine_res.passed_accuracy)

    # 5. Fatigue notch sensitivity and factor
    q, kf = calculate_notch_sensitivity(
        kt=fine_res.kt_fea,
        notch_radius_r=spec.notch_radius_r,
        tensile_strength_Sut=material.tensile_strength_Sut,
    )

    return NotchBenchmarkResult(
        theory=theory,
        coarse_result=coarse_res,
        fine_result=fine_res,
        stress_change_pct=float(stress_change_pct),
        passed_convergence=passed_convergence,
        passed_accuracy=passed_accuracy,
        accuracy_tolerance_pct=accuracy_tolerance_pct,
        convergence_tolerance_pct=convergence_tolerance_pct,
        notch_sensitivity_q=q,
        fatigue_notch_factor_kf=kf,
    )
