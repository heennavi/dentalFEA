"""
test_calculix_writer.py
Comprehensive unit and integration tests for CalculiX .inp writer.
"""

import os
import shutil
import subprocess
import tempfile
import numpy as np
import pytest

from fea.models import (
    MeshModel,
    MeshPart,
    MaterialDef,
    MaterialAssignment,
    BoundaryConditionSet,
    TieDef,
)
from solver.calculix_writer import write_inp, GlobalIndexMapper
from meshing.gmsh_mesher import fix_orientation


def create_simple_tetrahedron_mesh(order: int = 1):
    """Create a single regular tetrahedron mesh for testing."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [5.0, 8.66, 0.0],
        [5.0, 2.89, 8.16],
    ], dtype=np.float64)

    if order == 1:
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        element_type = "C3D4"
    else:
        # 10-node quadratic tetrahedron
        # Midside nodes 4~9 (0-based)
        mid_nodes = [
            (nodes[0] + nodes[1]) / 2.0,  # 0-1 (4)
            (nodes[1] + nodes[2]) / 2.0,  # 1-2 (5)
            (nodes[2] + nodes[0]) / 2.0,  # 2-0 (6)
            (nodes[0] + nodes[3]) / 2.0,  # 0-3 (7)
            (nodes[1] + nodes[3]) / 2.0,  # 1-3 (8)
            (nodes[2] + nodes[3]) / 2.0,  # 2-3 (9)
        ]
        all_nodes = np.vstack([nodes, np.array(mid_nodes)])
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
        element_type = "C3D10"
        nodes = all_nodes

    elements = fix_orientation(nodes, elements)
    return nodes, elements, element_type


def create_cube_mesh_c3d4():
    """Create a 10x10x10 mm cube mesh composed of 5 linear tetrahedra."""
    # 8 corner vertices of [0, 10] x [0, 10] x [0, 10]
    nodes = np.array([
        [0.0, 0.0, 0.0],    # 0
        [10.0, 0.0, 0.0],   # 1
        [10.0, 10.0, 0.0],  # 2
        [0.0, 10.0, 0.0],   # 3
        [0.0, 0.0, 10.0],   # 4
        [10.0, 0.0, 10.0],  # 5
        [10.0, 10.0, 10.0], # 6
        [0.0, 10.0, 10.0],  # 7
    ], dtype=np.float64)

    # Standard 5-tetrahedron subdivision
    tets = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ], dtype=np.int64)

    tets = fix_orientation(nodes, tets)
    return nodes, tets


class TestGlobalIndexMapper:
    def test_global_index_offsets(self):
        n1, e1, t1 = create_simple_tetrahedron_mesh(order=1)
        n2, e2, t2 = create_simple_tetrahedron_mesh(order=2)

        part1 = MeshPart(name="bone", nodes=n1, elements=e1, element_type=t1)
        part2 = MeshPart(name="implant", nodes=n2, elements=e2, element_type=t2)
        model = MeshModel(parts=[part1, part2])

        mapper = GlobalIndexMapper(model)

        assert mapper.total_nodes == len(n1) + len(n2)
        assert mapper.total_elements == len(e1) + len(e2)

        # Part 1: local 0 -> global 1
        assert mapper.node_id("bone", 0) == 1
        assert mapper.node_id("bone", 3) == 4
        assert mapper.elem_id("bone", 0) == 1

        # Part 2: local 0 -> global len(n1) + 1
        assert mapper.node_id("implant", 0) == len(n1) + 1
        assert mapper.elem_id("implant", 0) == len(e1) + 1

        # Array conversion
        g_nids = mapper.node_ids("implant", [0, 1])
        assert list(g_nids) == [len(n1) + 1, len(n1) + 2]


class TestCalculixWriterParsing:
    def test_small_assembly_inp_parsing(self):
        """Verify .inp file structure and keywords for a multi-part assembly."""
        n1, e1, t1 = create_simple_tetrahedron_mesh(order=1)
        n2, e2, t2 = create_simple_tetrahedron_mesh(order=2)

        part1 = MeshPart(name="bone", nodes=n1, elements=e1, element_type=t1)
        part2 = MeshPart(name="implant", nodes=n2, elements=e2, element_type=t2)
        model = MeshModel(parts=[part1, part2])

        mat_bone = MaterialDef(name="BONE_MAT", E=12000.0, nu=0.3, density=1.8e-9)
        mat_ti = MaterialDef(name="TITANIUM", E=110000.0, nu=0.34)

        materials = {
            "bone": MaterialAssignment(
                element_material={"bone": np.array([0], dtype=np.int32)},
                materials=[mat_bone]
            ),
            "implant": MaterialAssignment(
                element_material={"implant": np.array([0], dtype=np.int32)},
                materials=[mat_ti]
            ),
        }

        bc = BoundaryConditionSet(
            fixed_node_sets=[{
                "name": "FIX_BASE",
                "part": "bone",
                "node_ids": np.array([0, 1, 2], dtype=np.int64),
                "dofs": (1, 3)
            }],
            loads=[{
                "name": "LOAD_OCCLUSAL",
                "part": "implant",
                "node_ids": np.array([3], dtype=np.int64),
                "force_N": (10.0, 0.0, -100.0)
            }]
        )

        ties = [
            TieDef(
                name="TIE_BONE_IMP",
                master_part="bone",
                slave_part="implant",
                master_faces=np.array([[0, 1]], dtype=np.int64),
                slave_nodes=np.array([0, 1], dtype=np.int64),
                position_tolerance_mm=0.1
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            inp_path = os.path.join(tmpdir, "test_job.inp")
            write_inp(
                path=inp_path,
                model=model,
                materials=materials,
                bc=bc,
                ties=ties,
                title="Test Dental FEA Model"
            )

            assert os.path.exists(inp_path)
            with open(inp_path, "r", encoding="ascii") as f:
                content = f.read()

            # 1. Heading check
            assert "*HEADING" in content
            assert "Test Dental FEA Model" in content

            # 2. Node count check
            total_expected_nodes = len(n1) + len(n2)
            lines = content.splitlines()
            node_idx = lines.index("*NODE")
            # Collect lines until next keyword
            node_lines = []
            for line in lines[node_idx + 1:]:
                if line.startswith("*"):
                    break
                if line.strip():
                    node_lines.append(line)
            assert len(node_lines) == total_expected_nodes

            # 3. Element keywords check
            assert "*ELEMENT, TYPE=C3D4, ELSET=EALL_BONE" in content
            assert "*ELEMENT, TYPE=C3D10, ELSET=EALL_IMPLANT" in content

            # 4. Materials check
            assert "*MATERIAL, NAME=BONE_MAT" in content
            assert "*MATERIAL, NAME=TITANIUM" in content
            assert "*ELASTIC" in content
            assert "12000.000000, 0.300000" in content
            assert "*DENSITY" in content
            assert "1.800000000e-09" in content
            assert "*SOLID SECTION, ELSET=ELSET_BONE_MAT, MATERIAL=BONE_MAT" in content

            # 5. Boundary & Load sets check
            assert "*NSET, NSET=FIX_BASE" in content
            assert "*NSET, NSET=LOAD_OCCLUSAL" in content

            # 6. Tie check
            assert "*SURFACE, NAME=SURF_M_TIE_BONE_IMP, TYPE=ELEMENT" in content
            assert "*SURFACE, NAME=SURF_S_TIE_BONE_IMP, TYPE=NODE" in content
            assert "*TIE, NAME=TIE_BONE_IMP, POSITION TOLERANCE=0.100000" in content
            assert "SURF_S_TIE_BONE_IMP, SURF_M_TIE_BONE_IMP" in content

            # 7. Step & Solvers check
            assert "*STEP" in content
            assert "*STATIC" in content
            assert "*BOUNDARY" in content
            assert "FIX_BASE, 1, 3" in content
            assert "*CLOAD" in content
            assert "*NODE FILE" in content
            assert "*EL FILE" in content
            assert "*END STEP" in content

    def test_c3d10_node_order_and_continuation_lines(self):
        """Verify C3D10 node exchange and continuation line rule (trailing comma)."""
        nodes = np.zeros((10, 3), dtype=np.float64)
        for i in range(10):
            nodes[i] = [float(i), 0.0, 0.0]

        # Input connectivity in Gmsh 0-based order: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
        part = MeshPart(name="quad_part", nodes=nodes, elements=elements, element_type="C3D10")
        model = MeshModel(parts=[part])

        mat = MaterialDef(name="STEEL", E=210000.0, nu=0.3)
        materials = {
            "quad_part": MaterialAssignment(
                element_material={"quad_part": np.array([0], dtype=np.int32)},
                materials=[mat]
            )
        }
        bc = BoundaryConditionSet(fixed_node_sets=[], loads=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            inp_path = os.path.join(tmpdir, "c3d10_test.inp")
            write_inp(
                path=inp_path,
                model=model,
                materials=materials,
                bc=bc,
                ties=[],
                apply_ccx_ordering=True
            )

            with open(inp_path, "r", encoding="ascii") as f:
                lines = [line.strip() for line in f if line.strip()]

            elem_idx = lines.index("*ELEMENT, TYPE=C3D10, ELSET=EALL_QUAD_PART")
            line1 = lines[elem_idx + 1]
            line2 = lines[elem_idx + 2]

            # Line 1 must end with a trailing comma
            assert line1.endswith(","), f"Line 1 should end with comma: {line1}"

            # Nodes: 0-based [0..7] -> 1-based [1..8] on line 1
            expected_line1 = "1, 1, 2, 3, 4, 5, 6, 7, 8,"
            assert line1 == expected_line1

            # Gmsh [8, 9] should be converted to CalculiX [9, 8] -> 1-based [10, 9]
            expected_line2 = "10, 9"
            assert line2 == expected_line2

    def test_cload_resultant_force_conservation(self):
        """Verify that distributed CLOAD forces exactly sum up to resultant force."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        part = MeshPart(name="load_part", nodes=nodes, elements=elements, element_type="C3D4")
        model = MeshModel(parts=[part])

        target_nodes = np.array([0, 1, 2, 4], dtype=np.int64)  # 4 nodes
        input_force = (25.0, -50.0, 120.0)

        bc = BoundaryConditionSet(
            fixed_node_sets=[],
            loads=[{
                "name": "OCCLUSAL_DISTRIBUTED",
                "part": "load_part",
                "node_ids": target_nodes,
                "force_N": input_force
            }]
        )

        mat = MaterialDef(name="MAT", E=1000.0, nu=0.3)
        materials = {
            "load_part": MaterialAssignment(
                element_material={"load_part": np.array([0], dtype=np.int32)},
                materials=[mat]
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            inp_path = os.path.join(tmpdir, "load_test.inp")
            write_inp(
                path=inp_path,
                model=model,
                materials=materials,
                bc=bc,
                ties=[]
            )

            with open(inp_path, "r", encoding="ascii") as f:
                lines = f.readlines()

            cload_idx = -1
            for i, line in enumerate(lines):
                if line.strip() == "*CLOAD":
                    cload_idx = i
                    break
            assert cload_idx != -1

            sum_fx = 0.0
            sum_fy = 0.0
            sum_fz = 0.0

            for line in lines[cload_idx + 1:]:
                line = line.strip()
                if line.startswith("*"):
                    break
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                nid, dof, f_val = int(parts[0]), int(parts[1]), float(parts[2])
                if dof == 1:
                    sum_fx += f_val
                elif dof == 2:
                    sum_fy += f_val
                elif dof == 3:
                    sum_fz += f_val

            assert np.isclose(sum_fx, input_force[0], atol=1e-6)
            assert np.isclose(sum_fy, input_force[1], atol=1e-6)
            assert np.isclose(sum_fz, input_force[2], atol=1e-6)


class TestCalculixExecutionIntegration:
    def test_cube_compression_simulation_with_ccx(self):
        """
        Integration test: Solve a 10x10x10 mm cube under uniaxial compression using CalculiX ccx.
        Verifies that .frd result file is generated upon successful exit.
        """
        ccx_path = shutil.which("ccx")
        if not ccx_path:
            pytest.skip("CalculiX 'ccx' binary is not found on PATH; skipping execution test.")

        nodes, tets = create_cube_mesh_c3d4()
        part = MeshPart(name="cube", nodes=nodes, elements=tets, element_type="C3D4")
        model = MeshModel(parts=[part])

        # Bottom face nodes (z == 0)
        bot_mask = np.isclose(nodes[:, 2], 0.0)
        bot_nodes = np.where(bot_mask)[0].astype(np.int64)

        # Top face nodes (z == 10)
        top_mask = np.isclose(nodes[:, 2], 10.0)
        top_nodes = np.where(top_mask)[0].astype(np.int64)

        mat = MaterialDef(name="CUBE_MAT", E=1000.0, nu=0.3)
        materials = {
            "cube": MaterialAssignment(
                element_material={"cube": np.zeros(len(tets), dtype=np.int32)},
                materials=[mat]
            )
        }

        # Fix bottom in Z (and constrain rigid body motion in X, Y)
        # For simplicity, fix all DOFs of bottom face
        bc = BoundaryConditionSet(
            fixed_node_sets=[{
                "name": "FIX_BOTTOM",
                "part": "cube",
                "node_ids": bot_nodes,
                "dofs": (1, 3)
            }],
            loads=[{
                "name": "COMPRESS_TOP",
                "part": "cube",
                "node_ids": top_nodes,
                "force_N": (0.0, 0.0, -100.0)
            }]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            job_name = "cube_compression"
            inp_path = os.path.join(tmpdir, f"{job_name}.inp")
            write_inp(
                path=inp_path,
                model=model,
                materials=materials,
                bc=bc,
                ties=[],
                title="Cube Uniaxial Compression Test"
            )

            # Run ccx
            cmd = [ccx_path, job_name]
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30
            )

            assert proc.returncode == 0, f"ccx failed with returncode {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"

            frd_path = os.path.join(tmpdir, f"{job_name}.frd")
            assert os.path.exists(frd_path), f".frd output file was not generated: {frd_path}"
            assert os.path.getsize(frd_path) > 0, ".frd file is empty"
