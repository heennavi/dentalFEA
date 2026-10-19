"""
test_models.py
Unit tests for fea.models data contracts and helper methods.
"""

import json
import pytest
import numpy as np

from fea.models import (
    MeshPart,
    MeshModel,
    MaterialDef,
    MaterialAssignment,
    BoundaryConditionSet,
    TieDef,
    AnalysisConfig,
)


class TestFeaModels:
    """Test suite for Phase 3 FEA data structures."""

    def test_mesh_part_creation(self):
        """Verify MeshPart dataclass fields and types."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

        part = MeshPart(
            name="bone",
            nodes=nodes,
            elements=elements,
            element_type="C3D4",
            surface_tags={"top": np.array([0])}
        )

        assert part.name == "bone"
        assert part.nodes.shape == (4, 3)
        assert part.elements.shape == (1, 4)
        assert part.element_type == "C3D4"
        assert "top" in part.surface_tags
        assert len(part.surface_tags["top"]) == 1

    def test_mesh_model_offsets_and_totals(self):
        """Verify MeshModel global_node_offset, total_nodes, and total_elements."""
        # Part 1: Bone (10 nodes, 2 elements)
        bone_nodes = np.zeros((10, 3), dtype=np.float64)
        bone_elems = np.zeros((2, 4), dtype=np.int64)
        bone = MeshPart(name="bone", nodes=bone_nodes, elements=bone_elems, element_type="C3D4")

        # Part 2: Implant (6 nodes, 1 element)
        imp_nodes = np.ones((6, 3), dtype=np.float64)
        imp_elems = np.zeros((1, 4), dtype=np.int64)
        implant = MeshPart(name="implant", nodes=imp_nodes, elements=imp_elems, element_type="C3D4")

        # Part 3: Abutment (4 nodes, 1 element)
        abut_nodes = np.ones((4, 3), dtype=np.float64)
        abut_elems = np.zeros((1, 4), dtype=np.int64)
        abutment = MeshPart(name="abutment", nodes=abut_nodes, elements=abut_elems, element_type="C3D4")

        model = MeshModel(parts=[bone, implant, abutment])

        # Global node offsets
        assert model.global_node_offset("bone") == 0
        assert model.global_node_offset("implant") == 10
        assert model.global_node_offset("abutment") == 16

        # Missing part raises KeyError
        with pytest.raises(KeyError):
            model.global_node_offset("nonexistent_part")

        # Totals
        assert model.total_nodes() == 20
        assert model.total_elements() == 4

        # Part retrieval
        assert model.get_part("implant") is implant
        assert model.get_part("unknown") is None

    def test_analysis_config_serialization(self):
        """Verify AnalysisConfig to_dict and from_dict JSON round-trip."""
        config = AnalysisConfig(
            mesh_size_far_mm=2.5,
            mesh_size_near_mm=0.6,
            element_type="C3D10",
            ccx_path="/usr/bin/ccx",
            n_threads=8,
        )

        d = config.to_dict()
        assert d["mesh_size_far_mm"] == 2.5
        assert d["mesh_size_near_mm"] == 0.6
        assert d["element_type"] == "C3D10"
        assert d["ccx_path"] == "/usr/bin/ccx"
        assert d["n_threads"] == 8

        # JSON serialize & deserialize
        json_str = json.dumps(d)
        parsed_dict = json.loads(json_str)

        restored = AnalysisConfig.from_dict(parsed_dict)
        assert restored.mesh_size_far_mm == 2.5
        assert restored.mesh_size_near_mm == 0.6
        assert restored.element_type == "C3D10"
        assert restored.ccx_path == "/usr/bin/ccx"
        assert restored.n_threads == 8

    def test_material_contracts(self):
        """Verify MaterialDef and MaterialAssignment contracts."""
        m_cortical = MaterialDef(name="cortical", E=13700.0, nu=0.3, density=1.9e-9)
        m_cancellous = MaterialDef(name="cancellous", E=1370.0, nu=0.3)
        m_ti = MaterialDef(name="titanium", E=110000.0, nu=0.35, density=4.5e-9)

        materials = [m_cortical, m_cancellous, m_ti]

        bone_elem_mat = np.array([0, 1, 0, 1], dtype=np.int32)
        imp_elem_mat = np.array([2, 2], dtype=np.int32)

        mat_assign = MaterialAssignment(
            element_material={"bone": bone_elem_mat, "implant": imp_elem_mat},
            materials=materials
        )

        assert len(mat_assign.materials) == 3
        assert mat_assign.element_material["bone"].shape == (4,)
        assert mat_assign.element_material["implant"].shape == (2,)

    def test_boundary_condition_and_tie(self):
        """Verify BoundaryConditionSet and TieDef contracts."""
        fixed_bc = {
            "name": "base_fix",
            "part": "bone",
            "node_ids": np.array([0, 1, 2], dtype=np.int64),
            "dofs": (1, 3)
        }
        load = {
            "name": "occlusal_force",
            "part": "implant",
            "node_ids": np.array([10, 11], dtype=np.int64),
            "force_N": (0.0, -50.0, -100.0)
        }

        bc_set = BoundaryConditionSet(fixed_node_sets=[fixed_bc], loads=[load])
        assert len(bc_set.fixed_node_sets) == 1
        assert len(bc_set.loads) == 1
        assert bc_set.loads[0]["force_N"] == (0.0, -50.0, -100.0)

        # Tie definition
        master_faces = np.array([[100, 2], [101, 3]], dtype=np.int64)
        slave_nodes = np.array([0, 1, 2, 3], dtype=np.int64)
        tie = TieDef(
            name="bone_implant_tie",
            master_part="bone",
            slave_part="implant",
            master_faces=master_faces,
            slave_nodes=slave_nodes,
            position_tolerance_mm=0.1
        )
        assert tie.name == "bone_implant_tie"
        assert tie.master_faces.shape == (2, 2)
        assert tie.slave_nodes.shape == (4,)
        assert tie.position_tolerance_mm == 0.1

    def test_headless_import(self):
        """Verify that fea module can be imported without importing PySide6 or VTK."""
        import sys
        # Check that fea.models itself does not import PySide6 or vtk
        import fea.models
        # Ensure fea.models doesn't have PySide6 or vtk as globals
        assert not hasattr(fea.models, "PySide6")
        assert not hasattr(fea.models, "vtk")
