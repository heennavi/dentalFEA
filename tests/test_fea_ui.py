"""
test_fea_ui.py
Unit and integration tests for Phase 3 FEA UI components:
- FeaPanel button enable/disable state transitions and stale status warnings
- FeaController headless pipeline execution (material assignment, boundary conditions, .inp export)
"""

import os
import tempfile
import numpy as np
import pytest
import trimesh
import vtk
from vtk.util import numpy_support
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.geometry.boolean_ops import FeaGeometry
from dental_fea_app.ui.fea_panel import FeaPanel
from dental_fea_app.ui.fea_controller import FeaController
from fea.models import (
    MeshPart,
    MeshModel,
    MaterialDef,
    MaterialAssignment,
    BoundaryConditionSet,
    AnalysisConfig,
)
from fea.workers import MeshWorker
from tests.test_calculix_writer import create_cube_mesh_c3d4


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication instance exists for UI widget testing."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def create_dummy_vtk_volume(shape=(10, 10, 10), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Create a minimal vtkImageData with HU values."""
    vtk_img = vtk.vtkImageData()
    nz, ny, nx = shape
    vtk_img.SetDimensions(nx, ny, nz)
    vtk_img.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_img.SetOrigin(origin[0], origin[1], origin[2])

    data = np.full(shape, 500, dtype=np.int16)  # 500 HU (cancellous bone range)
    # Cortical layer on edges
    data[0, :, :] = 1200
    data[-1, :, :] = 1200

    vtk_arr = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    vtk_arr.SetName("HU_Scalars")
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img


class TestFeaPanelState:
    """Test FeaPanel UI state transitions based on StateManager events."""

    def test_initial_buttons_disabled_without_data(self, qapp):
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        # 초기 상태: 기하 없음
        assert not panel.btn_mesh.isEnabled(), "기하 데이터가 없을 때 [메쉬 생성]은 비활성화되어야 함"
        assert not panel.btn_assign_material.isEnabled(), "메쉬가 없을 때 [물성 할당]은 비활성화되어야 함"
        assert not panel.btn_validate_bc.isEnabled(), "메쉬가 없을 때 [경계조건 검증]은 비활성화되어야 함"
        assert not panel.btn_export_inp.isEnabled(), "필수 데이터 부재 시 [.inp 저장]은 비활성화되어야 함"
        assert not panel.btn_run_solver.isEnabled(), "필수 데이터 부재 시 [해석 실행]은 비활성화되어야 함"
        assert not panel.banner_stale.isVisible(), "초기 상태에서 stale 경고 배너는 숨겨져 있어야 함"

    def test_mesh_button_enabled_when_geometry_present(self, qapp):
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        box_cortical = trimesh.creation.box(extents=[10, 10, 10])
        box_cancellous = trimesh.creation.box(extents=[8, 8, 8])
        box_implant = trimesh.creation.cylinder(radius=2, height=10)

        geom = FeaGeometry(
            cortical=box_cortical,
            cancellous=box_cancellous,
            implant=box_implant,
            report={"volume": 1000},
        )

        state_mgr.set_fea_geometry(geom)

        assert panel.btn_mesh.isEnabled(), "FeaGeometry가 존재하면 [메쉬 생성] 버튼이 활성화되어야 함"
        assert not panel.btn_assign_material.isEnabled(), "메쉬가 없으면 [물성 할당]은 여전히 비활성화여야 함"

    def test_material_and_bc_buttons_enabled_when_mesh_present(self, qapp):
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        nodes, elements = create_cube_mesh_c3d4()
        elem_type = "C3D4"
        bone_part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type=elem_type)
        mesh_model = MeshModel(parts=[bone_part])

        state_mgr.set_mesh_model(mesh_model)

        assert panel.btn_assign_material.isEnabled(), "MeshModel이 존재하면 [물성 할당] 버튼이 활성화되어야 함"
        assert panel.btn_validate_bc.isEnabled(), "MeshModel이 존재하면 [경계조건] 버튼이 활성화되어야 함"
        assert not panel.btn_run_solver.isEnabled(), "물성과 BC가 없으면 [해석 실행]은 비활성화여야 함"

    def test_solver_buttons_enabled_when_all_prerequisites_met(self, qapp):
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        nodes, elements = create_cube_mesh_c3d4()
        elem_type = "C3D4"
        bone_part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type=elem_type)
        mesh_model = MeshModel(parts=[bone_part])

        mat = MaterialDef(name="BONE_01", E=1000.0, nu=0.3)
        mat_assign = MaterialAssignment(
            element_material={"bone": np.zeros(len(elements), dtype=np.int32)},
            materials=[mat],
        )

        bc = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX", "part": "bone", "node_ids": np.array([0, 1, 2]), "dofs": (1, 3)}],
            loads=[{"name": "LOAD", "part": "bone", "node_ids": np.array([3, 4]), "force_N": (0.0, 0.0, -100.0)}],
        )

        state_mgr.set_mesh_model(mesh_model)
        state_mgr.set_material_assignment(mat_assign)
        state_mgr.set_bc_set(bc)

        assert panel.btn_export_inp.isEnabled(), "메쉬, 물성, BC가 모두 존재하면 [.inp 저장] 버튼이 활성화되어야 함"
        assert panel.btn_run_solver.isEnabled(), "메쉬, 물성, BC가 모두 존재하면 [해석 실행] 버튼이 활성화되어야 함"

    def test_stale_warning_banner_triggers_on_upstream_change(self, qapp):
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        nodes, elements = create_cube_mesh_c3d4()
        elem_type = "C3D4"
        bone_part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type=elem_type)
        state_mgr.set_mesh_model(MeshModel(parts=[bone_part]))

        # 상류 데이터 변경 (기하 재생성 또는 임플란트 변경 신호 발생)
        box_c = trimesh.creation.box(extents=[10, 10, 10])
        new_geom = FeaGeometry(cortical=box_c, cancellous=box_c, implant=box_c, report={})
        state_mgr.set_fea_geometry(new_geom)

        panel.show()
        assert state_mgr.fea_is_stale, "상류 데이터 변경 시 StateManager.fea_is_stale이 True여야 함"
        assert not panel.banner_stale.isHidden(), "상류 데이터 변경 시 배너가 표시(visible) 상태여야 함"



class TestFeaControllerHeadless:
    """Headless unit test for FeaController pipeline from material mapping to .inp generation."""

    def test_material_assignment_and_summary(self, qapp):
        state_mgr = StateManager()
        controller = FeaController(state_mgr)

        # 1. 메쉬 설정
        nodes, elements = create_cube_mesh_c3d4()
        elem_type = "C3D4"
        bone_part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type=elem_type)
        mesh_model = MeshModel(parts=[bone_part])
        state_mgr.set_mesh_model(mesh_model)

        # 2. 볼륨 설정
        vtk_vol = create_dummy_vtk_volume(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        state_mgr.set_volume_data(vtk_vol, metadata={"series_path": "dummy"})

        # 3. 물성 할당 실행
        summary_results = []
        controller.material_assigned.connect(lambda s: summary_results.append(s))

        assignment = controller.assign_materials(n_bins=10)

        assert assignment is not None
        assert "bone" in assignment.element_material
        assert len(assignment.materials) > 0
        assert len(summary_results) == 1
        summary = summary_results[0]
        assert "e_min" in summary and "e_max" in summary and "e_mean" in summary
        assert summary["e_min"] <= summary["e_mean"] <= summary["e_max"]
        assert state_mgr.material_assignment is assignment

    def test_boundary_condition_setup_and_validation(self, qapp):
        state_mgr = StateManager()
        controller = FeaController(state_mgr)

        nodes, elements = create_cube_mesh_c3d4()
        elem_type = "C3D4"
        bone_part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type=elem_type)
        mesh_model = MeshModel(parts=[bone_part])
        state_mgr.set_mesh_model(mesh_model)

        issues = controller.setup_boundary_conditions(
            bottom_z=2.0,
            load_preset_key="axial_100N",
            load_mag=100.0,
            load_angle=0.0,
            top_margin_mm=3.0,
        )

        assert state_mgr.bc_set is not None
        assert len(state_mgr.bc_set.fixed_node_sets) == 1
        assert len(state_mgr.bc_set.loads) == 1
        # 큐브에서 z <= 2.0인 절점들이 고정됨
        assert len(state_mgr.bc_set.fixed_node_sets[0]["node_ids"]) > 0
        assert len(state_mgr.bc_set.loads[0]["node_ids"]) > 0

    def test_export_inp_file_generation(self, qapp):
        state_mgr = StateManager()
        controller = FeaController(state_mgr)

        # 메쉬, 볼륨, 물성, BC 구성
        nodes, elements = create_cube_mesh_c3d4()
        elem_type = "C3D4"
        bone_part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type=elem_type)
        mesh_model = MeshModel(parts=[bone_part])
        state_mgr.set_mesh_model(mesh_model)

        vtk_vol = create_dummy_vtk_volume(shape=(16, 16, 16))
        state_mgr.set_volume_data(vtk_vol)

        controller.assign_materials(n_bins=5)
        controller.setup_boundary_conditions(bottom_z=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp_path = os.path.join(tmpdir, "test_job.inp")
            exported = controller.export_inp(inp_path)

            assert exported == inp_path
            assert os.path.exists(inp_path)

            with open(inp_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "*HEADING" in content
            assert "*NODE" in content
            assert "*ELEMENT" in content
            assert "*MATERIAL" in content
            assert "*BOUNDARY" in content
            assert "*CLOAD" in content
            assert "*STEP" in content
            assert "*STATIC" in content
            assert "*END STEP" in content

    def test_mesh_worker_cancellation(self, qapp):
        box_c = trimesh.creation.box(extents=[10, 10, 10])
        geom = FeaGeometry(cortical=box_c, cancellous=box_c, implant=box_c, report={})
        cfg = AnalysisConfig(mesh_size_far_mm=5.0, mesh_size_near_mm=2.0, element_type="C3D4", ccx_path=None, n_threads=1)

        worker = MeshWorker(fea_geometry=geom, config=cfg)
        worker.cancel()
        assert worker._is_cancelled
