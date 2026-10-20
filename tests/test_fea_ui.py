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
        assert "CalculiX 실행 가능" in panel.lbl_solver_status.text()

    def test_smart_activation_when_mesh_and_bc_ready_without_material(self, qapp):
        """메쉬와 경계조건이 설정되면 물성 미할당 상태여도 해석 실행 버튼이 활성화되어야 함 (스마트 자동 실행)."""
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        nodes, elements = create_cube_mesh_c3d4()
        bone_part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type="C3D4")
        mesh_model = MeshModel(parts=[bone_part])

        bc = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX", "part": "bone", "node_ids": np.array([0, 1]), "dofs": (1, 3)}],
            loads=[{"name": "LOAD", "part": "bone", "node_ids": np.array([2, 3]), "force_N": (0.0, 0.0, -100.0)}],
        )

        state_mgr.set_mesh_model(mesh_model)
        state_mgr.set_bc_set(bc)

        # 물성은 아직 할당되지 않은 상태
        assert state_mgr.material_assignment is None
        assert panel.btn_run_solver.isEnabled(), "메쉬와 BC가 있으면 물성 미할당이어도 스마트 활성화되어야 함"
        assert panel.btn_export_inp.isEnabled(), "메쉬와 BC가 있으면 .inp 저장도 스마트 활성화되어야 함"
        assert "자동 할당" in panel.lbl_solver_status.text()

    def test_auto_material_assignment_on_run_solver_clicked(self, qapp, monkeypatch):
        """물성이 없는 상태에서 _on_run_solver_clicked 호출 시 자동 물성 할당이 수행됨."""
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        # 더미 볼륨 및 메쉬 설정
        vtk_img = create_dummy_vtk_volume(shape=(10, 10, 10))
        state_mgr.set_volume_data(vtk_img, metadata={"series_path": "dummy"})

        nodes, elements = create_cube_mesh_c3d4()
        bone_part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type="C3D4")
        mesh_model = MeshModel(parts=[bone_part])
        bc = BoundaryConditionSet(
            fixed_node_sets=[{"name": "FIX", "part": "bone", "node_ids": np.array([0, 1]), "dofs": (1, 3)}],
            loads=[{"name": "LOAD", "part": "bone", "node_ids": np.array([2, 3]), "force_N": (0.0, 0.0, -100.0)}],
        )

        state_mgr.set_mesh_model(mesh_model)
        state_mgr.set_bc_set(bc)

        # start_solver는 실제 ccx를 띄우지 않도록 모킹
        started = []
        monkeypatch.setattr(panel.controller, "start_solver", lambda *args, **kwargs: started.append(True))

        assert state_mgr.material_assignment is None
        panel._on_run_solver_clicked()

        # 물성 할당이 자동 완료되었고 start_solver가 호출되었는지 확인
        assert state_mgr.material_assignment is not None, "클릭 시 물성 자동 할당이 수행되어야 함"
        assert len(started) == 1, "물성 할당 후 start_solver가 호출되어야 함"

    def test_mesh_algorithm_and_optimization_ui_options(self, qapp, monkeypatch):
        """메쉬 생성 UI에서 3D 알고리즘 및 Netgen 최적화 옵션이 AnalysisConfig로 올바르게 전달됨."""
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        box_c = trimesh.creation.box(extents=[10, 10, 10])
        geom = FeaGeometry(cortical=box_c, cancellous=box_c, implant=box_c, report={})
        state_mgr.set_fea_geometry(geom)

        # UI에서 Netgen 알고리즘 및 옵션 선택
        idx = panel.combo_mesh_algo.findData("netgen")
        assert idx != -1
        panel.combo_mesh_algo.setCurrentIndex(idx)
        panel.chk_optimize_netgen.setChecked(True)

        captured_config = []
        monkeypatch.setattr(panel.controller, "start_meshing", lambda cfg: captured_config.append(cfg))

        panel._on_mesh_clicked()

        assert len(captured_config) == 1
        cfg = captured_config[0]
        assert cfg.mesh_algorithm_3d == "netgen"
        assert cfg.optimize_netgen is True



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

    def test_fea_panel_widget_layout_and_sizes(self, qapp):
        """사용자 요청에 따른 FEA 패널 위젯 크기 및 위치 조정 검증"""
        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)
        panel.resize(260, 800)
        panel.show()
        qapp.processEvents()

        # 1. 메쉬 생성: spin_size_far, spin_size_near 폭 60px (1/2로 축소)
        assert panel.spin_size_far.width() == 60
        assert panel.spin_size_near.width() == 60
        # combo_elem_type은 라벨 고정(52px)으로 좌측(x <= 70)으로 이동
        assert panel.combo_elem_type.x() <= 70

        # 2. 물성 할당: btn_choose_preset이 lbl_preset_path보다 좌측에 위치하고, 폭은 텍스트 맞춤
        assert panel.btn_choose_preset.x() < panel.lbl_preset_path.x()
        assert panel.btn_choose_preset.width() <= 55
        # spin_bins 폭 60px (1/2로 축소) 및 좌측 x=135 정렬
        assert panel.spin_bins.width() == 60
        assert panel.spin_bins.x() == panel.spin_size_far.x()

        # 3. 경계조건: spin_fixed_z 폭 80px (2/3 수준)
        assert panel.spin_fixed_z.width() == 80

    def test_control_panel_resizable_and_scroll(self, qapp):
        """ControlPanel이 고정폭이 아닌 유연한 크기 조절 및 가로 스크롤을 지원하는지 검증"""
        from dental_fea_app.ui.control_panel import ControlPanel
        state_mgr = StateManager()
        ctrl = ControlPanel(state_mgr, on_open_3d_viewer=lambda: None)

        assert ctrl.minimumWidth() <= 240
        assert ctrl.maximumWidth() >= 600

    def test_gmsh_thread_initialization(self):
        """서브 스레드에서 gmsh.initialize(interruptible=False)가 signal 오류 없이 성공하는지 검증"""
        import threading
        import gmsh

        thread_error = []

        def worker_func():
            try:
                if not gmsh.isInitialized():
                    gmsh.initialize(interruptible=False)
                assert gmsh.isInitialized() == 1
                gmsh.finalize()
            except Exception as e:
                thread_error.append(e)

        th = threading.Thread(target=worker_func)
        th.start()
        th.join(timeout=5.0)

        assert len(thread_error) == 0, f"Gmsh thread initialization failed: {thread_error}"

