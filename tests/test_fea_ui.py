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

        # 3. 경계조건: combo_fix_mode와 btn_sync_fix 존재 및 텍스트 확인
        assert panel.combo_fix_mode.count() == 2
        assert panel.combo_fix_mode.itemText(0) == "Z-축"
        assert panel.combo_fix_mode.itemText(1) == "축 지정"
        assert panel.btn_sync_fix.text() == "현재 고정위치 복사"

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

    def test_copy_material_button(self, qapp, monkeypatch):
        """현재 물성 복사 버튼 배치 및 클립보드 복사 동작 검증"""
        from unittest.mock import MagicMock
        from PySide6.QtWidgets import QMessageBox

        mock_info = MagicMock()
        monkeypatch.setattr(QMessageBox, "information", mock_info)

        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        # 1. 버튼 존재 및 크기 검증
        assert hasattr(panel, "btn_copy_mat")
        assert panel.btn_copy_mat.text() == "현재 물성 복사"

        # 2. 물성 미할당 상태에서 복사 클릭
        panel.btn_copy_mat.click()
        assert mock_info.called
        clipboard_text = QApplication.clipboard().text()
        assert "E 구간(Bin) 개수" in clipboard_text or "presets.json" in clipboard_text

        # 3. 물성 할당 상태에서 복사 클릭
        mat = MaterialDef(name="CORTICAL", E=13700.0, nu=0.3)
        mat_assign = MaterialAssignment(
            element_material={"bone": np.array([0, 0])},
            materials=[mat],
        )
        state_mgr.set_material_assignment(mat_assign)

        mock_info.reset_mock()
        panel.btn_copy_mat.click()
        assert mock_info.called
        clipboard_text = QApplication.clipboard().text()
        assert "CORTICAL" in clipboard_text
        assert "13,700" in clipboard_text or "13700" in clipboard_text

    def test_boundary_conditions_and_load_gizmo_ui(self, qapp, monkeypatch):
        """고정면 드롭다운, 복사 버튼, 하중 부여 및 스냅 체크박스 UI 동작 검증"""
        from unittest.mock import MagicMock
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(QMessageBox, "information", MagicMock())
        monkeypatch.setattr(QMessageBox, "warning", MagicMock())

        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        # 1. 고정 모드 전환 시그널 테스트
        mode_signals = []
        state_mgr.bc_fix_drawing_mode_toggled.connect(lambda b: mode_signals.append(b))

        # '축 지정' 선택
        idx_box = panel.combo_fix_mode.findData("box")
        panel.combo_fix_mode.setCurrentIndex(idx_box)
        assert len(mode_signals) > 0 and mode_signals[-1] is True
        assert "축 지정 모드" in panel.lbl_fix_info.text()

        # 'Z-축' 선택
        idx_z = panel.combo_fix_mode.findData("z_plane")
        panel.combo_fix_mode.setCurrentIndex(idx_z)
        assert len(mode_signals) > 1 and mode_signals[-1] is False

        # 2. 고정위치 복사 (Z-축 모드)
        state_mgr.set_coordinates(10.0, 20.0, -45.5)
        panel.btn_sync_fix.click()
        assert panel._fixed_z == -45.5
        assert "-45.5" in panel.lbl_fix_info.text()

        # 3. 고정위치 복사 (축 지정 모드)
        panel.combo_fix_mode.setCurrentIndex(idx_box)
        state_mgr.bc_fix_box = (10.0, 30.0, 5.0, 25.0, -50.0, 10.0)
        panel.btn_sync_fix.click()
        assert "지정 축 영역 복사됨" in panel.lbl_fix_info.text()
        assert "X:10~30" in panel.lbl_fix_info.text()

        # 4. 하중 부여 및 스냅 버튼
        assert hasattr(panel, "btn_apply_load")
        assert panel.chk_snap_implant.isChecked() is True

        # 프리셋 변경 연동 테스트
        idx_oblique = panel.combo_load_preset.findData("oblique_30deg_100N")
        assert idx_oblique >= 0
        panel.combo_load_preset.setCurrentIndex(idx_oblique)
        assert panel.spin_load_mag.value() == 100.0
        assert panel.spin_load_angle.value() == 30.0

        # 5. 모의 뷰어를 통한 교합 하중 부여 버튼 및 update_parameters 연동 검증
        import vtk
        from dental_fea_app.ui.geometry.occlusal_load_gizmo import OcclusalLoadGizmo
        from PySide6.QtWidgets import QMainWindow

        ren = vtk.vtkRenderer()
        gizmo = OcclusalLoadGizmo(ren)

        class MockFeaViewer:
            def __init__(self, g):
                self.load_gizmo = g
                self.on_load_params_changed_callback = None
            def set_load_gizmo_visible(self, vis):
                self.load_gizmo.set_visibility(vis)
            def render(self):
                pass
            def _get_implant_axis(self):
                return np.array([0.0, 0.0, -1.0])
            def _get_target_nodes(self):
                return np.array([[10.0, 20.0, 50.0], [10.0, 20.0, 30.0]])

        win = QMainWindow()
        class MockContainer:
            def __init__(self, viewer):
                self.fea_viewer = viewer
        win.mpr_container = MockContainer(MockFeaViewer(gizmo))
        panel_with_win = FeaPanel(state_mgr, parent=win)

        # 교합 하중 부여 버튼 클릭
        panel_with_win.btn_apply_load.click()
        assert gizmo.is_visible is True
        assert gizmo.gizmo_assembly.GetVisibility() == 1
        assert gizmo.origin is not None
        assert np.isclose(gizmo.magnitude_N, 100.0)

        # 각도 드래그 콜백 연동 검증
        panel_with_win._on_gizmo_angle_dragged(25.5)
        assert np.isclose(panel_with_win.spin_load_angle.value(), 25.5)

        # 경계조건 적용 클릭 시 gizmo 숨김 검증
        panel_with_win._on_setup_bc_clicked()
        assert gizmo.is_visible is False

    def test_occlusal_load_gizmo_selection_and_properties(self, qapp):
        """교합 하중 기즈모의 3D 조작 핸들 선택/숨김 토글, 노란색 텍스트, 원점 위치 검증"""
        from dental_fea_app.ui.geometry.occlusal_load_gizmo import OcclusalLoadGizmo

        gizmo = OcclusalLoadGizmo()
        assert gizmo.is_selected is False
        assert gizmo.gizmo_assembly.GetVisibility() == 0

        # 1. 전체 가시화 활성화 시 화살표와 3D 조작 기즈모 핸들이 함께 표시
        gizmo.set_visibility(True)
        assert gizmo.is_visible is True
        assert gizmo.arrow_assembly.GetVisibility() == 1
        assert gizmo.gizmo_assembly.GetVisibility() == 1
        assert gizmo.is_selected is True
        assert gizmo.text_actor.GetVisibility() == 1

        # 2. 빈 곳 클릭 시 기즈모 핸들 숨김
        gizmo.set_gizmo_handles_visible(False)
        assert gizmo.is_selected is False
        assert gizmo.gizmo_assembly.GetVisibility() == 0

        # 3. 화살표 다시 클릭 시 3D 조작 기즈모 핸들 표시
        gizmo.set_gizmo_handles_visible(True)
        assert gizmo.is_selected is True
        assert gizmo.gizmo_assembly.GetVisibility() == 1

        # 4. 노란색 텍스트 색상 검증
        tprop = gizmo.text_actor.GetTextProperty()
        r, g, b = tprop.GetColor()
        assert np.isclose(r, 1.0, atol=0.05)
        assert np.isclose(g, 0.95, atol=0.05)
        assert np.isclose(b, 0.15, atol=0.05)

        # 5. 기즈모 원점은 화살표 끝(Tip, position)과 일치
        test_pos = np.array([12.3, 45.6, 78.9])
        gizmo.update_parameters(origin=test_pos)
        pos = gizmo.gizmo_assembly.GetUserTransform().GetPosition()
        assert np.allclose(pos, test_pos)

        # 6. 화살표 화면 피킹 및 기즈모 재가시화 검증
        ren = vtk.vtkRenderer()
        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.AddRenderer(ren)
        rw.SetSize(400, 400)
        ren.AddActor(gizmo.arrow_assembly)
        ren.AddActor(gizmo.gizmo_assembly)
        ren.ResetCamera()
        rw.Render()

        ren.SetWorldPoint(float(test_pos[0]), float(test_pos[1]), float(test_pos[2]), 1.0)
        ren.WorldToDisplay()
        dp = ren.GetDisplayPoint()
        sx, sy = int(dp[0]), int(dp[1])

        # 기즈모가 숨겨진 상태에서 화살표(팁 또는 근처 5px) 클릭 시 'arrow' 검출
        gizmo.set_gizmo_handles_visible(False)
        assert gizmo.gizmo_assembly.GetVisibility() == 0
        hit = gizmo.pick(sx + 3, sy + 3, ren)
        assert hit == "arrow", f"화살표 클릭 시 'arrow'가 검출되어야 함: {hit}"
        # 화살표 클릭에 따라 기즈모 활성화
        gizmo.set_gizmo_handles_visible(True)
        assert gizmo.gizmo_assembly.GetVisibility() == 1

        # 7. rz (파란색 회전 링) 액터 및 3개 링 핸들 존재 검증
        assert "rz" in gizmo.handle_actors
        assert "rx" in gizmo.handle_actors
        assert "ry" in gizmo.handle_actors
        assert hasattr(gizmo, "rz_handle") and gizmo.rz_handle is not None

        # 8. 원 회전 시 다른 원들도 연동되어 회전하는 행렬 갱신(동기화) 검증
        initial_rot = np.copy(gizmo.rot_matrix)
        gizmo.start_drag("ry", sx + 25, sy)
        # 90도 회전에 해당하는 드래그 시뮬레이션
        gizmo.update_drag(sx, sy + 25, ren)
        gizmo.end_drag()
        # rot_matrix가 단위행렬에서 갱신되었으며 직교행렬 유지 검증
        assert not np.allclose(gizmo.rot_matrix, initial_rot)
        assert np.allclose(np.dot(gizmo.rot_matrix.T, gizmo.rot_matrix), np.eye(3), atol=1e-5)

        # 9. rz 핸들 회전 검증
        rot_before_rz = np.copy(gizmo.rot_matrix)
        gizmo.start_drag("rz", sx + 25, sy)
        gizmo.update_drag(sx, sy + 25, ren)
        gizmo.end_drag()
        assert not np.allclose(gizmo.rot_matrix, rot_before_rz)
        assert np.allclose(np.dot(gizmo.rot_matrix.T, gizmo.rot_matrix), np.eye(3), atol=1e-5)

    def test_app_icon_creation_and_mainwindow_integration(self, qapp):
        """대표 애플리케이션 아이콘 생성 및 설정 검증"""
        from dental_fea_app.ui.icons import IconFactory
        from PySide6.QtWidgets import QMainWindow

        icon = IconFactory.app_icon(256)
        assert not icon.isNull(), "생성된 대표 아이콘은 null이 아니어야 함"
        pix = icon.pixmap(256, 256)
        assert not pix.isNull()
        assert pix.width() == 256 and pix.height() == 256

        win = QMainWindow()
        win.setWindowIcon(IconFactory.app_icon())
        assert not win.windowIcon().isNull(), "설정된 대표 아이콘은 null이 아니어야 함"


    def test_setup_boundary_conditions_box_fix_by_region(self, qapp, monkeypatch):
        """'축 지정' 고정 모드(fix_by_region) 적용 시 NameError 없이 정상 수행되는지 검증"""
        from fea.models import MeshModel, MeshPart
        from dental_fea_app.ui.fea_controller import FeaController

        state_mgr = StateManager()
        controller = FeaController(state_mgr)

        # 골 절점 및 4면체 요소가 있는 메쉬 모델 모의 구성
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 10.0, 10.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type="C3D4")
        mesh_model = MeshModel(parts=[part])
        state_mgr.set_mesh_model(mesh_model)

        # 'box' 모드로 경계조건 설정 호출
        fix_box = (-1.0, 5.0, -1.0, 5.0, -1.0, 5.0)
        issues = controller.setup_boundary_conditions(
            fix_mode="box",
            fix_box=fix_box,
            load_preset_key="axial_100N",
            load_mag=100.0,
            load_angle=0.0,
        )
        assert state_mgr.bc_set is not None
        assert len(state_mgr.bc_set.fixed_node_sets) > 0
        fix_dict = state_mgr.bc_set.fixed_node_sets[0]
        assert fix_dict["name"] == "FIX_MANDIBLE_REGION"
        # 절점 0, 1, 2, 3 중 범위(-1~5) 내에 있는 절점들 고정 확인
        fixed_nodes = fix_dict["node_ids"]
        assert len(fixed_nodes) > 0

    def test_remesh_clears_downstream_state(self, qapp, monkeypatch):
        """체적 메쉬 재생성 시 물성, 경계조건, 해석 결과, 고정면 박스가 모두 제거되는지 검증"""
        from fea.models import MaterialAssignment, MaterialDef, BoundaryConditionSet

        state_mgr = StateManager()
        panel = FeaPanel(state_mgr)

        # 기존 하류 데이터 세팅
        mat = MaterialDef(name="CORTICAL", E=13700.0, nu=0.3)
        state_mgr.set_material_assignment(MaterialAssignment({"bone": np.array([0])}, [mat]))
        state_mgr.set_bc_set(BoundaryConditionSet(fixed_node_sets=[{"name": "FIX", "node_ids": np.array([0])}], loads=[]))
        state_mgr.set_last_frd_path("/tmp/test.frd")
        state_mgr.bc_fix_box = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
        panel._fixed_z = -10.0

        # _on_mesh_clicked 호출
        monkeypatch.setattr(panel.controller, "start_meshing", lambda cfg: None)
        panel._on_mesh_clicked()

        # 하류 상태가 모두 초기화되었는지 검증
        assert state_mgr.material_assignment is None
        assert state_mgr.bc_set is None
        assert state_mgr.last_frd_path is None
        assert state_mgr.bc_fix_box is None
        assert panel._fixed_z is None
        assert "물성 미할당" in panel.lbl_mat_summary.text()
        assert "경계조건 미설정" in panel.lbl_bc_status.text()

    def test_occlusal_load_gizmo_3d_handles_and_math(self, qapp):
        """교합 하중 기즈모의 양방향 3축 이동, 16mm 회전 링 가시성, 축 투영 이동, 회전 방향 수학 및 피킹 검증"""
        import vtk
        from dental_fea_app.ui.geometry.occlusal_load_gizmo import OcclusalLoadGizmo

        ren = vtk.vtkRenderer()
        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.AddRenderer(ren)
        rw.SetSize(600, 600)

        gizmo = OcclusalLoadGizmo(ren)
        start_pos = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        gizmo.update_parameters(origin=start_pos, magnitude_N=120.0, tilt_deg=10.0)
        gizmo.set_visibility(True)

        ren.ResetCamera()
        rw.Render()

        # 1. 3D 기즈모의 모든 핸들(tx, ty, tz, rx, ry) 가시성 및 조명 속성 검증
        assert "rx" in gizmo.handle_actors and "ry" in gizmo.handle_actors
        assert set(gizmo.handle_actors.keys()) >= {"tx", "ty", "tz", "rx", "ry"}
        for h_id, actor in gizmo.handle_actors.items():
            assert actor.GetVisibility() == 1, f"핸들 {h_id}는 가시 상태여야 함"
            prop = actor.GetProperty()
            assert prop.GetLighting() == 0, f"핸들 {h_id}는 고시인성을 위해 조명이 꺼져 있어야 함"
            assert np.isclose(prop.GetAmbient(), 1.0), f"핸들 {h_id}의 Ambient는 1.0이어야 함"

        # 1-1. 원 색상 검증: rx는 빨강, ry는 녹색
        rx_c = gizmo.handle_actors["rx"].GetProperty().GetColor()
        assert np.isclose(rx_c[0], 1.0, atol=0.05) and rx_c[1] < 0.4, "rx는 빨간색 원이어야 함"
        ry_c = gizmo.handle_actors["ry"].GetProperty().GetColor()
        assert ry_c[0] < 0.4 and np.isclose(ry_c[1], 0.95, atol=0.05), "ry는 녹색 원이어야 함"

        # 2. 3축 이동: 축 구속 정사영 이동 검증
        # 2-1) tx (X축 구속 이동)
        gizmo.start_drag("tx", 300, 300)
        assert gizmo.is_dragging is True
        moved = gizmo.update_drag(350, 300, ren)
        assert moved is True
        # Y와 Z는 변하지 않고 X만 이동해야 함
        assert np.isclose(gizmo.position[1], start_pos[1], atol=1e-5)
        assert np.isclose(gizmo.position[2], start_pos[2], atol=1e-5)
        assert not np.isclose(gizmo.position[0], start_pos[0])
        gizmo.end_drag()

        # 2-2) ty (Y축 구속 이동)
        gizmo.update_parameters(origin=start_pos)
        gizmo.start_drag("ty", 300, 300)
        moved = gizmo.update_drag(300, 350, ren)
        assert moved is True
        assert np.isclose(gizmo.position[0], start_pos[0], atol=1e-5)
        assert np.isclose(gizmo.position[2], start_pos[2], atol=1e-5)
        assert not np.isclose(gizmo.position[1], start_pos[1])
        gizmo.end_drag()

        # 2-3) tz (Z축 구속 이동)
        gizmo.update_parameters(origin=start_pos)
        gizmo.start_drag("tz", 300, 300)
        moved = gizmo.update_drag(300, 350, ren)
        assert moved is True
        assert np.isclose(gizmo.position[0], start_pos[0], atol=1e-5)
        assert np.isclose(gizmo.position[1], start_pos[1], atol=1e-5)
        assert not np.isclose(gizmo.position[2], start_pos[2])
        gizmo.end_drag()

        # 3. 회전 링: 빨간 원 rx (X축 중심 360도 회전) & 녹색 원 ry (Y축 중심 360도 회전)
        # 3-1) rx (빨간 원): X축 중심 회전 -> direction의 X성분은 불변, Y/Z 성분이 360도 회전
        init_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        gizmo.update_parameters(origin=start_pos, direction=init_dir)
        gizmo.start_drag("rx", 300, 350)
        moved_rx = gizmo.update_drag(350, 300, ren)
        assert moved_rx is True
        assert np.isclose(gizmo.direction[0], 0.0, atol=1e-4), "X축 중심 회전이므로 X성분은 0에 가까워야 함"
        assert not np.allclose(gizmo.direction, init_dir)
        gizmo.end_drag()

        # 3-2) ry (녹색 원): Y축 중심 회전 -> direction의 Y성분은 불변, X/Z 성분이 360도 회전
        gizmo.update_parameters(origin=start_pos, direction=init_dir)
        gizmo.start_drag("ry", 350, 300)
        moved_ry = gizmo.update_drag(300, 350, ren)
        assert moved_ry is True
        assert np.isclose(gizmo.direction[1], 0.0, atol=1e-4), "Y축 중심 회전이므로 Y성분은 0에 가까워야 함"
        assert not np.allclose(gizmo.direction, init_dir)
        gizmo.end_drag()

        # 4. 호버 하이라이트 검증
        gizmo.highlight_handle("rx")
        r, g, b = gizmo.handle_actors["rx"].GetProperty().GetColor()
        assert np.isclose(r, gizmo.highlight_colors["rx"][0])
        gizmo.highlight_handle(None)
        r, g, b = gizmo.handle_actors["rx"].GetProperty().GetColor()
        assert np.isclose(r, gizmo.base_colors["rx"][0])



