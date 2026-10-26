"""
test_result_panel_and_controller.py
Unit and headless integration tests for Step 4-6:
- ResultPanel UI initial offscreen state and controls disabled without results
- ResultController headless pipeline execution with synthetic MeshModel and .frd
- Verification that state_mgr.result_set, result_stats slots are populated
- Statistics table row count = num_parts + 1 (parts + peri-implant bone)
- Upstream invalidation resets result_set to None and marks UI stale
- Mismatched node count error handling without crashing
- ControlPanel workflow tab integration and auto-switch
"""

import os
import pytest
import numpy as np
from PySide6.QtWidgets import QApplication

from fea.models import MeshModel, MeshPart, BoundaryConditionSet
from dental_fea_app.core.state_manager import StateManager
from ui.result_panel import ResultPanel
from ui.result_controller import ResultController
from dental_fea_app.ui.control_panel import ControlPanel


os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def cube_frd_path():
    """Path to valid synthetic CalculiX .frd test fixture."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "tests", "fixtures", "cube_compression.frd")
    assert os.path.exists(path), f"Fixture not found: {path}"
    return path


@pytest.fixture
def matching_cube_model() -> MeshModel:
    """
    Create a 2-part MeshModel matching the 8 nodes in tests/fixtures/cube_compression.frd.
    Part 1: 'bone' (nodes 0..3, bottom surface)
    Part 2: 'implant' (nodes 4..7, top surface)
    Total nodes: 8.
    """
    all_coords = np.array([
        [0.0, 0.0, 0.0],    # 0 (1-based: 1)
        [10.0, 0.0, 0.0],   # 1 (1-based: 2)
        [10.0, 10.0, 0.0],  # 2 (1-based: 3)
        [0.0, 10.0, 0.0],   # 3 (1-based: 4)
        [0.0, 0.0, 10.0],   # 4 (1-based: 5)
        [10.0, 0.0, 10.0],  # 5 (1-based: 6)
        [10.0, 10.0, 10.0], # 6 (1-based: 7)
        [0.0, 10.0, 10.0],  # 7 (1-based: 8)
    ], dtype=np.float64)

    # Part 1: bone (4 nodes)
    bone_nodes = all_coords[:4]
    bone_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    bone_part = MeshPart(name="bone", nodes=bone_nodes, elements=bone_elements, element_type="C3D4")

    # Part 2: implant (4 nodes)
    implant_nodes = all_coords[4:]
    implant_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    implant_part = MeshPart(name="implant", nodes=implant_nodes, elements=implant_elements, element_type="C3D4")

    return MeshModel(parts=[bone_part, implant_part])


class TestResultPanelOffscreen:
    """Test ResultPanel initial state without results."""

    def test_initial_state_disabled(self, qapp):
        state_mgr = StateManager()
        panel = ResultPanel(state_mgr)

        # 결과 불러오기 버튼은 활성
        assert panel.btn_open_frd.isEnabled() is True

        # 표시 및 통계 그룹은 비활성
        assert panel.group_display.isEnabled() is False
        assert panel.group_stats.isEnabled() is False

        # 통계 표 행 수는 0
        assert panel.table_stats.rowCount() == 0

        # 상태 라벨
        assert "결과 없음" in panel.lbl_status.text()
        assert panel.banner_stale.isVisible() is False


class TestResultControllerHeadless:
    """Test ResultController and StateManager result slots."""

    def test_load_frd_populates_slots_and_table(self, qapp, cube_frd_path, matching_cube_model):
        state_mgr = StateManager()
        state_mgr.set_mesh_model(matching_cube_model)

        # 경계조건(반력 검산용)
        bc = BoundaryConditionSet(
            fixed_node_sets=[
                {"name": "FIX_BOTTOM", "part": "bone", "node_ids": np.array([0, 1, 2, 3], dtype=np.int64), "dofs": (1, 3)}
            ],
            loads=[
                {"name": "LOAD_TOP", "part": "implant", "node_ids": np.array([0, 1, 2, 3], dtype=np.int64), "force_N": (0.0, 0.0, -100.0)}
            ],
        )
        state_mgr.set_bc_set(bc)

        panel = ResultPanel(state_mgr)
        controller = ResultController(state_mgr, panel=panel)

        # 동기(blocking) 로드 실행
        controller.load_frd(cube_frd_path, blocking=True)

        # 1. StateManager 슬롯 검증
        assert state_mgr.result_set is not None
        assert state_mgr.result_stats is not None
        assert state_mgr.result_source_frd == cube_frd_path
        assert len(state_mgr.result_set.u_mag) == 8

        # 2. ResultPanel UI 갱신 검증
        assert panel.group_display.isEnabled() is True
        assert panel.group_stats.isEnabled() is True
        assert panel.btn_open_frd.isEnabled() is True
        assert "로드 완료" in panel.lbl_status.text()

        # 3. 통계 표 행 수: 파트 수(2) + 임플란트 주변 뼈(1) = 3
        expected_rows = len(matching_cube_model.parts) + 1
        assert panel.table_stats.rowCount() == expected_rows

        # 행 라벨 검사
        labels = [panel.table_stats.verticalHeaderItem(r).text() for r in range(expected_rows)]
        assert "bone" in labels
        assert "implant" in labels
        assert any("임플란트 주변 뼈" in l for l in labels)

        # 4. 최대 응력 표시 슬롯 호출 검증 (AttributeError 방지)
        controller.mark_max_stress("von_mises", "ALL")

    def test_upstream_invalidation_clears_result_set(self, qapp, cube_frd_path, matching_cube_model):
        state_mgr = StateManager()
        state_mgr.set_mesh_model(matching_cube_model)
        panel = ResultPanel(state_mgr)
        controller = ResultController(state_mgr, panel=panel)

        # 로드 성공
        panel.show()
        controller.load_frd(cube_frd_path, blocking=True)
        assert state_mgr.result_set is not None
        assert panel.group_display.isEnabled() is True

        # 상류 무효화 신호 발행 (예: 임플란트 이동 또는 형상 변경)
        state_mgr.set_fea_stale(True)

        # result_set이 None으로 초기화되고 패널이 비활성화되는지 검증
        assert state_mgr.result_set is None
        assert state_mgr.result_stats is None
        assert panel.group_display.isEnabled() is False
        assert not panel.banner_stale.isHidden()
        assert "결과가 현재 모델과 맞지 않음" in panel.banner_stale.text()

    def test_mismatched_model_shows_error(self, qapp, cube_frd_path):
        state_mgr = StateManager()
        # 5개 절점을 가진 불일치 모델 생성
        mismatched_part = MeshPart(
            name="wrong_part",
            nodes=np.zeros((5, 3)),
            elements=np.array([[0, 1, 2, 3]]),
            element_type="C3D4",
        )
        state_mgr.set_mesh_model(MeshModel(parts=[mismatched_part]))

        panel = ResultPanel(state_mgr)
        controller = ResultController(state_mgr, panel=panel)

        failed_messages = []
        controller.load_failed.connect(failed_messages.append)

        # 불일치 파일 로드 시도 -> 크래시 없이 load_failed 발생
        controller.load_frd(cube_frd_path, blocking=True)

        assert len(failed_messages) == 1
        assert "일치하지 않습니다" in failed_messages[0]
        assert state_mgr.result_set is None
        assert panel.group_display.isEnabled() is False


class TestControlPanelIntegration:
    """Test ControlPanel workflow tab 6 registration and auto-switch."""

    def test_result_tab_registration_and_activation(self, qapp, cube_frd_path, matching_cube_model):
        state_mgr = StateManager()
        state_mgr.set_mesh_model(matching_cube_model)

        control_panel = ControlPanel(
            state_manager=state_mgr,
            on_open_3d_viewer=lambda: None,
        )

        tabs = control_panel.workflow_tabs
        # 탭 6 "6. 결과"가 존재하는지 확인 (Phase 4 6개 탭 또는 Phase 5 7개 탭)
        assert tabs.count() in (6, 7)
        assert "결과" in tabs.tabText(5)

        # 초기에는 5번 탭 비활성
        assert tabs.isTabEnabled(5) is False

        # 결과 로드 수행
        control_panel.result_controller.load_frd(cube_frd_path, blocking=True)

        # 로드 완료 시 탭 활성화 및 전환
        assert tabs.isTabEnabled(5) is True
        assert tabs.currentIndex() == 5

    def test_ui_display_controls_forwarding(self, qapp, cube_frd_path, matching_cube_model):
        """Test user interaction widgets forward signals properly to controller."""
        state_mgr = StateManager()
        state_mgr.set_mesh_model(matching_cube_model)
        panel = ResultPanel(state_mgr)
        controller = ResultController(state_mgr, panel=panel)

        controller.load_frd(cube_frd_path, blocking=True)

        # 1. 필드 변경
        panel.combo_field.setCurrentIndex(1)  # u_mag
        assert panel.combo_field.currentData() == "u_mag"

        # 2. 컬러맵 변경
        panel.combo_cmap.setCurrentIndex(1)  # viridis
        assert panel.combo_cmap.currentData() == "viridis"

        # 3. 변형 스케일 슬라이더 -> 스핀박스 동기화
        panel.slider_scale.setValue(25)
        assert panel.spin_scale.value() == 25.0

        # 4. 프로브 모드 토글
        panel.btn_probe_toggle.setChecked(True)
        assert controller.probe._probe_enabled is True
        panel.btn_probe_toggle.setChecked(False)
        assert controller.probe._probe_enabled is False
