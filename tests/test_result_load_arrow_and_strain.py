"""
tests/test_result_load_arrow_and_strain.py
Unit tests for:
1. Equivalent strain (von Mises) calculation and microstrain conversion.
2. ResultViewController load arrow visualization without 3D gizmo and single-arrow enforcement.
3. FEAGeometryViewer single load arrow enforcement.
4. Default thread count 8 across solver and UI.
5. VolumeViewerWindow implant selectable checkbox and result view mode suppression/restoration.
"""

import os, sys, ctypes
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
local_xcb_cursor = os.path.join(project_root, ".local_libs", "usr", "lib", "x86_64-linux-gnu", "libxcb-cursor.so.0")
if os.path.exists(local_xcb_cursor):
    try:
        ctypes.CDLL(local_xcb_cursor)
    except Exception:
        pass

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import unittest
import numpy as np

import vtk
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from dental_fea_app.core.state_manager import StateManager
from fea.models import MeshModel, MeshPart, BoundaryConditionSet, AnalysisConfig
from fea.results import ResultSet
from solver.runner import CalculixRunner
from postprocess.result_fields import equivalent_strain
from postprocess.vtk_builder import build_part_grid, attach_results
from ui.result_view_controller import ResultViewController, FIELD_DISPLAY_CONFIG
from ui.result_panel import ResultPanel
from ui.result_controller import ResultController
from dental_fea_app.ui.fea_panel import FeaPanel
from dental_fea_app.ui.fea_controller import FeaController
from dental_fea_app.ui.geometry.geometry_panel import GeometryPanel
from dental_fea_app.ui.control_panel import ControlPanel
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow
from dental_fea_app.ui.geometry.fea_geometry_viewer import FeaGeometryViewWidget


class TestResultLoadArrowAndStrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        self.state_mgr = StateManager()

    def _create_mock_mesh_model(self) -> MeshModel:
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        part = MeshPart(name="bone", nodes=nodes, elements=elements, element_type="C3D4")
        return MeshModel(parts=[part])

    def test_01_equivalent_strain_calculation(self):
        """등가 변형률(Equivalent Strain) 공식 계산 및 특수 케이스 검증"""
        # 1) 순수 인장: exx = 0.003, 기타 0
        # diff_sq = 2 * (0.003)^2 = 1.8e-5
        # eq = (sqrt(2)/3) * sqrt(1.8e-5) = (sqrt(2)/3) * (sqrt(2) * 0.003) = (2/3) * 0.003 = 0.002
        strain_pure_tension = np.array([[0.003, 0.0, 0.0, 0.0, 0.0, 0.0]])
        eq_tens = equivalent_strain(strain_pure_tension)
        self.assertAlmostEqual(float(eq_tens[0]), 0.002, places=7)

        # 2) 순수 전단: exy = 0.001, 기타 0
        # shear_sq = (0.001)^2, 6 * shear_sq = 6e-6
        # eq = (sqrt(2)/3) * sqrt(6e-6) = (sqrt(12)/3) * 0.001 = (2*sqrt(3)/3) * 0.001
        strain_pure_shear = np.array([[0.0, 0.0, 0.0, 0.001, 0.0, 0.0]])
        eq_shear = equivalent_strain(strain_pure_shear)
        expected_shear = (2.0 * np.sqrt(3.0) / 3.0) * 0.001
        self.assertAlmostEqual(float(eq_shear[0]), expected_shear, places=7)

        # 3) 정수압 변형 (등방 팽창): exx=eyy=ezz, shear=0 -> 등가 변형률은 0이어야 함
        strain_hydro = np.array([[0.005, 0.005, 0.005, 0.0, 0.0, 0.0]])
        eq_hydro = equivalent_strain(strain_hydro)
        self.assertAlmostEqual(float(eq_hydro[0]), 0.0, places=7)

        # 4) NaN 포함 행 검증
        strain_nan = np.array([[np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]])
        eq_nan = equivalent_strain(strain_nan)
        self.assertTrue(np.isnan(eq_nan[0]))

    def test_02_attach_results_equiv_strain_in_microstrain(self):
        """attach_results 호출 시 VTK Grid에 equiv_strain이 µε 단위로 부착되는지 검증"""
        model = self._create_mock_mesh_model()
        grid = build_part_grid(model.parts[0], linear_only=True)

        strain6 = np.array([
            [0.003, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.003, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.003, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.003, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)

        rs = ResultSet(time=1.0, strain6=strain6)
        attach_results(grid, rs, model, "bone")

        arr = grid.GetPointData().GetArray("equiv_strain")
        self.assertIsNotNone(arr, "equiv_strain 배열이 VTK PointData에 부착되어야 합니다.")
        # eq = 0.002 -> microstrain = 0.002 * 1e6 = 2000.0 µε
        for i in range(4):
            self.assertAlmostEqual(arr.GetValue(i), 2000.0, places=3)

    def test_03_field_display_config_and_result_panel_combo(self):
        """FIELD_DISPLAY_CONFIG 및 ResultPanel combo_field에 equiv_strain 등록 확인"""
        self.assertIn("equiv_strain", FIELD_DISPLAY_CONFIG)
        self.assertEqual(FIELD_DISPLAY_CONFIG["equiv_strain"]["unit"], "µε")

        panel = ResultPanel(self.state_mgr)
        fields = [panel.combo_field.itemData(i) for i in range(panel.combo_field.count())]
        self.assertIn("equiv_strain", fields)

        # chk_show_load 체크박스 검증
        self.assertTrue(hasattr(panel, "chk_show_load"))
        signal_received = []
        panel.load_arrow_toggled.connect(lambda v: signal_received.append(v))
        panel.chk_show_load.setChecked(True)
        self.assertEqual(signal_received, [True])

    def test_04_default_thread_count_eight(self):
        """기본 스레드 수 8 검증"""
        # 1. AnalysisConfig
        cfg = AnalysisConfig.from_dict({})
        self.assertEqual(cfg.n_threads, 8)

        # 2. CalculixRunner
        runner = CalculixRunner()
        self.assertEqual(runner.n_threads, 8)

        # 3. FeaPanel
        panel = FeaPanel(self.state_mgr)
        self.assertEqual(panel.spin_threads.value(), 8)

    def test_05_result_view_controller_load_arrow_single_no_gizmo(self):
        """ResultViewController: 하중 화살표 표시 시 3D 기즈모 없이 단 1개만 생성 및 제거 검증"""
        renderer = vtk.vtkRenderer()
        ctrl = ResultViewController(renderer)
        model = self._create_mock_mesh_model()
        rs = ResultSet(time=1.0)
        ctrl.set_results(model, rs)

        bc = BoundaryConditionSet(
            fixed_node_sets=[],
            loads=[
                {"part": "bone", "node_ids": np.array([0, 1]), "force_N": (0.0, 0.0, -100.0)},
                {"part": "bone", "node_ids": np.array([2, 3]), "force_N": (10.0, 0.0, -50.0)},
            ]
        )

        # 하중 화살표 활성화
        ctrl.set_load_arrow_visible(True, bc)
        self.assertTrue(ctrl._load_arrow_visible)
        # 화살표 액터(1개) + 텍스트 액터(1개) = 총 2개의 Prop만 존재 (중복 로드 추가 없이 단 1개의 하중만 렌더링)
        self.assertEqual(len(ctrl._load_arrow_actors), 2)

        # 텍스트 라벨에 하중과 각도가 모두 포함되어 있는지 확인
        text_actor = ctrl._load_arrow_actors[1]
        self.assertIsInstance(text_actor, vtk.vtkBillboardTextActor3D)
        label_text = text_actor.GetInput()
        self.assertIn("N", label_text)
        self.assertIn("°", label_text)
        self.assertIn("100.0 N", label_text)
        self.assertIn("0.0°", label_text)

        # 비활성화 시 완전 제거
        ctrl.set_load_arrow_visible(False)
        self.assertEqual(len(ctrl._load_arrow_actors), 0)

    def test_06_volume_viewer_implant_selectable_and_result_mode(self):
        """VolumeViewerWindow: 임플란트 선택 가능 체크박스 및 해석 결과 뷰 모드 차단/복구 검증"""
        viewer = VolumeViewerWindow(self.state_mgr)
        self.assertTrue(hasattr(viewer, "chk_implant_selectable"))
        # 초기값: 체크 상태
        self.assertTrue(viewer.chk_implant_selectable.isChecked())

        # 해석 결과 뷰 모드 진입 시: 체크 해제
        viewer.set_result_view_mode(True)
        self.assertTrue(viewer.is_result_view_mode)
        self.assertFalse(viewer.chk_implant_selectable.isChecked())

        # 해석 결과 뷰 모드 해제 시: 원래대로 복구 (체크 활성화)
        viewer.set_result_view_mode(False)
        self.assertFalse(viewer.is_result_view_mode)
        self.assertTrue(viewer.chk_implant_selectable.isChecked())

        # 사용자가 직접 체크 해제한 경우 기즈모 detach 검증
        viewer.chk_implant_selectable.setChecked(False)
        self.assertFalse(viewer.chk_implant_selectable.isChecked())
        # 임플란트 선택 신호가 와도 기즈모 부착되지 않아야 함
        viewer._on_implant_selected_3d("non_existent_or_mock")
        if hasattr(viewer, "implant_gizmo_3d"):
            self.assertFalse(viewer.implant_gizmo_3d.is_attached)

        try:
            viewer.close()
        except Exception:
            pass

    def test_07_fea_geometry_viewer_single_arrow_enforcement(self):
        """FEAGeometryViewer: 하중 화살표 중복 방지 (2개 이상 표시 방지) 검증"""
        viewer = FeaGeometryViewWidget(self.state_mgr)
        model = self._create_mock_mesh_model()
        self.state_mgr.set_mesh_model(model)

        bc = BoundaryConditionSet(
            fixed_node_sets=[],
            loads=[
                {"part": "bone", "node_ids": np.array([0, 1]), "force_N": (0.0, 0.0, -100.0)},
                {"part": "bone", "node_ids": np.array([2, 3]), "force_N": (0.0, 0.0, -50.0)},
            ]
        )

        # 1) load_gizmo가 꺼져 있을 때: 단 1개의 화살표 액터만 bc_actors에 추가됨
        viewer.load_gizmo.set_visibility(False)
        viewer._on_bc_set_changed(bc)
        # bc_actors 내 액터 수가 1개여야 함 (2개 이상의 화살표 생성 금지)
        self.assertEqual(len(viewer.bc_actors), 1)

        # 2) load_gizmo가 켜져 있을 때: bc_actors에 중복 화살표가 추가되지 않음
        viewer.set_load_gizmo_visible(True)
        viewer._on_bc_set_changed(bc)
        self.assertEqual(len(viewer.bc_actors), 0)

    def test_08_tab_order_and_default_parameters(self):
        """탭 순서 ('2. 임플란트', '3. 분할') 및 기본 파라미터 (0.40mm) 검증"""
        # 1. ControlPanel 탭 순서 검증
        cp = ControlPanel(self.state_mgr, on_open_3d_viewer=lambda: None)
        tab_texts = [cp.workflow_tabs.tabText(i) for i in range(cp.workflow_tabs.count())]
        expected_tabs = ["1. 영상", "2. 임플란트", "3. 분할", "4. 기하 처리", "5. 해석", "6. 결과"]
        self.assertEqual(tab_texts, expected_tabs)

        # 2. GeometryPanel 삼각형 크기 기본값 0.40mm 검증
        gp = GeometryPanel(self.state_mgr)
        self.assertAlmostEqual(gp.spin_target_edge.value(), 0.40, places=2)

        # 3. FeaPanel 임플란트 주변 메쉬 크기 기본값 0.40mm 검증
        fp = FeaPanel(self.state_mgr)
        self.assertAlmostEqual(fp.spin_size_near.value(), 0.40, places=2)

        # 4. AnalysisConfig.from_dict 기본값 0.40mm 검증
        cfg = AnalysisConfig.from_dict({})
        self.assertAlmostEqual(cfg.mesh_size_near_mm, 0.40, places=2)


if __name__ == "__main__":
    unittest.main()
