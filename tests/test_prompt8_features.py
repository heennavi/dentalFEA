"""
tests/test_prompt8_features.py
Prompt 8 요구사항 검증 테스트:
1. 3D 뷰에서 임플란트 삭제 시 식립된 임플란트 목록 테이블 동기화 (즉시 삭제 반영)
2. '2. 분할' 탭 폭 조정 및 ControlPanel 워크플로 탭바 균등 확장
3. '3. 임플란트' 탭 UI:
   - '임플란트 및 보철' 버튼이 '임플란트 모델 / 사양' 그룹 내부로 이동
   - '규격' 레이블 및 규격 콤보박스 레이아웃 제거
   - '임플란트 및 보철' 버튼 텍스트 색상 검은색(#000000), 보통 글씨(font-weight: normal)
   - '외부 STL 파일 불러오기' 버튼 폭 글자/아이콘에 맞추어 축소
4. MPRWidget의 set_selected_measurement / set_selected_measure 호환성 및 안정 동작
"""

import sys
import os
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QSizePolicy, QGroupBox
from PySide6.QtCore import Qt

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.placement.implant_generator import ImplantSpec
from dental_fea_app.ui.placement.implant_panel import ImplantPanel
from dental_fea_app.ui.placement.implant_simulation_dialog import ImplantSimulationDialog
from dental_fea_app.ui.control_panel import ControlPanel
from dental_fea_app.ui.segmentation.segmentation_panel import SegmentationPanel
from dental_fea_app.ui.mpr_widget import MPRWidget
from dental_fea_app.config import AXIS_AXIAL


class TestPrompt8Features(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        self.state_mgr = StateManager()

    def test_01_implant_deletion_sync_with_placed_table(self):
        """3D 뷰 또는 placement_mgr에서 임플란트 삭제 시 식립된 임플란트 목록 테이블이 즉시 갱신되어 삭제되는지 검증"""
        from dental_fea_app.placement.implant_generator import ImplantLibraryManager
        mgr = self.state_mgr.implant_placement_mgr
        sim_dlg = ImplantSimulationDialog(mgr)

        presets = ImplantLibraryManager.list_all_implants()
        spec1 = presets[0]
        spec2 = presets[1]

        id1 = mgr.place_implant(spec1, initial_position=(10.0, 20.0, 30.0))
        id2 = mgr.place_implant(spec2, initial_position=(40.0, 50.0, 60.0))
        self.app.processEvents()

        # 테이블 행 수가 2개인지 확인
        self.assertEqual(sim_dlg.table_placed.rowCount(), 2)

        # mgr.delete_implant(id1) 실행 (3D 뷰의 컨텍스트 메뉴에서 호출하는 방식과 동일)
        mgr.delete_implant(id1)
        self.app.processEvents()

        # 식립된 임플란트 테이블에 즉시 반영되어 행 수가 1개가 되고, 남아있는 임플란트가 id2인지 확인
        self.assertEqual(sim_dlg.table_placed.rowCount(), 1)
        self.assertEqual(sim_dlg.table_placed.item(0, 1).text(), spec2.name)

        # 전체 삭제(delete_all_implants) 시 테이블이 0행으로 비워지는지 확인
        mgr.delete_all_implants()
        self.app.processEvents()
        self.assertEqual(sim_dlg.table_placed.rowCount(), 0)

    def test_02_segmentation_panel_width_and_tab_expanding(self):
        """'2. 분할' 탭 및 ControlPanel 탭바 폭 정책 검증"""
        ctrl = ControlPanel(self.state_mgr, on_open_3d_viewer=lambda: None)
        self.assertTrue(ctrl.workflow_tabs.tabBar().expanding())
        self.assertTrue(ctrl.workflow_tabs.tabBar().documentMode())

        seg_panel = ctrl.segmentation_panel
        self.assertIsInstance(seg_panel, SegmentationPanel)
        self.assertEqual(seg_panel.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)

        # 마진이 2로 축소되어 전체 패널 폭(260px)에 최적화되었는지 확인
        margins = seg_panel.layout().contentsMargins()
        self.assertEqual(margins.left(), 2)
        self.assertEqual(margins.right(), 2)

    def test_03_implant_panel_ui_rearrangement(self):
        """'3. 임플란트' 탭: 버튼 그룹 이동, 규격 제거, 스타일, 외부 STL 버튼 폭 축소 검증"""
        panel = ImplantPanel(self.state_mgr)

        # 1) '임플란트 및 보철' 버튼이 '임플란트 모델 / 사양' 그룹박스 내부의 첫 번째 항목에 위치하는지 확인
        parent_group = panel.btn_sim_launch.parent()
        self.assertIsInstance(parent_group, QGroupBox)
        self.assertEqual(parent_group.title(), "임플란트 모델 / 사양")

        # 2) '규격' 콤보박스는 UI에서 숨겨져 있어야 함
        self.assertTrue(panel.combo_library.isHidden())

        # 3) '임플란트 및 보철' 버튼의 텍스트 색상이 검은색(#000000), 보통 글씨(font-weight: normal)인지 확인
        style = panel.btn_sim_launch.styleSheet()
        self.assertIn("color: #000000;", style)
        self.assertIn("font-weight: normal;", style)

        # 4) '외부 STL 파일 불러오기' 버튼 폭이 QSizePolicy.Policy.Maximum으로 축소 설정되었는지 확인
        self.assertEqual(panel.btn_load_stl.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Maximum)

    def test_04_mpr_widget_set_selected_measurement_alias(self):
        """MPRWidget에서 set_selected_measurement 호출 시 오류 없이 정상 동작하는지 검증"""
        widget = MPRWidget(AXIS_AXIAL, self.state_mgr)
        self.assertTrue(hasattr(widget, "set_selected_measurement"))
        self.assertTrue(hasattr(widget, "set_selected_measure"))
        self.assertEqual(widget.set_selected_measurement, widget.set_selected_measure)

        # 예외 없이 호출 가능해야 함
        widget.set_selected_measurement(-1)
        self.assertEqual(widget.selected_measure_idx, -1)


if __name__ == "__main__":
    unittest.main()
