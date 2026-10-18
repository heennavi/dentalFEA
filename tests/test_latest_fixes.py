import os, sys, ctypes
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_xcb_cursor = os.path.join(project_root, ".local_libs", "usr", "lib", "x86_64-linux-gnu", "libxcb-cursor.so.0")
if os.path.exists(local_xcb_cursor):
    try:
        ctypes.CDLL(local_xcb_cursor)
    except Exception:
        pass
"""
test_latest_fixes.py
최신 버그 수정 및 개선 사항 검증 테스트:
1. 임플란트 패널: 위치 및 회전 미세조정 그룹 삭제(숨김) 및 임플란트 선택 시 모델/치수 갱신
2. 2D 뷰: 도구 아이콘 선택/내용 존재 시 흰색 테두리, 맨 왼쪽 초기화 아이콘 동작, Sagittal 뷰 기즈모 녹색 화살표
3. 주석 작성 창: RichText 방지 및 붙여넣기 시 plain text 처리
4. 계측/그리기 도구: 더블 클릭 시 편집 불가(NoEditTriggers, ~ItemIsEditable) 및 행 전체 선택
5. 3D 뷰: QMenu 정의 확인 및 크로스 선 불투명도 0.125(기존 0.5의 25%) 확인
"""

import unittest
import numpy as np
from PySide6.QtWidgets import QApplication, QTableWidget, QAbstractItemView
from PySide6.QtCore import Qt, QPoint

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.ui.mpr_container import MPRContainer
from dental_fea_app.ui.placement.implant_panel import ImplantPanel
from dental_fea_app.placement.implant_placement_manager import ImplantPlacementManager
from dental_fea_app.placement.implant_generator import ImplantSpec, ImplantLibraryManager
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow
from dental_fea_app.ui.annotation_dialog import AnnotationDialog
from dental_fea_app.ui.measurement_dialog import MeasurementDialog
from dental_fea_app.ui.draw_dialog import DrawDialog


app = QApplication.instance() or QApplication([])


class TestLatestFixes(unittest.TestCase):

    def setUp(self):
        self.state_mgr = StateManager()

    def test_implant_panel_adjust_group_and_selection_update(self):
        panel = ImplantPanel(self.state_mgr)
        # 1) 위치 및 회전 미세조정 그룹이 숨겨져 있거나 UI 레이아웃에 직접 추가되지 않음
        self.assertTrue(panel.btn_reset_pose.parent().isHidden())

        # 2) 임플란트 선택 시 모델 및 치수 갱신
        mgr = self.state_mgr.implant_placement_mgr
        implants = ImplantLibraryManager.list_all_implants()
        spec = implants[0] if implants else ImplantSpec(
            id="test1",
            name="Dentis_OneQ_Ø4.5x10",
            diameter_mm=4.5,
            length_mm=10.0,
            taper_deg=2.0,
            top_type="collar",
            thread_type="triangular",
            micro_threads=True
        )
        imp_id = mgr.place_implant(spec, [10.0, 20.0, 30.0], is_maxilla=False)
        self.state_mgr.implant_selected.emit(imp_id)

        self.assertIn("모델:", panel.lbl_stl_name.text())
        self.assertIn("치수:", panel.lbl_stl_dim.text())
        self.assertIn(f"{spec.diameter_mm:.1f}", panel.lbl_stl_dim.text())
        self.assertIn(f"{spec.length_mm:.1f}", panel.lbl_stl_dim.text())

    def test_2d_view_reset_and_white_borders(self):
        container = MPRContainer(self.state_mgr, on_open_3d_callback=lambda: None)
        axial = container.axial_panel
        axial.set_tools_enabled(True)

        # 1) 초기화 버튼이 도구 버튼 맨 왼쪽에 존재
        self.assertTrue(hasattr(axial, "btn_reset_view"))
        self.assertEqual(axial.btn_reset_view.toolTip(), "뷰 초기화 (회전/대칭 리셋)")

        # 임의의 회전 및 대칭 적용
        axial.mpr_widget.rotate_cw_90()
        axial.mpr_widget.flip_horizontal()

        # 초기화 버튼 클릭 (reset_view_transforms)
        axial.btn_reset_view.click()
        self.assertEqual(axial.mpr_widget._rotation_angle, 0)
        self.assertFalse(axial.mpr_widget._flip_h)
        self.assertFalse(axial.mpr_widget._flip_v)

        # 2) 아이콘 선택 시 흰색 테두리 확인
        self.assertIn("border: 1px solid #ffffff", axial.btn_reset_view.styleSheet())

        # 3) Sagittal 뷰 기즈모 H-axis 화살표 색상이 녹색인지 확인
        sagittal = container.sagittal_panel
        self.assertEqual(sagittal.mpr_widget.implant_gizmo_2d.color_h, (0.20, 0.85, 0.30))

    def test_annotation_plain_text_and_white_color(self):
        dlg = AnnotationDialog()
        self.assertFalse(dlg.editor.acceptRichText())
        pal = dlg.editor.palette()
        self.assertEqual(pal.color(pal.ColorRole.Text).name().lower(), "#ffffff")

    def test_table_no_edit_and_row_select(self):
        # 계측 테이블
        m_dlg = MeasurementDialog()
        self.assertEqual(m_dlg.table.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertEqual(m_dlg.table.selectionBehavior(), QTableWidget.SelectionBehavior.SelectRows)

        m_dlg.add_measurement({
            "type": "length", "value_str": "15.00 mm", "view_name": "Axial",
            "slice_idx": 10, "world_pts": [(0, 0, 0), (15, 0, 0)]
        })
        item0 = m_dlg.table.item(0, 0)
        self.assertFalse(item0.flags() & Qt.ItemFlag.ItemIsEditable)

        # 그리기 테이블
        d_dlg = DrawDialog()
        self.assertEqual(d_dlg.table.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertEqual(d_dlg.table.selectionBehavior(), QTableWidget.SelectionBehavior.SelectRows)

        d_dlg.add_shape({
            "shape_type": "line", "color": "#ff0000", "view_name": "Axial",
            "slice_idx": 5, "pts": [[0, 0, 0], [10, 10, 0]], "width": 1,
            "is_dashed": False, "end_cap": "none"
        })
        d_item0 = d_dlg.table.item(0, 0)
        self.assertFalse(d_item0.flags() & Qt.ItemFlag.ItemIsEditable)

    def test_volume_viewer_qmenu_and_cross_opacity(self):
        viewer = VolumeViewerWindow(self.state_mgr)
        # 1) QMenu 임포트 및 컨텍스트 메뉴 동작 검증
        self.assertTrue(hasattr(viewer, "_show_implant_context_menu"))

        # 2) 크로스 선 불투명도 0.125 확인
        self.assertAlmostEqual(viewer._cross_x_actor.GetProperty().GetOpacity(), 0.125, places=3)
        self.assertAlmostEqual(viewer._cross_y_actor.GetProperty().GetOpacity(), 0.125, places=3)
        self.assertAlmostEqual(viewer._cross_z_actor.GetProperty().GetOpacity(), 0.125, places=3)
        viewer.close()


if __name__ == "__main__":
    unittest.main()
