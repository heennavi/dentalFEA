"""
test_user_info_and_drawing.py
신규 고도화 기능 종합 단위 테스트:
1. 메뉴바 호버 및 선택 QSS 스타일 적용
2. 그리기 도구: '선택 취소' 및 신규 선 끝단(양쪽 원형 double_circle 등) 아이콘/옵션
3. 계측 도구: 2점 드래그 길이 측정 및 3점 순차 클릭 각도 측정
4. 툴버튼 테두리 활성화: 주석, 계측, 도형 존재 시 얇은 파란색 테두리 적용
5. User Info JSON 저장 및 로드 영속성 (saved/ 폴더 연동)
"""

import os
import sys
import json
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Qt xcb 라이브러리 프리로드
import ctypes
local_xcb_cursor = os.path.join(project_root, ".local_libs", "usr", "lib", "x86_64-linux-gnu", "libxcb-cursor.so.0")
if os.path.exists(local_xcb_cursor):
    try:
        ctypes.CDLL(local_xcb_cursor)
    except Exception:
        pass

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor
from dental_fea_app.config import DARK_THEME_QSS
from dental_fea_app.imaging.dicom_reader import DicomReader
from dental_fea_app.ui.main_window import MainWindow
from dental_fea_app.ui.draw_dialog import DrawDialog
from dental_fea_app.ui.measurement_dialog import MeasurementDialog
from tests.create_dummy_data import generate_synthetic_dental_ct


class TestUserInfoAndDrawing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

        cls.test_file = os.path.join(project_root, "tests", "sample_data", "test_mandible.mha")
        if not os.path.exists(cls.test_file):
            generate_synthetic_dental_ct(cls.test_file, size=(60, 60, 40), spacing=(1.0, 1.0, 1.0))

    def test_01_menubar_hover_qss(self):
        """메뉴바 및 메뉴 항목 마우스 호버/선택 스타일 확인"""
        self.assertIn("QMenuBar::item:selected", DARK_THEME_QSS)
        self.assertIn("QMenuBar::item:hover", DARK_THEME_QSS)
        self.assertIn("QMenu::item:selected", DARK_THEME_QSS)

    def test_02_draw_dialog_options_and_serialization(self):
        """그리기 도구 형태, 선분 끝단, 색상 및 to_dict/from_dict 직렬화 검증"""
        dlg = DrawDialog()
        # 0: 선택 취소, 1: 선분, 2: 화살표, 3: 사각형, 4: 원, 5: 프리폼
        dlg.bg_shape.button(0).click()
        self.assertEqual(dlg.shape_type, "select")

        dlg.bg_shape.button(3).click()
        self.assertEqual(dlg.shape_type, "rect")

        # 선 끝단 캡 (4: 양쪽 원형 double_circle)
        dlg.bg_cap.button(4).click()
        self.assertEqual(dlg.end_cap, "double_circle")

        # 도형 추가
        dlg.add_shape({
            "shape_type": "rect",
            "view_name": "Axial",
            "slice_idx": 20,
            "pts": [(10.0, 10.0, 5.0), (30.0, 30.0, 5.0)],
            "color": "#eb4b4b",
            "width": 2,
            "is_dashed": False,
            "end_cap": "double_circle"
        })
        self.assertEqual(len(dlg.shapes), 1)

        # 직렬화 및 역직렬화
        s_dict = dlg.to_dict()
        self.assertIsInstance(s_dict, list)
        self.assertEqual(len(s_dict), 1)
        self.assertEqual(s_dict[0]["shape_type"], "rect")

        dlg2 = DrawDialog()
        dlg2.from_dict(s_dict)
        self.assertEqual(len(dlg2.shapes), 1)
        self.assertEqual(dlg2.shapes[0]["shape_type"], "rect")
        dlg.close()
        dlg2.close()

    def test_03_measurement_geometry_builders(self):
        """MPRWidget 길이 및 각도 PolyData 생성 및 라벨 계산 검증"""
        main_win = MainWindow()
        vtk_img, meta = DicomReader.load_single_file(self.test_file)
        main_win.state_mgr.set_volume_data(vtk_img, meta)

        axial_w = main_win.mpr_container.axial_panel.mpr_widget

        # 1. 길이 측정 PolyData (pt1: (10, 10, 0), pt2: (40, 50, 0)) -> dist = hypot(30, 40) = 50.0
        poly_len, dist_val, mid_pt = axial_w._build_measure_length_polydata((10.0, 10.0, 0.0), (40.0, 50.0, 0.0))
        self.assertAlmostEqual(dist_val, 50.0, places=2)
        self.assertGreater(poly_len.GetNumberOfPoints(), 2)

        # 2. 각도 측정 PolyData (p1: (10, 0, 0), p2: (0, 0, 0), p3: (0, 10, 0)) -> 90.0도
        poly_ang, ang_val, label_pt = axial_w._build_measure_angle_polydata((10.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 10.0, 0.0))
        self.assertAlmostEqual(ang_val, 90.0, places=1)
        self.assertGreater(poly_ang.GetNumberOfPoints(), 2)

        main_win.close()

    def test_04_user_info_json_save_and_load(self):
        """User Info JSON 저장 및 로드 워크플로우 검증"""
        main_win = MainWindow()
        vtk_img, meta = DicomReader.load_single_file(self.test_file)
        main_win.state_mgr.set_volume_data(vtk_img, meta)

        panel = main_win.mpr_container.axial_panel
        panel._open_draw_dialog()
        panel._open_measure_dialog()
        panel._open_annotation_dialog()

        # 데이터 입력
        panel._annotation_dialog.editor.setPlainText("치과 임플란트 46번 부위 식립 계획")
        panel._draw_dialog.add_shape({
            "shape_type": "arrow", "view_name": "Axial", "slice_idx": 20,
            "pts": [(10.0, 10.0, 0.0), (30.0, 30.0, 0.0)], "color": "#00e676"
        })
        panel._measure_dialog.add_measurement({
            "type": "length", "value_str": "12.50 mm", "view_name": "Axial",
            "slice_idx": 20, "world_pts": [(10.0, 10.0, 0.0), (22.5, 10.0, 0.0)]
        })

        # 테두리 갱신 확인
        panel._update_tool_buttons_border()
        self.assertIn("border: 2px solid #2196f3", panel.btn_measure.styleSheet())
        self.assertIn("border: 2px solid #2196f3", panel.btn_annotation.styleSheet())
        self.assertIn("border: 2px solid #2196f3", panel.btn_draw.styleSheet())

        # 임시 JSON 파일 저장
        saved_dir = os.path.join(project_root, "saved")
        os.makedirs(saved_dir, exist_ok=True)
        test_json_path = os.path.join(saved_dir, "test_case_user_info.json")

        user_data = main_win.mpr_container.get_user_info_data()
        self.assertIn("치과 임플란트", user_data["annotations"]["text"])
        self.assertEqual(len(user_data["shapes"]), 1)
        self.assertEqual(len(user_data["measurements"]), 1)

        with open(test_json_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)

        # 새 창에서 로드 테스트
        main_win2 = MainWindow()
        main_win2.state_mgr.set_volume_data(vtk_img, meta)
        with open(test_json_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        main_win2.mpr_container.set_user_info_data(loaded_data)

        panel2 = main_win2.mpr_container.axial_panel
        self.assertIn("border: 2px solid #2196f3", panel2.btn_measure.styleSheet())
        self.assertIn("border: 2px solid #2196f3", panel2.btn_annotation.styleSheet())
        self.assertIn("border: 2px solid #2196f3", panel2.btn_draw.styleSheet())

        self.assertIn("치과 임플란트", panel2._annotation_dialog.editor.toPlainText())
        self.assertEqual(len(panel2._draw_dialog.shapes), 1)
        self.assertEqual(len(panel2._measure_dialog.measurements), 1)

        # 정리
        if os.path.exists(test_json_path):
            os.remove(test_json_path)
        main_win.close()
        main_win2.close()


if __name__ == "__main__":
    unittest.main()
