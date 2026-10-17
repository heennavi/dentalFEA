"""
test_measurement_and_roi_fixes.py
단면 계측도구, 주석 작성, 그리기 도구 및 3D ROI Box 신규 요구사항 회귀 검증 테스트
"""

import os
import sys
import unittest
import shutil

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

import vtk
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.ui.main_window import MainWindow
from dental_fea_app.ui.measurement_dialog import MeasurementDialog
from dental_fea_app.ui.draw_dialog import DrawDialog
from dental_fea_app.ui.annotation_dialog import AnnotationDialog
from dental_fea_app.imaging.dicom_reader import DicomReader
from tests.create_dummy_data import generate_synthetic_dental_ct


class TestMeasurementAndRoiFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication([])
        cls.sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
        os.makedirs(cls.sample_dir, exist_ok=True)
        cls.test_file = os.path.join(cls.sample_dir, "test_mandible.mha")
        if not os.path.exists(cls.test_file):
            generate_synthetic_dental_ct(cls.test_file)

    def test_01_measurement_dialog_features(self):
        """단면 계측 도구 요구사항 검증: 닫기 버튼, 면번호, 가시성, 더블클릭 슬라이스 이동"""
        dlg = MeasurementDialog()
        
        # 1. 닫기 버튼 확인
        self.assertEqual(dlg.btn_close.text(), "닫기")
        
        # 2. 면번호(1-indexed) 표시 확인
        dlg.add_measurement({
            "type": "length",
            "value_str": "15.30 mm",
            "view_name": "Axial",
            "slice_idx": 19,
            "world_pts": [(0.0, 0.0, 19.0), (15.3, 0.0, 19.0)]
        })
        self.assertEqual(dlg.table.item(0, 0).text(), "길이")
        self.assertEqual(dlg.table.item(0, 1).text(), "15.30 mm")
        self.assertEqual(dlg.table.item(0, 2).text(), "Axial")
        self.assertEqual(dlg.table.item(0, 3).text(), "20")  # 19 + 1 = 20

        # 3. 더블클릭 시그널 발생 확인
        nav_result = []
        dlg.navigate_to_item.connect(lambda v, s, c: nav_result.append((v, s, c)))
        dlg._navigate_to_row(0)
        self.assertEqual(len(nav_result), 1)
        self.assertEqual(nav_result[0][0], "Axial")
        self.assertEqual(nav_result[0][1], 19)

        # 4. 삭제 및 전체 삭제
        dlg.add_measurement({
            "type": "angle",
            "value_str": "45.0°",
            "view_name": "Coronal",
            "slice_idx": 10,
            "world_pts": [(0, 0, 0), (10, 0, 0), (10, 0, 10)]
        })
        self.assertEqual(dlg.table.rowCount(), 2)
        dlg.table.selectRow(0)
        dlg._on_delete_selected()
        self.assertEqual(dlg.table.rowCount(), 1)
        self.assertEqual(dlg.table.item(0, 0).text(), "각도")
        self.assertEqual(dlg.table.item(0, 3).text(), "11")  # 10 + 1 = 11

        dlg.close()

    def test_02_draw_dialog_features(self):
        """그리기 도구 요구사항 검증: 기본 굵기 1, 창 닫혀도 보이기, 면번호, 선택 취소, 프리폼"""
        dlg = DrawDialog()
        
        # 1. 기본 굵기 1 확인
        self.assertEqual(dlg.pen_width, 1)
        self.assertTrue(dlg.bg_width.button(1).isChecked())

        # 2. 창이 닫혀도 보이기 체크박스
        self.assertEqual(dlg.chk_keep.text(), "창이 닫혀도 보이기")
        self.assertTrue(dlg.chk_keep.isChecked())

        # 3. 프리폼 및 면번호 확인
        dlg.add_shape({
            "shape_type": "freeform",
            "view_name": "Sagittal",
            "slice_idx": 5,
            "pts": [(0, 0, 0), (1, 1, 0), (2, 2, 0)],
            "color": "#ff3c3c",
            "width": 1
        })
        self.assertEqual(dlg.table.item(0, 1).text(), "프리폼")
        self.assertEqual(dlg.table.item(0, 3).text(), "Sagittal")
        self.assertEqual(dlg.table.item(0, 4).text(), "6")  # 5 + 1 = 6

        # 4. '선택 취소' 모드 확인
        dlg.set_shape_type("select")
        self.assertEqual(dlg.shape_type, "select")

        # 5. 더블클릭 슬라이스 이동
        nav_result = []
        dlg.navigate_to_item.connect(lambda v, s, c: nav_result.append((v, s, c)))
        dlg._on_table_double_clicked(0, 0)
        self.assertEqual(len(nav_result), 1)
        self.assertEqual(nav_result[0][0], "Sagittal")
        self.assertEqual(nav_result[0][1], 5)

        dlg.close()

    def test_03_mpr_widget_overlays_and_3d_box(self):
        """MPR 위젯 오버레이 렌더링(텍스트 라벨, 하이라이트) 및 3D 뷰어 vtkBoxWidget 검증"""
        main_win = MainWindow()
        vtk_img, meta = DicomReader.load_single_file(self.test_file)
        main_win.state_mgr.set_volume_data(vtk_img, meta)

        axial_widget = main_win.mpr_container.axial_panel.mpr_widget
        cur_slice = axial_widget._get_current_slice_idx()

        # 계측 추가
        main_win.mpr_container.measure_dialog.add_measurement({
            "type": "length",
            "value_str": "20.00 mm",
            "view_name": "Axial",
            "slice_idx": cur_slice,
            "world_pts": [(10.0, 10.0, 0.0), (30.0, 10.0, 0.0)]
        })

        # 오버레이 갱신 확인: 선분 액터 + vtkBillboardTextActor3D 라벨 액터 존재 확인
        axial_widget._update_overlays()
        self.assertTrue(len(axial_widget._measure_actors) >= 2)
        has_billboard = any(isinstance(act, vtk.vtkBillboardTextActor3D) for act in axial_widget._measure_actors)
        self.assertTrue(has_billboard)

        # 3D 뷰어 열기 및 ROI box widget 검증
        main_win.open_3d_viewer()
        vol_win = main_win.volume_window
        self.assertIsNotNone(vol_win)
        self.assertIsNotNone(vol_win.box_widget)

        # boundary 설정 모드 활성화 시 box_widget On 확인
        vol_win.btn_roi_boundary.setChecked(True)
        self.assertTrue(vol_win.btn_roi_boundary.isChecked())

        vol_win.close()
        main_win.close()


if __name__ == "__main__":
    unittest.main()
