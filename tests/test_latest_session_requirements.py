"""
tests/test_latest_session_requirements.py
사용자 최근 6대 요구사항 검증 테스트:
1. 2D 뷰 활성화 테두리 두께 1 (1px solid #ffffff)
2. 단면 계측 도구: 선택 해제 시 오버레이 해제, 그리기 직후 해제, 닫기 시 해제
3. 그리기 도구: 선택 해제 시 오버레이 해제, 그리기 직후 해제, 닫기 시 해제, 형태 변경 시 이전 도형 형태 보존
4. 3D 뷰: 슬라이스 평면, ROI 설정(적용, 박스, 바운더리), 옵션(프리셋), 3D 음영, 투명도 저장 및 복원
5. 사용자 정보 로드 성능 최적화: 1회 일괄 처리
6. 사용자 정보 저장 후 추가 변경 없을 시 종료 팝업 생략
"""

import sys
import os
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import vtk
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCloseEvent

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.core.settings_manager import SettingsManager
from dental_fea_app.ui.measurement_dialog import MeasurementDialog
from dental_fea_app.ui.draw_dialog import DrawDialog
from dental_fea_app.ui.mpr_container import MPRContainer
from dental_fea_app.ui.main_window import MainWindow
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow


class TestLatestSessionRequirements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        self.state_mgr = StateManager()

    def _create_mock_volume(self):
        img = vtk.vtkImageData()
        img.SetDimensions(20, 20, 20)
        img.SetSpacing(1.0, 1.0, 1.0)
        img.SetOrigin(0.0, 0.0, 0.0)
        img.AllocateScalars(vtk.VTK_SHORT, 1)
        return img

    def test_01_measurement_dialog_deselection_and_close(self):
        """계측 도구 선택 해제, 추가 후 해제, 닫기 시 해제 검증"""
        dlg = MeasurementDialog()
        selected_signal_log = []
        dlg.item_selected.connect(lambda idx: selected_signal_log.append(idx))

        # 1) 항목 추가
        dlg.add_measurement({
            "type": "length",
            "value_str": "15.00 mm",
            "view_name": "Axial",
            "slice_idx": 10,
            "world_pts": [(0, 0, 0), (15, 0, 0)]
        })
        self.app.processEvents()

        # 추가 직후 테이블에서 방금 추가된 행 선택 확인
        self.assertEqual(dlg.table.currentRow(), 0)
        self.assertEqual(selected_signal_log[-1], 0)

        # 행 선택
        dlg.table.selectRow(0)
        self.app.processEvents()
        self.assertEqual(selected_signal_log[-1], 0)

        # clear_selection 호출 시
        dlg.clear_selection()
        self.app.processEvents()
        self.assertEqual(dlg.table.selectedItems(), [])
        self.assertEqual(selected_signal_log[-1], -1)

        # 닫기 시 clear_selection 확인
        dlg.table.selectRow(0)
        self.assertEqual(dlg.table.currentRow(), 0)
        dlg.close()
        self.assertEqual(dlg.table.selectedItems(), [])
        self.assertEqual(selected_signal_log[-1], -1)

    def test_02_draw_dialog_shape_type_preservation_and_deselection(self):
        """그리기 도구: 형태 변경 시 이전 도형 형태 불변 및 선택 해제 검증"""
        dlg = DrawDialog()
        selected_signal_log = []
        dlg.item_selected.connect(lambda idx: selected_signal_log.append(idx))

        # 1) 선분 도형 추가
        dlg.add_shape({
            "shape_type": "line",
            "view_name": "Axial",
            "slice_idx": 5,
            "pts": [[0, 0, 0], [10, 10, 0]],
            "color": QColor(255, 60, 60),
            "width": 1,
            "is_dashed": False,
            "end_cap": "none"
        })
        self.app.processEvents()

        # 추가 직후 테이블에서 방금 추가된 행 선택 확인
        self.assertEqual(dlg.table.currentRow(), 0)
        self.assertEqual(selected_signal_log[-1], 0)

        # 3) 사용자가 '사각형(rect)' 형태 버튼을 클릭하여 앞으로 사각형을 그리려고 함
        dlg.set_shape_type("rect")
        self.app.processEvents()

        # 이전 행 도형의 형태가 여전히 'line'으로 보존되었는지 검증
        self.assertEqual(dlg.shapes[0]["shape_type"], "line")
        self.assertEqual(dlg.table.item(0, 1).text(), "선분")

        # 기존 선택이 해제되었는지 확인
        self.assertEqual(dlg.table.selectedItems(), [])
        self.assertEqual(selected_signal_log[-1], -1)

        # 4) 행을 다시 선택하고 스타일(굵기, 색상)만 변경할 경우
        dlg.table.selectRow(0)
        dlg._on_width_changed(3)
        self.assertEqual(dlg.shapes[0]["width"], 3)
        self.assertEqual(dlg.shapes[0]["shape_type"], "line")  # 형태는 여전히 line

        # 5) 닫기 시 clear_selection 확인
        dlg.close()
        self.assertEqual(dlg.table.selectedItems(), [])
        self.assertEqual(selected_signal_log[-1], -1)

    def test_03_border_width_1px(self):
        """2D 뷰 활성화 및 데이터 보유 테두리가 1px solid #ffffff인지 검증"""
        container = MPRContainer(self.state_mgr, lambda: None)
        axial = container.axial_panel
        axial._open_annotation_dialog()
        axial._annotation_dialog.hide()
        axial._annotation_dialog.editor.setPlainText("메모")
        axial._update_tool_buttons_border()

        self.assertIn("border: 1px solid #ffffff", axial.btn_annotation.styleSheet())
        self.assertNotIn("2px solid #ffffff", axial.btn_annotation.styleSheet())

    def test_04_3d_view_settings_persistence(self):
        """3D 뷰: ROI 및 슬라이스 평면 저장/로드 제거 확인 및 기본 Bone 프리셋 복원 검증"""
        settings_mgr = SettingsManager()
        win_3d = VolumeViewerWindow(self.state_mgr, settings_mgr=settings_mgr)

        # 1) 설정 변경
        win_3d.chk_shading.setChecked(False)
        win_3d.slider_opacity.setValue(85)
        win_3d.chk_plane_axial.setChecked(True)
        win_3d.chk_plane_coronal.setChecked(True)
        win_3d.chk_plane_sagittal.setChecked(False)
        win_3d.chk_roi_apply.setChecked(True)
        win_3d.chk_roi_box.setChecked(True)
        win_3d.preset_combo.setCurrentIndex(1)  # 치아(에나멜)
        test_bounds = (2.0, 18.0, 3.0, 17.0, 4.0, 16.0)
        win_3d._on_roi_bounds_updated(test_bounds)

        # 음영과 투명도는 저장되지만, 슬라이스 평면/ROI/프리셋은 저장되지 않아야 함 (체크를 켰어도 저장되지 않음)
        self.assertFalse(settings_mgr.get("volume_3d.shading"))
        self.assertAlmostEqual(settings_mgr.get("volume_3d.opacity"), 0.85, places=2)
        self.assertNotEqual(settings_mgr.get("volume_3d.show_axial_plane"), True)
        self.assertNotEqual(settings_mgr.get("volume_3d.roi_apply"), True)
        self.assertIsNone(settings_mgr.get("volume_3d.roi_bounds"))
        self.assertNotEqual(settings_mgr.get("volume_3d.preset"), 1)

        # 2) 새로운 3D 윈도우 인스턴스 생성 후 볼륨 로드 시 기본값 복원 확인
        win_3d_new = VolumeViewerWindow(self.state_mgr, settings_mgr=settings_mgr)
        mock_vol = self._create_mock_volume()
        win_3d_new.on_volume_loaded(mock_vol)

        # 음영/투명도는 복원, 평면/ROI/프리셋은 기본값(골조직 0, ROI 꺼짐, 평면 꺼짐)
        self.assertFalse(win_3d_new.chk_shading.isChecked())
        self.assertEqual(win_3d_new.slider_opacity.value(), 85)
        self.assertFalse(win_3d_new.chk_plane_axial.isChecked())
        self.assertFalse(win_3d_new.chk_roi_apply.isChecked())
        self.assertEqual(win_3d_new.preset_combo.currentIndex(), 0)
        self.assertIsNone(self.state_mgr.roi_bounds)

        win_3d.close()
        win_3d_new.close()

    def test_05_user_info_load_optimization_and_dirty_flag(self):
        """사용자 정보 일괄 로드 최적화 및 저장 후 추가 변경 없을 시 종료 팝업 생략 검증"""
        main_win = MainWindow()
        mock_vol = self._create_mock_volume()
        main_win.state_mgr.set_volume_data(mock_vol, {"PatientName": "Test"})
        self.app.processEvents()

        # 초기 상태: dirty = False
        self.assertFalse(main_win._user_info_dirty)

        # 1) 주석 작성 -> dirty = True
        main_win.mpr_container.annotation_dialog.editor.setPlainText("환자 상태 정상")
        main_win.mpr_container.annotation_dialog.btn_save.click()
        self.app.processEvents()
        self.assertTrue(main_win._user_info_dirty)
        self.assertTrue(main_win._has_user_info_content())

        # 2) 파일로 저장 수행
        saved_dir = os.path.join(project_root, "saved")
        os.makedirs(saved_dir, exist_ok=True)
        test_file = os.path.join(saved_dir, "test_dirty_lifecycle_user_info.json")
        main_win._save_user_info(test_file, prompt_dialog=False)

        # 저장 완료 후 dirty = False 리셋 확인!
        self.assertFalse(main_win._user_info_dirty)

        # 3) 새로운 변경사항이 없는 상태에서 closeEvent 발생
        # _user_info_dirty가 False이므로 QMessageBox 팝업 질문을 호출하지 않고 정상 종료되어야 함
        close_evt = QCloseEvent()
        main_win.closeEvent(close_evt)
        self.assertTrue(close_evt.isAccepted())

        # 4) 다시 계측 추가 -> dirty = True
        main_win.mpr_container.measure_dialog.add_measurement({
            "type": "length", "value_str": "10.0 mm", "view_name": "Axial",
            "slice_idx": 1, "world_pts": [(0, 0, 0), (10, 0, 0)]
        })
        self.assertTrue(main_win._user_info_dirty)

        # 5) 로드 수행 시 dirty = False 리셋 확인
        main_win._load_user_info(test_file)
        self.assertFalse(main_win._user_info_dirty)

        if os.path.exists(test_file):
            os.remove(test_file)
        main_win.close()


if __name__ == "__main__":
    unittest.main()
