"""
test_new_enhancements.py
- 세그폴트 방지: ImplantSimulationDialog 연속 열기/닫기 무크래시 검증
- CT 미로드 시 비활성화 및 로드 시 활성화 검증
- User Info 임플란트 및 보철(Abutment, Crown) 직렬화/역직렬화 검증
- ImplantTransformDialog 실시간 변환 및 초기값 복원/취소/적용 검증
- 단면 계측 및 그리기 도구: 완료 시 하이라이트 해제, 8개 컬럼, 더블클릭 동기화, 파란색 테두리 검증
"""
import os
import sys
import unittest
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ctypes
local_xcb_cursor = os.path.join(project_root, ".local_libs", "usr", "lib", "x86_64-linux-gnu", "libxcb-cursor.so.0")
if os.path.exists(local_xcb_cursor):
    try:
        ctypes.CDLL(local_xcb_cursor)
    except Exception:
        pass

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.ui.main_window import MainWindow
from dental_fea_app.ui.control_panel import ControlPanel
from dental_fea_app.ui.mpr_container import MPRContainer
from dental_fea_app.ui.placement.implant_transform_dialog import ImplantTransformDialog
from dental_fea_app.placement.implant_generator import (
    ImplantSpec, AbutmentSpec, ProstheticSpec
)

app = QApplication.instance() or QApplication(sys.argv)


def create_test_volume(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    nz, ny, nx = shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*spacing)
    img.SetOrigin(*origin)
    data = np.full(shape, 200.0, dtype=np.float32)
    vtk_arr = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(vtk_arr)
    return img


class TestNewEnhancements(unittest.TestCase):

    def setUp(self):
        self.state_mgr = StateManager()

    def test_01_sim_dialog_open_close_multiple_times_no_segfault(self):
        """1. 임플란트 및 보철 창 10회 연속 열기/닫기 시 세그폴트 없이 안전한지 검증"""
        win = MainWindow()
        win.show()
        app.processEvents()

        implant_panel = win.control_panel.implant_panel

        for i in range(10):
            implant_panel._open_simulation_dialog()
            app.processEvents()
            self.assertTrue(implant_panel._sim_dialog.isVisible())

            implant_panel._sim_dialog.close()
            app.processEvents()
            self.assertFalse(implant_panel._sim_dialog.isVisible())

        win.close()
        app.processEvents()

    def test_02_disabled_tabs_and_icons_when_no_volume(self):
        """2. CT 미로드 시 탭(분할, 임플란트, 해석) 및 2D 툴바 버튼 비활성화, 로드 시 활성화 검증"""
        cp = ControlPanel(self.state_mgr, lambda: None)
        container = MPRContainer(self.state_mgr, lambda: None)
        app.processEvents()

        # 미로드 상태: 탭 1, 2, 3 비활성화 확인
        self.assertFalse(cp.workflow_tabs.isTabEnabled(1)) # 2. 분할
        self.assertFalse(cp.workflow_tabs.isTabEnabled(2)) # 3. 임플란트
        self.assertFalse(cp.workflow_tabs.isTabEnabled(3)) # 4. 해석

        # 2D 뷰 툴바 버튼 비활성화 확인
        for p in container.panels:
            self.assertFalse(p.btn_measure.isEnabled())
            self.assertFalse(p.btn_draw.isEnabled())
            self.assertFalse(p.btn_annotation.isEnabled())
            self.assertFalse(p.slider.isEnabled())

        # 볼륨 로드 시뮬레이션
        vol = create_test_volume()
        self.state_mgr.set_volume_data(vol, {})
        app.processEvents()

        # 로드 후: 탭 1, 2, 3 활성화 확인
        self.assertTrue(cp.workflow_tabs.isTabEnabled(1))
        self.assertTrue(cp.workflow_tabs.isTabEnabled(2))
        self.assertTrue(cp.workflow_tabs.isTabEnabled(3))

        # 2D 뷰 툴바 버튼 활성화 확인
        for p in container.panels:
            self.assertTrue(p.btn_measure.isEnabled())
            self.assertTrue(p.btn_draw.isEnabled())
            self.assertTrue(p.btn_annotation.isEnabled())
            self.assertTrue(p.slider.isEnabled())

    def test_03_user_info_implants_serialization(self):
        """3. User Info 연동: 임플란트, Abutment, Crown 데이터의 JSON to_dict / from_dict 복원 검증"""
        mgr = self.state_mgr.implant_placement_mgr

        spec = ImplantSpec(
            id="test_imp_1",
            name="MegaGen AnyRidge Ø4.0x10.0",
            diameter_mm=4.0,
            length_mm=10.0,
            taper_deg=2.0,
            top_type="collar",
            thread_type="triangular",
            micro_threads=True
        )
        imp_id = mgr.place_implant(spec, initial_position=(10.0, 20.0, 30.0))
        imp = mgr.get_implant(imp_id)
        imp.orientation = np.array([15.0, -10.0, 5.0])
        mgr._update_implant_transform(imp)

        # Abutment 장착
        ab_spec = AbutmentSpec(diameter_mm=4.0, flare_deg=35.0, collar_len_mm=4.5, upper_len_mm=6.0, color_type="gold")
        mgr.attach_abutment(imp.id, ab_spec)

        # Prosthetic 장착
        pr_spec = ProstheticSpec(tooth_number=16, crown_len_mm=8.0, crown_width_mm=10.0, crown_depth_mm=11.0, material="zirconia")
        mgr.attach_prosthetic(imp.id, pr_spec)

        # 직렬화
        data = mgr.to_dict()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], imp.id)
        self.assertEqual(data[0]["position"], [float(p) for p in imp.position])
        self.assertEqual(data[0]["orientation"], [15.0, -10.0, 5.0])
        self.assertIsNotNone(data[0]["abutment"])
        self.assertEqual(data[0]["abutment"]["flare_deg"], 35.0)
        self.assertIsNotNone(data[0]["prosthetic"])
        self.assertEqual(data[0]["prosthetic"]["tooth_number"], 16)

        # 전체 삭제 후 복원
        mgr.from_dict(data)
        self.assertEqual(len(mgr.implants), 1)
        restored = mgr.get_implant(imp.id)
        self.assertIsNotNone(restored)
        np.testing.assert_allclose(restored.position, imp.position)
        np.testing.assert_allclose(restored.orientation, [15.0, -10.0, 5.0])
        self.assertIsNotNone(restored.abutment)
        self.assertEqual(restored.abutment.flare_deg, 35.0)
        self.assertIsNotNone(restored.prosthetic)
        self.assertEqual(restored.prosthetic.tooth_number, 16)

    def test_04_implant_transform_dialog_controls(self):
        """4. ImplantTransformDialog 실시간 변경, 초기값 복원, 취소, 적용 동작 검증"""
        mgr = self.state_mgr.implant_placement_mgr
        spec = ImplantSpec(
            id="test_imp_2", name="Dentium SuperLine Ø3.5x8.5",
            diameter_mm=3.5, length_mm=8.5, taper_deg=0.0,
            top_type="flat", thread_type="square", micro_threads=False
        )
        imp_id = mgr.place_implant(spec, initial_position=(5.0, 5.0, 5.0))
        imp = mgr.get_implant(imp_id)
        imp.orientation = np.array([0.0, 0.0, 0.0])

        dlg = ImplantTransformDialog(imp, self.state_mgr)

        # 스핀박스 값 변경
        dlg.spin_x.setValue(12.5)
        dlg.spin_yaw.setValue(45.0)
        app.processEvents()

        self.assertAlmostEqual(imp.position[0], 12.5)
        self.assertAlmostEqual(imp.orientation[0], 45.0)

        # 초기값 복원 버튼 클릭
        dlg.btn_reset.click()
        app.processEvents()
        self.assertAlmostEqual(imp.position[0], 5.0)
        self.assertAlmostEqual(imp.orientation[0], 0.0)

        # 값 다시 변경 후 취소 클릭
        dlg.spin_x.setValue(20.0)
        dlg.btn_cancel.click()
        app.processEvents()
        self.assertAlmostEqual(imp.position[0], 5.0) # 원래 값 복원 확인

    def test_05_measurement_and_drawing_enhancements(self):
        """5. 단면 계측 및 그리기 도구 완료 시 선택 해제, 테이블 8컬럼, 파란 테두리 검증"""
        container = MPRContainer(self.state_mgr, lambda: None)
        app.processEvents()

        # 1) 계측 추가
        container.measure_dialog.add_measurement({
            "type": "length",
            "value_str": "15.0 mm",
            "view_name": "Axial",
            "slice_idx": 5,
            "world_pts": [[0, 0, 0], [15, 0, 0]]
        })
        app.processEvents()

        # 측정 완료 직후 테이블 행 선택 확인
        self.assertEqual(container.measure_dialog.table.currentRow(), 0)

        # 계측 항목이 있으므로 btn_measure에 1px 흰색 테두리 적용 확인
        for p in container.panels:
            self.assertIn("1px solid #ffffff", p.btn_measure.styleSheet())

        # 2) 그리기 추가
        container.draw_dialog.add_shape({
            "shape_type": "arrow",
            "color": "#ff0000",
            "view_name": "Coronal",
            "slice_idx": 8,
            "pts": [[0, 0, 0], [10, 10, 0]],
            "width": 2,
            "is_dashed": True,
            "end_cap": "single_arrow"
        })
        app.processEvents()

        # 그리기 완료 직후 테이블 행 선택 확인
        self.assertEqual(container.draw_dialog.table.currentRow(), 0)

        # 컬럼 8개 확인
        self.assertEqual(container.draw_dialog.table.columnCount(), 8)
        self.assertEqual(container.draw_dialog.table.item(0, 5).text(), "2") # 선굵기
        self.assertEqual(container.draw_dialog.table.item(0, 6).text(), "점선") # 선형태
        self.assertEqual(container.draw_dialog.table.item(0, 7).text(), "한쪽 화살표") # 선분끝 형태

        # 그리기 항목이 있으므로 btn_draw에 1px 흰색 테두리 적용 확인
        for p in container.panels:
            self.assertIn("1px solid #ffffff", p.btn_draw.styleSheet())


if __name__ == "__main__":
    unittest.main()
