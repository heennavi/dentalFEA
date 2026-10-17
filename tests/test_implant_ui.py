"""
test_implant_ui.py
Step 4: 임플란트 배치 UI, 2D 단면 윤곽선(Cutter), 3D 표시에 대한 통합 단위 테스트 (pytest).
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 로컬 libxcb-cursor.so.0 로드 (Qt 6.5+ xcb 의존성)
import ctypes
local_xcb_cursor = os.path.join(project_root, ".local_libs", "usr", "lib", "x86_64-linux-gnu", "libxcb-cursor.so.0")
if os.path.exists(local_xcb_cursor):
    try:
        ctypes.CDLL(local_xcb_cursor)
    except Exception:
        pass

from PySide6.QtWidgets import QApplication

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.placement.implant_manager import ImplantModel
from dental_fea_app.ui.placement.implant_panel import ImplantPanel
from dental_fea_app.ui.control_panel import ControlPanel
from dental_fea_app.ui.mpr_widget import MPRWidget
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow
from dental_fea_app.config import AXIS_AXIAL, AXIS_CORONAL, AXIS_SAGITTAL
from tests.test_implant_manager import create_cylinder_stl_file


app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


def create_test_vtk_volume(shape=(30, 30, 30), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """테스트용 vtkImageData 생성"""
    nz, ny, nx = shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(spacing[0], spacing[1], spacing[2])
    img.SetOrigin(origin[0], origin[1], origin[2])

    data = np.full(shape, 500.0, dtype=np.float32)
    vtk_arr = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(vtk_arr)
    return img


class TestImplantUI(unittest.TestCase):

    def setUp(self):
        self.state_mgr = StateManager()
        self.vtk_vol = create_test_vtk_volume()
        self.state_mgr.set_volume_data(self.vtk_vol, {"origin": (0, 0, 0), "spacing": (1, 1, 1)})

    def test_01_control_panel_has_implant_tab(self):
        """좌측 ControlPanel에 '3. 임플란트' 탭과 ImplantPanel이 정상 등록되어 있는지 확인"""
        ctrl = ControlPanel(self.state_mgr, on_open_3d_viewer=lambda: None)
        self.assertTrue(hasattr(ctrl, "implant_panel"))
        self.assertIsInstance(ctrl.implant_panel, ImplantPanel)

        # 탭 개수 및 이름 확인
        tab_widget = ctrl.workflow_tabs
        tab_titles = [tab_widget.tabText(i) for i in range(tab_widget.count())]
        self.assertIn("3. 임플란트", tab_titles)

    def test_02_implant_panel_ui_controls_and_crosshair_placement(self):
        """ImplantPanel에서 STL 로드 및 crosshair 배치 버튼 동작 검증"""
        panel = ImplantPanel(self.state_mgr)

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "test_fixture.stl")
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)

            # 1. 모델 직접 로드 및 등록
            model = ImplantModel(stl_file)
            self.state_mgr.set_implant_model(model)

            self.assertIsNotNone(panel.implant_model)
            self.assertIn("test_fixture.stl", panel.lbl_stl_name.text())
            self.assertIn("10.0 mm", panel.lbl_stl_dim.text())

            # 2. 십자선 좌표 설정 후 배치 버튼 클릭
            self.state_mgr.set_coordinates(15.0, 12.0, 8.0)
            panel.btn_place_crosshair.click()

            # Apex 위치가 (15, 12, 8)로 정렬되었는지 확인
            apex = model.get_world_apex()
            np.testing.assert_allclose(apex, [15.0, 12.0, 8.0], atol=1e-4)
            self.assertAlmostEqual(panel.spin_x.value(), model.position[0], places=2)
            self.assertAlmostEqual(panel.spin_y.value(), model.position[1], places=2)
            self.assertAlmostEqual(panel.spin_z.value(), model.position[2], places=2)

    def test_03_volume_viewer_3d_implant_display(self):
        """VolumeViewerWindow가 임플란트 3D 메쉬를 액터로 표시하고 포즈 변경 시 갱신하는지 검증"""
        viewer = VolumeViewerWindow(self.state_mgr)
        self.assertTrue(hasattr(viewer, "implant_actor"))
        self.assertFalse(viewer.implant_actor.GetVisibility())

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "test_fixture.stl")
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)
            model = ImplantModel(stl_file)

            # 모델 등록 -> 3D 액터 가시화
            self.state_mgr.set_implant_model(model)
            self.assertTrue(viewer.implant_actor.GetVisibility())

            # 포즈 이동
            model.translate(10.0, 0.0, 0.0)
            bounds = viewer.implant_actor.GetBounds()
            # xmin, xmax가 이동되었는지 확인
            self.assertGreater(bounds[0], 0.0)

            # 모델 해제 -> 액터 비가시화
            self.state_mgr.set_implant_model(None)
            self.assertFalse(viewer.implant_actor.GetVisibility())

        viewer.close()

    def test_04_mpr_widget_2d_cutter_contour(self):
        """2D MPR 뷰에서 vtkCutter를 통해 슬라이스와 만나는 단면 윤곽선이 생성/갱신되는지 검증"""
        axial_widget = MPRWidget(AXIS_AXIAL, self.state_mgr)
        self.assertTrue(hasattr(axial_widget, "implant_contour_actor"))
        self.assertFalse(axial_widget.implant_contour_actor.GetVisibility())

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "test_fixture.stl")
            # z: -5 ~ +5 범위의 10mm 원기둥
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)
            model = ImplantModel(stl_file)

            # z=15 지점에 임플란트 Apex 배치 -> z 범위: [15, 25]
            model.place_at(point_world=[15.0, 15.0, 15.0], axis_world=[0.0, 0.0, 1.0])
            self.state_mgr.set_implant_model(model)

            # 1. 슬라이스가 임플란트 범위 밖(z=5)에 있을 때: 윤곽선 비가시
            self.state_mgr.set_coordinates(15.0, 15.0, 5.0)
            axial_widget.update_slice()
            self.assertFalse(axial_widget.implant_contour_actor.GetVisibility())

            # 2. 슬라이스가 임플란트 내부(z=18)에 있을 때: 윤곽선 가시화 (원형 단면)
            self.state_mgr.set_coordinates(15.0, 15.0, 18.0)
            axial_widget.update_slice()
            self.assertTrue(axial_widget.implant_contour_actor.GetVisibility())

            # 커터 출력 정점 개수 확인 (> 0)
            cut_pts = axial_widget.implant_cutter.GetOutput().GetNumberOfPoints()
            self.assertGreater(cut_pts, 0)

            # 3. 임플란트를 다른 위치로 이동했을 때: 슬라이스 z=18에서 벗어나면 다시 비가시
            model.translate(0.0, 0.0, 50.0)  # z: [65, 75]
            axial_widget.update_slice()
            self.assertFalse(axial_widget.implant_contour_actor.GetVisibility())


if __name__ == "__main__":
    unittest.main()
