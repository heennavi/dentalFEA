"""
test_segmentation_ui.py
Step 2 분할 UI 및 2D MPR / 3D 오버레이 통합 단위 테스트.
- ControlPanel의 SegmentationPanel 인스턴스 존재 여부
- 분할 파라미터 제어 및 QThread 워커 비동기 실행 및 시그널 방출
- StateManager의 segmentation_result 및 combined_label_vtk 등록
- 2D MPRWidget의 마스크 오버레이 슬라이스(mask_image_slice) 및 LUT 알파 갱신
- 3D VolumeViewerWindow의 피질골/수질골 표면(FlyingEdges3D) 액터 생성 및 가시성 제어
"""

import os
import sys
import unittest
import numpy as np
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
from dental_fea_app.ui.control_panel import ControlPanel
from dental_fea_app.ui.segmentation.segmentation_panel import SegmentationPanel
from dental_fea_app.ui.segmentation.segmentation_worker import SegmentationWorker
from dental_fea_app.ui.mpr_widget import MPRWidget
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow
from dental_fea_app.segmentation.cortical_cancellous import (
    SegmentationResult,
    run_segmentation,
    mask_to_vtk_image
)
from dental_fea_app.config import AXIS_AXIAL


class TestSegmentationUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

        # 테스트용 합성 볼륨 생성 (충분한 크기)
        shape = (30, 30, 30)
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)

        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        cz, cy, cx = 15, 15, 15
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

        hu = np.full(shape, -1000.0, dtype=np.float32)
        hu[r <= 10.0] = 400.0   # 수질골 영역
        hu[(r > 7.0) & (r <= 10.0)] = 1200.0  # 피질골 외곽 영역

        cls.vtk_img = vtk.vtkImageData()
        cls.vtk_img.SetDimensions(shape[2], shape[1], shape[0])
        cls.vtk_img.SetSpacing(*spacing)
        cls.vtk_img.SetOrigin(*origin)

        flat_arr = hu.transpose(2, 1, 0).ravel()
        vtk_arr = numpy_support.numpy_to_vtk(flat_arr, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName("Scalars_")
        cls.vtk_img.GetPointData().SetScalars(vtk_arr)

    def test_01_control_panel_has_segmentation_panel(self):
        """ControlPanel 내부 탭 2에 SegmentationPanel이 올바르게 탑재되었는지 확인"""
        state_mgr = StateManager()
        panel = ControlPanel(state_manager=state_mgr, on_open_3d_viewer=lambda: None)

        self.assertTrue(hasattr(panel, "segmentation_panel"))
        self.assertIsInstance(panel.segmentation_panel, SegmentationPanel)
        self.assertIs(panel.segmentation_panel.state_mgr, state_mgr)

    def test_02_segmentation_worker_execution(self):
        """SegmentationWorker가 백그라운드 스레드에서 분할을 수행하고 finished_work 시그널을 발행하는지 검증"""
        params = {
            "bone_threshold": 200.0,
            "method": "distance",
            "cortical_thickness_mm": 1.5,
            "cortical_hu": 700.0,
            "keep_largest": True,
            "fill_holes": True,
        }

        worker = SegmentationWorker(self.vtk_img, params)

        received_results = []
        progress_values = []

        worker.progress.connect(lambda p, m: progress_values.append(p))
        worker.finished_work.connect(lambda res: received_results.append(res))

        worker.run()

        self.assertEqual(len(received_results), 1)
        res = received_results[0]
        self.assertIsInstance(res, SegmentationResult)
        self.assertEqual(res.bone_mask.shape, (30, 30, 30))
        self.assertGreater(res.cortical_mask.sum(), 0)
        self.assertGreater(res.cancellous_mask.sum(), 0)
        self.assertIn(100, progress_values)

    def test_03_state_manager_segmentation_result_handling(self):
        """StateManager가 SegmentationResult를 등록하고 combined_label_vtk를 생성하는지 검증"""
        state_mgr = StateManager()
        state_mgr.set_volume_data(self.vtk_img, {"origin": (0, 0, 0), "spacing": (1, 1, 1)})

        params = {"bone_threshold": 200.0, "method": "distance", "cortical_thickness_mm": 1.5}
        result = run_segmentation(self.vtk_img, params=params)

        sig_emitted = []
        state_mgr.segmentation_result_changed.connect(lambda r: sig_emitted.append(r))

        state_mgr.set_segmentation_result(result)

        self.assertEqual(len(sig_emitted), 1)
        self.assertIs(state_mgr.segmentation_result, result)
        self.assertIsNotNone(state_mgr.combined_label_vtk)

        scalar_range = state_mgr.combined_label_vtk.GetScalarRange()
        self.assertGreaterEqual(scalar_range[0], 0)
        self.assertLessEqual(scalar_range[1], 2)

    def test_04_mpr_widget_mask_overlay(self):
        """MPRWidget이 분할 결과를 수신하여 2D 마스크 오버레이 슬라이스를 정상 표시하는지 검증"""
        state_mgr = StateManager()
        state_mgr.set_volume_data(self.vtk_img, {"origin": (0, 0, 0), "spacing": (1, 1, 1)})

        mpr = MPRWidget(AXIS_AXIAL, state_mgr)
        self.assertTrue(hasattr(mpr, "mask_image_slice"))
        self.assertTrue(hasattr(mpr, "mask_lut"))

        # 분할 전에는 마스크가 비가시 상태
        self.assertFalse(mpr.mask_image_slice.GetVisibility())

        # 분할 결과 등록
        params = {"bone_threshold": 200.0, "method": "distance"}
        result = run_segmentation(self.vtk_img, params=params)
        state_mgr.set_segmentation_result(result)

        # 결과 등록 후 마스크가 가시 상태로 전환
        self.assertTrue(mpr.mask_image_slice.GetVisibility())

        # 가시성 토글 테스트
        state_mgr.set_segmentation_visibility({
            "show_cortical": False,
            "show_cancellous": False,
            "opacity": 0.5
        })
        self.assertFalse(mpr.mask_image_slice.GetVisibility())

        # 피질골만 활성화
        state_mgr.set_segmentation_visibility({
            "show_cortical": True,
            "show_cancellous": False,
            "opacity": 0.5
        })
        self.assertTrue(mpr.mask_image_slice.GetVisibility())
        val = [0.0, 0.0, 0.0, 0.0]
        mpr.mask_lut.GetTableValue(1, val)
        self.assertAlmostEqual(val[3], 0.5, delta=0.01)

    def test_05_volume_viewer_3d_segmentation_surfaces(self):
        """VolumeViewerWindow가 피질골/수질골 3D 표면 액터를 생성하고 가시성을 제어하는지 검증"""
        state_mgr = StateManager()
        state_mgr.set_volume_data(self.vtk_img, {"origin": (0, 0, 0), "spacing": (1, 1, 1)})

        viewer = VolumeViewerWindow(state_mgr)
        self.assertTrue(hasattr(viewer, "cortical_actor"))
        self.assertTrue(hasattr(viewer, "cancellous_actor"))

        # 분할 전에는 비가시
        self.assertFalse(viewer.cortical_actor.GetVisibility())
        self.assertFalse(viewer.cancellous_actor.GetVisibility())

        # 분할 결과 등록
        params = {"bone_threshold": 200.0, "method": "distance"}
        result = run_segmentation(self.vtk_img, params=params)
        state_mgr.set_segmentation_result(result)

        # 등록 후 3D 서피스 액터 활성화
        self.assertTrue(viewer.cortical_actor.GetVisibility())
        self.assertTrue(viewer.cancellous_actor.GetVisibility())

        # 불투명도 및 가시성 제어
        state_mgr.set_segmentation_visibility({
            "show_cortical": True,
            "show_cancellous": False,
            "opacity": 0.8
        })
        self.assertTrue(viewer.cortical_actor.GetVisibility())
        self.assertFalse(viewer.cancellous_actor.GetVisibility())
        self.assertAlmostEqual(viewer.cortical_actor.GetProperty().GetOpacity(), 0.8)

        viewer.close()


if __name__ == "__main__":
    unittest.main()
