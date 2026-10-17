"""
test_phase1.py
Phase 1 (DICOM 로더, 2D MPR 3분할 뷰어, 3D 볼륨 뷰어) 통합 단위 테스트.
"""

import os
import sys
import unittest
import numpy as np
import vtk

# 프로젝트 루트 경로 추가
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
from dental_fea_app.imaging.dicom_reader import DicomReader
from dental_fea_app.imaging.volume_data import itk_to_vtk_image
from dental_fea_app.ui.mpr_widget import MPRWidget
from dental_fea_app.ui.mpr_container import MPRContainer
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow
from dental_fea_app.ui.main_window import MainWindow
from tests.create_dummy_data import generate_synthetic_dental_ct


class TestPhase1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # QApplication 싱글톤 생성
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

        # 테스트용 합성 치과 CT 파일 생성
        cls.test_file = os.path.join(project_root, "tests", "sample_data", "test_mandible.mha")
        generate_synthetic_dental_ct(cls.test_file, size=(60, 60, 40), spacing=(1.0, 1.0, 1.0))

    def test_01_dicom_reader_loading(self):
        """DICOM / MHA 파일 로드 및 VTK ImageData 변환 검증"""
        vtk_img, meta = DicomReader.load_single_file(self.test_file)
        self.assertIsNotNone(vtk_img)
        self.assertIsInstance(vtk_img, vtk.vtkImageData)

        dims = vtk_img.GetDimensions()
        self.assertEqual(dims, (60, 60, 40))

        spacing = vtk_img.GetSpacing()
        self.assertEqual(spacing, (1.0, 1.0, 1.0))

        scalars = vtk_img.GetPointData().GetScalars()
        self.assertIsNotNone(scalars)
        s_range = scalars.GetRange()
        # HU 범위 검증 (피질골 1200, 치아 2200, 공기 -1000 주변)
        self.assertLess(s_range[0], -800)
        self.assertGreater(s_range[1], 1500)

    def test_02_state_manager(self):
        """StateManager 좌표 및 윈도우/레벨 신호 전파 검증"""
        state_mgr = StateManager()

        received_coords = []
        state_mgr.coordinates_changed.connect(lambda x, y, z: received_coords.append((x, y, z)))

        received_wl = []
        state_mgr.window_level_changed.connect(lambda w, l: received_wl.append((w, l)))

        vtk_img, meta = DicomReader.load_single_file(self.test_file)
        state_mgr.set_volume_data(vtk_img, meta)

        # 초기 중심 좌표 확인
        bnds = state_mgr.bounds
        expected_cx = 0.5 * (bnds[0] + bnds[1])
        expected_cy = 0.5 * (bnds[2] + bnds[3])
        expected_cz = 0.5 * (bnds[4] + bnds[5])

        cur_coords = state_mgr.current_coords
        self.assertAlmostEqual(cur_coords[0], expected_cx, places=2)
        self.assertAlmostEqual(cur_coords[1], expected_cy, places=2)
        self.assertAlmostEqual(cur_coords[2], expected_cz, places=2)

        # 좌표 변경 및 클램핑 테스트
        state_mgr.set_coordinates(0.0, 0.0, 10.0)
        self.assertTrue(len(received_coords) > 0)
        self.assertAlmostEqual(state_mgr.current_coords[2], 10.0, places=2)

        # 복셀 값 조회
        hu = state_mgr.get_voxel_value_at(0.0, 0.0, 10.0)
        self.assertIsNotNone(hu)

        # Window/Level 변경
        state_mgr.set_window_level(2000.0, 500.0)
        self.assertEqual(state_mgr.window, 2000.0)
        self.assertEqual(state_mgr.level, 500.0)
        self.assertEqual(received_wl[-1], (2000.0, 500.0))

    def test_03_ui_components_integration(self):
        """MPRContainer, VolumeViewerWindow, MainWindow UI 컴포넌트 통합 검증"""
        main_win = MainWindow()
        self.assertIsNotNone(main_win)

        # 샘플 데이터 원클릭 로드 테스트
        main_win.control_panel._load_sample_data()
        self.assertIsNotNone(main_win.state_mgr.vtk_image)
        self.assertEqual(main_win.state_mgr.dimensions, (120, 120, 80))

        # 3D 뷰어 창 열기
        main_win.open_3d_viewer()
        self.assertIsNotNone(main_win.volume_window)
        self.assertTrue(main_win.volume_window.isVisible())

        # 좌표 이동 시뮬레이션
        main_win.state_mgr.set_coordinates(5.0, -2.0, 15.0)

        # 2D MPR 뷰어들의 카메라 및 슬라이스 업데이트 검증
        mpr = main_win.mpr_container
        self.assertEqual(mpr.axial_panel.mpr_widget.axis, 2)
        self.assertEqual(mpr.coronal_panel.mpr_widget.axis, 1)
        self.assertEqual(mpr.sagittal_panel.mpr_widget.axis, 0)

        main_win.close()

    def test_04_real_sample_dicom_series(self):
        """실제 치과 CT 샘플 디렉터리(재귀 탐색 및 517 슬라이스) 로드 검증"""
        sample_dir = os.path.join(project_root, "samples", "Class 3 malocclusion")
        if os.path.isdir(sample_dir):
            vtk_img, meta = DicomReader.load_dicom_series(sample_dir)
            self.assertIsNotNone(vtk_img)
            self.assertEqual(vtk_img.GetDimensions(), (640, 640, 517))
            self.assertEqual(meta.get("file_count"), 517)


if __name__ == "__main__":
    unittest.main()
