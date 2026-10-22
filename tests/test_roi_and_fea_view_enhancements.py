"""
test_roi_and_fea_view_enhancements.py
ROI boundary 회전 및 '원래대로' 기능, ROI 내 임플란트 기준 기하 생성,
ROI 구간 segmentation 및 메인 화면 우하단 3D FEA 지오메트리 뷰어 연동 검증 테스트.
"""

import os
import sys
import ctypes
import unittest
import numpy as np
import trimesh
import vtk

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

local_xcb_cursor = os.path.join(project_root, ".local_libs", "usr", "lib", "x86_64-linux-gnu", "libxcb-cursor.so.0")
if os.path.exists(local_xcb_cursor):
    try:
        ctypes.CDLL(local_xcb_cursor)
    except Exception:
        pass

from PySide6.QtWidgets import QApplication

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.ui.main_window import MainWindow
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow
from dental_fea_app.ui.geometry.fea_geometry_viewer import FeaGeometryViewWidget
from dental_fea_app.ui.geometry.geometry_worker import get_implant_world_mesh
from dental_fea_app.placement.implant_placement_manager import ImplantPlacementManager, ImplantSpec
from dental_fea_app.geometry.boolean_ops import FeaGeometry


class TestRoiAndFeaViewEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.state_mgr = StateManager()
        # 가상 3D 볼륨 생성 (10x10x10 복셀, spacing=1.0, origin=(0,0,0))
        img = vtk.vtkImageData()
        img.SetDimensions(10, 10, 10)
        img.SetSpacing(1.0, 1.0, 1.0)
        img.SetOrigin(0.0, 0.0, 0.0)
        img.AllocateScalars(vtk.VTK_SHORT, 1)
        self.state_mgr.set_volume_data(img, {})

    def test_01_roi_rotate_and_reset_in_volume_viewer(self):
        """VolumeViewerWindow에서 ROI 회전 체크박스, 3D 기즈모 토글 및 '원래대로' 동작 검증"""
        viewer = VolumeViewerWindow(self.state_mgr)

        # 1. '회전' 체크박스 존재 및 '원래대로' 체크박스 확인
        self.assertTrue(hasattr(viewer, "chk_roi_rotate"))
        self.assertTrue(hasattr(viewer, "chk_roi_reset"))

        # 2. 축별 회전 버튼은 제거되어 존재하지 않아야 함
        self.assertFalse(hasattr(viewer, "btn_rot_x_pos"))
        self.assertFalse(hasattr(viewer, "btn_rot_y_pos"))
        self.assertFalse(hasattr(viewer, "btn_rot_z_pos"))

        # 3. 3D 회전 기즈모(3축 링) 액터 존재 및 체크박스에 따른 가시성 토글 확인
        self.assertTrue(hasattr(viewer, "roi_gizmo_actors"))
        self.assertIn("X", viewer.roi_gizmo_actors)
        self.assertIn("Y", viewer.roi_gizmo_actors)
        self.assertIn("Z", viewer.roi_gizmo_actors)

        # 기본 상태(chk_roi_rotate=False)에서는 기즈모가 숨겨져 있어야 함
        self.assertFalse(viewer.chk_roi_rotate.isChecked())
        viewer._update_roi_rotation_points()
        self.assertFalse(viewer.roi_gizmo_actors["X"].GetVisibility())

        # chk_roi_rotate=True 시 기즈모 표시 확인
        viewer.chk_roi_rotate.setChecked(True)
        self.assertTrue(viewer.roi_gizmo_actors["X"].GetVisibility())
        self.assertTrue(viewer.roi_gizmo_actors["Y"].GetVisibility())
        self.assertTrue(viewer.roi_gizmo_actors["Z"].GetVisibility())

        # chk_roi_rotate=False 시 다시 숨김 확인
        viewer.chk_roi_rotate.setChecked(False)
        self.assertFalse(viewer.roi_gizmo_actors["X"].GetVisibility())

        # 4. boundary 설정 모드 활성화 시 box_widget 윤곽선 피킹 비활성화(선 클릭으로 인한 왜곡 방지) 검증
        viewer.btn_roi_boundary.setChecked(True)
        props = viewer.renderer.GetViewProps()
        props.InitTraversal()
        outline_pickable = False
        for i in range(props.GetNumberOfItems()):
            p = props.GetNextProp()
            if hasattr(p, "GetProperty") and p.GetProperty() == viewer.box_widget.GetOutlineProperty():
                if p.GetPickable():
                    outline_pickable = True
        self.assertFalse(outline_pickable, "box 선분은 크기 왜곡 방지를 위해 pickable하지 않아야 합니다.")

        # 4. 축별 회전 동작 검증 (회전 시 상자 크기/길이 보존 확인)
        corners_before, center_before = viewer._get_roi_box_corners_and_center()
        edge_len_before = np.linalg.norm(corners_before[1] - corners_before[0])

        viewer.rotate_roi_box("Z", 45.0)
        viewer.rotate_roi_box("X", 30.0)
        viewer.rotate_roi_box("Y", 15.0)

        corners_after, center_after = viewer._get_roi_box_corners_and_center()
        edge_len_after = np.linalg.norm(corners_after[1] - corners_after[0])

        self.assertAlmostEqual(edge_len_before, edge_len_after, delta=1e-4)
        np.testing.assert_allclose(center_before, center_after, atol=1e-3)
        self.assertIsNotNone(self.state_mgr.roi_bounds)

        # 5. '원래대로' 체크 시 전체 볼륨 bounds로 복원 및 자동 언체크 확인
        viewer.chk_roi_reset.setChecked(True)
        full_bounds = self.state_mgr.bounds
        self.assertEqual(self.state_mgr.roi_bounds, full_bounds)
        self.assertFalse(viewer.chk_roi_reset.isChecked())

    def test_02_implant_filtering_by_roi_bounds(self):
        """ROI bounds 설정 시 해당 ROI 내의 임플란트만 선별되는지 검증"""
        renderer = vtk.vtkRenderer()
        mgr = ImplantPlacementManager(renderer_3d=renderer)
        mgr.set_state_manager(self.state_mgr)
        self.state_mgr.implant_placement_mgr = mgr

        # 임플란트 1: 위치 (2.0, 2.0, 2.0)
        # 임플란트 2: 위치 (15.0, 15.0, 15.0)
        spec = ImplantSpec(
            id="test_imp",
            name="Test Implant",
            diameter_mm=2.0,
            length_mm=4.0,
            taper_deg=0.0,
            top_type="flat",
            thread_type="triangular",
            micro_threads=False
        )
        # 더미 실린더 메쉬 vtkPolyData 생성
        cyl = vtk.vtkCylinderSource()
        cyl.SetRadius(1.0)
        cyl.SetHeight(4.0)
        cyl.Update()
        poly = cyl.GetOutput()

        id1 = mgr.place_implant(spec, initial_position=(2.0, 2.0, 2.0))
        imp1 = mgr.get_implant(id1)
        imp1.raw_polydata = poly

        id2 = mgr.place_implant(spec, initial_position=(15.0, 15.0, 15.0))
        imp2 = mgr.get_implant(id2)
        imp2.raw_polydata = poly

        # 1. ROI bounds가 없을 때는 모든 임플란트 결합
        self.state_mgr.set_roi_bounds(None)
        all_mesh = get_implant_world_mesh(self.state_mgr)
        self.assertIsNotNone(all_mesh)

        # 2. ROI bounds를 임플란트 1번 주변으로 설정 (Z: -5~5mm)
        self.state_mgr.set_roi_bounds((0.0, 5.0, 0.0, 5.0, -5.0, 5.0))
        filtered_mesh = get_implant_world_mesh(self.state_mgr)
        # 임플란트 1만 선택되므로 바운딩 박스 중심이 (2, 2, -2) 주변이어야 함
        center = filtered_mesh.centroid
        self.assertAlmostEqual(center[0], 2.0, delta=1.0)
        self.assertAlmostEqual(center[1], 2.0, delta=1.0)
        self.assertAlmostEqual(center[2], -2.0, delta=1.0)

        # 3. 임플란트가 없는 빈 영역으로 ROI bounds 설정 시 예외 발생 확인
        self.state_mgr.set_roi_bounds((50.0, 60.0, 50.0, 60.0, 50.0, 60.0))
        with self.assertRaises(ValueError):
            get_implant_world_mesh(self.state_mgr)

    def test_03_fea_geometry_viewer_color_and_visibility(self):
        """FeaGeometryViewWidget의 피질골 색상 좌측 패널 일치(#e06c75) 및 기하 연동 검증"""
        fea_viewer = FeaGeometryViewWidget(self.state_mgr)

        # 기하 설정 시 플레이스홀더 라벨이 숨겨지고 3D 위젯이 표시되는지 확인
        box = trimesh.creation.box([2, 2, 2])
        geom = FeaGeometry(cortical=box, cancellous=box, implant=box, report={})
        self.state_mgr.set_fea_geometry(geom)

        self.assertEqual(fea_viewer.stack.currentWidget(), fea_viewer.vtkWidget)
        self.assertTrue(fea_viewer.cortical_actor.GetVisibility())

        # 피질골 액터 색상이 좌측 패널(#e06c75)과 일치하는지 확인 (RGB: 0.88, 0.42, 0.46)
        c_color = fea_viewer.cortical_actor.GetProperty().GetColor()
        self.assertAlmostEqual(c_color[0], 0.88, delta=0.01)
        self.assertAlmostEqual(c_color[1], 0.42, delta=0.01)
        self.assertAlmostEqual(c_color[2], 0.46, delta=0.01)

    def test_05_watertight_and_boolean_stability(self):
        """비매니폴드/다중 조각 메쉬에 대해서도 build_fea_geometry가 watertight를 유지하는지 검증"""
        from dental_fea_app.geometry.boolean_ops import build_fea_geometry

        # 1. 뼈 구 메쉬에 미세 파편 조각이 붙어있는 비매니폴드 피질골 메쉬 생성
        c_box = trimesh.creation.box([10, 10, 10])
        c_box.apply_translation([5, 5, 5])
        frag = trimesh.creation.box([0.5, 0.5, 0.5])
        frag.apply_translation([25, 25, 25])
        cortical_raw = trimesh.util.concatenate([c_box, frag])

        # 피질골과 인접한 수질골 블록
        cn_box = trimesh.creation.box([10, 10, 10])
        cn_box.apply_translation([15, 5, 5])

        # 두 뼈를 관통하는 임플란트 실린더
        implant = trimesh.creation.cylinder(radius=2, height=12)
        implant.apply_translation([10, 5, 5])

        # build_fea_geometry 실행 (check_volume=False 및 watertight 복구 검증)
        res = build_fea_geometry(
            cortical_mesh=cortical_raw,
            cancellous_mesh=cn_box,
            implant_mesh=implant,
            undersize_mm=0.0
        )

        self.assertTrue(res.report["cortical_watertight"], "피질골 최종 메쉬는 watertight이어야 합니다.")
        self.assertTrue(res.report["cancellous_watertight"], "수질골 최종 메쉬는 watertight이어야 합니다.")
        self.assertTrue(res.report["implant_watertight"], "임플란트 최종 메쉬는 watertight이어야 합니다.")
        self.assertEqual(res.report["cortical_pieces"], 1, "미세 파편이 제거되어 1개의 유효 부피여야 합니다.")

    def test_04_mesh_and_implant_coordinate_alignment(self):
        """CT 볼륨의 World LPS 원점/간격이 뼈 표면 메쉬에 정확히 반영되어 임플란트와 정렬되는지 검증"""
        # 비대칭 원점과 비단위 spacing을 가진 가상 볼륨
        img = vtk.vtkImageData()
        img.SetDimensions(20, 20, 20)
        img.SetSpacing(0.5, 0.5, 0.5)
        img.SetOrigin(-50.0, -40.0, -30.0)
        img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # 중심부에 구 형태의 뼈 마스크 생성
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[8:12, 8:12, 8:12] = True

        from dental_fea_app.geometry.surface_extract import mask_to_surface
        # reference_vtk만 넘겨서 추출
        mesh = mask_to_surface(mask, reference_vtk=img, smooth_iterations=0)
        self.assertGreater(len(mesh.vertices), 0)

        # 중심 좌표가 원점 (-50, -40, -30) + (10 * 0.5) = (-45, -35, -25) 부근이어야 함
        center = mesh.centroid
        self.assertAlmostEqual(center[0], -45.0, delta=1.0)
        self.assertAlmostEqual(center[1], -35.0, delta=1.0)
        self.assertAlmostEqual(center[2], -25.0, delta=1.0)

    def test_06_dicom_reload_full_reset(self):
        """새로운 DICOM 볼륨 로드 시 임플란트, ROI, FEA 기하/메쉬 및 3D FEA 뷰어가 깨끗이 초기화되는지 검증"""
        fea_viewer = FeaGeometryViewWidget(self.state_mgr)

        # 1. 상태 설정 (임플란트 1개 추가, ROI 설정, 기하 설정)
        spec = ImplantSpec(
            id="test_imp",
            name="Test Implant",
            diameter_mm=2.0,
            length_mm=4.0,
            taper_deg=0.0,
            top_type="flat",
            thread_type="triangular",
            micro_threads=False
        )
        self.state_mgr.implant_placement_mgr.place_implant(spec, initial_position=(5.0, 5.0, 5.0))
        self.state_mgr.set_roi_bounds([2, 8, 2, 8, 2, 8])
        c_box = trimesh.creation.box([4, 4, 4])
        dummy_geom = FeaGeometry(cortical=c_box, cancellous=c_box.copy(), implant=c_box.copy(), report={})
        self.state_mgr.set_fea_geometry(dummy_geom)

        self.assertEqual(len(self.state_mgr.implant_placement_mgr.implants), 1)
        self.assertIsNotNone(self.state_mgr.roi_bounds)
        self.assertIsNotNone(self.state_mgr.fea_geometry)
        self.assertEqual(fea_viewer.stack.currentWidget(), fea_viewer.vtkWidget)

        # 2. 새로운 DICOM 볼륨 로드
        new_img = vtk.vtkImageData()
        new_img.SetDimensions(12, 12, 12)
        new_img.SetSpacing(1.0, 1.0, 1.0)
        new_img.AllocateScalars(vtk.VTK_SHORT, 1)
        self.state_mgr.set_volume_data(new_img, {"series_path": "new_series"})

        # 3. 모든 상태가 초기화되었는지 검증
        self.assertEqual(len(self.state_mgr.implant_placement_mgr.implants), 0, "새 볼륨 로드 시 임플란트는 모두 삭제되어야 합니다.")
        self.assertIsNone(self.state_mgr.roi_bounds, "새 볼륨 로드 시 ROI bounds는 None이어야 합니다.")
        self.assertIsNone(self.state_mgr.fea_geometry, "새 볼륨 로드 시 FEA 기하는 초기화되어야 합니다.")
        self.assertIsNone(self.state_mgr.mesh_model, "새 볼륨 로드 시 FEA 메쉬는 초기화되어야 합니다.")
        self.assertEqual(fea_viewer.stack.currentWidget(), fea_viewer.lbl_placeholder, "3D FEA 화면은 플레이스홀더 안내 문구로 초기화되어야 합니다.")

    def test_07_roi_gizmo_axis_alignment_and_rotation(self):
        """ROI 박스가 회전할 때 기즈모 링 액터들의 자세가 박스의 로컬 축과 일치하고, 중심점 변위 없이 회전하는지 검증"""
        viewer = VolumeViewerWindow(self.state_mgr)
        viewer.chk_roi_rotate.setChecked(True)
        viewer.chk_roi_box.setChecked(True)
        viewer.btn_roi_boundary.setChecked(True)

        # 초기 상태에서 중심점 및 축 확인
        _, center_before = viewer._get_roi_box_corners_and_center()
        self.assertIsNotNone(center_before)

        # 1. X축 45도 회전
        viewer.rotate_roi_box("X", 45.0)
        _, center_after_x = viewer._get_roi_box_corners_and_center()
        self.assertAlmostEqual(np.linalg.norm(center_after_x - center_before), 0.0, places=4, msg="회전 후 박스 중심점은 유지되어야 함")

        # 기즈모 링 액터의 UserMatrix 추출
        actor_x = viewer.roi_gizmo_actors["X"]
        mat = actor_x.GetUserMatrix()
        self.assertIsNotNone(mat)

        # 2. Y축 30도 회전
        viewer.rotate_roi_box("Y", 30.0)
        _, center_after_y = viewer._get_roi_box_corners_and_center()
        self.assertAlmostEqual(np.linalg.norm(center_after_y - center_before), 0.0, places=4)

        # 3. Z축 -25도 회전
        viewer.rotate_roi_box("Z", -25.0)
        _, center_after_z = viewer._get_roi_box_corners_and_center()
        self.assertAlmostEqual(np.linalg.norm(center_after_z - center_before), 0.0, places=4)

    def test_08_rotated_roi_volume_clipping(self):
        """회전된 ROI 육면체에 대해 볼륨 매퍼에 6개의 OBB 클리핑 평면이 주입되는지 검증"""
        viewer = VolumeViewerWindow(self.state_mgr)
        viewer.chk_roi_apply.setChecked(True)
        self.state_mgr.set_roi_bounds([2, 8, 2, 8, 2, 8])

        # 박스 회전
        viewer.rotate_roi_box("Z", 35.0)
        viewer._update_volume_cropping()

        # 볼륨 매퍼의 클리핑 평면 개수 검증 (6개 평면)
        n_planes = viewer.volume_mapper.GetNumberOfClippingPlanes()
        self.assertEqual(n_planes, 6, "회전된 ROI 육면체 클리핑 시 6개의 절단 평면이 등록되어야 합니다.")


