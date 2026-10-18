"""
test_geometry_ui_and_export.py
Step 7: Boolean UI 연결 및 결과 검증 (GeometryPanel, Worker, Export, VolumeViewer 연동) 테스트 슈트.
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import pytest
import trimesh

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
from dental_fea_app.geometry.boolean_ops import FeaGeometry
from dental_fea_app.ui.control_panel import ControlPanel
from dental_fea_app.ui.geometry.geometry_panel import GeometryPanel
from dental_fea_app.ui.geometry.geometry_worker import GeometryWorker
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


def create_dummy_fea_geometry():
    """테스트용 간단한 FeaGeometry 객체 생성 (구와 원통)"""
    sphere_outer = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    sphere_inner = trimesh.creation.icosphere(subdivisions=2, radius=6.0)
    cylinder = trimesh.creation.cylinder(radius=2.0, height=12.0)

    # 단순한 리포트 dict
    report = {
        "cortical": {"volume_mm3": 100.0, "is_watertight": True, "euler_number": 2, "connected_components": 1},
        "cancellous": {"volume_mm3": 80.0, "is_watertight": True, "euler_number": 2, "connected_components": 1},
        "implant": {"volume_mm3": 30.0, "is_watertight": True, "euler_number": 2, "connected_components": 1},
        "interface": {"cortical_contact_area_mm2": 15.5, "cancellous_contact_area_mm2": 25.2, "undersize_mm": 0.0},
        "warnings": []
    }
    return FeaGeometry(cortical=sphere_outer, cancellous=sphere_inner, implant=cylinder, report=report)


class TestGeometryUIAndExport(unittest.TestCase):

    def setUp(self):
        self.state_mgr = StateManager()

    def test_01_control_panel_has_geometry_tab(self):
        """ControlPanel에 '4. 기하 처리' 탭이 추가되었고 초기 비활성화/활성화 상태가 올바른지 확인"""
        cp = ControlPanel(self.state_mgr, lambda: None)
        self.assertTrue(hasattr(cp, "geometry_panel"))
        self.assertEqual(cp.workflow_tabs.count(), 5)
        self.assertEqual(cp.workflow_tabs.tabText(3), "4. 기하 처리")
        self.assertEqual(cp.workflow_tabs.tabText(4), "5. 해석")

        # 볼륨 로드 전에는 비활성화 (Tab index 3)
        self.assertFalse(cp.workflow_tabs.isTabEnabled(3))

        # 더미 볼륨 로드 시그널 트리거 시 활성화
        cp._on_volume_loaded(object())
        self.assertTrue(cp.workflow_tabs.isTabEnabled(3))

    def test_02_geometry_panel_initial_state_and_stale_detection(self):
        """GeometryPanel 초기 상태 및 stale(오래됨) 감지 배지 동작 검증"""
        panel = GeometryPanel(self.state_mgr)

        # 초기 값 확인
        self.assertEqual(panel.spin_roi_margin.value(), 12.0)
        self.assertEqual(panel.spin_smooth.value(), 15)
        self.assertEqual(panel.spin_target_edge.value(), 1.0)
        self.assertEqual(panel.spin_undersize.value(), 0.0)

        # 초기에는 결과 없음
        self.assertFalse(panel.btn_export_stl.isEnabled())
        self.assertFalse(panel.group_vis.isEnabled())

        # 기하 설정
        fea_geom = create_dummy_fea_geometry()
        self.state_mgr.set_fea_geometry(fea_geom)

        self.assertTrue(panel.btn_export_stl.isEnabled())
        self.assertTrue(panel.group_vis.isEnabled())
        self.assertIn("최신 상태", panel.lbl_stale_status.text())

        # 임플란트 변경 시그널 발생 시 STALE 상태로 전환되는지 확인
        self.state_mgr.implants_updated.emit()
        self.assertIn("재생성 필요", panel.lbl_stale_status.text())

    def test_03_export_stl_and_manifest(self):
        """STL 내보내기 및 manifest.json 파일 생성 검증"""
        panel = GeometryPanel(self.state_mgr)
        fea_geom = create_dummy_fea_geometry()
        self.state_mgr.set_fea_geometry(fea_geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            success, msg = panel.export_fea_files(tmpdir)
            self.assertTrue(success, f"Export failed: {msg}")

            cortical_path = os.path.join(tmpdir, "cortical.stl")
            cancellous_path = os.path.join(tmpdir, "cancellous.stl")
            implant_path = os.path.join(tmpdir, "implant.stl")
            manifest_path = os.path.join(tmpdir, "manifest.json")

            self.assertTrue(os.path.exists(cortical_path))
            self.assertTrue(os.path.exists(cancellous_path))
            self.assertTrue(os.path.exists(implant_path))
            self.assertTrue(os.path.exists(manifest_path))

            # STL 파일 크기 검증
            self.assertGreater(os.path.getsize(cortical_path), 0)
            self.assertGreater(os.path.getsize(cancellous_path), 0)
            self.assertGreater(os.path.getsize(implant_path), 0)

            # Manifest 내용 검증
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["coordinate_system"], "LPS")
            self.assertEqual(manifest["unit"], "mm")
            self.assertIn("parameters", manifest)
            self.assertEqual(manifest["parameters"]["roi_margin_mm"], 12.0)
            self.assertEqual(manifest["parameters"]["undersize_mm"], 0.0)
            self.assertIn("report", manifest)
            self.assertEqual(manifest["report"]["interface"]["cortical_contact_area_mm2"], 15.5)

    def test_04_volume_viewer_fea_actor_and_clipping(self):
        """VolumeViewerWindow에서 FeaGeometry를 수신하여 VTK 액터 및 클리핑 평면이 정상 작동하는지 확인"""
        viewer = VolumeViewerWindow(self.state_mgr)

        self.assertTrue(hasattr(viewer, "fea_cortical_actor"))
        self.assertTrue(hasattr(viewer, "fea_cancellous_actor"))
        self.assertTrue(hasattr(viewer, "fea_implant_actor"))
        self.assertTrue(hasattr(viewer, "fea_clip_plane"))

        # 기하 설정
        fea_geom = create_dummy_fea_geometry()
        self.state_mgr.set_fea_geometry(fea_geom)

        # 액터가 렌더러에 추가되었는지 확인
        actors = viewer.renderer.GetActors()
        self.assertTrue(actors.IsItemPresent(viewer.fea_cortical_actor))
        self.assertTrue(actors.IsItemPresent(viewer.fea_cancellous_actor))
        self.assertTrue(actors.IsItemPresent(viewer.fea_implant_actor))

        # 가시성 토글 테스트
        self.state_mgr.set_fea_visibility({"cortical": False, "cancellous": True, "implant": True})
        self.assertEqual(viewer.fea_cortical_actor.GetVisibility(), 0)
        self.assertEqual(viewer.fea_cancellous_actor.GetVisibility(), 1)

        # 단면 클리핑 슬라이더 조절 테스트
        self.state_mgr.set_fea_clipping({"enabled": True, "axis": "Z", "value_pct": 50.0, "inverted": False})
        plane_normal = viewer.fea_clip_plane.GetNormal()
        self.assertEqual(plane_normal, (0.0, 0.0, 1.0))
