"""
fea_geometry_viewer.py
메인 화면 우측 하단 창에 FEA 기하(피질골, 수질골, 임플란트) 3D 솔리드를 렌더링하고,
가시성 토글 및 단면 컷팅(클리핑 평면)을 실시간으로 반영하는 독립 3D 뷰어 위젯.
"""

from typing import Optional, Dict, Any
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QToolButton, QStackedLayout
)
from PySide6.QtCore import Qt, QSize

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.geometry.surface_extract import trimesh_to_vtk_polydata
from dental_fea_app.geometry.boolean_ops import FeaGeometry
from dental_fea_app.ui.icons import IconFactory


class FeaGeometryViewWidget(QWidget):
    """
    메인 화면 우측 하단(사분면 4번) 전용 3D FEA 지오메트리 뷰어 위젯.
    """

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self._fea_bounds = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 1. 상단 미니멀 헤더
        header = QFrame(self)
        header.setFixedHeight(30)
        header.setStyleSheet("""
            QFrame {
                background-color: #222225;
                border: 1px solid #333338;
                border-radius: 3px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 4, 0)
        header_layout.setSpacing(4)

        self.title_label = QLabel("3D FEA Geometry (기하 형상)", header)
        self.title_label.setStyleSheet("font-weight: bold; color: #61afef; font-size: 11px;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        btn_qss = """
            QToolButton {
                background-color: #2c2c32;
                border: 1px solid #444450;
                border-radius: 3px;
                padding: 2px;
            }
            QToolButton:hover {
                background-color: #3e3e48;
                border-color: #61afef;
            }
        """
        self.btn_reset_cam = QToolButton(header)
        self.btn_reset_cam.setIcon(IconFactory.reset_view(22))
        self.btn_reset_cam.setIconSize(QSize(18, 18))
        self.btn_reset_cam.setFixedSize(26, 24)
        self.btn_reset_cam.setToolTip("3D 시점 초기화 (카메라 리셋)")
        self.btn_reset_cam.setStyleSheet(btn_qss)
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        header_layout.addWidget(self.btn_reset_cam)

        layout.addWidget(header)

        # 2. 3D VTK 렌더링 위젯 및 오버레이 스택
        self.render_container = QFrame(self)
        self.render_container.setStyleSheet("background-color: #121214; border: 1px solid #282830; border-radius: 4px;")
        self.stack = QStackedLayout(self.render_container)
        self.stack.setContentsMargins(0, 0, 0, 0)

        # 안내 문구 오버레이 라벨
        self.lbl_placeholder = QLabel(
            "기하 생성 후 3D 형상(피질골, 수질골, 임플란트)이\n여기에 표시됩니다.\n\n좌측 '4. 기하 처리' 탭에서 [기하 생성]을 실행하십시오.",
            self.render_container
        )
        self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_placeholder.setStyleSheet("color: #666675; font-size: 12px; font-weight: bold; background: transparent;")
        self.stack.addWidget(self.lbl_placeholder)

        self.vtkWidget = QVTKRenderWindowInteractor(self.render_container)
        self.stack.addWidget(self.vtkWidget)

        self.stack.setCurrentWidget(self.lbl_placeholder)

        layout.addWidget(self.render_container, stretch=1)

        # 3. VTK 렌더러 및 카메라 인터랙터 설정
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.08, 0.10)
        self.renderer.SetBackground2(0.14, 0.14, 0.18)
        self.renderer.SetGradientBackground(True)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidget.SetInteractorStyle(style)

        # 4. 방향 마커 (Orientation Marker)
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(15.0, 15.0, 15.0)
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 0.3, 0.3)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.3, 1, 0.3)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.3, 0.6, 1)

        self.marker_widget = vtk.vtkOrientationMarkerWidget()
        self.marker_widget.SetOrientationMarker(axes)
        self.marker_widget.SetInteractor(self.vtkWidget._Iren)
        self.marker_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
        self.marker_widget.SetEnabled(1)
        self.marker_widget.InteractiveOff()

        # 5. FEA 액터 및 클리핑 평면 초기화
        self._init_actors()

        # 6. StateManager 신호 연결
        self.state_mgr.fea_geometry_changed.connect(self._on_fea_geometry_changed)
        self.state_mgr.fea_visibility_changed.connect(self._on_fea_visibility_changed)
        self.state_mgr.fea_clipping_changed.connect(self._on_fea_clipping_changed)

        # 이미 생성된 기하가 있을 경우 즉시 표시
        geom = getattr(self.state_mgr, "fea_geometry", None)
        if geom is not None:
            self._on_fea_geometry_changed(geom)

        self.vtkWidget.Initialize()
        self.vtkWidget.Start()

    def _init_actors(self):
        """피질골, 수질골, 임플란트 3개 액터 및 단면 클리핑 평면 생성"""
        self.fea_clip_plane = vtk.vtkPlane()
        self.fea_clip_plane.SetOrigin(0.0, 0.0, 0.0)
        self.fea_clip_plane.SetNormal(0.0, 0.0, 1.0)

        # 1. 피질골 솔리드 (좌측 패널 #e06c75 색상과 일치화)
        self.cortical_mapper = vtk.vtkPolyDataMapper()
        self.cortical_actor = vtk.vtkActor()
        self.cortical_actor.SetMapper(self.cortical_mapper)
        self.cortical_actor.GetProperty().SetColor(0.88, 0.42, 0.46)
        self.cortical_actor.GetProperty().SetOpacity(0.85)
        self.cortical_actor.GetProperty().SetSpecular(0.2)
        self.cortical_actor.GetProperty().SetSpecularPower(20.0)
        self.cortical_actor.SetVisibility(False)
        self.renderer.AddActor(self.cortical_actor)

        # 2. 수질골 솔리드 (수질골 연파랑/산호)
        self.cancellous_mapper = vtk.vtkPolyDataMapper()
        self.cancellous_actor = vtk.vtkActor()
        self.cancellous_actor.SetMapper(self.cancellous_mapper)
        self.cancellous_actor.GetProperty().SetColor(0.38, 0.69, 0.94)
        self.cancellous_actor.GetProperty().SetOpacity(0.85)
        self.cancellous_actor.GetProperty().SetSpecular(0.2)
        self.cancellous_actor.GetProperty().SetSpecularPower(20.0)
        self.cancellous_actor.SetVisibility(False)
        self.renderer.AddActor(self.cancellous_actor)

        # 3. 임플란트 솔리드 (골드/티타늄)
        self.implant_mapper = vtk.vtkPolyDataMapper()
        self.implant_actor = vtk.vtkActor()
        self.implant_actor.SetMapper(self.implant_mapper)
        self.implant_actor.GetProperty().SetColor(0.90, 0.75, 0.40)
        self.implant_actor.GetProperty().SetOpacity(1.0)
        self.implant_actor.GetProperty().SetSpecular(0.6)
        self.implant_actor.GetProperty().SetSpecularPower(40.0)
        self.implant_actor.SetVisibility(False)
        self.renderer.AddActor(self.implant_actor)

    def _on_fea_geometry_changed(self, fea_geom: Optional[FeaGeometry]):
        """FEA 기하 솔리드가 생성/갱신되었을 때 매핑 및 카메라 리셋"""
        if fea_geom is None:
            self.cortical_actor.SetVisibility(False)
            self.cancellous_actor.SetVisibility(False)
            self.implant_actor.SetVisibility(False)
            self.stack.setCurrentWidget(self.lbl_placeholder)
            self.render()
            return

        self.stack.setCurrentWidget(self.vtkWidget)

        poly_c = trimesh_to_vtk_polydata(fea_geom.cortical)
        poly_cn = trimesh_to_vtk_polydata(fea_geom.cancellous)
        poly_i = trimesh_to_vtk_polydata(fea_geom.implant)

        self.cortical_mapper.SetInputData(poly_c)
        self.cancellous_mapper.SetInputData(poly_cn)
        self.implant_mapper.SetInputData(poly_i)

        all_bounds = []
        for poly in [poly_c, poly_cn, poly_i]:
            if poly.GetNumberOfPoints() > 0:
                all_bounds.append(poly.GetBounds())
        if all_bounds:
            b_arr = np.array(all_bounds)
            self._fea_bounds = (
                float(b_arr[:, 0].min()), float(b_arr[:, 1].max()),
                float(b_arr[:, 2].min()), float(b_arr[:, 2].max()),
                float(b_arr[:, 4].min()), float(b_arr[:, 5].max()),
            )

        self._apply_fea_visibility()
        self._apply_fea_clipping()
        self.reset_camera()

    def _on_fea_visibility_changed(self, vis: dict):
        self._apply_fea_visibility()
        self.render()

    def _apply_fea_visibility(self):
        geom = getattr(self.state_mgr, "fea_geometry", None)
        if geom is None:
            self.cortical_actor.SetVisibility(False)
            self.cancellous_actor.SetVisibility(False)
            self.implant_actor.SetVisibility(False)
            return

        vis = self.state_mgr.fea_visibility
        opacity = float(vis.get("opacity", 0.85))

        self.cortical_actor.SetVisibility(bool(vis.get("cortical", True)))
        self.cortical_actor.GetProperty().SetOpacity(opacity)

        self.cancellous_actor.SetVisibility(bool(vis.get("cancellous", True)))
        self.cancellous_actor.GetProperty().SetOpacity(opacity)

        self.implant_actor.SetVisibility(bool(vis.get("implant", True)))
        self.implant_actor.GetProperty().SetOpacity(1.0)

    def _on_fea_clipping_changed(self, clip: dict):
        self._apply_fea_clipping()
        self.render()

    def _apply_fea_clipping(self):
        clip = getattr(self.state_mgr, "fea_clipping", None)
        if clip is None or not clip.get("enabled", False):
            self.cortical_mapper.RemoveAllClippingPlanes()
            self.cancellous_mapper.RemoveAllClippingPlanes()
            self.implant_mapper.RemoveAllClippingPlanes()
            self.render()
            return

        axis = clip.get("axis", "Z")
        pct = float(clip.get("value_pct", 50.0)) / 100.0
        inverted = bool(clip.get("inverted", False))

        bounds = self._fea_bounds or self.state_mgr.bounds
        normal = [0.0, 0.0, 0.0]
        origin = [0.0, 0.0, 0.0]

        if axis == "X":
            x_min, x_max = bounds[0], bounds[1]
            origin[0] = x_min + (x_max - x_min) * pct
            origin[1] = (bounds[2] + bounds[3]) / 2.0
            origin[2] = (bounds[4] + bounds[5]) / 2.0
            normal[0] = -1.0 if inverted else 1.0
        elif axis == "Y":
            y_min, y_max = bounds[2], bounds[3]
            origin[0] = (bounds[0] + bounds[1]) / 2.0
            origin[1] = y_min + (y_max - y_min) * pct
            origin[2] = (bounds[4] + bounds[5]) / 2.0
            normal[1] = -1.0 if inverted else 1.0
        else:
            z_min, z_max = bounds[4], bounds[5]
            origin[0] = (bounds[0] + bounds[1]) / 2.0
            origin[1] = (bounds[2] + bounds[3]) / 2.0
            origin[2] = z_min + (z_max - z_min) * pct
            normal[2] = -1.0 if inverted else 1.0

        self.fea_clip_plane.SetOrigin(origin[0], origin[1], origin[2])
        self.fea_clip_plane.SetNormal(normal[0], normal[1], normal[2])

        for mapper in [self.cortical_mapper, self.cancellous_mapper, self.implant_mapper]:
            mapper.RemoveAllClippingPlanes()
            mapper.AddClippingPlane(self.fea_clip_plane)

        self.render()

    def reset_camera(self):
        """카메라를 메쉬 전체가 보이도록 리셋"""
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.Elevation(20.0)
        cam.Azimuth(30.0)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def render(self):
        rw = self.vtkWidget.GetRenderWindow()
        if rw:
            rw.Render()
