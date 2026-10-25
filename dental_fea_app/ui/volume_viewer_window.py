"""
volume_viewer_window.py
독립 창으로 구동되는 3D 볼륨 렌더러 윈도우.
- 우측 상단 ⚙ 설정 아이콘 및 플로팅 제어 패널
- 높이 약 20mm 수직 투명도 슬라이더 + 음영(Shading) 체크박스
- 2D 좌표와 실시간 동조되는 반투명 Axial/Coronal/Sagittal 평면 및 토글
- [A], [C], [S] 평면 컷팅 (좌클릭: 정방향, 우클릭: 역방향) + [O] 전체보기
- 설정 영속화 (SettingsManager) 연동
"""

from typing import Optional
import numpy as np
import vtk
from vtk.util import numpy_support
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QComboBox, QPushButton, QLabel, QStatusBar,
    QCheckBox, QSlider, QFrame, QToolButton, QGroupBox, QMenu
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QEvent
from PySide6.QtGui import QAction, QMouseEvent, QIcon, QColor

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.core.settings_manager import SettingsManager
from dental_fea_app.ui.icons import IconFactory


def trimesh_to_vtk_polydata(mesh) -> vtk.vtkPolyData:
    """trimesh.Trimesh를 vtkPolyData로 변환합니다."""
    poly = vtk.vtkPolyData()
    if mesh is None or len(mesh.faces) == 0:
        return poly

    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float32), deep=True))
    poly.SetPoints(pts)

    cells = vtk.vtkCellArray()
    conn = np.ascontiguousarray(mesh.faces.ravel(), dtype=np.int64)
    vtk_id_arr = numpy_support.numpy_to_vtkIdTypeArray(conn, deep=True)
    cells.SetData(3, vtk_id_arr)
    poly.SetPolys(cells)

    norm_filter = vtk.vtkPolyDataNormals()
    norm_filter.SetInputData(poly)
    norm_filter.ComputePointNormalsOn()
    norm_filter.ComputeCellNormalsOff()
    norm_filter.SplittingOff()
    norm_filter.Update()
    return norm_filter.GetOutput()


class ClippingButton(QToolButton):
    """
    좌클릭 및 우클릭 이벤트를 모두 지원하는 클리핑 토글 버튼. (순수 아이콘)
    """
    left_clicked = Signal()
    right_clicked = Signal()

    def __init__(self, icon: QIcon, tooltip: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setIcon(icon)
        self.setIconSize(QSize(22, 22))
        self.setFixedSize(32, 32)
        self.setToolTip(tooltip)
        self.setStyleSheet("""
            QToolButton {
                background-color: #2c2c34;
                border: 1px solid #4a4a58;
                border-radius: 4px;
                padding: 2px;
            }
            QToolButton:hover {
                background-color: #3d3d4a;
                border-color: #61afef;
            }
        """)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.left_clicked.emit()
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)


class VolumeViewerWindow(QMainWindow):
    """
    독립된 3D Volume Rendering 윈도우.
    """
    window_closed = Signal()

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None, settings_mgr: Optional[SettingsManager] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.settings_mgr = settings_mgr if settings_mgr is not None else SettingsManager()
        self.setWindowTitle("Dental FEA - 3D Volume & Mesh Viewer")
        self.resize(920, 720)

        # 3D 평면 액터 및 클리핑 플레인 상태
        self.clip_planes = []
        self._current_clip_mode = "O"

        # 중앙 위젯 (수평 분할: 3D 뷰어 + 우측 설정 사이드바)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # VTK Interactor Widget (좌측 뷰어)
        self.vtkWidget = QVTKRenderWindowInteractor(central_widget)
        self.main_layout.addWidget(self.vtkWidget, stretch=1)

        # VTK 렌더러
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.12, 0.12, 0.14)
        self.renderer.SetBackground2(0.04, 0.04, 0.05)
        self.renderer.SetGradientBackground(True)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidget.SetInteractorStyle(style)

        # 3D 렌더링 스로틀 타이머 (마우스 연속 드래그 시 3D 재렌더링 부하 방지)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(60)  # 최대 약 16 FPS로 3D 렌더 제한
        self._render_timer.timeout.connect(self._do_render)

        self.is_result_view_mode: bool = False

        # 볼륨 렌더링 파이프라인
        self.volume_mapper = vtk.vtkSmartVolumeMapper()
        self.volume_mapper.SetRequestedRenderModeToGPU()

        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetAmbient(0.25)
        self.volume_property.SetDiffuse(0.75)
        self.volume_property.SetSpecular(0.3)
        self.volume_property.SetSpecularPower(20.0)

        self.color_func = vtk.vtkColorTransferFunction()
        self.opacity_func = vtk.vtkPiecewiseFunction()
        self.volume_property.SetColor(self.color_func)
        self.volume_property.SetScalarOpacity(self.opacity_func)

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)
        self.volume.SetVisibility(False)
        self.renderer.AddVolume(self.volume)

        # 3D 슬라이스 평면 액터 초기화 (Axial, Coronal, Sagittal)
        self._init_3d_planes()

        # 3D 축 마커
        self._init_orientation_marker()

        # 3D ROI 바운딩 박스 (파란색 육면체 외곽선)
        self._init_roi_box()

        # 3D 현재 위치 표시자 (2mm 빨간 구 및 축 색상 긴 십자선)
        self._init_position_indicators()

        # 우측 설정 사이드바 패널 생성 (기본 활성화)
        self.panel = self._create_settings_sidebar()
        self.main_layout.addWidget(self.panel, stretch=0)
        self.panel.setVisible(True)

        # 상단 툴바 구성 (모든 버튼 텍스트 없이 아이콘만 표시)
        self._init_toolbar()

        # 기본 골조직 프리셋
        self.apply_bone_preset()


        # 저장된 3D 설정 로드 및 적용
        self._load_saved_settings()

        # 상태바
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("3D 볼륨 뷰어가 준비되었습니다.")

        # 상태 관리자 신호 연결
        self.state_mgr.volume_loaded.connect(self.on_volume_loaded)
        self.state_mgr.coordinates_changed.connect(self.on_coordinates_changed)
        self.state_mgr.segmentation_result_changed.connect(self._on_segmentation_result_changed)
        self.state_mgr.segmentation_visibility_changed.connect(self._on_segmentation_visibility_changed)
        self.state_mgr.implant_model_changed.connect(self._on_implant_model_changed)
        self.state_mgr.implant_pose_changed.connect(self._on_implant_pose_changed)
        self.state_mgr.render_all_requested.connect(self.request_render)

        # 3D 분할 표면 메쉬 액터 초기화
        self._init_segmentation_surface_actors()

        # 3D 임플란트 메쉬 액터 초기화
        self._init_implant_actor()

        # 다중 임플란트 3D 렌더러 등록 및 3D Transform 기즈모 초기화
        self.state_mgr.implant_placement_mgr.set_renderer_3d(self.renderer)
        from dental_fea_app.placement.implant_gizmo import ImplantTransformGizmo
        self.implant_gizmo_3d = ImplantTransformGizmo(self.renderer, self.state_mgr.implant_placement_mgr)

        self.state_mgr.implant_selected.connect(self._on_implant_selected_3d)
        self.state_mgr.implant_transformed.connect(self._on_implant_transformed_3d)

        # FEA 체적 메쉬 및 경계조건 액터는 3D Volume 뷰가 아닌 3D FEA Geometry 뷰에서만 전담 표시
        self.fea_mesh_actors = {}
        self.fea_bc_actors = []


        # 이미 선택된 임플란트가 있을 경우 기즈모 즉시 부착
        sel_imp = self.state_mgr.implant_placement_mgr.get_selected_implant()
        if sel_imp and hasattr(self, "implant_gizmo_3d"):
            self.implant_gizmo_3d.attach_to_implant(sel_imp.id)

        # 마우스 상호작용 이벤트 필터
        self.vtkWidget.installEventFilter(self)
        self.vtkWidget.setMouseTracking(True)

        if self.state_mgr.vtk_image is not None:
            self.on_volume_loaded(self.state_mgr.vtk_image)

        if self.state_mgr.segmentation_result is not None:
            self._on_segmentation_result_changed(self.state_mgr.segmentation_result)

        if self.state_mgr.implant_model is not None:
            self._on_implant_model_changed(self.state_mgr.implant_model)

    def _init_3d_planes(self):
        """3D 공간 내 반투명 슬라이스 평면(Axial, Coronal, Sagittal) 액터 초기화"""
        # 1. Axial Plane (XY 평면, 연파랑)
        self.axial_plane_source = vtk.vtkPlaneSource()
        mapper_a = vtk.vtkPolyDataMapper()
        mapper_a.SetInputConnection(self.axial_plane_source.GetOutputPort())
        self.axial_plane_actor = vtk.vtkActor()
        self.axial_plane_actor.SetMapper(mapper_a)
        self.axial_plane_actor.GetProperty().SetColor(0.25, 0.65, 1.0)
        self.axial_plane_actor.GetProperty().SetOpacity(0.24)
        self.axial_plane_actor.SetVisibility(False)
        self.renderer.AddActor(self.axial_plane_actor)

        # 2. Coronal Plane (XZ 평면, 연초록)
        self.coronal_plane_source = vtk.vtkPlaneSource()
        mapper_c = vtk.vtkPolyDataMapper()
        mapper_c.SetInputConnection(self.coronal_plane_source.GetOutputPort())
        self.coronal_plane_actor = vtk.vtkActor()
        self.coronal_plane_actor.SetMapper(mapper_c)
        self.coronal_plane_actor.GetProperty().SetColor(0.25, 0.95, 0.3)
        self.coronal_plane_actor.GetProperty().SetOpacity(0.24)
        self.coronal_plane_actor.SetVisibility(False)
        self.renderer.AddActor(self.coronal_plane_actor)

        # 3. Sagittal Plane (YZ 평면, 연빨강)
        self.sagittal_plane_source = vtk.vtkPlaneSource()
        mapper_s = vtk.vtkPolyDataMapper()
        mapper_s.SetInputConnection(self.sagittal_plane_source.GetOutputPort())
        self.sagittal_plane_actor = vtk.vtkActor()
        self.sagittal_plane_actor.SetMapper(mapper_s)
        self.sagittal_plane_actor.GetProperty().SetColor(1.0, 0.35, 0.35)
        self.sagittal_plane_actor.GetProperty().SetOpacity(0.24)
        self.sagittal_plane_actor.SetVisibility(False)
        self.renderer.AddActor(self.sagittal_plane_actor)

    def _init_roi_box(self):
        """파란색 Bounding Box (두께 1) 및 VTK Cropping 파이프라인"""
        self.roi_box_source = vtk.vtkOutlineSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.roi_box_source.GetOutputPort())

        self.roi_box_actor = vtk.vtkActor()
        self.roi_box_actor.SetMapper(mapper)
        self.roi_box_actor.GetProperty().SetColor(0.2, 0.5, 1.0)  # Blue
        self.roi_box_actor.GetProperty().SetLineWidth(1.0)
        self.roi_box_actor.GetProperty().LightingOff()
        self.roi_box_actor.SetVisibility(False)
        self.renderer.AddActor(self.roi_box_actor)

        # 3D 대화형 조작용 vtkBoxWidget (선분/면/핸들로 각 축 크기 조절)
        self.box_widget = vtk.vtkBoxWidget()
        self.box_widget.SetInteractor(self.vtkWidget._Iren)
        self.box_widget.SetPlaceFactor(1.0)
        self.box_widget.SetRotationEnabled(True)  # 회전 기능 활성화
        self.box_widget.SetTranslationEnabled(True)
        self.box_widget.SetScalingEnabled(True)
        
        # 윤곽선 및 핸들 스타일
        self.box_widget.GetOutlineProperty().SetColor(0.2, 0.5, 1.0)
        self.box_widget.GetOutlineProperty().SetLineWidth(1.5)
        self.box_widget.GetHandleProperty().SetColor(1.0, 0.65, 0.0)  # Orange 핸들
        self.box_widget.GetHandleProperty().SetOpacity(0.85)
        self.box_widget.GetSelectedHandleProperty().SetColor(1.0, 1.0, 0.2)
        self.box_widget.AddObserver(vtk.vtkCommand.InteractionEvent, self._on_box_widget_interaction)
        self._box_widget_updating = False

        # StateManager 신호 연결
        self.state_mgr.roi_bounds_changed.connect(self._on_roi_bounds_updated)

        # 3D ROI 각 축별 회전 링 기즈모(Gizmo) 초기화
        self._init_roi_rotation_gizmo()

    def _init_roi_rotation_gizmo(self):
        """ROI boundary box의 중심에 배치되는 3D 회전 기즈모(X: Pitch 빨강, Y: Roll 초록, Z: Yaw 파랑 링) 초기화"""
        self._roi_dragging_axis = None
        self._roi_drag_start_x = 0
        self._roi_drag_start_y = 0
        self.roi_gizmo_actors = {}
        self.roi_gizmo_tube_filters = {}

        self._roi_gizmo_base_colors = {
            "X": (1.0, 0.22, 0.22),
            "Y": (0.22, 0.95, 0.28),
            "Z": (0.25, 0.65, 1.00),
        }
        self._roi_gizmo_highlight_colors = {
            "X": (1.0, 0.70, 0.70),
            "Y": (0.70, 1.00, 0.75),
            "Z": (0.70, 0.90, 1.00),
        }

        # Z: XY 평면 (Z축 수직 회전)
        # X: YZ 평면 (X축 수직 회전)
        # Y: XZ 평면 (Y축 수직 회전)
        rings_def = [
            ("Z", 0.0, 0.0, 0.0),
            ("X", 0.0, 90.0, 0.0),
            ("Y", 90.0, 0.0, 0.0),
        ]

        for axis, rx, ry, rz in rings_def:
            circle = vtk.vtkRegularPolygonSource()
            circle.SetNumberOfSides(64)
            circle.SetRadius(1.0)
            circle.GeneratePolygonOff()
            circle.Update()

            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(circle.GetOutputPort())
            tube.SetRadius(0.04)
            tube.SetNumberOfSides(16)
            tube.Update()
            self.roi_gizmo_tube_filters[axis] = tube

            trans = vtk.vtkTransform()
            trans.RotateX(rx)
            trans.RotateY(ry)
            trans.RotateZ(rz)

            tf = vtk.vtkTransformFilter()
            tf.SetInputConnection(tube.GetOutputPort())
            tf.SetTransform(trans)
            tf.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tf.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            c = self._roi_gizmo_base_colors[axis]
            actor.GetProperty().SetColor(c[0], c[1], c[2])
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            actor.GetProperty().SetLighting(False)
            actor.SetVisibility(False)
            self.renderer.AddActor(actor)
            self.roi_gizmo_actors[axis] = actor

        # 호환성을 위한 액터 참조 별칭
        self.roi_rot_actor_x = self.roi_gizmo_actors["X"]
        self.roi_rot_actor_y = self.roi_gizmo_actors["Y"]
        self.roi_rot_actor_z = self.roi_gizmo_actors["Z"]

    def _disable_box_widget_outline_picking(self):
        """vtkBoxWidget의 테두리 선분 및 면을 클릭했을 때 박스가 왜곡되거나 원치않게 스케일링되는 현상 방지"""
        if not hasattr(self, "box_widget"):
            return
        props = self.renderer.GetViewProps()
        props.InitTraversal()
        for i in range(props.GetNumberOfItems()):
            p = props.GetNextProp()
            if hasattr(p, "GetProperty"):
                prop = p.GetProperty()
                if prop in (self.box_widget.GetOutlineProperty(),
                            self.box_widget.GetSelectedOutlineProperty(),
                            self.box_widget.GetFaceProperty(),
                            self.box_widget.GetSelectedFaceProperty()):
                    p.SetPickable(0)

    def _get_roi_box_corners_and_center(self):
        """ROI box widget의 현재 8개 모서리 및 정확한 중심 좌표 반환"""
        if not hasattr(self, "box_widget"):
            return None, None
        poly = vtk.vtkPolyData()
        self.box_widget.GetPolyData(poly)
        pts = poly.GetPoints()
        if not pts or pts.GetNumberOfPoints() < 8:
            return None, None
        corners = [np.array(pts.GetPoint(i), dtype=np.float64) for i in range(8)]
        center = np.mean(corners, axis=0)
        return corners, center

    def _update_roi_rotation_points(self):
        """ROI boundary box의 현재 위치/크기/자세에 맞추어 3D 회전 기즈모(링) 위치 및 반경 갱신"""
        if not hasattr(self, "box_widget") or not hasattr(self, "roi_gizmo_actors"):
            return
        is_rotate_checked = hasattr(self, "chk_roi_rotate") and self.chk_roi_rotate.isChecked()
        is_box_visible = hasattr(self, "chk_roi_box") and self.chk_roi_box.isChecked()
        if not (is_rotate_checked and is_box_visible):
            for actor in self.roi_gizmo_actors.values():
                actor.SetVisibility(False)
            return

        corners, center = self._get_roi_box_corners_and_center()
        if center is None:
            return

        # 박스의 8개 꼭짓점으로부터 세 로컬 축 방향 벡터 계산
        vec_x = corners[1] - corners[0]
        vec_y = corners[3] - corners[0]
        vec_z = corners[4] - corners[0]
        norm_x = np.linalg.norm(vec_x)
        norm_y = np.linalg.norm(vec_y)
        norm_z = np.linalg.norm(vec_z)
        if norm_x < 1e-6 or norm_y < 1e-6 or norm_z < 1e-6:
            return

        dir_x = vec_x / norm_x
        dir_y = vec_y / norm_y
        dir_z = vec_z / norm_z

        # 박스의 대각선 길이를 기준으로 적절한 링 반경 계산
        dists = [np.linalg.norm(c - center) for c in corners]
        max_dist = max(dists) if dists else 10.0
        gizmo_radius = max(8.0, max_dist * 0.85)
        tube_radius = max(0.4, gizmo_radius * 0.035)

        # 박스의 자세 및 중심점, 스케일을 반영하는 4x4 행렬 구성
        mat = vtk.vtkMatrix4x4()
        cols = [gizmo_radius * dir_x, gizmo_radius * dir_y, gizmo_radius * dir_z]
        for row in range(3):
            for col in range(3):
                mat.SetElement(row, col, cols[col][row])
            mat.SetElement(row, 3, center[row])
        mat.SetElement(3, 3, 1.0)

        for axis, actor in self.roi_gizmo_actors.items():
            self.roi_gizmo_tube_filters[axis].SetRadius(tube_radius / gizmo_radius)
            actor.SetPosition(0.0, 0.0, 0.0)
            actor.SetScale(1.0, 1.0, 1.0)
            actor.SetUserMatrix(mat)
            actor.SetVisibility(True)

    def rotate_roi_box(self, axis: str, angle_deg: float):
        """ROI boundary box를 크기 왜곡 없이 중심점 기준으로 지정한 로컬 축(X, Y, Z)에 따라 angle_deg만큼 정확히 회전"""
        if not hasattr(self, "box_widget"):
            return

        corners, center = self._get_roi_box_corners_and_center()
        if center is None:
            return

        vec_x = corners[1] - corners[0]
        vec_y = corners[3] - corners[0]
        vec_z = corners[4] - corners[0]
        norm_x = np.linalg.norm(vec_x)
        norm_y = np.linalg.norm(vec_y)
        norm_z = np.linalg.norm(vec_z)
        if norm_x < 1e-6 or norm_y < 1e-6 or norm_z < 1e-6:
            return

        dir_x = vec_x / norm_x
        dir_y = vec_y / norm_y
        dir_z = vec_z / norm_z

        ax = axis.upper()
        if ax == "X":
            axis_vec = dir_x
        elif ax == "Y":
            axis_vec = dir_y
        elif ax == "Z":
            axis_vec = dir_z
        else:
            return

        current_trans = vtk.vtkTransform()
        self.box_widget.GetTransform(current_trans)

        rot_trans = vtk.vtkTransform()
        rot_trans.PostMultiply()
        rot_trans.Concatenate(current_trans)
        rot_trans.Translate(-center[0], -center[1], -center[2])
        rot_trans.RotateWXYZ(angle_deg, axis_vec[0], axis_vec[1], axis_vec[2])
        rot_trans.Translate(center[0], center[1], center[2])

        self._box_widget_updating = True
        self.box_widget.SetTransform(rot_trans)
        self._box_widget_updating = False

        self._on_box_widget_interaction(None, None)
        self._update_volume_cropping()
        self._update_roi_rotation_points()
        self.render()

    def _on_box_widget_interaction(self, obj, event):
        """3D 창에서 박스의 핸들을 드래그할 때 실시간 크롭 및 바운더리 갱신"""
        if self._box_widget_updating:
            return
        poly = vtk.vtkPolyData()
        self.box_widget.GetPolyData(poly)
        bounds = poly.GetBounds()
        self._box_widget_updating = True
        self.state_mgr.set_roi_bounds(bounds)
        self._box_widget_updating = False
        self._update_volume_cropping()
        self._update_roi_rotation_points()

    def _init_position_indicators(self):
        """2mm 직경 빨간 구 및 축 색상 긴 십자선 액터 초기화"""
        # 1. 빨간 구 (직경 2mm -> 반경 1mm)
        self._sphere_source = vtk.vtkSphereSource()
        self._sphere_source.SetRadius(1.0)
        self._sphere_source.SetThetaResolution(16)
        self._sphere_source.SetPhiResolution(16)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(self._sphere_source.GetOutputPort())

        self._sphere_actor = vtk.vtkActor()
        self._sphere_actor.SetMapper(sphere_mapper)
        self._sphere_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self._sphere_actor.GetProperty().SetAmbient(0.3)
        self._sphere_actor.GetProperty().SetDiffuse(0.8)
        self._sphere_actor.SetVisibility(False)
        self.renderer.AddActor(self._sphere_actor)

        # 2. 축 색상 긴 십자선 (X: 빨강, Y: 초록, Z: 파랑)
        self._cross_x_source = vtk.vtkLineSource()
        mapper_x = vtk.vtkPolyDataMapper()
        mapper_x.SetInputConnection(self._cross_x_source.GetOutputPort())
        self._cross_x_actor = vtk.vtkActor()
        self._cross_x_actor.SetMapper(mapper_x)
        self._cross_x_actor.GetProperty().SetColor(1.0, 0.2, 0.2)
        self._cross_x_actor.GetProperty().SetLineWidth(1.5)
        self._cross_x_actor.GetProperty().SetOpacity(0.125)
        self._cross_x_actor.GetProperty().LightingOff()
        self._cross_x_actor.SetVisibility(False)
        self.renderer.AddActor(self._cross_x_actor)

        self._cross_y_source = vtk.vtkLineSource()
        mapper_y = vtk.vtkPolyDataMapper()
        mapper_y.SetInputConnection(self._cross_y_source.GetOutputPort())
        self._cross_y_actor = vtk.vtkActor()
        self._cross_y_actor.SetMapper(mapper_y)
        self._cross_y_actor.GetProperty().SetColor(0.2, 1.0, 0.2)
        self._cross_y_actor.GetProperty().SetLineWidth(1.5)
        self._cross_y_actor.GetProperty().SetOpacity(0.125)
        self._cross_y_actor.GetProperty().LightingOff()
        self._cross_y_actor.SetVisibility(False)
        self.renderer.AddActor(self._cross_y_actor)

        self._cross_z_source = vtk.vtkLineSource()
        mapper_z = vtk.vtkPolyDataMapper()
        mapper_z.SetInputConnection(self._cross_z_source.GetOutputPort())
        self._cross_z_actor = vtk.vtkActor()
        self._cross_z_actor.SetMapper(mapper_z)
        self._cross_z_actor.GetProperty().SetColor(0.2, 0.5, 1.0)
        self._cross_z_actor.GetProperty().SetLineWidth(1.5)
        self._cross_z_actor.GetProperty().SetOpacity(0.125)
        self._cross_z_actor.GetProperty().LightingOff()
        self._cross_z_actor.SetVisibility(False)
        self.renderer.AddActor(self._cross_z_actor)

    def _update_position_indicators(self, x: float, y: float, z: float):
        if hasattr(self, "_sphere_source"):
            self._sphere_source.SetCenter(x, y, z)
            self._sphere_source.Update()
        if hasattr(self, "_cross_x_source"):
            span = 200.0
            self._cross_x_source.SetPoint1(x - span, y, z)
            self._cross_x_source.SetPoint2(x + span, y, z)
            self._cross_x_source.Update()

            self._cross_y_source.SetPoint1(x, y - span, z)
            self._cross_y_source.SetPoint2(x, y + span, z)
            self._cross_y_source.Update()

            self._cross_z_source.SetPoint1(x, y, z - span)
            self._cross_z_source.SetPoint2(x, y, z + span)
            self._cross_z_source.Update()

    def _init_orientation_marker(self):

        self.axes_actor = vtk.vtkAxesActor()
        self.axes_actor.SetShaftTypeToCylinder()
        self.axes_actor.SetXAxisLabelText("R")
        self.axes_actor.SetYAxisLabelText("A")
        self.axes_actor.SetZAxisLabelText("S")
        self.axes_actor.SetTotalLength(25, 25, 25)

        self.orient_widget = vtk.vtkOrientationMarkerWidget()
        self.orient_widget.SetOrientationMarker(self.axes_actor)
        self.orient_widget.SetInteractor(self.vtkWidget._Iren)
        self.orient_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
        self._orient_widget_enabled = False

    def _init_toolbar(self):
        toolbar = QToolBar("3D 제어 툴바", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel(" 렌더링 프리셋: ", self))
        self.preset_combo = QComboBox(self)
        self.preset_combo.addItems([
            "Bone (골조직 강조)",
            "Dental Enamel (고밀도 치아)",
            "Soft & Bone (연조직+골조직)",
            "MIP (최대강도투영)"
        ])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        toolbar.addWidget(self.preset_combo)

        toolbar.addSeparator()

        # 도구 버튼 공통 스타일
        btn_qss = """
            QToolButton {
                background-color: #2c2c34;
                border: 1px solid #444452;
                border-radius: 3px;
                padding: 2px;
            }
            QToolButton:hover {
                background-color: #3e3e4a;
                border-color: #61afef;
            }
            QToolButton:checked {
                background-color: #1e3a5f;
                border-color: #61afef;
            }
        """

        toolbar.addWidget(QLabel(" 시점: ", self))

        # 시점 목록: (라벨/툴팁, 방향벡터(6개), 큐브강조면)
        # Front: Anterior 정면 응시 (0, -1, 0, 0, 0, 1)
        # Rear: Posterior 후면 응시 (0, 1, 0, 0, 0, 1)
        # Left: 환자 좌측 응시 (-1, 0, 0, 0, 0, 1)
        # Right: 환자 우측 응시 (1, 0, 0, 0, 0, 1)
        # Top: 상단 응시 (0, 0, 1, 0, 1, 0)
        # Bottom: 하단 응시 (0, 0, -1, 0, -1, 0)
        # Isometric: 등각 투영 (1, -1, 1, 0, 0, 1)
        views = [
            ("정면 시점 (Front)", (0, -1, 0, 0, 0, 1), "front"),
            ("후면 시점 (Rear)", (0, 1, 0, 0, 0, 1), "rear"),
            ("좌측면 시점 (Left)", (-1, 0, 0, 0, 0, 1), "left"),
            ("우측면 시점 (Right)", (1, 0, 0, 0, 0, 1), "right"),
            ("상단 시점 (Top)", (0, 0, 1, 0, 1, 0), "top"),
            ("하단 시점 (Bottom)", (0, 0, -1, 0, -1, 0), "bottom"),
            ("등각 시점 (Isometric)", (1.0, -1.0, 1.0, 0.0, 0.0, 1.0), "iso")
        ]

        for tip, params, face_key in views:
            btn = QToolButton(self)
            btn.setIcon(IconFactory.view_cube(face_key, 22))
            btn.setIconSize(QSize(20, 20))
            btn.setFixedSize(28, 26)
            btn.setToolTip(tip)
            btn.setStyleSheet(btn_qss)
            btn.clicked.connect(lambda _, p=params: self.set_view_direction(*p))
            toolbar.addWidget(btn)

        toolbar.addSeparator()

        # 카메라 리셋 버튼 (아이콘 전용)
        self.btn_reset = QToolButton(self)
        self.btn_reset.setIcon(IconFactory.reset_camera(22))
        self.btn_reset.setIconSize(QSize(20, 20))
        self.btn_reset.setFixedSize(28, 26)
        self.btn_reset.setToolTip("카메라 위치 및 줌 리셋")
        self.btn_reset.setStyleSheet(btn_qss)
        self.btn_reset.clicked.connect(self.reset_camera)
        toolbar.addWidget(self.btn_reset)

        # 우측 여백 후 설정 아이콘 버튼
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding, spacer.sizePolicy().verticalPolicy().Preferred)
        toolbar.addWidget(spacer)

        self.btn_toggle_settings = QToolButton(self)
        self.btn_toggle_settings.setIcon(IconFactory.settings(22))
        self.btn_toggle_settings.setIconSize(QSize(20, 20))
        self.btn_toggle_settings.setFixedSize(28, 26)
        self.btn_toggle_settings.setCheckable(True)
        self.btn_toggle_settings.setChecked(True)
        self.btn_toggle_settings.setToolTip("3D 설정 사이드바 패널 열기/닫기")
        self.btn_toggle_settings.setStyleSheet(btn_qss)
        self.btn_toggle_settings.clicked.connect(self._toggle_settings_panel)
        toolbar.addWidget(self.btn_toggle_settings)

    def _create_settings_sidebar(self) -> QFrame:
        """우측에 고정 배치되는 콤팩트한 3D 설정 사이드바 패널 (너비 145px)"""
        panel = QFrame(self)
        panel.setFixedWidth(145)
        panel.setStyleSheet("""
            QFrame {
                background-color: #212126;
                border-left: 1px solid #3c3c46;
                color: #e0e0e0;
            }
            QCheckBox {
                font-size: 11px;
                spacing: 4px;
            }
            QLabel {
                font-size: 11px;
                color: #aaaaaa;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # 패널 타이틀
        lbl_panel_title = QLabel("3D 설정", panel)
        lbl_panel_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #61afef;")
        layout.addWidget(lbl_panel_title)

        # 1. 음영효과 체크박스
        self.chk_shading = QCheckBox("3D 음영 사용", panel)
        self.chk_shading.setChecked(True)
        self.chk_shading.toggled.connect(self._on_shading_toggled)
        layout.addWidget(self.chk_shading)

        # 2. 가로 투명도 슬라이더 바 (끝에 % 라벨)
        lbl_op_title = QLabel("투명도:", panel)
        layout.addWidget(lbl_op_title)

        slider_box = QHBoxLayout()
        slider_box.setSpacing(4)
        slider_box.setContentsMargins(0, 0, 0, 0)

        self.slider_opacity = QSlider(Qt.Orientation.Horizontal, panel)
        self.slider_opacity.setRange(5, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.valueChanged.connect(self._on_opacity_slider_changed)
        slider_box.addWidget(self.slider_opacity)

        self.lbl_opacity_val = QLabel("100%", panel)
        self.lbl_opacity_val.setFixedWidth(34)
        slider_box.addWidget(self.lbl_opacity_val)
        layout.addLayout(slider_box)

        # 구분선
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setStyleSheet("border-color: #383842; margin-top: 2px; margin-bottom: 2px;")
        layout.addWidget(line1)

        # 3. 3축 슬라이스 평면 가시성 체크박스 ('슬라이스 평면' 및 'plane')
        layout.addWidget(QLabel("슬라이스 평면", panel))

        self.chk_plane_axial = QCheckBox("Axial plane (Z)", panel)
        self.chk_plane_axial.toggled.connect(lambda c: self._set_plane_visible("axial", c))
        layout.addWidget(self.chk_plane_axial)

        self.chk_plane_coronal = QCheckBox("Coronal plane (Y)", panel)
        self.chk_plane_coronal.toggled.connect(lambda c: self._set_plane_visible("coronal", c))
        layout.addWidget(self.chk_plane_coronal)

        self.chk_plane_sagittal = QCheckBox("Sagittal plane (X)", panel)
        self.chk_plane_sagittal.toggled.connect(lambda c: self._set_plane_visible("sagittal", c))
        layout.addWidget(self.chk_plane_sagittal)

        # 구분선
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("border-color: #383842; margin-top: 2px; margin-bottom: 2px;")
        layout.addWidget(line2)

        # 4. 단면 보기 ('단면 보기' 글자 우측으로 가로 배치된 작은 버튼들)
        clip_row = QHBoxLayout()
        clip_row.setSpacing(2)
        clip_row.setContentsMargins(0, 0, 0, 0)

        lbl_clip = QLabel("단면 보기", panel)
        lbl_clip.setToolTip("좌클릭: 해당 평면 정방향 컷팅\n우클릭: 해당 평면 역방향 컷팅\n재클릭: 해제")
        lbl_clip.setStyleSheet("font-size: 11px; font-weight: 500;")
        clip_row.addWidget(lbl_clip)
        clip_row.addStretch()

        self.btn_clip_a = ClippingButton(IconFactory.letter_icon("A", 16), "Axial 컷팅 (좌:위 / 우:아래)", panel)
        self.btn_clip_a.setFixedSize(20, 20)
        self.btn_clip_a.left_clicked.connect(lambda: self._toggle_clip("A", +1))
        self.btn_clip_a.right_clicked.connect(lambda: self._toggle_clip("A", -1))
        clip_row.addWidget(self.btn_clip_a)

        self.btn_clip_c = ClippingButton(IconFactory.letter_icon("C", 16), "Coronal 컷팅 (좌:앞 / 우:뒤)", panel)
        self.btn_clip_c.setFixedSize(20, 20)
        self.btn_clip_c.left_clicked.connect(lambda: self._toggle_clip("C", +1))
        self.btn_clip_c.right_clicked.connect(lambda: self._toggle_clip("C", -1))
        clip_row.addWidget(self.btn_clip_c)

        self.btn_clip_s = ClippingButton(IconFactory.letter_icon("S", 16), "Sagittal 컷팅 (좌:좌 / 우:우)", panel)
        self.btn_clip_s.setFixedSize(20, 20)
        self.btn_clip_s.left_clicked.connect(lambda: self._toggle_clip("S", +1))
        self.btn_clip_s.right_clicked.connect(lambda: self._toggle_clip("S", -1))
        clip_row.addWidget(self.btn_clip_s)

        self.btn_clip_o = ClippingButton(
            IconFactory.letter_icon("O", 16, bg_color=QColor(30, 58, 95), text_color=QColor(97, 175, 239)),
            "전체보기 (컷팅 완전 해제)", panel
        )
        self.btn_clip_o.setFixedSize(20, 20)
        self.btn_clip_o.left_clicked.connect(self._reset_all_clipping)
        clip_row.addWidget(self.btn_clip_o)

        layout.addLayout(clip_row)

        # 구분선
        line3 = QFrame()
        line3.setFrameShape(QFrame.Shape.HLine)
        line3.setStyleSheet("border-color: #383842; margin-top: 2px; margin-bottom: 2px;")
        layout.addWidget(line3)

        # 5. ROI 그룹 (ROI 적용, boundary 설정, Box 보이기)
        roi_group = QGroupBox("ROI 설정", panel)
        roi_group.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                font-weight: bold;
                color: #e0e0e0;
                border: 1px solid #3c3c46;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 3px;
            }
        """)
        roi_layout = QVBoxLayout(roi_group)
        roi_layout.setContentsMargins(4, 4, 4, 4)
        roi_layout.setSpacing(4)

        self.chk_roi_apply = QCheckBox("ROI 적용", roi_group)
        self.chk_roi_apply.toggled.connect(self._on_roi_apply_toggled)
        roi_layout.addWidget(self.chk_roi_apply)

        self.btn_roi_boundary = QPushButton("boundary 설정", roi_group)
        self.btn_roi_boundary.setCheckable(True)
        self.btn_roi_boundary.setStyleSheet("""
            QPushButton {
                background-color: #2c2c34;
                color: #ffffff;
                border: 1px solid #444452;
                border-radius: 3px;
                padding: 3px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3e3e4a;
                border-color: #ffa500;
            }
            QPushButton:checked {
                background-color: #b35900;
                border-color: #ffa500;
                color: #ffffff;
            }
        """)
        self.btn_roi_boundary.toggled.connect(self._on_roi_boundary_btn_toggled)
        roi_layout.addWidget(self.btn_roi_boundary)

        # 3D 회전 기즈모 표시/숨김 체크박스
        self.chk_roi_rotate = QCheckBox("회전", roi_group)
        self.chk_roi_rotate.setChecked(False)
        self.chk_roi_rotate.setToolTip("체크 시 3D 뷰에 ROI 박스 회전용 3D 기즈모(링)를 표시합니다.")
        self.chk_roi_rotate.toggled.connect(self._on_roi_rotate_toggled)
        roi_layout.addWidget(self.chk_roi_rotate)

        # 원래대로 리셋 기능 체크박스
        self.chk_roi_reset = QCheckBox("원래대로", roi_group)
        self.chk_roi_reset.setChecked(False)
        self.chk_roi_reset.toggled.connect(self._on_roi_reset_toggled)
        roi_layout.addWidget(self.chk_roi_reset)

        self.chk_roi_box = QCheckBox("Box 보이기", roi_group)
        self.chk_roi_box.setChecked(True)
        self.chk_roi_box.toggled.connect(self._on_roi_box_toggled)
        roi_layout.addWidget(self.chk_roi_box)

        layout.addWidget(roi_group)

        # 6. 옵션 그룹 (선택 좌표 동기화, 현재 위치 표시, 현재 위치 크로스 선)
        opt_group = QGroupBox("옵션", panel)
        opt_group.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                font-weight: bold;
                color: #e0e0e0;
                border: 1px solid #3c3c46;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 3px;
            }
        """)
        opt_layout = QVBoxLayout(opt_group)
        opt_layout.setContentsMargins(4, 4, 4, 4)
        opt_layout.setSpacing(4)

        self.chk_sync_coords = QCheckBox("선택 좌표 동기화", opt_group)
        self.chk_sync_coords.setChecked(True)
        self.chk_sync_coords.setToolTip("임플란트를 선택했을 때 해당 위치 좌표를 2D/3D 모든 창과 동조합니다.")
        opt_layout.addWidget(self.chk_sync_coords)

        self.chk_show_sphere = QCheckBox("현재 위치 표시", opt_group)
        self.chk_show_sphere.setChecked(False)
        self.chk_show_sphere.setToolTip("선택된 좌표 위치에 직경 2mm의 빨간색 구를 표시합니다.")
        self.chk_show_sphere.toggled.connect(self._on_show_sphere_toggled)
        opt_layout.addWidget(self.chk_show_sphere)

        self.chk_show_crosslines = QCheckBox("현재 위치 크로스 선", opt_group)
        self.chk_show_crosslines.setChecked(False)
        self.chk_show_crosslines.setToolTip("선택된 좌표를 중심으로 축 색상의 긴 십자선을 표시합니다.")
        self.chk_show_crosslines.toggled.connect(self._on_show_crosslines_toggled)
        opt_layout.addWidget(self.chk_show_crosslines)

        self.chk_implant_selectable = QCheckBox("임플란트 선택 가능", opt_group)
        self.chk_implant_selectable.setChecked(True)
        self.chk_implant_selectable.setToolTip("3D 창에서 임플란트를 마우스로 클릭하여 선택 및 조작할 수 있도록 허용합니다.")
        self.chk_implant_selectable.toggled.connect(self._on_implant_selectable_toggled)
        opt_layout.addWidget(self.chk_implant_selectable)

        layout.addWidget(opt_group)
        layout.addStretch()

        return panel

    def _on_show_sphere_toggled(self, checked: bool):
        if hasattr(self, "_sphere_actor"):
            self._sphere_actor.SetVisibility(checked)
            if checked:
                self._update_position_indicators(*self.state_mgr.current_coords)
            self.render()

    def _on_show_crosslines_toggled(self, checked: bool):
        if hasattr(self, "_cross_x_actor"):
            self._cross_x_actor.SetVisibility(checked)
            self._cross_y_actor.SetVisibility(checked)
            self._cross_z_actor.SetVisibility(checked)
            if checked:
                self._update_position_indicators(*self.state_mgr.current_coords)
            self.render()

    def _on_implant_selectable_toggled(self, checked: bool):
        """임플란트 선택 가능/불가능 토글 슬롯"""
        if not checked:
            if hasattr(self, "implant_gizmo_3d"):
                self.implant_gizmo_3d.detach()
            if hasattr(self.state_mgr, "implant_placement_mgr") and self.state_mgr.implant_placement_mgr:
                for imp in self.state_mgr.implant_placement_mgr.implants.values():
                    if imp.axes_assembly:
                        imp.axes_assembly.SetVisibility(False)
            self.render()
        else:
            sel_imp = self.state_mgr.implant_placement_mgr.get_selected_implant()
            if sel_imp and hasattr(self, "implant_gizmo_3d") and not getattr(self, "is_result_view_mode", False):
                self.implant_gizmo_3d.attach_to_implant(sel_imp.id)
                if sel_imp.axes_assembly:
                    sel_imp.axes_assembly.SetVisibility(True)
            self.render()

    def set_result_view_mode(self, enabled: bool):
        """
        해석 결과 3D 뷰 모드 설정.
        enabled=True: 임플란트 선택 체크박스를 해제하고 기즈모 및 축선을 숨김.
        enabled=False: 원래 상태로 복구(임플란트 선택 체크박스를 다시 체크하고 기즈모 및 축선 복구).
        """
        self.is_result_view_mode = enabled
        if hasattr(self, "chk_implant_selectable"):
            self.chk_implant_selectable.blockSignals(True)
            self.chk_implant_selectable.setChecked(not enabled)
            self.chk_implant_selectable.blockSignals(False)

        if enabled:
            if hasattr(self, "implant_gizmo_3d"):
                self.implant_gizmo_3d.detach()
            if hasattr(self.state_mgr, "implant_placement_mgr") and self.state_mgr.implant_placement_mgr:
                for imp in self.state_mgr.implant_placement_mgr.implants.values():
                    if imp.axes_assembly:
                        imp.axes_assembly.SetVisibility(False)
            self.render()
        else:
            sel_imp = self.state_mgr.implant_placement_mgr.get_selected_implant()
            if sel_imp and hasattr(self, "implant_gizmo_3d"):
                self.implant_gizmo_3d.attach_to_implant(sel_imp.id)
                if sel_imp.axes_assembly:
                    sel_imp.axes_assembly.SetVisibility(True)
            self.render()


    def _toggle_settings_panel(self, checked: bool):
        self.panel.setVisible(checked)

    def _load_saved_settings(self):
        shading = self.settings_mgr.get("volume_3d.shading", True)
        self.chk_shading.setChecked(shading)
        self.volume_property.SetShade(shading)

        opacity_val = int(self.settings_mgr.get("volume_3d.opacity", 1.0) * 100)
        self.slider_opacity.setValue(opacity_val)
        self.lbl_opacity_val.setText(f"{opacity_val}%")

        # 윈도우 위치/크기 복원
        geom = self.settings_mgr.get("volume_3d.geometry")
        if geom:
            try:
                from PySide6.QtCore import QByteArray
                self.restoreGeometry(QByteArray.fromHex(geom.encode("utf-8")))
            except Exception:
                pass

    def _on_roi_apply_toggled(self, checked: bool):
        self._update_volume_cropping()

    def _on_roi_boundary_btn_toggled(self, checked: bool):
        # 2D 뷰에 사각형 그리기 모드 알림
        self.state_mgr.roi_setting_mode_toggled.emit(checked)
        if checked:
            self.statusBar().showMessage("2D 뷰에서 사각형을 그리거나, 3D 창에서 박스의 선분/면을 드래그하여 크기를 조절하세요.")
            # 3D 상에 ROI bounds가 없으면 전체 볼륨 bounds로 초기화
            if self.state_mgr.roi_bounds is None and self.state_mgr.vtk_image is not None:
                self.state_mgr.set_roi_bounds(self.state_mgr.bounds)
        else:
            self.statusBar().showMessage("ROI boundary 설정 모드가 종료되었습니다.")
        self._update_roi_box_visibility()
        self._update_roi_rotation_points()

    def _on_roi_rotate_toggled(self, checked: bool):
        """회전 체크박스 토글 시 3D 회전 기즈모 표시/숨김 갱신"""
        self._update_roi_rotation_points()
        self.render()
        if checked:
            self.statusBar().showMessage("ROI 3D 회전 기즈모가 활성화되었습니다. 링을 클릭/드래그하여 회전하세요.")
        else:
            self.statusBar().showMessage("ROI 3D 회전 기즈모가 숨겨졌습니다.")

    def _on_roi_reset_toggled(self, checked: bool):
        if checked:
            if self.state_mgr.vtk_image is not None:
                full_bounds = self.state_mgr.bounds
                self._box_widget_updating = True
                try:
                    self.box_widget.PlaceWidget(full_bounds)
                    t_identity = vtk.vtkTransform()
                    self.box_widget.SetTransform(t_identity)
                except Exception:
                    pass
                self.state_mgr.set_roi_bounds(full_bounds)
                self._box_widget_updating = False
                self._update_volume_cropping()
                self._update_roi_box_visibility()
                self._update_roi_rotation_points()
                self.render()
                self.statusBar().showMessage("ROI boundary가 전체 볼륨 원본 크기 및 각도로 복원되었습니다.")
            self.chk_roi_reset.setChecked(False)

    def _on_roi_box_toggled(self, checked: bool):
        self.state_mgr.roi_box_visibility_changed.emit(checked)
        self._update_roi_box_visibility()

    def _on_roi_bounds_updated(self, bounds):
        """2D 뷰 또는 3D 위젯에서 사각형이 조절되어 새로운 바운더리가 설정되었을 때 호출"""
        if bounds is not None:
            if hasattr(self, "roi_box_source"):
                self.roi_box_source.SetBounds(bounds)
            if hasattr(self, "box_widget") and not self._box_widget_updating:
                try:
                    t_identity = vtk.vtkTransform()
                    self.box_widget.SetTransform(t_identity)
                    self.box_widget.PlaceWidget(bounds)
                except Exception:
                    pass
            # 사각형이 그려지거나 조절되면 자동으로 ROI 적용 활성화
            if not self.chk_roi_apply.isChecked():
                self.chk_roi_apply.setChecked(True)
            else:
                self._update_volume_cropping()
        else:
            self._update_volume_cropping()
        self._update_roi_box_visibility()

    def _update_volume_cropping(self):
        """VTK Volume Cropping 적용/해제 (회전된 OBB 육면체 평면 클리핑 반영)"""
        if self.state_mgr.vtk_image is None:
            return

        apply_roi = self.chk_roi_apply.isChecked()
        bounds = self.state_mgr.roi_bounds

        if apply_roi and bounds is not None:
            self.volume_mapper.SetCropping(1)
            self.volume_mapper.SetCroppingRegionPlanes(
                bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
            )
            self.volume_mapper.SetCroppingRegionFlagsToSubVolume()
        else:
            self.volume_mapper.SetCropping(0)

        self._apply_clipping_plane()
        self._update_roi_box_visibility()
        self.render()

    def _update_roi_box_visibility(self):
        show_box = self.chk_roi_box.isChecked()
        is_boundary_mode = self.btn_roi_boundary.isChecked()
        has_bounds = self.state_mgr.roi_bounds is not None
        has_volume = self.state_mgr.vtk_image is not None

        if show_box and is_boundary_mode and has_bounds and has_volume:
            # boundary 설정 모드에서는 3D 상에서 직접 핸들로 크기 조절할 수 있도록 box_widget 활성화
            try:
                if not self.box_widget.GetEnabled():
                    self.box_widget.PlaceWidget(self.state_mgr.roi_bounds)
                    self.box_widget.On()
                self._disable_box_widget_outline_picking()
            except Exception:
                pass
            self.roi_box_actor.SetVisibility(False)
        elif show_box and has_bounds and has_volume:
            try:
                self.box_widget.Off()
            except Exception:
                pass
            self.roi_box_actor.SetVisibility(True)
        else:
            try:
                self.box_widget.Off()
            except Exception:
                pass
            self.roi_box_actor.SetVisibility(False)

        self._update_roi_rotation_points()
        self.render()


    def _on_shading_toggled(self, checked: bool):
        self.volume_property.SetShade(checked)
        self.settings_mgr.set("volume_3d.shading", checked)
        self.render()

    def _on_opacity_slider_changed(self, value: int):
        self.lbl_opacity_val.setText(f"{value}%")
        scale = value / 100.0
        self.settings_mgr.set("volume_3d.opacity", scale)
        self._rescale_opacity(scale)
        self.render()

    def _rescale_opacity(self, scale: float):
        """기본 골조직 불투명도 곡선에 스케일 배율 적용"""
        self.opacity_func.RemoveAllPoints()
        self.opacity_func.AddPoint(-1000.0, 0.0)
        self.opacity_func.AddPoint(180.0, 0.0)
        self.opacity_func.AddPoint(350.0, 0.20 * scale)
        self.opacity_func.AddPoint(750.0, 0.65 * scale)
        self.opacity_func.AddPoint(1500.0, 0.85 * scale)
        self.opacity_func.AddPoint(3000.0, 0.95 * scale)

    def _set_plane_visible(self, plane_type: str, visible: bool):
        has_volume = self.state_mgr.vtk_image is not None
        if plane_type == "axial":
            self.axial_plane_actor.SetVisibility(visible and has_volume)
        elif plane_type == "coronal":
            self.coronal_plane_actor.SetVisibility(visible and has_volume)
        elif plane_type == "sagittal":
            self.sagittal_plane_actor.SetVisibility(visible and has_volume)

        if has_volume:
            self._update_plane_coordinates(*self.state_mgr.current_coords)
        self.render()

    def _update_plane_coordinates(self, x: float, y: float, z: float):
        """현재 십자선 월드 좌표로 3D 슬라이스 평면 지오메트리 갱신"""
        bx0, bx1, by0, by1, bz0, bz1 = self.state_mgr.bounds

        # Axial Plane: XY 평면, Z = z
        if self.axial_plane_actor.GetVisibility():
            self.axial_plane_source.SetOrigin(bx0, by0, z)
            self.axial_plane_source.SetPoint1(bx1, by0, z)
            self.axial_plane_source.SetPoint2(bx0, by1, z)
            self.axial_plane_source.Update()

        # Coronal Plane: XZ 평면, Y = y
        if self.coronal_plane_actor.GetVisibility():
            self.coronal_plane_source.SetOrigin(bx0, y, bz0)
            self.coronal_plane_source.SetPoint1(bx1, y, bz0)
            self.coronal_plane_source.SetPoint2(bx0, y, bz1)
            self.coronal_plane_source.Update()

        # Sagittal Plane: YZ 평면, X = x
        if self.sagittal_plane_actor.GetVisibility():
            self.sagittal_plane_source.SetOrigin(x, by0, bz0)
            self.sagittal_plane_source.SetPoint1(x, by1, bz0)
            self.sagittal_plane_source.SetPoint2(x, by0, bz1)
            self.sagittal_plane_source.Update()

    def on_coordinates_changed(self, x: float, y: float, z: float):
        """2D MPR에서 클릭/이동된 월드 좌표에 맞춰 3D 슬라이스 평면 및 위치 표시자 위치 동조"""
        if self.state_mgr.vtk_image is None or not self.isVisible():
            return

        # 3D 슬라이스 평면이 하나라도 켜져 있는지 확인
        has_visible_plane = (
            self.axial_plane_actor.GetVisibility()
            or self.coronal_plane_actor.GetVisibility()
            or self.sagittal_plane_actor.GetVisibility()
        )
        has_active_clip = (self._current_clip_mode != "O")
        has_sphere = hasattr(self, "_sphere_actor") and self._sphere_actor.GetVisibility()
        has_cross = hasattr(self, "_cross_x_actor") and self._cross_x_actor.GetVisibility()

        if not has_visible_plane and not has_active_clip and not has_sphere and not has_cross:
            return

        if has_visible_plane:
            self._update_plane_coordinates(x, y, z)

        if has_sphere or has_cross:
            self._update_position_indicators(x, y, z)

        # 만약 활성화된 클리핑 플레인이 있다면 위치 업데이트
        if has_active_clip:
            self._apply_clipping_plane()

        # 마우스 연속 드래그 시 3D 렌더링 스로틀링 적용
        self.request_render()

    def _toggle_clip(self, axis_char: str, sign: int):
        """
        A, C, S 평면 클리핑 토글.
        sign == +1: 정방향, sign == -1: 역방향.
        이미 같은 모드면 해제.
        """
        target_mode = f"{axis_char}{'+' if sign > 0 else '-'}"
        if self._current_clip_mode == target_mode:
            self._reset_all_clipping()
            return

        self._current_clip_mode = target_mode
        self._apply_clipping_plane()

        # 버튼 강조 스타일 갱신
        self._update_clip_button_styles()

    def _apply_clipping_plane(self):
        """ROI 육면체 클리핑 평면 및 단면 컷팅 평면 일괄 적용"""
        self.volume_mapper.RemoveAllClippingPlanes()

        # 1. ROI 적용:
        # 직교 ROI인 경우 SetCropping(1) 및 SetCroppingRegionPlanes로 하드웨어 레벨에서 정확하게 크롭합니다.
        # 박스가 실제로 회전된 OBB 상태일 때만 box_widget 6평면 클리핑을 적용합니다.
        apply_roi = hasattr(self, "chk_roi_apply") and self.chk_roi_apply.isChecked()
        bounds = self.state_mgr.roi_bounds

        is_rotated = False
        if hasattr(self, "box_widget"):
            t = vtk.vtkTransform()
            self.box_widget.GetTransform(t)
            m = t.GetMatrix()
            for r in range(3):
                for c in range(3):
                    exp = 1.0 if r == c else 0.0
                    if abs(m.GetElement(r, c) - exp) > 1e-3:
                        is_rotated = True
                        break
                if is_rotated:
                    break

        if apply_roi and bounds is not None and is_rotated and hasattr(self, "box_widget"):
            planes = vtk.vtkPlanes()
            self.box_widget.GetPlanes(planes)

            # VTK SmartVolumeMapper는 평면의 음의 반공간을 클리핑하므로, 안쪽을 향하도록 법선 반전
            planes_inv = vtk.vtkPlanes()
            planes_inv.SetPoints(planes.GetPoints())
            norms = planes.GetNormals()
            if norms and norms.GetNumberOfTuples() > 0:
                inv_norms = vtk.vtkDoubleArray()
                inv_norms.SetNumberOfComponents(3)
                inv_norms.SetNumberOfTuples(norms.GetNumberOfTuples())
                for i in range(norms.GetNumberOfTuples()):
                    n = norms.GetTuple(i)
                    inv_norms.SetTuple(i, (-n[0], -n[1], -n[2]))
                planes_inv.SetNormals(inv_norms)
                self.volume_mapper.SetClippingPlanes(planes_inv)

        # 2. 축 단면 컷팅 평면 추가 (A, C, S)
        if self._current_clip_mode != "O":
            x, y, z = self.state_mgr.current_coords
            axis_char = self._current_clip_mode[0]
            sign = +1 if self._current_clip_mode[1] == '+' else -1

            clip_plane = vtk.vtkPlane()
            if axis_char == "A":
                clip_plane.SetOrigin(x, y, z)
                clip_plane.SetNormal(0.0, 0.0, -1.0 * sign)
            elif axis_char == "C":
                clip_plane.SetOrigin(x, y, z)
                clip_plane.SetNormal(0.0, -1.0 * sign, 0.0)
            elif axis_char == "S":
                clip_plane.SetOrigin(x, y, z)
                clip_plane.SetNormal(-1.0 * sign, 0.0, 0.0)

            self.volume_mapper.AddClippingPlane(clip_plane)

        self.render()

    def _reset_all_clipping(self):
        self._current_clip_mode = "O"
        self._apply_clipping_plane()
        self._update_clip_button_styles()

    def _update_clip_button_styles(self):
        def style(btn, active):
            if active:
                btn.setStyleSheet("""
                    QToolButton {
                        background-color: #1e3a5f;
                        border: 2px solid #61afef;
                        border-radius: 4px;
                        padding: 2px;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QToolButton {
                        background-color: #2c2c34;
                        border: 1px solid #4a4a58;
                        border-radius: 4px;
                        padding: 2px;
                    }
                    QToolButton:hover {
                        background-color: #3d3d4a;
                        border-color: #61afef;
                    }
                """)

        style(self.btn_clip_a, self._current_clip_mode.startswith("A"))
        style(self.btn_clip_c, self._current_clip_mode.startswith("C"))
        style(self.btn_clip_s, self._current_clip_mode.startswith("S"))

    def on_volume_loaded(self, vtk_image: vtk.vtkImageData):
        self.volume_mapper.SetInputData(vtk_image)
        self.volume.SetVisibility(True)

        # 1. 3D 음영사용 및 투명도 설정 적용
        shading = self.chk_shading.isChecked()
        self.volume_property.SetShade(shading)
        opacity_scale = self.slider_opacity.value() / 100.0

        # 2. 기본 골조직 프리셋 항상 적용 (원상 복구)
        self.preset_combo.setCurrentIndex(0)
        self.apply_bone_preset()
        self._rescale_opacity(opacity_scale)

        # 3. 슬라이스 평면 액터 초기화 (기본 꺼짐)
        self.chk_plane_axial.setChecked(False)
        self.chk_plane_coronal.setChecked(False)
        self.chk_plane_sagittal.setChecked(False)
        self._set_plane_visible("axial", False)
        self._set_plane_visible("coronal", False)
        self._set_plane_visible("sagittal", False)

        # 4. ROI 설정 초기화 (기존 사용자가 설정한 바운더리가 없을 때만 None으로 초기화)
        bounds = self.state_mgr.roi_bounds
        self.chk_roi_apply.setChecked(bounds is not None)
        self.chk_roi_box.setChecked(True)
        if bounds is not None:
            if hasattr(self, "roi_box_source"):
                self.roi_box_source.SetBounds(bounds)
            if hasattr(self, "box_widget"):
                try:
                    t_identity = vtk.vtkTransform()
                    self.box_widget.SetTransform(t_identity)
                    self.box_widget.PlaceWidget(bounds)
                except Exception:
                    pass
        else:
            self.state_mgr.set_roi_bounds(None)
        self._update_volume_cropping()
        self._update_roi_box_visibility()

        self.on_coordinates_changed(*self.state_mgr.current_coords)
        # 초기 오픈 시 요구사항에 따라 항상 정면 시점(Front View)으로 설정
        self.set_view_direction(0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
        self.render()

    def apply_bone_preset(self):

        self.volume_mapper.SetBlendModeToComposite()
        self.color_func.RemoveAllPoints()
        self.color_func.AddRGBPoint(-1000.0, 0.0, 0.0, 0.0)
        self.color_func.AddRGBPoint(150.0, 0.6, 0.25, 0.15)
        self.color_func.AddRGBPoint(400.0, 0.88, 0.78, 0.65)
        self.color_func.AddRGBPoint(800.0, 0.95, 0.92, 0.82)
        self.color_func.AddRGBPoint(1600.0, 1.0, 1.0, 0.95)
        self.color_func.AddRGBPoint(3000.0, 1.0, 1.0, 1.0)
        self._rescale_opacity(self.slider_opacity.value() / 100.0)

    def apply_enamel_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.color_func.RemoveAllPoints()
        self.color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        self.color_func.AddRGBPoint(800.0, 0.8, 0.8, 0.8)
        self.color_func.AddRGBPoint(1400.0, 1.0, 0.98, 0.92)
        self.color_func.AddRGBPoint(3000.0, 1.0, 1.0, 1.0)
        scale = self.slider_opacity.value() / 100.0
        self.opacity_func.RemoveAllPoints()
        self.opacity_func.AddPoint(600.0, 0.0)
        self.opacity_func.AddPoint(900.0, 0.3 * scale)
        self.opacity_func.AddPoint(1400.0, 0.85 * scale)
        self.opacity_func.AddPoint(3000.0, 0.95 * scale)

    def apply_soft_bone_preset(self):
        self.volume_mapper.SetBlendModeToComposite()
        self.color_func.RemoveAllPoints()
        self.color_func.AddRGBPoint(-1000.0, 0.0, 0.0, 0.0)
        self.color_func.AddRGBPoint(40.0, 0.8, 0.3, 0.3)
        self.color_func.AddRGBPoint(300.0, 0.85, 0.75, 0.65)
        self.color_func.AddRGBPoint(1200.0, 1.0, 1.0, 0.9)
        scale = self.slider_opacity.value() / 100.0
        self.opacity_func.RemoveAllPoints()
        self.opacity_func.AddPoint(-100.0, 0.0)
        self.opacity_func.AddPoint(40.0, 0.12 * scale)
        self.opacity_func.AddPoint(250.0, 0.25 * scale)
        self.opacity_func.AddPoint(800.0, 0.75 * scale)

    def apply_mip_preset(self):
        self.volume_mapper.SetBlendModeToMaximumIntensity()
        self.color_func.RemoveAllPoints()
        self.color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        self.color_func.AddRGBPoint(1500.0, 1.0, 1.0, 1.0)
        self.opacity_func.RemoveAllPoints()
        self.opacity_func.AddPoint(0.0, 0.0)
        self.opacity_func.AddPoint(1500.0, 1.0)

    def _on_preset_changed(self, index: int):
        if index == 0:
            self.apply_bone_preset()
        elif index == 1:
            self.apply_enamel_preset()
        elif index == 2:
            self.apply_soft_bone_preset()
        elif index == 3:
            self.apply_mip_preset()
        self.render()

    def set_view_direction(self, dir_x: float, dir_y: float, dir_z: float, up_x: float, up_y: float, up_z: float):
        bounds = self.state_mgr.bounds
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])
        dist = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2.5
        if dist <= 0:
            dist = 400.0

        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx + dir_x * dist, cy + dir_y * dist, cz + dir_z * dist)
        cam.SetViewUp(up_x, up_y, up_z)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def set_isometric_view(self):
        self.set_view_direction(1.0, -1.0, 1.0, 0.0, 0.0, 1.0)

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.render()

    def request_render(self):
        """3D 렌더링 요청을 스로틀링하여 UI 반응성을 확보합니다."""
        if not self.isVisible() or self.state_mgr.vtk_image is None:
            return
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _do_render(self):
        """실제 VTK RenderWindow 렌더링 실행"""
        if not self.isVisible() or self.state_mgr.vtk_image is None:
            return
        if self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()
            self.vtkWidget.update()

    def render(self):
        """즉시 3D 렌더링을 실행합니다."""
        if self._render_timer.isActive():
            self._render_timer.stop()
        self._do_render()

    def showEvent(self, event):
        super().showEvent(event)
        try:
            self.vtkWidget.Initialize()
        except Exception:
            pass
        if not self._orient_widget_enabled:
            try:
                self.orient_widget.SetEnabled(1)
                self.orient_widget.InteractiveOff()
                self._orient_widget_enabled = True
            except Exception:
                pass
        if self.state_mgr.vtk_image is not None:
            self._update_volume_cropping()
            self.render()

    def closeEvent(self, event):
        """창이 닫힐 때 설정 저장 및 시그널을 안전하게 해제하여 재오픈 시 충돌 방지"""
        try:
            geom = bytes(self.saveGeometry().toHex()).decode("utf-8")
            self.settings_mgr.set("volume_3d.geometry", geom, auto_save=True)
        except Exception:
            pass

        try:
            self.state_mgr.volume_loaded.disconnect(self.on_volume_loaded)
        except Exception:
            pass
        try:
            self.state_mgr.coordinates_changed.disconnect(self.on_coordinates_changed)
        except Exception:
            pass
        try:
            self.state_mgr.roi_bounds_changed.disconnect(self._on_roi_bounds_updated)
        except Exception:
            pass
        try:
            self.state_mgr.segmentation_result_changed.disconnect(self._on_segmentation_result_changed)
        except Exception:
            pass
        try:
            self.state_mgr.segmentation_visibility_changed.disconnect(self._on_segmentation_visibility_changed)
        except Exception:
            pass
        try:
            self.state_mgr.implant_model_changed.disconnect(self._on_implant_model_changed)
        except Exception:
            pass
        try:
            self.state_mgr.implant_pose_changed.disconnect(self._on_implant_pose_changed)
        except Exception:
            pass
        try:
            if self._orient_widget_enabled and self.orient_widget:
                self.orient_widget.SetEnabled(0)
                self._orient_widget_enabled = False
        except Exception:
            pass

        try:
            self.state_mgr.implant_selected.disconnect(self._on_implant_selected_3d)
        except Exception:
            pass
        try:
            self.state_mgr.implant_transformed.disconnect(self._on_implant_transformed_3d)
        except Exception:
            pass

        try:
            self.state_mgr.implant_placement_mgr.set_renderer_3d(None)
        except Exception:
            pass

        self.window_closed.emit()
        super().closeEvent(event)

    def eventFilter(self, obj, event):
        """3D VTK 위젯 마우스 이벤트를 가로채어 3D 임플란트 기즈모 조작 및 선택 처리"""
        if obj == self.vtkWidget:
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    pos = event.position().toPoint()
                    screen_x = pos.x()
                    screen_y = self.vtkWidget.height() - pos.y()

                    # 0) 3D ROI 회전 기즈모(링) 피킹 체크 (boundary 설정 모드일 때)
                    if hasattr(self, "btn_roi_boundary") and self.btn_roi_boundary.isChecked() and hasattr(self, "roi_gizmo_actors"):
                        is_any_gizmo_visible = any(a.GetVisibility() for a in self.roi_gizmo_actors.values())
                        if is_any_gizmo_visible:
                            picker = vtk.vtkPropPicker()
                            if picker.Pick(screen_x, screen_y, 0, self.renderer):
                                hit_actor = picker.GetActor()
                                for axis, actor in self.roi_gizmo_actors.items():
                                    if hit_actor == actor:
                                        self._roi_dragging_axis = axis
                                        self.statusBar().showMessage(f"ROI {axis}축 회전 기즈모를 드래그 중입니다...")
                                        self._roi_drag_start_x = screen_x
                                        self._roi_drag_start_y = screen_y
                                        # 활성 축 하이라이트
                                        for ax_key, act in self.roi_gizmo_actors.items():
                                            c = self._roi_gizmo_highlight_colors[ax_key] if ax_key == axis else self._roi_gizmo_base_colors[ax_key]
                                            act.GetProperty().SetColor(c[0], c[1], c[2])
                                        self.render()
                                        return True

                    # 1) 3D 기즈모 핸들 조작 체크
                    can_select = (not getattr(self, "is_result_view_mode", False)) and (
                        not hasattr(self, "chk_implant_selectable") or self.chk_implant_selectable.isChecked()
                    )
                    if can_select:
                        if hasattr(self, "implant_gizmo_3d") and self.implant_gizmo_3d.is_attached:
                            if self.implant_gizmo_3d.start_drag(None, screen_x, screen_y, self.renderer):
                                self.render()
                                return True

                        # 2) 3D 임플란트 피킹 체크
                        hit_imp = self.state_mgr.implant_placement_mgr.find_implant_by_screen_ray(self.renderer, screen_x, screen_y)
                        if hit_imp:
                            self.state_mgr.implant_placement_mgr.select_implant(hit_imp.id, multi_select=False)
                            self.implant_gizmo_3d.attach_to_implant(hit_imp.id)
                            if hasattr(self, "chk_sync_coords") and self.chk_sync_coords.isChecked():
                                self.state_mgr.set_coordinates(float(hit_imp.position[0]), float(hit_imp.position[1]), float(hit_imp.position[2]))
                            self.render()
                            return True
                        elif hasattr(self, "implant_gizmo_3d") and self.implant_gizmo_3d.is_attached:
                            # 빈 영역 클릭 시 기즈모 해제
                            self.state_mgr.implant_placement_mgr.clear_selection()
                            self.implant_gizmo_3d.detach()
                            self.render()

                elif event.button() == Qt.MouseButton.RightButton:
                    # 해석 결과 뷰 모드일 때는 우클릭 팝업 메뉴를 띄우지 않고 기본 카메라 줌 등으로 전달
                    if getattr(self, "is_result_view_mode", False):
                        return super().eventFilter(obj, event)

                    pos = event.position().toPoint()
                    screen_x = pos.x()
                    screen_y = self.vtkWidget.height() - pos.y()
                    hit_imp = self.state_mgr.implant_placement_mgr.find_implant_by_screen_ray(self.renderer, screen_x, screen_y)
                    if not hit_imp and self.state_mgr.implant_placement_mgr.selected_ids:
                        hit_imp = self.state_mgr.implant_placement_mgr.get_selected_implant()

                    if hit_imp:
                        self.state_mgr.implant_placement_mgr.select_implant(hit_imp.id, multi_select=False)
                        self.implant_gizmo_3d.attach_to_implant(hit_imp.id)
                        self.render()

                    self._show_implant_context_menu(event.globalPosition().toPoint(), hit_imp)
                    return True

            elif etype == QEvent.Type.MouseMove:
                pos = event.position().toPoint()
                screen_x = pos.x()
                screen_y = self.vtkWidget.height() - pos.y()

                # ROI 회전 조절점 드래그 중인 경우
                if getattr(self, "_roi_dragging_axis", None) is not None:
                    dx = screen_x - self._roi_drag_start_x
                    dy = screen_y - self._roi_drag_start_y
                    delta_angle = (dx + dy) * 0.4
                    if abs(delta_angle) >= 0.4:
                        self.rotate_roi_box(self._roi_dragging_axis, delta_angle)
                        self._roi_drag_start_x = screen_x
                        self._roi_drag_start_y = screen_y
                    return True

                # ROI 기즈모 마우스 호버 하이라이트 체크
                if hasattr(self, "btn_roi_boundary") and self.btn_roi_boundary.isChecked() and hasattr(self, "roi_gizmo_actors"):
                    is_any_gizmo_visible = any(a.GetVisibility() for a in self.roi_gizmo_actors.values())
                    if is_any_gizmo_visible and not (hasattr(self, "implant_gizmo_3d") and self.implant_gizmo_3d.is_dragging):
                        picker = vtk.vtkPropPicker()
                        hovered_axis = None
                        if picker.Pick(screen_x, screen_y, 0, self.renderer):
                            hit = picker.GetActor()
                            for ax_key, act in self.roi_gizmo_actors.items():
                                if hit == act:
                                    hovered_axis = ax_key
                                    break
                        changed = False
                        for ax_key, act in self.roi_gizmo_actors.items():
                            c = self._roi_gizmo_highlight_colors[ax_key] if ax_key == hovered_axis else self._roi_gizmo_base_colors[ax_key]
                            old_c = act.GetProperty().GetColor()
                            if old_c != c:
                                act.GetProperty().SetColor(c[0], c[1], c[2])
                                changed = True
                        if changed:
                            self.render()

                if hasattr(self, "implant_gizmo_3d") and self.implant_gizmo_3d.is_dragging:
                    if self.implant_gizmo_3d.update_drag(screen_x, screen_y, self.renderer):
                        self.render()
                    return True
                elif hasattr(self, "implant_gizmo_3d") and self.implant_gizmo_3d.is_attached:
                    if self.implant_gizmo_3d.highlight_handle(None, screen_x, screen_y, self.renderer):
                        self.render()

            elif etype == QEvent.Type.MouseButtonRelease:
                if getattr(self, "_roi_dragging_axis", None) is not None:
                    self.statusBar().showMessage(f"ROI {self._roi_dragging_axis}축 회전이 완료되었습니다.")
                    self._roi_dragging_axis = None
                    if hasattr(self, "roi_gizmo_actors"):
                        for ax_key, act in self.roi_gizmo_actors.items():
                            c = self._roi_gizmo_base_colors[ax_key]
                            act.GetProperty().SetColor(c[0], c[1], c[2])
                        self.render()
                    return True

                if event.button() == Qt.MouseButton.LeftButton:
                    if hasattr(self, "implant_gizmo_3d") and self.implant_gizmo_3d.is_dragging:
                        self.implant_gizmo_3d.end_drag()
                        self.render()
                        return True

        return super().eventFilter(obj, event)

    def _show_implant_context_menu(self, global_pos, target_imp):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #0284c7;
            }
            QMenu::item:disabled {
                color: #64748b;
            }
        """)

        act_change_pos = menu.addAction("위치 변경")
        act_change_pos.setEnabled(target_imp is not None)

        act_del_this = menu.addAction("이 임플란트 삭제")
        act_del_this.setEnabled(target_imp is not None)

        act_del_all = menu.addAction("전체 임플란트 삭제")
        has_implants = bool(self.state_mgr.implant_placement_mgr.implants)
        act_del_all.setEnabled(has_implants)

        action = menu.exec(global_pos)
        if action == act_change_pos and target_imp is not None:
            self._open_implant_transform_dialog(target_imp)
        elif action == act_del_this and target_imp is not None:
            self.state_mgr.implant_placement_mgr.delete_implant(target_imp.id)
            if hasattr(self, "implant_gizmo_3d"):
                self.implant_gizmo_3d.detach()
            self.render()
        elif action == act_del_all:
            self.state_mgr.implant_placement_mgr.delete_all_implants()
            if hasattr(self, "implant_gizmo_3d"):
                self.implant_gizmo_3d.detach()
            self.render()

    def _open_implant_transform_dialog(self, imp):
        from dental_fea_app.ui.placement.implant_transform_dialog import ImplantTransformDialog
        if hasattr(self, "_transform_dialog") and self._transform_dialog is not None:
            try:
                self._transform_dialog.close()
            except Exception:
                pass
        self._transform_dialog = ImplantTransformDialog(imp, self.state_mgr, self)
        self._transform_dialog.show()
        self._transform_dialog.raise_()
        self._transform_dialog.activateWindow()

    def _init_segmentation_surface_actors(self):
        """3D 분할 표면 메쉬(피질골, 수질골) 파이프라인 초기화"""
        # 1. 피질골 서피스 파이프라인 (빨간색/분홍색 계열)
        self.cortical_extractor = vtk.vtkFlyingEdges3D()
        self.cortical_extractor.ComputeNormalsOn()
        self.cortical_extractor.SetValue(0, 0.5)  # cortical_mask (0 or 1) 경계

        self.cortical_mapper = vtk.vtkPolyDataMapper()
        self.cortical_mapper.SetInputConnection(self.cortical_extractor.GetOutputPort())
        self.cortical_mapper.ScalarVisibilityOff()

        self.cortical_actor = vtk.vtkActor()
        self.cortical_actor.SetMapper(self.cortical_mapper)
        self.cortical_actor.GetProperty().SetColor(0.9, 0.4, 0.4)
        self.cortical_actor.GetProperty().SetOpacity(0.5)
        self.cortical_actor.GetProperty().SetSpecular(0.2)
        self.cortical_actor.GetProperty().SetSpecularPower(10.0)
        self.cortical_actor.SetVisibility(False)
        self.renderer.AddActor(self.cortical_actor)

        # 2. 수질골 서피스 파이프라인 (파란색/연청색 계열)
        self.cancellous_extractor = vtk.vtkFlyingEdges3D()
        self.cancellous_extractor.ComputeNormalsOn()
        self.cancellous_extractor.SetValue(0, 0.5)  # cancellous_mask (0 or 1) 경계

        self.cancellous_mapper = vtk.vtkPolyDataMapper()
        self.cancellous_mapper.SetInputConnection(self.cancellous_extractor.GetOutputPort())
        self.cancellous_mapper.ScalarVisibilityOff()

        self.cancellous_actor = vtk.vtkActor()
        self.cancellous_actor.SetMapper(self.cancellous_mapper)
        self.cancellous_actor.GetProperty().SetColor(0.3, 0.7, 0.9)
        self.cancellous_actor.GetProperty().SetOpacity(0.5)
        self.cancellous_actor.GetProperty().SetSpecular(0.2)
        self.cancellous_actor.GetProperty().SetSpecularPower(10.0)
        self.cancellous_actor.SetVisibility(False)
        self.renderer.AddActor(self.cancellous_actor)

    def _on_segmentation_result_changed(self, result):
        """분할 결과가 변경되었을 때 3D 표면 추출기 입력 연결 및 갱신"""
        if result is not None and self.state_mgr.vtk_image is not None:
            from dental_fea_app.segmentation.cortical_cancellous import mask_to_vtk_image
            # 피질골 마스크 VTK 생성 및 연결
            vtk_cortical = mask_to_vtk_image(result.cortical_mask, self.state_mgr.vtk_image)
            self.cortical_extractor.SetInputData(vtk_cortical)

            # 수질골 마스크 VTK 생성 및 연결
            vtk_cancellous = mask_to_vtk_image(result.cancellous_mask, self.state_mgr.vtk_image)
            self.cancellous_extractor.SetInputData(vtk_cancellous)

            self._apply_segmentation_visibility()
        else:
            self.cortical_actor.SetVisibility(False)
            self.cancellous_actor.SetVisibility(False)
        self.render()

    def _on_segmentation_visibility_changed(self, vis: dict):
        """피질골/수질골 3D 표면 가시성 및 불투명도 변경 반영"""
        self._apply_segmentation_visibility()
        self.render()

    def _apply_segmentation_visibility(self):
        if self.state_mgr.segmentation_result is None:
            self.cortical_actor.SetVisibility(False)
            self.cancellous_actor.SetVisibility(False)
            return

        vis = self.state_mgr.segmentation_visibility
        opacity = vis.get("opacity", 0.5)
        # show_3d_surface가 명시적으로 지정되지 않은 경우(예: 직접 state_mgr 호출)는 기본 True 처리
        show_3d = vis.get("show_3d_surface", True)
        show_cortical = vis.get("show_cortical", True) and show_3d
        show_cancellous = vis.get("show_cancellous", True) and show_3d

        self.cortical_actor.SetVisibility(show_cortical)
        self.cortical_actor.GetProperty().SetOpacity(opacity)

        self.cancellous_actor.SetVisibility(show_cancellous)
        self.cancellous_actor.GetProperty().SetOpacity(opacity)

    def _init_implant_actor(self):
        """3D 임플란트 서피스 액터 파이프라인 초기화 (골드/티타늄 색상)"""
        self.implant_mapper = vtk.vtkPolyDataMapper()
        self.implant_actor = vtk.vtkActor()
        self.implant_actor.SetMapper(self.implant_mapper)

        # 치과 임플란트 티타늄/골드 메탈릭 재질
        prop = self.implant_actor.GetProperty()
        prop.SetColor(1.0, 0.82, 0.28)  # 밝은 골드/티타늄
        prop.SetAmbient(0.3)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.6)
        prop.SetSpecularPower(30.0)
        prop.SetOpacity(1.0)
        self.implant_actor.SetVisibility(False)
        self.renderer.AddActor(self.implant_actor)

    def _on_implant_model_changed(self, model):
        """임플란트 모델이 변경되었을 때 3D 액터 갱신"""
        self._update_implant_actor()

    def _on_implant_pose_changed(self, model):
        """임플란트 포즈(위치/자세)가 변경되었을 때 3D 액터 갱신"""
        self._update_implant_actor()

    def _update_implant_actor(self):
        model = self.state_mgr.implant_model
        if model is not None and model.raw_polydata is not None:
            poly_w = model.get_transformed_polydata()
            self.implant_mapper.SetInputData(poly_w)
            self.implant_actor.SetVisibility(True)
        else:
            self.implant_actor.SetVisibility(False)
        self.request_render()

    def _on_implant_selected_3d(self, implant_id: str):
        """2D 뷰 또는 다이얼로그에서 임플란트 선택 변경 시 3D 기즈모 및 좌표 동기화"""
        if not hasattr(self, "implant_gizmo_3d"):
            return
        can_select = (not getattr(self, "is_result_view_mode", False)) and (
            not hasattr(self, "chk_implant_selectable") or self.chk_implant_selectable.isChecked()
        )
        if not can_select:
            self.implant_gizmo_3d.detach()
            if hasattr(self.state_mgr, "implant_placement_mgr") and self.state_mgr.implant_placement_mgr:
                for imp in self.state_mgr.implant_placement_mgr.implants.values():
                    if imp.axes_assembly:
                        imp.axes_assembly.SetVisibility(False)
            return
        if implant_id and implant_id in self.state_mgr.implant_placement_mgr.implants:
            self.implant_gizmo_3d.attach_to_implant(implant_id)
            if hasattr(self, "chk_sync_coords") and self.chk_sync_coords.isChecked():
                imp = self.state_mgr.implant_placement_mgr.implants[implant_id]
                self.state_mgr.set_coordinates(float(imp.position[0]), float(imp.position[1]), float(imp.position[2]))
        else:
            self.implant_gizmo_3d.detach()
        self.render()

    def _on_implant_transformed_3d(self, implant_id: str):
        """2D 뷰 또는 기즈모 드래그로 임플란트 포즈 변경 시 3D 기즈모 및 3D 뷰 동기화"""
        if not hasattr(self, "implant_gizmo_3d"):
            return
        active_id = getattr(self.implant_gizmo_3d, "active_implant_id", None)
        if self.implant_gizmo_3d.is_attached and active_id == implant_id:
            self.implant_gizmo_3d.update_transform()
        self.render()

    def _on_fea_mesh_model_changed(self, mesh_model):
        """체적 메쉬는 3D Volume 창에 표시하지 않고 3D FEA Geometry 뷰에만 표시"""
        for act in self.fea_mesh_actors.values():
            self.renderer.RemoveActor(act)
        self.fea_mesh_actors.clear()
        self.request_render()

    def _on_fea_bc_set_changed(self, bc_set):
        """경계조건은 3D Volume 창에 표시하지 않고 3D FEA Geometry 뷰에만 표시"""
        for act in self.fea_bc_actors:
            self.renderer.RemoveActor(act)
        self.fea_bc_actors.clear()
        self.request_render()




