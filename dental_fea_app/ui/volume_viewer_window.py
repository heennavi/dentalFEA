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
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QComboBox, QPushButton, QLabel, QStatusBar,
    QCheckBox, QSlider, QFrame, QToolButton, QGroupBox
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QAction, QMouseEvent, QIcon, QColor

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.core.settings_manager import SettingsManager
from dental_fea_app.ui.icons import IconFactory


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

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.settings_mgr = SettingsManager()
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

        if self.state_mgr.vtk_image is not None:
            self.on_volume_loaded(self.state_mgr.vtk_image)

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
        self.box_widget.SetRotationEnabled(False)  # 각 축에 정렬된 직육면체 유지
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

    def _on_box_widget_interaction(self, obj, event):
        """3D 창에서 박스의 선분/면/핸들을 드래그할 때 실시간 크롭 및 바운더리 갱신"""
        if self._box_widget_updating:
            return
        poly = vtk.vtkPolyData()
        self.box_widget.GetPolyData(poly)
        bounds = poly.GetBounds()
        self._box_widget_updating = True
        self.state_mgr.set_roi_bounds(bounds)
        self._box_widget_updating = False

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

        self.chk_roi_box = QCheckBox("Box 보이기", roi_group)
        self.chk_roi_box.setChecked(True)
        self.chk_roi_box.toggled.connect(self._on_roi_box_toggled)
        roi_layout.addWidget(self.chk_roi_box)

        layout.addWidget(roi_group)
        layout.addStretch()

        return panel


    def _toggle_settings_panel(self, checked: bool):
        self.panel.setVisible(checked)

    def _load_saved_settings(self):
        shading = self.settings_mgr.get("volume_3d.shading", True)
        self.chk_shading.setChecked(shading)
        self.volume_property.SetShade(shading)

        opacity_val = int(self.settings_mgr.get("volume_3d.opacity", 1.0) * 100)
        self.slider_opacity.setValue(opacity_val)
        self.lbl_opacity_val.setText(f"{opacity_val}%")

        self.chk_plane_axial.setChecked(self.settings_mgr.get("volume_3d.show_axial_plane", False))
        self.chk_plane_coronal.setChecked(self.settings_mgr.get("volume_3d.show_coronal_plane", False))
        self.chk_plane_sagittal.setChecked(self.settings_mgr.get("volume_3d.show_sagittal_plane", False))

        # ROI 설정 복원
        roi_apply = self.settings_mgr.get("volume_3d.roi_apply", False)
        roi_box = self.settings_mgr.get("volume_3d.roi_box", True)
        self.chk_roi_apply.setChecked(roi_apply)
        self.chk_roi_box.setChecked(roi_box)

        # 윈도우 위치/크기 복원
        geom = self.settings_mgr.get("volume_3d.geometry")
        if geom:
            try:
                from PySide6.QtCore import QByteArray
                self.restoreGeometry(QByteArray.fromHex(geom.encode("utf-8")))
            except Exception:
                pass

    def _on_roi_apply_toggled(self, checked: bool):
        self.settings_mgr.set("volume_3d.roi_apply", checked)
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

    def _on_roi_box_toggled(self, checked: bool):
        self.settings_mgr.set("volume_3d.roi_box", checked)
        self.state_mgr.roi_box_visibility_changed.emit(checked)
        self._update_roi_box_visibility()

    def _on_roi_bounds_updated(self, bounds):
        """2D 뷰 또는 3D 위젯에서 사각형이 조절되어 새로운 바운더리가 설정되었을 때 호출"""
        if bounds is not None:
            self.roi_box_source.SetBounds(bounds)
            if not self._box_widget_updating:
                try:
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
        """VTK Volume Cropping 적용/해제"""
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

        self._update_roi_box_visibility()
        self.render()

    def _update_roi_box_visibility(self):
        show_box = self.chk_roi_box.isChecked()
        is_boundary_mode = self.btn_roi_boundary.isChecked()
        has_bounds = self.state_mgr.roi_bounds is not None
        has_volume = self.state_mgr.vtk_image is not None

        if show_box and is_boundary_mode and has_bounds and has_volume:
            # boundary 설정 모드에서는 3D 상에서 직접 핸들/면/선분을 잡고 크기 조절할 수 있도록 box_widget 활성화
            try:
                self.box_widget.PlaceWidget(self.state_mgr.roi_bounds)
                self.box_widget.On()
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
            self.settings_mgr.set("volume_3d.show_axial_plane", visible)
        elif plane_type == "coronal":
            self.coronal_plane_actor.SetVisibility(visible and has_volume)
            self.settings_mgr.set("volume_3d.show_coronal_plane", visible)
        elif plane_type == "sagittal":
            self.sagittal_plane_actor.SetVisibility(visible and has_volume)
            self.settings_mgr.set("volume_3d.show_sagittal_plane", visible)
        self.render()

    def on_coordinates_changed(self, x: float, y: float, z: float):
        """2D MPR에서 클릭/이동된 월드 좌표에 맞춰 3D 슬라이스 평면 위치 동조"""
        if self.state_mgr.vtk_image is None:
            return

        bx0, bx1, by0, by1, bz0, bz1 = self.state_mgr.bounds

        # Axial Plane: XY 평면, Z = z
        self.axial_plane_source.SetOrigin(bx0, by0, z)
        self.axial_plane_source.SetPoint1(bx1, by0, z)
        self.axial_plane_source.SetPoint2(bx0, by1, z)

        # Coronal Plane: XZ 평면, Y = y
        self.coronal_plane_source.SetOrigin(bx0, y, bz0)
        self.coronal_plane_source.SetPoint1(bx1, y, bz0)
        self.coronal_plane_source.SetPoint2(bx0, y, bz1)

        # Sagittal Plane: YZ 평면, X = x
        self.sagittal_plane_source.SetOrigin(x, by0, bz0)
        self.sagittal_plane_source.SetPoint1(x, by1, bz0)
        self.sagittal_plane_source.SetPoint2(x, by0, bz1)

        # 만약 활성화된 클리핑 플레인이 있다면 위치 업데이트
        if self._current_clip_mode != "O":
            self._apply_clipping_plane()

        self.render()

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
        self.volume_mapper.RemoveAllClippingPlanes()

        if self._current_clip_mode == "O":
            self.render()
            return

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
        self.volume_mapper.RemoveAllClippingPlanes()
        self._update_clip_button_styles()
        self.render()

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

        # 평면 액터 가시성 복원
        self._set_plane_visible("axial", self.chk_plane_axial.isChecked())
        self._set_plane_visible("coronal", self.chk_plane_coronal.isChecked())
        self._set_plane_visible("sagittal", self.chk_plane_sagittal.isChecked())

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

    def render(self):
        if not self.isVisible() or self.state_mgr.vtk_image is None:
            return
        if self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()

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
            if self._orient_widget_enabled and self.orient_widget:
                self.orient_widget.SetEnabled(0)
                self._orient_widget_enabled = False
        except Exception:
            pass
        self.window_closed.emit()
        super().closeEvent(event)

