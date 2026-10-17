"""
mpr_container.py
Axial, Coronal, Sagittal 3개 MPR 뷰어 및 상단 툴바(십자선/자 토글, 90도 회전, 좌우/상하 대칭)와
3D 뷰어 퀵 컨트롤을 배치하는 2x2 컨테이너.
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFrame, QGroupBox,
    QToolButton, QSplitter
)
from PySide6.QtCore import Qt, QSize

from dental_fea_app.config import (
    AXIS_AXIAL, AXIS_CORONAL, AXIS_SAGITTAL
)
from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.ui.mpr_widget import MPRWidget
from dental_fea_app.ui.icons import IconFactory


class ViewPanel(QWidget):
    """
    MPRWidget과 상단 툴바(회전, 대칭, 십자선/자 토글), 하단 슬라이더를 포함하는 패널.
    """

    def __init__(self, axis: int, title: str, state_mgr: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.axis = axis
        self.state_mgr = state_mgr
        self._is_updating_slider = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 1. 상단 미니멀 헤더 & 툴바 (색상 두 줄 제거됨)
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
        header_layout.setSpacing(2)

        # 뷰 제목
        self.title_label = QLabel(title, self)
        self.title_label.setStyleSheet("font-weight: bold; color: #e5e5e5; font-size: 11px;")
        header_layout.addWidget(self.title_label)

        # 슬라이스 정보
        self.info_label = QLabel("- / - (0.00 mm)", self)
        self.info_label.setStyleSheet("color: #888899; font-size: 10px;")
        header_layout.addWidget(self.info_label)


        header_layout.addStretch()

        # 도구 버튼 공통 스타일 (텍스트 없이 순수 아이콘만 표시)
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
            QToolButton:checked {
                background-color: #1e3a5f;
                border-color: #61afef;
            }
        """

        # MPR 위젯 먼저 생성
        self.mpr_widget = MPRWidget(axis, state_mgr, self)
        icon_size = QSize(18, 18)

        # 1) 90도 시계방향 회전
        self.btn_rot_cw = QToolButton(header)
        self.btn_rot_cw.setIcon(IconFactory.rotate_cw_90(22))
        self.btn_rot_cw.setIconSize(icon_size)
        self.btn_rot_cw.setFixedSize(26, 24)
        self.btn_rot_cw.setToolTip("시계방향 90도 회전")
        self.btn_rot_cw.setStyleSheet(btn_qss)
        self.btn_rot_cw.clicked.connect(self.mpr_widget.rotate_cw_90)
        header_layout.addWidget(self.btn_rot_cw)

        # 2) -90도 반시계방향 회전
        self.btn_rot_ccw = QToolButton(header)
        self.btn_rot_ccw.setIcon(IconFactory.rotate_ccw_90(22))
        self.btn_rot_ccw.setIconSize(icon_size)
        self.btn_rot_ccw.setFixedSize(26, 24)
        self.btn_rot_ccw.setToolTip("반시계방향 90도 회전")
        self.btn_rot_ccw.setStyleSheet(btn_qss)
        self.btn_rot_ccw.clicked.connect(self.mpr_widget.rotate_ccw_90)
        header_layout.addWidget(self.btn_rot_ccw)

        # 3) 좌우 대칭 (Horizontal Flip)
        self.btn_flip_h = QToolButton(header)
        self.btn_flip_h.setIcon(IconFactory.flip_h(22))
        self.btn_flip_h.setIconSize(icon_size)
        self.btn_flip_h.setFixedSize(26, 24)
        self.btn_flip_h.setToolTip("좌우 대칭")
        self.btn_flip_h.setStyleSheet(btn_qss)
        self.btn_flip_h.clicked.connect(self.mpr_widget.flip_horizontal)
        header_layout.addWidget(self.btn_flip_h)

        # 4) 상하 대칭 (Vertical Flip)
        self.btn_flip_v = QToolButton(header)
        self.btn_flip_v.setIcon(IconFactory.flip_v(22))
        self.btn_flip_v.setIconSize(icon_size)
        self.btn_flip_v.setFixedSize(26, 24)
        self.btn_flip_v.setToolTip("상하 대칭")
        self.btn_flip_v.setStyleSheet(btn_qss)
        self.btn_flip_v.clicked.connect(self.mpr_widget.flip_vertical)
        header_layout.addWidget(self.btn_flip_v)

        # 5) 자(Ruler) 보기/감추기 토글
        self.btn_ruler = QToolButton(header)
        self.btn_ruler.setIcon(IconFactory.ruler(22))
        self.btn_ruler.setIconSize(icon_size)
        self.btn_ruler.setFixedSize(26, 24)
        self.btn_ruler.setCheckable(True)
        self.btn_ruler.setChecked(self.mpr_widget.is_ruler_visible())
        self.btn_ruler.setToolTip("눈금자(Ruler) 보기/감추기")
        self.btn_ruler.setStyleSheet(btn_qss)
        self.btn_ruler.clicked.connect(self._on_toggle_ruler)
        header_layout.addWidget(self.btn_ruler)

        # 6) 십자선 보기/감추기 토글
        self.btn_crosshair = QToolButton(header)
        self.btn_crosshair.setIcon(IconFactory.crosshair(22))
        self.btn_crosshair.setIconSize(icon_size)
        self.btn_crosshair.setFixedSize(26, 24)
        self.btn_crosshair.setCheckable(True)
        self.btn_crosshair.setChecked(self.mpr_widget.is_crosshair_visible())
        self.btn_crosshair.setToolTip("십자선(Crosshair) 보기/감추기")
        self.btn_crosshair.setStyleSheet(btn_qss)
        self.btn_crosshair.clicked.connect(self._on_toggle_crosshair)
        header_layout.addWidget(self.btn_crosshair)

        # 7) 단면 계측 도구 (가운데 구멍 뚫린 삼각자)
        self.btn_measure = QToolButton(header)
        self.btn_measure.setIcon(IconFactory.triangle_ruler(22))
        self.btn_measure.setIconSize(icon_size)
        self.btn_measure.setFixedSize(26, 24)
        self.btn_measure.setToolTip("단면 계측 도구 (길이/각도 측정)")
        self.btn_measure.setStyleSheet(btn_qss)
        self.btn_measure.clicked.connect(self._open_measure_dialog)
        header_layout.addWidget(self.btn_measure)

        # 8) 주석 작성 (노트)
        self.btn_annotation = QToolButton(header)
        self.btn_annotation.setIcon(IconFactory.note(22))
        self.btn_annotation.setIconSize(icon_size)
        self.btn_annotation.setFixedSize(26, 24)
        self.btn_annotation.setToolTip("주석 작성 (소견 메모 및 DICOM 정보)")
        self.btn_annotation.setStyleSheet(btn_qss)
        self.btn_annotation.clicked.connect(self._open_annotation_dialog)
        header_layout.addWidget(self.btn_annotation)

        # 9) 그리기 도구 (파레트)
        self.btn_draw = QToolButton(header)
        self.btn_draw.setIcon(IconFactory.palette(22))
        self.btn_draw.setIconSize(icon_size)
        self.btn_draw.setFixedSize(26, 24)
        self.btn_draw.setToolTip("그리기 도구 (선분, 화살표, 사각형, 원, 프리폼)")
        self.btn_draw.setStyleSheet(btn_qss)
        self.btn_draw.clicked.connect(self._open_draw_dialog)
        header_layout.addWidget(self.btn_draw)

        layout.addWidget(header)


        # 2. MPR 렌더링 위젯
        layout.addWidget(self.mpr_widget, stretch=1)

        # 3. 하단 슬라이스 슬라이더
        slider_bar = QHBoxLayout()
        slider_bar.setContentsMargins(4, 0, 4, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_bar.addWidget(self.slider)

        layout.addLayout(slider_bar)

        # 상태 관리자 신호 연결
        self.state_mgr.coordinates_changed.connect(self._update_info)
        self.state_mgr.volume_loaded.connect(self._on_volume_loaded)

    def _on_toggle_crosshair(self, checked: bool):
        self.mpr_widget.set_crosshair_visible(checked)

    def _on_toggle_ruler(self, checked: bool):
        self.mpr_widget.set_ruler_visible(checked)

    def _on_volume_loaded(self, vtk_image):
        dims = self.state_mgr.dimensions
        max_idx = max(0, dims[self.axis] - 1)
        self.slider.setRange(0, max_idx)
        self._update_info(*self.state_mgr.current_coords)

    def _update_info(self, x: float, y: float, z: float):
        dims = self.state_mgr.dimensions
        if dims[self.axis] <= 0:
            return

        coords = (x, y, z)
        coord = coords[self.axis]
        origin = self.state_mgr.origin[self.axis]
        spacing = self.state_mgr.spacing[self.axis]

        idx = int(round((coord - origin) / spacing))
        idx = max(0, min(dims[self.axis] - 1, idx))

        self.info_label.setText(f"{idx + 1} / {dims[self.axis]} ({coord:.2f} mm)")

        self._is_updating_slider = True
        self.slider.setValue(idx)
        self._is_updating_slider = False

    def _on_slider_changed(self, value: int):
        if self._is_updating_slider or self.state_mgr.vtk_image is None:
            return

        origin = self.state_mgr.origin[self.axis]
        spacing = self.state_mgr.spacing[self.axis]
        new_coord = origin + value * spacing
        self.state_mgr.set_axis_coordinate(self.axis, new_coord)

    @property
    def _measure_dialog(self):
        if hasattr(self, "container") and self.container is not None:
            return self.container.measure_dialog
        return getattr(self, "_local_measure_dialog", None)

    @property
    def _annotation_dialog(self):
        if hasattr(self, "container") and self.container is not None:
            return self.container.annotation_dialog
        return getattr(self, "_local_annotation_dialog", None)

    @property
    def _draw_dialog(self):
        if hasattr(self, "container") and self.container is not None:
            return self.container.draw_dialog
        return getattr(self, "_local_draw_dialog", None)

    def _open_measure_dialog(self):
        if hasattr(self, "container") and self.container is not None:
            self.container.open_measure_dialog()
        else:
            from dental_fea_app.ui.measurement_dialog import MeasurementDialog
            if getattr(self, "_local_measure_dialog", None) is None:
                self._local_measure_dialog = MeasurementDialog(self)
                self._local_measure_dialog.mode_changed.connect(self._on_measure_mode_changed)
                self._local_measure_dialog.item_deleted.connect(self._on_measurement_updated)
                self._local_measure_dialog.all_cleared.connect(self._on_measurement_updated)
                self.mpr_widget.measure_dialog = self._local_measure_dialog
            self._local_measure_dialog.show()
            self._local_measure_dialog.raise_()
            self._local_measure_dialog.activateWindow()

    def _on_measurement_updated(self):
        self._update_tool_buttons_border()
        self.mpr_widget.render()

    def _on_measure_mode_changed(self, mode: str):
        if mode == "length":
            self.state_mgr.interaction_mode_changed.emit("measure_length")
        elif mode == "angle":
            self.state_mgr.interaction_mode_changed.emit("measure_angle")
        else:
            self.state_mgr.interaction_mode_changed.emit("normal")

    def _open_annotation_dialog(self):
        if hasattr(self, "container") and self.container is not None:
            self.container.open_annotation_dialog()
        else:
            from dental_fea_app.ui.annotation_dialog import AnnotationDialog
            if not hasattr(self, "_annotation_dialog") or self._annotation_dialog is None:
                self._annotation_dialog = AnnotationDialog(self, dicom_meta=self.state_mgr.metadata)
                self._annotation_dialog.saved.connect(lambda _: self._update_tool_buttons_border())
            else:
                self._annotation_dialog.update_dicom_meta(self.state_mgr.metadata)
            self._annotation_dialog.show()
            self._annotation_dialog.raise_()
            self._annotation_dialog.activateWindow()

    def _open_draw_dialog(self):
        if hasattr(self, "container") and self.container is not None:
            self.container.open_draw_dialog()
        else:
            from dental_fea_app.ui.draw_dialog import DrawDialog
            if not hasattr(self, "_draw_dialog") or self._draw_dialog is None:
                self._draw_dialog = DrawDialog(self)
                self._draw_dialog.active_toggled.connect(self._on_draw_active_toggled)
                self._draw_dialog.style_changed.connect(self._on_draw_style_changed)
                self._draw_dialog.item_deleted.connect(self._on_draw_updated)
                self._draw_dialog.all_cleared.connect(self._on_draw_updated)
                self.mpr_widget.draw_dialog = self._draw_dialog
            self._draw_dialog.show()
            self._draw_dialog.raise_()
            self._draw_dialog.activateWindow()

    def _on_draw_updated(self):
        self._update_tool_buttons_border()
        self.mpr_widget.render()

    def _on_draw_active_toggled(self, active: bool):
        if active:
            stype = getattr(self._draw_dialog, "shape_type", "line")
            if stype == "select":
                self.state_mgr.interaction_mode_changed.emit("draw_select")
            else:
                self.state_mgr.interaction_mode_changed.emit("draw")
        else:
            self.state_mgr.interaction_mode_changed.emit("normal")

    def _on_draw_style_changed(self, style: dict):
        stype = style.get("shape_type", "line")
        if stype == "select":
            self.state_mgr.interaction_mode_changed.emit("draw_select")
        else:
            self.state_mgr.interaction_mode_changed.emit("draw")

    def _update_tool_buttons_border(self):
        """내용(주석, 계측, 그린 도형) 존재 여부에 따라 얇은 파란색 테두리 적용"""
        has_measure = False
        has_annot = False
        has_draw = False

        if hasattr(self, "container") and self.container is not None:
            has_measure = bool(getattr(self.container.measure_dialog, "measurements", []))
            has_annot = bool(self.container.annotation_dialog.editor.toPlainText().strip())
            has_draw = bool(getattr(self.container.draw_dialog, "shapes", []))
        else:
            has_measure = hasattr(self, "_measure_dialog") and bool(getattr(self._measure_dialog, "measurements", []))
            has_annot = hasattr(self, "_annotation_dialog") and bool(self._annotation_dialog.editor.toPlainText().strip())
            has_draw = hasattr(self, "_draw_dialog") and bool(getattr(self._draw_dialog, "shapes", []))

        def set_border(btn, active):
            if active:
                btn.setStyleSheet("""
                    QToolButton {
                        background-color: #2c2c32;
                        border: 2px solid #2196f3;
                        border-radius: 3px;
                        padding: 2px;
                    }
                    QToolButton:hover {
                        background-color: #3e3e48;
                        border-color: #64b5f6;
                    }
                """)
            else:
                btn.setStyleSheet("""
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
                """)

        set_border(self.btn_measure, has_measure)
        set_border(self.btn_annotation, has_annot)
        set_border(self.btn_draw, has_draw)

    def minimumSizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(200, 150)


class MPRContainer(QWidget):
    """
    Axial, Coronal, Sagittal 뷰 및 3D 볼륨 뷰어 제어 4분할 컨테이너.
    - 얇은 회색(2px, #555555) 핸들의 QSplitter 4분할 (Top/Bottom 및 Left/Right)
    - 상하 스플리터의 좌우 분할 크기 동기화
    - 크기 상태 저장 및 복원 (SettingsManager 연동)
    - 공유 다이얼로그(계측 도구, 주석, 그리기 도구) 통합 관리
    """

    def __init__(self, state_manager: StateManager, on_open_3d_callback, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.on_open_3d_callback = on_open_3d_callback

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 3개 뷰 공용 다이얼로그 생성
        from dental_fea_app.ui.measurement_dialog import MeasurementDialog
        from dental_fea_app.ui.annotation_dialog import AnnotationDialog
        from dental_fea_app.ui.draw_dialog import DrawDialog

        self.measure_dialog = MeasurementDialog(self)
        self.annotation_dialog = AnnotationDialog(self, dicom_meta=self.state_mgr.metadata)
        self.draw_dialog = DrawDialog(self)

        # 2. 3개의 MPR 패널 생성
        self.axial_panel = ViewPanel(AXIS_AXIAL, "Axial (Z)", self.state_mgr, self)
        self.axial_panel.container = self
        self.coronal_panel = ViewPanel(AXIS_CORONAL, "Coronal (Y)", self.state_mgr, self)
        self.coronal_panel.container = self
        self.sagittal_panel = ViewPanel(AXIS_SAGITTAL, "Sagittal (X)", self.state_mgr, self)
        self.sagittal_panel.container = self

        self.panels = [self.axial_panel, self.coronal_panel, self.sagittal_panel]
        for p in self.panels:
            p.mpr_widget.measure_dialog = self.measure_dialog
            p.mpr_widget.draw_dialog = self.draw_dialog

        # 3. 다이얼로그 신호 연결
        self._init_shared_dialog_signals()

        # 4번째 슬롯: 빈 프레임 (우하단)
        self.control_quadrant = self._create_control_quadrant()

        # 상단 수평 스플리터 (Axial | Coronal)
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.top_splitter.addWidget(self.axial_panel)
        self.top_splitter.addWidget(self.coronal_panel)
        self.top_splitter.setOpaqueResize(True)
        self.top_splitter.setChildrenCollapsible(False)

        # 하단 수평 스플리터 (Sagittal | Control Quadrant)
        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.bottom_splitter.addWidget(self.sagittal_panel)
        self.bottom_splitter.addWidget(self.control_quadrant)
        self.bottom_splitter.setOpaqueResize(True)
        self.bottom_splitter.setChildrenCollapsible(False)

        # 각 스플리터 패널의 stretch factor 동일 배분
        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 1)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 1)

        # 메인 수직 스플리터 (Top / Bottom)
        self.v_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.v_splitter.addWidget(self.top_splitter)
        self.v_splitter.addWidget(self.bottom_splitter)
        self.v_splitter.setOpaqueResize(True)
        self.v_splitter.setChildrenCollapsible(False)
        self.v_splitter.setStretchFactor(0, 1)
        self.v_splitter.setStretchFactor(1, 1)

        # 얇고 선명한 회색 스플리터 바 스타일시트
        splitter_qss = """
            QSplitter::handle {
                background-color: #555555;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
            QSplitter::handle:vertical {
                height: 2px;
            }
        """
        self.top_splitter.setStyleSheet(splitter_qss)
        self.bottom_splitter.setStyleSheet(splitter_qss)
        self.v_splitter.setStyleSheet(splitter_qss)

        main_layout.addWidget(self.v_splitter)

        # 상하 수평 스플리터 간 좌우 이동 동기화
        self._syncing_splitters = False
        self.top_splitter.splitterMoved.connect(
            lambda pos, idx: self._sync_horizontal(self.top_splitter, self.bottom_splitter)
        )
        self.bottom_splitter.splitterMoved.connect(
            lambda pos, idx: self._sync_horizontal(self.bottom_splitter, self.top_splitter)
        )

        # 메타데이터 업데이트 시 주석 다이얼로그 동기화
        self.state_mgr.volume_loaded.connect(self._on_volume_loaded_for_dialogs)

    def _init_shared_dialog_signals(self):
        # 계측 다이얼로그
        self.measure_dialog.mode_changed.connect(self._on_measure_mode_changed)
        self.measure_dialog.item_selected.connect(self._on_measure_item_selected)
        self.measure_dialog.item_deleted.connect(self._on_measure_updated)
        self.measure_dialog.all_cleared.connect(self._on_measure_updated)
        self.measure_dialog.visibility_toggled.connect(self._on_measure_visibility_toggled)
        self.measure_dialog.navigate_to_item.connect(self._on_navigate_to_item)

        # 주석 다이얼로그
        self.annotation_dialog.saved.connect(lambda _: self._update_all_borders())

        # 그리기 다이얼로그
        self.draw_dialog.active_toggled.connect(self._on_draw_active_toggled)
        self.draw_dialog.style_changed.connect(self._on_draw_style_changed)
        self.draw_dialog.item_selected.connect(self._on_draw_item_selected)
        self.draw_dialog.item_deleted.connect(self._on_draw_updated)
        self.draw_dialog.all_cleared.connect(self._on_draw_updated)
        self.draw_dialog.visibility_toggled.connect(self._on_draw_visibility_toggled)
        self.draw_dialog.navigate_to_item.connect(self._on_navigate_to_item)

    def _on_volume_loaded_for_dialogs(self, vtk_image):
        self.annotation_dialog.update_dicom_meta(self.state_mgr.metadata)

    def open_measure_dialog(self):
        self.measure_dialog.show()
        self.measure_dialog.raise_()
        self.measure_dialog.activateWindow()

    def open_annotation_dialog(self):
        self.annotation_dialog.update_dicom_meta(self.state_mgr.metadata)
        self.annotation_dialog.show()
        self.annotation_dialog.raise_()
        self.annotation_dialog.activateWindow()

    def open_draw_dialog(self):
        self.draw_dialog.show()
        self.draw_dialog.raise_()
        self.draw_dialog.activateWindow()

    def _on_measure_mode_changed(self, mode: str):
        if mode == "length":
            self.state_mgr.interaction_mode_changed.emit("measure_length")
        elif mode == "angle":
            self.state_mgr.interaction_mode_changed.emit("measure_angle")
        else:
            self.state_mgr.interaction_mode_changed.emit("normal")

    def _on_measure_item_selected(self, idx: int):
        for p in self.panels:
            p.mpr_widget.set_selected_measure(idx)

    def _on_measure_updated(self):
        self._update_all_borders()
        for p in self.panels:
            p.mpr_widget._update_overlays()
            p.mpr_widget.render()

    def _on_measure_visibility_toggled(self, visible: bool):
        for p in self.panels:
            p.mpr_widget.set_measurements_visible(visible)

    def _on_draw_active_toggled(self, active: bool):
        if active:
            stype = getattr(self.draw_dialog, "shape_type", "line")
            if stype == "select":
                self.state_mgr.interaction_mode_changed.emit("draw_select")
            else:
                self.state_mgr.interaction_mode_changed.emit("draw")
        else:
            self.state_mgr.interaction_mode_changed.emit("normal")

    def _on_draw_style_changed(self, style: dict):
        stype = style.get("shape_type", "line")
        if stype == "select":
            self.state_mgr.interaction_mode_changed.emit("draw_select")
        else:
            self.state_mgr.interaction_mode_changed.emit("draw")

    def _on_draw_item_selected(self, idx: int):
        for p in self.panels:
            p.mpr_widget.set_selected_shape(idx)

    def _on_draw_updated(self):
        self._update_all_borders()
        for p in self.panels:
            p.mpr_widget._update_overlays()
            p.mpr_widget.render()

    def _on_draw_visibility_toggled(self, visible: bool):
        for p in self.panels:
            p.mpr_widget.set_drawn_shapes_visible(visible)

    def _update_all_borders(self):
        for p in self.panels:
            p._update_tool_buttons_border()

    def _on_navigate_to_item(self, view_name: str, slice_idx: int, center: object):
        """더블 클릭 시 해당 단면 및 슬라이스로 이동"""
        axis_map = {
            "Axial": AXIS_AXIAL,
            "Coronal": AXIS_CORONAL,
            "Sagittal": AXIS_SAGITTAL
        }
        axis = axis_map.get(view_name)
        if axis is None or self.state_mgr.vtk_image is None:
            return

        if center is not None and isinstance(center, (list, tuple)) and len(center) >= 3:
            # 점의 정확한 3차원 월드 좌표로 이동
            self.state_mgr.set_coordinates(float(center[0]), float(center[1]), float(center[2]))
        else:
            origin = self.state_mgr.origin[axis]
            spacing = self.state_mgr.spacing[axis]
            coord = origin + slice_idx * spacing
            self.state_mgr.set_axis_coordinate(axis, coord)

    def _sync_horizontal(self, src: QSplitter, dst: QSplitter):
        """상하 수평 스플리터의 좌우 분할 크기를 일치시켜 십자 분할선 유지"""
        if self._syncing_splitters:
            return
        self._syncing_splitters = True
        dst.setSizes(src.sizes())
        self._syncing_splitters = False

    def save_splitter_state(self, settings_mgr):
        """스플리터 분할 비율을 SettingsManager에 저장"""
        settings_mgr.set("mpr_container.v_sizes", self.v_splitter.sizes(), auto_save=False)
        settings_mgr.set("mpr_container.h_sizes", self.top_splitter.sizes(), auto_save=True)

    def restore_splitter_state(self, settings_mgr):
        """저장된 스플리터 분할 비율 복원"""
        v_sizes = settings_mgr.get("mpr_container.v_sizes")
        h_sizes = settings_mgr.get("mpr_container.h_sizes")
        if v_sizes and len(v_sizes) == 2 and sum(v_sizes) > 0:
            self.v_splitter.setSizes(v_sizes)
        if h_sizes and len(h_sizes) == 2 and sum(h_sizes) > 0:
            self.top_splitter.setSizes(h_sizes)
            self.bottom_splitter.setSizes(h_sizes)

    def reset_all_views(self):
        """
        모든 뷰를 처음 로드한 상태의 이미지 보기로 완전 초기화합니다.
        (회전 0도, 대칭 해제, 십자선/자 표시, 볼륨 중앙 슬라이스 및 기본 W/L)
        """
        self.axial_panel.mpr_widget.reset_view_state()
        self.coronal_panel.mpr_widget.reset_view_state()
        self.sagittal_panel.mpr_widget.reset_view_state()

        if self.state_mgr.vtk_image is not None:
            b = self.state_mgr.bounds
            cx = 0.5 * (b[0] + b[1])
            cy = 0.5 * (b[2] + b[3])
            cz = 0.5 * (b[4] + b[5])
            self.state_mgr.set_coordinates(cx, cy, cz)

            from dental_fea_app.config import WL_PRESETS, DEFAULT_PRESET_NAME
            preset = WL_PRESETS[DEFAULT_PRESET_NAME]
            self.state_mgr.set_window_level(preset.window, preset.level)

    def reset_all_cameras(self):
        """모든 2D 뷰의 카메라 시점 및 배율만 리셋"""
        self.axial_panel.mpr_widget.reset_camera()
        self.coronal_panel.mpr_widget.reset_camera()
        self.sagittal_panel.mpr_widget.reset_camera()

    def get_user_info_data(self) -> dict:
        """저장을 위한 3종 데이터(주석, 계측, 그린 도형) 수집"""
        res = {
            "annotations": {},
            "measurements": [],
            "shapes": []
        }
        # 첫 번째 패널 또는 등록된 패널들의 다이얼로그로부터 데이터 수집
        panels = [self.axial_panel, self.coronal_panel, self.sagittal_panel]
        for p in panels:
            if hasattr(p, "_annotation_dialog") and p._annotation_dialog:
                t = p._annotation_dialog.to_dict().get("text", "")
                if t:
                    res["annotations"]["text"] = t
                    break
        for p in panels:
            if hasattr(p, "_measure_dialog") and p._measure_dialog:
                m_list = p._measure_dialog.to_dict()
                if m_list:
                    res["measurements"] = m_list
                    break
        for p in panels:
            if hasattr(p, "_draw_dialog") and p._draw_dialog:
                s_list = p._draw_dialog.to_dict()
                if s_list:
                    res["shapes"] = s_list
                    break
        return res

    def set_user_info_data(self, data: dict):
        """불러온 사용자 정보 데이터를 다이얼로그 및 뷰에 적용하고 테두리 갱신"""
        panels = [self.axial_panel, self.coronal_panel, self.sagittal_panel]
        annot_data = data.get("annotations", {})
        measure_data = data.get("measurements", [])
        shapes_data = data.get("shapes", [])

        for p in panels:
            if not hasattr(p, "_annotation_dialog") or p._annotation_dialog is None:
                p._open_annotation_dialog()
                p._annotation_dialog.hide()
            p._annotation_dialog.from_dict(annot_data)

            if not hasattr(p, "_measure_dialog") or p._measure_dialog is None:
                p._open_measure_dialog()
                p._measure_dialog.hide()
            p._measure_dialog.from_dict(measure_data)

            if not hasattr(p, "_draw_dialog") or p._draw_dialog is None:
                p._open_draw_dialog()
                p._draw_dialog.hide()
            p._draw_dialog.from_dict(shapes_data)

            p._update_tool_buttons_border()

    def _create_control_quadrant(self) -> QWidget:
        """
        우하단 사분면 패널.
        사용자 요청에 따라 내부 버튼 및 라벨 등 내용물을 모두 제거한 빈 검은색 패널을 제공합니다.
        """
        container = QFrame(self)
        container.setStyleSheet("background-color: #121214; border: 1px solid #282830; border-radius: 4px;")
        container.setMinimumSize(100, 100)
        return container




