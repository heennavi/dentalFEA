"""
implant_simulation_dialog.py
임플란트 및 보철 시뮬레이션 (Implant Placement & Prosthetics Simulation) 대화상자.
- 상단 3개 탭: [임플란트 (Implant)], [Abutment], [보철물 (Crown)]
- 하단 고정 영역: 식립된 임플란트 목록 (Placed Implants Table) 및 상세정보, 우클릭 컨텍스트 메뉴,
  [임플란트 전체 삭제], [선택한 임플란트 지우기], [식립 정보 저장], [식립 정보 불러오기], [닫기]
- 상단 탭 내용:
  * 탭 1 (임플란트): 라이브러리 선택, 직경/길이 필터, 테이블, 3D 미리보기, [상악에 식립], [Replace], [Place]
    조작 방식(3D 기즈모 / 팝업 윈도우), [임플란트 높이 정렬], [임플란트 각도 정렬], 간격 슬라이더 + [간격 정렬], [되돌리기 (Undo)]
    (중간의 임플란트 가이드 그룹은 사용자 요청에 따라 제거됨)
  * 탭 2 (Abutment): Collar Flare 각도, Collar 길이, 상부 Morse 길이, Angulation 기울기/방향, 색상,
    [Abutment 장착하기], [Abutment 제거], [Abutment 교체], [Abutment 전부 제거], 3D 미리보기
  * 탭 3 (보철물 Crown): 인터랙티브 Tooth Chart (683x145, 골드/루비 번호), 치관 길이/폭/협설폭/회전/재질,
    [보철물 장착하기], [보철물 제거], [보철물 교체], [보철물 전부 제거], 3D 해부학적 미리보기
"""
import copy
import os
import json
from typing import Optional, List, Dict, Tuple
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QSlider, QComboBox, QRadioButton, QButtonGroup, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QTextBrowser,
    QSplitter, QMenu, QCheckBox, QTabWidget, QScrollArea, QFrame, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QItemSelectionModel
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap

from dental_fea_app.placement.implant_generator import (
    ImplantSpec, ImplantLibraryManager,
    AbutmentSpec, generate_abutment_polydata,
    ProstheticSpec, generate_prosthetic_polydata,
    get_material_color_rgb, get_default_tooth_dimensions
)
from dental_fea_app.placement.implant_placement_manager import (
    ImplantPlacementManager, PlacedImplant
)
from dental_fea_app.ui.icons import IconFactory
from dental_fea_app.ui.settings_dialog import Implant3DPreviewWidget


# FDI 치아 번호 좌표 매핑 (ToothChartWidget)
TOOTH_NUMBER_POSITIONS = {
    17: (25, 45), 16: (88, 44), 15: (144, 46), 14: (189, 46), 13: (229, 45), 12: (266, 45), 11: (310, 44),
    21: (359, 43), 22: (403, 48), 23: (447, 45), 24: (490, 48), 25: (528, 48), 26: (584, 49), 27: (652, 45),
    47: (49, 98), 46: (114, 99), 45: (166, 102), 44: (210, 103), 43: (250, 103), 42: (287, 101), 41: (317, 99),
    31: (353, 100), 32: (386, 99), 33: (426, 103), 34: (473, 104), 35: (517, 102), 36: (569, 97), 37: (634, 100)
}

UPPER_TOOTH_RANGES = [
    (17, 0, 65), (16, 65, 125), (15, 125, 170), (14, 170, 209), (13, 209, 247), (12, 247, 288), (11, 288, 334),
    (21, 334, 381), (22, 381, 425), (23, 425, 468), (24, 468, 509), (25, 509, 556), (26, 556, 618), (27, 618, 683)
]

LOWER_TOOTH_RANGES = [
    (47, 0, 81), (46, 81, 140), (45, 140, 188), (44, 188, 230), (43, 230, 268), (42, 268, 302), (41, 302, 335),
    (31, 335, 369), (32, 369, 406), (33, 406, 449), (34, 449, 495), (35, 495, 543), (36, 543, 601), (37, 601, 683)
]

TOOTH_NAMES = {
    17: "상악 우측 제2대구치 (#17)", 16: "상악 우측 제1대구치 (#16)", 15: "상악 우측 제2소구치 (#15)", 14: "상악 우측 제1소구치 (#14)",
    13: "상악 우측 견치 (#13)", 12: "상악 우측 측절치 (#12)", 11: "상악 우측 중절치 (#11)",
    21: "상악 좌측 중절치 (#21)", 22: "상악 좌측 측절치 (#22)", 23: "상악 좌측 견치 (#23)",
    24: "상악 좌측 제1소구치 (#24)", 25: "상악 좌측 제2소구치 (#25)", 26: "상악 좌측 제1대구치 (#26)", 27: "상악 좌측 제2대구치 (#27)",
    47: "하악 우측 제2대구치 (#47)", 46: "하악 우측 제1대구치 (#46)", 45: "하악 우측 제2소구치 (#45)", 44: "하악 우측 제1소구치 (#44)",
    43: "하악 우측 견치 (#43)", 42: "하악 우측 측절치 (#42)", 41: "하악 우측 중절치 (#41)",
    31: "하악 좌측 중절치 (#31)", 32: "하악 좌측 측절치 (#32)", 33: "하악 좌측 견치 (#33)",
    34: "하악 좌측 제1소구치 (#34)", 35: "하악 좌측 제2소구치 (#35)", 36: "하악 좌측 제1대구치 (#36)", 37: "하악 좌측 제2대구치 (#37)"
}


class ToothChartWidget(QWidget):
    """골드 및 루비 색상 번호 인터랙티브 치아 차트 위젯 (683 x 145)"""
    tooth_selected = Signal(int)
    tooth_hovered = Signal(int, str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.selected_tooth = 16
        self.hovered_tooth: Optional[int] = None
        self.ruby_patches: Dict[int, Tuple[int, int, int, int, QPixmap]] = {}

        # 리소스 이미지 로드
        img_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "tooth_chart.png")
        self.base_pixmap = QPixmap(img_path)
        if self.base_pixmap.isNull():
            alt_path = "/home/kim/Documents/project/dentalSeeTee/dental_viewer/resources/tooth_chart.png"
            if os.path.exists(alt_path):
                self.base_pixmap = QPixmap(alt_path)

        self.img_w = 683
        self.img_h = 145
        self.setFixedSize(self.img_w, self.img_h)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._create_ruby_patches()

    def _create_ruby_patches(self):
        if self.base_pixmap.isNull():
            return
        base_img = self.base_pixmap.toImage()
        img_w, img_h = base_img.width(), base_img.height()

        for t_num, (cx, cy) in TOOTH_NUMBER_POSITIONS.items():
            x1 = max(0, cx - 18)
            y1 = max(0, cy - 13)
            x2 = min(img_w, cx + 18)
            y2 = min(img_h, cy + 13)
            pw, ph = x2 - x1, y2 - y1

            patch_img = base_img.copy(x1, y1, pw, ph)
            for py in range(ph):
                for px in range(pw):
                    col = patch_img.pixelColor(px, py)
                    r, g, b, a = col.red(), col.green(), col.blue(), col.alpha()
                    if r > 140 and g > 100 and b < 100 and (r > b + 45):
                        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                        new_r = min(255, int(245 * lum * 1.15))
                        new_g = min(255, int(22 * lum))
                        new_b = min(255, int(100 * lum * 1.1))
                        patch_img.setPixelColor(px, py, QColor(new_r, new_g, new_b, a))

            self.ruby_patches[t_num] = (x1, y1, pw, ph, QPixmap.fromImage(patch_img))

    def set_selected_tooth(self, tooth_num: int):
        if tooth_num != self.selected_tooth:
            self.selected_tooth = tooth_num
            self.update()

    def _get_tooth_from_pos(self, pos: QPoint) -> Optional[int]:
        if self.width() <= 0 or self.height() <= 0:
            return None
        x_orig = int(round(pos.x() * (float(self.img_w) / float(self.width()))))
        y_orig = int(round(pos.y() * (float(self.img_h) / float(self.height()))))

        if 10 <= y_orig <= 73:
            for t_num, x1, x2 in UPPER_TOOTH_RANGES:
                if x1 <= x_orig < x2:
                    return t_num
        elif 74 <= y_orig <= 138:
            for t_num, x1, x2 in LOWER_TOOTH_RANGES:
                if x1 <= x_orig < x2:
                    return t_num
        return None

    def mouseMoveEvent(self, event):
        tooth = self._get_tooth_from_pos(event.pos())
        if tooth != self.hovered_tooth:
            self.hovered_tooth = tooth
            self.update()
            if tooth and tooth in TOOTH_NAMES:
                name = TOOTH_NAMES[tooth]
                self.tooth_hovered.emit(tooth, name)
                self.setToolTip(name)
            else:
                self.tooth_hovered.emit(0, "")
                self.setToolTip("")
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self.hovered_tooth is not None:
            self.hovered_tooth = None
            self.update()
            self.tooth_hovered.emit(0, "")
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            tooth = self._get_tooth_from_pos(event.pos())
            if tooth:
                self.selected_tooth = tooth
                self.update()
                self.tooth_selected.emit(tooth)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        target_rect = self.rect()
        if not self.base_pixmap.isNull():
            painter.drawPixmap(target_rect, self.base_pixmap)
        else:
            painter.fillRect(target_rect, QColor("#1e293b"))

        sx = target_rect.width() / float(self.img_w)
        sy = target_rect.height() / float(self.img_h)

        active_tooth = self.hovered_tooth if self.hovered_tooth is not None else self.selected_tooth
        if active_tooth and active_tooth in self.ruby_patches:
            px, py, pw, ph, patch_pix = self.ruby_patches[active_tooth]
            dest_rect = QRect(int(round(px * sx)), int(round(py * sy)),
                              int(round(pw * sx)), int(round(ph * sy)))
            painter.drawPixmap(dest_rect, patch_pix)


class ImplantSimulationDialog(QDialog):
    """임플란트 및 보철 시뮬레이션 종합 관리 대화상자"""

    placement_requested = Signal(object, bool) # (ImplantSpec, is_maxilla: bool)
    implant_selected = Signal(str)            # implant_id
    alignment_changed = Signal()

    def __init__(self, placement_mgr: ImplantPlacementManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.placement_mgr = placement_mgr
        self.setWindowTitle("임플란트 및 보철 시뮬레이션 (Implant Placement & Prosthetics Simulation)")
        self.resize(880, 850)
        self.setStyleSheet("""
            QDialog { background-color: #0f172a; color: #f8fafc; }
            QTabWidget::pane { border: 1px solid #1e293b; background-color: #0f172a; }
            QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 7px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-weight: bold; margin-right: 2px; }
            QTabBar::tab:selected { background: #0284c7; color: #ffffff; }
            QGroupBox { border: 1px solid #1e293b; border-radius: 6px; margin-top: 10px; font-weight: bold; color: #38bdf8; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QTableWidget { background-color: #090d16; alternate-background-color: #111827; gridline-color: #1e293b; color: #f8fafc; border: 1px solid #1e293b; }
            QTableWidget::item:selected { background-color: #0284c7; color: #ffffff; }
            QHeaderView::section { background-color: #1e293b; color: #cbd5e1; font-weight: bold; padding: 4px; border: 1px solid #0f172a; }
        """)

        self.all_implants: List[ImplantSpec] = []
        self.filtered_implants: List[ImplantSpec] = []
        self.selected_lib_spec: Optional[ImplantSpec] = None

        self._init_ui()
        self._load_implant_list()
        self.refresh_placed_implants_table()
        self._update_abutment_preview()
        self._update_prosthetic_preview()

        if hasattr(self.placement_mgr, "state_mgr") and self.placement_mgr.state_mgr:
            self.placement_mgr.state_mgr.implant_selected.connect(self._on_external_implant_selected)
            self.placement_mgr.state_mgr.implant_transformed.connect(self._on_external_implant_transformed)
            if hasattr(self.placement_mgr.state_mgr, "implants_updated"):
                self.placement_mgr.state_mgr.implants_updated.connect(self.refresh_placed_implants_table)

        self.alignment_changed.connect(self._on_alignment_changed)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        self.splitter = QSplitter(Qt.Orientation.Vertical, self)

        # 상단 탭 위젯
        self.tab_widget = QTabWidget(self.splitter)

        # Tab 1: 임플란트
        tab_implant = self._create_implant_tab()
        self.tab_widget.addTab(tab_implant, "임플란트 (Implant)")

        # Tab 2: Abutment
        tab_abutment = self._create_abutment_tab()
        self.tab_widget.addTab(tab_abutment, "Abutment")

        # Tab 3: 보철물 (Crown)
        tab_prosthetic = self._create_prosthetic_tab()
        self.tab_widget.addTab(tab_prosthetic, "보철물 (Crown)")

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # 하단: 식립된 임플란트 목록
        grp_placed = self._create_placed_implants_group()
        self.splitter.addWidget(grp_placed)

        self.splitter.setStretchFactor(0, 7)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setSizes([595, 255])
        main_layout.addWidget(self.splitter)

    # -------------------------------------------------------------
    # Tab 1: 임플란트 탭
    # -------------------------------------------------------------
    def _create_implant_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 1. 라이브러리 그룹
        grp_lib = QGroupBox("식립할 임플란트 선택 및 사양 (Library)", container)
        lib_vbox = QVBoxLayout(grp_lib)
        lib_vbox.setContentsMargins(10, 8, 10, 8)
        lib_vbox.setSpacing(6)

        filter_bar = QHBoxLayout()
        filter_bar.addWidget(QLabel("직경 필터:"))
        self.combo_filter_dia = QComboBox(grp_lib)
        self.combo_filter_dia.addItems(["전체 직경 (모두보기)", "Ø3.0mm", "Ø3.5mm", "Ø4.0mm", "Ø4.5mm", "Ø5.0mm", "Ø6.0mm"])
        self.combo_filter_dia.currentTextChanged.connect(self._apply_filters)
        filter_bar.addWidget(self.combo_filter_dia)

        filter_bar.addSpacing(15)
        filter_bar.addWidget(QLabel("길이 필터:"))
        self.combo_filter_len = QComboBox(grp_lib)
        self.combo_filter_len.addItems(["전체 길이 (모두보기)", "5.0mm", "6.0mm", "7.0mm", "8.5mm", "10.0mm", "11.5mm"])
        self.combo_filter_len.currentTextChanged.connect(self._apply_filters)
        filter_bar.addWidget(self.combo_filter_len)
        filter_bar.addStretch()
        lib_vbox.addLayout(filter_bar)

        lib_inner = QHBoxLayout()
        self.table_library = QTableWidget(grp_lib)
        self.table_library.setColumnCount(6)
        self.table_library.setHorizontalHeaderLabels(["이름 / 사양", "직경", "길이", "Taper", "상부형태", "구분"])
        self.table_library.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_library.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_library.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_library.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_library.itemSelectionChanged.connect(self._on_lib_implant_selected)
        lib_inner.addWidget(self.table_library, 3)

        detail_container = QWidget(grp_lib)
        detail_container.setFixedWidth(170)
        d_layout = QVBoxLayout(detail_container)
        d_layout.setContentsMargins(0, 0, 0, 0)
        d_layout.setSpacing(4)
        self.library_3d_view = Implant3DPreviewWidget(detail_container)
        d_layout.addWidget(self.library_3d_view, 1)
        self.txt_implant_spec = QTextBrowser(detail_container)
        self.txt_implant_spec.setFixedHeight(85)
        self.txt_implant_spec.setStyleSheet("background-color: #090d16; border: 1px solid #1e293b; font-size: 11px;")
        d_layout.addWidget(self.txt_implant_spec)
        lib_inner.addWidget(detail_container, 1)
        lib_vbox.addLayout(lib_inner)

        # 액션 바
        action_bar = QHBoxLayout()
        self.chk_maxilla = QCheckBox("상악에 식립", grp_lib)
        self.chk_maxilla.toggled.connect(self.library_3d_view.set_inverted)
        action_bar.addWidget(self.chk_maxilla)
        action_bar.addStretch()

        self.btn_replace = QPushButton("Replace (선택한 임플란트로 교체)", grp_lib)
        self.btn_replace.setStyleSheet("background-color: #334155; color: #f8fafc; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_replace.clicked.connect(self._on_replace_clicked)
        action_bar.addWidget(self.btn_replace)

        self.btn_place = QPushButton("Place (3D 볼륨 및 단면에 식립)", grp_lib)
        self.btn_place.setIcon(IconFactory.implant(18))
        self.btn_place.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; padding: 6px 16px; border-radius: 4px;")
        self.btn_place.clicked.connect(self._on_place_clicked)
        action_bar.addWidget(self.btn_place)
        lib_vbox.addLayout(action_bar)
        layout.addWidget(grp_lib)

        # 2. 조작 방식 및 정렬 도구 그룹 (임플란트 가이드 그룹은 제거됨)
        grp_middle = QGroupBox("조작 방식 및 임플란트 정렬 도구 (Tools & Mode)", container)
        mid_layout = QVBoxLayout(grp_middle)
        mid_layout.setContentsMargins(10, 8, 10, 8)
        mid_layout.setSpacing(6)

        mode_bar = QHBoxLayout()
        mode_bar.addWidget(QLabel("<b>조작 방식:</b>"))
        self.rb_gizmo = QRadioButton("3D 기즈모(Gizmo)로 조절", grp_middle)
        self.rb_popup = QRadioButton("팝업 윈도우로 조절", grp_middle)
        self.rb_gizmo.setChecked(True)
        mode_bar.addWidget(self.rb_gizmo)
        mode_bar.addWidget(self.rb_popup)
        mode_bar.addStretch()
        mid_layout.addLayout(mode_bar)

        align_row = QHBoxLayout()
        self.btn_align_height = QPushButton("임플란트 높이 정렬", grp_middle)
        self.btn_align_height.setStyleSheet("background-color: #1e293b; color: #38bdf8; padding: 5px 10px; border-radius: 4px;")
        self.btn_align_height.clicked.connect(self._on_align_height_clicked)
        align_row.addWidget(self.btn_align_height)

        self.btn_align_angle = QPushButton("임플란트 각도 정렬", grp_middle)
        self.btn_align_angle.setStyleSheet("background-color: #1e293b; color: #38bdf8; padding: 5px 10px; border-radius: 4px;")
        self.btn_align_angle.clicked.connect(self._on_align_angle_clicked)
        align_row.addWidget(self.btn_align_angle)

        align_row.addSpacing(15)
        align_row.addWidget(QLabel("간격:"))
        self.slider_spacing = QSlider(Qt.Orientation.Horizontal, grp_middle)
        self.slider_spacing.setRange(10, 24)
        self.slider_spacing.setValue(14)  # 7.0 mm
        self.lbl_spacing_val = QLabel("7.0 mm", grp_middle)
        self.lbl_spacing_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_spacing.valueChanged.connect(lambda v: self.lbl_spacing_val.setText(f"{v*0.5:.1f} mm"))
        align_row.addWidget(self.slider_spacing)
        align_row.addWidget(self.lbl_spacing_val)

        self.btn_align_spacing = QPushButton("간격 정렬", grp_middle)
        self.btn_align_spacing.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 5px 12px; border-radius: 4px;")
        self.btn_align_spacing.clicked.connect(self._on_align_spacing_clicked)
        align_row.addWidget(self.btn_align_spacing)

        align_row.addSpacing(15)
        self.btn_undo = QPushButton("되돌리기 (Undo)", grp_middle)
        self.btn_undo.setStyleSheet("background-color: #334155; color: #f8fafc; padding: 5px 10px; border-radius: 4px;")
        self.btn_undo.clicked.connect(self._on_undo_clicked)
        align_row.addWidget(self.btn_undo)
        mid_layout.addLayout(align_row)

        layout.addWidget(grp_middle)
        return container

    # -------------------------------------------------------------
    # Tab 2: Abutment 탭
    # -------------------------------------------------------------
    def _create_abutment_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.lbl_abut_target_info = QLabel("선택된 임플란트: 없음", container)
        self.lbl_abut_target_info.setStyleSheet("background-color: #1e293b; color: #38bdf8; padding: 6px 10px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(self.lbl_abut_target_info)

        body_layout = QHBoxLayout()
        abut_box = QGroupBox("Abutment 사양 설정", container)
        a_grid = QGridLayout(abut_box)
        a_grid.setContentsMargins(10, 10, 10, 10)
        a_grid.setHorizontalSpacing(10)
        a_grid.setVerticalSpacing(8)

        self.lbl_abut_bottom_desc = QLabel("하부 규격: 임플란트 직경 연동, Hex 돌기 3.0mm 하부 돌출", abut_box)
        self.lbl_abut_bottom_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        a_grid.addWidget(self.lbl_abut_bottom_desc, 0, 0, 1, 3)

        a_grid.addWidget(QLabel("Collar Flare 각도:"), 1, 0)
        self.slider_collar_flare = QSlider(Qt.Orientation.Horizontal, abut_box)
        self.slider_collar_flare.setRange(5, 45)
        self.slider_collar_flare.setValue(30)
        self.lbl_flare_val = QLabel("30°", abut_box)
        self.lbl_flare_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_collar_flare.valueChanged.connect(self._on_abutment_param_changed)
        a_grid.addWidget(self.slider_collar_flare, 1, 1)
        a_grid.addWidget(self.lbl_flare_val, 1, 2)

        a_grid.addWidget(QLabel("Collar 길이:"), 2, 0)
        self.slider_collar_len = QSlider(Qt.Orientation.Horizontal, abut_box)
        self.slider_collar_len.setRange(4, 20)
        self.slider_collar_len.setValue(8) # 4.0 mm
        self.lbl_collar_len_val = QLabel("4.0 mm", abut_box)
        self.lbl_collar_len_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_collar_len.valueChanged.connect(self._on_abutment_param_changed)
        a_grid.addWidget(self.slider_collar_len, 2, 1)
        a_grid.addWidget(self.lbl_collar_len_val, 2, 2)

        a_grid.addWidget(QLabel("상부 (Morse) 길이:"), 3, 0)
        self.slider_upper_len = QSlider(Qt.Orientation.Horizontal, abut_box)
        self.slider_upper_len.setRange(6, 40)
        self.slider_upper_len.setValue(10) # 5.0 mm
        self.lbl_upper_len_val = QLabel("5.0 mm", abut_box)
        self.lbl_upper_len_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_upper_len.valueChanged.connect(self._on_abutment_param_changed)
        a_grid.addWidget(self.slider_upper_len, 3, 1)
        a_grid.addWidget(self.lbl_upper_len_val, 3, 2)

        a_grid.addWidget(QLabel("Angulation 기울기:"), 4, 0)
        self.slider_angulation = QSlider(Qt.Orientation.Horizontal, abut_box)
        self.slider_angulation.setRange(0, 30)
        self.slider_angulation.setValue(0)
        self.lbl_angulation_val = QLabel("0°", abut_box)
        self.lbl_angulation_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_angulation.valueChanged.connect(self._on_abutment_param_changed)
        a_grid.addWidget(self.slider_angulation, 4, 1)
        a_grid.addWidget(self.lbl_angulation_val, 4, 2)

        a_grid.addWidget(QLabel("기울기 방향 (회전):"), 5, 0)
        self.slider_ang_dir = QSlider(Qt.Orientation.Horizontal, abut_box)
        self.slider_ang_dir.setRange(0, 72)
        self.slider_ang_dir.setValue(0)
        self.lbl_ang_dir_val = QLabel("0°", abut_box)
        self.lbl_ang_dir_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_ang_dir.valueChanged.connect(self._on_abutment_param_changed)
        a_grid.addWidget(self.slider_ang_dir, 5, 1)
        a_grid.addWidget(self.lbl_ang_dir_val, 5, 2)

        c_box = QHBoxLayout()
        c_box.addWidget(QLabel("색상:"))
        self.rb_abut_gold = QRadioButton("Gold", abut_box)
        self.rb_abut_gold.setChecked(True)
        self.rb_abut_zirconia = QRadioButton("Zirconia", abut_box)
        self.rb_abut_metal = QRadioButton("Metal", abut_box)
        self.rb_abut_gold.toggled.connect(self._on_abutment_param_changed)
        self.rb_abut_zirconia.toggled.connect(self._on_abutment_param_changed)
        self.rb_abut_metal.toggled.connect(self._on_abutment_param_changed)
        c_box.addWidget(self.rb_abut_gold)
        c_box.addWidget(self.rb_abut_zirconia)
        c_box.addWidget(self.rb_abut_metal)
        c_box.addStretch()
        a_grid.addLayout(c_box, 6, 0, 1, 3)

        body_layout.addWidget(abut_box, 3)

        prev_box = QGroupBox("Abutment 3D 미리보기", container)
        p_vbox = QVBoxLayout(prev_box)
        p_vbox.setContentsMargins(6, 6, 6, 6)
        self.abutment_3d_view = Implant3DPreviewWidget(prev_box)
        p_vbox.addWidget(self.abutment_3d_view)
        body_layout.addWidget(prev_box, 2)
        layout.addLayout(body_layout)

        # Abutment 액션 버튼들
        btn_bar = QHBoxLayout()
        self.btn_attach_abutment = QPushButton("Abutment 장착하기", container)
        self.btn_attach_abutment.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_attach_abutment.clicked.connect(self._on_attach_abutment_clicked)
        btn_bar.addWidget(self.btn_attach_abutment)

        self.btn_remove_abutment = QPushButton("Abutment 제거", container)
        self.btn_remove_abutment.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_remove_abutment.clicked.connect(self._on_remove_abutment_clicked)
        btn_bar.addWidget(self.btn_remove_abutment)

        self.btn_replace_abutment = QPushButton("Abutment 교체", container)
        self.btn_replace_abutment.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_replace_abutment.clicked.connect(self._on_replace_abutment_clicked)
        btn_bar.addWidget(self.btn_replace_abutment)

        self.btn_remove_all_abutments = QPushButton("Abutment 전부 제거", container)
        self.btn_remove_all_abutments.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_remove_all_abutments.clicked.connect(self._on_remove_all_abutments_clicked)
        btn_bar.addWidget(self.btn_remove_all_abutments)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)
        return container

    # -------------------------------------------------------------
    # Tab 3: 보철물 (Crown) 탭
    # -------------------------------------------------------------
    def _create_prosthetic_tab(self) -> QWidget:
        container = QWidget()
        scroll = QScrollArea(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.lbl_prost_target_info = QLabel("선택된 임플란트: 없음", content)
        self.lbl_prost_target_info.setStyleSheet("background-color: #1e293b; color: #f59e0b; padding: 6px 10px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(self.lbl_prost_target_info)

        # 치아 차트 그룹
        grp_tooth = QGroupBox("치아 번호 선택 (Tooth Chart)", content)
        t_vbox = QVBoxLayout(grp_tooth)
        t_vbox.setContentsMargins(10, 8, 10, 8)
        self.tooth_chart = ToothChartWidget(grp_tooth)
        self.tooth_chart.tooth_selected.connect(self._on_chart_tooth_selected)
        self.tooth_chart.tooth_hovered.connect(lambda num, name: self.lbl_tooth_hover.setText(f"선택/호버 치아: {name}" if name else ""))
        t_vbox.addWidget(self.tooth_chart)

        self.lbl_tooth_hover = QLabel("선택된 치아: 16 (상악 우측 제1대구치)", grp_tooth)
        self.lbl_tooth_hover.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        t_vbox.addWidget(self.lbl_tooth_hover)
        layout.addWidget(grp_tooth)

        # 치관 사양 및 미리보기
        body_box = QHBoxLayout()
        spec_box = QGroupBox("보철물 (Crown) 사양 설정", content)
        s_grid = QGridLayout(spec_box)
        s_grid.setContentsMargins(10, 10, 10, 10)
        s_grid.setHorizontalSpacing(10)
        s_grid.setVerticalSpacing(8)

        self.lbl_crown_bottom_desc = QLabel("하단 직경: Abutment Collar 상단 직경 - 1mm 자동 연동", spec_box)
        self.lbl_crown_bottom_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        s_grid.addWidget(self.lbl_crown_bottom_desc, 0, 0, 1, 3)

        s_grid.addWidget(QLabel("치아 길이:"), 1, 0)
        self.slider_tooth_len = QSlider(Qt.Orientation.Horizontal, spec_box)
        self.slider_tooth_len.setRange(50, 150)
        self.slider_tooth_len.setValue(75) # 7.5 mm
        self.lbl_tooth_len_val = QLabel("7.5 mm", spec_box)
        self.lbl_tooth_len_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_tooth_len.valueChanged.connect(self._on_prosthetic_param_changed)
        s_grid.addWidget(self.slider_tooth_len, 1, 1)
        s_grid.addWidget(self.lbl_tooth_len_val, 1, 2)

        s_grid.addWidget(QLabel("근원심 폭경:"), 2, 0)
        self.slider_tooth_w = QSlider(Qt.Orientation.Horizontal, spec_box)
        self.slider_tooth_w.setRange(40, 150)
        self.slider_tooth_w.setValue(100) # 10.0 mm
        self.lbl_tooth_w_val = QLabel("10.0 mm", spec_box)
        self.lbl_tooth_w_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_tooth_w.valueChanged.connect(self._on_prosthetic_param_changed)
        s_grid.addWidget(self.slider_tooth_w, 2, 1)
        s_grid.addWidget(self.lbl_tooth_w_val, 2, 2)

        s_grid.addWidget(QLabel("협설 폭경:"), 3, 0)
        self.slider_tooth_d = QSlider(Qt.Orientation.Horizontal, spec_box)
        self.slider_tooth_d.setRange(40, 150)
        self.slider_tooth_d.setValue(110) # 11.0 mm
        self.lbl_tooth_d_val = QLabel("11.0 mm", spec_box)
        self.lbl_tooth_d_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_tooth_d.valueChanged.connect(self._on_prosthetic_param_changed)
        s_grid.addWidget(self.slider_tooth_d, 3, 1)
        s_grid.addWidget(self.lbl_tooth_d_val, 3, 2)

        s_grid.addWidget(QLabel("치아 회전 방향:"), 4, 0)
        self.slider_tooth_rot = QSlider(Qt.Orientation.Horizontal, spec_box)
        self.slider_tooth_rot.setRange(-180, 180)
        self.slider_tooth_rot.setValue(0)
        self.lbl_tooth_rot_val = QLabel("0°", spec_box)
        self.lbl_tooth_rot_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_tooth_rot.valueChanged.connect(self._on_prosthetic_param_changed)
        s_grid.addWidget(self.slider_tooth_rot, 4, 1)
        s_grid.addWidget(self.lbl_tooth_rot_val, 4, 2)

        m_box = QHBoxLayout()
        m_box.addWidget(QLabel("재료:"))
        self.rb_mat_ceramic = QRadioButton("Ceramic", spec_box)
        self.rb_mat_ceramic.setChecked(True)
        self.rb_mat_zirconia = QRadioButton("Zirconia", spec_box)
        self.rb_mat_gold = QRadioButton("Gold", spec_box)
        self.rb_mat_metal = QRadioButton("Metal", spec_box)
        self.rb_mat_ceramic.toggled.connect(self._on_prosthetic_param_changed)
        self.rb_mat_zirconia.toggled.connect(self._on_prosthetic_param_changed)
        self.rb_mat_gold.toggled.connect(self._on_prosthetic_param_changed)
        self.rb_mat_metal.toggled.connect(self._on_prosthetic_param_changed)
        m_box.addWidget(self.rb_mat_ceramic)
        m_box.addWidget(self.rb_mat_zirconia)
        m_box.addWidget(self.rb_mat_gold)
        m_box.addWidget(self.rb_mat_metal)
        m_box.addStretch()
        s_grid.addLayout(m_box, 5, 0, 1, 3)

        body_box.addWidget(spec_box, 3)

        crown_prev_box = QGroupBox("보철물 3D 해부학적 미리보기", content)
        c_vbox = QVBoxLayout(crown_prev_box)
        c_vbox.setContentsMargins(6, 6, 6, 6)
        self.prosthetic_3d_view = Implant3DPreviewWidget(crown_prev_box)
        c_vbox.addWidget(self.prosthetic_3d_view)
        body_box.addWidget(crown_prev_box, 2)
        layout.addLayout(body_box)

        p_btn_bar = QHBoxLayout()
        self.btn_attach_crown = QPushButton("보철물 장착하기", content)
        self.btn_attach_crown.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_attach_crown.clicked.connect(self._on_attach_prosthetic_clicked)
        p_btn_bar.addWidget(self.btn_attach_crown)

        self.btn_remove_crown = QPushButton("보철물 제거", content)
        self.btn_remove_crown.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_remove_crown.clicked.connect(self._on_remove_prosthetic_clicked)
        p_btn_bar.addWidget(self.btn_remove_crown)

        self.btn_replace_crown = QPushButton("보철물 교체", content)
        self.btn_replace_crown.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_replace_crown.clicked.connect(self._on_replace_prosthetic_clicked)
        p_btn_bar.addWidget(self.btn_replace_crown)

        self.btn_remove_all_crowns = QPushButton("보철물 전부 제거", content)
        self.btn_remove_all_crowns.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_remove_all_crowns.clicked.connect(self._on_remove_all_prosthetics_clicked)
        p_btn_bar.addWidget(self.btn_remove_all_crowns)
        p_btn_bar.addStretch()
        layout.addLayout(p_btn_bar)

        scroll.setWidget(content)
        top_layout = QVBoxLayout(container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(scroll)
        return container

    # -------------------------------------------------------------
    # 하단: 식립된 임플란트 목록 그룹
    # -------------------------------------------------------------
    def _create_placed_implants_group(self) -> QWidget:
        grp = QGroupBox("식립된 임플란트 목록 (Placed Implants - 다중선택: Ctrl + Click)", self)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        inner = QHBoxLayout()
        self.table_placed = QTableWidget(grp)
        self.table_placed.setColumnCount(9)
        self.table_placed.setHorizontalHeaderLabels([
            "ID / 순서", "이름 / 사양", "직경", "길이", "보철",
            "위치 (X, Y, Z)", "각도 (R, P, Y)", "선택상태", "보이기"
        ])
        h = self.table_placed.horizontalHeader()
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_placed.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_placed.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table_placed.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_placed.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_placed.customContextMenuRequested.connect(self._on_placed_table_context_menu)
        self.table_placed.itemSelectionChanged.connect(self._on_placed_table_selection_changed)
        inner.addWidget(self.table_placed, 3)

        self.txt_placed_detail = QTextBrowser(grp)
        self.txt_placed_detail.setStyleSheet("background-color: #090d16; border: 1px solid #1e293b; font-size: 11px; padding: 4px;")
        inner.addWidget(self.txt_placed_detail, 1)
        layout.addLayout(inner)

        # 하단 액션 버튼 바
        bot_bar = QHBoxLayout()
        self.btn_delete_all = QPushButton("임플란트 전체 삭제", grp)
        self.btn_delete_all.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_delete_all.clicked.connect(self._on_delete_all_clicked)
        bot_bar.addWidget(self.btn_delete_all)

        self.btn_delete_selected = QPushButton("선택한 임플란트 지우기", grp)
        self.btn_delete_selected.setStyleSheet("background-color: #b91c1c; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.btn_delete_selected.clicked.connect(self._on_delete_selected_clicked)
        bot_bar.addWidget(self.btn_delete_selected)

        bot_bar.addSpacing(15)
        self.btn_save_info = QPushButton("식립 정보 저장 (Save)", grp)
        self.btn_save_info.setStyleSheet("background-color: #334155; color: #f8fafc; padding: 6px 12px; border-radius: 4px;")
        self.btn_save_info.clicked.connect(self._on_save_info_clicked)
        bot_bar.addWidget(self.btn_save_info)

        self.btn_load_info = QPushButton("식립 정보 불러오기 (Load)", grp)
        self.btn_load_info.setStyleSheet("background-color: #334155; color: #f8fafc; padding: 6px 12px; border-radius: 4px;")
        self.btn_load_info.clicked.connect(self._on_load_info_clicked)
        bot_bar.addWidget(self.btn_load_info)

        bot_bar.addStretch()
        btn_close = QPushButton("닫기", grp)
        btn_close.setFixedWidth(90)
        btn_close.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; padding: 6px 14px; border-radius: 4px;")
        btn_close.clicked.connect(self.hide)
        bot_bar.addWidget(btn_close)

        layout.addLayout(bot_bar)
        return grp

    # -------------------------------------------------------------
    # Handlers & Logic
    # -------------------------------------------------------------
    def _load_implant_list(self):
        self.all_implants = ImplantLibraryManager.list_all_implants()
        self._apply_filters()

    def _apply_filters(self):
        sel_dia = self.combo_filter_dia.currentText()
        sel_len = self.combo_filter_len.currentText()
        self.filtered_implants = []
        for imp in self.all_implants:
            if sel_dia != "전체 직경 (모두보기)":
                val = float(sel_dia.replace("Ø", "").replace("mm", "").strip())
                if abs(imp.diameter_mm - val) > 0.05:
                    continue
            if sel_len != "전체 길이 (모두보기)":
                val = float(sel_len.replace("mm", "").strip())
                if abs(imp.length_mm - val) > 0.05:
                    continue
            self.filtered_implants.append(imp)

        self.table_library.setRowCount(len(self.filtered_implants))
        for row, imp in enumerate(self.filtered_implants):
            self.table_library.setItem(row, 0, QTableWidgetItem(imp.name))
            self.table_library.setItem(row, 1, QTableWidgetItem(f"Ø{imp.diameter_mm:.1f}"))
            self.table_library.setItem(row, 2, QTableWidgetItem(f"{imp.length_mm:.1f}"))
            self.table_library.setItem(row, 3, QTableWidgetItem(f"{imp.taper_deg:.0f}°"))
            self.table_library.setItem(row, 4, QTableWidgetItem(imp.top_type.capitalize()))
            self.table_library.setItem(row, 5, QTableWidgetItem("사용자정의" if imp.is_custom else "기본제공"))
        if self.filtered_implants:
            self.table_library.selectRow(0)

    def _on_lib_implant_selected(self):
        rows = self.table_library.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self.filtered_implants):
            self.selected_lib_spec = self.filtered_implants[row]
            poly = ImplantLibraryManager.load_implant_polydata(self.selected_lib_spec)
            self.library_3d_view.set_polydata(poly, force_reset_camera=True)
            self.txt_implant_spec.setHtml(f"""
                <b>{self.selected_lib_spec.name}</b><br>
                규격: Ø{self.selected_lib_spec.diameter_mm:.1f} x {self.selected_lib_spec.length_mm:.1f} mm | Taper {self.selected_lib_spec.taper_deg:.0f}°<br>
                상부: {self.selected_lib_spec.top_type} | 나사산: {self.selected_lib_spec.thread_type}
            """)

    def _on_place_clicked(self):
        if not self.selected_lib_spec:
            QMessageBox.warning(self, "경고", "식립할 임플란트를 먼저 선택해주세요.")
            return
        is_maxilla = self.chk_maxilla.isChecked()

        # 현재 커서 위치에 새 임플란트 식립 (기존 임플란트를 대체하지 않고 새 항목 추가)
        pos = self.placement_mgr.current_cursor
        if (pos[0] == 0.0 and pos[1] == 0.0 and pos[2] == 0.0):
            parent_w = self.parent()
            if hasattr(parent_w, "state_mgr"):
                pos = parent_w.state_mgr.current_coords

        new_imp_id = self.placement_mgr.place_implant(
            spec=self.selected_lib_spec,
            initial_position=pos,
            is_maxilla=is_maxilla
        )
        self.placement_mgr.select_implant(new_imp_id)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()
        self.placement_requested.emit(self.selected_lib_spec, is_maxilla)

    def _on_replace_clicked(self):
        if not self.selected_lib_spec or not self.placement_mgr.selected_ids:
            QMessageBox.warning(self, "경고", "교체할 임플란트 사양과 식립된 임플란트를 선택해주세요.")
            return
        target_id = self.placement_mgr.selected_ids[0]
        self.placement_mgr.replace_implant(target_id, self.selected_lib_spec)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_align_height_clicked(self):
        self.placement_mgr.align_heights_to_reference()
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_align_angle_clicked(self):
        self.placement_mgr.align_angles_to_reference()
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_align_spacing_clicked(self):
        spacing = self.slider_spacing.value() * 0.5
        self.placement_mgr.align_spacing_sequence(spacing)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_undo_clicked(self):
        self.placement_mgr.undo()
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    # -------------------------------------------------------------
    # 식립된 임플란트 목록 갱신 및 테이블 이벤트
    # -------------------------------------------------------------
    def refresh_placed_implants_table(self):
        implants = list(self.placement_mgr.implants.values())
        self.table_placed.blockSignals(True)
        self.table_placed.setRowCount(len(implants))

        for row, imp in enumerate(implants):
            self.table_placed.setItem(row, 0, QTableWidgetItem(f"#{imp.sequence_idx}"))
            self.table_placed.setItem(row, 1, QTableWidgetItem(imp.spec.name))
            self.table_placed.setItem(row, 2, QTableWidgetItem(f"Ø{imp.spec.diameter_mm:.1f}"))
            self.table_placed.setItem(row, 3, QTableWidgetItem(f"{imp.spec.length_mm:.1f}"))

            prost_text = "없음"
            if imp.prosthetic:
                prost_text = f"#{imp.prosthetic.tooth_number}"
            elif imp.abutment:
                prost_text = "Abutment"
            self.table_placed.setItem(row, 4, QTableWidgetItem(prost_text))

            pos_str = f"({imp.position[0]:.1f}, {imp.position[1]:.1f}, {imp.position[2]:.1f})"
            self.table_placed.setItem(row, 5, QTableWidgetItem(pos_str))

            rot_str = f"({imp.orientation[1]:.0f}°, {imp.orientation[2]:.0f}°, {imp.orientation[0]:.0f}°)"
            self.table_placed.setItem(row, 6, QTableWidgetItem(rot_str))

            sel_text = "-"
            if imp.id in self.placement_mgr.selected_ids:
                idx = self.placement_mgr.selected_ids.index(imp.id)
                sel_text = "1st 기준 (Blue)" if idx == 0 else "2nd+ (Green)"
            self.table_placed.setItem(row, 7, QTableWidgetItem(sel_text))

            chk_item = QTableWidgetItem("보임" if imp.is_visible else "숨김")
            chk_item.setCheckState(Qt.CheckState.Checked if imp.is_visible else Qt.CheckState.Unchecked)
            self.table_placed.setItem(row, 8, chk_item)

            if imp.is_selected:
                self.table_placed.selectionModel().select(
                    self.table_placed.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
                )

        self.table_placed.blockSignals(False)
        self._update_placed_detail_panel()
        self._update_tab_targets()

    def _on_placed_table_selection_changed(self):
        rows = self.table_placed.selectionModel().selectedRows()
        implants = list(self.placement_mgr.implants.values())
        if not rows:
            self.placement_mgr.clear_selection()
        else:
            multi = len(rows) > 1
            if not multi:
                idx = rows[0].row()
                if 0 <= idx < len(implants):
                    self.placement_mgr.select_implant(implants[idx].id, multi_select=False)
                    self.implant_selected.emit(implants[idx].id)
            else:
                self.placement_mgr.clear_selection()
                for r in rows:
                    idx = r.row()
                    if 0 <= idx < len(implants):
                        self.placement_mgr.select_implant(implants[idx].id, multi_select=True)

        self._update_placed_detail_panel()
        self._update_tab_targets()
        # 선택 변경 시 뷰포트 즉시 렌더링 동기화
        if hasattr(self.placement_mgr, "state_mgr") and self.placement_mgr.state_mgr:
            self.placement_mgr.state_mgr.request_render_all()

    def _update_placed_detail_panel(self):
        sel_imp = self.placement_mgr.get_selected_implant()
        if not sel_imp:
            self.txt_placed_detail.setText("[선택된 임플란트 상세]\n선택된 항목 없음")
            return

        detail_text = f"""<b>[선택된 임플란트 상세]</b><br>
        • 이름: {sel_imp.spec.name}<br>
        • 규격: Ø{sel_imp.spec.diameter_mm:.1f} x {sel_imp.spec.length_mm:.1f} mm<br>
        • 위치: ({sel_imp.position[0]:.1f}, {sel_imp.position[1]:.1f}, {sel_imp.position[2]:.1f}) mm<br>
        • 각도: Yaw={sel_imp.orientation[0]:.1f}°, Pitch={sel_imp.orientation[1]:.1f}°, Roll={sel_imp.orientation[2]:.1f}°<br>
        • Abutment: {'장착됨' if sel_imp.abutment else '없음'}<br>
        • 보철물: {f'치아 #{sel_imp.prosthetic.tooth_number}' if sel_imp.prosthetic else '없음'}
        """
        self.txt_placed_detail.setHtml(detail_text)

    def _update_tab_targets(self):
        sel_imp = self.placement_mgr.get_selected_implant()
        if sel_imp:
            self.lbl_abut_target_info.setText(f"선택된 임플란트: #{sel_imp.sequence_idx} {sel_imp.spec.name}")
            if sel_imp.abutment:
                self.lbl_prost_target_info.setText(f"선택된 임플란트: #{sel_imp.sequence_idx} (Abutment 장착됨)")
            else:
                self.lbl_prost_target_info.setText(f"선택된 임플란트: #{sel_imp.sequence_idx} (Abutment 없음 - 보철물을 장착하려면 먼저 Abutment를 장착하세요)")
        else:
            self.lbl_abut_target_info.setText("선택된 임플란트: 없음 (목록에서 임플란트를 선택하세요)")
            self.lbl_prost_target_info.setText("선택된 임플란트: 없음")

    def _on_placed_table_context_menu(self, pos: QPoint):
        row = self.table_placed.rowAt(pos.y())
        implants = list(self.placement_mgr.implants.values())
        target_imp = implants[row] if 0 <= row < len(implants) else None

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 4px; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background-color: #0284c7; color: #ffffff; }
        """)

        act_show_all = menu.addAction("전체 보이기")
        act_hide_all = menu.addAction("전체 감추기")
        menu.addSeparator()

        act_toggle = None
        act_delete = None
        if target_imp:
            act_toggle = menu.addAction("이 임플란트 감추기" if target_imp.is_visible else "이 임플란트 보이기")
            act_delete = menu.addAction("이 임플란트 삭제")
            menu.addSeparator()

        act_delete_all = menu.addAction("임플란트 전체 삭제")

        action = menu.exec(self.table_placed.mapToGlobal(pos))
        if action == act_show_all:
            self.placement_mgr.set_visibility_all(True)
            self.refresh_placed_implants_table()
            self.alignment_changed.emit()
        elif action == act_hide_all:
            self.placement_mgr.set_visibility_all(False)
            self.refresh_placed_implants_table()
            self.alignment_changed.emit()
        elif act_toggle and action == act_toggle:
            self.placement_mgr.set_implant_visibility(target_imp.id, not target_imp.is_visible)
            self.refresh_placed_implants_table()
            self.alignment_changed.emit()
        elif act_delete and action == act_delete:
            self.placement_mgr.delete_implant(target_imp.id)
            self.refresh_placed_implants_table()
            self.alignment_changed.emit()
        elif action == act_delete_all:
            self._on_delete_all_clicked()

    def _on_delete_selected_clicked(self):
        if not self.placement_mgr.selected_ids:
            QMessageBox.information(self, "알림", "삭제할 임플란트를 먼저 선택해주세요.")
            return
        reply = QMessageBox.question(
            self, "삭제 확인", "선택한 임플란트를 지우시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            for s_id in list(self.placement_mgr.selected_ids):
                self.placement_mgr.delete_implant(s_id)
            self.refresh_placed_implants_table()
            self.alignment_changed.emit()

    def _on_delete_all_clicked(self):
        if not self.placement_mgr.implants:
            return
        reply = QMessageBox.question(
            self, "전체 삭제 확인", "식립된 모든 임플란트를 전체 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.placement_mgr.delete_all_implants()
            self.refresh_placed_implants_table()
            self.alignment_changed.emit()

    def _on_save_info_clicked(self):
        if not self.placement_mgr.implants:
            QMessageBox.information(self, "알림", "저장할 식립 임플란트 정보가 없습니다.")
            return
        fpath, _ = QFileDialog.getSaveFileName(self, "식립 정보 저장", "", "JSON 파일 (*.json)")
        if not fpath:
            return
        data = []
        for imp in self.placement_mgr.implants.values():
            item = {
                "id": imp.id,
                "spec": copy.deepcopy(imp.spec.__dict__),
                "position": imp.position.tolist(),
                "orientation": imp.orientation.tolist(),
            }
            if imp.abutment:
                item["abutment"] = copy.deepcopy(imp.abutment.__dict__)
            if imp.prosthetic:
                item["prosthetic"] = copy.deepcopy(imp.prosthetic.__dict__)
            data.append(item)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, "저장 완료", "식립 정보가 성공적으로 저장되었습니다.")

    def _on_load_info_clicked(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "식립 정보 불러오기", "", "JSON 파일 (*.json)")
        if not fpath or not os.path.exists(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.placement_mgr.delete_all_implants()
            for item in data:
                spec = ImplantSpec(**item["spec"])
                imp_id = self.placement_mgr.place_implant(spec, initial_position=tuple(item["position"]))
                imp = self.placement_mgr.implants[imp_id]
                imp.orientation = np.array(item["orientation"])
                if "abutment" in item:
                    self.placement_mgr.attach_abutment(imp_id, AbutmentSpec(**item["abutment"]))
                if "prosthetic" in item:
                    self.placement_mgr.attach_prosthetic(imp_id, ProstheticSpec(**item["prosthetic"]))
                self.placement_mgr._update_implant_transform(imp)
            self.refresh_placed_implants_table()
            self.alignment_changed.emit()
            QMessageBox.information(self, "불러오기 완료", f"{len(data)}개의 임플란트 정보를 불러왔습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"불러오기 실패:\n{str(e)}")

    # -------------------------------------------------------------
    # Abutment 및 Prosthetic 설정 변경 & 3D 미리보기
    # -------------------------------------------------------------
    def _on_tab_changed(self, index: int):
        self._update_tab_targets()
        if index == 1:
            self._update_abutment_preview()
        elif index == 2:
            self._update_prosthetic_preview()

    def _on_abutment_param_changed(self):
        self.lbl_flare_val.setText(f"{self.slider_collar_flare.value()}°")
        self.lbl_collar_len_val.setText(f"{self.slider_collar_len.value()*0.5:.1f} mm")
        self.lbl_upper_len_val.setText(f"{self.slider_upper_len.value()*0.5:.1f} mm")
        self.lbl_angulation_val.setText(f"{self.slider_angulation.value()}°")
        self.lbl_ang_dir_val.setText(f"{self.slider_ang_dir.value()*5.0:.0f}°")
        self._update_abutment_preview()

    def _get_current_abutment_spec(self) -> AbutmentSpec:
        sel_imp = self.placement_mgr.get_selected_implant()
        dia = sel_imp.spec.diameter_mm if sel_imp else 4.0
        flare = float(self.slider_collar_flare.value())
        c_len = self.slider_collar_len.value() * 0.5
        u_len = self.slider_upper_len.value() * 0.5
        ang = float(self.slider_angulation.value())
        ang_dir = float(self.slider_ang_dir.value() * 5.0)

        col_type = "gold"
        col_rgb = (0.92, 0.75, 0.20)
        if self.rb_abut_zirconia.isChecked():
            col_type = "Zirconia"
            col_rgb = (0.92, 0.88, 0.78)
        elif self.rb_abut_metal.isChecked():
            col_type = "Metal"
            col_rgb = (0.35, 0.38, 0.40)

        return AbutmentSpec(
            diameter_mm=dia,
            flare_deg=flare,
            collar_len_mm=c_len,
            upper_len_mm=u_len,
            upper_angle_deg=ang,
            upper_dir_deg=ang_dir,
            color_type=col_type,
            color_rgb=col_rgb
        )

    def _update_abutment_preview(self):
        spec = self._get_current_abutment_spec()
        poly = generate_abutment_polydata(spec)
        self.abutment_3d_view.set_polydata(poly, force_reset_camera=True)
        ar, ag, ab = spec.color_rgb
        self.abutment_3d_view.actor.GetProperty().SetColor(ar, ag, ab)

    def _on_attach_abutment_clicked(self):
        sel_imp = self.placement_mgr.get_selected_implant()
        if not sel_imp:
            QMessageBox.warning(self, "경고", "Abutment를 장착할 임플란트를 먼저 선택해주세요.")
            return
        spec = self._get_current_abutment_spec()
        self.placement_mgr.attach_abutment(sel_imp.id, spec)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_remove_abutment_clicked(self):
        sel_imp = self.placement_mgr.get_selected_implant()
        if not sel_imp:
            return
        self.placement_mgr.remove_abutment(sel_imp.id)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_replace_abutment_clicked(self):
        sel_imp = self.placement_mgr.get_selected_implant()
        if not sel_imp:
            return
        spec = self._get_current_abutment_spec()
        self.placement_mgr.replace_abutment(sel_imp.id, spec)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_remove_all_abutments_clicked(self):
        self.placement_mgr.remove_all_abutments()
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_chart_tooth_selected(self, tooth_num: int):
        dims = get_default_tooth_dimensions(tooth_num)
        self.slider_tooth_len.setValue(int(round(dims[0] * 10)))
        self.slider_tooth_w.setValue(int(round(dims[1] * 10)))
        self.slider_tooth_d.setValue(int(round(dims[2] * 10)))
        self.lbl_tooth_hover.setText(f"선택된 치아: {tooth_num} ({TOOTH_NAMES.get(tooth_num, '')})")
        self._update_prosthetic_preview()

    def _on_prosthetic_param_changed(self):
        self.lbl_tooth_len_val.setText(f"{self.slider_tooth_len.value()*0.1:.1f} mm")
        self.lbl_tooth_w_val.setText(f"{self.slider_tooth_w.value()*0.1:.1f} mm")
        self.lbl_tooth_d_val.setText(f"{self.slider_tooth_d.value()*0.1:.1f} mm")
        self.lbl_tooth_rot_val.setText(f"{self.slider_tooth_rot.value()}°")
        self._update_prosthetic_preview()

    def _get_current_prosthetic_spec(self) -> ProstheticSpec:
        sel_imp = self.placement_mgr.get_selected_implant()
        b_dia = 3.5
        if sel_imp and sel_imp.abutment:
            b_dia = max(1.5, sel_imp.abutment.collar_top_diameter_mm - 1.0)

        t_num = self.tooth_chart.selected_tooth
        length = self.slider_tooth_len.value() * 0.1
        width = self.slider_tooth_w.value() * 0.1
        depth = self.slider_tooth_d.value() * 0.1
        rot = float(self.slider_tooth_rot.value())

        mat = "Ceramic"
        if self.rb_mat_zirconia.isChecked():
            mat = "Zirconia"
        elif self.rb_mat_gold.isChecked():
            mat = "Gold"
        elif self.rb_mat_metal.isChecked():
            mat = "Metal"

        return ProstheticSpec(
            tooth_number=t_num,
            base_diameter_mm=b_dia,
            length_mm=length,
            width_mm=width,
            depth_mm=depth,
            material=mat,
            rotation_deg=rot
        )

    def _update_prosthetic_preview(self):
        spec = self._get_current_prosthetic_spec()
        poly = generate_prosthetic_polydata(spec)
        self.prosthetic_3d_view.set_polydata(poly, force_reset_camera=True)
        pr, pg, pb = get_material_color_rgb(spec.material)
        self.prosthetic_3d_view.actor.GetProperty().SetColor(pr, pg, pb)

    def _on_attach_prosthetic_clicked(self):
        sel_imp = self.placement_mgr.get_selected_implant()
        if not sel_imp:
            QMessageBox.warning(self, "경고", "보철물을 장착할 임플란트를 먼저 선택해주세요.")
            return
        if not sel_imp.abutment:
            QMessageBox.warning(self, "경고", "보철물을 장착하려면 먼저 Abutment를 장착해야 합니다.")
            return
        spec = self._get_current_prosthetic_spec()
        self.placement_mgr.attach_prosthetic(sel_imp.id, spec)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_remove_prosthetic_clicked(self):
        sel_imp = self.placement_mgr.get_selected_implant()
        if not sel_imp:
            return
        self.placement_mgr.remove_prosthetic(sel_imp.id)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_replace_prosthetic_clicked(self):
        sel_imp = self.placement_mgr.get_selected_implant()
        if not sel_imp or not sel_imp.abutment:
            return
        spec = self._get_current_prosthetic_spec()
        self.placement_mgr.replace_prosthetic(sel_imp.id, spec)
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_remove_all_prosthetics_clicked(self):
        self.placement_mgr.remove_all_prosthetics()
        self.refresh_placed_implants_table()
        self.alignment_changed.emit()

    def _on_external_implant_selected(self, implant_id: str):
        """2D 뷰 또는 3D 뷰에서 임플란트 선택 변경 시 테이블 선택 동기화"""
        implants = list(self.placement_mgr.implants.values())
        self.table_placed.blockSignals(True)
        if not implant_id:
            self.table_placed.clearSelection()
        else:
            for row, imp in enumerate(implants):
                if imp.id == implant_id:
                    self.table_placed.selectRow(row)
                    break
        self.table_placed.blockSignals(False)
        self._update_placed_detail_panel()
        self._update_tab_targets()

    def _on_external_implant_transformed(self, implant_id: str):
        """2D/3D 기즈모 조작으로 임플란트 위치/각도가 변경될 때 테이블 좌표/각도 실시간 갱신"""
        implants = list(self.placement_mgr.implants.values())
        self.table_placed.blockSignals(True)
        for row, imp in enumerate(implants):
            if imp.id == implant_id:
                pos_str = f"({imp.position[0]:.1f}, {imp.position[1]:.1f}, {imp.position[2]:.1f})"
                rot_str = f"({imp.orientation[1]:.0f}°, {imp.orientation[2]:.0f}°, {imp.orientation[0]:.0f}°)"
                if self.table_placed.item(row, 5):
                    self.table_placed.item(row, 5).setText(pos_str)
                if self.table_placed.item(row, 6):
                    self.table_placed.item(row, 6).setText(rot_str)
                break
        self.table_placed.blockSignals(False)
        self._update_placed_detail_panel()

    def _on_alignment_changed(self):
        """정렬 및 편집 적용 시 식립 목록 테이블 및 2D/3D 뷰포트 즉시 갱신"""
        self.refresh_placed_implants_table()
        if hasattr(self.placement_mgr, "state_mgr") and self.placement_mgr.state_mgr:
            self.placement_mgr.state_mgr.request_render_all()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
