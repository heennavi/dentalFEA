"""
result_panel.py
Left sidebar control panel widget for Phase 4 FEA simulation results.

Contains 4 sections:
1. Result Import (.frd file status, open button, increment/step combo)
2. Display Settings (field, colormap, range mode, deformation scale, part visibility/opacity, section clip, edges)
3. Probe & Statistics (peak stress marker, interactive click probe, statistics table, equilibrium balance)
4. Export Placeholders (Report, CSV export buttons for Step 4-7)
"""

from typing import Optional, Dict, Any, List
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QRadioButton, QButtonGroup, QDoubleSpinBox, QSlider,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QSizePolicy, QFileDialog, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, Signal

from fea.models import MeshModel
from fea.results import ResultSet
from dental_fea_app.core.state_manager import StateManager


class ResultPanel(QWidget):
    """
    Control panel widget for viewing and interrogating FEA results.
    """
    # UI Interaction Signals
    open_frd_requested = Signal()
    step_changed = Signal(int)
    field_changed = Signal(str)
    colormap_changed = Signal(str)
    range_mode_changed = Signal(str, object)  # (mode, manual_tuple or None)
    active_part_changed = Signal(str)
    deform_scale_changed = Signal(float)
    part_visibility_changed = Signal(str, bool)
    part_opacity_changed = Signal(str, float)
    clip_changed = Signal(bool, str, float)   # (enabled, axis_str, offset_mm)
    edges_toggled = Signal(bool)
    load_arrow_toggled = Signal(bool)
    mark_max_requested = Signal(str, str)     # (field, part_name)
    probe_toggled = Signal(bool)
    peri_radius_changed = Signal(float)



    # 내보내기 및 번들 시그널 (Step 4-7)
    export_csv_requested = Signal()
    export_stats_requested = Signal()
    save_screenshot_requested = Signal(int)  # scale factor (1 or 2)
    save_bundle_requested = Signal()
    load_bundle_requested = Signal()


    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self._part_widgets: Dict[str, Dict[str, Any]] = {}
        self._is_updating = False

        self._init_ui()
        self.set_controls_enabled(False)

    def _init_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # 0. 경고 및 상태 배너
        self.banner_stale = QLabel("⚠️ 결과가 현재 모델과 맞지 않음, 다시 해석 필요", self)
        self.banner_stale.setStyleSheet("""
            background-color: #5c3818;
            color: #e5c07b;
            border: 1px solid #d19a66;
            border-radius: 4px;
            padding: 6px;
            font-size: 11px;
            font-weight: bold;
        """)
        self.banner_stale.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner_stale.setWordWrap(True)
        self.banner_stale.setVisible(False)
        main_layout.addWidget(self.banner_stale)

        # 스크롤 영역
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        self.vbox_content = QVBoxLayout(scroll_content)
        self.vbox_content.setContentsMargins(2, 2, 2, 2)
        self.vbox_content.setSpacing(8)

        # 섹션 1. 결과 불러오기
        self.group_import = self._create_import_section()
        self.vbox_content.addWidget(self.group_import)

        # 섹션 2. 표시
        self.group_display = self._create_display_section()
        self.vbox_content.addWidget(self.group_display)

        # 섹션 3. 프로브와 통계
        self.group_stats = self._create_stats_section()
        self.vbox_content.addWidget(self.group_stats)

        # 섹션 4. 내보내기
        self.group_export = self._create_export_section()
        self.vbox_content.addWidget(self.group_export)

        self.vbox_content.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # 섹션 1. 결과 불러오기
    # -------------------------------------------------------------------------
    def _create_import_section(self) -> QGroupBox:
        group = QGroupBox("1. 결과 불러오기 (Result Import)", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # 상태 안내 라벨
        self.lbl_status = QLabel("결과 없음 (해석을 실행하거나 .frd를 여십시오)", group)
        self.lbl_status.setStyleSheet("color: #abb2bf; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        vbox.addWidget(self.lbl_status)

        # .frd 열기 버튼
        self.btn_open_frd = QPushButton(".frd 파일 열기...", group)
        self.btn_open_frd.setStyleSheet("""
            QPushButton {
                background-color: #2c313a;
                color: #dcdfe4;
                border: 1px solid #4b5263;
                border-radius: 3px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3e4451;
                border-color: #61afef;
            }
        """)
        self.btn_open_frd.clicked.connect(self.open_frd_requested.emit)
        vbox.addWidget(self.btn_open_frd)

        # Increment / Step 선택 콤보
        row_step = QHBoxLayout()
        lbl_step = QLabel("스텝/블록:", group)
        lbl_step.setFixedWidth(65)
        row_step.addWidget(lbl_step)

        self.combo_step = QComboBox(group)
        self.combo_step.addItem("마지막 스텝 (Last Step)", -1)
        self.combo_step.currentIndexChanged.connect(self._on_step_combo_changed)
        row_step.addWidget(self.combo_step)
        vbox.addLayout(row_step)

        return group

    # -------------------------------------------------------------------------
    # 섹션 2. 표시 (Display)
    # -------------------------------------------------------------------------
    def _create_display_section(self) -> QGroupBox:
        group = QGroupBox("2. 표시 (Display)", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # 필드 선택
        row_field = QHBoxLayout()
        lbl_f = QLabel("표시 필드:", group)
        lbl_f.setFixedWidth(65)
        row_field.addWidget(lbl_f)
        self.combo_field = QComboBox(group)
        self.combo_field.addItem("von Mises 응력", "von_mises")
        self.combo_field.addItem("변위 크기 (Displacement)", "u_mag")
        self.combo_field.addItem("최대 주응력 (Max Principal)", "s_max_principal")
        self.combo_field.addItem("최소 주응력 (Min Principal)", "s_min_principal")
        self.combo_field.addItem("등가 변형률 (Equivalent Strain, µε)", "equiv_strain")
        self.combo_field.addItem("요소 탄성계수 (Modulus)", "E_MPa")
        self.combo_field.currentIndexChanged.connect(self._on_field_changed)
        row_field.addWidget(self.combo_field)
        vbox.addLayout(row_field)

        # 컬러맵 선택
        row_cmap = QHBoxLayout()
        lbl_c = QLabel("컬러맵:", group)
        lbl_c.setFixedWidth(65)
        row_cmap.addWidget(lbl_c)
        self.combo_cmap = QComboBox(group)
        self.combo_cmap.addItem("Rainbow (FEA 표준)", "rainbow")
        self.combo_cmap.addItem("Viridis (지각적 선형)", "viridis")
        self.combo_cmap.addItem("CoolWarm (인장/압축)", "coolwarm")
        self.combo_cmap.currentIndexChanged.connect(self._on_cmap_changed)
        row_cmap.addWidget(self.combo_cmap)
        vbox.addLayout(row_cmap)

        # 범위 모드 라디오 버튼
        vbox.addWidget(QLabel("컬러 범위 모드:", group))
        row_range = QHBoxLayout()
        self.radio_global = QRadioButton("전체", group)
        self.radio_per_part = QRadioButton("파트별", group)
        self.radio_manual = QRadioButton("수동", group)
        self.radio_global.setChecked(True)

        self.btn_group_range = QButtonGroup(group)
        self.btn_group_range.addButton(self.radio_global, 0)
        self.btn_group_range.addButton(self.radio_per_part, 1)
        self.btn_group_range.addButton(self.radio_manual, 2)
        self.btn_group_range.idToggled.connect(self._on_range_mode_toggled)

        row_range.addWidget(self.radio_global)
        row_range.addWidget(self.radio_per_part)
        row_range.addWidget(self.radio_manual)
        vbox.addLayout(row_range)

        # 수동 min/max 입력 필드
        self.frame_manual = QFrame(group)
        row_manual = QHBoxLayout(self.frame_manual)
        row_manual.setContentsMargins(0, 0, 0, 0)
        row_manual.setSpacing(4)
        row_manual.addWidget(QLabel("Min:", self.frame_manual))
        self.spin_range_min = QDoubleSpinBox(self.frame_manual)
        self.spin_range_min.setRange(-1e6, 1e6)
        self.spin_range_min.setDecimals(2)
        row_manual.addWidget(self.spin_range_min)

        row_manual.addWidget(QLabel("Max:", self.frame_manual))
        self.spin_range_max = QDoubleSpinBox(self.frame_manual)
        self.spin_range_max.setRange(-1e6, 1e6)
        self.spin_range_max.setDecimals(2)
        row_manual.addWidget(self.spin_range_max)

        self.btn_apply_manual = QPushButton("적용", self.frame_manual)
        self.btn_apply_manual.setFixedWidth(40)
        self.btn_apply_manual.clicked.connect(self._on_manual_range_applied)
        row_manual.addWidget(self.btn_apply_manual)
        self.frame_manual.setVisible(False)
        vbox.addWidget(self.frame_manual)

        # 변형 스케일 슬라이더 + 숫자 입력 (0 ~ 500배)
        vbox.addWidget(QLabel("변형 증폭 스케일 (배수):", group))
        row_scale = QHBoxLayout()
        self.slider_scale = QSlider(Qt.Orientation.Horizontal, group)
        self.slider_scale.setRange(0, 500)
        self.slider_scale.setValue(0)
        self.slider_scale.valueChanged.connect(self._on_scale_slider_changed)
        row_scale.addWidget(self.slider_scale)

        self.spin_scale = QDoubleSpinBox(group)
        self.spin_scale.setRange(0.0, 500.0)
        self.spin_scale.setValue(0.0)
        self.spin_scale.setSingleStep(1.0)
        self.spin_scale.setFixedWidth(65)
        self.spin_scale.valueChanged.connect(self._on_scale_spin_changed)
        row_scale.addWidget(self.spin_scale)
        vbox.addLayout(row_scale)

        # 파트 표시 목록 및 투명도 컨테이너
        vbox.addWidget(QLabel("파트 표시 및 투명도:", group))
        self.vbox_parts = QVBoxLayout()
        self.vbox_parts.setSpacing(4)
        vbox.addLayout(self.vbox_parts)

        # 단면 클립
        self.chk_clip = QCheckBox("단면 클리핑 (Section Cut)", group)
        self.chk_clip.toggled.connect(self._on_clip_toggled)
        vbox.addWidget(self.chk_clip)

        self.frame_clip = QFrame(group)
        vbox_clip = QVBoxLayout(self.frame_clip)
        vbox_clip.setContentsMargins(6, 2, 2, 2)
        vbox_clip.setSpacing(4)

        row_clip_axis = QHBoxLayout()
        row_clip_axis.addWidget(QLabel("절단축:", self.frame_clip))
        self.combo_clip_axis = QComboBox(self.frame_clip)
        self.combo_clip_axis.addItem("Y축 (Coronal 관상면)", "Y")
        self.combo_clip_axis.addItem("X축 (Sagittal 시상면)", "X")
        self.combo_clip_axis.addItem("Z축 (Axial 횡단면)", "Z")
        self.combo_clip_axis.addItem("임플란트 장축 (Implant Axis)", "IMPLANT")
        self.combo_clip_axis.currentIndexChanged.connect(self._emit_clip_changed)
        row_clip_axis.addWidget(self.combo_clip_axis)
        vbox_clip.addLayout(row_clip_axis)

        row_clip_pos = QHBoxLayout()
        row_clip_pos.addWidget(QLabel("위치 (mm):", self.frame_clip))
        self.slider_clip_pos = QSlider(Qt.Orientation.Horizontal, self.frame_clip)
        self.slider_clip_pos.setRange(-100, 100)
        self.slider_clip_pos.setValue(0)
        self.slider_clip_pos.valueChanged.connect(self._emit_clip_changed)
        row_clip_pos.addWidget(self.slider_clip_pos)
        vbox_clip.addLayout(row_clip_pos)
        self.frame_clip.setVisible(False)
        vbox.addWidget(self.frame_clip)

        # 메쉬 선 표시 체크박스
        self.chk_edges = QCheckBox("메쉬 모서리선 표시 (Wireframe)", group)
        self.chk_edges.toggled.connect(self.edges_toggled.emit)
        vbox.addWidget(self.chk_edges)

        # 교합 하중 화살표 표시 체크박스
        self.chk_show_load = QCheckBox("교합 하중 화살표 표시 (Occlusal Load)", group)
        self.chk_show_load.setChecked(False)
        self.chk_show_load.toggled.connect(self.load_arrow_toggled.emit)
        vbox.addWidget(self.chk_show_load)

        return group

    # -------------------------------------------------------------------------
    # 섹션 3. 프로브와 통계
    # -------------------------------------------------------------------------
    def _create_stats_section(self) -> QGroupBox:
        group = QGroupBox("3. 프로브와 통계 (Probe & Stats)", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # 최대 응력 위치 표시 버튼 + 파트 선택
        row_peak = QHBoxLayout()
        self.combo_max_part = QComboBox(group)
        self.combo_max_part.addItem("전체 모델", "ALL")
        self.combo_max_part.addItem("보이지 않기", "NONE")
        self.combo_max_part.currentIndexChanged.connect(self._on_max_part_combo_changed)
        row_peak.addWidget(self.combo_max_part)

        self.btn_mark_max = QPushButton("최대 응력 위치 표시", group)
        self.btn_mark_max.clicked.connect(self._on_mark_max_clicked)
        row_peak.addWidget(self.btn_mark_max)
        vbox.addLayout(row_peak)

        # 클릭 프로브 토글
        self.btn_probe_toggle = QPushButton("3D 클릭 프로브 모드 켜기", group)
        self.btn_probe_toggle.setCheckable(True)
        self.btn_probe_toggle.toggled.connect(self._on_probe_toggled)
        vbox.addWidget(self.btn_probe_toggle)

        # 프로브 값 표시 라벨
        self.lbl_probe_info = QLabel("절점을 클릭하거나 최대 위치를 표시하십시오.", group)
        self.lbl_probe_info.setStyleSheet("""
            background-color: #21252b;
            color: #abb2bf;
            border: 1px solid #3c4048;
            border-radius: 3px;
            padding: 4px;
            font-size: 11px;
        """)
        self.lbl_probe_info.setWordWrap(True)
        vbox.addWidget(self.lbl_probe_info)

        # 주변 뼈 통계 반경 조절
        row_radius = QHBoxLayout()
        row_radius.addWidget(QLabel("임플란트 주변 뼈 분석 반경:", group))
        self.spin_peri_radius = QDoubleSpinBox(group)
        self.spin_peri_radius.setRange(0.1, 10.0)
        self.spin_peri_radius.setValue(1.0)
        self.spin_peri_radius.setSingleStep(0.2)
        self.spin_peri_radius.setSuffix(" mm")
        self.spin_peri_radius.valueChanged.connect(self.peri_radius_changed.emit)
        row_radius.addWidget(self.spin_peri_radius)
        vbox.addLayout(row_radius)

        # 통계 표 (QTableWidget)
        self.table_stats = QTableWidget(group)
        self.table_stats.setColumnCount(6)
        self.table_stats.setHorizontalHeaderLabels([
            "최대 변위\n(mm)", "최대 vM\n(MPa)", "99%\n(MPa)",
            "평균\n(MPa)", "최대 인장\n(MPa)", "최대 압축\n(MPa)"
        ])
        self.table_stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_stats.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_stats.setMinimumHeight(150)
        self.table_stats.setStyleSheet("""
            QTableWidget {
                background-color: #21252b;
                color: #dcdfe4;
                gridline-color: #3e4451;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #282c34;
                color: #abb2bf;
                font-size: 10px;
                padding: 2px;
                border: 1px solid #3e4451;
            }
        """)
        vbox.addWidget(self.table_stats)

        # 힘 평형 점검 결과 라벨
        self.lbl_equilibrium = QLabel("힘 평형: --", group)
        self.lbl_equilibrium.setStyleSheet("color: #abb2bf; font-size: 11px;")
        self.lbl_equilibrium.setWordWrap(True)
        vbox.addWidget(self.lbl_equilibrium)

        return group

    # -------------------------------------------------------------------------
    # 섹션 4. 내보내기 (Export)
    # -------------------------------------------------------------------------
    def _create_export_section(self) -> QGroupBox:
        group = QGroupBox("4. 내보내기 및 저장 (Export)", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # 1. CSV 저장 (절점 데이터)
        self.btn_export_csv = QPushButton("절점 결과 CSV 저장...", group)
        self.btn_export_csv.clicked.connect(self.export_csv_requested.emit)
        vbox.addWidget(self.btn_export_csv)

        # 2. 통계 저장 (CSV / JSON)
        self.btn_export_stats = QPushButton("통계 표 저장 (CSV/JSON)...", group)
        self.btn_export_stats.clicked.connect(self.export_stats_requested.emit)
        vbox.addWidget(self.btn_export_stats)

        # 3. 3D 화면 이미지 저장 (PNG, 해상도 선택)
        row_img = QHBoxLayout()
        self.btn_save_screenshot = QPushButton("3D 화면 이미지 저장...", group)
        self.btn_save_screenshot.clicked.connect(self._on_screenshot_clicked)
        row_img.addWidget(self.btn_save_screenshot, stretch=2)

        self.combo_screen_scale = QComboBox(group)
        self.combo_screen_scale.addItem("1x (현재)", 1)
        self.combo_screen_scale.addItem("2x (고해상도)", 2)
        row_img.addWidget(self.combo_screen_scale, stretch=1)
        vbox.addLayout(row_img)

        # 구분선
        line = QFrame(group)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        vbox.addWidget(line)

        # 4. 결과 폴더 저장 (Bundle)
        self.btn_save_bundle = QPushButton("결과 폴더 저장 (Bundle)...", group)
        self.btn_save_bundle.clicked.connect(self.save_bundle_requested.emit)
        vbox.addWidget(self.btn_save_bundle)

        # 5. 결과 폴더 불러오기 (Bundle)
        self.btn_load_bundle = QPushButton("결과 폴더 불러오기 (Bundle)...", group)
        self.btn_load_bundle.clicked.connect(self.load_bundle_requested.emit)
        vbox.addWidget(self.btn_load_bundle)

        return group

    def _on_screenshot_clicked(self) -> None:
        scale = self.combo_screen_scale.currentData() or 1
        self.save_screenshot_requested.emit(int(scale))

    # -------------------------------------------------------------------------
    # 제어 및 상태 동기화 API
    # -------------------------------------------------------------------------
    def set_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable result display and statistics widgets."""
        self.group_display.setEnabled(enabled)
        self.group_stats.setEnabled(enabled)
        self.btn_export_csv.setEnabled(enabled)
        self.btn_export_stats.setEnabled(enabled)
        self.btn_save_screenshot.setEnabled(enabled)
        self.combo_screen_scale.setEnabled(enabled)
        self.btn_save_bundle.setEnabled(enabled)
        self.btn_load_bundle.setEnabled(True)
        if not enabled:
            self.lbl_probe_info.setText("절점을 클릭하거나 최대 위치를 표시하십시오.")
            self.table_stats.setRowCount(0)

            self.lbl_equilibrium.setText("힘 평형: --")

    def set_loading_state(self, is_loading: bool, message: str = "") -> None:
        """Set UI state during background result loading."""
        if is_loading:
            self.lbl_status.setText(message or "결과 읽는 중...")
            self.lbl_status.setStyleSheet("color: #61afef; font-weight: bold; font-size: 11px;")
            self.btn_open_frd.setEnabled(False)
            self.set_controls_enabled(False)
        else:
            self.btn_open_frd.setEnabled(True)

    def set_stale_state(self, message: str) -> None:
        """Show stale notification and disable result view controls."""
        self.banner_stale.setText(f"⚠️ {message}")
        self.banner_stale.setVisible(True)
        self.lbl_status.setText("결과 무효화됨 (모델 재해석 필요)")
        self.lbl_status.setStyleSheet("color: #e5c07b; font-size: 11px;")
        self.set_controls_enabled(False)
        self.btn_open_frd.setEnabled(True)

    def show_error(self, err_msg: str) -> None:
        """Show error message in status label and dialog without crashing."""
        self.lbl_status.setText(f"오류: {err_msg}")
        self.lbl_status.setStyleSheet("color: #e06c75; font-size: 11px; font-weight: bold;")
        self.btn_open_frd.setEnabled(True)
        QMessageBox.critical(self, "결과 로드 오류", f"결과 파일을 불러오는 중 오류가 발생했습니다:\n\n{err_msg}")

    def populate_results(self, rs: ResultSet, stats: dict, model: MeshModel) -> None:
        """
        Populate part list, statistics table, and equilibrium info after load.
        """
        self._is_updating = True
        try:
            self.set_loading_state(False)
            self.btn_open_frd.setEnabled(True)
            self.banner_stale.setVisible(False)
            source_name = getattr(rs, "source_frd", "") or "해석 결과"
            num_nodes = stats.get("num_nodes", 0)
            self.lbl_status.setText(f"로드 완료: {source_name}\n(절점: {num_nodes:,}개, 시간: {rs.time:.3f}s)")
            self.lbl_status.setStyleSheet("color: #98c379; font-size: 11px;")

            self.set_controls_enabled(True)

            # 스텝/블록 콤보박스 갱신 (다중 스텝/인크리먼트 지원)
            step_items = stats.get("step_items")
            if step_items:
                self.combo_step.blockSignals(True)
                self.combo_step.clear()
                self.combo_step.addItem("마지막 스텝 (Last Step)", -1)
                for item_label, item_idx in step_items:
                    self.combo_step.addItem(item_label, item_idx)
                # 현재 선택된 스텝 반영
                cur_step = stats.get("step", -1)
                for i in range(self.combo_step.count()):
                    if self.combo_step.itemData(i) == cur_step:
                        self.combo_step.setCurrentIndex(i)
                        break
                self.combo_step.blockSignals(False)

            # 파트 위젯 목록 동적 생성
            self._rebuild_part_list(model)

            # 최대 응력 파트 콤보 갱신
            self.combo_max_part.blockSignals(True)
            self.combo_max_part.clear()
            self.combo_max_part.addItem("전체 모델", "ALL")
            self.combo_max_part.addItem("보이지 않기", "NONE")
            for part in model.parts:
                self.combo_max_part.addItem(part.name, part.name)
            self.combo_max_part.blockSignals(False)

            # 통계 표 갱신
            self._update_stats_table(stats, model)

            # 힘 평형 라벨 갱신
            self._update_equilibrium_label(stats.get("reaction_balance"))

        finally:
            self._is_updating = False

    def _rebuild_part_list(self, model: MeshModel) -> None:
        """Rebuild individual part checkboxes and opacity sliders."""
        # 기존 위젯 정리
        while self.vbox_parts.count() > 0:
            item = self.vbox_parts.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._part_widgets.clear()

        for part in model.parts:
            p_name = part.name
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(6)

            chk = QCheckBox(p_name, row_w)
            chk.setChecked(True)
            chk.toggled.connect(lambda vis, name=p_name: self.part_visibility_changed.emit(name, vis))
            row.addWidget(chk)

            slider = QSlider(Qt.Orientation.Horizontal, row_w)
            slider.setRange(0, 100)
            slider.setValue(100)
            slider.setFixedWidth(70)
            slider.valueChanged.connect(
                lambda val, name=p_name: self.part_opacity_changed.emit(name, val / 100.0)
            )
            row.addWidget(slider)

            self.vbox_parts.addWidget(row_w)
            self._part_widgets[p_name] = {"check": chk, "slider": slider}

    def _update_stats_table(self, stats: dict, model: MeshModel) -> None:
        """Fill QTableWidget with part statistics and peri-implant bone stats."""
        part_stats = stats.get("parts", {})
        peri_stats = stats.get("peri_implant", None)
        radius = stats.get("radius_mm", self.spin_peri_radius.value())

        rows: List[tuple] = []
        for part in model.parts:
            p_stat = part_stats.get(part.name, {})
            rows.append((part.name, p_stat))

        if peri_stats is not None:
            label = f"임플란트 주변 뼈 (반경 {radius:.1f} mm)"
            rows.append((label, peri_stats))

        self.table_stats.setRowCount(len(rows))

        def _fmt(val: Optional[float], decimals: int = 2) -> str:
            if val is None or np.isnan(val):
                return "--"
            return f"{val:.{decimals}f}"

        for r_idx, (name, st) in enumerate(rows):
            # 행 라벨
            self.table_stats.setVerticalHeaderItem(r_idx, QTableWidgetItem(name))

            # 열 값
            u_max = st.get("max_u_mag")
            vm_max = st.get("max_von_mises")
            vm_p99 = st.get("p99_von_mises")
            vm_mean = st.get("mean_von_mises")
            s_max = st.get("max_s_max_principal")
            s_min = st.get("min_s_min_principal")

            vals = [
                _fmt(u_max, 4),
                _fmt(vm_max, 2),
                _fmt(vm_p99, 2),
                _fmt(vm_mean, 2),
                _fmt(s_max, 2),
                _fmt(s_min, 2),
            ]
            for c_idx, val_str in enumerate(vals):
                item = QTableWidgetItem(val_str)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table_stats.setItem(r_idx, c_idx, item)

    def _update_equilibrium_label(self, eq: Optional[dict]) -> None:
        """Update equilibrium text and color based on relative error threshold."""
        if not eq:
            self.lbl_equilibrium.setText("힘 평형: 산출 불가 (반력 또는 구속 조건 부재)")
            self.lbl_equilibrium.setStyleSheet("color: #abb2bf; font-size: 11px;")
            return

        app_mag = eq.get("applied_mag", 0.0)
        rxn_mag = eq.get("reaction_mag", 0.0)
        rel_err = eq.get("relative_error", 0.0)
        pct_err = rel_err * 100.0

        text = f"적용 하중: {app_mag:.2f} N  /  반력 합: {rxn_mag:.2f} N  /  오차: {pct_err:.2f}%"
        if pct_err > 1.0:
            self.lbl_equilibrium.setText(
                f"{text}\n⚠️ 오차 1% 초과: 구속/하중 정의 또는 단위계를 확인하십시오."
            )
            self.lbl_equilibrium.setStyleSheet("color: #e06c75; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_equilibrium.setText(f"{text} (정적 평형 만족)")
            self.lbl_equilibrium.setStyleSheet("color: #98c379; font-size: 11px;")

    def display_probe_info(self, data: dict) -> None:
        """Show probed nodal values in info label."""
        p_name = data.get("part", "")
        n_idx = data.get("node", -1)
        xyz = data.get("xyz", [0, 0, 0])
        u_mag = data.get("u_mag", 0.0)
        vm = data.get("von_mises", 0.0)
        s_max = data.get("s_max_principal", 0.0)
        s_min = data.get("s_min_principal", 0.0)

        txt = (
            f"[{p_name}] 절점 #{n_idx}\n"
            f"위치: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) mm\n"
            f"변위: {u_mag:.4f} mm | von Mises: {vm:.2f} MPa\n"
            f"주응력: Max {s_max:.2f} MPa, Min {s_min:.2f} MPa"
        )
        self.lbl_probe_info.setText(txt)

    def display_max_marked_info(self, data: dict) -> None:
        """Show peak stress localization info."""
        p_name = data.get("part_name", "")
        n_idx = data.get("node_idx", -1)
        coord = data.get("coord", [0, 0, 0])
        val = data.get("value", 0.0)
        f_name = data.get("field", "von_mises")

        txt = (
            f"[최대 {f_name}] 파트: {p_name} (절점 #{n_idx})\n"
            f"위치: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}) mm\n"
            f"최대값: {val:.2f}"
        )
        self.lbl_probe_info.setText(txt)

    # -------------------------------------------------------------------------
    # 내부 이벤트 바인딩
    # -------------------------------------------------------------------------
    def _on_step_combo_changed(self) -> None:
        if self._is_updating:
            return
        step_val = self.combo_step.currentData()
        if step_val is not None:
            self.step_changed.emit(step_val)

    def _on_field_changed(self) -> None:
        if self._is_updating:
            return
        field_id = self.combo_field.currentData()
        self.field_changed.emit(field_id)

    def _on_cmap_changed(self) -> None:
        if self._is_updating:
            return
        cmap_id = self.combo_cmap.currentData()
        self.colormap_changed.emit(cmap_id)

    def _on_range_mode_toggled(self, button_id: int, checked: bool) -> None:
        if not checked or self._is_updating:
            return
        mode = "global" if button_id == 0 else ("per_part" if button_id == 1 else "manual")
        self.frame_manual.setVisible(mode == "manual")
        manual_range = (self.spin_range_min.value(), self.spin_range_max.value()) if mode == "manual" else None
        self.range_mode_changed.emit(mode, manual_range)

    def _on_manual_range_applied(self) -> None:
        cmin = self.spin_range_min.value()
        cmax = self.spin_range_max.value()
        if cmin >= cmax:
            cmax = cmin + 1.0
            self.spin_range_max.setValue(cmax)
        self.range_mode_changed.emit("manual", (cmin, cmax))

    def _on_scale_slider_changed(self, val: int) -> None:
        if self._is_updating:
            return
        self._is_updating = True
        self.spin_scale.setValue(float(val))
        self._is_updating = False
        self.deform_scale_changed.emit(float(val))

    def _on_scale_spin_changed(self, val: float) -> None:
        if self._is_updating:
            return
        self._is_updating = True
        self.slider_scale.setValue(int(round(val)))
        self._is_updating = False
        self.deform_scale_changed.emit(float(val))

    def _on_clip_toggled(self, checked: bool) -> None:
        self.frame_clip.setVisible(checked)
        self._emit_clip_changed()

    def _emit_clip_changed(self) -> None:
        if self._is_updating:
            return
        enabled = self.chk_clip.isChecked()
        axis_str = self.combo_clip_axis.currentData() or "Y"
        offset_mm = float(self.slider_clip_pos.value())
        self.clip_changed.emit(enabled, axis_str, offset_mm)

    def _on_max_part_combo_changed(self, index: int) -> None:
        if self._is_updating:
            return
        part = self.combo_max_part.currentData()
        if part == "NONE":
            field = self.combo_field.currentData()
            self.mark_max_requested.emit(field, "NONE")

    def _on_mark_max_clicked(self) -> None:
        field = self.combo_field.currentData()
        part = self.combo_max_part.currentData()
        self.mark_max_requested.emit(field, part)

    def _on_probe_toggled(self, checked: bool) -> None:
        self.btn_probe_toggle.setText("3D 클릭 프로브 모드 끄기" if checked else "3D 클릭 프로브 모드 켜기")
        if not checked:
            self.lbl_probe_info.setText("절점을 클릭하거나 최대 위치를 표시하십시오.")
        self.probe_toggled.emit(checked)
