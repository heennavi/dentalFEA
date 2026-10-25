"""
control_panel.py
좌측 워크플로 컨트롤 패널 (영상 로드, W/L, 좌표 제어, 파이프라인 탭).
"""

from typing import Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QSlider, QDoubleSpinBox,
    QGroupBox, QTabWidget, QFileDialog, QMessageBox,
    QScrollArea, QFrame
)
from PySide6.QtCore import Qt

from dental_fea_app.config import WL_PRESETS, DEFAULT_PRESET_NAME
from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.imaging.dicom_reader import DicomReader
from dental_fea_app.ui.segmentation.segmentation_panel import SegmentationPanel
from dental_fea_app.ui.placement.implant_panel import ImplantPanel
from dental_fea_app.ui.geometry.geometry_panel import GeometryPanel
from dental_fea_app.ui.fea_panel import FeaPanel
from ui.result_panel import ResultPanel
from ui.result_controller import ResultController
import math


class ControlPanel(QWidget):
    """
    애플리케이션의 좌측 제어 패널.
    DICOM I/O, 영상 Window/Level 조절, 3D 커서 네비게이션 및 단계별 워크플로 UI를 담당합니다.
    """

    def __init__(self, state_manager: StateManager, on_open_3d_viewer: Callable[[], None], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.on_open_3d_viewer = on_open_3d_viewer
        self.setMinimumWidth(276)  # 기존 240px에서 15% 확대
        self.setMaximumWidth(600)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 2, 0, 2)
        main_layout.setSpacing(4)

        # 스크롤 영역으로 감싸기
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 1. 단계별 워크플로 탭
        layout.addWidget(self._create_workflow_tabs())

        layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # StateManager 신호 바인딩
        self.state_mgr.volume_loaded.connect(self._on_volume_loaded)

    def _create_io_group(self) -> QGroupBox:
        group = QGroupBox("영상 데이터 불러오기", self)
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        btn_folder = QPushButton("📁 DICOM 폴더 열기 (Series)", group)
        btn_folder.clicked.connect(self._load_dicom_folder)
        vbox.addWidget(btn_folder)

        btn_file = QPushButton("📄 단일 파일 열기 (.dcm, .mha, .nii)", group)
        btn_file.clicked.connect(self._load_single_file)
        vbox.addWidget(btn_file)

        btn_sample = QPushButton("🎯 샘플 치과 CT 로드 (하악골)", group)
        btn_sample.setStyleSheet("background-color: #1e3a5f; border-color: #3878b2; font-weight: bold;")
        btn_sample.clicked.connect(self._load_sample_data)
        vbox.addWidget(btn_sample)

        return group

    def _create_meta_group(self) -> QGroupBox:
        self.meta_group = QGroupBox("영상 정보 (Metadata)", self)
        vbox = QVBoxLayout(self.meta_group)
        vbox.setSpacing(4)

        self.lbl_meta_dim = QLabel("크기 (Voxels): --", self.meta_group)
        self.lbl_meta_spacing = QLabel("복셀 간격 (mm): --", self.meta_group)
        self.lbl_meta_range = QLabel("HU 범위: --", self.meta_group)

        vbox.addWidget(self.lbl_meta_dim)
        vbox.addWidget(self.lbl_meta_spacing)
        vbox.addWidget(self.lbl_meta_range)

        return self.meta_group

    def _create_wl_group(self) -> QGroupBox:
        group = QGroupBox("Window / Level (명암비 조절)", self)
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        # 프리셋 콤보박스
        vbox.addWidget(QLabel("프리셋 선택:", group))
        self.combo_wl_preset = QComboBox(group)
        for name in WL_PRESETS.keys():
            self.combo_wl_preset.addItem(name)
        self.combo_wl_preset.setCurrentText(DEFAULT_PRESET_NAME)
        self.combo_wl_preset.currentTextChanged.connect(self._on_preset_selected)
        vbox.addWidget(self.combo_wl_preset)

        # Window 폭
        vbox.addWidget(QLabel("Window (폭):", group))
        h_w = QHBoxLayout()
        self.slider_w = QSlider(Qt.Orientation.Horizontal, group)
        self.slider_w.setRange(10, 4000)
        self.slider_w.setValue(int(self.state_mgr.window))
        self.slider_w.valueChanged.connect(self._on_slider_w_changed)
        self.spin_w = QDoubleSpinBox(group)
        self.spin_w.setRange(1.0, 8000.0)
        self.spin_w.setValue(self.state_mgr.window)
        self.spin_w.valueChanged.connect(self._on_spin_w_changed)
        h_w.addWidget(self.slider_w)
        h_w.addWidget(self.spin_w)
        vbox.addLayout(h_w)

        # Level 중심
        vbox.addWidget(QLabel("Level (중심 HU):", group))
        h_l = QHBoxLayout()
        self.slider_l = QSlider(Qt.Orientation.Horizontal, group)
        self.slider_l.setRange(-1024, 3000)
        self.slider_l.setValue(int(self.state_mgr.level))
        self.slider_l.valueChanged.connect(self._on_slider_l_changed)
        self.spin_l = QDoubleSpinBox(group)
        self.spin_l.setRange(-2048.0, 5000.0)
        self.spin_l.setValue(self.state_mgr.level)
        self.spin_l.valueChanged.connect(self._on_spin_l_changed)
        h_l.addWidget(self.slider_l)
        h_l.addWidget(self.spin_l)
        vbox.addLayout(h_l)

        return group

    def _create_nav_group(self) -> QGroupBox:
        group = QGroupBox("3D 커서 좌표 네비게이션 (mm)", self)
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        # X 좌표 (Sagittal)
        h_x = QHBoxLayout()
        h_x.addWidget(QLabel("X:", group))
        self.spin_x = QDoubleSpinBox(group)
        self.spin_x.setRange(-9999.0, 9999.0)
        self.spin_x.setSingleStep(1.0)
        self.spin_x.valueChanged.connect(lambda v: self.state_mgr.set_axis_coordinate(0, v))
        h_x.addWidget(self.spin_x)
        vbox.addLayout(h_x)

        # Y 좌표 (Coronal)
        h_y = QHBoxLayout()
        h_y.addWidget(QLabel("Y:", group))
        self.spin_y = QDoubleSpinBox(group)
        self.spin_y.setRange(-9999.0, 9999.0)
        self.spin_y.setSingleStep(1.0)
        self.spin_y.valueChanged.connect(lambda v: self.state_mgr.set_axis_coordinate(1, v))
        h_y.addWidget(self.spin_y)
        vbox.addLayout(h_y)

        # Z 좌표 (Axial)
        h_z = QHBoxLayout()
        h_z.addWidget(QLabel("Z:", group))
        self.spin_z = QDoubleSpinBox(group)
        self.spin_z.setRange(-9999.0, 9999.0)
        self.spin_z.setSingleStep(1.0)
        self.spin_z.valueChanged.connect(lambda v: self.state_mgr.set_axis_coordinate(2, v))
        h_z.addWidget(self.spin_z)
        vbox.addLayout(h_z)

        return group

    def _create_workflow_tabs(self) -> QTabWidget:
        self.workflow_tabs = QTabWidget(self)
        tabs = self.workflow_tabs
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c3c46;
                padding: 0px;
                margin: 0px;
            }
            QTabBar::tab {
                background: #21252b;
                color: #abb2bf;
                padding: 6px 6px;
                border: 1px solid #3c3c46;
                border-bottom: none;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #282c34;
                color: #61afef;
                font-weight: bold;
            }
        """)
        tabs.tabBar().setExpanding(True)
        tabs.tabBar().setDocumentMode(True)

        # 탭 1: Phase 1 영상 정보
        t1 = QWidget()
        t1_layout = QVBoxLayout(t1)
        t1_layout.addWidget(QLabel("Phase 1: 2D MPR 3분할 뷰어\n및 실시간 십자선 동기화 활성 상태", t1))
        t1_layout.addStretch()
        tabs.addTab(t1, "1. 영상")

        # 탭 2: Phase 2 임플란트 배치 패널
        self.implant_panel = ImplantPanel(self.state_mgr, self)
        tabs.addTab(self.implant_panel, "2. 임플란트")

        # 탭 3: Phase 2 분할 (피질골/수질골 분할 패널)
        self.segmentation_panel = SegmentationPanel(self.state_mgr, self)
        tabs.addTab(self.segmentation_panel, "3. 분할")

        # 탭 4: Phase 2 기하 처리 패널 (Boolean 차집합 및 메쉬 준비)
        self.geometry_panel = GeometryPanel(self.state_mgr, self)
        tabs.addTab(self.geometry_panel, "4. 기하 처리")

        # 탭 5: Phase 3 FEA 유한요소해석 패널
        self.fea_panel = FeaPanel(self.state_mgr, self)
        tabs.addTab(self.fea_panel, "5. 해석")

        # 탭 6: Phase 4 FEA 해석 결과 패널
        self.result_panel = ResultPanel(self.state_mgr, self)
        self.result_controller = ResultController(self.state_mgr, panel=self.result_panel, parent=self)
        self.result_controller.open_3d_viewer_callback = getattr(self, "on_open_3d_viewer", None)
        self.result_panel.controller = self.result_controller
        tabs.addTab(self.result_panel, "6. 결과")

        # 결과 로드/무효화 시 탭 활성화 및 전환 연결
        self.result_controller.results_loaded.connect(self._on_results_loaded)
        self.result_controller.results_cleared.connect(self._on_results_cleared)

        tabs.currentChanged.connect(self._on_workflow_tab_changed)

        # CT 이미지가 열리기 전에는 분할(1), 임플란트(2), 기하 처리(3), 해석(4), 결과(5) 비활성화
        tabs.setTabEnabled(1, False)
        tabs.setTabEnabled(2, False)
        tabs.setTabEnabled(3, False)
        tabs.setTabEnabled(4, False)
        tabs.setTabEnabled(5, False)

        return tabs

    def _on_workflow_tab_changed(self, index: int):
        """워크플로 탭 전환 시 3D 볼륨 창의 결과 뷰 모드 및 임플란트 기즈모 복구 연동"""
        is_result_tab = (index == 5)
        mw = self.window()
        if mw is not None and hasattr(mw, "volume_window") and mw.volume_window is not None:
            mw.volume_window.set_result_view_mode(is_result_tab)

    # --- 이벤트 핸들러 ---
    def _load_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "DICOM 시리즈 디렉터리 선택")
        if not folder:
            return

        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QCursor
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()

        try:
            vtk_img, meta = DicomReader.load_dicom_series(folder)
            self.state_mgr.set_volume_data(vtk_img, meta)
        except Exception as e:
            QMessageBox.critical(self, "DICOM 로드 오류", f"DICOM 폴더를 읽는 중 오류가 발생했습니다:\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def _load_single_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "의료 영상 파일 선택", "",
            "의료 영상 (*.dcm *.ima *.mha *.mhd *.nii *.nii.gz *.nrrd);;모든 파일 (*.*)"
        )
        if not file_path:
            return

        try:
            vtk_img, meta = DicomReader.load_single_file(file_path)
            self.state_mgr.set_volume_data(vtk_img, meta)
        except Exception as e:
            QMessageBox.critical(self, "파일 로드 오류", f"파일을 읽는 중 오류가 발생했습니다:\n{str(e)}")

    def _load_sample_data(self):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sample_path = os.path.join(base_dir, "tests", "sample_data", "synthetic_mandible_ct.mha")

        if not os.path.exists(sample_path):
            from tests.create_dummy_data import generate_synthetic_dental_ct
            generate_synthetic_dental_ct(sample_path)

        try:
            vtk_img, meta = DicomReader.load_single_file(sample_path)
            self.state_mgr.set_volume_data(vtk_img, meta)
        except Exception as e:
            QMessageBox.critical(self, "샘플 로드 오류", f"샘플 데이터를 로드하는 중 오류가 발생했습니다:\n{str(e)}")

    def _on_results_loaded(self, rs, stats):
        if hasattr(self, "workflow_tabs"):
            self.workflow_tabs.setTabEnabled(5, True)
            self.workflow_tabs.setCurrentIndex(5)

    def _on_results_cleared(self):
        if hasattr(self, "workflow_tabs"):
            self.workflow_tabs.setTabEnabled(5, False)

    def _on_volume_loaded(self, vtk_image):
        has_vol = (vtk_image is not None)
        if hasattr(self, "workflow_tabs"):
            self.workflow_tabs.setTabEnabled(1, has_vol)
            self.workflow_tabs.setTabEnabled(2, has_vol)
            self.workflow_tabs.setTabEnabled(3, has_vol)
            self.workflow_tabs.setTabEnabled(4, has_vol)
            self.workflow_tabs.setTabEnabled(5, False)

        if hasattr(self, "lbl_meta_dim"):
            dims = self.state_mgr.dimensions
            spc = self.state_mgr.spacing
            s_min, s_max = self.state_mgr.scalar_range
            self.lbl_meta_dim.setText(f"크기 (Voxels): {dims[0]} × {dims[1]} × {dims[2]}")
            self.lbl_meta_spacing.setText(f"복셀 간격 (mm): {spc[0]:.2f}, {spc[1]:.2f}, {spc[2]:.2f}")
            self.lbl_meta_range.setText(f"HU 범위: [{s_min:.1f}, {s_max:.1f}]")

        bnds = self.state_mgr.bounds
        # 스핀박스가 존재하는 경우에만 범위 갱신
        if hasattr(self, "spin_x"):
            self.spin_x.blockSignals(True)
            self.spin_y.blockSignals(True)
            self.spin_z.blockSignals(True)
            self.spin_x.setRange(bnds[0], bnds[1])
            self.spin_y.setRange(bnds[2], bnds[3])
            self.spin_z.setRange(bnds[4], bnds[5])
            self.spin_x.setValue(self.state_mgr.current_coords[0])
            self.spin_y.setValue(self.state_mgr.current_coords[1])
            self.spin_z.setValue(self.state_mgr.current_coords[2])
            self.spin_x.blockSignals(False)
            self.spin_y.blockSignals(False)
            self.spin_z.blockSignals(False)

    def _on_coords_changed(self, x: float, y: float, z: float):
        if hasattr(self, "spin_x"):
            self.spin_x.blockSignals(True)
            self.spin_y.blockSignals(True)
            self.spin_z.blockSignals(True)
            self.spin_x.setValue(x)
            self.spin_y.setValue(y)
            self.spin_z.setValue(z)
            self.spin_x.blockSignals(False)
            self.spin_y.blockSignals(False)
            self.spin_z.blockSignals(False)

    def _on_preset_selected(self, preset_name: str):
        if preset_name in WL_PRESETS:
            p = WL_PRESETS[preset_name]
            self.state_mgr.set_window_level(p.window, p.level)

    def _on_slider_w_changed(self, val: int):
        self.state_mgr.set_window_level(float(val), self.state_mgr.level)

    def _on_spin_w_changed(self, val: float):
        self.state_mgr.set_window_level(val, self.state_mgr.level)

    def _on_slider_l_changed(self, val: int):
        self.state_mgr.set_window_level(self.state_mgr.window, float(val))

    def _on_spin_l_changed(self, val: float):
        self.state_mgr.set_window_level(self.state_mgr.window, val)

    def _on_wl_changed(self, window: float, level: float):
        self.slider_w.blockSignals(True)
        self.spin_w.blockSignals(True)
        self.slider_l.blockSignals(True)
        self.spin_l.blockSignals(True)

        self.slider_w.setValue(int(window))
        self.spin_w.setValue(window)
        self.slider_l.setValue(int(level))
        self.spin_l.setValue(level)

        self.slider_w.blockSignals(False)
        self.spin_w.blockSignals(False)
        self.slider_l.blockSignals(False)
        self.spin_l.blockSignals(False)
