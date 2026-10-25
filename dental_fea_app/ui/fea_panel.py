"""
fea_panel.py
Left workflow control panel tab widget for Phase 3 FEA operations:
1. Volume meshing (Gmsh C3D4/C3D10)
2. Material assignment (Bone HU mapping & Titanium)
3. Boundary conditions & occlusal loads
4. CalculiX solver execution & real-time monitoring
"""

import os
import math
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QProgressBar, QGroupBox, QSlider, QLineEdit,
    QPlainTextEdit, QFileDialog, QMessageBox, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, Signal

from dental_fea_app.core.state_manager import StateManager
from fea.models import AnalysisConfig
from fea.boundary_conditions import default_load_presets
from dental_fea_app.ui.fea_controller import FeaController
from dental_fea_app.ui.icons import IconFactory


class FeaPanel(QWidget):
    """FEA 파이프라인 제어 패널 위젯"""

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.controller = FeaController(state_manager, parent=self)
        self._fixed_z: Optional[float] = None

        self._init_ui()
        self._bind_signals()
        self._update_ui_state()

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(6)

        # 0. 상류 데이터 변경 경고 배너 (평소 숨김)
        self.banner_stale = QLabel("⚠️ 기하 형상이 변경되었습니다.\n메쉬를 다시 생성해야 합니다.", self)
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
        self.banner_stale.setVisible(False)
        layout.addWidget(self.banner_stale)

        # 1. 메쉬 생성 섹션
        group_mesh = QGroupBox("1. 메쉬 생성 (Meshing)", self)
        vbox_mesh = QVBoxLayout(group_mesh)
        vbox_mesh.setContentsMargins(6, 6, 6, 6)
        vbox_mesh.setSpacing(4)

        # 요소 차수 선택
        row_elem = QHBoxLayout()
        lbl_elem = QLabel("요소 차수:", group_mesh)
        lbl_elem.setFixedWidth(52)  # 글자 2개분만큼 좌측으로 당겨지도록 폭 고정
        row_elem.addWidget(lbl_elem)
        self.combo_elem_type = QComboBox(group_mesh)
        self.combo_elem_type.addItem("C3D10 (2차 10절점 고정밀)", "C3D10")
        self.combo_elem_type.addItem("C3D4 (1차 4절점 미리보기)", "C3D4")
        row_elem.addWidget(self.combo_elem_type)
        vbox_mesh.addLayout(row_elem)

        # 원거리 메쉬 크기
        row_far = QHBoxLayout()
        lbl_far = QLabel("먼 영역 크기 (mm):", group_mesh)
        lbl_far.setFixedWidth(122)
        row_far.addWidget(lbl_far)
        self.spin_size_far = QDoubleSpinBox(group_mesh)
        self.spin_size_far.setRange(0.5, 10.0)
        self.spin_size_far.setValue(2.0)
        self.spin_size_far.setSingleStep(0.2)
        self.spin_size_far.setFixedWidth(60)  # 기존 폭의 1/2로 축소
        row_far.addWidget(self.spin_size_far)
        row_far.addStretch()
        vbox_mesh.addLayout(row_far)

        # 임플란트 주변 메쉬 크기
        row_near = QHBoxLayout()
        lbl_near = QLabel("임플란트 주변 (mm):", group_mesh)
        lbl_near.setFixedWidth(130)
        row_near.addWidget(lbl_near)
        self.spin_size_near = QDoubleSpinBox(group_mesh)
        self.spin_size_near.setRange(0.1, 5.0)
        self.spin_size_near.setValue(0.40)
        self.spin_size_near.setSingleStep(0.05)
        self.spin_size_near.setFixedWidth(60)
        row_near.addWidget(self.spin_size_near)
        row_near.addStretch()
        vbox_mesh.addLayout(row_near)

        # 지대주(Abutment) 메쉬 크기
        row_abut = QHBoxLayout()
        lbl_abut = QLabel("지대주(Abutment) (mm):", group_mesh)
        lbl_abut.setFixedWidth(130)
        row_abut.addWidget(lbl_abut)
        self.spin_size_abut = QDoubleSpinBox(group_mesh)
        self.spin_size_abut.setRange(0.1, 5.0)
        self.spin_size_abut.setValue(0.40)
        self.spin_size_abut.setSingleStep(0.05)
        self.spin_size_abut.setFixedWidth(60)
        row_abut.addWidget(self.spin_size_abut)
        row_abut.addStretch()
        vbox_mesh.addLayout(row_abut)

        # 스크류(Screw) 메쉬 크기
        row_screw = QHBoxLayout()
        lbl_screw = QLabel("스크류(Screw) (mm):", group_mesh)
        lbl_screw.setFixedWidth(130)
        row_screw.addWidget(lbl_screw)
        self.spin_size_screw = QDoubleSpinBox(group_mesh)
        self.spin_size_screw.setRange(0.1, 5.0)
        self.spin_size_screw.setValue(0.35)
        self.spin_size_screw.setSingleStep(0.05)
        self.spin_size_screw.setFixedWidth(60)
        row_screw.addWidget(self.spin_size_screw)
        row_screw.addStretch()
        vbox_mesh.addLayout(row_screw)

        # 보철물(Crown) 메쉬 크기
        row_crown = QHBoxLayout()
        lbl_crown = QLabel("보철물(Crown) (mm):", group_mesh)
        lbl_crown.setFixedWidth(130)
        row_crown.addWidget(lbl_crown)
        self.spin_size_crown = QDoubleSpinBox(group_mesh)
        self.spin_size_crown.setRange(0.1, 5.0)
        self.spin_size_crown.setValue(0.40)
        self.spin_size_crown.setSingleStep(0.05)
        self.spin_size_crown.setFixedWidth(60)
        row_crown.addWidget(self.spin_size_crown)
        row_crown.addStretch()
        vbox_mesh.addLayout(row_crown)

        # 3D 메싱 알고리즘 선택
        row_algo = QHBoxLayout()
        lbl_algo = QLabel("메싱 알고리즘:", group_mesh)
        lbl_algo.setFixedWidth(85)
        row_algo.addWidget(lbl_algo)
        self.combo_mesh_algo = QComboBox(group_mesh)
        self.combo_mesh_algo.addItem("자동 (Auto - Delaunay/HXT)", "auto")
        self.combo_mesh_algo.addItem("Delaunay 3D (안정 표준형)", "delaunay")
        self.combo_mesh_algo.addItem("HXT 3D (고속 병렬)", "hxt")
        self.combo_mesh_algo.addItem("Netgen 3D (정밀 CAD형)", "netgen")
        row_algo.addWidget(self.combo_mesh_algo)
        vbox_mesh.addLayout(row_algo)

        # 메쉬 품질 최적화 옵션
        self.chk_optimize_netgen = QCheckBox("3D 볼륨 메쉬 품질 최적화 적용 (Relocate3D 노드 정돈)", group_mesh)
        self.chk_optimize_netgen.setChecked(True)
        self.chk_optimize_netgen.setToolTip("사면체 요소의 형상 왜곡을 개선하고 노드 위치를 정돈합니다.")
        self.chk_optimize_netgen.setStyleSheet("font-size: 10px; color: #abb2bf;")
        vbox_mesh.addWidget(self.chk_optimize_netgen)

        # 메쉬 생성 버튼 및 진행바
        self.btn_mesh = QPushButton("⚙️ 체적 메쉬 생성", group_mesh)
        self.btn_mesh.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_mesh.clicked.connect(self._on_mesh_clicked)
        vbox_mesh.addWidget(self.btn_mesh)

        self.mesh_progress_bar = QProgressBar(group_mesh)
        self.mesh_progress_bar.setRange(0, 100)
        self.mesh_progress_bar.setValue(0)
        self.mesh_progress_bar.setFixedHeight(14)
        self.mesh_progress_bar.setTextVisible(False)
        self.mesh_progress_bar.setVisible(False)
        vbox_mesh.addWidget(self.mesh_progress_bar)

        self.lbl_mesh_status = QLabel("대기 중", group_mesh)
        self.lbl_mesh_status.setStyleSheet("color: #888899; font-size: 10px;")
        vbox_mesh.addWidget(self.lbl_mesh_status)

        self.lbl_mesh_summary = QLabel("절점: -- | 요소: --", group_mesh)
        self.lbl_mesh_summary.setStyleSheet("color: #abb2bf; font-size: 11px; font-weight: bold;")
        vbox_mesh.addWidget(self.lbl_mesh_summary)

        layout.addWidget(group_mesh)

        # 2. 물성 할당 섹션
        group_mat = QGroupBox("2. 물성 할당 (Material)", self)
        vbox_mat = QVBoxLayout(group_mat)
        vbox_mat.setContentsMargins(6, 6, 6, 6)
        vbox_mat.setSpacing(4)

        # 프리셋 선택
        row_preset = QHBoxLayout()
        row_preset.addWidget(QLabel("프리셋:", group_mat))
        self.btn_choose_preset = QPushButton("선택...", group_mat)
        self.btn_choose_preset.setStyleSheet("padding: 2px 6px; font-size: 11px;")
        self.btn_choose_preset.adjustSize()
        self.btn_choose_preset.setFixedWidth(self.btn_choose_preset.sizeHint().width())
        self.btn_choose_preset.clicked.connect(self._on_choose_preset)
        row_preset.addWidget(self.btn_choose_preset)

        self.lbl_preset_path = QLabel("기본 presets.json", group_mat)
        self.lbl_preset_path.setStyleSheet("color: #61afef; font-size: 10px;")
        row_preset.addWidget(self.lbl_preset_path, stretch=1)
        vbox_mat.addLayout(row_preset)
        self._custom_preset_path: Optional[str] = None

        # Bin 개수
        row_bins = QHBoxLayout()
        lbl_bins = QLabel("E 구간(Bin) 개수:", group_mat)
        lbl_bins.setFixedWidth(122)
        row_bins.addWidget(lbl_bins)
        self.spin_bins = QSpinBox(group_mat)
        self.spin_bins.setRange(5, 100)
        self.spin_bins.setValue(30)
        self.spin_bins.setFixedWidth(60)  # 기존 폭의 1/2로 축소
        row_bins.addWidget(self.spin_bins)

        self.btn_copy_mat = QPushButton("현재 물성 복사", group_mat)
        self.btn_copy_mat.setStyleSheet("padding: 2px 6px; font-size: 11px;")
        self.btn_copy_mat.adjustSize()
        self.btn_copy_mat.setFixedWidth(self.btn_copy_mat.sizeHint().width())
        self.btn_copy_mat.setToolTip("현재 물성 값을 클립보드에 복사")
        self.btn_copy_mat.clicked.connect(self._on_copy_material_clicked)
        row_bins.addWidget(self.btn_copy_mat)

        row_bins.addStretch()
        vbox_mat.addLayout(row_bins)

        # 물성 할당 버튼
        self.btn_assign_material = QPushButton("🔬 물성 할당 실행", group_mat)
        self.btn_assign_material.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_assign_material.clicked.connect(self._on_assign_material_clicked)
        vbox_mat.addWidget(self.btn_assign_material)

        self.lbl_mat_summary = QLabel("탄성계수(E): -- MPa\n할당 재료 수: --", group_mat)
        self.lbl_mat_summary.setStyleSheet("color: #abb2bf; font-size: 10px;")
        vbox_mat.addWidget(self.lbl_mat_summary)

        layout.addWidget(group_mat)

        # 3. 경계조건 및 하중 섹션
        group_bc = QGroupBox("3. 경계조건 및 하중 (BC & Loads)", self)
        vbox_bc = QVBoxLayout(group_bc)
        vbox_bc.setContentsMargins(6, 6, 6, 6)
        vbox_bc.setSpacing(4)

        # 고정면 설정 (Z-축 또는 축 지정)
        vbox_bc.addWidget(QLabel("하단 고정면 설정:", group_bc))
        row_fix = QHBoxLayout()
        self.combo_fix_mode = QComboBox(group_bc)
        self.combo_fix_mode.addItem("Z-축", "z_plane")
        self.combo_fix_mode.addItem("축 지정", "box")
        self.combo_fix_mode.currentIndexChanged.connect(self._on_fix_mode_changed)

        self.btn_sync_fix = QPushButton("현재 고정위치 복사", group_bc)
        self.btn_sync_fix.setToolTip("Z-축: 현재 슬라이스 Z 위치 설정 / 축 지정: 2D 단면 사각형 영역 고정면 설정")
        self.btn_sync_fix.clicked.connect(self._on_sync_current_fix)

        row_fix.addWidget(self.combo_fix_mode, stretch=1)
        row_fix.addWidget(self.btn_sync_fix, stretch=2)
        vbox_bc.addLayout(row_fix)

        self.lbl_fix_info = QLabel("고정면: Z-축 기본값 미지정", group_bc)
        self.lbl_fix_info.setStyleSheet("color: #abb2bf; font-size: 10px;")
        vbox_bc.addWidget(self.lbl_fix_info)

        # 교합 하중 부여 버튼 및 임플란트 부착 체크박스
        row_load_ctrl = QHBoxLayout()
        self.btn_apply_load = QPushButton("교합 하중 부여", group_bc)
        self.btn_apply_load.setIcon(IconFactory.occlusal_load(16))
        self.btn_apply_load.setToolTip("3D 뷰어에 교합 하중 조작 화살표 및 기즈모를 표시합니다.")
        self.btn_apply_load.setStyleSheet("font-weight: bold;")
        self.btn_apply_load.clicked.connect(self._on_apply_load_clicked)

        self.chk_snap_implant = QCheckBox("임플란트에 부착", group_bc)
        self.chk_snap_implant.setChecked(True)
        self.chk_snap_implant.setToolTip("체크 시 하중 화살표가 가장 가까운 임플란트 표면에 자동으로 스냅됩니다.")
        self.chk_snap_implant.toggled.connect(self._on_snap_implant_toggled)

        row_load_ctrl.addWidget(self.btn_apply_load)
        row_load_ctrl.addWidget(self.chk_snap_implant)
        vbox_bc.addLayout(row_load_ctrl)

        # 하중 프리셋
        vbox_bc.addWidget(QLabel("교합 하중 프리셋:", group_bc))
        self.combo_load_preset = QComboBox(group_bc)
        self._load_presets = default_load_presets()
        for k, v in self._load_presets.items():
            self.combo_load_preset.addItem(v["name"], k)
        self.combo_load_preset.currentIndexChanged.connect(self._on_load_preset_changed)
        vbox_bc.addWidget(self.combo_load_preset)

        # 하중 세부 크기 및 경사
        row_load = QHBoxLayout()
        row_load.addWidget(QLabel("크기(N):", group_bc))
        self.spin_load_mag = QDoubleSpinBox(group_bc)
        self.spin_load_mag.setRange(1.0, 2000.0)
        self.spin_load_mag.setValue(100.0)
        self.spin_load_mag.valueChanged.connect(self._on_load_mag_changed)
        row_load.addWidget(self.spin_load_mag)

        row_load.addWidget(QLabel("기울기(°):", group_bc))
        self.spin_load_angle = QDoubleSpinBox(group_bc)
        self.spin_load_angle.setRange(0.0, 90.0)
        self.spin_load_angle.setValue(0.0)
        self.spin_load_angle.valueChanged.connect(self._on_load_angle_changed)
        row_load.addWidget(self.spin_load_angle)
        vbox_bc.addLayout(row_load)

        # 하중 적용 상단 마진
        row_margin = QHBoxLayout()
        row_margin.addWidget(QLabel("하중 적용 영역 (상단 mm):", group_bc))
        self.spin_load_margin = QDoubleSpinBox(group_bc)
        self.spin_load_margin.setRange(0.2, 10.0)
        self.spin_load_margin.setValue(2.0)
        row_margin.addWidget(self.spin_load_margin)
        vbox_bc.addLayout(row_margin)

        # 스크류 체결 예하중 (Bolt Preload) 설정
        row_preload = QHBoxLayout()
        self.chk_preload = QCheckBox("스크류 예하중(Preload) 적용", group_bc)
        self.chk_preload.setChecked(True)
        self.chk_preload.setToolTip("어버트먼트 스크류 조임 토크로 발생하는 축방향 인장 예하중을 적용합니다.")
        self.chk_preload.setStyleSheet("font-weight: bold; color: #d1b87a;")
        row_preload.addWidget(self.chk_preload)

        lbl_pre_force = QLabel("예하중(N):", group_bc)
        lbl_pre_force.setStyleSheet("font-size: 11px;")
        row_preload.addWidget(lbl_pre_force)
        self.spin_preload_force = QDoubleSpinBox(group_bc)
        self.spin_preload_force.setRange(10.0, 1000.0)
        self.spin_preload_force.setValue(200.0)
        self.spin_preload_force.setSingleStep(10.0)
        self.spin_preload_force.setFixedWidth(70)
        self.spin_preload_force.setToolTip("스크류 체결 인장 예하중 (통상 150~300 N)")
        row_preload.addWidget(self.spin_preload_force)
        row_preload.addStretch()
        vbox_bc.addLayout(row_preload)

        self.chk_preload.toggled.connect(self.spin_preload_force.setEnabled)

        # 경계조건 검증 버튼
        self.btn_validate_bc = QPushButton("✔️ 경계조건 적용 및 검증", group_bc)
        self.btn_validate_bc.setStyleSheet("font-weight: bold; padding: 5px;")
        self.btn_validate_bc.clicked.connect(self._on_setup_bc_clicked)
        vbox_bc.addWidget(self.btn_validate_bc)

        self.lbl_bc_status = QLabel("경계조건 미설정", group_bc)
        self.lbl_bc_status.setStyleSheet("color: #888899; font-size: 10px;")
        self.lbl_bc_status.setWordWrap(True)
        vbox_bc.addWidget(self.lbl_bc_status)

        layout.addWidget(group_bc)

        # 4. 유한요소해석 섹션
        group_solver = QGroupBox("4. CalculiX 해석 (Solver)", self)
        vbox_solver = QVBoxLayout(group_solver)
        vbox_solver.setContentsMargins(6, 6, 6, 6)
        vbox_solver.setSpacing(4)

        # ccx 경로
        row_ccx = QHBoxLayout()
        row_ccx.addWidget(QLabel("ccx 경로:", group_solver))
        self.edit_ccx_path = QLineEdit(group_solver)
        from solver.runner import CalculixRunner
        dummy_runner = CalculixRunner()
        detected_ccx = dummy_runner.resolve_ccx_path()
        self.edit_ccx_path.setText(detected_ccx or "")
        self.edit_ccx_path.setPlaceholderText("ccx 실행 경로 (기본 PATH 탐색)")
        self.btn_browse_ccx = QPushButton("...", group_solver)
        self.btn_browse_ccx.setFixedWidth(28)
        self.btn_browse_ccx.clicked.connect(self._on_browse_ccx)
        row_ccx.addWidget(self.edit_ccx_path)
        row_ccx.addWidget(self.btn_browse_ccx)
        vbox_solver.addLayout(row_ccx)

        # 스레드 수
        row_threads = QHBoxLayout()
        row_threads.addWidget(QLabel("스레드 수:", group_solver))
        self.spin_threads = QSpinBox(group_solver)
        self.spin_threads.setRange(1, 64)
        self.spin_threads.setValue(8)
        row_threads.addWidget(self.spin_threads)
        vbox_solver.addLayout(row_threads)

        # 버튼들
        row_btns = QHBoxLayout()
        self.btn_export_inp = QPushButton("📄 .inp 저장", group_solver)
        self.btn_export_inp.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:disabled {
                color: #6c707e;
            }
        """)
        self.btn_export_inp.clicked.connect(self._on_export_inp_clicked)
        row_btns.addWidget(self.btn_export_inp)

        self.btn_run_solver = QPushButton("▶ 해석 실행", group_solver)
        self.btn_run_solver.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: #ffffff;
                border: 1px solid #3878b2;
                border-radius: 4px;
                font-weight: bold;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #2b5288;
                border-color: #5294e2;
            }
            QPushButton:disabled {
                background-color: #2b2d30;
                color: #6c707e;
                border: 1px solid #3e4249;
            }
        """)
        self.btn_run_solver.clicked.connect(self._on_run_solver_clicked)
        row_btns.addWidget(self.btn_run_solver)

        self.btn_cancel_solver = QPushButton("⏹ 취소", group_solver)
        self.btn_cancel_solver.setEnabled(False)
        self.btn_cancel_solver.clicked.connect(self._on_cancel_solver_clicked)
        row_btns.addWidget(self.btn_cancel_solver)
        vbox_solver.addLayout(row_btns)

        # 진행바
        self.solver_progress_bar = QProgressBar(group_solver)
        self.solver_progress_bar.setRange(0, 100)
        self.solver_progress_bar.setValue(0)
        self.solver_progress_bar.setFixedHeight(14)
        self.solver_progress_bar.setTextVisible(False)
        vbox_solver.addWidget(self.solver_progress_bar)

        self.lbl_solver_status = QLabel("해석 준비", group_solver)
        self.lbl_solver_status.setStyleSheet("color: #888899; font-size: 10px;")
        vbox_solver.addWidget(self.lbl_solver_status)

        # 로그창
        self.txt_solver_log = QPlainTextEdit(group_solver)
        self.txt_solver_log.setReadOnly(True)
        self.txt_solver_log.setMaximumHeight(90)
        self.txt_solver_log.setStyleSheet("background-color: #1a1a1f; color: #abb2bf; font-family: monospace; font-size: 9px;")
        vbox_solver.addWidget(self.txt_solver_log)

        layout.addWidget(group_solver)
        layout.addStretch()

    def _bind_signals(self):
        # StateManager 신호
        self.state_mgr.fea_geometry_changed.connect(lambda *_: self._update_ui_state())
        self.state_mgr.mesh_model_changed.connect(lambda *_: self._update_ui_state())
        self.state_mgr.material_assignment_changed.connect(lambda *_: self._update_ui_state())
        self.state_mgr.bc_set_changed.connect(lambda *_: self._update_ui_state())
        self.state_mgr.fea_stale_changed.connect(self._on_stale_changed)

        # FeaController 신호
        self.controller.meshing_progress.connect(self._on_mesh_progress)
        self.controller.meshing_finished.connect(self._on_mesh_finished)
        self.controller.meshing_error.connect(self._on_mesh_error)

        self.controller.material_assigned.connect(self._on_mat_assigned)
        self.controller.material_error.connect(self._on_mat_error)

        self.controller.bc_validated.connect(self._on_bc_validated)
        self.controller.bc_error.connect(self._on_bc_error)

        self.controller.solver_started.connect(self._on_solver_started)
        self.controller.solver_progress.connect(self._on_solver_progress)
        self.controller.solver_log.connect(self._on_solver_log)
        self.controller.solver_finished.connect(self._on_solver_finished)
        self.controller.solver_error.connect(self._on_solver_error)

    def _update_ui_state(self):
        """기하, 메쉬, 물성, BC 상태에 따라 버튼 활성/비활성화 제어"""
        has_geom = (self.state_mgr.fea_geometry is not None)
        has_mesh = (self.state_mgr.mesh_model is not None)
        has_mat = (self.state_mgr.material_assignment is not None)
        has_bc = (self.state_mgr.bc_set is not None)

        self.btn_mesh.setEnabled(has_geom)
        self.btn_assign_material.setEnabled(has_mesh)
        self.btn_validate_bc.setEnabled(has_mesh)

        # 메쉬와 경계조건이 충족되면 해석 및 내보내기 활성화 (물성은 미할당 시 실행 시점에 자동 할당)
        can_run = has_mesh and has_bc
        self.btn_export_inp.setEnabled(can_run)
        self.btn_run_solver.setEnabled(can_run)

        # 2단계 물성 요약 라벨 피드백
        if has_mesh and not has_mat:
            self.lbl_mat_summary.setText("⚠️ 물성 미할당 (해석 실행 시 자동 할당 가능)")
            self.lbl_mat_summary.setStyleSheet("color: #e5c07b; font-size: 10px;")

        # 4단계 해석 상태 피드백 및 툴팁 업데이트
        if not has_mesh:
            self.lbl_solver_status.setText("해석 준비: 1단계 [메쉬 생성] 필요")
            self.lbl_solver_status.setStyleSheet("color: #888899; font-size: 10px;")
            self.btn_run_solver.setToolTip("1단계 [메쉬 생성]을 먼저 완료해야 합니다.")
            self.btn_export_inp.setToolTip("1단계 [메쉬 생성]을 먼저 완료해야 합니다.")
        elif not has_bc:
            self.lbl_solver_status.setText("⚠️ 3단계 [경계조건 적용 및 검증] 필요")
            self.lbl_solver_status.setStyleSheet("color: #e5c07b; font-size: 10px;")
            self.btn_run_solver.setToolTip("3단계 [경계조건 적용 및 검증]을 먼저 완료해야 합니다.")
            self.btn_export_inp.setToolTip("3단계 [경계조건 적용 및 검증]을 먼저 완료해야 합니다.")
        elif not has_mat:
            self.lbl_solver_status.setText("💡 해석 준비 완료 (실행 시 물성 자동 할당)")
            self.lbl_solver_status.setStyleSheet("color: #61afef; font-size: 10px;")
            self.btn_run_solver.setToolTip("해석 실행 시 2단계 물성 할당이 자동으로 함께 수행됩니다.")
            self.btn_export_inp.setToolTip(".inp 저장 시 2단계 물성 할당이 자동으로 함께 수행됩니다.")
        else:
            self.lbl_solver_status.setText("✅ 해석 준비 완료 (CalculiX 실행 가능)")
            self.lbl_solver_status.setStyleSheet("color: #98c379; font-size: 10px;")
            self.btn_run_solver.setToolTip("CalculiX 유한요소해석을 시작합니다.")
            self.btn_export_inp.setToolTip("CalculiX 해석용 .inp 데크 파일을 생성합니다.")

        # 초기 고정면 Z 기본값 설정 (볼륨 또는 지오메트리 바운드 기준)
        if has_geom and self._fixed_z is None:
            bounds = getattr(self.state_mgr, "bounds", None)
            if bounds and len(bounds) == 6:
                z_min = bounds[4]
                self._fixed_z = z_min + 2.0
                if self.combo_fix_mode.currentData() == "z_plane":
                    self.lbl_fix_info.setText(f"고정면: Z-축 기본값 ({self._fixed_z:.1f} mm)")

    def _on_stale_changed(self, is_stale: bool):
        self.banner_stale.setVisible(is_stale)
        if is_stale:
            self.lbl_mesh_status.setText("⚠️ 상류 지오메트리가 변경됨 (재생성 권장)")
            self.lbl_mesh_status.setStyleSheet("color: #e5c07b; font-size: 10px;")

    # --- 메쉬 슬롯 ---
    def _on_mesh_clicked(self):
        # 중간에 체적 메쉬 생성을 다시하면 물성, 하단면 고정, 교합하중 등 이후 단계 데이터 모두 제거
        self._fixed_z = None
        self.state_mgr.set_material_assignment(None)
        self.state_mgr.set_bc_set(None)
        self.state_mgr.set_last_frd_path(None)
        self.state_mgr.set_bc_fix_box(None)

        self.lbl_mat_summary.setText("물성 미할당 (골 HU 샘플링 필요)")
        self.lbl_mat_summary.setStyleSheet("color: #888899; font-size: 10px;")
        self.lbl_bc_status.setText("경계조건 미설정")
        self.lbl_bc_status.setStyleSheet("color: #888899; font-size: 10px;")
        self.lbl_fix_info.setText("고정면: Z-축 기본값 미지정")
        self.lbl_solver_status.setText("⚠️ 1단계 [메쉬 생성] 필요")
        self.lbl_solver_status.setStyleSheet("color: #888899; font-size: 10px;")

        viewer = self._get_fea_viewer()
        if viewer:
            viewer.set_load_gizmo_visible(False)
            for act in viewer.bc_actors:
                viewer.renderer.RemoveActor(act)
            viewer.bc_actors.clear()
            viewer.render()

        elem_type = self.combo_elem_type.currentData()
        config = AnalysisConfig(
            mesh_size_far_mm=self.spin_size_far.value(),
            mesh_size_near_mm=self.spin_size_near.value(),
            element_type=elem_type,
            ccx_path=self.edit_ccx_path.text() or None,
            n_threads=self.spin_threads.value(),
            mesh_algorithm_3d=self.combo_mesh_algo.currentData(),
            optimize_netgen=self.chk_optimize_netgen.isChecked(),
            mesh_size_abutment_mm=self.spin_size_abut.value(),
            mesh_size_screw_mm=self.spin_size_screw.value(),
            mesh_size_crown_mm=self.spin_size_crown.value(),
            screw_contact_tol_mm=0.20,
            preload_enabled=self.chk_preload.isChecked(),
            preload_force_n=self.spin_preload_force.value(),
        )
        self.btn_mesh.setEnabled(False)
        self.mesh_progress_bar.setVisible(True)
        self.mesh_progress_bar.setValue(0)
        self.lbl_mesh_status.setText("메쉬 생성 시작...")
        self.controller.start_meshing(config)

    def _on_mesh_progress(self, pct: int, msg: str):
        self.mesh_progress_bar.setValue(pct)
        self.lbl_mesh_status.setText(msg)

    def _on_mesh_finished(self, report_str: str):
        self.mesh_progress_bar.setVisible(False)
        self.btn_mesh.setEnabled(True)
        self.lbl_mesh_status.setText("✅ 메쉬 생성 성공")
        self.lbl_mesh_status.setStyleSheet("color: #98c379; font-size: 10px;")

        mesh = self.state_mgr.mesh_model
        if mesh:
            self.lbl_mesh_summary.setText(f"절점: {mesh.total_nodes():,}개 | 요소: {mesh.total_elements():,}개")
        self.banner_stale.setVisible(False)
        self._update_ui_state()

    def _on_mesh_error(self, err_msg: str):
        self.mesh_progress_bar.setVisible(False)
        self.btn_mesh.setEnabled(True)
        self.lbl_mesh_status.setText("❌ 메쉬 생성 오류")
        self.lbl_mesh_status.setStyleSheet("color: #e06c75; font-size: 10px;")
        QMessageBox.critical(self, "메쉬 오류", err_msg)

    # --- 물성 슬롯 ---
    def _on_copy_material_clicked(self):
        """현재 물성 정보 또는 프리셋 정보를 클립보드에 복사하고 알림 메시지를 표시"""
        lines = []
        mat_assign = self.state_mgr.material_assignment
        if mat_assign is not None and mat_assign.materials:
            lines.append("=== 현재 할당된 재료 물성 (Material Assignment) ===")
            for i, m in enumerate(mat_assign.materials):
                dens_str = f", 밀도: {m.density:.3e} t/mm³" if m.density is not None else ""
                lines.append(f"[{i:02d}] {m.name}: E = {m.E:,.1f} MPa, nu = {m.nu}{dens_str}")
            if mat_assign.element_material:
                lines.append("\n=== 부품별 요소 할당 현황 ===")
                for part_name, elem_mats in mat_assign.element_material.items():
                    lines.append(f"- {part_name}: {len(elem_mats):,}개 요소")
        else:
            lines.append("=== 물성 설정 정보 (Preset & Settings) ===")
            preset_name = os.path.basename(self._custom_preset_path) if self._custom_preset_path else "기본 presets.json"
            lines.append(f"프리셋: {preset_name}")
            lines.append(f"E 구간(Bin) 개수: {self.spin_bins.value()}")
            preset_path = self._custom_preset_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "material", "presets.json"
            )
            if os.path.exists(preset_path):
                try:
                    import json
                    with open(preset_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    lines.append("\n[프리셋 세부 내용]")
                    lines.append(json.dumps(data, indent=2, ensure_ascii=False))
                except Exception as ex:
                    lines.append(f"(프리셋 파일 읽기 오류: {ex})")

        text_to_copy = "\n".join(lines)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text_to_copy)

        QMessageBox.information(
            self,
            "물성 복사",
            "현재 물성 값이 클립보드에 복사되었습니다."
        )

    def _on_choose_preset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "물성 매핑 프리셋 JSON 선택", "", "JSON 파일 (*.json);;모든 파일 (*.*)"
        )
        if file_path:
            self._custom_preset_path = file_path
            self.lbl_preset_path.setText(os.path.basename(file_path))

    def _on_assign_material_clicked(self):
        self.btn_assign_material.setEnabled(False)
        self.lbl_mat_summary.setText("물성 샘플링 및 Binning 계산 중...")
        self.controller.assign_materials(
            preset_path=self._custom_preset_path,
            n_bins=self.spin_bins.value(),
        )

    def _on_mat_assigned(self, summary: dict):
        self.btn_assign_material.setEnabled(True)
        e_min = summary.get("e_min", 0.0)
        e_max = summary.get("e_max", 0.0)
        e_mean = summary.get("e_mean", 0.0)
        n_mats = summary.get("num_materials", 0)
        self.lbl_mat_summary.setText(
            f"탄성계수(E): {e_min:.0f} ~ {e_max:.0f} MPa (평균 {e_mean:.0f} MPa)\n"
            f"생성된 재료 수: {n_mats}개 (CORTICAL, TITANIUM 포함)"
        )
        self._update_ui_state()

    def _on_mat_error(self, err_msg: str):
        self.btn_assign_material.setEnabled(True)
        self.lbl_mat_summary.setText(f"❌ 오류: {err_msg}")
        QMessageBox.warning(self, "물성 할당 오류", err_msg)

    # --- 경계조건 슬롯 ---
    def _get_fea_viewer(self):
        """윈도우 트리에서 FeaGeometryViewWidget 인스턴스 탐색"""
        win = self.window()
        if hasattr(win, "mpr_container") and hasattr(win.mpr_container, "fea_viewer"):
            viewer = win.mpr_container.fea_viewer
            if viewer and viewer.on_load_params_changed_callback != self._on_gizmo_angle_dragged:
                viewer.on_load_params_changed_callback = self._on_gizmo_angle_dragged
            return viewer
        return None

    def _on_gizmo_angle_dragged(self, tilt_deg: float):
        """화살표 기즈모 드래그 회전 시 스핀박스 각도 실시간 동기화"""
        self.spin_load_angle.blockSignals(True)
        self.spin_load_angle.setValue(round(tilt_deg, 1))
        self.spin_load_angle.blockSignals(False)

    def _on_fix_mode_changed(self, idx_or_text):
        """고정면 모드(Z-축 또는 축 지정) 변경 슬롯"""
        mode = self.combo_fix_mode.currentData()
        if mode == "box":
            self.state_mgr.bc_fix_drawing_mode_toggled.emit(True)
            self.lbl_fix_info.setText("축 지정 모드: 2D 단면 뷰에서 사각형 영역을 드래그하세요.")
            self.lbl_fix_info.setStyleSheet("color: #61afef; font-size: 10px;")
        else:
            self.state_mgr.bc_fix_drawing_mode_toggled.emit(False)
            if self._fixed_z is not None:
                self.lbl_fix_info.setText(f"고정면: Z-축 ({self._fixed_z:.1f} mm)")
            else:
                self.lbl_fix_info.setText("고정면: Z-축 기본값 미지정")
            self.lbl_fix_info.setStyleSheet("color: #abb2bf; font-size: 10px;")

    def _on_sync_current_fix(self):
        """현재 고정위치 복사 슬롯"""
        mode = self.combo_fix_mode.currentData()
        if mode == "box":
            self.state_mgr.bc_fix_drawing_mode_toggled.emit(False)
            box = self.state_mgr.bc_fix_box
            if box is not None:
                self.lbl_fix_info.setText(
                    f"지정 축 영역 복사됨 (X:{box[0]:.0f}~{box[1]:.0f}, Y:{box[2]:.0f}~{box[3]:.0f}, Z:{box[4]:.0f}~{box[5]:.0f})"
                )
                self.lbl_fix_info.setStyleSheet("color: #98c379; font-size: 10px;")
                QMessageBox.information(self, "고정 위치 복사", "2D 사각형 및 축 전체 길이와 결합된 3D 고정 영역이 복사되었습니다.")
            else:
                self.lbl_fix_info.setText("지정된 2D 영역 없음 (2D 단면에서 먼저 사각형을 그리세요)")
                self.lbl_fix_info.setStyleSheet("color: #e5c07b; font-size: 10px;")
                QMessageBox.warning(self, "고정 영역 미지정", "2D 뷰에서 먼저 사각형을 드래그하여 영역을 지정해 주세요.")
        else:
            curr_z = self.state_mgr.current_coords[2]
            self._fixed_z = curr_z
            self.lbl_fix_info.setText(f"고정면: Z={curr_z:.1f} mm 복사 완료")
            self.lbl_fix_info.setStyleSheet("color: #98c379; font-size: 10px;")
            QMessageBox.information(self, "고정 위치 복사", f"현재 슬라이스 Z 위치({curr_z:.1f} mm)가 하단 고정면으로 복사되었습니다.")

    def _on_apply_load_clicked(self):
        """교합 하중 부여 버튼 클릭 슬롯: 이전 화살표를 제거하고 새로운 화살표 생성 (기존 힘과 각도는 유지)"""
        viewer = self._get_fea_viewer()
        if viewer:
            current_mag = self.spin_load_mag.value()
            current_angle = self.spin_load_angle.value()
            if hasattr(viewer, "recreate_load_gizmo"):
                viewer.recreate_load_gizmo(magnitude_N=current_mag, tilt_deg=current_angle)
            else:
                viewer.set_load_gizmo_visible(True)
            self._update_gizmo_parameters()

    def _on_snap_implant_toggled(self, checked: bool):
        """임플란트 스냅 체크 토글 슬롯"""
        viewer = self._get_fea_viewer()
        if viewer and viewer.load_gizmo:
            viewer.load_gizmo.snap_to_implant = checked

    def _on_load_mag_changed(self, val: float):
        self._update_gizmo_parameters()

    def _on_load_angle_changed(self, val: float):
        self._update_gizmo_parameters()

    def _on_load_preset_changed(self):
        key = self.combo_load_preset.currentData()
        if key in self._load_presets:
            p = self._load_presets[key]
            self.spin_load_mag.blockSignals(True)
            self.spin_load_angle.blockSignals(True)
            self.spin_load_mag.setValue(p["magnitude_N"])
            self.spin_load_angle.setValue(p["tilt_deg"])
            self.spin_load_mag.blockSignals(False)
            self.spin_load_angle.blockSignals(False)
            self._update_gizmo_parameters()

    def _update_gizmo_parameters(self):
        """현재 UI 입력값을 3D 뷰어 화살표 기즈모에 반영"""
        viewer = self._get_fea_viewer()
        if viewer and viewer.load_gizmo:
            axis = viewer._get_implant_axis()

            # 임플란트 상단 중심점 위치 획득
            origin = None
            if hasattr(viewer, "_get_implant_top_position"):
                origin = viewer._get_implant_top_position()

            if origin is None:
                nodes = viewer._get_target_nodes()
                if nodes is not None and len(nodes) > 0:
                    max_idx = int(np.argmax(nodes[:, 2]))
                    origin = nodes[max_idx]
                elif viewer.load_gizmo.origin is not None:
                    origin = viewer.load_gizmo.origin
                else:
                    origin = np.array([0.0, 0.0, 0.0])

            tilt = self.spin_load_angle.value()
            if abs(tilt) > 1e-4:
                up = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
                side = np.cross(axis, up)
                side /= max(1e-6, np.linalg.norm(side))
                rad = math.radians(tilt)
                direction = axis * math.cos(rad) + np.cross(side, axis) * math.sin(rad)
                direction /= max(1e-6, np.linalg.norm(direction))
            else:
                direction = axis.copy()

            viewer.load_gizmo.update_parameters(
                origin=origin,
                direction=direction,
                magnitude_N=self.spin_load_mag.value(),
                tilt_deg=tilt
            )
            viewer.render()

    def _on_setup_bc_clicked(self):
        # 1. 3D 뷰어 교합 하중 화살표 감추기
        viewer = self._get_fea_viewer()
        load_pos = None
        load_dir = None
        if viewer:
            if viewer.load_gizmo and viewer.load_gizmo.is_visible:
                load_pos = viewer.load_gizmo.origin
                load_dir = viewer.load_gizmo.direction
            viewer.set_load_gizmo_visible(False)

        # 2. 파라미터 수집
        fix_mode_data = self.combo_fix_mode.currentData()
        fix_box = self.state_mgr.bc_fix_box if fix_mode_data == "box" else None

        # 3. 경계조건 설정 및 검증 실행
        issues = self.controller.setup_boundary_conditions(
            bottom_z=self._fixed_z,
            load_preset_key=self.combo_load_preset.currentData(),
            load_mag=self.spin_load_mag.value(),
            load_angle=self.spin_load_angle.value(),
            top_margin_mm=self.spin_load_margin.value(),
            fix_mode=fix_mode_data,
            fix_box=fix_box,
            load_position=load_pos,
            load_direction=load_dir,
            preload_enabled=self.chk_preload.isChecked(),
            preload_force_n=self.spin_preload_force.value(),
        )

    def _on_bc_validated(self, issues: list):
        if not issues:
            self.lbl_bc_status.setText("✅ 경계조건 및 하중 검증 통과")
            self.lbl_bc_status.setStyleSheet("color: #98c379; font-size: 10px;")
        else:
            txt = "\n".join(issues)
            self.lbl_bc_status.setText(f"⚠️ 검증 주의사항:\n{txt}")
            self.lbl_bc_status.setStyleSheet("color: #e5c07b; font-size: 10px;")
        self._update_ui_state()

    def _on_bc_error(self, err_msg: str):
        self.lbl_bc_status.setText(f"❌ 설정 오류: {err_msg}")
        self.lbl_bc_status.setStyleSheet("color: #e06c75; font-size: 10px;")
        QMessageBox.warning(self, "경계조건 오류", err_msg)

    # --- 해석 슬롯 ---
    def _on_browse_ccx(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "CalculiX(ccx) 실행 파일 선택", "", "모든 파일 (*)")
        if file_path:
            self.edit_ccx_path.setText(file_path)

    def _ensure_material_assigned(self) -> bool:
        """물성이 할당되지 않은 경우 기본 설정으로 자동 할당 시도"""
        if self.state_mgr.material_assignment is not None:
            return True
        self.lbl_solver_status.setText("물성 자동 할당 중 (CT HU 샘플링)...")
        self.lbl_solver_status.setStyleSheet("color: #61afef; font-size: 10px;")
        self.txt_solver_log.appendPlainText("[자동 물성 할당] 2단계 물성 할당을 자동으로 실행합니다...")
        QApplication.processEvents()
        mat = self.controller.assign_materials(n_bins=self.spin_bins.value())
        if mat is None:
            self.txt_solver_log.appendPlainText("[자동 물성 할당 실패] 물성 할당에 실패하여 중단합니다.")
            return False
        self.txt_solver_log.appendPlainText("[자동 물성 할당 완료] CT HU 기반 이종 물성이 성공적으로 할당되었습니다.")
        return True

    def _on_export_inp_clicked(self):
        if not self._ensure_material_assigned():
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "CalculiX .inp 저장", "dental_fea_job.inp", "CalculiX Deck (*.inp)")
        if file_path:
            out = self.controller.export_inp(file_path)
            if out:
                QMessageBox.information(self, ".inp 저장 완료", f"입력 데크 파일이 성공적으로 생성되었습니다:\n{out}")

    def _on_run_solver_clicked(self):
        self.txt_solver_log.clear()
        self.solver_progress_bar.setValue(0)
        if not self._ensure_material_assigned():
            return
        ccx = self.edit_ccx_path.text().strip() or None
        self.controller.start_solver(
            inp_path=None,  # 기본 saved/ 폴더에 자동 생성
            ccx_path=ccx,
            n_threads=self.spin_threads.value(),
        )

    def _on_cancel_solver_clicked(self):
        self.controller.cancel_solver()

    def _on_solver_started(self):
        self.btn_run_solver.setEnabled(False)
        self.btn_cancel_solver.setEnabled(True)
        self.lbl_solver_status.setText("해석 진행 중...")
        self.lbl_solver_status.setStyleSheet("color: #61afef; font-size: 10px;")

    def _on_solver_progress(self, pct: int, msg: str):
        self.solver_progress_bar.setValue(pct)
        self.lbl_solver_status.setText(msg)

    def _on_solver_log(self, line: str):
        self.txt_solver_log.appendPlainText(line)

    def _on_solver_finished(self, success: bool, frd_path: str, msg: str):
        self.btn_run_solver.setEnabled(True)
        self.btn_cancel_solver.setEnabled(False)
        if success:
            self.lbl_solver_status.setText("✅ 해석 완료 (.frd 생성)")
            self.lbl_solver_status.setStyleSheet("color: #98c379; font-size: 10px;")
            QMessageBox.information(self, "해석 완료", f"CalculiX 정적 해석이 완료되었습니다.\n결과 파일: {frd_path}")
        else:
            self.lbl_solver_status.setText("❌ 해석 실패")
            self.lbl_solver_status.setStyleSheet("color: #e06c75; font-size: 10px;")
            QMessageBox.critical(self, "해석 실패", f"해석 중 오류가 발생했습니다:\n{msg}")

    def _on_solver_error(self, err_msg: str):
        self.btn_run_solver.setEnabled(True)
        self.btn_cancel_solver.setEnabled(False)
        self.lbl_solver_status.setText("❌ 실행 오류")
        self.lbl_solver_status.setStyleSheet("color: #e06c75; font-size: 10px;")
        QMessageBox.warning(self, "솔버 오류", err_msg)
