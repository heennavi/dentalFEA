"""
segmentation_panel.py
좌측 제어 패널 내 "2. 분할" 워크플로 탭 UI 위젯.
- 뼈 임계값, 분할 방식(distance/threshold), 피질골 두께, HU 파라미터 조절
- QThread 비동기 실행 및 프로그레스바
- 2D/3D 오버레이 표시 제어 (피질골/수질골 체크박스 및 불투명도)
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QGroupBox, QSlider, QFrame, QMessageBox
)
from PySide6.QtCore import Qt

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.ui.segmentation.segmentation_worker import SegmentationWorker


class SegmentationPanel(QWidget):
    """
    피질골/수질골 분할 제어 및 마스크 표시 설정 패널 위젯.
    """

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.worker: Optional[SegmentationWorker] = None

        self._init_ui()

        # StateManager 신호 연결
        self.state_mgr.volume_loaded.connect(self._on_volume_loaded)
        self.state_mgr.segmentation_result_changed.connect(self._on_segmentation_result_changed)

    def _init_ui(self):
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(6)

        # 1. 분할 파라미터 그룹
        param_group = QGroupBox("분할 설정 (Parameters)", self)
        param_layout = QVBoxLayout(param_group)
        param_layout.setContentsMargins(6, 6, 6, 6)
        param_layout.setSpacing(5)

        # 뼈 하한 임계값 (HU)
        row_bone = QHBoxLayout()
        row_bone.addWidget(QLabel("뼈 임계값 (HU):", param_group))
        self.spin_bone_th = QDoubleSpinBox(param_group)
        self.spin_bone_th.setRange(-500.0, 3000.0)
        self.spin_bone_th.setValue(200.0)
        self.spin_bone_th.setSingleStep(50.0)
        self.spin_bone_th.setToolTip("뼈 조직을 추출할 최소 CT HU 값 (기본 200 HU)")
        row_bone.addWidget(self.spin_bone_th)
        param_layout.addLayout(row_bone)

        # 분할 방식 (Method)
        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("분할 방식:", param_group))
        self.combo_method = QComboBox(param_group)
        self.combo_method.addItem("거리 기반 (Distance)", "distance")
        self.combo_method.addItem("임계값 기반 (Threshold)", "threshold")
        self.combo_method.setToolTip("치과 CBCT에서는 두께가 일관된 거리 기반(Distance) 방식을 권장합니다.")
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        row_method.addWidget(self.combo_method)
        param_layout.addLayout(row_method)

        # 피질골 두께 (mm) - Distance 방식용
        self.row_thickness = QHBoxLayout()
        self.lbl_thickness = QLabel("피질골 두께 (mm):", param_group)
        self.row_thickness.addWidget(self.lbl_thickness)
        self.spin_thickness = QDoubleSpinBox(param_group)
        self.spin_thickness.setRange(0.5, 6.0)
        self.spin_thickness.setValue(1.5)
        self.spin_thickness.setSingleStep(0.1)
        self.spin_thickness.setToolTip("하악골 피질골의 목표 두께 (mm 단위)")
        self.row_thickness.addWidget(self.spin_thickness)
        param_layout.addLayout(self.row_thickness)

        # 피질골 기준 HU - Threshold 방식용
        self.row_cortical_hu = QHBoxLayout()
        self.lbl_cortical_hu = QLabel("피질골 HU:", param_group)
        self.row_cortical_hu.addWidget(self.lbl_cortical_hu)
        self.spin_cortical_hu = QDoubleSpinBox(param_group)
        self.spin_cortical_hu.setRange(200.0, 3000.0)
        self.spin_cortical_hu.setValue(700.0)
        self.spin_cortical_hu.setSingleStep(50.0)
        self.spin_cortical_hu.setToolTip("임계값 방식 시 고밀도 피질골 판별 기준 HU")
        self.row_cortical_hu.addWidget(self.spin_cortical_hu)
        param_layout.addLayout(self.row_cortical_hu)
        self._update_method_inputs()

        # 최대 연결 성분만 유지
        self.chk_keep_largest = QCheckBox("최대 연결 성분만 유지", param_group)
        self.chk_keep_largest.setChecked(True)
        self.chk_keep_largest.setToolTip("하악골/상악골 본체 외의 고립된 미세 노이즈 제거")
        param_layout.addWidget(self.chk_keep_largest)

        # 구멍 메우기
        self.chk_fill_holes = QCheckBox("슬라이스별 구멍 메우기", param_group)
        self.chk_fill_holes.setChecked(True)
        param_layout.addWidget(self.chk_fill_holes)

        # 분할 실행 버튼
        self.btn_run = QPushButton("⚡ 분할 실행 (Run Segmentation)", param_group)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 4px;
                border-radius: 4px;
                border: 1px solid #3878b2;
            }
            QPushButton:hover {
                background-color: #275284;
                border-color: #61afef;
            }
            QPushButton:disabled {
                background-color: #2c2c34;
                color: #666670;
                border-color: #444452;
            }
        """)
        self.btn_run.clicked.connect(self._run_segmentation)
        param_layout.addWidget(self.btn_run)

        # 진행 표시 프로그레스바 및 상태 라벨
        self.progress_bar = QProgressBar(param_group)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setVisible(False)
        param_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("영상 데이터를 로드하십시오.", param_group)
        self.lbl_status.setStyleSheet("color: #888899; font-size: 11px;")
        param_layout.addWidget(self.lbl_status)

        layout.addWidget(param_group)

        # 2. 오버레이 표시 제어 그룹
        view_group = QGroupBox("마스크 표시 (Overlay)", self)
        view_layout = QVBoxLayout(view_group)
        view_layout.setContentsMargins(6, 6, 6, 6)
        view_layout.setSpacing(4)

        # 피질골 체크박스 (빨강 색상 인디케이터)
        self.chk_show_cortical = QCheckBox("피질골 (Cortical) ■", view_group)
        self.chk_show_cortical.setStyleSheet("color: #ff6666; font-weight: bold;")
        self.chk_show_cortical.setChecked(True)
        self.chk_show_cortical.toggled.connect(self._on_display_changed)
        view_layout.addWidget(self.chk_show_cortical)

        # 수질골 체크박스 (파랑 색상 인디케이터)
        self.chk_show_cancellous = QCheckBox("수질골 (Cancellous) ■", view_group)
        self.chk_show_cancellous.setStyleSheet("color: #66a3ff; font-weight: bold;")
        self.chk_show_cancellous.setChecked(True)
        self.chk_show_cancellous.toggled.connect(self._on_display_changed)
        view_layout.addWidget(self.chk_show_cancellous)

        # 뼈 전체 마스크 체크박스
        self.chk_show_bone = QCheckBox("뼈 전체 외곽 (Bone)", view_group)
        self.chk_show_bone.setStyleSheet("color: #cccccc;")
        self.chk_show_bone.setChecked(False)
        self.chk_show_bone.toggled.connect(self._on_display_changed)
        view_layout.addWidget(self.chk_show_bone)

        # 3D 표면 표시 체크박스 (기본 해제 - 필요한 경우에만 켜서 성능 확보)
        self.chk_show_3d_surface = QCheckBox("3D 창 표면 표시", view_group)
        self.chk_show_3d_surface.setStyleSheet("color: #e5c07b; font-weight: bold;")
        self.chk_show_3d_surface.setChecked(False)
        self.chk_show_3d_surface.setToolTip("3D 볼륨 창에 피질골/수질골 표면 메쉬를 렌더링합니다.")
        self.chk_show_3d_surface.toggled.connect(self._on_display_changed)
        view_layout.addWidget(self.chk_show_3d_surface)

        # 오버레이 불투명도 슬라이더
        view_layout.addWidget(QLabel("오버레이 불투명도:", view_group))
        h_op = QHBoxLayout()
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal, view_group)
        self.slider_opacity.setRange(5, 100)
        self.slider_opacity.setValue(45)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        h_op.addWidget(self.slider_opacity)

        self.lbl_opacity = QLabel("45%", view_group)
        self.lbl_opacity.setFixedWidth(32)
        h_op.addWidget(self.lbl_opacity)
        view_layout.addLayout(h_op)

        layout.addWidget(view_group)
        layout.addStretch()

        self._update_button_states()

    def _on_method_changed(self):
        self._update_method_inputs()

    def _update_method_inputs(self):
        method = self.combo_method.currentData()
        is_distance = (method == "distance")
        self.lbl_thickness.setVisible(is_distance)
        self.spin_thickness.setVisible(is_distance)
        self.lbl_cortical_hu.setVisible(not is_distance)
        self.spin_cortical_hu.setVisible(not is_distance)

    def _update_button_states(self):
        has_volume = (self.state_mgr.vtk_image is not None)
        is_running = (self.worker is not None and self.worker.isRunning())

        self.btn_run.setEnabled(has_volume and not is_running)
        if not has_volume:
            self.lbl_status.setText("영상 데이터를 로드하십시오.")
        elif not is_running:
            if self.state_mgr.segmentation_result is None:
                self.lbl_status.setText("분할 파라미터를 설정하고 실행하세요.")

    def _on_volume_loaded(self, vtk_image):
        self._update_button_states()
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def _run_segmentation(self):
        if self.state_mgr.vtk_image is None:
            QMessageBox.warning(self, "영상 없음", "먼저 CT 영상을 로드하십시오.")
            return

        method = self.combo_method.currentData()
        params = {
            "bone_threshold": self.spin_bone_th.value(),
            "method": method,
            "cortical_thickness_mm": self.spin_thickness.value(),
            "cortical_hu": self.spin_cortical_hu.value(),
            "keep_largest": self.chk_keep_largest.isChecked(),
            "fill_holes": self.chk_fill_holes.isChecked(),
        }

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("분할 연산 시작...")

        # QThread 워커 생성 및 시작
        self.worker = SegmentationWorker(
            volume_data=self.state_mgr.vtk_image,
            params=params,
            parent=self
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished_work.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()

    def _on_worker_progress(self, percent: int, msg: str):
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(msg)

    def _on_worker_finished(self, result):
        self.progress_bar.setValue(100)
        self.lbl_status.setText("분할 완료 (결과가 화면에 반영되었습니다).")
        self.btn_run.setEnabled(True)

        # StateManager에 분할 결과 등록 -> 2D/3D 뷰어에 신호 전달
        self.state_mgr.set_segmentation_result(result)
        self._emit_visibility()

    def _on_worker_failed(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("분할 중 오류 발생")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "분할 오류", error_msg)

    def _on_display_changed(self):
        self._emit_visibility()

    def _on_opacity_changed(self, value: int):
        self.lbl_opacity.setText(f"{value}%")
        self._emit_visibility()

    def _emit_visibility(self):
        vis = {
            "show_cortical": self.chk_show_cortical.isChecked(),
            "show_cancellous": self.chk_show_cancellous.isChecked(),
            "show_bone": self.chk_show_bone.isChecked(),
            "show_3d_surface": self.chk_show_3d_surface.isChecked(),
            "opacity": self.slider_opacity.value() / 100.0,
        }
        self.state_mgr.set_segmentation_visibility(vis)

    def _on_segmentation_result_changed(self, result):
        if result is None:
            self.lbl_status.setText("분할 대기 중")
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
        self._update_button_states()
