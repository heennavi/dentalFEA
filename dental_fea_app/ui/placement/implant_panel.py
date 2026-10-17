"""
implant_panel.py
좌측 제어 패널에 임베딩되는 "임플란트 배치" 제어 위젯.
- STL 불러오기 버튼 + 파일명/치수 정보 표시
- 위치 X/Y/Z (mm) 스핀박스, 회전 yaw/pitch/roll (도) 스핀박스
- [현재 crosshair 위치에 배치] 버튼
- [식립 방향 지정] : 두 점(시작점/끝점)을 crosshair로 캡처하여 축 결정
- 이동 스텝 크기 선택 (0.1 / 0.5 / 1.0 mm)
- [포즈 리셋] 버튼
- 뼈 겹침 비율 (bone_overlap_ratio) 계산 및 표시
- 스핀박스 연속 조작 시 약 30ms 디바운스/스로틀링 타이머 적용
"""

import os
from typing import Optional
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QDoubleSpinBox, QComboBox,
    QGroupBox, QFileDialog, QMessageBox, QFrame, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.placement.implant_manager import (
    ImplantModel,
    bone_overlap_ratio,
)


class ImplantPanel(QWidget):
    """임플란트 STL 로드 및 위치/자세 실시간 편집 패널"""

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.implant_model: Optional[ImplantModel] = None

        # 식립 방향 지정(두 점 캡처) 버퍼
        self._dir_point1: Optional[np.ndarray] = None
        self._dir_point2: Optional[np.ndarray] = None

        # 스핀박스 연속 변경 디바운스 타이머 (30ms)
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(30)
        self._update_timer.timeout.connect(self._apply_pose_from_spinboxes)

        self._block_spin_signals = False

        self._init_ui()

        # StateManager 신호 바인딩
        self.state_mgr.volume_loaded.connect(self._on_volume_loaded)
        self.state_mgr.implant_model_changed.connect(self._on_implant_model_changed)
        self.state_mgr.implant_pose_changed.connect(self._on_implant_pose_changed)
        self.state_mgr.segmentation_result_changed.connect(self._on_segmentation_changed)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. STL 파일 불러오기 그룹
        stl_group = QGroupBox("임플란트 STL 파일", self)
        stl_vbox = QVBoxLayout(stl_group)
        stl_vbox.setContentsMargins(6, 6, 6, 6)
        stl_vbox.setSpacing(4)

        self.btn_load_stl = QPushButton("📂 STL 불러오기...", stl_group)
        self.btn_load_stl.setStyleSheet("""
            QPushButton {
                background-color: #2b3a4a;
                color: #61afef;
                border: 1px solid #45627d;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #354a5f;
            }
        """)
        self.btn_load_stl.clicked.connect(self._load_stl_dialog)
        stl_vbox.addWidget(self.btn_load_stl)

        self.lbl_stl_name = QLabel("파일: (로드되지 않음)", stl_group)
        self.lbl_stl_name.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.lbl_stl_name.setWordWrap(True)
        stl_vbox.addWidget(self.lbl_stl_name)

        self.lbl_stl_dim = QLabel("치수 (D x L): --", stl_group)
        self.lbl_stl_dim.setStyleSheet("color: #98c379; font-size: 11px; font-weight: bold;")
        stl_vbox.addWidget(self.lbl_stl_dim)

        layout.addWidget(stl_group)

        # 2. 빠른 배치 및 식립축 그룹
        place_group = QGroupBox("배치 및 식립축", self)
        place_vbox = QVBoxLayout(place_group)
        place_vbox.setContentsMargins(6, 6, 6, 6)
        place_vbox.setSpacing(4)

        self.btn_place_crosshair = QPushButton("🎯 현재 Crosshair에 배치", place_group)
        self.btn_place_crosshair.setToolTip("현재 2D/3D 십자선 교차점 위치를 임플란트 Apex(식립끝)로 지정합니다.")
        self.btn_place_crosshair.setStyleSheet("""
            QPushButton {
                background-color: #3e3e28;
                color: #e5c07b;
                border: 1px solid #7d7d45;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555535;
            }
        """)
        self.btn_place_crosshair.clicked.connect(self._place_at_crosshair)
        place_vbox.addWidget(self.btn_place_crosshair)

        # 식립 방향 지정 (두 점 캡처)
        dir_box = QHBoxLayout()
        dir_box.setSpacing(4)
        self.btn_capture_p1 = QPushButton("1) 시작점(Apex)", place_group)
        self.btn_capture_p1.setToolTip("현재 십자선 위치를 식립 시작점(Apex)으로 등록")
        self.btn_capture_p1.clicked.connect(self._capture_dir_p1)
        dir_box.addWidget(self.btn_capture_p1)

        self.btn_capture_p2 = QPushButton("2) 끝점(Top)", place_group)
        self.btn_capture_p2.setToolTip("현재 십자선 위치를 식립 끝점(Top)으로 등록하여 축 설정")
        self.btn_capture_p2.clicked.connect(self._capture_dir_p2)
        dir_box.addWidget(self.btn_capture_p2)
        place_vbox.addLayout(dir_box)

        self.lbl_dir_status = QLabel("방향: Z축 수직 (기본값)", place_group)
        self.lbl_dir_status.setStyleSheet("color: #888888; font-size: 10px;")
        place_vbox.addWidget(self.lbl_dir_status)

        layout.addWidget(place_group)

        # 3. 위치 및 회전 미세조정 그룹
        adjust_group = QGroupBox("위치 및 회전 미세조정", self)
        adj_layout = QVBoxLayout(adjust_group)
        adj_layout.setContentsMargins(6, 6, 6, 6)
        adj_layout.setSpacing(4)

        # 이동 스텝 선택
        step_box = QHBoxLayout()
        step_box.addWidget(QLabel("스텝:", adjust_group))
        self.combo_step = QComboBox(adjust_group)
        self.combo_step.addItems(["0.1 mm", "0.5 mm", "1.0 mm"])
        self.combo_step.setCurrentIndex(1)  # 기본 0.5mm
        self.combo_step.currentIndexChanged.connect(self._on_step_changed)
        step_box.addWidget(self.combo_step)
        step_box.addStretch()

        self.btn_reset_pose = QPushButton("↺ 리셋", adjust_group)
        self.btn_reset_pose.setToolTip("초기 원점 위치 및 각도로 리셋")
        self.btn_reset_pose.clicked.connect(self._reset_pose)
        step_box.addWidget(self.btn_reset_pose)
        adj_layout.addLayout(step_box)

        # 위치 X, Y, Z (mm) 스핀박스
        grid_pos = QGridLayout()
        grid_pos.setSpacing(4)

        self.spin_x = self._create_double_spin(-1000.0, 1000.0, 0.5)
        self.spin_y = self._create_double_spin(-1000.0, 1000.0, 0.5)
        self.spin_z = self._create_double_spin(-1000.0, 1000.0, 0.5)

        grid_pos.addWidget(QLabel("X (mm):", adjust_group), 0, 0)
        grid_pos.addWidget(self.spin_x, 0, 1)
        grid_pos.addWidget(QLabel("Y (mm):", adjust_group), 1, 0)
        grid_pos.addWidget(self.spin_y, 1, 1)
        grid_pos.addWidget(QLabel("Z (mm):", adjust_group), 2, 0)
        grid_pos.addWidget(self.spin_z, 2, 1)
        adj_layout.addLayout(grid_pos)

        # 회전 Yaw, Pitch, Roll (deg) 스핀박스
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("border-color: #383842; margin-top: 2px; margin-bottom: 2px;")
        adj_layout.addWidget(line)

        grid_rot = QGridLayout()
        grid_rot.setSpacing(4)

        self.spin_yaw = self._create_double_spin(-180.0, 180.0, 1.0)
        self.spin_pitch = self._create_double_spin(-180.0, 180.0, 1.0)
        self.spin_roll = self._create_double_spin(-180.0, 180.0, 1.0)

        grid_rot.addWidget(QLabel("Yaw (°):", adjust_group), 0, 0)
        grid_rot.addWidget(self.spin_yaw, 0, 1)
        grid_rot.addWidget(QLabel("Pitch (°):", adjust_group), 1, 0)
        grid_rot.addWidget(self.spin_pitch, 1, 1)
        grid_rot.addWidget(QLabel("Roll (°):", adjust_group), 2, 0)
        grid_rot.addWidget(self.spin_roll, 2, 1)
        adj_layout.addLayout(grid_rot)

        layout.addWidget(adjust_group)

        # 4. 뼈 겹침 비율 및 적합도 정보
        fit_group = QGroupBox("식립 적합도 (Bone Fit)", self)
        fit_vbox = QVBoxLayout(fit_group)
        fit_vbox.setContentsMargins(6, 6, 6, 6)
        fit_vbox.setSpacing(4)

        self.lbl_overlap_ratio = QLabel("뼈 겹침 비율: -- %", fit_group)
        self.lbl_overlap_ratio.setStyleSheet("color: #61afef; font-weight: bold; font-size: 11px;")
        fit_vbox.addWidget(self.lbl_overlap_ratio)

        self.btn_calc_fit = QPushButton("적합도 재계산", fit_group)
        self.btn_calc_fit.clicked.connect(self._update_bone_fit_ratio)
        fit_vbox.addWidget(self.btn_calc_fit)

        layout.addWidget(fit_group)
        layout.addStretch()

        self._update_controls_enabled()

    def _create_double_spin(self, min_val: float, max_val: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #21252b;
                color: #dcdfe4;
                border: 1px solid #3c3c46;
                border-radius: 3px;
                padding: 2px;
            }
        """)
        spin.valueChanged.connect(self._on_spinbox_value_changed)
        return spin

    def _update_controls_enabled(self):
        has_implant = (self.implant_model is not None and self.implant_model.raw_polydata is not None)
        has_volume = (self.state_mgr.vtk_image is not None)

        self.btn_place_crosshair.setEnabled(has_implant and has_volume)
        self.btn_capture_p1.setEnabled(has_implant and has_volume)
        self.btn_capture_p2.setEnabled(has_implant and has_volume)
        self.btn_reset_pose.setEnabled(has_implant)
        self.btn_calc_fit.setEnabled(has_implant and has_volume)

        for spin in [self.spin_x, self.spin_y, self.spin_z, self.spin_yaw, self.spin_pitch, self.spin_roll]:
            spin.setEnabled(has_implant)

    def _on_step_changed(self, index: int):
        step_val = [0.1, 0.5, 1.0][index]
        self.spin_x.setSingleStep(step_val)
        self.spin_y.setSingleStep(step_val)
        self.spin_z.setSingleStep(step_val)

    def _load_stl_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "임플란트 STL 파일 선택", "", "STL 3D 모델 (*.stl);;모든 파일 (*.*)"
        )
        if not file_path:
            return

        model = ImplantModel()
        try:
            info = model.load_stl(file_path)
            if info.get("warning"):
                QMessageBox.warning(self, "단위 주의", info["warning"])
        except Exception as e:
            QMessageBox.critical(self, "STL 로드 실패", f"STL 파일을 로드하는 중 오류가 발생했습니다:\n{str(e)}")
            return

        # 모델 등록
        self.state_mgr.set_implant_model(model)

        # 초기 배치: 볼륨이 로드되어 있으면 현재 십자선 위치로 자동 이동
        if self.state_mgr.vtk_image is not None:
            self._place_at_crosshair()

    def _on_implant_model_changed(self, model: Optional[ImplantModel]):
        self.implant_model = model
        self._update_controls_enabled()

        if model is not None and model.raw_polydata is not None:
            fname = os.path.basename(model.stl_path) if model.stl_path else "메모리 STL 모델"
            self.lbl_stl_name.setText(f"파일: {fname}")
            bounds = model.get_local_bounds()
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            diameter = max(dx, dy)
            self.lbl_stl_dim.setText(f"치수: Ø {diameter:.1f} mm x L {dz:.1f} mm")
            self._sync_spinboxes_from_model()
            self._update_bone_fit_ratio()
        else:
            self.lbl_stl_name.setText("파일: (로드되지 않음)")
            self.lbl_stl_dim.setText("치수 (D x L): --")
            self.lbl_overlap_ratio.setText("뼈 겹침 비율: -- %")

    def _on_implant_pose_changed(self, model: Optional[ImplantModel]):
        if model is not None and not self._block_spin_signals:
            self._sync_spinboxes_from_model()
            self._update_bone_fit_ratio()

    def _sync_spinboxes_from_model(self):
        if self.implant_model is None:
            return
        self._block_spin_signals = True
        try:
            pos = self.implant_model.position
            self.spin_x.setValue(pos[0])
            self.spin_y.setValue(pos[1])
            self.spin_z.setValue(pos[2])

            yaw, pitch, roll = self.implant_model.get_euler_angles("zyx")
            self.spin_yaw.setValue(yaw)
            self.spin_pitch.setValue(pitch)
            self.spin_roll.setValue(roll)
        finally:
            self._block_spin_signals = False

    def _on_spinbox_value_changed(self):
        if self._block_spin_signals or self.implant_model is None:
            return
        # 연속 입력 스로틀링 (30ms 타이머 재시작)
        self._update_timer.start()

    def _apply_pose_from_spinboxes(self):
        if self.implant_model is None or self._block_spin_signals:
            return

        pos = [self.spin_x.value(), self.spin_y.value(), self.spin_z.value()]
        yaw = self.spin_yaw.value()
        pitch = self.spin_pitch.value()
        roll = self.spin_roll.value()

        # 모델 포즈 갱신
        self.implant_model.position = pos
        self.implant_model.set_yaw_pitch_roll(yaw, pitch, roll)

    def _place_at_crosshair(self):
        if self.implant_model is None:
            return
        # 현재 십자선 월드 좌표
        cx, cy, cz = self.state_mgr.current_coords
        # 기본 장축: +Z 축 수직 상향
        current_axis = self.implant_model.get_world_axis()
        if np.linalg.norm(current_axis) < 1e-4:
            current_axis = [0.0, 0.0, 1.0]

        self.implant_model.place_at(point_world=[cx, cy, cz], axis_world=current_axis)
        self.lbl_dir_status.setText(f"Apex: ({cx:.1f}, {cy:.1f}, {cz:.1f})")

    def _capture_dir_p1(self):
        cx, cy, cz = self.state_mgr.current_coords
        self._dir_point1 = np.array([cx, cy, cz], dtype=np.float64)
        self.lbl_dir_status.setText(f"P1(Apex): ({cx:.1f}, {cy:.1f}, {cz:.1f}) 등록됨. P2를 찍으세요.")

    def _capture_dir_p2(self):
        if self._dir_point1 is None:
            QMessageBox.warning(self, "순서 오류", "먼저 1) 시작점(Apex)을 등록하십시오.")
            return

        cx, cy, cz = self.state_mgr.current_coords
        self._dir_point2 = np.array([cx, cy, cz], dtype=np.float64)

        axis_vec = self._dir_point2 - self._dir_point1
        norm = np.linalg.norm(axis_vec)
        if norm < 0.1:
            QMessageBox.warning(self, "좌표 오류", "시작점과 끝점 사이의 거리가 너무 가깝습니다.")
            return

        axis_unit = axis_vec / norm
        if self.implant_model is not None:
            self.implant_model.place_at(point_world=self._dir_point1, axis_world=axis_unit)
            self.lbl_dir_status.setText(f"식립축 정렬 완료 (길이: {norm:.1f} mm)")

    def _reset_pose(self):
        if self.implant_model is not None:
            self.implant_model.reset_pose()

    def _on_volume_loaded(self, vtk_img):
        self._update_controls_enabled()
        self._update_bone_fit_ratio()

    def _on_segmentation_changed(self, result):
        self._update_bone_fit_ratio()

    def _update_bone_fit_ratio(self):
        if self.implant_model is None or self.state_mgr.vtk_image is None:
            self.lbl_overlap_ratio.setText("뼈 겹침 비율: -- %")
            return

        seg_res = self.state_mgr.segmentation_result
        if seg_res is None or seg_res.bone_mask is None:
            self.lbl_overlap_ratio.setText("뼈 겹침 비율: (분할 마스크 필요)")
            return

        poly_transformed = self.implant_model.get_transformed_polydata()
        mask_geom = {
            "origin": self.state_mgr.origin,
            "spacing": self.state_mgr.spacing,
        }
        ratio = bone_overlap_ratio(poly_transformed, seg_res.bone_mask, mask_geom)
        pct = ratio * 100.0
        self.lbl_overlap_ratio.setText(f"뼈 겹침 비율: {pct:.1f} %")
