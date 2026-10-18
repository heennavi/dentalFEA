"""
implant_transform_dialog.py
식립된 임플란트의 위치 이동(Translation) 및 각도 회전(Rotation)을 조절하는 모달리스 다이얼로그.
사용자 첨부 이미지(media_1792292139292.png, media_1792292139293.png, media_1792292139296.png)의
라벨 및 레이아웃을 충실히 반영하고 실시간 2D/3D 연동을 제공합니다.
"""
from typing import Optional
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QPushButton
)
from PySide6.QtCore import Qt, Signal

from dental_fea_app.placement.implant_placement_manager import ImplantPlacementManager, PlacedImplant
from dental_fea_app.core.state_manager import StateManager


class ImplantTransformDialog(QDialog):
    """
    임플란트 위치 및 각도 미세조정 모달리스 대화상자
    """
    applied = Signal()
    cancelled = Signal()

    def __init__(self, implant: PlacedImplant, state_mgr: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.implant = implant
        self.state_mgr = state_mgr
        self.placement_mgr = state_mgr.implant_placement_mgr

        # 초기 값 저장 (취소 및 초기값 복원용)
        self.initial_pos = np.copy(implant.position)
        self.initial_orient = np.copy(implant.orientation)

        self._block_signals = False

        self.setWindowTitle(f"임플란트 위치 및 각도 변경 - [{implant.spec.name}]")
        self.setModal(False)
        self.resize(320, 290)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self._init_ui()
        self._load_current_values()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 1. 위치 이동 (Translation) 그룹 (media_1792292139292.png)
        pos_group = QGroupBox("위치 이동 (Translation)", self)
        pos_grid = QGridLayout(pos_group)
        pos_grid.setContentsMargins(10, 10, 10, 10)
        pos_grid.setSpacing(6)

        pos_grid.addWidget(QLabel("X (좌우, mm):", pos_group), 0, 0)
        self.spin_x = self._create_double_spin(-1000.0, 1000.0, 0.5, 2)
        pos_grid.addWidget(self.spin_x, 0, 1)

        pos_grid.addWidget(QLabel("Y (전후, mm):", pos_group), 1, 0)
        self.spin_y = self._create_double_spin(-1000.0, 1000.0, 0.5, 2)
        pos_grid.addWidget(self.spin_y, 1, 1)

        pos_grid.addWidget(QLabel("Z (상하, mm):", pos_group), 2, 0)
        self.spin_z = self._create_double_spin(-1000.0, 1000.0, 0.5, 2)
        pos_grid.addWidget(self.spin_z, 2, 1)

        layout.addWidget(pos_group)

        # 2. 각도 회전 (Rotation) 그룹 (media_1792292139293.png)
        rot_group = QGroupBox("각도 회전 (Rotation)", self)
        rot_grid = QGridLayout(rot_group)
        rot_grid.setContentsMargins(10, 10, 10, 10)
        rot_grid.setSpacing(6)

        rot_grid.addWidget(QLabel("Yaw (Z축 회전, °):", rot_group), 0, 0)
        self.spin_yaw = self._create_double_spin(-180.0, 180.0, 1.0, 1)
        rot_grid.addWidget(self.spin_yaw, 0, 1)

        rot_grid.addWidget(QLabel("Pitch (X축 회전, °):", rot_group), 1, 0)
        self.spin_pitch = self._create_double_spin(-180.0, 180.0, 1.0, 1)
        rot_grid.addWidget(self.spin_pitch, 1, 1)

        rot_grid.addWidget(QLabel("Roll (Y축 회전, °):", rot_group), 2, 0)
        self.spin_roll = self._create_double_spin(-180.0, 180.0, 1.0, 1)
        rot_grid.addWidget(self.spin_roll, 2, 1)

        layout.addWidget(rot_group)

        # 3. 하단 액션 버튼 바 (media_1792292139296.png)
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(6)

        self.btn_reset = QPushButton("초기값 복원", self)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 6px 10px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        self.btn_reset.clicked.connect(self._on_reset_clicked)

        self.btn_cancel = QPushButton("취소", self)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)

        self.btn_apply = QPushButton("적용 완료", self)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_apply.clicked.connect(self._on_apply_clicked)

        btn_bar.addWidget(self.btn_reset)
        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_cancel)
        btn_bar.addWidget(self.btn_apply)

        layout.addLayout(btn_bar)

        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 6px;
                margin-top: 8px;
                font-weight: bold;
                color: #38bdf8;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
            }
            QDoubleSpinBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 3px;
                padding: 2px 4px;
                font-size: 11px;
            }
            QDoubleSpinBox:focus {
                border-color: #38bdf8;
            }
        """)

    def _create_double_spin(self, min_val: float, max_val: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(self._on_value_changed)
        return spin

    def _load_current_values(self):
        self._block_signals = True
        try:
            self.spin_x.setValue(float(self.implant.position[0]))
            self.spin_y.setValue(float(self.implant.position[1]))
            self.spin_z.setValue(float(self.implant.position[2]))

            self.spin_yaw.setValue(float(self.implant.orientation[0]))
            self.spin_pitch.setValue(float(self.implant.orientation[1]))
            self.spin_roll.setValue(float(self.implant.orientation[2]))
        finally:
            self._block_signals = False

    def _on_value_changed(self):
        if self._block_signals:
            return

        new_pos = np.array([self.spin_x.value(), self.spin_y.value(), self.spin_z.value()], dtype=np.float64)
        new_rot = np.array([self.spin_yaw.value(), self.spin_pitch.value(), self.spin_roll.value()], dtype=np.float64)

        self.implant.position = new_pos
        self.implant.orientation = new_rot

        self.placement_mgr._update_implant_transform(self.implant)
        self.placement_mgr.update_2d_slice_contours(self.state_mgr.current_coords)
        self.state_mgr.implant_transformed.emit(self.implant.id)
        self.state_mgr.request_render_all()

    def _on_reset_clicked(self):
        """초기 좌표 및 각도로 복원하고 뷰 갱신 (창은 유지)"""
        self._block_signals = True
        try:
            self.spin_x.setValue(float(self.initial_pos[0]))
            self.spin_y.setValue(float(self.initial_pos[1]))
            self.spin_z.setValue(float(self.initial_pos[2]))

            self.spin_yaw.setValue(float(self.initial_orient[0]))
            self.spin_pitch.setValue(float(self.initial_orient[1]))
            self.spin_roll.setValue(float(self.initial_orient[2]))
        finally:
            self._block_signals = False

        self._on_value_changed()

    def _on_cancel_clicked(self):
        """원래 값으로 복구 후 창 닫기"""
        self._on_reset_clicked()
        self.cancelled.emit()
        self.reject()

    def _on_apply_clicked(self):
        """현재 값 확정 후 창 닫기"""
        self.applied.emit()
        self.accept()
