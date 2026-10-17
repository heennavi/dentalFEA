"""
image_info_dialog.py
현재 로드된 영상의 상세 메타데이터를 표시하는 팝업 대화상자.
- 검은 바탕(#121214)에 흰 글씨(#ffffff)로 통일된 가독성 높은 테이블
- 클립보드 복사(정보 복사) 기능 제공
"""

from typing import Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QApplication
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

from dental_fea_app.core.state_manager import StateManager


class ImageInfoDialog(QDialog):
    """
    현재 로드된 3D/2D 영상의 세부 규격 및 메타데이터를 정밀하게 표시하는 팝업 대화상자.
    """

    def __init__(self, state_manager: StateManager, parent: Optional[QDialog] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.setWindowTitle("현재 영상 상세 정보")
        self.resize(540, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1e;
                color: #ffffff;
            }
            QTableWidget {
                background-color: #121214;
                gridline-color: #2a2a32;
                border: 1px solid #383842;
                border-radius: 4px;
                color: #ffffff;
                selection-background-color: #264f78;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #1a1a20;
                color: #61afef;
                font-weight: bold;
                padding: 6px;
                border: 1px solid #2a2a32;
            }
            QPushButton {
                background-color: #2b5588;
                border: 1px solid #61afef;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 18px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3b70ab;
            }
            QPushButton#btn_copy {
                background-color: #2c323c;
                border: 1px solid #5c6370;
            }
            QPushButton#btn_copy:hover {
                background-color: #3e4451;
                border-color: #98c379;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 타이틀 헤더
        header_lbl = QLabel("의료 영상 (CT/DICOM) 상세 속성 정보", self)
        header_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #61afef;")
        layout.addWidget(header_lbl)

        # 테이블 위젯 (검은 바탕, 흰 글씨 통일)
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["항목 (Property)", "값 (Value)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(False)

        layout.addWidget(self.table)

        # 하단 버튼 영역 (정보 복사 및 확인)
        btn_box = QHBoxLayout()

        self.btn_copy = QPushButton("정보 복사", self)
        self.btn_copy.setObjectName("btn_copy")
        self.btn_copy.setToolTip("표시된 모든 영상 정보를 클립보드에 복사합니다.")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_box.addWidget(self.btn_copy)

        btn_box.addStretch()

        btn_close = QPushButton("확인", self)
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)

        layout.addLayout(btn_box)

        # 데이터 채우기
        self._populate_info()

    def _copy_to_clipboard(self):
        """테이블 내용을 탭 구분 텍스트로 클립보드에 복사"""
        lines = []
        for row in range(self.table.rowCount()):
            item_p = self.table.item(row, 0)
            item_v = self.table.item(row, 1)
            prop = item_p.text() if item_p else ""
            val = item_v.text() if item_v else ""
            lines.append(f"{prop}\t{val}")

        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(lines))

        # 복사 완료 피드백
        orig_text = self.btn_copy.text()
        self.btn_copy.setText("✔ 복사 완료!")
        QTimer.singleShot(2000, lambda: self.btn_copy.setText(orig_text))

    def _populate_info(self):
        items = []

        if self.state_mgr.vtk_image is None:
            items.append(("상태", "로드된 영상 데이터가 없습니다."))
        else:
            img = self.state_mgr.vtk_image
            dims = self.state_mgr.dimensions
            spc = self.state_mgr.spacing
            origin = self.state_mgr.origin
            bounds = self.state_mgr.bounds
            hu_range = self.state_mgr.scalar_range

            phys_x = (dims[0] - 1) * spc[0] if dims[0] > 1 else 0.0
            phys_y = (dims[1] - 1) * spc[1] if dims[1] > 1 else 0.0
            phys_z = (dims[2] - 1) * spc[2] if dims[2] > 1 else 0.0

            items.append(("영상 차원 (Dimensions)", f"{dims[0]} × {dims[1]} × {dims[2]} voxels"))
            items.append(("총 슬라이스 수 (Z-slices)", f"{dims[2]} slices"))
            items.append(("복셀 간격 (Voxel Spacing)", f"{spc[0]:.4f} × {spc[1]:.4f} × {spc[2]:.4f} mm"))
            items.append(("물리적 크기 (FOV / Extent)", f"{phys_x:.2f} × {phys_y:.2f} × {phys_z:.2f} mm"))
            items.append(("좌표 원점 (Origin)", f"({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}) mm"))
            items.append(("월드 X 범위 (Sagittal Bounds)", f"[{bounds[0]:.2f}, {bounds[1]:.2f}] mm"))
            items.append(("월드 Y 범위 (Coronal Bounds)", f"[{bounds[2]:.2f}, {bounds[3]:.2f}] mm"))
            items.append(("월드 Z 범위 (Axial Bounds)", f"[{bounds[4]:.2f}, {bounds[5]:.2f}] mm"))
            items.append(("HU 명암도 범위 (Scalar Range)", f"최소 {hu_range[0]:.1f} ~ 최대 {hu_range[1]:.1f} HU"))
            items.append(("스칼라 데이터 타입", str(img.GetScalarTypeAsString())))
            items.append(("스칼라 컴포넌트 수", str(img.GetNumberOfScalarComponents())))

            # 추가 메타데이터
            meta = self.state_mgr.metadata
            if meta:
                for k, v in meta.items():
                    if k not in ["spacing", "origin", "dimensions", "bounds", "scalar_range"]:
                        items.append((str(k), str(v)))

        self.table.setRowCount(len(items))
        for row, (prop, val) in enumerate(items):
            item_p = QTableWidgetItem(prop)
            item_v = QTableWidgetItem(val)
            # 검은 바탕에 흰 글씨 통일
            item_p.setForeground(QColor("#ffffff"))
            item_v.setForeground(QColor("#ffffff"))
            item_p.setBackground(QColor("#121214"))
            item_v.setBackground(QColor("#121214"))
            self.table.setItem(row, 0, item_p)
            self.table.setItem(row, 1, item_v)
