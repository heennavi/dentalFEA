"""
measurement_dialog.py
단면 계측 도구 다이얼로그.
사용자 첨부 이미지(media_1792079476507.png, media_1792079476509.png)의 레이아웃과 기능을 충실히 반영:
- [길이] [각도] [삭제] [모두 지우기] [선분 보이기 체크] [캡처 저장]
- 계측 목록 테이블 (타입, 측정값, 단면, 슬라이스)
- [바로가기] 버튼
"""

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QLabel, QFileDialog, QMessageBox,
    QAbstractItemView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from .icons import IconFactory


class DeselectableTableWidget(QTableWidget):
    """빈 영역 클릭 시 선택을 해제하는 테이블 위젯"""
    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item is None:
            self.clearSelection()
            self.setCurrentItem(None)
            self.setCurrentCell(-1, -1)
        super().mousePressEvent(event)


class MeasurementDialog(QDialog):
    """
    단면 계측 도구 전용 창
    """
    # 측정 모드 변경 시그널: mode ('none', 'length', 'angle')
    mode_changed = Signal(str)
    # 특정 계측 항목 선택 시그널: index (-1 for none)
    item_selected = Signal(int)
    # 특정 계측 항목 삭제 시그널: index
    item_deleted = Signal(int)
    # 전체 계측 삭제 시그널
    all_cleared = Signal()
    # 선분 보이기 토글 시그널: bool
    visibility_toggled = Signal(bool)
    # 특정 항목 바로가기 (단면 뷰 및 슬라이스 이동): axis_name, slice_idx, world_coord
    navigate_to_item = Signal(str, int, object)
    # 캡처 저장 요청 시그널
    request_capture = Signal()
    # 계측 목록 변경(추가/삭제) 시그널
    measurements_updated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("단면 계측 도구")
        self.resize(520, 360)

        self.current_mode = "none"  # 'length', 'angle', 'none'
        self.measurements: List[Dict[str, Any]] = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 상단 도구 버튼 바
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(6)

        self.btn_length = QPushButton("길이")
        self.btn_length.setCheckable(True)
        self.btn_length.clicked.connect(lambda: self._on_mode_button_clicked("length"))

        self.btn_angle = QPushButton("각도")
        self.btn_angle.setCheckable(True)
        self.btn_angle.clicked.connect(lambda: self._on_mode_button_clicked("angle"))

        self.btn_delete = QPushButton("삭제")
        self.btn_delete.clicked.connect(self._on_delete_selected)

        self.btn_clear = QPushButton("모두 지우기")
        self.btn_clear.clicked.connect(self._on_clear_all)

        self.chk_show_lines = QCheckBox("선분 보이기")
        self.chk_show_lines.setChecked(True)
        self.chk_show_lines.toggled.connect(self.visibility_toggled.emit)

        self.btn_capture = QPushButton("캡처 저장")
        self.btn_capture.clicked.connect(self._on_capture_clicked)

        btn_bar.addWidget(self.btn_length)
        btn_bar.addWidget(self.btn_angle)
        btn_bar.addWidget(self.btn_delete)
        btn_bar.addWidget(self.btn_clear)
        btn_bar.addWidget(self.chk_show_lines)
        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_capture)

        layout.addLayout(btn_bar)

        # 계측 목록 테이블 (슬라이스 -> 면번호)
        self.table = DeselectableTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["타입", "측정값", "단면", "면번호"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.cellDoubleClicked.connect(self._on_table_double_clicked)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e24;
                color: #ffffff;
                gridline-color: #33333d;
                border: 1px solid #33333d;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2b2b36;
                color: #dcdce1;
                padding: 4px;
                border: 1px solid #383846;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background-color: #3b5998;
                color: #ffffff;
            }
        """)
        layout.addWidget(self.table)

        # 하단 닫기 버튼 바 ('바로가기' -> '닫기'로 변경)
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        self.btn_close = QPushButton("닫기")
        self.btn_close.clicked.connect(self._on_close_clicked)
        bottom_bar.addWidget(self.btn_close)
        layout.addLayout(bottom_bar)

        # 다이얼로그 전체 스타일시트
        self.setStyleSheet("""
            QDialog {
                background-color: #18181c;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #2c2c34;
                color: #ffffff;
                border: 1px solid #3e3e4a;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3a3a46;
                border-color: #555566;
            }
            QPushButton:checked {
                background-color: #1976d2;
                border-color: #2196f3;
                color: #ffffff;
            }
            QCheckBox {
                color: #dcdce1;
                spacing: 6px;
            }
        """)

    def _on_selection_changed(self):
        selected = self.table.selectedItems()
        if not selected:
            self.item_selected.emit(-1)
            return
        row = self.table.currentRow()
        self.item_selected.emit(row)

    def clear_selection(self):
        """테이블 선택 및 오버레이 하이라이트 완전 해제"""
        self.table.clearSelection()
        self.table.setCurrentItem(None)
        self.table.setCurrentCell(-1, -1)
        self.item_selected.emit(-1)

    def _on_table_double_clicked(self, row: int, col: int):
        self.table.selectRow(row)
        self._navigate_to_row(row)

    def _on_mode_button_clicked(self, mode: str):
        if mode == "length":
            if self.btn_length.isChecked():
                self.btn_angle.setChecked(False)
                self.current_mode = "length"
            else:
                self.current_mode = "none"
        elif mode == "angle":
            if self.btn_angle.isChecked():
                self.btn_length.setChecked(False)
                self.current_mode = "angle"
            else:
                self.current_mode = "none"
        self.mode_changed.emit(self.current_mode)

    def set_mode(self, mode: str):
        self.current_mode = mode
        self.btn_length.setChecked(mode == "length")
        self.btn_angle.setChecked(mode == "angle")
        self.mode_changed.emit(mode)

    def add_measurement(self, item: Dict[str, Any]):
        """
        item keys:
        - type: 'length' or 'angle'
        - value_str: '24.50 mm' or '68.5°'
        - view_name: 'Axial', 'Coronal', 'Sagittal'
        - slice_idx: 120 (면번호는 slice_idx + 1)
        - world_pts: [...]
        """
        self.measurements.append(item)
        row = self.table.rowCount()
        self.table.insertRow(row)

        type_text = "길이" if item.get("type") == "length" else "각도"
        s_idx = item.get("slice_idx", 0)
        plane_no = s_idx + 1

        for c, text in enumerate([type_text, item.get("value_str", ""), item.get("view_name", ""), str(plane_no)]):
            t_item = QTableWidgetItem(text)
            t_item.setFlags(t_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, c, t_item)

        self.table.selectRow(row)
        self.measurements_updated.emit()

    def _on_delete_selected(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.measurements):
            self.measurements.pop(row)
            self.table.removeRow(row)
            self.item_deleted.emit(row)
            self.clear_selection()
            self.measurements_updated.emit()

    def _on_clear_all(self):
        if not self.measurements:
            return
        reply = QMessageBox.question(
            self, "확인", "모든 계측 항목을 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.measurements.clear()
            self.table.setRowCount(0)
            self.all_cleared.emit()
            self.clear_selection()
            self.measurements_updated.emit()

    def _navigate_to_row(self, row: int):
        if 0 <= row < len(self.measurements):
            item = self.measurements[row]
            v_name = item.get("view_name", "")
            s_idx = item.get("slice_idx", 0)
            pts = item.get("world_pts", [])
            center = pts[0] if pts else None
            self.navigate_to_item.emit(v_name, s_idx, center)

    def _on_capture_clicked(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "단면 계측 화면 캡처 저장", "measurement_capture.png",
            "PNG 이미지 (*.png);;JPEG 이미지 (*.jpg);;모든 파일 (*.*)"
        )
        if filepath:
            self.request_capture.emit()

    def to_dict(self) -> List[Dict[str, Any]]:
        """JSON 저장을 위한 계측 데이터 직렬화"""
        return list(self.measurements)

    def from_dict(self, data_list: List[Dict[str, Any]]):
        """JSON 데이터로부터 계측 목록 복원 (일괄 로드 최적화)"""
        self.measurements.clear()
        self.table.setRowCount(0)
        self.table.blockSignals(True)
        for d in data_list:
            item = dict(d)
            self.measurements.append(item)
            row = self.table.rowCount()
            self.table.insertRow(row)

            type_text = "길이" if item.get("type") == "length" else "각도"
            s_idx = item.get("slice_idx", 0)
            plane_no = s_idx + 1

            for c, text in enumerate([type_text, item.get("value_str", ""), item.get("view_name", ""), str(plane_no)]):
                t_item = QTableWidgetItem(text)
                t_item.setFlags(t_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, c, t_item)

        self.table.blockSignals(False)
        self.clear_selection()
        self.measurements_updated.emit()

    def _on_close_clicked(self):
        self.clear_selection()
        self.close()

    def closeEvent(self, event):
        # 창 닫힐 때 측정 모드 및 선택 해제
        self.set_mode("none")
        self.clear_selection()
        # 선분 보이기 상태를 확실히 전달
        self.visibility_toggled.emit(self.chk_show_lines.isChecked())
        super().closeEvent(event)


