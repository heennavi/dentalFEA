"""
draw_dialog.py
그리기 도구 다이얼로그.
사용자 첨부 이미지(media_1792079796390.png)의 레이아웃과 옵션을 충실히 반영:
- [그리기 도구 옵션 설정]
  - 굵기: [ 1 ] [ 2 ] [ 3 ] [ 4 ]
  - 선 형태: [ 실선 ] [ 점선 ]
  - 색상: [ 빨강 ] [ 노랑 ] [ 초록 ] [ 파랑 ] [ 주황 ] [ 사용자 색상... ]
  - 선분 끝 형태: [ 없음 ] [ 한쪽 화살표 ] [ 양쪽 화살표 ] [ 원형 ]
  - 그리기 형태: [ 선분 ] [ 화살표 ] [ 사각형 ] [ 원 ] [ 프리폼 ]
- [그린 도형 목록]
  - 테이블: 번호, 형태, 단면, 색상, 삭제
- 하단: [창이 닫혀도 도형 유지] 체크박스, [모두 지우기]
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QRadioButton, QButtonGroup, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QColorDialog, QMessageBox, QLabel
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush

from .icons import IconFactory


class DrawDialog(QDialog):
    """
    2D 단면 그리기 도구 설정 및 도형 관리 다이얼로그
    """
    # 속성 변경 시그널 (width, is_dashed, color, end_cap, shape_type)
    style_changed = Signal(dict)
    # 특정 도형 선택 시그널: index (-1 for none)
    item_selected = Signal(int)
    # 특정 도형 삭제 시그널: index
    item_deleted = Signal(int)
    # 전체 도형 삭제 시그널
    all_cleared = Signal()
    # 그리기 모드 활성화/비활성화 시그널: bool
    active_toggled = Signal(bool)
    # 화면 표시 가시성 토글 시그널: bool
    visibility_toggled = Signal(bool)
    # 특정 도형 바로가기 (단면 뷰 및 슬라이스 이동): axis_name, slice_idx, world_coord
    navigate_to_item = Signal(str, int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("그리기 도구")
        self.resize(540, 520)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        # 현재 스타일 설정값 (기본 굵기 1)
        self.pen_width = 1
        self.is_dashed = False
        self.current_color = QColor(255, 60, 60)  # 기본 빨강
        self.end_cap = "none"  # 'none', 'single_arrow', 'double_arrow', 'circle', 'double_circle'
        self.shape_type = "line"  # 'select', 'line', 'arrow', 'rect', 'circle', 'freeform'
        self.keep_visible_after_close = True

        self.shapes: List[Dict[str, Any]] = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 1. 상단: 그리기 도구 옵션 설정 그룹
        opt_group = QGroupBox("그리기 도구 옵션 설정")
        opt_layout = QVBoxLayout(opt_group)
        opt_layout.setSpacing(8)

        # 굵기 (기본값 1)
        w_row = QHBoxLayout()
        w_row.addWidget(QLabel("굵기:"))
        self.bg_width = QButtonGroup(self)
        for w in [1, 2, 3, 4]:
            btn = QPushButton(str(w))
            btn.setCheckable(True)
            btn.setFixedWidth(36)
            if w == self.pen_width:
                btn.setChecked(True)
            self.bg_width.addButton(btn, w)
            w_row.addWidget(btn)
        self.bg_width.idClicked.connect(self._on_width_changed)
        w_row.addStretch()
        opt_layout.addLayout(w_row)

        # 선 형태
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("선 형태:"))
        self.btn_solid = QPushButton("실선")
        self.btn_solid.setCheckable(True)
        self.btn_solid.setChecked(True)
        self.btn_dash = QPushButton("점선")
        self.btn_dash.setCheckable(True)

        self.bg_style = QButtonGroup(self)
        self.bg_style.addButton(self.btn_solid, 0)
        self.bg_style.addButton(self.btn_dash, 1)
        self.bg_style.idClicked.connect(self._on_style_changed)

        style_row.addWidget(self.btn_solid)
        style_row.addWidget(self.btn_dash)
        style_row.addStretch()
        opt_layout.addLayout(style_row)

        # 색상 (해당 색상을 배경으로 설정)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("색상:"))
        preset_colors = [
            ("빨강", QColor(235, 75, 75), "#ffffff"),
            ("노랑", QColor(245, 215, 30), "#111111"),
            ("초록", QColor(46, 190, 80), "#ffffff"),
            ("파랑", QColor(50, 130, 245), "#ffffff"),
            ("주황", QColor(255, 140, 20), "#ffffff"),
        ]
        self.bg_color = QButtonGroup(self)
        for idx, (cname, col, txt_col) in enumerate(preset_colors):
            btn = QPushButton(cname)
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {col.name()};
                    color: {txt_col};
                    font-weight: bold;
                    border: 2px solid transparent;
                    border-radius: 4px;
                    padding: 4px 10px;
                }}
                QPushButton:hover {{
                    border-color: #ffffff;
                }}
                QPushButton:checked {{
                    border-color: #ffffff;
                    text-decoration: underline;
                }}
            """)
            if idx == 0:
                btn.setChecked(True)
            self.bg_color.addButton(btn, idx)
            color_row.addWidget(btn)

        self.btn_custom_color = QPushButton("사용자 색상...")
        self.btn_custom_color.clicked.connect(self._choose_custom_color)
        color_row.addWidget(self.btn_custom_color)
        color_row.addStretch()

        self._preset_colors = [c[1] for c in preset_colors]
        self.bg_color.idClicked.connect(self._on_color_preset_clicked)
        opt_layout.addLayout(color_row)

        # 선분 끝 형태 (아이콘 적용, 양쪽 원형 포함)
        cap_row = QHBoxLayout()
        cap_row.addWidget(QLabel("선분 끝 형태:"))
        caps = [
            ("없음", "none", IconFactory.cap_none(18)),
            ("한쪽 화살표", "single_arrow", IconFactory.cap_single_arrow(18)),
            ("양쪽 화살표", "double_arrow", IconFactory.cap_double_arrow(18)),
            ("한쪽 원형", "circle", IconFactory.cap_single_circle(18)),
            ("양쪽 원형", "double_circle", IconFactory.cap_double_circle(18)),
        ]
        self.bg_cap = QButtonGroup(self)
        for idx, (label, cap_id, icon) in enumerate(caps):
            btn = QPushButton(label)
            btn.setIcon(icon)
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
            self.bg_cap.addButton(btn, idx)
            cap_row.addWidget(btn)
        self._caps = [c[1] for c in caps]
        self.bg_cap.idClicked.connect(self._on_cap_changed)
        cap_row.addStretch()
        opt_layout.addLayout(cap_row)

        # 그리기 형태 (아이콘 적용, '선택 취소' 버튼 포함)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("그리기 형태:"))
        shapes = [
            ("선택 취소", "select", IconFactory.draw_select(18)),
            ("선분", "line", IconFactory.draw_line(18)),
            ("화살표", "arrow", IconFactory.draw_arrow(18)),
            ("사각형", "rect", IconFactory.draw_rect(18)),
            ("원", "circle", IconFactory.draw_circle(18)),
            ("프리폼", "freeform", IconFactory.draw_freeform(18)),
        ]
        self.bg_shape = QButtonGroup(self)
        for idx, (label, sid, icon) in enumerate(shapes):
            btn = QPushButton(label)
            btn.setIcon(icon)
            btn.setCheckable(True)
            # 기본은 선분(idx=1)
            if idx == 1:
                btn.setChecked(True)
            self.bg_shape.addButton(btn, idx)
            shape_row.addWidget(btn)
        self._shapes = [s[1] for s in shapes]
        self.bg_shape.idClicked.connect(self._on_shape_changed)
        shape_row.addStretch()
        opt_layout.addLayout(shape_row)

        layout.addWidget(opt_group)

        # 2. 중간: 그린 도형 목록 테이블 (번호, 형태, 색상, 단면, 면번호)
        list_group = QGroupBox("그린 도형 목록")
        list_layout = QVBoxLayout(list_group)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["번호", "형태", "색상", "단면", "면번호"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
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
        list_layout.addWidget(self.table)
        layout.addWidget(list_group)

        # 3. 하단 컨트롤 바 ("창이 닫혀도 도형유지" -> "창이 닫혀도 보이기")
        bottom_bar = QHBoxLayout()
        self.chk_keep = QCheckBox("창이 닫혀도 보이기")
        self.chk_keep.setChecked(True)
        self.chk_keep.toggled.connect(self._on_keep_visible_toggled)

        self.btn_del_sel = QPushButton("선택 삭제")
        self.btn_del_sel.clicked.connect(self._delete_selected)

        self.btn_clear_all = QPushButton("모두 지우기")
        self.btn_clear_all.clicked.connect(self._clear_all)

        bottom_bar.addWidget(self.chk_keep)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.btn_del_sel)
        bottom_bar.addWidget(self.btn_clear_all)
        layout.addLayout(bottom_bar)

        # 다이얼로그 전체 스타일
        self.setStyleSheet("""
            QDialog {
                background-color: #18181c;
                color: #e0e0e0;
            }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #33333e;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel {
                color: #cccccc;
                font-weight: 500;
            }
            QPushButton {
                background-color: #282830;
                color: #e0e0e0;
                border: 1px solid #3a3a46;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #383844;
            }
            QPushButton:checked {
                background-color: #1976d2;
                border-color: #2196f3;
                color: #ffffff;
                font-weight: bold;
            }
            QCheckBox {
                color: #dcdce1;
            }
        """)

    def _on_selection_changed(self):
        row = self.table.currentRow()
        self.item_selected.emit(row)

    def _on_table_double_clicked(self, row: int, col: int):
        if 0 <= row < len(self.shapes):
            shape_info = self.shapes[row]
            v_name = shape_info.get("view_name", "")
            s_idx = shape_info.get("slice_idx", 0)
            pts = shape_info.get("pts", [])
            center = pts[0] if pts else None
            self.navigate_to_item.emit(v_name, s_idx, center)

    def _on_keep_visible_toggled(self, checked: bool):
        self.keep_visible_after_close = checked
        self.visibility_toggled.emit(checked)

    def _emit_style(self):
        style = {
            "width": self.pen_width,
            "is_dashed": self.is_dashed,
            "color": self.current_color,
            "end_cap": self.end_cap,
            "shape_type": self.shape_type
        }
        self.style_changed.emit(style)

    def _on_width_changed(self, w_id: int):
        self.pen_width = w_id
        self._emit_style()

    def _on_style_changed(self, s_id: int):
        self.is_dashed = (s_id == 1)
        self._emit_style()

    def _on_color_preset_clicked(self, c_id: int):
        if 0 <= c_id < len(self._preset_colors):
            self.current_color = self._preset_colors[c_id]
            self._emit_style()

    def _choose_custom_color(self):
        col = QColorDialog.getColor(self.current_color, self, "색상 선택")
        if col.isValid():
            self.current_color = col
            for b in self.bg_color.buttons():
                b.setChecked(False)
            self._emit_style()

    def _on_cap_changed(self, cap_id: int):
        if 0 <= cap_id < len(self._caps):
            self.end_cap = self._caps[cap_id]
            self._emit_style()

    def _on_shape_changed(self, sh_id: int):
        if 0 <= sh_id < len(self._shapes):
            self.shape_type = self._shapes[sh_id]
            self._emit_style()

    def set_shape_type(self, shape_type: str):
        if shape_type in self._shapes:
            idx = self._shapes.index(shape_type)
            btn = self.bg_shape.button(idx)
            if btn:
                btn.setChecked(True)
            self.shape_type = shape_type
            self._emit_style()

    def get_current_style(self) -> dict:
        return {
            "width": self.pen_width,
            "is_dashed": self.is_dashed,
            "color": self.current_color,
            "end_cap": self.end_cap,
            "shape_type": self.shape_type
        }

    def add_shape(self, shape_info: Dict[str, Any]):
        self.shapes.append(shape_info)
        row = self.table.rowCount()
        self.table.insertRow(row)

        name_map = {
            "select": "선택", "line": "선분", "arrow": "화살표",
            "rect": "사각형", "circle": "원", "freeform": "프리폼"
        }
        stype = shape_info.get("shape_type", "line")
        col = shape_info.get("color")
        color_str = col.name() if isinstance(col, QColor) else str(col or "#eb4b4b")

        s_idx = shape_info.get("slice_idx", 0)
        plane_no = s_idx + 1

        self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.table.setItem(row, 1, QTableWidgetItem(name_map.get(stype, stype)))
        
        col_item = QTableWidgetItem(color_str)
        col_item.setForeground(QBrush(QColor(color_str)))
        self.table.setItem(row, 2, col_item)

        self.table.setItem(row, 3, QTableWidgetItem(shape_info.get("view_name", "")))
        self.table.setItem(row, 4, QTableWidgetItem(str(plane_no)))
        self.table.selectRow(row)

    def to_dict(self) -> List[Dict[str, Any]]:
        res = []
        for s in self.shapes:
            item = dict(s)
            if isinstance(item.get("color"), QColor):
                item["color"] = item["color"].name()
            res.append(item)
        return res

    def from_dict(self, data_list: List[Dict[str, Any]]):
        self.shapes.clear()
        self.table.setRowCount(0)
        for d in data_list:
            item = dict(d)
            if isinstance(item.get("color"), str):
                item["color"] = QColor(item["color"])
            self.add_shape(item)

    def _delete_selected(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.shapes):
            self.shapes.pop(row)
            self.table.removeRow(row)
            self.item_deleted.emit(row)
            new_row = self.table.currentRow()
            self.item_selected.emit(new_row)

    def _clear_all(self):
        if not self.shapes:
            return
        reply = QMessageBox.question(
            self, "확인", "그려진 모든 도형을 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.shapes.clear()
            self.table.setRowCount(0)
            self.all_cleared.emit()
            self.item_selected.emit(-1)

    def showEvent(self, event):
        self.active_toggled.emit(True)
        # 창이 열리면 화면에 항상 보이도록
        self.visibility_toggled.emit(True)
        self._emit_style()
        super().showEvent(event)

    def closeEvent(self, event):
        self.active_toggled.emit(False)
        # 체크 해제 시 리스트는 유지하되 창 닫히면 2D 화면에서 도형 숨김
        if not self.keep_visible_after_close:
            self.visibility_toggled.emit(False)
        else:
            self.visibility_toggled.emit(True)
        super().closeEvent(event)
