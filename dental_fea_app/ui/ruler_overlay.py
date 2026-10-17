"""
ruler_overlay.py
VTK 뷰어 위에 투명 오버레이 레이어로 배치되어
CT 영상의 실제 물리적 길이(mm)를 표시하는 고성능 2D 눈금자(Ruler) 위젯.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtCore import Qt


class RulerOverlay(QWidget):
    """
    VTK Interactor 위에 겹쳐지는 투명 눈금자 레이어.
    마우스 이벤트를 완전히 투과(TransparentForMouseEvents)시켜 성능 저하가 없습니다.
    """

    def __init__(self, mpr_widget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.mpr = mpr_widget
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._visible: bool = True

        # 폰트 설정
        self._font = QFont("monospace", 8)
        self._font.setStyleHint(QFont.StyleHint.Monospace)

    def set_ruler_visible(self, visible: bool):
        self._visible = visible
        self.setVisible(visible)
        self.update()

    def is_ruler_visible(self) -> bool:
        return self._visible

    def paintEvent(self, event):
        if not self._visible or self.mpr.state_mgr.vtk_image is None:
            return

        cam = self.mpr.camera
        if cam is None:
            return

        w = self.width()
        h = self.height()
        if w < 50 or h < 50:
            return

        # 병렬 투영 스케일로부터 픽셀당 mm 계산
        # ParallelScale = 화면 높이 절반에 해당하는 월드 mm
        parallel_scale = cam.GetParallelScale()
        if parallel_scale <= 0:
            return

        world_height_mm = 2.0 * parallel_scale
        pixels_per_mm = h / world_height_mm
        mm_per_pixel = world_height_mm / h

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setFont(self._font)

        # 색상 설정 (시안/화이트 톤)
        ruler_bg = QColor(20, 20, 24, 160)
        ruler_line_col = QColor(200, 200, 210, 220)
        ruler_text_col = QColor(240, 240, 240, 240)
        pen_thick = QPen(ruler_line_col, 1)
        pen_thin = QPen(QColor(160, 160, 170, 180), 1)

        # 1. 하단 가로 자 (Horizontal Ruler)
        h_ruler_height = 20
        h_y_base = h - h_ruler_height
        # 반투명 배경
        painter.fillRect(0, h_y_base, w - 24, h_ruler_height, ruler_bg)

        # 자 눈금 단위 자동 결정 (10mm 또는 5mm)
        step_mm = 10.0
        if pixels_per_mm < 1.5:
            step_mm = 20.0
        elif pixels_per_mm > 8.0:
            step_mm = 5.0

        # 중심점 기준 눈금 렌더링
        cx = w / 2.0
        painter.setPen(pen_thick)
        painter.drawLine(0, h_y_base, w - 24, h_y_base)

        # 오른쪽/왼쪽으로 눈금 그리기
        max_dist_mm = (w / 2.0) / pixels_per_mm
        num_ticks = int(max_dist_mm / step_mm) + 1

        for i in range(-num_ticks, num_ticks + 1):
            cur_mm = i * step_mm
            px = int(cx + cur_mm * pixels_per_mm)
            if 0 <= px <= (w - 30):
                # 큰 눈금
                painter.setPen(pen_thick)
                painter.drawLine(px, h_y_base, px, h_y_base + 8)
                # 라벨 표시 (mm 숫자)
                painter.setPen(ruler_text_col)
                label_txt = f"{abs(int(cur_mm))}"
                painter.drawText(px - 15, h_y_base + 9, 30, 10, Qt.AlignmentFlag.AlignCenter, label_txt)

                # 중간 눈금 (5mm 또는 step/2)
                mid_px = int(cx + (cur_mm + step_mm / 2.0) * pixels_per_mm)
                if 0 <= mid_px <= (w - 30):
                    painter.setPen(pen_thin)
                    painter.drawLine(mid_px, h_y_base, mid_px, h_y_base + 4)

        # 2. 우측 세로 자 (Vertical Ruler)
        v_ruler_width = 24
        v_x_base = w - v_ruler_width
        painter.fillRect(v_x_base, 0, v_ruler_width, h - h_ruler_height, ruler_bg)

        painter.setPen(pen_thick)
        painter.drawLine(v_x_base, 0, v_x_base, h - h_ruler_height)

        cy = (h - h_ruler_height) / 2.0
        v_max_dist_mm = cy / pixels_per_mm
        v_num_ticks = int(v_max_dist_mm / step_mm) + 1

        for i in range(-v_num_ticks, v_num_ticks + 1):
            cur_mm = i * step_mm
            py = int(cy + cur_mm * pixels_per_mm)
            if 0 <= py <= (h - h_ruler_height):
                painter.setPen(pen_thick)
                painter.drawLine(v_x_base, py, v_x_base + 8, py)
                painter.setPen(ruler_text_col)
                v_label = f"{abs(int(cur_mm))}"
                painter.drawText(v_x_base + 9, py - 6, 15, 12, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, v_label)

                mid_py = int(cy + (cur_mm + step_mm / 2.0) * pixels_per_mm)
                if 0 <= mid_py <= (h - h_ruler_height):
                    painter.setPen(pen_thin)
                    painter.drawLine(v_x_base, mid_py, v_x_base + 4, mid_py)

        # 단위 라벨 (우측 하단 코너)
        painter.fillRect(w - 24, h - 20, 24, 20, QColor(30, 30, 35, 220))
        painter.setPen(QColor(97, 175, 239))
        painter.drawText(w - 24, h - 20, 24, 20, Qt.AlignmentFlag.AlignCenter, "mm")

        painter.end()
