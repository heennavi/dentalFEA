"""
icons.py
텍스트 없는 세련된 벡터 그래픽 아이콘(QIcon)을 동적으로 렌더링하는 팩토리.
플랫폼(Windows, macOS, Linux)에 무관하게 항상 선명한 픽토그램을 제공합니다.
"""

from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QPainterPath, QBrush, QFont
from PySide6.QtCore import Qt, QPointF, QRectF


class IconFactory:
    """
    QPainter를 사용하여 일관된 다크 테마 벡터 아이콘을 생성하는 팩토리 클래스.
    """

    COLOR_DEFAULT = QColor(220, 220, 225)
    COLOR_ACCENT = QColor(97, 175, 239)

    @classmethod
    def _create_canvas(cls, size: int = 24) -> tuple[QPixmap, QPainter]:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        return pixmap, painter

    @classmethod
    def rotate_cw_90(cls, size: int = 24, color: QColor = None) -> QIcon:
        """시계방향 90도 회전 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        # 3/4 원호 (상단->우측->하단)
        rect = QRectF(4, 4, size - 8, size - 8)
        painter.drawArc(rect, 45 * 16, -270 * 16)

        # 화살표 머리
        arrow = QPainterPath()
        arrow.moveTo(size - 3, size / 2 - 2)
        arrow.lineTo(size - 4, size / 2 + 4)
        arrow.lineTo(size - 10, size / 2 + 2)
        painter.fillPath(arrow, QBrush(color))

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def rotate_ccw_90(cls, size: int = 24, color: QColor = None) -> QIcon:
        """반시계방향 90도 회전 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        # 3/4 원호 (상단->좌측->하단)
        rect = QRectF(4, 4, size - 8, size - 8)
        painter.drawArc(rect, 135 * 16, 270 * 16)

        # 화살표 머리
        arrow = QPainterPath()
        arrow.moveTo(3, size / 2 - 2)
        arrow.lineTo(4, size / 2 + 4)
        arrow.lineTo(10, size / 2 + 2)
        painter.fillPath(arrow, QBrush(color))

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def flip_h(cls, size: int = 24, color: QColor = None) -> QIcon:
        """좌우 대칭 아이콘 (세로 중심선 및 양방향 가로 화살표)"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)

        # 중앙 점선
        pen_dash = QPen(QColor(120, 120, 130), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen_dash)
        painter.drawLine(size / 2, 3, size / 2, size - 3)

        # 양방향 수평 화살표
        pen_arrow = QPen(color, 2)
        pen_arrow.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_arrow)
        cy = size / 2
        painter.drawLine(4, cy, size - 4, cy)

        # 좌측 화살촉
        painter.drawLine(4, cy, 8, cy - 3)
        painter.drawLine(4, cy, 8, cy + 3)
        # 우측 화살촉
        painter.drawLine(size - 4, cy, size - 8, cy - 3)
        painter.drawLine(size - 4, cy, size - 8, cy + 3)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def flip_v(cls, size: int = 24, color: QColor = None) -> QIcon:
        """상하 대칭 아이콘 (가로 중심선 및 양방향 세로 화살표)"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)

        # 중앙 수평 점선
        pen_dash = QPen(QColor(120, 120, 130), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(pen_dash)
        painter.drawLine(3, size / 2, size - 3, size / 2)

        # 양방향 세로 화살표
        pen_arrow = QPen(color, 2)
        pen_arrow.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_arrow)
        cx = size / 2
        painter.drawLine(cx, 4, cx, size - 4)

        # 상단 화살촉
        painter.drawLine(cx, 4, cx - 3, 8)
        painter.drawLine(cx, 4, cx + 3, 8)
        # 하단 화살촉
        painter.drawLine(cx, size - 4, cx - 3, size - 8)
        painter.drawLine(cx, size - 4, cx + 3, size - 8)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def ruler(cls, size: int = 24, color: QColor = None) -> QIcon:
        """눈금자 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        painter.setPen(QPen(color, 1.8))

        # 대각선 자 바디
        painter.save()
        painter.translate(size / 2, size / 2)
        painter.rotate(45)

        rw, rh = size - 6, 8
        painter.drawRect(int(-rw / 2), int(-rh / 2), rw, rh)

        # 눈금들
        painter.setPen(QPen(color, 1.2))
        for x in range(int(-rw / 2) + 3, int(rw / 2) - 1, 3):
            painter.drawLine(x, int(-rh / 2), x, int(-rh / 2) + 3)

        painter.restore()
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def crosshair(cls, size: int = 24, color: QColor = None) -> QIcon:
        """십자선 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        painter.setPen(QPen(color, 1.8))

        c = size / 2
        # 원
        painter.drawEllipse(QPointF(c, c), 6, 6)
        # 십자선
        painter.drawLine(c, 3, c, size - 3)
        painter.drawLine(3, c, size - 3, c)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def settings(cls, size: int = 24, color: QColor = None) -> QIcon:
        """톱니바퀴 ⚙ 설정 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        painter.setPen(QPen(color, 2))

        c = size / 2
        painter.drawEllipse(QPointF(c, c), 4, 4)

        # 6개 이빨
        pen_teeth = QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_teeth)
        import math
        for i in range(6):
            angle = i * (math.pi / 3)
            x1 = c + 5.5 * math.cos(angle)
            y1 = c + 5.5 * math.sin(angle)
            x2 = c + 8.5 * math.cos(angle)
            y2 = c + 8.5 * math.sin(angle)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def reset_camera(cls, size: int = 24, color: QColor = None) -> QIcon:
        """카메라 원점 리셋 아이콘 (타겟/포커스 뷰)"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        painter.setPen(QPen(color, 2))

        # 4개의 코너 브래킷
        m = 4
        d = 5
        # 좌상단
        painter.drawLine(m, m, m + d, m)
        painter.drawLine(m, m, m, m + d)
        # 우상단
        painter.drawLine(size - m, m, size - m - d, m)
        painter.drawLine(size - m, m, size - m, m + d)
        # 좌하단
        painter.drawLine(m, size - m, m + d, size - m)
        painter.drawLine(m, size - m, m, size - m - d)
        # 우하단
        painter.drawLine(size - m, size - m, size - m - d, size - m)
        painter.drawLine(size - m, size - m, size - m, size - m - d)

        # 중앙 점
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(size / 2, size / 2), 2, 2)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def view_cube(cls, highlight_face: str, size: int = 24) -> QIcon:
        """
        3D 큐브 시점 아이콘.
        highlight_face: 'front', 'rear', 'top', 'bottom', 'left', 'right', 'iso'
        """
        pixmap, painter = cls._create_canvas(size)
        c = size / 2

        # 큐브 기본 외곽선
        pen_outline = QPen(QColor(150, 150, 165), 1.2)
        painter.setPen(pen_outline)

        # 큐브 와이어프레임 또는 면
        brush_normal = QBrush(QColor(50, 50, 60))
        brush_hl = QBrush(cls.COLOR_ACCENT)

        if highlight_face == "iso":
            # 아이소메트릭 큐브 3면
            p_top = QPainterPath()
            p_top.moveTo(c, 3)
            p_top.lineTo(size - 4, 8)
            p_top.lineTo(c, 13)
            p_top.lineTo(4, 8)
            p_top.closeSubpath()
            painter.fillPath(p_top, brush_hl)
            painter.drawPath(p_top)

            p_left = QPainterPath()
            p_left.moveTo(4, 8)
            p_left.lineTo(c, 13)
            p_left.lineTo(c, size - 3)
            p_left.lineTo(4, size - 8)
            p_left.closeSubpath()
            painter.fillPath(p_left, QBrush(QColor(80, 140, 200)))
            painter.drawPath(p_left)

            p_right = QPainterPath()
            p_right.moveTo(c, 13)
            p_right.lineTo(size - 4, 8)
            p_right.lineTo(size - 4, size - 8)
            p_right.lineTo(c, size - 3)
            p_right.closeSubpath()
            painter.fillPath(p_right, QBrush(QColor(60, 110, 170)))
            painter.drawPath(p_right)

        else:
            # 2D 큐브 투영 사각 박스
            pad = 5
            rect = QRectF(pad, pad, size - pad * 2, size - pad * 2)
            painter.fillRect(rect, brush_normal)
            painter.drawRect(rect)

            # 강조 영역
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(brush_hl)
            if highlight_face == "front":
                painter.drawRect(QRectF(pad + 2, pad + 2, size - (pad + 2) * 2, size - (pad + 2) * 2))
            elif highlight_face == "rear":
                painter.setPen(QPen(cls.COLOR_ACCENT, 2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(pad + 2, pad + 2, size - (pad + 2) * 2, size - (pad + 2) * 2))
            elif highlight_face == "top":
                painter.drawRect(QRectF(pad, pad, size - pad * 2, 4))
            elif highlight_face == "bottom":
                painter.drawRect(QRectF(pad, size - pad - 4, size - pad * 2, 4))
            elif highlight_face == "left":
                painter.drawRect(QRectF(pad, pad, 4, size - pad * 2))
            elif highlight_face == "right":
                painter.drawRect(QRectF(size - pad - 4, pad, 4, size - pad * 2))

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def letter_icon(cls, letter: str, size: int = 24, bg_color: QColor = None, text_color: QColor = None) -> QIcon:
        """미니멀 글자 심볼 아이콘 (A, C, S, O 등)"""
        pixmap, painter = cls._create_canvas(size)
        bg_color = bg_color or QColor(44, 44, 52)
        text_color = text_color or cls.COLOR_DEFAULT

        # 둥근 사각 배경
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(QColor(70, 70, 85), 1))
        painter.drawRoundedRect(1, 1, size - 2, size - 2, 4, 4)

        # 글자 렌더링
        painter.setPen(text_color)
        font = QFont("sans-serif", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, letter)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def folder(cls, size: int = 24, color: QColor = None) -> QIcon:
        """폴더 열기 벡터 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 1.8)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        # 폴더 탭 + 바디
        path = QPainterPath()
        path.moveTo(3, 7)
        path.lineTo(9, 7)
        path.lineTo(12, 10)
        path.lineTo(size - 3, 10)
        path.lineTo(size - 3, size - 5)
        path.lineTo(3, size - 5)
        path.closeSubpath()
        painter.drawPath(path)

        # 폴더 앞판 덮개 (살짝 열린 효과)
        front = QPainterPath()
        front.moveTo(3, 12)
        front.lineTo(size - 5, 12)
        front.lineTo(size - 3, size - 5)
        front.lineTo(5, size - 5)
        front.closeSubpath()
        painter.fillPath(front, QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        painter.drawPath(front)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def file(cls, size: int = 24, color: QColor = None) -> QIcon:
        """단일 파일 열기 벡터 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 1.8)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        # 문서 외곽선 (우상단 접힘)
        path = QPainterPath()
        path.moveTo(5, 4)
        path.lineTo(size - 9, 4)
        path.lineTo(size - 4, 9)
        path.lineTo(size - 4, size - 4)
        path.lineTo(5, size - 4)
        path.closeSubpath()
        painter.drawPath(path)

        # 접힌 모서리 선
        corner = QPainterPath()
        corner.moveTo(size - 9, 4)
        corner.lineTo(size - 9, 9)
        corner.lineTo(size - 4, 9)
        painter.drawPath(corner)

        # 문서 내부 텍스트 라인 2줄
        pen_line = QPen(QColor(color.red(), color.green(), color.blue(), 180), 1.4)
        painter.setPen(pen_line)
        painter.drawLine(8, 14, size - 8, 14)
        painter.drawLine(8, 18, size - 11, 18)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def cube_3d(cls, size: int = 24, color: QColor = None) -> QIcon:
        """3D 볼륨 뷰어 벡터 아이콘"""
        color = color or cls.COLOR_ACCENT
        pixmap, painter = cls._create_canvas(size)
        c = size / 2

        # 3차원 투시 큐브 (입체감 있는 메쉬 느낌)
        p_top = QPainterPath()
        p_top.moveTo(c, 3)
        p_top.lineTo(size - 4, 8)
        p_top.lineTo(c, 13)
        p_top.lineTo(4, 8)
        p_top.closeSubpath()
        painter.fillPath(p_top, QBrush(color))
        painter.setPen(QPen(QColor(230, 240, 255), 1.2))
        painter.drawPath(p_top)

        p_left = QPainterPath()
        p_left.moveTo(4, 8)
        p_left.lineTo(c, 13)
        p_left.lineTo(c, size - 3)
        p_left.lineTo(4, size - 8)
        p_left.closeSubpath()
        painter.fillPath(p_left, QBrush(QColor(50, 110, 175)))
        painter.drawPath(p_left)

        p_right = QPainterPath()
        p_right.moveTo(c, 13)
        p_right.lineTo(size - 4, 8)
        p_right.lineTo(size - 4, size - 8)
        p_right.lineTo(c, size - 3)
        p_right.closeSubpath()
        painter.fillPath(p_right, QBrush(QColor(35, 80, 140)))
        painter.drawPath(p_right)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def triangle_ruler(cls, size: int = 24, color: QColor = None) -> QIcon:
        """가운데 삼각형 구멍이 뚫린 직각 삼각자 아이콘 (단면 계측용)"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        
        # 외곽 직각삼각형 (좌하단 직각)
        # (3, 3) -> (3, size-3) -> (size-3, size-3)
        outer = QPainterPath()
        outer.moveTo(4, 4)
        outer.lineTo(4, size - 4)
        outer.lineTo(size - 4, size - 4)
        outer.closeSubpath()

        # 내부 삼각형 구멍
        inner = QPainterPath()
        inner.moveTo(8, 12)
        inner.lineTo(8, size - 8)
        inner.lineTo(size - 12, size - 8)
        inner.closeSubpath()

        # 바디 = 외곽 - 내부
        body = outer.subtracted(inner)
        painter.fillPath(body, QBrush(QColor(color.red(), color.green(), color.blue(), 180)))
        painter.setPen(QPen(color, 1.4))
        painter.drawPath(outer)
        painter.drawPath(inner)

        # 빗변 및 밑변 눈금 표시
        painter.setPen(QPen(QColor(30, 30, 35), 1.2))
        for x in range(7, size - 6, 3):
            painter.drawLine(x, size - 4, x, size - 6)
        for y in range(7, size - 6, 3):
            painter.drawLine(4, y, 6, y)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def note(cls, size: int = 24, color: QColor = None) -> QIcon:
        """주석 작성 (노트 및 펜) 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)

        # 노트 패드 사각형
        painter.setPen(QPen(color, 1.6))
        painter.drawRoundedRect(QRectF(4, 3, size - 10, size - 6), 2, 2)

        # 노트 줄 3개
        pen_line = QPen(QColor(color.red(), color.green(), color.blue(), 160), 1.3)
        painter.setPen(pen_line)
        painter.drawLine(7, 7, size - 9, 7)
        painter.drawLine(7, 11, size - 9, 11)
        painter.drawLine(7, 15, size - 12, 15)

        # 연필/펜 (우하단 대각선)
        pen_pencil = QPen(cls.COLOR_ACCENT, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_pencil)
        painter.drawLine(size - 3, size - 9, size - 8, size - 4)
        painter.drawLine(size - 8, size - 4, size - 10, size - 4)

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def palette(cls, size: int = 24, color: QColor = None) -> QIcon:
        """그리기 도구 (아트 파레트) 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)

        painter.setPen(QPen(color, 1.6))
        # 파레트 외형 타원
        painter.drawEllipse(QRectF(3, 4, size - 6, size - 8))

        # 파레트 엄지 구멍
        painter.setBrush(QBrush(QColor(30, 30, 35)))
        painter.drawEllipse(QRectF(size - 9, size / 2 - 1, 4, 4))

        # 물감 방울 3개
        colors = [QColor(235, 87, 87), QColor(242, 201, 76), QColor(97, 175, 239)]
        pts = [(7, 9), (12, 7), (8, 14)]
        for pt, c in zip(pts, colors):
            painter.setBrush(QBrush(c))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(pt[0], pt[1], 3.5, 3.5))

        painter.end()
        return QIcon(pixmap)

    # --- 그리기 형태 아이콘 (선분, 화살표, 사각형, 원, 프리폼, 선택 취소) ---
    @classmethod
    def draw_select(cls, size: int = 20, color: QColor = None) -> QIcon:
        """선택 커서 (포인터 화살표) 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        path = QPainterPath()
        path.moveTo(4, 3)
        path.lineTo(4, size - 4)
        path.lineTo(size / 2 - 1, size / 2 + 1)
        path.lineTo(size - 4, size - 3)
        path.closeSubpath()
        painter.fillPath(path, QBrush(color))
        painter.setPen(QPen(QColor(20, 20, 25), 1))
        painter.drawPath(path)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def draw_line(cls, size: int = 20, color: QColor = None) -> QIcon:
        """직선 선분 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(3, size - 4, size - 3, 4)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def draw_arrow(cls, size: int = 20, color: QColor = None) -> QIcon:
        """화살표 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(3, size - 4, size - 4, 4)
        # 화살촉
        painter.drawLine(size - 4, 4, size - 10, 4)
        painter.drawLine(size - 4, 4, size - 4, 10)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def draw_rect(cls, size: int = 20, color: QColor = None) -> QIcon:
        """사각형 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        painter.setPen(QPen(color, 1.8))
        painter.drawRect(QRectF(3, 4, size - 6, size - 8))
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def draw_circle(cls, size: int = 20, color: QColor = None) -> QIcon:
        """원 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        painter.setPen(QPen(color, 1.8))
        painter.drawEllipse(QRectF(3, 3, size - 6, size - 6))
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def draw_freeform(cls, size: int = 20, color: QColor = None) -> QIcon:
        """프리폼 (곡선) 아이콘"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        path = QPainterPath()
        path.moveTo(3, size - 5)
        path.cubicTo(6, 4, size / 2, size - 4, size - 3, 4)
        painter.drawPath(path)
        painter.end()
        return QIcon(pixmap)

    # --- 선분 끝 형태 아이콘 (없음, 한쪽 화살표, 양쪽 화살표, 원형, 양쪽 원형) ---
    @classmethod
    def cap_none(cls, size: int = 20, color: QColor = None) -> QIcon:
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLine(3, size / 2, size - 3, size / 2)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def cap_single_arrow(cls, size: int = 20, color: QColor = None) -> QIcon:
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        cy = size / 2
        painter.drawLine(3, cy, size - 3, cy)
        painter.drawLine(size - 3, cy, size - 7, cy - 3.5)
        painter.drawLine(size - 3, cy, size - 7, cy + 3.5)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def cap_double_arrow(cls, size: int = 20, color: QColor = None) -> QIcon:
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        cy = size / 2
        painter.drawLine(3, cy, size - 3, cy)
        # 좌측 촉
        painter.drawLine(3, cy, 7, cy - 3.5)
        painter.drawLine(3, cy, 7, cy + 3.5)
        # 우측 촉
        painter.drawLine(size - 3, cy, size - 7, cy - 3.5)
        painter.drawLine(size - 3, cy, size - 7, cy + 3.5)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def cap_single_circle(cls, size: int = 20, color: QColor = None) -> QIcon:
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2)
        painter.setPen(pen)
        cy = size / 2
        painter.drawLine(3, cy, size - 6, cy)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(size - 4, cy), 2.5, 2.5)
        painter.end()
        return QIcon(pixmap)

    @classmethod
    def cap_double_circle(cls, size: int = 20, color: QColor = None) -> QIcon:
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2)
        painter.setPen(pen)
        cy = size / 2
        painter.drawLine(6, cy, size - 6, cy)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(4, cy), 2.5, 2.5)
        painter.drawEllipse(QPointF(size - 4, cy), 2.5, 2.5)
        painter.end()
        return QIcon(pixmap)


