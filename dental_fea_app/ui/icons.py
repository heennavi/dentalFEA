"""
icons.py
텍스트 없는 세련된 벡터 그래픽 아이콘(QIcon)을 동적으로 렌더링하는 팩토리.
플랫폼(Windows, macOS, Linux)에 무관하게 항상 선명한 픽토그램을 제공합니다.
"""

from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QPainterPath, QBrush, QFont, QPolygonF
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
    def reset_view(cls, size: int = 24, color: QColor = None) -> QIcon:
        """2D 뷰 초기화 아이콘 (회전 및 대칭 리셋 - 원형 복원 화살표)"""
        color = color or cls.COLOR_DEFAULT
        pixmap, painter = cls._create_canvas(size)
        painter.setPen(QPen(color, 2))

        # 3/4 원호 (반시계 또는 시계방향 리셋 화살표)
        c = size / 2.0
        r = size * 0.35
        rect = QRectF(c - r, c - r, 2 * r, 2 * r)
        # 45도부터 270도 회전 원호
        painter.drawArc(rect, int(45 * 16), int(270 * 16))

        # 끝부분 화살표 머리
        # 45도 지점: (c + r*cos(45), c - r*sin(45))
        import math
        ang = math.radians(45)
        ax = c + r * math.cos(ang)
        ay = c - r * math.sin(ang)

        arrow = QPolygonF([
            QPointF(ax - 1, ay - 4),
            QPointF(ax + 4, ay - 1),
            QPointF(ax, ay + 3)
        ])
        painter.setBrush(QBrush(color))
        painter.drawPolygon(arrow)

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

    @classmethod
    def implant(cls, size: int = 22, color: QColor = None) -> QIcon:
        """치과 임플란트 픽스처 벡터 아이콘 (칼라, 테이퍼 바디, 에이펙스, 나사선)"""
        color = color or cls.COLOR_ACCENT
        pixmap, painter = cls._create_canvas(size)

        cx = size / 2.0
        # 1. 상단 칼라(Collar / Platform)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(cx - size * 0.24, size * 0.14, size * 0.48, size * 0.12), 1.0, 1.0)

        # 2. 메인 테이퍼 바디 (Tapered Body)
        from PySide6.QtGui import QPolygonF
        body = QPolygonF([
            QPointF(cx - size * 0.20, size * 0.26),
            QPointF(cx + size * 0.20, size * 0.26),
            QPointF(cx + size * 0.12, size * 0.76),
            QPointF(cx - size * 0.12, size * 0.76)
        ])
        painter.drawPolygon(body)

        # 3. 에이펙스 (둥근 하단 끝)
        apex = QPainterPath()
        apex.moveTo(cx - size * 0.12, size * 0.76)
        apex.quadTo(QPointF(cx, size * 0.88), QPointF(cx + size * 0.12, size * 0.76))
        painter.drawPath(apex)

        # 4. 나사선 그루브 (Screw Thread Grooves)
        groove_color = QColor("#1e222b")
        pen_groove = QPen(groove_color, max(1.2, size * 0.07), Qt.PenStyle.SolidLine)
        pen_groove.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_groove)
        painter.drawLine(QPointF(cx - size * 0.18, size * 0.38), QPointF(cx + size * 0.18, size * 0.42))
        painter.drawLine(QPointF(cx - size * 0.16, size * 0.51), QPointF(cx + size * 0.16, size * 0.55))
        painter.drawLine(QPointF(cx - size * 0.14, size * 0.64), QPointF(cx + size * 0.14, size * 0.68))

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def occlusal_load(cls, size: int = 22, color: QColor = None) -> QIcon:
        """교합 하중(타깃 및 아래 방향 하중 화살표) 벡터 아이콘"""
        color = color or QColor(235, 80, 80)
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        # 상단에서 아래로 향하는 화살표 축
        cx = size / 2.0
        painter.drawLine(QPointF(cx, 3), QPointF(cx, size - 6))

        # 화살촉 (아래 방향)
        arrow = QPainterPath()
        arrow.moveTo(cx, size - 3)
        arrow.lineTo(cx - 4.5, size - 8.5)
        arrow.lineTo(cx + 4.5, size - 8.5)
        arrow.closeSubpath()
        painter.fillPath(arrow, QBrush(color))

        # 상단 타깃 바
        painter.drawLine(QPointF(cx - 5.0, 3), QPointF(cx + 5.0, 3))

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def visibility_eye(cls, visible: bool = True, size: int = 20, color: QColor = None) -> QIcon:
        """가시성 보기/감추기 눈동자 아이콘"""
        color = color or (cls.COLOR_ACCENT if visible else QColor(130, 130, 140))
        pixmap, painter = cls._create_canvas(size)
        pen = QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        # 눈 윤곽 (상/하 호)
        cx, cy = size / 2.0, size / 2.0
        eye_path = QPainterPath()
        eye_path.moveTo(3, cy)
        eye_path.quadTo(QPointF(cx, cy - 5.5), QPointF(size - 3, cy))
        eye_path.quadTo(QPointF(cx, cy + 5.5), QPointF(3, cy))
        painter.drawPath(eye_path)

        # 눈동자 (원)
        if visible:
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(cx, cy), 2.8, 2.8)
        else:
            # 사선 긋기
            painter.drawLine(QPointF(4, size - 4), QPointF(size - 4, 4))

        painter.end()
        return QIcon(pixmap)

    @classmethod
    def app_icon(cls, size: int = 256) -> QIcon:
        """치과 임플란트 생체역학 및 FEA 유한요소해석을 상징하는 대표 애플리케이션 아이콘 생성"""
        import os
        from PySide6.QtGui import QLinearGradient

        icon = QIcon()

        # resources/app_icon.png 파일이 존재하면 기본 추가
        res_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "app_icon.png")
        if os.path.exists(res_path):
            icon.addFile(res_path)

        # 다중 해상도(16, 32, 48, 64, 128, 256) 벡터 직접 렌더링
        for s in (16, 32, 48, 64, 128, 256):
            pixmap = QPixmap(s, s)
            pixmap.fill(Qt.GlobalColor.transparent)
            p = QPainter(pixmap)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            scale = s / 256.0

            # 1. 다크 스퀴클 배경
            bg = QPainterPath()
            bg.addRoundedRect(QRectF(10 * scale, 10 * scale, 236 * scale, 236 * scale), 48 * scale, 48 * scale)
            grad = QLinearGradient(0, 0, 0, 256 * scale)
            grad.setColorAt(0.0, QColor(28, 34, 46))
            grad.setColorAt(1.0, QColor(14, 18, 26))
            p.fillPath(bg, QBrush(grad))
            p.setPen(QPen(QColor(70, 140, 240, 180), max(1.0, 3.0 * scale)))
            p.drawPath(bg)

            # 2. 치아 치관(Crown) 실루엣 + FEA 응력 그라데이션
            tooth = QPainterPath()
            tooth.moveTo(60 * scale, 110 * scale)
            tooth.cubicTo(55 * scale, 70 * scale, 85 * scale, 52 * scale, 110 * scale, 55 * scale)
            tooth.cubicTo(120 * scale, 56 * scale, 125 * scale, 62 * scale, 128 * scale, 62 * scale)
            tooth.cubicTo(131 * scale, 62 * scale, 136 * scale, 56 * scale, 146 * scale, 55 * scale)
            tooth.cubicTo(171 * scale, 52 * scale, 201 * scale, 70 * scale, 196 * scale, 110 * scale)
            tooth.cubicTo(192 * scale, 125 * scale, 180 * scale, 132 * scale, 160 * scale, 135 * scale)
            tooth.lineTo(96 * scale, 135 * scale)
            tooth.cubicTo(76 * scale, 132 * scale, 64 * scale, 125 * scale, 60 * scale, 110 * scale)
            tooth.closeSubpath()

            stress_grad = QLinearGradient(60 * scale, 55 * scale, 196 * scale, 135 * scale)
            stress_grad.setColorAt(0.0, QColor(235, 70, 70, 230))   # 빨강
            stress_grad.setColorAt(0.3, QColor(245, 175, 40, 220))  # 주황/노랑
            stress_grad.setColorAt(0.65, QColor(60, 200, 120, 210)) # 초록
            stress_grad.setColorAt(1.0, QColor(50, 140, 240, 220))  # 파랑
            p.fillPath(tooth, QBrush(stress_grad))
            p.setPen(QPen(QColor(255, 255, 255, 210), max(0.8, 2.0 * scale)))
            p.drawPath(tooth)

            # 메쉬 와이어프레임 라인 (32px 이상일 때)
            if s >= 32:
                mesh_pen = QPen(QColor(255, 255, 255, 100), max(0.6, 1.2 * scale))
                p.setPen(mesh_pen)
                pts = [
                    (60, 110), (85, 80), (110, 55), (128, 62), (146, 55), (171, 80), (196, 110),
                    (96, 110), (128, 95), (160, 110), (96, 135), (128, 135), (160, 135)
                ]
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                    (0, 7), (1, 7), (2, 8), (3, 8), (4, 8), (5, 9), (6, 9),
                    (7, 8), (8, 9), (7, 10), (8, 11), (9, 12), (10, 11), (11, 12)
                ]
                for i1, i2 in edges:
                    p.drawLine(QPointF(pts[i1][0] * scale, pts[i1][1] * scale),
                               QPointF(pts[i2][0] * scale, pts[i2][1] * scale))

            # 3. 임플란트 픽스처 (하단 지대주 및 나사산 구조)
            fixture = QPainterPath()
            fixture.moveTo(105 * scale, 135 * scale)
            fixture.lineTo(151 * scale, 135 * scale)
            fixture.lineTo(145 * scale, 150 * scale)
            fixture.lineTo(111 * scale, 150 * scale)
            fixture.closeSubpath()
            p.fillPath(fixture, QBrush(QColor(210, 220, 235)))

            thread_pts = [
                (111, 150), (145, 150),
                (148, 158), (142, 163),
                (145, 171), (139, 176),
                (142, 184), (136, 189),
                (139, 197), (133, 202),
                (134, 210), (122, 218), (122, 210),
                (123, 202), (117, 197),
                (120, 189), (114, 184),
                (117, 176), (111, 171),
                (114, 163), (108, 158)
            ]
            imp_grad = QLinearGradient(100 * scale, 135 * scale, 160 * scale, 220 * scale)
            imp_grad.setColorAt(0.0, QColor(190, 205, 225))
            imp_grad.setColorAt(0.5, QColor(130, 145, 165))
            imp_grad.setColorAt(1.0, QColor(85, 100, 120))

            f_path = QPainterPath()
            f_path.moveTo(thread_pts[0][0] * scale, thread_pts[0][1] * scale)
            for x, y in thread_pts[1:]:
                f_path.lineTo(x * scale, y * scale)
            f_path.closeSubpath()

            p.fillPath(f_path, QBrush(imp_grad))
            p.setPen(QPen(QColor(230, 240, 255, 200), max(0.8, 1.8 * scale)))
            p.drawPath(f_path)

            # 4. 상단 교합 하중 화살표
            arrow = QPainterPath()
            arrow.moveTo(128 * scale, 48 * scale)
            arrow.lineTo(121 * scale, 34 * scale)
            arrow.lineTo(125 * scale, 34 * scale)
            arrow.lineTo(125 * scale, 20 * scale)
            arrow.lineTo(131 * scale, 20 * scale)
            arrow.lineTo(131 * scale, 34 * scale)
            arrow.lineTo(135 * scale, 34 * scale)
            arrow.closeSubpath()

            arrow_grad = QLinearGradient(128 * scale, 20 * scale, 128 * scale, 48 * scale)
            arrow_grad.setColorAt(0.0, QColor(255, 120, 50))
            arrow_grad.setColorAt(1.0, QColor(255, 50, 50))
            p.fillPath(arrow, QBrush(arrow_grad))
            p.setPen(QPen(QColor(255, 230, 200), max(0.6, 1.2 * scale)))
            p.drawPath(arrow)

            p.end()
            icon.addPixmap(pixmap)

        return icon


