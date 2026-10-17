"""
events.py
Dental FEA 애플리케이션의 이벤트 시그널 정의.
"""

from PySide6.QtCore import QObject, Signal


class AppEvents(QObject):
    """
    애플리케이션 전역에서 발생하는 주요 이벤트 버스.
    """
    # 3차원 월드 좌표 변경: (x, y, z) in mm
    coordinates_changed = Signal(float, float, float)

    # 윈도우/레벨 변경: (window, level)
    window_level_changed = Signal(float, float)

    # 영상 데이터 로드 완료: vtkImageData 인스턴스
    volume_loaded = Signal(object)

    # 상태 표시줄 메시지: (message, timeout_ms)
    status_message = Signal(str, int)

    # 3D 볼륨 뷰어 토글 요청
    toggle_3d_viewer = Signal()
