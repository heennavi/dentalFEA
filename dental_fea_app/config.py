"""
config.py
Dental FEA 애플리케이션의 전역 설정, 상수, 윈도우/레벨 프리셋, 단위계 정의.
"""

from dataclasses import dataclass
from typing import Tuple


# --- 축 방향 정의 ---
AXIS_SAGITTAL = 0  # X-cut (Sagittal plane: YZ)
AXIS_CORONAL = 1   # Y-cut (Coronal plane: XZ)
AXIS_AXIAL = 2     # Z-cut (Axial plane: XY)

AXIS_NAMES = {
    AXIS_SAGITTAL: "Sagittal",
    AXIS_CORONAL: "Coronal",
    AXIS_AXIAL: "Axial"
}

# --- 뷰어 색상 체계 (의료 표준 컬러) ---
COLOR_SAGITTAL: Tuple[float, float, float] = (1.0, 0.2, 0.2)  # Red
COLOR_CORONAL: Tuple[float, float, float] = (0.2, 0.9, 0.2)   # Green
COLOR_AXIAL: Tuple[float, float, float] = (0.2, 0.6, 1.0)     # Blue

# --- Window / Level 프리셋 (치과 및 두경부 CT 기준) ---
@dataclass(frozen=True)
class WindowLevelPreset:
    name: str
    window: float
    level: float

WL_PRESETS = {
    "Bone (골조직)": WindowLevelPreset("Bone", window=1500.0, level=300.0),
    "Dental / Enamel (치아/에나멜)": WindowLevelPreset("Dental", window=3000.0, level=1000.0),
    "Mandible Cortical (하악 피질골)": WindowLevelPreset("Mandible Cortical", window=2000.0, level=800.0),
    "Soft Tissue (연조직)": WindowLevelPreset("Soft Tissue", window=400.0, level=40.0),
    "Full Dynamic Range (전체 범위)": WindowLevelPreset("Full Range", window=4000.0, level=1000.0),
}

DEFAULT_PRESET_NAME = "Bone (골조직)"

# --- 물리 단위계 (FEA 해석 표준) ---
# 길이: mm
# 힘: N
# 응력/탄성계수: MPa (N/mm^2)
# 밀도: tonne/mm^3 (g/cm^3 * 1e-9)
UNIT_LENGTH = "mm"
UNIT_FORCE = "N"
UNIT_STRESS = "MPa"

# UI 테마
DARK_THEME_QSS = """
QMainWindow, QDialog {
    background-color: #1e1e1e;
    color: #f0f0f0;
}
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Noto Sans KR', sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 6px;
    margin-top: 10px;
    font-weight: bold;
    color: #61afef;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #2b2b2b;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #383838;
    border-color: #61afef;
}
QPushButton:pressed {
    background-color: #1f5080;
}
QPushButton:disabled {
    background-color: #1a1a1a;
    border-color: #333333;
    color: #666666;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #252525;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ffffff;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #61afef;
}
QSlider::groove:horizontal {
    border: 1px solid #3c3c44;
    height: 6px;
    background: #252528;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #b0b0b8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e0e0e5;
    border: 1px solid #666670;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #9999aa;
}
QTabWidget::pane {
    border: 1px solid #3c3c3c;
    background-color: #232323;
}
QTabBar::tab {
    background: #2b2b2b;
    border: 1px solid #3c3c3c;
    padding: 8px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #1e1e1e;
    border-bottom-color: #1e1e1e;
    color: #61afef;
    font-weight: bold;
}
QSplitter::handle {
    background-color: #2f2f2f;
}
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: #444444;
    min-height: 20px;
    border-radius: 4px;
}
QMenuBar {
    background-color: #1e1e22;
    color: #e0e0e0;
    border-bottom: 1px solid #333338;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 8px;
    border-radius: 3px;
}
QMenuBar::item:selected, QMenuBar::item:hover {
    background-color: #2b3d56;
    color: #ffffff;
}
QMenu {
    background-color: #222228;
    color: #e0e0e0;
    border: 1px solid #3c3c46;
    padding: 4px 0px;
}
QMenu::item {
    padding: 6px 24px;
    background-color: transparent;
}
QMenu::item:selected {
    background-color: #1976d2;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background: #383844;
    margin: 4px 8px;
}
"""

