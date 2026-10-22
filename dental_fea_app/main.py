"""
main.py
Dental FEA 애플리케이션 진입점.
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# 프로젝트 루트를 Python Path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Qt 6.5+ XCB 플러그인용 libxcb-cursor 자동 로드
import ctypes
local_xcb_cursor = os.path.join(project_root, ".local_libs", "usr", "lib", "x86_64-linux-gnu", "libxcb-cursor.so.0")
if os.path.exists(local_xcb_cursor):
    try:
        ctypes.CDLL(local_xcb_cursor)
    except Exception:
        pass

# NVIDIA 드라이버 버전 불일치 또는 GLX 에러 방어: 안전한 OpenGL 폴백
try:
    import subprocess
    proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=1)
    if proc.returncode != 0 and "version mismatch" in (proc.stderr + proc.stdout):
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "mesa"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
except Exception:
    pass

# Linux 환경에서 fcitx5 데몬 실행 시 PySide6 내장 IBus 플러그인을 통해 한글 IME 및 한/영 전환 연동
if sys.platform.startswith("linux"):
    im_mod = os.environ.get("QT_IM_MODULE", "")
    if im_mod in ("fcitx", "fcitx5") or not im_mod:
        os.environ["QT_IM_MODULE"] = "ibus"

from dental_fea_app.config import DARK_THEME_QSS
from dental_fea_app.ui.main_window import MainWindow
from dental_fea_app.imaging.dicom_reader import DicomReader
from dental_fea_app.ui.icons import IconFactory


def main():
    # PySide6 고해상도 지원 및 애플리케이션 초기화
    app = QApplication(sys.argv)
    app.setApplicationName("Dental FEA")
    app.setOrganizationName("DentalBiomech")
    app.setWindowIcon(IconFactory.app_icon())

    # 다크 테마 적용
    app.setStyleSheet(DARK_THEME_QSS)

    window = MainWindow()
    window.show()

    # 명령행 인자로 DICOM/영상 파일이나 폴더가 전달된 경우 자동 로드
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        if os.path.exists(target_path):
            try:
                vtk_img, meta = DicomReader.load_path(target_path)
                window.state_mgr.set_volume_data(vtk_img, meta)
            except Exception as e:
                print(f"[Dental FEA] 영상 자동 로드 실패: {e}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
