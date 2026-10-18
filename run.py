"""
run.py
Dental FEA 애플리케이션 편리 실행 스크립트.
"""

import sys
import os
import ctypes

# 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Qt 6.5+ XCB 플러그인용 libxcb-cursor 자동 로드
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

from dental_fea_app.main import main

if __name__ == "__main__":
    main()
