"""
settings_manager.py
애플리케이션 UI 상태, 뷰어 설정(십자선, 자, 회전/대칭), 3D 뷰어 설정(투명도, 음영, 평면)을
영속적으로 저장하고 복원하는 설정 관리자.
"""

import os
import json
from typing import Any, Dict


DEFAULT_SETTINGS: Dict[str, Any] = {
    "mpr_views": {
        "Axial": {
            "show_crosshair": True,
            "show_ruler": True,
            "rotation": 0,
            "flip_h": False,
            "flip_v": False
        },
        "Coronal": {
            "show_crosshair": True,
            "show_ruler": True,
            "rotation": 0,
            "flip_h": False,
            "flip_v": False
        },
        "Sagittal": {
            "show_crosshair": True,
            "show_ruler": True,
            "rotation": 0,
            "flip_h": False,
            "flip_v": False
        }
    },
    "volume_3d": {
        "shading": True,
        "opacity": 1.0
    },
    "window_level": {
        "window": 1500.0,
        "level": 300.0,
        "preset": "Bone (골조직)"
    }
}


class SettingsManager:
    """
    JSON 기반 설정 저장 및 로드 매니저 싱글톤.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SettingsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_dir: str = None):
        if self._initialized:
            return
        if config_dir is None:
            config_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = os.path.join(config_dir, "user_settings.json")
        self.settings: Dict[str, Any] = {}
        self.load()
        self._initialized = True

    def load(self) -> Dict[str, Any]:
        """설정 파일에서 로드하거나 기본값을 병합합니다."""
        self.settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        if os.path.isfile(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self._deep_update(self.settings, loaded)
            except Exception as e:
                print(f"[SettingsManager] 설정 파일 읽기 오류: {e}")
        return self.settings

    def save(self):
        """현재 설정을 파일에 저장합니다."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SettingsManager] 설정 파일 저장 오류: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """점(.) 구분 경로로 설정값을 조회합니다. 예: 'mpr_views.Axial.show_crosshair'"""
        keys = key_path.split(".")
        val = self.settings
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key_path: str, value: Any, auto_save: bool = True):
        """점(.) 구분 경로로 설정값을 지정합니다."""
        keys = key_path.split(".")
        curr = self.settings
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = value
        if auto_save:
            self.save()

    def _deep_update(self, base: dict, update: dict):
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v
