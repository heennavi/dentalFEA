"""
conftest.py
테스트 도중 뜨는 확인 팝업창(QMessageBox.question)은 사용자 규칙에 따라 모두 No를 반환하도록 자동 모의(mock)합니다.
"""
import pytest
from PySide6.QtWidgets import QMessageBox

@pytest.fixture(autouse=True)
def auto_no_messagebox(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
