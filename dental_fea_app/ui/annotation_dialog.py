"""
annotation_dialog.py
주석 작성 다이얼로그.
사용자 첨부 이미지(media_1792079672912.png)의 레이아웃과 기능을 충실히 반영:
- [DICOM 정보 가져오기]
- 상단 툴바: [복사] [붙여넣기] [선택 지우기] [전체 지우기]
- 대형 텍스트 편집기 (QTextEdit)
- 하단: [저장 및 닫기] [취소]
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QMessageBox, QApplication
)
from PySide6.QtCore import Qt, Signal


class AnnotationDialog(QDialog):
    """
    영상 주석 및 판독 메모 작성 창
    """
    # 저장 완료 시그널: text
    saved = Signal(str)

    def __init__(self, parent=None, dicom_meta: dict = None, initial_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("주석 작성")
        self.resize(560, 420)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self.dicom_meta = dicom_meta or {}
        self.initial_text = initial_text

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 상단 툴바 행
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        self.btn_import_dicom = QPushButton("DICOM 정보 가져오기")
        self.btn_import_dicom.clicked.connect(self._import_dicom_info)

        self.btn_copy = QPushButton("복사")
        self.btn_copy.clicked.connect(self._copy_text)

        self.btn_paste = QPushButton("붙여넣기")
        self.btn_paste.clicked.connect(self._paste_text)

        self.btn_clear_sel = QPushButton("선택 지우기")
        self.btn_clear_sel.clicked.connect(self._clear_selected)

        self.btn_clear_all = QPushButton("전체 지우기")
        self.btn_clear_all.clicked.connect(self._clear_all)

        top_bar.addWidget(self.btn_import_dicom)
        top_bar.addWidget(self.btn_copy)
        top_bar.addWidget(self.btn_paste)
        top_bar.addWidget(self.btn_clear_sel)
        top_bar.addWidget(self.btn_clear_all)
        top_bar.addStretch()

        layout.addLayout(top_bar)

        # 본문 텍스트 에디터
        self.editor = QTextEdit()
        self.editor.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.editor.setPlainText(self.initial_text)
        self.editor.setPlaceholderText("환자 상태, 진단 소견, 분석 메모 등을 입력하세요...")
        self.editor.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a20;
                color: #ffffff;
                font-family: 'Noto Sans CKR', 'NanumGothic', 'Malgun Gothic', 'Apple SD Gothic Neo', 'Consolas', 'NanumGothicCoding', sans-serif, monospace;
                font-size: 13px;
                line-height: 1.5;
                border: 1px solid #33333e;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.editor)

        # 하단 버튼 바 (저장 및 닫기 / 취소)
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()

        self.btn_save = QPushButton("저장 및 닫기")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #2196f3;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
        """)
        self.btn_save.clicked.connect(self._on_save)

        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.clicked.connect(self.reject)

        bottom_bar.addWidget(self.btn_save)
        bottom_bar.addWidget(self.btn_cancel)
        layout.addLayout(bottom_bar)

        # 공통 스타일
        self.setStyleSheet("""
            QDialog {
                background-color: #18181c;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #2c2c34;
                color: #ffffff;
                border: 1px solid #3e3e4a;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3a3a46;
                border-color: #555566;
            }
        """)

    def update_dicom_meta(self, meta: dict):
        self.dicom_meta = meta or {}

    def _import_dicom_info(self):
        """DICOM 메타데이터를 서식화하여 에디터 커서 위치 또는 맨 위에 삽입"""
        if not self.dicom_meta:
            QMessageBox.information(self, "알림", "로드된 DICOM 메타데이터 정보가 없습니다.")
            return

        lines = [
            "================ [DICOM 정보] ================",
            f"환자 이름: {self.dicom_meta.get('PatientName', 'N/A')}",
            f"환자 ID:   {self.dicom_meta.get('PatientID', 'N/A')}",
            f"생년월일: {self.dicom_meta.get('PatientBirthDate', 'N/A')}",
            f"성별:     {self.dicom_meta.get('PatientSex', 'N/A')}",
            f"검사 일시: {self.dicom_meta.get('StudyDate', '')} {self.dicom_meta.get('StudyTime', '')}",
            f"장비 모델: {self.dicom_meta.get('Manufacturer', '')} {self.dicom_meta.get('ManufacturerModelName', '')}",
            f"볼륨 규격: {self.dicom_meta.get('Dimensions', 'N/A')} (Spacing: {self.dicom_meta.get('Spacing', 'N/A')})",
            "==============================================\n"
        ]
        info_text = "\n".join(lines)
        self.editor.insertPlainText(info_text)

    def _copy_text(self):
        self.editor.copy()

    def _paste_text(self):
        self.editor.paste()

    def _clear_selected(self):
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            cursor.removeSelectedText()

    def _clear_all(self):
        if not self.editor.toPlainText().strip():
            return
        reply = QMessageBox.question(
            self, "확인", "작성된 주석 내용을 모두 지우시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.editor.clear()

    def _on_save(self):
        text = self.editor.toPlainText()
        self.saved.emit(text)
        self.accept()

    def to_dict(self) -> dict:
        return {"text": self.editor.toPlainText()}

    def from_dict(self, data: dict):
        text = data.get("text", "")
        self.editor.setPlainText(text)
        self.saved.emit(text)

