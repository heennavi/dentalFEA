import os
import json
from typing import Optional, List
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QStatusBar, QMenuBar, QMenu, QMessageBox, QToolBar,
    QToolButton, QFileDialog
)
from PySide6.QtCore import Qt, QSize, QDateTime
from PySide6.QtGui import QAction

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.core.settings_manager import SettingsManager
from dental_fea_app.imaging.dicom_reader import DicomReader
from dental_fea_app.ui.control_panel import ControlPanel
from dental_fea_app.ui.mpr_container import MPRContainer
from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow
from dental_fea_app.ui.image_info_dialog import ImageInfoDialog
from dental_fea_app.ui.icons import IconFactory


class MainWindow(QMainWindow):
    """
    Dental FEA 메인 윈도우.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Dental FEA - Biomechanics & Implant Planning System")
        self.resize(1400, 900)

        # 코어 상태 및 설정 관리자
        self.state_mgr = StateManager(self)
        self.settings_mgr = SettingsManager()

        # 독립 3D 볼륨 뷰어 창 인스턴스
        self.volume_window: Optional[VolumeViewerWindow] = None

        # UI 컴포넌트 생성
        self._init_ui()

        # 창 위치 및 크기 복원
        self._restore_window_geometry()

        # 상단 메인 툴바 생성 (순수 아이콘)
        self._init_main_toolbar()

        # 메뉴바 및 상태표시줄 구성
        self._init_menu_and_status()

    def _restore_window_geometry(self):
        geom = self.settings_mgr.get("main_window.geometry")
        if geom:
            try:
                from PySide6.QtCore import QByteArray
                self.restoreGeometry(QByteArray.fromHex(geom.encode("utf-8")))
            except Exception:
                pass


    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 메인 스플리터 (좌: 컨트롤 패널, 우: MPR 영역)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # 좌측 컨트롤 패널 (슬림 240px)
        self.control_panel = ControlPanel(
            state_manager=self.state_mgr,
            on_open_3d_viewer=self.open_3d_viewer,
            parent=self
        )
        self.splitter.addWidget(self.control_panel)

        # 우측 MPR 컨테이너 (Axial, Coronal, Sagittal, 3D 트리거)
        self.mpr_container = MPRContainer(
            state_manager=self.state_mgr,
            on_open_3d_callback=self.open_3d_viewer,
            parent=self
        )
        self.splitter.addWidget(self.mpr_container)

        self.splitter.setSizes([240, 1160])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)

        main_layout.addWidget(self.splitter)

        # 4분할 MPR 스플리터 분할 상태 복원
        self.mpr_container.restore_splitter_state(self.settings_mgr)

    def _init_main_toolbar(self):
        """상단 메인 툴바 (텍스트 없는 고해상도 벡터 아이콘 전용)"""
        toolbar = QToolBar("메인 툴바", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(22, 22))
        self.addToolBar(toolbar)

        btn_qss = """
            QToolButton {
                background-color: #2c2c34;
                border: 1px solid #444452;
                border-radius: 4px;
                padding: 4px;
                margin-right: 4px;
            }
            QToolButton:hover {
                background-color: #3e3e4a;
                border-color: #61afef;
            }
        """

        # 1. DICOM 폴더 열기 버튼
        self.btn_tb_folder = QToolButton(self)
        self.btn_tb_folder.setIcon(IconFactory.folder(22))
        self.btn_tb_folder.setToolTip("DICOM 폴더 열기 (시리즈 자동 탐색)")
        self.btn_tb_folder.setStyleSheet(btn_qss)
        self.btn_tb_folder.clicked.connect(self._load_dicom_folder)
        toolbar.addWidget(self.btn_tb_folder)

        # 2. 단일 영상 파일 열기 버튼
        self.btn_tb_file = QToolButton(self)
        self.btn_tb_file.setIcon(IconFactory.file(22))
        self.btn_tb_file.setToolTip("단일 영상 파일 열기 (.dcm, .mha, .nii)")
        self.btn_tb_file.setStyleSheet(btn_qss)
        self.btn_tb_file.clicked.connect(self._load_single_file)
        toolbar.addWidget(self.btn_tb_file)

        toolbar.addSeparator()

        # 3. 3D 볼륨 뷰어 열기 버튼
        self.btn_tb_3d = QToolButton(self)
        self.btn_tb_3d.setIcon(IconFactory.cube_3d(22))
        self.btn_tb_3d.setToolTip("독립 3D 볼륨 & 메쉬 뷰어 열기 (3D Window)")
        self.btn_tb_3d.setStyleSheet(btn_qss)
        self.btn_tb_3d.clicked.connect(self.open_3d_viewer)
        toolbar.addWidget(self.btn_tb_3d)

    def _init_menu_and_status(self):
        menubar = self.menuBar()

        # 1. 파일 메뉴
        self.menu_file = menubar.addMenu("파일(&F)")

        act_open_folder = QAction("DICOM 폴더 열기...", self)
        act_open_folder.setShortcut("Ctrl+O")
        act_open_folder.triggered.connect(self._load_dicom_folder)
        self.menu_file.addAction(act_open_folder)

        act_open_file = QAction("단일 영상 파일 열기...", self)
        act_open_file.setShortcut("Ctrl+Shift+O")
        act_open_file.triggered.connect(self._load_single_file)
        self.menu_file.addAction(act_open_file)

        # [요청 반영] 단일 영상파일 열기 바로 아래에 샘플 치과 CT 로드 배치
        act_open_sample = QAction("샘플 치과 CT 로드 (하악골)", self)
        act_open_sample.triggered.connect(self._load_sample_data)
        self.menu_file.addAction(act_open_sample)

        self.menu_file.addSeparator()

        # [요청 반영] 최근 파일/폴더 기록 (최대 5개) 서브메뉴
        self.menu_recent = self.menu_file.addMenu("최근 파일/폴더 열기(&R)")
        self._update_recent_menu()

        self.menu_file.addSeparator()

        act_save_user_info = QAction("사용자 정보 저장(&S)...", self)
        act_save_user_info.setShortcut("Ctrl+S")
        act_save_user_info.triggered.connect(self._save_user_info)
        self.menu_file.addAction(act_save_user_info)

        act_load_user_info = QAction("사용자 정보 로드(&L)...", self)
        act_load_user_info.setShortcut("Ctrl+L")
        act_load_user_info.triggered.connect(self._load_user_info)
        self.menu_file.addAction(act_load_user_info)

        self.menu_file.addSeparator()

        act_exit = QAction("종료(&X)", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        self.menu_file.addAction(act_exit)

        # 2. 뷰 메뉴
        menu_view = menubar.addMenu("뷰(&V)")

        act_open_3d = QAction("독립 3D 볼륨 뷰어 열기", self)
        act_open_3d.setShortcut("Ctrl+3")
        act_open_3d.triggered.connect(self.open_3d_viewer)
        menu_view.addAction(act_open_3d)

        act_reset_all_views = QAction("뷰 초기화(&I)", self)
        act_reset_all_views.setShortcut("Ctrl+Shift+R")
        act_reset_all_views.setToolTip("모든 뷰를 처음 로드한 상태의 이미지 보기(회전, 대칭, 슬라이스 위치, W/L)로 초기화합니다.")
        act_reset_all_views.triggered.connect(self.mpr_container.reset_all_views)
        menu_view.addAction(act_reset_all_views)

        act_reset_cam = QAction("모든 2D 뷰 카메라 리셋", self)
        act_reset_cam.setShortcut("Ctrl+R")
        act_reset_cam.triggered.connect(self.mpr_container.reset_all_cameras)
        menu_view.addAction(act_reset_cam)

        # 3. 도움말 메뉴
        menu_help = menubar.addMenu("도움말(&H)")

        # [요청 반영] 현재 영상정보 메뉴 및 팝업
        act_img_info = QAction("현재 영상정보(&I)...", self)
        act_img_info.triggered.connect(self._show_image_info)
        menu_help.addAction(act_img_info)

        act_about = QAction("정보(&A)", self)
        act_about.triggered.connect(self._show_about)
        menu_help.addAction(act_about)

        # 상태표시줄
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Dental FEA 준비 완료. 영상 데이터를 로드하십시오.")

        # 상태 관리자 신호 연결
        self.state_mgr.volume_loaded.connect(self._on_volume_loaded)

    def _update_recent_menu(self):
        """최근 파일/폴더 5개 메뉴 갱신"""
        self.menu_recent.clear()
        recent_list = self.settings_mgr.get("recent_files", [])
        if not recent_list:
            act_empty = QAction("(최근 파일 기록 없음)", self)
            act_empty.setEnabled(False)
            self.menu_recent.addAction(act_empty)
            return

        for path in recent_list[:5]:
            display_name = os.path.basename(path.rstrip("/\\")) or path
            act = QAction(f"{display_name} ({path})", self)
            act.triggered.connect(lambda _, p=path: self._load_path_direct(p))
            self.menu_recent.addAction(act)

    def _add_recent_path(self, path: str):
        recent_list = self.settings_mgr.get("recent_files", [])
        if path in recent_list:
            recent_list.remove(path)
        recent_list.insert(0, path)
        recent_list = recent_list[:5]
        self.settings_mgr.set("recent_files", recent_list, auto_save=True)
        self._update_recent_menu()

    def _load_dicom_folder(self):
        self.control_panel._load_dicom_folder()
        # 로드 성공 시 최근 목록에 반영
        if self.state_mgr.series_path:
            self._add_recent_path(self.state_mgr.series_path)

    def _load_single_file(self):
        self.control_panel._load_single_file()
        if self.state_mgr.series_path:
            self._add_recent_path(self.state_mgr.series_path)

    def _load_sample_data(self):
        self.control_panel._load_sample_data()
        if self.state_mgr.series_path:
            self._add_recent_path(self.state_mgr.series_path)

    def _load_path_direct(self, path: str):
        """최근 파일 목록에서 선택한 경로 직접 로드"""
        if not os.path.exists(path):
            QMessageBox.warning(self, "파일 열기 실패", f"지정한 경로가 존재하지 않습니다:\n{path}")
            return

        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QCursor
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()

        try:
            if os.path.isdir(path):
                vtk_img, meta = DicomReader.load_dicom_series(path)
            else:
                vtk_img, meta = DicomReader.load_single_file(path)
            self.state_mgr.set_volume_data(vtk_img, meta)
            self._add_recent_path(path)
        except Exception as e:
            QMessageBox.critical(self, "영상 로드 오류", f"영상을 읽는 중 오류가 발생했습니다:\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def _show_image_info(self):
        """현재 영상 정보 상세 팝업 창 띄우기"""
        dlg = ImageInfoDialog(self.state_mgr, self)
        dlg.exec()

    def open_3d_viewer(self):
        """
        독립 3D 볼륨 뷰어를 안전하게 엽니다.
        창을 닫았다가 다시 열 때 OpenGL 컨텍스트 충돌/배경 비침(Ghosting) 및 세그폴트를 방지하기 위해
        닫힌 창은 안전하게 파괴하고 새 인스턴스로 생성합니다.
        """
        if self.volume_window is not None:
            try:
                if self.volume_window.isVisible():
                    self.volume_window.raise_()
                    self.volume_window.activateWindow()
                    return
                else:
                    self.volume_window.close()
                    self.volume_window = None
            except Exception:
                self.volume_window = None

        self.volume_window = VolumeViewerWindow(self.state_mgr)
        self.volume_window.window_closed.connect(self._on_volume_window_closed)
        self.volume_window.show()
        self.volume_window.raise_()
        self.volume_window.activateWindow()

    def _on_volume_window_closed(self):
        self.volume_window = None

    def _on_volume_loaded(self, vtk_image):
        dims = self.state_mgr.dimensions
        spc = self.state_mgr.spacing
        self.status_bar.showMessage(
            f"영상 로드 성공 | 차원: {dims[0]}x{dims[1]}x{dims[2]} | "
            f"복셀 간격: {spc[0]:.2f}x{spc[1]:.2f}x{spc[2]:.2f} mm"
        )
        series_path = getattr(self.state_mgr, "series_path", None)
        if series_path:
            self._add_recent_path(series_path)
            self._check_load_matching_user_info(series_path)

    def _show_about(self):
        QMessageBox.about(
            self,
            "Dental FEA 안내",
            "<b>Dental FEA (치과 생체역학 유한요소해석 시스템)</b><br><br>"
            "Phase 1: DICOM 2D MPR 3분할 연동 뷰어 & 3D 볼륨 뷰어<br>"
            "- Axial, Coronal, Sagittal 실시간 직교 슬라이싱 및 고정밀 2D 눈금자 연동<br>"
            "- VTK Smart/GPU Volume Ray Casting 기반 3D 볼륨 렌더링"
        )

    def _save_user_info(self, default_path: Optional[str] = None):
        """그리기, 계측, 주석 데이터를 saved/ 폴더에 JSON 파일로 저장"""
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "saved")
        os.makedirs(base_dir, exist_ok=True)

        if not default_path:
            # 기본 파일명: DICOM 시리즈 이름 또는 timestamp 기반
            series_p = getattr(self.state_mgr, "series_path", "")
            if series_p:
                clean_name = os.path.basename(series_p.rstrip("/\\"))
                def_name = f"{clean_name}_user_info.json"
            else:
                from datetime import datetime
                def_name = f"user_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            default_path = os.path.join(base_dir, def_name)

        filepath, _ = QFileDialog.getSaveFileName(
            self, "사용자 정보 저장 (주석, 계측, 그리기)", default_path,
            "JSON 파일 (*.json);;모든 파일 (*.*)"
        )
        if not filepath:
            return False

        try:
            data = self.mpr_container.get_user_info_data()
            data["series_path"] = getattr(self.state_mgr, "series_path", "")
            data["saved_at"] = str(QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate))
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.status_bar.showMessage(f"사용자 정보 저장 완료: {os.path.basename(filepath)}", 4000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"사용자 정보 저장 중 오류가 발생했습니다:\n{str(e)}")
            return False

    def _load_user_info(self, default_path: Optional[str] = None):
        """JSON 파일로부터 그리기, 계측, 주석 데이터 복원"""
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "saved")
        os.makedirs(base_dir, exist_ok=True)

        if not default_path:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "사용자 정보 로드", base_dir,
                "JSON 파일 (*.json);;모든 파일 (*.*)"
            )
        else:
            filepath = default_path

        if not filepath or not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.mpr_container.set_user_info_data(data)
            self.status_bar.showMessage(f"사용자 정보 로드 완료: {os.path.basename(filepath)}", 4000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "로드 오류", f"사용자 정보 로드 중 오류가 발생했습니다:\n{str(e)}")
            return False

    def _check_load_matching_user_info(self, series_path: str):
        """DICOM 로드 후 일치하거나 관련된 사용자 정보 파일이 있을 때 로드 여부 질문"""
        if not series_path:
            return
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "saved")
        clean_name = os.path.basename(series_path.rstrip("/\\"))
        matching_file = os.path.join(base_dir, f"{clean_name}_user_info.json")

        if os.path.exists(matching_file):
            reply = QMessageBox.question(
                self, "사용자 정보 로드",
                f"현재 영상과 일치하는 사용자 정보(주석/계측/도형) 파일이 발견되었습니다:\n'{os.path.basename(matching_file)}'\n\n불러오시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._load_user_info(matching_file)

    def _has_user_info_content(self) -> bool:
        """저장할 주석, 계측, 또는 도형 데이터가 존재하는지 검사"""
        data = self.mpr_container.get_user_info_data()
        has_annot = bool(data.get("annotations", {}).get("text", "").strip())
        has_measure = bool(data.get("measurements", []))
        has_shapes = bool(data.get("shapes", []))
        return has_annot or has_measure or has_shapes

    def closeEvent(self, event):
        """
        메인 윈도우 닫힐 때:
        1. 작성된 사용자 정보(주석, 계측, 도형)가 있을 경우 저장 여부 확인 팝업
        2. 설정 저장 및 3D 독립 창 정리
        """
        if self._has_user_info_content():
            reply = QMessageBox.question(
                self, "사용자 정보 저장 확인",
                "작성 중인 사용자 정보(주석, 계측, 그리기 도형)가 있습니다.\n종료하기 전에 파일로 저장하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            elif reply == QMessageBox.StandardButton.Yes:
                if not self._save_user_info():
                    event.ignore()
                    return

        try:
            geom = bytes(self.saveGeometry().toHex()).decode("utf-8")
            self.settings_mgr.set("main_window.geometry", geom, auto_save=False)
            self.mpr_container.save_splitter_state(self.settings_mgr)
        except Exception:
            pass
        self.settings_mgr.save()

        if self.volume_window is not None:
            try:
                self.volume_window.close()
            except Exception:
                pass
        super().closeEvent(event)

