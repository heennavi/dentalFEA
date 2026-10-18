"""
settings_dialog.py
환경설정 (Settings) 대화상자.
- 메뉴 Tools > 환경설정 에서 실행
- 상단 탭: [초기설정], [파노라마 설정], [임플란트 설정]
- 임플란트 설정 탭 (media_1792270674530.png 일치):
  1) 임플란트 형상 만들기 (3D Implant Modeler)
     - 직경(Diameter) 슬라이더 (2.0 ~ 7.0 mm, 0.5 step)
     - Taper(각도) 슬라이더 (0 ~ 15 °, 1 step)
     - 길이(Length) 슬라이더 (5.0 ~ 15.0 mm, 0.5 step)
     - 상부 형태 콤보박스 (flat / collar 2mm 칼라형 5° Flare)
     - 나사 형태 콤보박스 (삼각형 V-Thread / 사각형(핀) Square-Fin)
     - 상부 Micro-thread 체크박스
     - [임플란트 저장 (asset/implant/custom)] 녹색 버튼
     - 우측 3D 실시간 미리보기 (Implant3DPreviewWidget)
  2) 임플란트 리스트 및 사양 (Implant Library)
     - 직경 필터 / 길이 필터
     - 6컬럼 테이블 (이름/사양, 직경, 길이, Taper, 상부형태, 구분)
     - 우측: 선택된 임플란트 사양 및 3D 형상 + 커스텀 삭제 버튼 + 상세 스펙
"""
import os
from typing import Optional, List
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox,
    QGridLayout, QLabel, QSlider, QComboBox, QCheckBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QFrame,
    QSplitter, QTextBrowser, QSpinBox, QColorDialog
)
from PySide6.QtCore import Qt, Signal, QSettings, QSize
from PySide6.QtGui import QColor

from dental_fea_app.placement.implant_generator import (
    ImplantSpec, Implant3DGenerator, ImplantLibraryManager
)
from dental_fea_app.ui.icons import IconFactory


class Implant3DPreviewWidget(QWidget):
    """안정화된 카메라 시점을 갖춘 임플란트 3D VTK 프리뷰 위젯"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.10, 0.12, 0.16)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidget.SetInteractorStyle(self.style)

        self.polydata_mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.polydata_mapper)

        prop = self.actor.GetProperty()
        prop.SetColor(0.70, 0.72, 0.75)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.85)
        prop.SetSpecular(0.4)
        prop.SetSpecularPower(25.0)
        prop.SetInterpolationToPhong()
        prop.BackfaceCullingOff()

        self.renderer.AddActor(self.actor)

        light1 = vtk.vtkLight()
        light1.SetPosition(10, 20, 30)
        light1.SetFocalPoint(0, 0, 5)
        light1.SetColor(1.0, 1.0, 1.0)
        light1.SetIntensity(0.75)
        self.renderer.AddLight(light1)

        light2 = vtk.vtkLight()
        light2.SetPosition(-15, -10, 10)
        light2.SetFocalPoint(0, 0, 5)
        light2.SetColor(0.8, 0.9, 1.0)
        light2.SetIntensity(0.4)
        self.renderer.AddLight(light2)

        self._camera_initialized = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._camera_initialized:
            try:
                self.vtkWidget.Initialize()
                self._camera_initialized = True
                self.reset_camera()
            except Exception:
                pass
        elif self.isVisible() and self.vtkWidget.GetRenderWindow():
            try:
                self.vtkWidget.update()
            except Exception:
                pass

    def set_polydata(self, polydata: vtk.vtkPolyData, force_reset_camera: bool = False):
        self.polydata_mapper.SetInputData(polydata)
        self.polydata_mapper.Update()
        if not self._camera_initialized or not self.isVisible():
            return
        if force_reset_camera:
            self.reset_camera()
        else:
            self.renderer.ResetCameraClippingRange()
            if self.vtkWidget.GetRenderWindow():
                self.vtkWidget.GetRenderWindow().Render()

    def set_inverted(self, inverted: bool):
        if inverted:
            self.actor.SetOrientation(180, 0, 0)
        else:
            self.actor.SetOrientation(0, 0, 0)
        self.reset_camera()

    def reset_camera(self):
        if not self._camera_initialized or not self.isVisible():
            return
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        bounds = self.actor.GetBounds()
        cz = (bounds[4] + bounds[5]) / 2.0 if bounds[5] > bounds[4] else 5.0
        cam.SetFocalPoint(0, 0, cz)
        cam.SetPosition(0, -26, cz + 7)
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        if self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()


class SettingsDialog(QDialog):
    """환경설정 대화상자 (Settings Dialog)"""

    implant_saved = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("환경설정")
        self.resize(1020, 750)
        self.setStyleSheet("""
            QDialog {
                background-color: #121824;
                color: #e2e8f0;
            }
            QTabWidget::pane {
                border: 1px solid #1e293b;
                background-color: #0f172a;
            }
            QTabBar::tab {
                background: #1e293b;
                color: #94a3b8;
                padding: 8px 18px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #0284c7;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                color: #38bdf8;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
        """)

        self.all_implants: List[ImplantSpec] = []
        self.filtered_implants: List[ImplantSpec] = []
        self.selected_library_spec: Optional[ImplantSpec] = None

        self._init_ui()
        self._load_implant_list()
        self._update_creator_preview(initial=True)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.addTab(self._create_general_tab(), "초기설정")
        self.tab_widget.addTab(self._create_panoramic_tab(), "파노라마 설정")
        self.tab_widget.addTab(self._create_implant_tab(), "임플란트 설정")
        self.tab_widget.setCurrentIndex(2)  # 임플란트 설정 탭 기본 선택
        main_layout.addWidget(self.tab_widget)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_close = QPushButton("닫기", self)
        btn_close.setFixedWidth(100)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        main_layout.addLayout(btn_layout)

    def _create_general_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        grp = QGroupBox("기본 뷰포트 설정", tab)
        g_layout = QGridLayout(grp)
        g_layout.addWidget(QLabel("기본 W/L 프리셋:"), 0, 0)
        combo = QComboBox(grp)
        combo.addItems(["Bone & Teeth", "Soft Tissue", "Airway"])
        g_layout.addWidget(combo, 0, 1)
        layout.addWidget(grp)
        layout.addStretch()
        return tab

    def _create_panoramic_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        grp = QGroupBox("파노라마 CPR 뷰포트 설정", tab)
        g_layout = QGridLayout(grp)
        g_layout.addWidget(QLabel("기본 횡단면 개수:"), 0, 0)
        spin = QSpinBox(grp)
        spin.setRange(8, 64)
        spin.setValue(24)
        g_layout.addWidget(spin, 0, 1)
        layout.addWidget(grp)
        layout.addStretch()
        return tab

    def _create_implant_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(10, 10, 10, 10)
        tab_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Vertical, tab)

        # 1. 상단: 임플란트 형상 만들기
        grp_create = QGroupBox("임플란트 형상 만들기 (3D Implant Modeler)", splitter)
        create_layout = QHBoxLayout(grp_create)
        create_layout.setContentsMargins(10, 10, 10, 10)
        create_layout.setSpacing(12)

        ctrl_frame = QWidget(grp_create)
        ctrl_layout = QGridLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(8)

        # 직경 슬라이더 (2.0 ~ 7.0 mm, step 0.5 mm)
        ctrl_layout.addWidget(QLabel("직경 (Diameter):"), 0, 0)
        self.slider_dia = QSlider(Qt.Orientation.Horizontal, ctrl_frame)
        self.slider_dia.setRange(4, 14)
        self.slider_dia.setValue(8)  # 4.0 mm
        self.lbl_dia_val = QLabel("4.0 mm", ctrl_frame)
        self.lbl_dia_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_dia.valueChanged.connect(self._on_creator_param_changed)
        ctrl_layout.addWidget(self.slider_dia, 0, 1)
        ctrl_layout.addWidget(self.lbl_dia_val, 0, 2)

        # Taper 슬라이더 (0 ~ 15 °, step 1 °)
        ctrl_layout.addWidget(QLabel("Taper (각도):"), 1, 0)
        self.slider_taper = QSlider(Qt.Orientation.Horizontal, ctrl_frame)
        self.slider_taper.setRange(0, 15)
        self.slider_taper.setValue(0)
        self.lbl_taper_val = QLabel("0 °", ctrl_frame)
        self.lbl_taper_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_taper.valueChanged.connect(self._on_creator_param_changed)
        ctrl_layout.addWidget(self.slider_taper, 1, 1)
        ctrl_layout.addWidget(self.lbl_taper_val, 1, 2)

        # 길이 슬라이더 (5.0 ~ 15.0 mm, step 0.5 mm)
        ctrl_layout.addWidget(QLabel("길이 (Length):"), 2, 0)
        self.slider_len = QSlider(Qt.Orientation.Horizontal, ctrl_frame)
        self.slider_len.setRange(10, 30)
        self.slider_len.setValue(20)  # 10.0 mm
        self.lbl_len_val = QLabel("10.0 mm", ctrl_frame)
        self.lbl_len_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.slider_len.valueChanged.connect(self._on_creator_param_changed)
        ctrl_layout.addWidget(self.slider_len, 2, 1)
        ctrl_layout.addWidget(self.lbl_len_val, 2, 2)

        # 상부 형태
        ctrl_layout.addWidget(QLabel("상부 형태:"), 3, 0)
        self.combo_top = QComboBox(ctrl_frame)
        self.combo_top.addItems(["flat (편평형)", "collar (2mm 칼라형, 5° Flare)"])
        self.combo_top.currentIndexChanged.connect(self._on_creator_param_changed)
        ctrl_layout.addWidget(self.combo_top, 3, 1, 1, 2)

        # 나사 형태
        ctrl_layout.addWidget(QLabel("나사 형태:"), 4, 0)
        self.combo_thread = QComboBox(ctrl_frame)
        self.combo_thread.addItems(["삼각형 나사산 (V-Thread, 0.6mm)", "사각형(핀) 나사산 (Square-Fin, 0.6mm)"])
        self.combo_thread.currentIndexChanged.connect(self._on_creator_param_changed)
        ctrl_layout.addWidget(self.combo_thread, 4, 1, 1, 2)

        # Micro-thread 체크박스
        self.chk_micro = QCheckBox("상부 Micro-thread (상부 1.5mm 미세 나선 홈) 적용", ctrl_frame)
        self.chk_micro.setChecked(True)
        self.chk_micro.toggled.connect(self._on_creator_param_changed)
        ctrl_layout.addWidget(self.chk_micro, 5, 0, 1, 3)

        # 저장 버튼
        self.btn_save_implant = QPushButton("임플란트 저장 (asset/implant/custom)", ctrl_frame)
        self.btn_save_implant.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: #ffffff;
                font-weight: bold;
                padding: 7px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #047857;
            }
        """)
        self.btn_save_implant.clicked.connect(self._on_save_custom_implant)
        ctrl_layout.addWidget(self.btn_save_implant, 6, 0, 1, 3)

        create_layout.addWidget(ctrl_frame, 3)

        # 3D 실시간 미리보기
        prev_container = QWidget(grp_create)
        prev_layout = QVBoxLayout(prev_container)
        prev_layout.setContentsMargins(0, 0, 0, 0)
        prev_lbl = QLabel("3D 실시간 미리보기 (마우스 좌클릭: 회전, 우클릭: 줌)", prev_container)
        prev_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        prev_layout.addWidget(prev_lbl)
        self.creator_3d_view = Implant3DPreviewWidget(prev_container)
        prev_layout.addWidget(self.creator_3d_view, 1)
        create_layout.addWidget(prev_container, 2)

        splitter.addWidget(grp_create)

        # 2. 하단: 임플란트 리스트 및 사양
        grp_list = QGroupBox("임플란트 리스트 및 사양 (Implant Library)", splitter)
        list_outer = QVBoxLayout(grp_list)
        list_outer.setContentsMargins(10, 10, 10, 10)
        list_outer.setSpacing(8)

        filter_bar = QHBoxLayout()
        filter_bar.addWidget(QLabel("직경 필터:"))
        self.combo_filter_dia = QComboBox(grp_list)
        self.combo_filter_dia.addItems(["전체 직경 (모두보기)", "Ø3.0mm", "Ø3.5mm", "Ø4.0mm", "Ø4.5mm", "Ø5.0mm", "Ø6.0mm"])
        self.combo_filter_dia.currentTextChanged.connect(self._apply_implant_filters)
        filter_bar.addWidget(self.combo_filter_dia)

        filter_bar.addSpacing(15)
        filter_bar.addWidget(QLabel("길이 필터:"))
        self.combo_filter_len = QComboBox(grp_list)
        self.combo_filter_len.addItems(["전체 길이 (모두보기)", "5.0mm", "6.0mm", "7.0mm", "8.5mm", "10.0mm", "11.5mm"])
        self.combo_filter_len.currentTextChanged.connect(self._apply_implant_filters)
        filter_bar.addWidget(self.combo_filter_len)
        filter_bar.addStretch()
        list_outer.addLayout(filter_bar)

        list_inner = QHBoxLayout()
        list_inner.setSpacing(12)

        self.table_implants = QTableWidget(grp_list)
        self.table_implants.setColumnCount(6)
        self.table_implants.setHorizontalHeaderLabels(["이름 / 사양", "직경", "길이", "Taper", "상부형태", "구분"])
        self.table_implants.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_implants.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_implants.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_implants.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_implants.itemSelectionChanged.connect(self._on_implant_selected)
        list_inner.addWidget(self.table_implants, 3)

        detail_container = QWidget(grp_list)
        d_layout = QVBoxLayout(detail_container)
        d_layout.setContentsMargins(0, 0, 0, 0)
        d_layout.setSpacing(6)

        d_header = QHBoxLayout()
        d_lbl = QLabel("선택된 임플란트 사양 및 3D 형상", detail_container)
        d_lbl.setStyleSheet("font-weight: bold; color: #f8fafc;")
        d_header.addWidget(d_lbl)
        d_header.addStretch()

        self.btn_delete_custom = QPushButton("삭제", detail_container)
        self.btn_delete_custom.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: #ffffff;
                font-weight: bold;
                padding: 4px 10px;
                border-radius: 3px;
            }
        """)
        self.btn_delete_custom.clicked.connect(self._on_delete_custom_implant)
        self.btn_delete_custom.hide()
        d_header.addWidget(self.btn_delete_custom)
        d_layout.addLayout(d_header)

        self.library_3d_view = Implant3DPreviewWidget(detail_container)
        d_layout.addWidget(self.library_3d_view, 1)

        self.txt_implant_spec = QTextBrowser(detail_container)
        self.txt_implant_spec.setFixedHeight(110)
        self.txt_implant_spec.setStyleSheet("""
            QTextBrowser {
                background-color: #090d16;
                border: 1px solid #1e293b;
                color: #94a3b8;
                font-size: 11px;
                padding: 4px;
            }
        """)
        d_layout.addWidget(self.txt_implant_spec)

        list_inner.addWidget(detail_container, 2)
        list_outer.addLayout(list_inner)

        splitter.addWidget(grp_list)
        tab_layout.addWidget(splitter)
        return tab

    def _get_current_creator_spec(self) -> ImplantSpec:
        dia = self.slider_dia.value() * 0.5
        taper = float(self.slider_taper.value())
        length = self.slider_len.value() * 0.5
        top = "flat" if self.combo_top.currentIndex() == 0 else "collar"
        thread = "triangular" if self.combo_thread.currentIndex() == 0 else "square"
        micro = self.chk_micro.isChecked()

        name = f"Custom Ø{dia:.1f} x {length:.1f}mm ({int(taper)}° {top.capitalize()})"
        return ImplantSpec(
            id="temp_creator",
            name=name,
            diameter_mm=dia,
            length_mm=length,
            taper_deg=taper,
            top_type=top,
            thread_type=thread,
            micro_threads=micro,
            pitch_mm=0.6,
            depth_mm=0.3,
            is_custom=True
        )

    def _on_creator_param_changed(self):
        self.lbl_dia_val.setText(f"{self.slider_dia.value() * 0.5:.1f} mm")
        self.lbl_taper_val.setText(f"{self.slider_taper.value()} °")
        self.lbl_len_val.setText(f"{self.slider_len.value() * 0.5:.1f} mm")
        self._update_creator_preview(initial=False)

    def _update_creator_preview(self, initial: bool = False):
        spec = self._get_current_creator_spec()
        poly = Implant3DGenerator.generate_implant_polydata(spec)
        self.creator_3d_view.set_polydata(poly, force_reset_camera=initial)

    def _on_save_custom_implant(self):
        spec = self._get_current_creator_spec()
        success, msg = ImplantLibraryManager.save_custom_implant(spec)
        if success:
            QMessageBox.information(self, "저장 성공", f"커스텀 임플란트가 저장되었습니다:\nID: {msg}")
            self._load_implant_list()
            self.implant_saved.emit(msg)
        else:
            QMessageBox.critical(self, "저장 오류", f"임플란트 저장 실패:\n{msg}")

    def _load_implant_list(self):
        self.all_implants = ImplantLibraryManager.list_all_implants()
        self._apply_implant_filters()

    def _apply_implant_filters(self):
        sel_dia = self.combo_filter_dia.currentText()
        sel_len = self.combo_filter_len.currentText()

        self.filtered_implants = []
        for imp in self.all_implants:
            if sel_dia != "전체 직경 (모두보기)":
                val = float(sel_dia.replace("Ø", "").replace("mm", "").strip())
                if abs(imp.diameter_mm - val) > 0.05:
                    continue
            if sel_len != "전체 길이 (모두보기)":
                val = float(sel_len.replace("mm", "").strip())
                if abs(imp.length_mm - val) > 0.05:
                    continue
            self.filtered_implants.append(imp)

        self.table_implants.setRowCount(len(self.filtered_implants))
        for row, imp in enumerate(self.filtered_implants):
            self.table_implants.setItem(row, 0, QTableWidgetItem(imp.name))
            self.table_implants.setItem(row, 1, QTableWidgetItem(f"Ø{imp.diameter_mm:.1f}mm"))
            self.table_implants.setItem(row, 2, QTableWidgetItem(f"{imp.length_mm:.1f}mm"))
            self.table_implants.setItem(row, 3, QTableWidgetItem(f"{imp.taper_deg:.0f}°"))
            self.table_implants.setItem(row, 4, QTableWidgetItem(imp.top_type.capitalize()))
            self.table_implants.setItem(row, 5, QTableWidgetItem("사용자정의" if imp.is_custom else "기본제공"))

        if self.filtered_implants:
            self.table_implants.selectRow(0)

    def _on_implant_selected(self):
        rows = self.table_implants.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self.filtered_implants):
            self.selected_library_spec = self.filtered_implants[row]
            poly = ImplantLibraryManager.load_implant_polydata(self.selected_library_spec)
            self.library_3d_view.set_polydata(poly, force_reset_camera=True)

            self.btn_delete_custom.setVisible(self.selected_library_spec.is_custom)
            html = f"""
            <b>이름:</b> {self.selected_library_spec.name}<br>
            <b>직경 / 길이:</b> Ø{self.selected_library_spec.diameter_mm:.1f} mm / {self.selected_library_spec.length_mm:.1f} mm | <b>Taper:</b> {self.selected_library_spec.taper_deg:.0f}°<br>
            <b>상부 형태:</b> {self.selected_library_spec.top_type} | <b>나사 형태:</b> {self.selected_library_spec.thread_type}<br>
            <b>표면/재질:</b> SLA Titanium | <b>구분:</b> {'사용자정의' if self.selected_library_spec.is_custom else '기본제공'}
            """
            self.txt_implant_spec.setHtml(html)

    def _on_delete_custom_implant(self):
        if not self.selected_library_spec or not self.selected_library_spec.is_custom:
            return
        reply = QMessageBox.question(
            self, "삭제 확인",
            f"선택한 커스텀 임플란트 '{self.selected_library_spec.name}'을(를) 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            ImplantLibraryManager.delete_custom_implant(self.selected_library_spec.id)
            self._load_implant_list()
