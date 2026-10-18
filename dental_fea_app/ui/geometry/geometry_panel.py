"""
geometry_panel.py
좌측 제어 패널 내 "4. 기하 처리" 워크플로 탭 UI 위젯.
- 임플란트 주변 ROI 여유(mm), 스무딩 강도, 목표 삼각형 크기, 언더사이즈 파라미터 조절
- QThread 비동기 실행 및 프로그레스/단계 텍스트 표시
- 3D 뷰어 피질골/수질골/임플란트 가시성 및 단면 클리핑 제어
- 분석 리포트(부피, 접촉면적, watertight 무결성) 표시 (이상 시 빨간색 강조)
- STL 3개 솔리드 및 manifest.json 내보내기
- 임플란트 포즈/분할 변경 시 stale(오래됨) 상태 감지 및 재생성 유도
"""

import os
import json
import datetime
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QGroupBox, QSlider, QComboBox,
    QTextBrowser, QFileDialog, QMessageBox, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QTimer

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.geometry.boolean_ops import FeaGeometry, export_stl
from dental_fea_app.ui.geometry.geometry_worker import GeometryWorker


class GeometryPanel(QWidget):
    """FEA 기하 처리 및 메쉬 준비 제어 패널 위젯"""

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.worker: Optional[GeometryWorker] = None
        self._is_stale = False
        self._current_geometry: Optional[FeaGeometry] = None

        self._init_ui()

        # StateManager 신호 바인딩
        self.state_mgr.volume_loaded.connect(self._on_volume_loaded)
        self.state_mgr.segmentation_result_changed.connect(self._on_segmentation_changed)
        self.state_mgr.implant_pose_changed.connect(self._on_implant_changed)
        self.state_mgr.implant_transformed.connect(self._on_implant_changed)
        if hasattr(self.state_mgr, "placed_implants_changed"):
            self.state_mgr.placed_implants_changed.connect(self._on_implant_changed)
        if hasattr(self.state_mgr, "implants_updated"):
            self.state_mgr.implants_updated.connect(self._on_implant_changed)
        self.state_mgr.fea_geometry_changed.connect(self._on_fea_geometry_changed)

        if getattr(self.state_mgr, "fea_geometry", None) is not None:
            self._on_fea_geometry_changed(self.state_mgr.fea_geometry)

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(6)

        # 1. 기하 파라미터 그룹
        param_group = QGroupBox("기하 파라미터 (Parameters)", self)
        param_layout = QVBoxLayout(param_group)
        param_layout.setContentsMargins(6, 6, 6, 6)
        param_layout.setSpacing(4)

        # ROI 여유 (mm)
        row_roi = QHBoxLayout()
        row_roi.addWidget(QLabel("ROI 여유 (mm):", param_group))
        self.spin_roi_margin = QDoubleSpinBox(param_group)
        self.spin_roi_margin.setRange(2.0, 50.0)
        self.spin_roi_margin.setValue(12.0)
        self.spin_roi_margin.setSingleStep(1.0)
        self.spin_roi_margin.setToolTip("임플란트 주변 추출 여유 마진 (기본 12mm)")
        row_roi.addWidget(self.spin_roi_margin)
        param_layout.addLayout(row_roi)

        # 표면 스무딩 강도
        row_smooth = QHBoxLayout()
        row_smooth.addWidget(QLabel("스무딩 강도:", param_group))
        self.spin_smooth = QSpinBox(param_group)
        self.spin_smooth.setRange(0, 50)
        self.spin_smooth.setValue(15)
        self.spin_smooth.setToolTip("Windowed-Sinc 스무딩 반복 횟수 (기본 15)")
        row_smooth.addWidget(self.spin_smooth)
        param_layout.addLayout(row_smooth)

        # 목표 삼각형 크기 (mm)
        row_edge = QHBoxLayout()
        row_edge.addWidget(QLabel("삼각형 크기 (mm):", param_group))
        self.spin_target_edge = QDoubleSpinBox(param_group)
        self.spin_target_edge.setRange(0.2, 5.0)
        self.spin_target_edge.setValue(1.0)
        self.spin_target_edge.setSingleStep(0.2)
        self.spin_target_edge.setToolTip("목표 삼각형 변 길이 (기본 1.0mm, Quadric Decimation)")
        row_edge.addWidget(self.spin_target_edge)
        param_layout.addLayout(row_edge)

        # 드릴링 언더사이즈 (mm)
        row_under = QHBoxLayout()
        row_under.addWidget(QLabel("언더사이즈 (mm):", param_group))
        self.spin_undersize = QDoubleSpinBox(param_group)
        self.spin_undersize.setRange(-1.0, 2.0)
        self.spin_undersize.setValue(0.0)
        self.spin_undersize.setSingleStep(0.1)
        self.spin_undersize.setToolTip("임플란트 식립 드릴링 공간 수축량 (기본 0mm = 정확한 맞닿음)")
        row_under.addWidget(self.spin_undersize)
        param_layout.addLayout(row_under)

        layout.addWidget(param_group)

        # 2. 실행 및 상태 인디케이터
        exec_group = QGroupBox("FEA 솔리드 생성", self)
        exec_layout = QVBoxLayout(exec_group)
        exec_layout.setContentsMargins(6, 6, 6, 6)
        exec_layout.setSpacing(4)

        # 상태 라벨
        self.lbl_stale_status = QLabel("기하 데이터 미생성", exec_group)
        self.lbl_stale_status.setStyleSheet("color: #888899; font-size: 11px; font-weight: bold;")
        exec_layout.addWidget(self.lbl_stale_status)

        # 실행 버튼
        self.btn_build_geom = QPushButton("⚡ 기하 생성 (Build FEA Geometry)", exec_group)
        self.btn_build_geom.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 4px;
                border-radius: 4px;
                border: 1px solid #3878b2;
            }
            QPushButton:hover {
                background-color: #275284;
                border-color: #61afef;
            }
            QPushButton:disabled {
                background-color: #2c2c34;
                color: #666670;
                border-color: #444452;
            }
        """)
        self.btn_build_geom.clicked.connect(self._run_geometry_generation)
        exec_layout.addWidget(self.btn_build_geom)

        # 진행 프로그레스바 및 단계 라벨
        self.progress_bar = QProgressBar(exec_group)
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setVisible(False)
        exec_layout.addWidget(self.progress_bar)

        self.lbl_step_status = QLabel("", exec_group)
        self.lbl_step_status.setStyleSheet("color: #61afef; font-size: 10px;")
        self.lbl_step_status.setWordWrap(True)
        self.lbl_step_status.setVisible(False)
        exec_layout.addWidget(self.lbl_step_status)

        layout.addWidget(exec_group)

        # 3. 3D 가시성 및 단면 클리핑 제어 그룹
        self.group_vis = QGroupBox("3D 시각화 및 단면 컷팅", self)
        view_layout = QVBoxLayout(self.group_vis)
        view_layout.setContentsMargins(6, 6, 6, 6)
        view_layout.setSpacing(4)

        # 가시성 체크박스 3개 (피질골, 수질골, 임플란트)
        chk_layout = QHBoxLayout()
        chk_layout.setSpacing(6)
        self.chk_cortical = QCheckBox("피질골 ■", self.group_vis)
        self.chk_cortical.setStyleSheet("color: #e06c75; font-weight: bold;")
        self.chk_cortical.setChecked(True)
        self.chk_cortical.toggled.connect(self._on_display_toggled)
        chk_layout.addWidget(self.chk_cortical)

        self.chk_cancellous = QCheckBox("수질골 ■", self.group_vis)
        self.chk_cancellous.setStyleSheet("color: #61afef; font-weight: bold;")
        self.chk_cancellous.setChecked(True)
        self.chk_cancellous.toggled.connect(self._on_display_toggled)
        chk_layout.addWidget(self.chk_cancellous)

        self.chk_implant = QCheckBox("임플란트 ■", self.group_vis)
        self.chk_implant.setStyleSheet("color: #d19a66; font-weight: bold;")
        self.chk_implant.setChecked(True)
        self.chk_implant.toggled.connect(self._on_display_toggled)
        chk_layout.addWidget(self.chk_implant)
        view_layout.addLayout(chk_layout)

        # 단면 클리핑 (평면 컷팅)
        clip_row = QHBoxLayout()
        self.chk_clip_enable = QCheckBox("단면 클리핑", self.group_vis)
        self.chk_clip_enable.setChecked(False)
        self.chk_clip_enable.toggled.connect(self._on_clipping_toggled)
        clip_row.addWidget(self.chk_clip_enable)

        self.combo_clip_axis = QComboBox(self.group_vis)
        self.combo_clip_axis.addItems(["Z (Axial)", "Y (Coronal)", "X (Sagittal)"])
        self.combo_clip_axis.currentIndexChanged.connect(self._on_clipping_changed)
        clip_row.addWidget(self.combo_clip_axis)

        self.chk_clip_invert = QCheckBox("반전", self.group_vis)
        self.chk_clip_invert.setChecked(False)
        self.chk_clip_invert.toggled.connect(self._on_clipping_changed)
        clip_row.addWidget(self.chk_clip_invert)
        view_layout.addLayout(clip_row)

        # 클리핑 슬라이더 (0 ~ 100%)
        self.slider_clip = QSlider(Qt.Orientation.Horizontal, self.group_vis)
        self.slider_clip.setRange(0, 100)
        self.slider_clip.setValue(50)
        self.slider_clip.valueChanged.connect(self._on_clipping_changed)
        view_layout.addWidget(self.slider_clip)

        layout.addWidget(self.group_vis)

        # 4. 분석 보고서 (Report)
        rep_group = QGroupBox("형상 분석 보고서 (Report)", self)
        rep_layout = QVBoxLayout(rep_group)
        rep_layout.setContentsMargins(4, 4, 4, 4)
        rep_layout.setSpacing(4)

        self.text_report = QTextBrowser(rep_group)
        self.text_report.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e24;
                color: #abb2bf;
                border: 1px solid #3c3c46;
                border-radius: 4px;
                font-size: 11px;
            }
        """)
        self.text_report.setFixedHeight(120)
        self.text_report.setHtml("<span style='color: #666670;'>[기하 생성]을 실행하면 상세 분석 리포트가 여기에 표시됩니다.</span>")
        rep_layout.addWidget(self.text_report)

        layout.addWidget(rep_group)

        # 5. STL 및 Manifest 내보내기 버튼
        self.btn_export_stl = QPushButton("💾 STL 및 Manifest 내보내기...", self)
        self.btn_export_stl.setEnabled(False)
        self.btn_export_stl.setStyleSheet("""
            QPushButton {
                background-color: #2c3848;
                color: #61afef;
                border: 1px solid #4d698e;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #38485c;
                border-color: #61afef;
                color: #ffffff;
            }
            QPushButton:disabled {
                background-color: #21252b;
                color: #555560;
                border-color: #33333e;
            }
        """)
        self.btn_export_stl.clicked.connect(self._export_stl_and_manifest)
        layout.addWidget(self.btn_export_stl)

        layout.addStretch()
        self._update_controls_enabled()

    def _update_controls_enabled(self):
        has_vol = (self.state_mgr.vtk_image is not None)
        has_seg = (self.state_mgr.segmentation_result is not None)
        has_imp = bool(
            (getattr(self.state_mgr, "implant_placement_mgr", None) and self.state_mgr.implant_placement_mgr.implants)
            or (self.state_mgr.implant_model is not None)
        )

        can_build = has_vol and has_seg and has_imp
        self.btn_build_geom.setEnabled(can_build)
        has_geom = (self._current_geometry is not None)
        self.group_vis.setEnabled(has_geom)
        self.btn_export_stl.setEnabled(has_geom)

    def _mark_stale(self):
        if self._current_geometry is not None:
            self._is_stale = True
            self.lbl_stale_status.setText("⚠️ [오래됨] 데이터 변경됨: 재생성 필요")
            self.lbl_stale_status.setStyleSheet("color: #e5c07b; font-size: 11px; font-weight: bold;")
            self.btn_build_geom.setStyleSheet("""
                QPushButton {
                    background-color: #8c5210;
                    color: #ffffff;
                    font-weight: bold;
                    padding: 6px 4px;
                    border-radius: 4px;
                    border: 1px solid #d19a66;
                }
                QPushButton:hover {
                    background-color: #a86414;
                }
            """)

    def _on_volume_loaded(self, vtk_img):
        self._update_controls_enabled()

    def _on_segmentation_changed(self, result):
        self._update_controls_enabled()
        self._mark_stale()

    def _on_implant_changed(self, *args):
        self._update_controls_enabled()
        self._mark_stale()

    def _run_geometry_generation(self):
        self.btn_build_geom.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_step_status.setVisible(True)
        self.lbl_step_status.setText("기하 생성 파이프라인 초기화 중...")

        self.worker = GeometryWorker(
            state_manager=self.state_mgr,
            roi_margin_mm=self.spin_roi_margin.value(),
            smooth_iterations=self.spin_smooth.value(),
            smooth_passband=0.10,
            target_edge_mm=self.spin_target_edge.value(),
            undersize_mm=self.spin_undersize.value(),
            parent=self
        )
        self.worker.progress_updated.connect(self._on_worker_progress)
        self.worker.finished_success.connect(self._on_worker_success)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()

    def _on_worker_progress(self, step: int, total: int, msg: str):
        self.progress_bar.setValue(step)
        self.lbl_step_status.setText(msg)

    def _on_fea_geometry_changed(self, fea_geometry: Optional[FeaGeometry]):
        self._current_geometry = fea_geometry
        if fea_geometry is None:
            self._is_stale = False
            self.lbl_stale_status.setText("대기 중: 분할 마스크와 임플란트 필요")
            self.lbl_stale_status.setStyleSheet("color: #abb2bf; font-size: 11px;")
            self.btn_export_stl.setEnabled(False)
            self.group_vis.setEnabled(False)
            self.text_report.setHtml("<span style='color: #666;'>기하 생성 후 분석 보고서가 여기에 표시됩니다.</span>")
            return

        self._is_stale = False
        self.lbl_stale_status.setText("✅ [최신 상태] FEA 기하 솔리드 준비 완료")
        self.lbl_stale_status.setStyleSheet("color: #98c379; font-size: 11px; font-weight: bold;")
        self.btn_export_stl.setEnabled(True)
        self.group_vis.setEnabled(True)
        self._update_controls_enabled()
        if hasattr(fea_geometry, "report") and fea_geometry.report:
            self._render_report(fea_geometry.report)

    def _on_worker_success(self, fea_geometry: FeaGeometry):
        self.progress_bar.setVisible(False)
        self.lbl_step_status.setVisible(False)
        self.btn_build_geom.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 4px;
                border-radius: 4px;
                border: 1px solid #3878b2;
            }
            QPushButton:hover {
                background-color: #275284;
            }
        """)

        # StateManager를 통해 전체 시스템(3D 뷰어, 패널 등)에 갱신 전파
        self.state_mgr.set_fea_geometry(fea_geometry)
        self._on_display_toggled()
        self._on_clipping_changed()

    def _on_worker_failed(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.lbl_step_status.setVisible(False)
        self.lbl_stale_status.setText("❌ 기하 생성 오류 발생")
        self.lbl_stale_status.setStyleSheet("color: #e06c75; font-size: 11px; font-weight: bold;")
        self._update_controls_enabled()
        QMessageBox.critical(self, "기하 생성 실패", f"FEA 기하 솔리드 생성 중 오류가 발생했습니다:\n\n{error_msg}")

    def _render_report(self, rep: Dict[str, Any]):
        """분석 보고서를 렌더링하고 이상 수치(비-watertight, 조각 수 이상)를 빨간색으로 강조"""
        c_wt = rep.get("cortical_watertight", False)
        cn_wt = rep.get("cancellous_watertight", False)
        i_wt = rep.get("implant_watertight", False)

        c_wt_html = "<span style='color:#98c379;'>정상 (Watertight)</span>" if c_wt else "<span style='color:#ff5555; font-weight:bold;'>비-watertight (오류)</span>"
        cn_wt_html = "<span style='color:#98c379;'>정상 (Watertight)</span>" if cn_wt else "<span style='color:#ff5555; font-weight:bold;'>비-watertight (오류)</span>"
        i_wt_html = "<span style='color:#98c379;'>정상 (Watertight)</span>" if i_wt else "<span style='color:#ff5555; font-weight:bold;'>비-watertight (오류)</span>"

        c_pc = rep.get("cortical_pieces", 1)
        cn_pc = rep.get("cancellous_pieces", 1)
        c_pc_html = f"{c_pc}개" if c_pc == 1 else f"<span style='color:#ff5555; font-weight:bold;'>{c_pc}개 (조각 분리됨)</span>"
        cn_pc_html = f"{cn_pc}개" if cn_pc == 1 else f"<span style='color:#ff5555; font-weight:bold;'>{cn_pc}개 (조각 분리됨)</span>"

        html = f"""
        <div style='line-height: 140%;'>
            <b style='color:#61afef;'>[체적 분석 (Volume)]</b><br>
            • 피질골: {rep.get('cortical_volume', 0.0):.1f} mm³ ({c_wt_html}, {c_pc_html})<br>
            • 수질골: {rep.get('cancellous_volume', 0.0):.1f} mm³ ({cn_wt_html}, {cn_pc_html})<br>
            • 임플란트: {rep.get('implant_volume', 0.0):.1f} mm³ ({i_wt_html})<br>
            • 원본 대비 제거된 뼈: <b style='color:#e5c07b;'>{rep.get('removed_bone_volume', 0.0):.1f} mm³</b><br>
            <hr style='border: 0; border-top: 1px solid #3c3c46; margin: 3px 0;'>
            <b style='color:#98c379;'>[접촉 면적 (Contact Area, &le;0.05mm)]</b><br>
            • 임플란트-피질골: <b>{rep.get('implant_cortical_contact_area', 0.0):.2f} mm²</b><br>
            • 임플란트-수질골: <b>{rep.get('implant_cancellous_contact_area', 0.0):.2f} mm²</b><br>
            • 총 뼈 접촉 면적: <b style='color:#61afef;'>{rep.get('total_contact_area', 0.0):.2f} mm²</b>
        </div>
        """
        self.text_report.setHtml(html)

    def _on_display_toggled(self):
        vis = {
            "cortical": self.chk_cortical.isChecked(),
            "cancellous": self.chk_cancellous.isChecked(),
            "implant": self.chk_implant.isChecked(),
            "opacity": 0.85
        }
        self.state_mgr.set_fea_visibility(vis)

    def _on_clipping_toggled(self, enabled: bool):
        self._on_clipping_changed()

    def _on_clipping_changed(self):
        axis_map = {0: "Z", 1: "Y", 2: "X"}
        axis = axis_map.get(self.combo_clip_axis.currentIndex(), "Z")
        clip = {
            "enabled": self.chk_clip_enable.isChecked(),
            "axis": axis,
            "value_pct": float(self.slider_clip.value()),
            "inverted": self.chk_clip_invert.isChecked()
        }
        self.state_mgr.set_fea_clipping(clip)

    def export_fea_files(self, out_dir: str):
        """STL 3개 및 manifest.json 파일을 지정된 디렉터리에 내보냅니다."""
        if self._current_geometry is None:
            return False, "내보낼 FEA 기하 데이터가 없습니다."
        try:
            paths = export_stl(self._current_geometry, out_dir)
            manifest_data = {
                "format_version": "1.0",
                "generated_at": datetime.datetime.now().isoformat(),
                "coordinate_system": "LPS",
                "unit": "mm",
                "files": {
                    "cortical_stl": os.path.basename(paths["cortical"]),
                    "cancellous_stl": os.path.basename(paths["cancellous"]),
                    "implant_stl": os.path.basename(paths["implant"]),
                },
                "parameters": {
                    "roi_margin_mm": self.spin_roi_margin.value(),
                    "smooth_iterations": self.spin_smooth.value(),
                    "target_edge_mm": self.spin_target_edge.value(),
                    "undersize_mm": self.spin_undersize.value(),
                },
                "report": self._current_geometry.report
            }
            manifest_path = os.path.join(out_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            return True, f"저장 완료: {out_dir}"
        except Exception as e:
            return False, str(e)

    def _export_stl_and_manifest(self):
        if self._current_geometry is None:
            QMessageBox.warning(self, "내보내기 오류", "내보낼 FEA 기하 데이터가 없습니다.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "STL 및 Manifest 저장 폴더 선택")
        if not out_dir:
            return

        success, msg = self.export_fea_files(out_dir)
        if success:
            QMessageBox.information(
                self,
                "내보내기 완료",
                f"FEA 솔리드 STL 3개 및 manifest.json 저장이 완료되었습니다:\n\n"
                f"• 저장 폴더: {out_dir}\n"
                f"• cortical.stl, cancellous.stl, implant.stl\n"
                f"• manifest.json"
            )
        else:
            QMessageBox.critical(self, "저장 오류", f"파일을 저장하는 중 오류가 발생했습니다:\n{msg}")
