"""
fatigue_panel.py
Phase 5 High-Cycle Fatigue Analysis Control Panel (Workflow Tab 7).

Features:
1. Always visible clinical disclaimer and explicit simulation assumptions banner.
2. Material selection (materials_fatigue.yaml) with property inspection card.
3. Cyclic fatigue loading setup (R ratio, cycles, mean stress correction, stress measure).
4. Full fatigue evaluation execution and color map switching (FS, Nf, D, sigma_a, sigma_m).
5. 3D Hotspot marking with 3D callout.
6. Embedded S-N (Wöhler) curve and Haigh diagram.
7. Step 5.3 & Step 5.6 benchmark rerun and numerical validation against analytical/script results.
8. Comprehensive HTML and text report generation.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox, QScrollArea,
    QFileDialog, QMessageBox, QFrame, QSizePolicy, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Signal

from dental_fea_app.core.state_manager import StateManager
from fatigue.models import (
    FatigueMaterial,
    FatigueLoadCase,
    MeanStressCorrectionMethod,
    StressMeasureType,
)
from fatigue.report_generator import (
    generate_fatigue_report_html,
    generate_fatigue_report_text,
    DISCLAIMER_TEXT,
)
from fatigue.cantilever_benchmark import CantileverBenchmarkResult
from fatigue.iso14801 import ISO14801BatchResult
from dental_fea_app.ui.fatigue_diagrams import FatigueDiagramWidget
from dental_fea_app.ui.fatigue_worker import FatigueEvalWorker, BenchmarkWorker


class FatiguePanel(QWidget):
    """
    Control panel widget for High-Cycle Fatigue analysis (Tab 7).
    """
    field_changed_requested = Signal(str)
    hotspot_marked_requested = Signal(dict)
    hotspot_clear_requested = Signal()

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self._materials_db: Dict[str, FatigueMaterial] = {}
        self._current_material: Optional[FatigueMaterial] = None
        self._active_worker: Optional[FatigueEvalWorker] = None
        self._active_bm_worker: Optional[BenchmarkWorker] = None
        self._last_hotspot: Optional[Dict[str, Any]] = None
        self._hotspot_marked: bool = False
        self._last_benchmark_5_3: Optional[CantileverBenchmarkResult] = None
        self._last_benchmark_5_6: Optional[ISO14801BatchResult] = None

        self._load_materials_db()
        self._init_ui()
        self._update_assumption_banner()

        # StateManager 신호 바인딩
        self.state_mgr.fea_stale_changed.connect(self._on_fea_stale_changed)

    def _load_materials_db(self) -> None:
        """Load standard fatigue materials from config/materials_fatigue.yaml."""
        # Project root config path
        base_dir = Path(__file__).resolve().parent.parent.parent
        yaml_path = base_dir / "config" / "materials_fatigue.yaml"
        if not yaml_path.is_file():
            # Fallback to local cwd
            yaml_path = Path("config/materials_fatigue.yaml")

        if yaml_path.is_file():
            import yaml
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                mats = data.get("materials", {})
                for mat_id in mats.keys():
                    try:
                        self._materials_db[mat_id] = FatigueMaterial.from_yaml(yaml_path, mat_id)
                    except Exception:
                        pass
            except Exception:
                pass

        # Fallback default material if yaml load failed
        if not self._materials_db:
            self._materials_db["ti_6al_4v_eli"] = FatigueMaterial(
                name="Ti-6Al-4V ELI (Grade 23)",
                standard="ASTM F136",
                E=110000.0,
                yield_strength_Sy=795.0,
                tensile_strength_Sut=860.0,
                fatigue_limit_Se=510.0,
                sources=["ASTM F136", "MMPDS-01"],
            )

    def _init_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # ---------------------------------------------------------------------
        # 0. 화면 상시 표시 배너 (임상 판단 보조용, 시험 대체 불가 및 가정)
        # ---------------------------------------------------------------------
        self.banner_frame = QFrame(self)
        self.banner_frame.setStyleSheet("""
            QFrame {
                background-color: #3b2816;
                border: 2px solid #e5c07b;
                border-radius: 5px;
                padding: 6px;
            }
        """)
        vbox_banner = QVBoxLayout(self.banner_frame)
        vbox_banner.setContentsMargins(4, 4, 4, 4)
        vbox_banner.setSpacing(4)

        self.lbl_disclaimer = QLabel(
            "⚠️ [주의] 임상 판단 보조용, 시험 대체 불가\n"
            "(For Clinical Decision Support Only - Cannot Replace Physical ISO 14801 Testing)",
            self.banner_frame
        )
        self.lbl_disclaimer.setStyleSheet("color: #e5c07b; font-size: 11px; font-weight: bold;")
        self.lbl_disclaimer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_disclaimer.setWordWrap(True)
        vbox_banner.addWidget(self.lbl_disclaimer)

        self.lbl_assumptions = QLabel("가정: 재료=-- | 척도=-- | 보정=-- | R=0.10 | N=5.0e+06", self.banner_frame)
        self.lbl_assumptions.setStyleSheet("color: #dcdfe4; font-size: 10px; background: #282c34; padding: 4px; border-radius: 3px;")
        self.lbl_assumptions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_assumptions.setWordWrap(True)
        vbox_banner.addWidget(self.lbl_assumptions)

        main_layout.addWidget(self.banner_frame)

        # 스크롤 영역
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        vbox_content = QVBoxLayout(scroll_content)
        vbox_content.setContentsMargins(2, 2, 2, 2)
        vbox_content.setSpacing(8)

        # 1. 피로 해석 조건 및 재료 설정
        vbox_content.addWidget(self._create_params_group())

        # 2. 해석 결과 요약 및 뷰어 컬러 맵 전환
        vbox_content.addWidget(self._create_results_group())

        # 3. S-N 및 Haigh 선도 다이어그램
        vbox_content.addWidget(self._create_diagrams_group())

        # 4. Step 5.3 & Step 5.6 벤치마크 재실행 검증
        vbox_content.addWidget(self._create_benchmark_group())

        # 5. 피로 평가 보고서 생성
        vbox_content.addWidget(self._create_report_group())

        vbox_content.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # 1. 파라미터 제어 그룹
    # -------------------------------------------------------------------------
    def _create_params_group(self) -> QGroupBox:
        group = QGroupBox("1. 피로 재료 및 하중 조건 설정", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # 재료 선택 콤보박스
        row_mat = QHBoxLayout()
        row_mat.addWidget(QLabel("피로 재료:", group))
        self.combo_material = QComboBox(group)
        for mat_id, mat in self._materials_db.items():
            self.combo_material.addItem(f"{mat.name} ({mat.standard})", mat_id)
        self.combo_material.currentIndexChanged.connect(self._on_material_changed)
        row_mat.addWidget(self.combo_material)
        vbox.addLayout(row_mat)

        # 재료 물성 정보 카드 라벨
        self.lbl_mat_info = QLabel("", group)
        self.lbl_mat_info.setStyleSheet("""
            background-color: #21252b;
            color: #abb2bf;
            border: 1px solid #3c4048;
            border-radius: 3px;
            padding: 6px;
            font-size: 10px;
        """)
        self.lbl_mat_info.setWordWrap(True)
        vbox.addWidget(self.lbl_mat_info)

        # 주기 하중비 (Stress Ratio R = sigma_min / sigma_max)
        row_r = QHBoxLayout()
        row_r.addWidget(QLabel("응력비 (R = σ_min/σ_max):", group))
        self.spin_r = QDoubleSpinBox(group)
        self.spin_r.setRange(-1.0, 0.95)
        self.spin_r.setSingleStep(0.05)
        self.spin_r.setValue(0.10)
        self.spin_r.valueChanged.connect(self._update_assumption_banner)
        row_r.addWidget(self.spin_r)
        vbox.addLayout(row_r)

        # 목표 사이클 수 (Target Cycles)
        row_cycles = QHBoxLayout()
        row_cycles.addWidget(QLabel("목표 사이클 (Cycles):", group))
        self.combo_cycles = QComboBox(group)
        self.combo_cycles.addItem("5,000,000 회 (ISO 14801 표준)", 5.0e6)
        self.combo_cycles.addItem("2,000,000 회 (생체 수용액 환경)", 2.0e6)
        self.combo_cycles.addItem("10,000,000 회 (영구 내구한도)", 1.0e7)
        self.combo_cycles.addItem("1,000,000 회 (단기 저작)", 1.0e6)
        self.combo_cycles.currentIndexChanged.connect(self._update_assumption_banner)
        row_cycles.addWidget(self.combo_cycles)
        vbox.addLayout(row_cycles)

        # 평균응력 보정 모델
        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("평균응력 보정:", group))
        self.combo_method = QComboBox(group)
        self.combo_method.addItem("Goodman (치과 임플란트 표준 권장)", MeanStressCorrectionMethod.GOODMAN.value)
        self.combo_method.addItem("Gerber (연성 금속 포물선 모델)", MeanStressCorrectionMethod.GERBER.value)
        self.combo_method.addItem("Soderberg (항복강도 기준 보수적 모델)", MeanStressCorrectionMethod.SODERBERG.value)
        self.combo_method.currentIndexChanged.connect(self._update_assumption_banner)
        row_method.addWidget(self.combo_method)
        vbox.addLayout(row_method)

        # 등가 응력 척도
        row_measure = QHBoxLayout()
        row_measure.addWidget(QLabel("등가 응력 척도:", group))
        self.combo_measure = QComboBox(group)
        self.combo_measure.addItem("Signed von Mises (인장/압축 부호 보존)", StressMeasureType.SIGNED_VON_MISES.value)
        self.combo_measure.addItem("von Mises (표준 등가 응력)", StressMeasureType.VON_MISES.value)
        self.combo_measure.addItem("Max Principal (최대 주응력)", StressMeasureType.MAX_PRINCIPAL.value)
        self.combo_measure.currentIndexChanged.connect(self._update_assumption_banner)
        row_measure.addWidget(self.combo_measure)
        vbox.addLayout(row_measure)

        # 해석 실행 버튼
        self.btn_run_eval = QPushButton("⚡ 피로 해석 실행 (Evaluate Fatigue)", group)
        self.btn_run_eval.setStyleSheet("""
            QPushButton {
                background-color: #2c5282;
                color: #ffffff;
                border: 1px solid #3182ce;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
            QPushButton:disabled {
                background-color: #2d3748;
                color: #718096;
                border-color: #4a5568;
            }
        """)
        self.btn_run_eval.clicked.connect(self.run_fatigue_evaluation)
        vbox.addWidget(self.btn_run_eval)

        # 진행 바
        self.progress_bar = QProgressBar(group)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        vbox.addWidget(self.progress_bar)

        self._on_material_changed()
        return group

    # -------------------------------------------------------------------------
    # 2. 결과 요약 및 컬러 맵 전환 그룹
    # -------------------------------------------------------------------------
    def _create_results_group(self) -> QGroupBox:
        group = QGroupBox("2. 피로 해석 결과 및 뷰어 연계", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # 안전계수 및 수명 요약 라벨
        self.lbl_fs_summary = QLabel("피로 안전계수: 해석 미수행", group)
        self.lbl_fs_summary.setStyleSheet("""
            background-color: #21252b;
            color: #dcdfe4;
            border: 1px solid #3c4048;
            border-radius: 3px;
            padding: 6px;
            font-size: 11px;
            font-weight: bold;
        """)
        self.lbl_fs_summary.setWordWrap(True)
        vbox.addWidget(self.lbl_fs_summary)

        # 핫스팟 요약 라벨
        self.lbl_hotspot_summary = QLabel("취약 핫스팟 부위: --", group)
        self.lbl_hotspot_summary.setStyleSheet("color: #abb2bf; font-size: 11px;")
        self.lbl_hotspot_summary.setWordWrap(True)
        vbox.addWidget(self.lbl_hotspot_summary)

        # 핫스팟 3D 표시 버튼
        self.btn_mark_hotspot = QPushButton("🎯 3D 핫스팟 위치 마커 표시", group)
        self.btn_mark_hotspot.setStyleSheet("""
            QPushButton {
                background-color: #7c2d12;
                color: #ffedd5;
                border: 1px solid #ea580c;
                border-radius: 3px;
                padding: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9a3412;
            }
        """)
        self.btn_mark_hotspot.clicked.connect(self._on_mark_hotspot_clicked)
        self.btn_mark_hotspot.setEnabled(False)
        vbox.addWidget(self.btn_mark_hotspot)

        # 뷰어 컬러 맵 전환 버튼 행
        vbox.addWidget(QLabel("후처리 3D 뷰어 컬러 맵 전환:", group))
        grid_btns = QVBoxLayout()

        row_c1 = QHBoxLayout()
        self.btn_view_fs = QPushButton("안전계수 (FS)", group)
        self.btn_view_fs.clicked.connect(lambda: self._request_fatigue_field("fatigue_fs"))
        self.btn_view_life = QPushButton("피로 수명 (Nf)", group)
        self.btn_view_life.clicked.connect(lambda: self._request_fatigue_field("fatigue_life"))
        row_c1.addWidget(self.btn_view_fs)
        row_c1.addWidget(self.btn_view_life)
        grid_btns.addLayout(row_c1)

        row_c2 = QHBoxLayout()
        self.btn_view_dmg = QPushButton("누적 손상 (D)", group)
        self.btn_view_dmg.clicked.connect(lambda: self._request_fatigue_field("fatigue_damage"))
        self.btn_view_sa = QPushButton("응력 진폭 (σa)", group)
        self.btn_view_sa.clicked.connect(lambda: self._request_fatigue_field("fatigue_sigma_a"))
        self.btn_view_sm = QPushButton("평균 응력 (σm)", group)
        self.btn_view_sm.clicked.connect(lambda: self._request_fatigue_field("fatigue_sigma_m"))
        row_c2.addWidget(self.btn_view_dmg)
        row_c2.addWidget(self.btn_view_sa)
        row_c2.addWidget(self.btn_view_sm)
        grid_btns.addLayout(row_c2)

        vbox.addLayout(grid_btns)
        return group

    # -------------------------------------------------------------------------
    # 3. S-N 및 Haigh 선도 다이어그램 그룹
    # -------------------------------------------------------------------------
    def _create_diagrams_group(self) -> QGroupBox:
        group = QGroupBox("3. 피로 특성 선도 (S-N / Haigh)", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        self.diagram_widget = FatigueDiagramWidget(group)
        vbox.addWidget(self.diagram_widget)
        return group

    # -------------------------------------------------------------------------
    # 4. Step 5.3 & Step 5.6 벤치마크 재실행 검증 그룹
    # -------------------------------------------------------------------------
    def _create_benchmark_group(self) -> QGroupBox:
        group = QGroupBox("4. 규격 및 벤치마크 검증 (Step 5.3 / 5.6)", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        row_bm = QHBoxLayout()
        self.btn_bm_5_3 = QPushButton("5.3 외팔보 벤치마크 재실행", group)
        self.btn_bm_5_3.setToolTip("외팔보 굽힘 해석해(sigma=Mc/I=600MPa, FS=1.0951) 일치 검증")
        self.btn_bm_5_3.clicked.connect(self._run_benchmark_5_3)
        row_bm.addWidget(self.btn_bm_5_3)

        self.btn_bm_5_6 = QPushButton("5.6 ISO 14801 배치 검증", group)
        self.btn_bm_5_6.setToolTip("ISO 14801 표준 기하, 핫스팟 식별, 피로한계 하중 검증")
        self.btn_bm_5_6.clicked.connect(self._run_benchmark_5_6)
        row_bm.addWidget(self.btn_bm_5_6)
        vbox.addLayout(row_bm)

        # 벤치마크 결과 라벨
        self.lbl_bm_status = QLabel("벤치마크 상태: 대기 중", group)
        self.lbl_bm_status.setStyleSheet("color: #abb2bf; font-size: 11px;")
        self.lbl_bm_status.setWordWrap(True)
        vbox.addWidget(self.lbl_bm_status)

        return group

    # -------------------------------------------------------------------------
    # 5. 피로 평가 보고서 생성 그룹
    # -------------------------------------------------------------------------
    def _create_report_group(self) -> QGroupBox:
        group = QGroupBox("5. 피로 평가 보고서", self)
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        row_rep = QHBoxLayout()
        self.btn_export_html = QPushButton("HTML 보고서 저장...", group)
        self.btn_export_html.clicked.connect(self._export_report_html)
        row_rep.addWidget(self.btn_export_html)

        self.btn_export_txt = QPushButton("텍스트 요약 저장...", group)
        self.btn_export_txt.clicked.connect(self._export_report_text)
        row_rep.addWidget(self.btn_export_txt)
        vbox.addLayout(row_rep)

        return group

    # -------------------------------------------------------------------------
    # 이벤트 핸들러 및 슬롯
    # -------------------------------------------------------------------------
    def _on_material_changed(self) -> None:
        mat_id = self.combo_material.currentData()
        self._current_material = self._materials_db.get(mat_id)
        if self._current_material is not None:
            m = self._current_material
            src = ", ".join(m.sources[:2]) if m.sources else "materials_fatigue.yaml"
            self.lbl_mat_info.setText(
                f"• 탄성계수 E: {m.E:.0f} MPa | 포아송비 ν: {m.nu:.2f}\n"
                f"• 항복강도 Sy: {m.yield_strength_Sy:.0f} MPa | 인장강도 Sut: {m.tensile_strength_Sut:.0f} MPa\n"
                f"• 피로한도 Se: {m.fatigue_limit_Se:.0f} MPa (N_e={m.endurance_cycles_Ne:.0e})\n"
                f"• 출처: {src}"
            )
        self._update_assumption_banner()

    def _update_assumption_banner(self) -> None:
        """Dynamically update always-visible assumption text."""
        mat_name = self._current_material.name if self._current_material else "--"
        mat_source = self._current_material.standard if self._current_material else "--"
        r_val = self.spin_r.value()
        n_val = self.combo_cycles.currentData() or 5.0e6
        method = self.combo_method.currentData() or "goodman"
        measure = self.combo_measure.currentData() or "signed_von_mises"

        text = (
            f"가정: 재료={mat_name} ({mat_source}) | 척도={measure} | "
            f"보정={method} | R={r_val:.2f} | N={n_val:.1e}"
        )
        self.lbl_assumptions.setText(text)

    def _on_fea_stale_changed(self, is_stale: bool) -> None:
        if is_stale:
            self.lbl_fs_summary.setText("피로 안전계수: 모델 기하가 변경됨 (재해석 필요)")
            self.btn_mark_hotspot.setEnabled(False)
            if self._hotspot_marked:
                self.hotspot_clear_requested.emit()
                self._hotspot_marked = False
                self.btn_mark_hotspot.setText("🎯 3D 핫스팟 위치 마커 표시")

    def _request_fatigue_field(self, field_name: str) -> None:
        """
        Request 3D postprocess viewer to display the specified fatigue scalar field.
        If fatigue results are not yet calculated, automatically runs evaluation first.
        """
        rs = self.state_mgr.result_set
        if rs is None or getattr(rs, field_name, None) is None:
            if rs is not None and rs.stress6 is not None:
                self.run_fatigue_evaluation(blocking=True)
            else:
                QMessageBox.warning(self, "결과 부재", "유효한 FEA 응력 결과가 없습니다. 먼저 정적 해석을 실행하거나 .frd를 불러오십시오.")
                return

        self.field_changed_requested.emit(field_name)

    def run_fatigue_evaluation(self, blocking: bool = False) -> None:
        """Execute fatigue evaluation on the current active ResultSet in StateManager."""
        rs = self.state_mgr.result_set
        model = self.state_mgr.mesh_model
        if rs is None or model is None or rs.stress6 is None:
            QMessageBox.warning(self, "결과 부재", "유효한 FEA 응력 결과가 없습니다. 먼저 정적 해석을 실행하거나 .frd를 불러오십시오.")
            return

        if self._current_material is None:
            return

        load_case = FatigueLoadCase(
            target_cycles=float(self.combo_cycles.currentData()),
            stress_ratio_R=float(self.spin_r.value()),
            mean_stress_method=MeanStressCorrectionMethod(self.combo_method.currentData()),
            stress_measure=StressMeasureType(self.combo_measure.currentData()),
        )

        if blocking:
            # 동기 실행 (단위 테스트 등에 활용)
            from fatigue.ui_bridge import evaluate_fatigue_on_result_set
            updated_rs, hotspot_info = evaluate_fatigue_on_result_set(
                rs=rs,
                model=model,
                material=self._current_material,
                load_case=load_case,
            )
            self._on_evaluation_finished(updated_rs, hotspot_info)
        else:
            self.btn_run_eval.setEnabled(False)
            self.progress_bar.setVisible(True)
            self._active_worker = FatigueEvalWorker(
                rs=rs,
                model=model,
                material=self._current_material,
                load_case=load_case,
                parent=self,
            )
            self._active_worker.finished.connect(self._on_evaluation_finished)
            self._active_worker.failed.connect(self._on_evaluation_failed)
            self._active_worker.start()

    def _on_evaluation_finished(self, updated_rs: Any, hotspot_info: dict) -> None:
        self.btn_run_eval.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._last_hotspot = hotspot_info

        # StateManager 및 3D 뷰어에 피로 해석 결과 등록 및 VTK 그리드 갱신
        self.state_mgr.set_result_set(updated_rs)
        parent_panel = self.parent()
        if hasattr(parent_panel, "result_controller") and parent_panel.result_controller is not None:
            parent_panel.result_controller.update_results(updated_rs)
        elif self.window() is not None and hasattr(self.window(), "control_panel"):
            cp = self.window().control_panel
            if hasattr(cp, "result_controller") and cp.result_controller is not None:
                cp.result_controller.update_results(updated_rs)

        # 결과 요약 UI 갱신
        min_fs = hotspot_info.get("min_safety_factor", 0.0)
        is_safe = (min_fs >= 1.0)
        status_txt = "안전 (Safe)" if is_safe else "피로 파손 위험 (Failure Risk)"
        color_code = "#98c379" if is_safe else "#e06c75"

        self.lbl_fs_summary.setStyleSheet(f"""
            background-color: #21252b;
            color: {color_code};
            border: 1px solid #3c4048;
            border-radius: 3px;
            padding: 6px;
            font-size: 11px;
            font-weight: bold;
        """)
        self.lbl_fs_summary.setText(
            f"최소 피로 안전계수: {min_fs:.4f} ({status_txt})\n"
            f"기준 피로한도 Se: {hotspot_info.get('fatigue_limit_Se', 510.0):.1f} MPa"
        )

        part = hotspot_info.get("part_name", "")
        node = hotspot_info.get("node_id", -1)
        region = hotspot_info.get("hotspot_region_name", "General")
        self.lbl_hotspot_summary.setText(
            f"취약 핫스팟: {region} ({part}, #{node})\n"
            f"최대 응력: {hotspot_info.get('max_stress_MPa', 0.0):.1f} MPa"
        )
        self.btn_mark_hotspot.setEnabled(True)
        self._hotspot_marked = False
        self.btn_mark_hotspot.setText("🎯 3D 핫스팟 위치 마커 표시")
        self.btn_mark_hotspot.setStyleSheet("""
            QPushButton {
                background-color: #7c2d12;
                color: #ffedd5;
                border: 1px solid #ea580c;
                border-radius: 3px;
                padding: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9a3412;
            }
        """)

        # S-N 및 Haigh 선도 갱신
        if self._current_material is not None:
            local_idx = hotspot_info.get("local_node_id", 0)
            p_slice = updated_rs.slice_for_part(self.state_mgr.mesh_model, part) if part else slice(0, 1)
            sa = float(updated_rs.fatigue_sigma_a[p_slice][local_idx]) if updated_rs.fatigue_sigma_a is not None else 0.0
            sm = float(updated_rs.fatigue_sigma_m[p_slice][local_idx]) if updated_rs.fatigue_sigma_m is not None else 0.0
            life = float(updated_rs.fatigue_life[p_slice][local_idx]) if updated_rs.fatigue_life is not None else float("inf")

            load_case = FatigueLoadCase(
                target_cycles=float(self.combo_cycles.currentData()),
                stress_ratio_R=float(self.spin_r.value()),
                mean_stress_method=MeanStressCorrectionMethod(self.combo_method.currentData()),
                stress_measure=StressMeasureType(self.combo_measure.currentData()),
            )
            self.diagram_widget.update_diagrams(
                material=self._current_material,
                load_case=load_case,
                sigma_a_hotspot=sa,
                sigma_m_hotspot=sm,
                life_hotspot=life,
            )

        # 컨트롤러에 안전계수 컬러 맵 전환 자동 요청
        self.field_changed_requested.emit("fatigue_fs")

    def _on_evaluation_failed(self, err_msg: str) -> None:
        self.btn_run_eval.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "피로 해석 실패", f"피로 계산 중 오류가 발생했습니다:\n{err_msg}")

    def _on_mark_hotspot_clicked(self) -> None:
        if self._hotspot_marked:
            self.hotspot_clear_requested.emit()
            self._hotspot_marked = False
            self.btn_mark_hotspot.setText("🎯 3D 핫스팟 위치 마커 표시")
            self.btn_mark_hotspot.setStyleSheet("""
                QPushButton {
                    background-color: #7c2d12;
                    color: #ffedd5;
                    border: 1px solid #ea580c;
                    border-radius: 3px;
                    padding: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #9a3412;
                }
            """)
        else:
            if self._last_hotspot is not None:
                self.hotspot_marked_requested.emit(self._last_hotspot)
                self._hotspot_marked = True
                self.btn_mark_hotspot.setText("❌ 3D 핫스팟 마커 해제")
                self.btn_mark_hotspot.setStyleSheet("""
                    QPushButton {
                        background-color: #374151;
                        color: #f3f4f6;
                        border: 1px solid #6b7280;
                        border-radius: 3px;
                        padding: 4px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #4b5563;
                    }
                """)


    # -------------------------------------------------------------------------
    # 벤치마크 실행 및 검증 (Step 5.3 & 5.6)
    # -------------------------------------------------------------------------
    def _run_benchmark_5_3(self, blocking: bool = False) -> None:
        self.lbl_bm_status.setText("Step 5.3 외팔보 벤치마크 실행 중...")
        if blocking:
            from fatigue.ui_bridge import run_benchmark_case_5_3
            res = run_benchmark_case_5_3(mesh_size=2.5)
            self._on_benchmark_5_3_finished(res)
        else:
            self.btn_bm_5_3.setEnabled(False)
            self._active_bm_worker = BenchmarkWorker("5.3", mesh_size=2.5, parent=self)
            self._active_bm_worker.finished_5_3.connect(self._on_benchmark_5_3_finished)
            self._active_bm_worker.failed.connect(self._on_benchmark_failed)
            self._active_bm_worker.start()

    def _on_benchmark_5_3_finished(self, result: CantileverBenchmarkResult) -> None:
        self.btn_bm_5_3.setEnabled(True)
        self._last_benchmark_5_3 = result
        pass_str = "PASS (일치 확인)" if result.passed else "FAIL (오차 초과)"
        color = "#98c379" if result.passed else "#e06c75"
        self.lbl_bm_status.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
        self.lbl_bm_status.setText(
            f"[Step 5.3 외팔보 벤치마크: {pass_str}]\n"
            f"• 이론해 sigma_max: {result.theory.sigma_max:.1f} MPa | FEA: {result.fea_sigma_max_mean:.1f} MPa (오차 {result.mean_stress_error_pct:.2f}%)\n"
            f"• 이론해 Goodman FS: {result.theory.fs_goodman:.4f} | FEA: {result.fea_fs_goodman_mean:.4f} (오차 {result.mean_fs_error_pct:.2f}%)"
        )

    def _run_benchmark_5_6(self, blocking: bool = False) -> None:
        self.lbl_bm_status.setText("Step 5.6 ISO 14801 배치 벤치마크 실행 중...")
        if blocking:
            from fatigue.ui_bridge import run_benchmark_case_5_6
            res = run_benchmark_case_5_6()
            self._on_benchmark_5_6_finished(res)
        else:
            self.btn_bm_5_6.setEnabled(False)
            self._active_bm_worker = BenchmarkWorker("5.6", parent=self)
            self._active_bm_worker.finished_5_6.connect(self._on_benchmark_5_6_finished)
            self._active_bm_worker.failed.connect(self._on_benchmark_failed)
            self._active_bm_worker.start()

    def _on_benchmark_5_6_finished(self, result: ISO14801BatchResult) -> None:
        self.btn_bm_5_6.setEnabled(True)
        self._last_benchmark_5_6 = result
        pass_str = "PASS (일치 확인)" if result.passed else "FAIL"
        color = "#98c379" if result.passed else "#e06c75"
        self.lbl_bm_status.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
        self.lbl_bm_status.setText(
            f"[Step 5.6 ISO 14801 벤치마크: {pass_str}]\n"
            f"• 규격 기하 검증: {'합격' if result.validation_report.passed else '불합격'}\n"
            f"• 핫스팟 식별 부위: {result.hotspot.hotspot_region_name} (임상유효: {result.hotspot.is_clinically_valid})\n"
            f"• 예측 피로한계 하중 (FS=1.0): {result.predicted_fatigue_limit_N:.1f} N\n"
            f"• 문헌 교차 검증: {result.benchmark_comparison.status_display_text}"
        )

    def _on_benchmark_failed(self, err_msg: str) -> None:
        self.btn_bm_5_3.setEnabled(True)
        self.btn_bm_5_6.setEnabled(True)
        self.lbl_bm_status.setStyleSheet("color: #e06c75; font-size: 11px;")
        self.lbl_bm_status.setText(f"벤치마크 실행 실패: {err_msg}")

    # -------------------------------------------------------------------------
    # 보고서 내보내기
    # -------------------------------------------------------------------------
    def _export_report_html(self) -> None:
        rs = self.state_mgr.result_set
        if rs is None or rs.fatigue_fs is None:
            QMessageBox.warning(self, "보고서 생성 불가", "피로 해석을 먼저 실행하십시오.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "피로 해석 평가 보고서 저장 (HTML)", "fatigue_report.html", "HTML Files (*.html)"
        )
        if not file_path:
            return

        load_case = FatigueLoadCase(
            target_cycles=float(self.combo_cycles.currentData()),
            stress_ratio_R=float(self.spin_r.value()),
            mean_stress_method=MeanStressCorrectionMethod(self.combo_method.currentData()),
            stress_measure=StressMeasureType(self.combo_measure.currentData()),
        )
        html = generate_fatigue_report_html(
            rs=rs,
            model=self.state_mgr.mesh_model,
            material=self._current_material,
            load_case=load_case,
            hotspot_info=self._last_hotspot,
            benchmark_comp=getattr(self._last_benchmark_5_6, "benchmark_comparison", None),
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
        QMessageBox.information(self, "저장 완료", f"피로 보고서가 성공적으로 저장되었습니다:\n{file_path}")

    def _export_report_text(self) -> None:
        rs = self.state_mgr.result_set
        if rs is None or rs.fatigue_fs is None:
            QMessageBox.warning(self, "보고서 생성 불가", "피로 해석을 먼저 실행하십시오.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "피로 해석 요약 보고서 저장 (Text)", "fatigue_report.txt", "Text Files (*.txt)"
        )
        if not file_path:
            return

        load_case = FatigueLoadCase(
            target_cycles=float(self.combo_cycles.currentData()),
            stress_ratio_R=float(self.spin_r.value()),
            mean_stress_method=MeanStressCorrectionMethod(self.combo_method.currentData()),
            stress_measure=StressMeasureType(self.combo_measure.currentData()),
        )
        txt = generate_fatigue_report_text(
            rs=rs,
            model=self.state_mgr.mesh_model,
            material=self._current_material,
            load_case=load_case,
            hotspot_info=self._last_hotspot,
            benchmark_comp=getattr(self._last_benchmark_5_6, "benchmark_comparison", None),
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(txt)
        QMessageBox.information(self, "저장 완료", f"피로 요약 보고서가 성공적으로 저장되었습니다:\n{file_path}")
