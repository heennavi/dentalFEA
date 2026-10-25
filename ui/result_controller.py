"""
result_controller.py
Controller mediating between StateManager, ResultPanel UI, background QThread workers,
ResultViewController (3D VTK renderer), and ResultProbe (3D peak/nodal probe).
"""

import os
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from dental_fea_app.core.state_manager import StateManager
from fea.models import MeshModel
from fea.results import ResultSet
from postprocess.workers import ResultLoadWorker, PeriImplantRecalcWorker
from postprocess.exporters import (
    export_nodal_csv as exp_nodal_csv,
    export_stats_csv as exp_stats_csv,
    export_stats_json as exp_stats_json,
    save_result_bundle as exp_save_bundle,
    load_result_bundle as exp_load_bundle,
)
from ui.result_panel import ResultPanel
from ui.result_view_controller import ResultViewController
from ui.result_probe import ResultProbe



class ResultController(QObject):
    """
    Controller connecting the ResultPanel UI to post-processing logic and 3D visualization.

    Signals:
        results_loaded (object, dict): Emitted when results are loaded into StateManager.
        results_cleared: Emitted when results are invalidated due to upstream changes.
        load_failed (str): Emitted when loading fails with error message.
    """
    results_loaded = Signal(object, dict)
    results_cleared = Signal()
    load_failed = Signal(str)

    def __init__(
        self,
        state_manager: StateManager,
        panel: Optional[ResultPanel] = None,
        view_controller: Optional[ResultViewController] = None,
        probe: Optional[ResultProbe] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.state_mgr = state_manager
        self.panel = panel
        self.view_controller = view_controller
        self.probe = probe or ResultProbe(self)
        self.open_3d_viewer_callback: Optional[Callable[[], None]] = None

        self._current_worker: Optional[ResultLoadWorker] = None
        self._recalc_worker: Optional[PeriImplantRecalcWorker] = None
        self._last_loaded_frd_path: Optional[str] = None

        if self.view_controller is not None:
            self.probe.attach(self.view_controller)

        # 프로브 이벤트 바인딩
        if self.panel is not None:
            self.probe.probed.connect(self.panel.display_probe_info)
            self.probe.max_marked.connect(self.panel.display_max_marked_info)

        # 패널 시그널 바인딩
        if self.panel is not None:
            self._connect_panel_signals()

        # 전역 상태 변경 바인딩
        self.state_mgr.last_frd_path_changed.connect(self._on_last_frd_path_changed)
        self.state_mgr.fea_stale_changed.connect(self._on_fea_stale_changed)

    def set_view_controller(self, view_controller: ResultViewController) -> None:
        """
        Bind or update active 3D ResultViewController and re-attach probe.
        """
        self.view_controller = view_controller
        if self.view_controller is not None:
            self.probe.attach(self.view_controller)
            # 이미 로드된 결과가 있으면 즉시 3D 시각화 적용
            rs = self.state_mgr.result_set
            model = self.state_mgr.mesh_model
            if rs is not None and model is not None:
                self.view_controller.set_results(
                    model, rs, self.state_mgr.material_assignment
                )

    def _connect_panel_signals(self) -> None:
        """Connect ResultPanel user action signals to controller slots."""
        p = self.panel
        if p is None:
            return

        p.open_frd_requested.connect(self._on_open_frd_requested)
        p.step_changed.connect(self._on_step_changed)
        p.field_changed.connect(self.set_field)
        p.colormap_changed.connect(self.set_colormap)
        p.range_mode_changed.connect(self.set_range_mode)
        p.deform_scale_changed.connect(self.set_deform_scale)
        p.part_visibility_changed.connect(self.set_part_visibility)
        p.part_opacity_changed.connect(self.set_part_opacity)
        p.clip_changed.connect(self.set_clip)
        p.edges_toggled.connect(self.set_edges_visible)
        p.load_arrow_toggled.connect(self.set_load_arrow_visible)
        p.mark_max_requested.connect(self.mark_max_stress)
        p.probe_toggled.connect(self.set_probe_enabled)
        p.peri_radius_changed.connect(self.recalc_peri_implant)
        p.export_csv_requested.connect(self.export_nodal_csv)
        p.export_stats_requested.connect(self.export_stats)
        p.save_screenshot_requested.connect(self.save_screenshot)
        p.save_bundle_requested.connect(self.save_bundle)
        p.load_bundle_requested.connect(self.load_bundle)


    # -------------------------------------------------------------------------
    # 결과 파일 로드 파이프라인
    # -------------------------------------------------------------------------
    def load_frd(
        self,
        path: str,
        step: Optional[int] = -1,
        blocking: bool = False,
    ) -> None:
        """
        Parse and load CalculiX .frd result file using background QThread worker.

        Args:
            path: Absolute or relative path to .frd file.
            step: Increment/step index (-1 for final step).
            blocking: If True, executes worker synchronously (useful for headless tests).
        """
        model = self.state_mgr.mesh_model
        if model is None:
            err_msg = "메쉬 모델이 없습니다. 먼저 메쉬를 생성하거나 해석을 수행하십시오."
            if self.panel is not None:
                self.panel.show_error(err_msg)
            self.load_failed.emit(err_msg)
            return

        materials = self.state_mgr.material_assignment
        bc = self.state_mgr.bc_set
        radius = self.panel.spin_peri_radius.value() if self.panel is not None else 1.0

        if self.panel is not None:
            self.panel.set_loading_state(True, "결과 읽는 중...")

        worker = ResultLoadWorker(
            frd_path=path,
            model=model,
            materials=materials,
            bc=bc,
            step=step,
            radius_mm=radius,
            parent=self,
        )

        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        self._current_worker = worker

        if blocking:
            # 동기 실행 (단위 테스트 및 CLI용)
            worker.run()
            return

        # 비동기 실행
        worker.start()

    def _on_worker_finished(self, rs: ResultSet, stats: dict, frd_path: str) -> None:
        """Handle successful parsing and statistics calculation."""
        self._last_loaded_frd_path = frd_path

        # 1. 전역 상태 갱신
        self.state_mgr.set_result_set(rs)
        self.state_mgr.set_result_stats(stats)
        self.state_mgr.set_result_source_frd(frd_path)

        # 2. 3D 뷰어 갱신
        if self.view_controller is not None:
            self.view_controller.set_results(
                self.state_mgr.mesh_model,
                rs,
                self.state_mgr.material_assignment,
            )

        # 3. 패널 UI 갱신
        if self.panel is not None:
            self.panel.populate_results(rs, stats, self.state_mgr.mesh_model)

        self.results_loaded.emit(rs, stats)

    def _on_worker_error(self, err_msg: str) -> None:
        """Handle error from background worker."""
        if self.panel is not None:
            self.panel.set_loading_state(False)
            self.panel.show_error(err_msg)
        self.load_failed.emit(err_msg)

    def _on_last_frd_path_changed(self, frd_path: str) -> None:
        """CalculiX solver finished handler: auto-load last .frd."""
        if not frd_path:
            return
        # 이미 로드된 동일 파일이 아니라면 자동 로드 실행
        if frd_path != self._last_loaded_frd_path:
            self.load_frd(frd_path, step=-1, blocking=False)

    def _on_fea_stale_changed(self, stale: bool) -> None:
        """Upstream data changed handler: invalidate results."""
        if stale:
            if self.state_mgr.result_set is not None or self.state_mgr.result_stats is not None:
                self.state_mgr.set_result_set(None)
                self.state_mgr.set_result_stats(None)
                self.state_mgr.set_result_source_frd(None)
            if self.view_controller is not None:
                self.view_controller.clear()
            if self.panel is not None:
                self.panel.set_stale_state("결과가 현재 모델과 맞지 않음, 다시 해석 필요")
            self._last_loaded_frd_path = None
            self.results_cleared.emit()

    def _on_open_frd_requested(self) -> None:
        """Prompt file dialog to select external .frd file."""
        if self.panel is None:
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self.panel,
            "CalculiX 결과 파일 (.frd) 선택",
            "",
            "CalculiX Result (*.frd);;All Files (*.*)",
        )
        if filepath:
            self.load_frd(filepath)

    def _on_step_changed(self, step_idx: int) -> None:
        """Reload result fields for selected increment/step."""
        path = self._last_loaded_frd_path or self.state_mgr.result_source_frd
        if path:
            self.load_frd(path, step=step_idx, blocking=False)

    def recalc_peri_implant(self, radius_mm: float) -> None:
        """Recompute peri-implant bone stress statistics asynchronously."""
        rs = self.state_mgr.result_set
        model = self.state_mgr.mesh_model
        if rs is None or model is None:
            return

        worker = PeriImplantRecalcWorker(
            rs=rs,
            model=model,
            radius_mm=radius_mm,
            materials=self.state_mgr.material_assignment,
            parent=self,
        )
        worker.finished.connect(self._on_peri_recalc_finished)
        self._recalc_worker = worker
        worker.start()

    def _on_peri_recalc_finished(self, peri_stats: dict) -> None:
        stats = self.state_mgr.result_stats
        if stats is not None:
            stats["peri_implant"] = peri_stats
            stats["radius_mm"] = self.panel.spin_peri_radius.value() if self.panel else 1.0
            self.state_mgr.set_result_stats(stats)
            if self.panel is not None and self.state_mgr.mesh_model is not None:
                self.panel._update_stats_table(stats, self.state_mgr.mesh_model)

    # -------------------------------------------------------------------------
    # 3D 뷰어 표시 위임 메서드
    # -------------------------------------------------------------------------
    def set_field(self, field_name: str) -> None:
        if self.view_controller is not None:
            self.view_controller.set_field(field_name)

    def set_colormap(self, cmap_name: str) -> None:
        if self.view_controller is not None:
            self.view_controller.set_colormap(cmap_name)

    def set_range_mode(self, mode: str, manual_range: Optional[Tuple[float, float]] = None) -> None:
        if self.view_controller is not None:
            self.view_controller.set_range_mode(mode, manual_range)

    def set_deform_scale(self, scale: float) -> None:
        if self.view_controller is not None:
            self.view_controller.set_deform_scale(scale)

    def set_part_visibility(self, part_name: str, visible: bool) -> None:
        if self.view_controller is not None:
            self.view_controller.set_part_visible(part_name, visible)

    def set_part_opacity(self, part_name: str, opacity: float) -> None:
        if self.view_controller is not None:
            self.view_controller.set_opacity(part_name, opacity)

    def set_clip(self, enabled: bool, axis_str: str, offset_mm: float) -> None:
        if self.view_controller is None:
            return

        if not enabled:
            self.view_controller.set_clip_plane(False)
            return

        # 법선 벡터 계산
        if axis_str == "X":
            normal = (1.0, 0.0, 0.0)
        elif axis_str == "Z":
            normal = (0.0, 0.0, 1.0)
        elif axis_str == "IMPLANT":
            # 임플란트 장축 (Z축 기본 방향)
            normal = (0.0, 0.0, 1.0)
        else:  # "Y"
            normal = (0.0, 1.0, 0.0)

        # 원점 계산: 중심 + normal * offset_mm
        model = self.state_mgr.mesh_model
        if model is not None and model.parts:
            # 임플란트 파트 우선 탐색
            target_part = None
            for p in model.parts:
                if "implant" in p.name.lower() or "fixture" in p.name.lower():
                    target_part = p
                    break
            if target_part is None:
                target_part = model.parts[0]
            center = np.mean(target_part.nodes, axis=0)
            origin = (
                float(center[0] + normal[0] * offset_mm),
                float(center[1] + normal[1] * offset_mm),
                float(center[2] + normal[2] * offset_mm),
            )
        else:
            origin = (0.0, offset_mm, 0.0)

        self.view_controller.set_clip_plane(True, origin=origin, normal=normal)

    def set_edges_visible(self, visible: bool) -> None:
        if self.view_controller is not None:
            self.view_controller.set_edges_visible(visible)

    def set_load_arrow_visible(self, visible: bool) -> None:
        """Toggle occlusal load arrow visibility in the 3D result view."""
        if self.view_controller is not None and hasattr(self.view_controller, "set_load_arrow_visible"):
            self.view_controller.set_load_arrow_visible(visible, self.state_mgr.bc_set)

    def mark_max_stress(self, field: str, part_name: str) -> None:
        if self.view_controller is None or self.view_controller.renderer is None:
            if self.panel is not None:
                self.panel.lbl_probe_info.setText("⚠️ 3D 뷰어가 열려있지 않습니다. 3D 뷰어를 먼저 열어주십시오.")
            return

        if part_name in ("NONE", "none", "보이지 않기", "hide"):
            if hasattr(self.probe, "clear_max_marker"):
                self.probe.clear_max_marker()
            if self.panel is not None:
                self.panel.lbl_probe_info.setText("최대 응력 마커를 숨겼습니다.")
            return

        actual_part = None if part_name == "ALL" else part_name
        if hasattr(self.probe, "mark_maximum"):
            self.probe.mark_maximum(field=field, part_name=actual_part)
        elif hasattr(self.probe, "mark_max_stress"):
            self.probe.mark_max_stress(field=field, part_name=actual_part)

    def set_probe_enabled(self, enabled: bool) -> None:
        if enabled:
            # 1. 3D 뷰어 창이 아직 열리지 않았거나 닫힌 경우 자동 오픈
            if (self.view_controller is None or self.view_controller.renderer is None) and callable(self.open_3d_viewer_callback):
                try:
                    self.open_3d_viewer_callback()
                except Exception:
                    pass

            # 2. 3D 뷰어 창이 존재하면 결과 모드 보장 및 전면 활성화
            if self.view_controller is not None and getattr(self.view_controller, "window", None) is not None:
                win = self.view_controller.window
                try:
                    if hasattr(win, "set_result_view_mode"):
                        win.set_result_view_mode(True)
                    if hasattr(win, "show"):
                        win.show()
                    if hasattr(win, "raise_"):
                        win.raise_()
                    if hasattr(win, "activateWindow"):
                        win.activateWindow()
                except Exception:
                    pass

        if hasattr(self.probe, "enable_click_probe"):
            self.probe.enable_click_probe(enabled)
        elif hasattr(self.probe, "set_probe_enabled"):
            self.probe.set_probe_enabled(enabled)

        if enabled:
            if self.panel is not None:
                if self.view_controller is None or self.view_controller.renderer is None:
                    self.panel.lbl_probe_info.setText("ℹ️ 3D 클릭 프로브 활성화됨 (3D 뷰어 창에서 원하는 메쉬 표면을 클릭하세요)")
                else:
                    self.panel.lbl_probe_info.setText("ℹ️ 3D 클릭 프로브 활성화됨 (3D 뷰어의 메쉬 표면을 클릭하세요)")
        else:
            if self.panel is not None and not self.panel.lbl_probe_info.text().startswith("["):
                self.panel.lbl_probe_info.setText("절점을 클릭하거나 최대 위치를 표시하십시오.")

    # -------------------------------------------------------------------------
    # 내보내기 및 번들 저장/불러오기 메서드 (Step 4-7)
    # -------------------------------------------------------------------------
    def _get_default_save_dir(self) -> str:
        """Get default directory for exports (job folder or saved/)."""
        src = self.state_mgr.last_frd_path or self.state_mgr.result_source_frd
        if src and os.path.exists(os.path.dirname(src)):
            return os.path.dirname(os.path.abspath(src))
        saved_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved"))
        if os.path.exists(saved_dir):
            return saved_dir
        return os.getcwd()

    def export_nodal_csv(self, file_path: Optional[str] = None) -> bool:
        """Export all nodal fields across all parts to CSV."""
        rs = self.state_mgr.result_set
        model = self.state_mgr.mesh_model
        if rs is None or model is None:
            if self.panel is not None:
                self.panel.show_error("내보낼 해석 결과 또는 메쉬 모델이 없습니다.")
            return False

        if not file_path:
            default_path = os.path.join(self._get_default_save_dir(), "nodal_results.csv")
            file_path, _ = QFileDialog.getSaveFileName(
                self.panel,
                "절점 결과 CSV 파일 저장",
                default_path,
                "CSV Files (*.csv);;All Files (*.*)",
            )
        if not file_path:
            return False

        try:
            exp_nodal_csv(file_path, rs, model)
            if self.panel is not None:
                QMessageBox.information(
                    self.panel,
                    "CSV 저장 완료",
                    f"절점 결과 데이터가 성공적으로 저장되었습니다:\n{file_path}",
                )
            return True
        except Exception as e:
            if self.panel is not None:
                self.panel.show_error(f"CSV 저장 실패: {e}")
            return False

    def export_stats(self, file_path: Optional[str] = None) -> bool:
        """Export summary statistics table to CSV or JSON."""
        stats = self.state_mgr.result_stats
        if stats is None:
            if self.panel is not None:
                self.panel.show_error("내보낼 통계 데이터가 없습니다.")
            return False

        if not file_path:
            default_path = os.path.join(self._get_default_save_dir(), "simulation_stats.csv")
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self.panel,
                "통계 표 파일 저장",
                default_path,
                "CSV Files (*.csv);;JSON Files (*.json);;All Files (*.*)",
            )
        if not file_path:
            return False

        try:
            if file_path.lower().endswith(".json"):
                exp_stats_json(file_path, stats)
            else:
                exp_stats_csv(file_path, stats)

            if self.panel is not None:
                QMessageBox.information(
                    self.panel,
                    "통계 저장 완료",
                    f"통계 데이터가 성공적으로 저장되었습니다:\n{file_path}",
                )
            return True
        except Exception as e:
            if self.panel is not None:
                self.panel.show_error(f"통계 저장 실패: {e}")
            return False

    def save_screenshot(self, scale: int = 1, file_path: Optional[str] = None) -> bool:
        """Capture 3D simulation visualization screenshot to PNG."""
        if self.view_controller is None or self.view_controller.renderer is None:
            if self.panel is not None:
                self.panel.show_error("3D 뷰어가 활성화되어 있지 않습니다.")
            return False

        if not file_path:
            field_name = getattr(self.view_controller, "_current_field", "result")
            default_path = os.path.join(self._get_default_save_dir(), f"fea_3d_{field_name}.png")
            file_path, _ = QFileDialog.getSaveFileName(
                self.panel,
                "3D 화면 이미지 저장 (PNG)",
                default_path,
                "PNG Images (*.png);;All Files (*.*)",
            )
        if not file_path:
            return False

        ok = self.view_controller.save_screenshot(file_path, scale=scale)
        if ok:
            if self.panel is not None:
                QMessageBox.information(
                    self.panel,
                    "이미지 저장 완료",
                    f"3D 화면 이미지가 성공적으로 저장되었습니다 (배율: {scale}x):\n{file_path}",
                )
            return True
        else:
            if self.panel is not None:
                self.panel.show_error("3D 화면 캡처 저장에 실패했습니다.")
            return False

    def save_bundle(self, dir_path: Optional[str] = None) -> bool:
        """Save results, grids, statistics, and metadata to a bundle folder."""
        rs = self.state_mgr.result_set
        model = self.state_mgr.mesh_model
        stats = self.state_mgr.result_stats
        if rs is None or model is None:
            if self.panel is not None:
                self.panel.show_error("저장할 해석 결과 또는 메쉬 모델이 없습니다.")
            return False

        if not dir_path:
            default_dir = os.path.join(self._get_default_save_dir(), "fea_bundle")
            dir_path = QFileDialog.getExistingDirectory(
                self.panel,
                "결과 번들을 저장할 폴더 선택",
                default_dir,
            )
        if not dir_path:
            return False

        # 기존 번들 파일 존재 시 덮어쓰기 확인
        npz_check = os.path.join(dir_path, "results.npz")
        if os.path.exists(npz_check) and self.panel is not None:
            reply = QMessageBox.question(
                self.panel,
                "덮어쓰기 확인",
                f"선택한 폴더에 이미 결과 번들이 존재합니다:\n{dir_path}\n\n덮어쓰시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        # 메타데이터 구성
        bc = self.state_mgr.bc_set
        meta = {
            "source_frd": self.state_mgr.result_source_frd or self.state_mgr.last_frd_path or "",
            "active_field": getattr(self.view_controller, "_current_field", "von_mises") if self.view_controller else "von_mises",
        }
        if bc is not None:
            meta["loads_summary"] = [
                {"part": ld.get("part"), "force_N": ld.get("force_N")}
                for ld in getattr(bc, "loads", [])
            ]
            meta["fixed_sets"] = [
                {"part": fx.get("part"), "num_nodes": len(fx.get("node_ids", []))}
                for fx in getattr(bc, "fixed_node_sets", [])
            ]

        try:
            exp_save_bundle(dir_path, model, rs, stats or {}, meta)
            if self.panel is not None:
                QMessageBox.information(
                    self.panel,
                    "결과 번들 저장 완료",
                    f"해석 결과 번들이 성공적으로 저장되었습니다:\n{dir_path}",
                )
            return True
        except Exception as e:
            if self.panel is not None:
                self.panel.show_error(f"결과 번들 저장 실패: {e}")
            return False

    def load_bundle(self, dir_path: Optional[str] = None) -> bool:
        """Load results, statistics, and metadata from a bundle folder."""
        if not dir_path:
            dir_path = QFileDialog.getExistingDirectory(
                self.panel,
                "불러올 결과 번들 폴더 선택",
                self._get_default_save_dir(),
            )
        if not dir_path:
            return False

        try:
            rs, stats, meta = exp_load_bundle(dir_path)

            # 모델과의 정합성 검사 (절점 수)
            cur_model = self.state_mgr.mesh_model
            if cur_model is not None:
                rs_nodes = len(rs.u) if rs.u is not None else 0
                tot_nodes = cur_model.total_nodes()
                if rs_nodes > 0 and rs_nodes != tot_nodes:
                    if self.panel is not None:
                        reply = QMessageBox.warning(
                            self.panel,
                            "모델 불일치 경고",
                            f"불러온 결과의 절점 수({rs_nodes:,}개)가 현재 활성 메쉬 모델의 절점 수({tot_nodes:,}개)와 일치하지 않습니다.\n\n계속 불러오시겠습니까?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No,
                        )

                        if reply != QMessageBox.StandardButton.Yes:
                            return False

            # 상태 갱신
            self._last_loaded_frd_path = dir_path
            self.state_mgr.set_result_set(rs)
            self.state_mgr.set_result_stats(stats)
            self.state_mgr.set_result_source_frd(dir_path)

            if self.view_controller is not None and cur_model is not None:
                self.view_controller.set_results(
                    cur_model, rs, self.state_mgr.material_assignment
                )

            if self.panel is not None and cur_model is not None:
                self.panel.populate_results(rs, stats, cur_model)

            self.results_loaded.emit(rs, stats)

            if self.panel is not None:
                QMessageBox.information(
                    self.panel,
                    "결과 번들 로드 완료",
                    f"해석 결과 번들을 성공적으로 불러왔습니다:\n{dir_path}",
                )
            return True
        except Exception as e:
            if self.panel is not None:
                self.panel.show_error(f"결과 번들 로드 실패: {e}")
            return False

