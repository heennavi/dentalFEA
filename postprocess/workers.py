"""
workers.py
QThread background workers for asynchronous .frd parsing and FEA statistics computation.

Keeps heavy file I/O and numpy/scipy tensor calculations off the UI main thread.
Headless module: Does not import any PySide6 GUI widgets or VTK rendering modules.
"""

from typing import Optional, Dict, Any, List, Tuple
from PySide6.QtCore import QThread, Signal, QObject

from fea.models import MeshModel, MaterialAssignment, BoundaryConditionSet
from fea.results import ResultSet
from postprocess.frd_parser import parse_frd
from postprocess.result_fields import build_result_set, part_statistics, reaction_balance
from postprocess.peri_implant import peri_implant_stats


class ResultLoadWorker(QThread):
    """
    Background worker thread to parse .frd result file, build ResultSet,
    and compute summary statistics and reaction equilibrium.

    Signals:
        started_work: Emitted when thread execution begins.
        progress (int, str): Progress percentage (0~100) and localized status message.
        finished (object, dict, str): Emitted on success with (ResultSet, stats_dict, frd_path).
        error (str): Emitted when parsing or validation fails (e.g. ValueError).
    """
    started_work = Signal()
    progress = Signal(int, str)
    finished = Signal(object, dict, str)
    error = Signal(str)

    def __init__(
        self,
        frd_path: str,
        model: MeshModel,
        materials: Optional[MaterialAssignment] = None,
        bc: Optional[BoundaryConditionSet] = None,
        step: Optional[int] = -1,
        radius_mm: float = 1.0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.frd_path = str(frd_path)
        self.model = model
        self.materials = materials
        self.bc = bc
        self.step = step
        self.radius_mm = float(radius_mm)

    def run(self) -> None:
        """Execute parsing and statistics computation in background thread."""
        try:
            self.started_work.emit()
            self.progress.emit(10, "CalculiX 결과 파일 읽는 중...")

            frd_data = parse_frd(self.frd_path)

            self.progress.emit(45, "응력 및 변위 필드 텐서 변환 중...")
            rs = build_result_set(frd_data, self.model, step=self.step)

            self.progress.emit(75, "파트별 및 임플란트 주변 통계 산출 중...")

            # 1. 파트별 기본 통계 계산
            part_stats_map: Dict[str, Any] = {}
            for part in self.model.parts:
                try:
                    part_stats_map[part.name] = part_statistics(rs, self.model, part.name)
                except Exception:
                    pass

            # 2. 임플란트 주변 뼈 응력 통계 계산
            peri_stats: Optional[Dict[str, Any]] = None
            try:
                peri_stats = peri_implant_stats(
                    rs,
                    self.model,
                    radius_mm=self.radius_mm,
                    materials=self.materials,
                )
            except Exception:
                peri_stats = None

            # 3. 반력 평형 점검
            eq_balance: Optional[Dict[str, Any]] = None
            if self.bc is not None:
                try:
                    eq_balance = reaction_balance(rs, self.bc, self.model)
                except Exception:
                    eq_balance = None

            # 블록 목록 추출 (사용 가능한 DISP/STRESS 스텝 정보)
            block_names: List[str] = [b.name for b in frd_data.blocks]

            # DISP/STRESS 블록에서 개별 스텝/인크리먼트 목록 추출
            step_blocks = [b for b in frd_data.blocks if b.name.strip().upper() == "DISP"]
            if not step_blocks:
                step_blocks = [b for b in frd_data.blocks if b.name.strip().upper() == "STRESS"]

            step_items: List[Tuple[str, int]] = []
            for i, blk in enumerate(step_blocks):
                time_str = f"{blk.time:.3f}s" if blk.time is not None else ""
                label = f"스텝 #{blk.step + 1}" + (f" (t={time_str})" if time_str else "")
                step_items.append((label, i))

            stats: Dict[str, Any] = {
                "parts": part_stats_map,
                "peri_implant": peri_stats,
                "reaction_balance": eq_balance,
                "blocks": block_names,
                "step_items": step_items,
                "num_nodes": len(frd_data.node_ids),
                "step": self.step,
                "radius_mm": self.radius_mm,
            }

            self.progress.emit(100, "결과 로드 및 통계 계산 완료")
            self.finished.emit(rs, stats, self.frd_path)

        except Exception as exc:
            self.error.emit(str(exc))


class PeriImplantRecalcWorker(QThread):
    """
    Lightweight background worker thread to recompute peri-implant bone stress
    statistics when the inspection radius parameter changes in the UI.

    Signals:
        finished (dict): Recomputed peri-implant statistics dictionary.
        error (str): Error message string on failure.
    """
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        rs: ResultSet,
        model: MeshModel,
        radius_mm: float,
        materials: Optional[MaterialAssignment] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.rs = rs
        self.model = model
        self.radius_mm = float(radius_mm)
        self.materials = materials

    def run(self) -> None:
        """Execute peri-implant statistics computation."""
        try:
            peri_stats = peri_implant_stats(
                self.rs,
                self.model,
                radius_mm=self.radius_mm,
                materials=self.materials,
            )
            self.finished.emit(peri_stats)
        except Exception as exc:
            self.error.emit(str(exc))
