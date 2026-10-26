"""
fatigue_worker.py
QThread background workers for fatigue evaluation and benchmark execution.
Ensures UI responsiveness during nodal evaluations and CalculiX simulation.
"""

from typing import Optional, Dict, Any
from PySide6.QtCore import QThread, Signal

from fea.models import MeshModel
from fea.results import ResultSet
from fatigue.models import FatigueMaterial, FatigueLoadCase
from fatigue.ui_bridge import (
    evaluate_fatigue_on_result_set,
    run_benchmark_case_5_3,
    run_benchmark_case_5_6,
)


class FatigueEvalWorker(QThread):
    """Background worker for evaluating nodal fatigue across a ResultSet."""
    finished = Signal(object, dict)  # (updated_rs, hotspot_info)
    failed = Signal(str)

    def __init__(
        self,
        rs: ResultSet,
        model: MeshModel,
        material: FatigueMaterial,
        load_case: FatigueLoadCase,
        target_part_name: Optional[str] = None,
        parent: Optional[QThread] = None,
    ):
        super().__init__(parent)
        self.rs = rs
        self.model = model
        self.material = material
        self.load_case = load_case
        self.target_part_name = target_part_name

    def run(self) -> None:
        try:
            updated_rs, hotspot_info = evaluate_fatigue_on_result_set(
                rs=self.rs,
                model=self.model,
                material=self.material,
                load_case=self.load_case,
                target_part_name=self.target_part_name,
            )
            self.finished.emit(updated_rs, hotspot_info)
        except Exception as e:
            self.failed.emit(str(e))


class BenchmarkWorker(QThread):
    """Background worker for executing Step 5.3 & Step 5.6 validation benchmarks."""
    finished_5_3 = Signal(object)
    finished_5_6 = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        benchmark_type: str = "5.3",
        mesh_size: float = 2.5,
        ccx_path: Optional[str] = None,
        parent: Optional[QThread] = None,
    ):
        super().__init__(parent)
        self.benchmark_type = benchmark_type
        self.mesh_size = mesh_size
        self.ccx_path = ccx_path

    def run(self) -> None:
        try:
            if self.benchmark_type == "5.3":
                res = run_benchmark_case_5_3(mesh_size=self.mesh_size, ccx_path=self.ccx_path)
                self.finished_5_3.emit(res)
            elif self.benchmark_type == "5.6":
                res = run_benchmark_case_5_6(ccx_path=self.ccx_path)
                self.finished_5_6.emit(res)
            else:
                raise ValueError(f"Unknown benchmark type: {self.benchmark_type}")
        except Exception as e:
            self.failed.emit(str(e))
