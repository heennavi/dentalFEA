"""
segmentation_worker.py
QThread 기반 비동기 피질골/수질골 분할 연산 워커.
대용량 볼륨 연산 중에도 UI 이벤트 루프가 멈추지 않도록 백그라운드 스레드에서 수행합니다.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PySide6.QtCore import QThread, Signal
import vtk
from vtk.util import numpy_support

from dental_fea_app.segmentation.cortical_cancellous import run_segmentation, SegmentationResult


class SegmentationWorker(QThread):
    """
    백그라운드 스레드에서 뼈 및 피질골/수질골 분할 파이프라인을 실행하는 QThread 워커.
    """
    started_work = Signal()
    progress = Signal(int, str)             # (퍼센트 0~100, 상태 메시지)
    finished_work = Signal(object)          # SegmentationResult 인스턴스
    failed = Signal(str)                    # 예외 발생 시 에러 메시지

    def __init__(
        self,
        volume_data: Union[vtk.vtkImageData, np.ndarray],
        spacing: Optional[Tuple[float, float, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        roi_ijk: Optional[Tuple[int, int, int, int, int, int]] = None,
        parent=None
    ):
        super().__init__(parent)
        self.volume_data = volume_data
        self.spacing = spacing
        self.params = params or {}
        self.roi_ijk = roi_ijk

    def run(self):
        try:
            self.started_work.emit()
            self.progress.emit(10, "분할 파라미터 및 볼륨 데이터 준비 중...")

            # vtkImageData인 경우 안전하게 numpy 배열과 spacing을 사전에 추출하여 스레드 안전성 확보
            if isinstance(self.volume_data, vtk.vtkImageData):
                spc = self.volume_data.GetSpacing()
                spacing = (float(spc[0]), float(spc[1]), float(spc[2]))
                nx, ny, nz = self.volume_data.GetDimensions()
                scalars = self.volume_data.GetPointData().GetScalars()
                if scalars is None:
                    raise ValueError("입력 볼륨 데이터에 스칼라가 존재하지 않습니다.")
                hu_1d = numpy_support.vtk_to_numpy(scalars)
                hu = hu_1d.reshape((nz, ny, nx))
            else:
                hu = self.volume_data
                spacing = self.spacing or (1.0, 1.0, 1.0)

            self.progress.emit(30, "뼈(Bone) 영역 추출 및 형태학 필터링 중...")

            # 분할 실행
            result = run_segmentation(
                volume_data=hu,
                spacing=spacing,
                params=self.params,
                roi_ijk=self.roi_ijk
            )

            self.progress.emit(85, "피질골/수질골 마스크 및 통계 생성 중...")
            self.progress.emit(100, "분할 완료")
            self.finished_work.emit(result)

        except Exception as e:
            self.failed.emit(f"분할 실패: {str(e)}")
