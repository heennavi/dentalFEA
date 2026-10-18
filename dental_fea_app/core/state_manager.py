"""
state_manager.py
전역 데이터 및 뷰 동기화 상태 관리자.
"""

from typing import Tuple, Optional
import numpy as np
from PySide6.QtCore import QObject, Signal
import vtk

from dental_fea_app.config import WL_PRESETS, DEFAULT_PRESET_NAME


class StateManager(QObject):
    """
    영상 데이터 상태, 3D 월드 커서 좌표, 윈도우/레벨 상태를 관리하고
    변경 시 PySide6 Signal로 브로드캐스트하는 싱글톤 스타일 상태 관리자.
    """
    # Signals
    coordinates_changed = Signal(float, float, float)   # (x, y, z) in mm
    window_level_changed = Signal(float, float)         # (window, level)
    volume_loaded = Signal(object)                      # vtkImageData
    status_message = Signal(str, int)                   # (msg, timeout)

    # ROI Signals
    roi_bounds_changed = Signal(object)                 # (xmin, xmax, ymin, ymax, zmin, zmax) or None
    roi_setting_mode_toggled = Signal(bool)             # True: 2D 사각형 그리기 모드, False: 일반 모드
    roi_box_visibility_changed = Signal(bool)           # 3D 파란색 Bounding Box 가시성

    # 2D 인터랙션 모드 ('normal', 'draw', 'measure_length', 'measure_angle')
    interaction_mode_changed = Signal(str)

    # 분할(Segmentation) Signals
    segmentation_result_changed = Signal(object)        # SegmentationResult or None
    segmentation_visibility_changed = Signal(dict)      # {'show_bone': bool, 'show_cortical': bool, 'show_cancellous': bool, 'opacity': float}

    # 임플란트(Implant) Signals
    implant_model_changed = Signal(object)              # ImplantModel or None
    implant_pose_changed = Signal(object)               # ImplantModel or None
    implant_selected = Signal(str)                      # implant_id (선택 해제 시 "")
    implant_transformed = Signal(str)                   # implant_id (2D/3D 기즈모 변환 동기화)
    implants_updated = Signal()                         # 식립된 임플란트 목록 변경(추가/삭제/수정) 브로드캐스트
    render_all_requested = Signal()                     # 전역 렌더링 요청 Signal

    # FEA 기하 처리(Geometry) Signals
    fea_geometry_changed = Signal(object)               # FeaGeometry or None
    fea_visibility_changed = Signal(dict)               # {'cortical': bool, 'cancellous': bool, 'implant': bool, 'opacity': float}
    fea_clipping_changed = Signal(dict)                 # {'enabled': bool, 'axis': str, 'value_pct': float, 'inverted': bool}

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        # 3D 월드 좌표 (mm)
        self._current_x: float = 0.0
        self._current_y: float = 0.0
        self._current_z: float = 0.0

        # Window / Level
        default_preset = WL_PRESETS[DEFAULT_PRESET_NAME]
        self._window: float = default_preset.window
        self._level: float = default_preset.level

        # Volume 데이터
        self._vtk_image: Optional[vtk.vtkImageData] = None
        self._dimensions: Tuple[int, int, int] = (0, 0, 0)
        self._spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._bounds: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._scalar_range: Tuple[float, float] = (-1024.0, 3071.0)
        self._metadata: dict = {}
        self.series_path: Optional[str] = None
        self._roi_bounds: Optional[Tuple[float, float, float, float, float, float]] = None

        # 분할(Segmentation) 상태
        self._segmentation_result = None
        self._seg_visibility = {
            "show_bone": False,
            "show_cortical": True,
            "show_cancellous": True,
            "opacity": 0.45
        }
        self._combined_label_vtk: Optional[vtk.vtkImageData] = None

        # 임플란트(Implant) 상태
        self._implant_model = None
        from dental_fea_app.placement.implant_placement_manager import ImplantPlacementManager
        self.implant_placement_mgr = ImplantPlacementManager()
        self.implant_placement_mgr.set_state_manager(self)
        self.coordinates_changed.connect(lambda x, y, z: self.implant_placement_mgr.update_2d_slice_contours((x, y, z)))

        # FEA 기하(Geometry) 상태
        self._fea_geometry = None
        self._fea_visibility = {
            "cortical": True,
            "cancellous": True,
            "implant": True,
            "opacity": 0.85
        }
        self._fea_clipping = {
            "enabled": False,
            "axis": "Z",
            "value_pct": 50.0,
            "inverted": False
        }

    @property
    def implant_model(self):
        return self._implant_model

    def set_implant_model(self, model):
        """임플란트 모델 인스턴스 등록 및 옵저버 바인딩"""
        if self._implant_model is not None:
            self._implant_model.remove_observer(self._on_implant_pose_updated)

        self._implant_model = model

        if self._implant_model is not None:
            self._implant_model.add_observer(self._on_implant_pose_updated)

        self.implant_model_changed.emit(self._implant_model)
        self.implant_pose_changed.emit(self._implant_model)

    def _on_implant_pose_updated(self, model):
        """ImplantModel 내부 옵저버 콜백 -> Qt Signal 발행"""
        self.implant_pose_changed.emit(model)

    @property
    def segmentation_result(self):
        return self._segmentation_result

    @property
    def segmentation_visibility(self) -> dict:
        return self._seg_visibility

    @property
    def combined_label_vtk(self) -> Optional[vtk.vtkImageData]:
        return self._combined_label_vtk

    def set_segmentation_result(self, result):
        """
        분할 결과를 등록하고 2D MPR 및 3D 표면용 vtkImageData 라벨 볼륨을 생성합니다.
        라벨 매핑: 0: 배경, 1: 피질골(Cortical), 2: 수질골(Cancellous)
        """
        self._segmentation_result = result
        if result is not None and self._vtk_image is not None:
            from dental_fea_app.segmentation.cortical_cancellous import mask_to_vtk_image
            # 라벨 맵 구성: uint8
            nz, ny, nx = result.bone_mask.shape
            label_map = np.zeros((nz, ny, nx), dtype=np.uint8)
            label_map[result.cancellous_mask] = 2
            label_map[result.cortical_mask] = 1

            self._combined_label_vtk = mask_to_vtk_image(
                mask=label_map,
                reference_vtk=self._vtk_image,
                foreground_value=1
            )
        else:
            self._combined_label_vtk = None

        self.segmentation_result_changed.emit(self._segmentation_result)

    def set_segmentation_visibility(self, visibility_dict: dict):
        """
        분할 마스크 가시성 및 불투명도 설정을 갱신하고 신호를 발생합니다.
        """
        self._seg_visibility.update(visibility_dict)
        self.segmentation_visibility_changed.emit(self._seg_visibility)

    @property
    def roi_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        return self._roi_bounds

    def set_roi_bounds(self, bounds: Optional[Tuple[float, float, float, float, float, float]]):
        self._roi_bounds = bounds
        self.roi_bounds_changed.emit(bounds)

    # FEA 기하(Geometry) 프로퍼티 및 제어
    @property
    def fea_geometry(self):
        return self._fea_geometry

    def set_fea_geometry(self, geom):
        self._fea_geometry = geom
        self.fea_geometry_changed.emit(geom)

    @property
    def fea_visibility(self) -> dict:
        return self._fea_visibility

    def set_fea_visibility(self, vis: dict):
        self._fea_visibility.update(vis)
        self.fea_visibility_changed.emit(self._fea_visibility)

    @property
    def fea_clipping(self) -> dict:
        return self._fea_clipping

    def set_fea_clipping(self, clip: dict):
        self._fea_clipping.update(clip)
        self.fea_clipping_changed.emit(self._fea_clipping)


    # --- Property Getters ---
    @property
    def current_coords(self) -> Tuple[float, float, float]:
        return (self._current_x, self._current_y, self._current_z)

    @property
    def window(self) -> float:
        return self._window

    @property
    def level(self) -> float:
        return self._level

    @property
    def vtk_image(self) -> Optional[vtk.vtkImageData]:
        return self._vtk_image

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        return self._bounds

    @property
    def spacing(self) -> Tuple[float, float, float]:
        return self._spacing

    @property
    def origin(self) -> Tuple[float, float, float]:
        return self._origin

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        return self._dimensions

    @property
    def scalar_range(self) -> Tuple[float, float]:
        return self._scalar_range

    @property
    def metadata(self) -> dict:
        return self._metadata

    # --- Coordinates Control ---
    def set_coordinates(self, x: float, y: float, z: float, force: bool = False):
        """
        월드 좌표(mm)를 설정하고 범위 내로 클램핑한 후 변경 신호를 발행합니다.
        """
        if self._vtk_image is not None:
            bx0, bx1, by0, by1, bz0, bz1 = self._bounds
            # 약간의 여유(0.001) 허용 클램핑
            x = float(np.clip(x, bx0, bx1))
            y = float(np.clip(y, by0, by1))
            z = float(np.clip(z, bz0, bz1))

        if not force and np.isclose(self._current_x, x, atol=1e-4) and \
           np.isclose(self._current_y, y, atol=1e-4) and \
           np.isclose(self._current_z, z, atol=1e-4):
            return

        self._current_x = x
        self._current_y = y
        self._current_z = z
        self.coordinates_changed.emit(self._current_x, self._current_y, self._current_z)

    def set_axis_coordinate(self, axis: int, value: float):
        """
        단일 축(0: Sagittal/X, 1: Coronal/Y, 2: Axial/Z)의 좌표를 변경합니다.
        """
        if axis == 0:
            self.set_coordinates(value, self._current_y, self._current_z)
        elif axis == 1:
            self.set_coordinates(self._current_x, value, self._current_z)
        elif axis == 2:
            self.set_coordinates(self._current_x, self._current_y, value)

    # --- Window / Level Control ---
    def set_window_level(self, window: float, level: float, force: bool = False):
        window = max(1.0, float(window))
        level = float(level)

        if not force and np.isclose(self._window, window, atol=1e-2) and \
           np.isclose(self._level, level, atol=1e-2):
            return

        self._window = window
        self._level = level
        self.window_level_changed.emit(self._window, self._level)

    # --- Volume Data Lifecycle ---
    def set_volume_data(self, vtk_image: vtk.vtkImageData, metadata: Optional[dict] = None):
        """
        새로운 VTK 볼륨 데이터를 등록하고 기본 좌표 및 바운드를 갱신합니다.
        """
        self._vtk_image = vtk_image
        self._metadata = metadata or {}
        if self._metadata.get("series_path"):
            self.series_path = self._metadata.get("series_path")
        elif self._metadata.get("filepath"):
            self.series_path = self._metadata.get("filepath")

        self._dimensions = vtk_image.GetDimensions()
        self._spacing = vtk_image.GetSpacing()
        self._origin = vtk_image.GetOrigin()
        self._bounds = vtk_image.GetBounds()

        scalars = vtk_image.GetPointData().GetScalars()
        if scalars:
            self._scalar_range = scalars.GetRange()
        else:
            self._scalar_range = (-1024.0, 3071.0)

        # 초기 커서 좌표를 볼륨 정중앙으로 설정
        center_x = 0.5 * (self._bounds[0] + self._bounds[1])
        center_y = 0.5 * (self._bounds[2] + self._bounds[3])
        center_z = 0.5 * (self._bounds[4] + self._bounds[5])
        self._current_x = center_x
        self._current_y = center_y
        self._current_z = center_z

        self.set_segmentation_result(None)

        self.volume_loaded.emit(self._vtk_image)
        self.coordinates_changed.emit(self._current_x, self._current_y, self._current_z)
        self.window_level_changed.emit(self._window, self._level)

    def get_voxel_value_at(self, x: float, y: float, z: float) -> Optional[float]:
        """
        주어진 3D 월드 좌표(x, y, z)에 위치한 복셀의 스칼라 값(HU)을 반환합니다.
        """
        if self._vtk_image is None:
            return None

        # 월드 좌표 -> 복셀 인덱스 변환
        ix = int(round((x - self._origin[0]) / self._spacing[0]))
        iy = int(round((y - self._origin[1]) / self._spacing[1]))
        iz = int(round((z - self._origin[2]) / self._spacing[2]))

        nx, ny, nz = self._dimensions
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            return float(self._vtk_image.GetScalarComponentAsDouble(ix, iy, iz, 0))
        return None

    def request_render_all(self):
        """2D MPR 및 3D 뷰어 등 모든 활성 뷰포트에 즉각 렌더링을 요청합니다."""
        self.render_all_requested.emit()
