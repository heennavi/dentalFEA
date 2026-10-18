"""
mpr_widget.py
단일 직교 단면(Axial, Coronal, Sagittal)을 렌더링하는 MPR VTK 위젯.
- 이중 십자선: 마우스 호버 시 검은색 십자선, 클릭 시 회색 고정 십자선(두께 1)
- Coronal 뷰 십자선 정상 렌더링 보장
- 더블클릭 스핀 회전 차단 및 2D 뷰 잠금
- 90도/-90도 회전, 좌우대칭, 상하대칭 기능
- 실시간 눈금자(Ruler) 오버레이 레이어 연동
"""

import math
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QColor

from dental_fea_app.config import (
    AXIS_AXIAL, AXIS_CORONAL, AXIS_SAGITTAL, AXIS_NAMES
)
from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.core.settings_manager import SettingsManager
from dental_fea_app.ui.mpr_ruler import MPRRuler


class MPRWidget(QWidget):
    """
    단일 단면(Axial, Coronal, Sagittal) MPR 뷰어 위젯.
    """

    def __init__(self, axis: int, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.axis = axis
        self.axis_name = AXIS_NAMES.get(axis, "Axial")
        self.state_mgr = state_manager
        self.settings_mgr = SettingsManager()

        # UI 레이아웃
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # VTK Interactor Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtkWidget)

        # VTK 렌더러 및 윈도우
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.08, 0.09)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # 2D 전용 이미지 인터랙터 스타일 (3D 스핀 회전 차단)
        self.style = vtk.vtkInteractorStyleImage()
        self.vtkWidget.SetInteractorStyle(self.style)

        # 카메라 설정 (2D 병렬 투영)
        self.camera = self.renderer.GetActiveCamera()
        self.camera.ParallelProjectionOn()

        # 이미지 파이프라인
        self.slice_plane = vtk.vtkPlane()
        self.reslice_mapper = vtk.vtkImageResliceMapper()
        self.reslice_mapper.SliceFacesCameraOff()
        self.reslice_mapper.SetSlicePlane(self.slice_plane)

        self.image_property = vtk.vtkImageProperty()
        self.image_property.SetInterpolationTypeToLinear()

        self.image_slice = vtk.vtkImageSlice()
        self.image_slice.SetMapper(self.reslice_mapper)
        self.image_slice.SetProperty(self.image_property)
        self.image_slice.SetVisibility(False)
        self.image_slice.PickableOff()
        self.renderer.AddViewProp(self.image_slice)

        # 십자선 생성 (고정 십자선: 회색, 호버 십자선: 검은색)
        self._init_crosshairs()

        # VTK 2D 네이티브 고정밀 눈금자 (1mm, 5mm, 10mm 틱 및 숫자 라벨)
        self.ruler = MPRRuler(self.renderer, self.axis)

        # ROI 설정 모드 및 사각형 액터 (Orange, 두께 2)
        self._is_roi_setting_mode = False
        self._roi_rect_start: Optional[Tuple[float, float, float]] = None
        self._init_roi_rect_actor()

        # 계측 및 그리기 도구 상태
        self.measure_mode = "none"  # 'none', 'length', 'angle'
        self.measure_points = []
        self.draw_mode_active = False
        self.current_draw_style = {}

        # 마우스 드래그 상태
        self._left_pressed = False
        self._middle_pressed = False
        self._right_pressed = False
        self._last_mouse_pos = QPoint()

        # 회전 및 반전 상태
        self._rotation_angle = 0
        self._flip_h = False
        self._flip_v = False
        self._show_crosshair = True
        self._show_ruler = True

        # 기본 평면 법선 및 방향 초기화
        self._setup_axis_orientation()

        # 저장된 설정 불러오기
        self._load_saved_settings()

        # StateManager 신호 바인딩
        self.state_mgr.coordinates_changed.connect(self.on_coordinates_changed)
        self.state_mgr.window_level_changed.connect(self.on_window_level_changed)
        self.state_mgr.volume_loaded.connect(self.on_volume_loaded)
        self.state_mgr.roi_setting_mode_toggled.connect(self._on_roi_mode_toggled)
        self.state_mgr.interaction_mode_changed.connect(self._on_interaction_mode_changed)
        self.state_mgr.segmentation_result_changed.connect(self._on_segmentation_result_changed)
        self.state_mgr.segmentation_visibility_changed.connect(self._on_segmentation_visibility_changed)
        self.state_mgr.implant_model_changed.connect(self._on_implant_model_changed)
        self.state_mgr.implant_pose_changed.connect(self._on_implant_pose_changed)

        # 2D 마스크 오버레이 파이프라인 초기화
        self._init_mask_overlay()

        # 2D 임플란트 단면 윤곽선(Cutter) 파이프라인 초기화
        self._init_implant_contour()

        # 2D 인터랙티브 Transform 기즈모 초기화 (Axial/Coronal/Sagittal)
        self.view_name = "axial" if axis == AXIS_AXIAL else ("coronal" if axis == AXIS_CORONAL else "sagittal")
        from dental_fea_app.placement.implant_2d_gizmo import Implant2DTransformGizmo
        self.implant_gizmo_2d = Implant2DTransformGizmo(self.view_name, self.renderer, self.state_mgr.implant_placement_mgr)

        # 2D 드로잉 및 계측 오버레이 초기화
        self._init_drawing_and_measurement_actors()

        # 연속 렌더링 스로틀 타이머 (30ms 디바운스)
        self._implant_render_timer = QTimer(self)
        self._implant_render_timer.setSingleShot(True)
        self._implant_render_timer.setInterval(30)
        self._implant_render_timer.timeout.connect(self._do_implant_render)

        # 창 크기 변경 스로틀 타이머 (30ms 디바운스)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(30)
        self._resize_timer.timeout.connect(self._on_resize_timeout)

        # 전역 렌더링 요청 신호 연결
        self.state_mgr.render_all_requested.connect(self.render)

        # 이벤트 필터
        self.vtkWidget.installEventFilter(self)
        self.setMouseTracking(True)
        self.vtkWidget.setMouseTracking(True)


    def _load_saved_settings(self):
        prefix = f"mpr_views.{self.axis_name}"
        self._show_crosshair = self.settings_mgr.get(f"{prefix}.show_crosshair", True)
        self._show_ruler = self.settings_mgr.get(f"{prefix}.show_ruler", True)
        self._rotation_angle = self.settings_mgr.get(f"{prefix}.rotation", 0)
        self._flip_h = self.settings_mgr.get(f"{prefix}.flip_h", False)
        self._flip_v = self.settings_mgr.get(f"{prefix}.flip_v", False)

        self.ruler.set_visibility(self._show_ruler)
        self._update_crosshair_visibility()

    def _setup_axis_orientation(self):
        """
        축(Axial, Coronal, Sagittal)에 따라 표준 슬라이스 평면 법선 및 카메라 ViewUp 설정.
        """
        if self.axis == AXIS_AXIAL:
            # Z-cut, XY plane: Anterior가 위 (+Y), Right가 오른쪽 (+X)
            self.normal = (0.0, 0.0, 1.0)
            self.base_view_up = (0.0, 1.0, 0.0)
        elif self.axis == AXIS_CORONAL:
            # Y-cut, XZ plane: Anterior 정면 응시 (-Y -> +Y), Superior가 위 (+Z), Right가 오른쪽 (+X)
            self.normal = (0.0, -1.0, 0.0)
            self.base_view_up = (0.0, 0.0, 1.0)
        elif self.axis == AXIS_SAGITTAL:
            # X-cut, YZ plane: 환자 우측에서 좌측 응시, Superior가 위 (+Z), Anterior가 오른쪽 (+Y)
            self.normal = (1.0, 0.0, 0.0)
            self.base_view_up = (0.0, 0.0, 1.0)

        self.slice_plane.SetNormal(self.normal)

    def _init_crosshairs(self):
        """
        1) 기준 고정 십자선 (회색 실선, 두께 1)
        2) 마우스 이동 호버 십자선 (검은색 실선, 두께 1)
        """
        # --- 기준 고정 십자선 (회색: Gray 0.65) ---
        gray_col = (0.65, 0.65, 0.65)

        self.line_h_source = vtk.vtkLineSource()
        self.line_h_mapper = vtk.vtkPolyDataMapper()
        self.line_h_mapper.SetInputConnection(self.line_h_source.GetOutputPort())
        self.line_h_actor = vtk.vtkActor()
        self.line_h_actor.SetMapper(self.line_h_mapper)
        self.line_h_actor.GetProperty().SetColor(*gray_col)
        self.line_h_actor.GetProperty().SetLineWidth(1.0)
        self.line_h_actor.GetProperty().LightingOff()
        self.line_h_actor.SetVisibility(False)

        self.line_v_source = vtk.vtkLineSource()
        self.line_v_mapper = vtk.vtkPolyDataMapper()
        self.line_v_mapper.SetInputConnection(self.line_v_source.GetOutputPort())
        self.line_v_actor = vtk.vtkActor()
        self.line_v_actor.SetMapper(self.line_v_mapper)
        self.line_v_actor.GetProperty().SetColor(*gray_col)
        self.line_v_actor.GetProperty().SetLineWidth(1.0)
        self.line_v_actor.GetProperty().LightingOff()
        self.line_v_actor.SetVisibility(False)

        self.renderer.AddActor(self.line_h_actor)
        self.renderer.AddActor(self.line_v_actor)

        # --- 마우스 이동 호버 십자선 (검은색: Black 0.1) ---
        black_col = (0.08, 0.08, 0.08)

        self.hover_h_source = vtk.vtkLineSource()
        self.hover_h_mapper = vtk.vtkPolyDataMapper()
        self.hover_h_mapper.SetInputConnection(self.hover_h_source.GetOutputPort())
        self.hover_h_actor = vtk.vtkActor()
        self.hover_h_actor.SetMapper(self.hover_h_mapper)
        self.hover_h_actor.GetProperty().SetColor(*black_col)
        self.hover_h_actor.GetProperty().SetLineWidth(1.0)
        self.hover_h_actor.GetProperty().LightingOff()
        self.hover_h_actor.SetVisibility(False)

        self.hover_v_source = vtk.vtkLineSource()
        self.hover_v_mapper = vtk.vtkPolyDataMapper()
        self.hover_v_mapper.SetInputConnection(self.hover_v_source.GetOutputPort())
        self.hover_v_actor = vtk.vtkActor()
        self.hover_v_actor.SetMapper(self.hover_v_mapper)
        self.hover_v_actor.GetProperty().SetColor(*black_col)
        self.hover_v_actor.GetProperty().SetLineWidth(1.0)
        self.hover_v_actor.GetProperty().LightingOff()
        self.hover_v_actor.SetVisibility(False)

        self.renderer.AddActor(self.hover_h_actor)
        self.renderer.AddActor(self.hover_v_actor)

    def _init_roi_rect_actor(self):
        """ROI 지정용 주황색(Orange #ffa500, 두께 2) 사각형 액터 초기화"""
        self.roi_rect_points = vtk.vtkPoints()
        self.roi_rect_lines = vtk.vtkCellArray()
        self.roi_rect_poly = vtk.vtkPolyData()
        self.roi_rect_poly.SetPoints(self.roi_rect_points)
        self.roi_rect_poly.SetLines(self.roi_rect_lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.roi_rect_poly)

        self.roi_rect_actor = vtk.vtkActor()
        self.roi_rect_actor.SetMapper(mapper)
        self.roi_rect_actor.GetProperty().SetColor(1.0, 0.647, 0.0)  # Orange
        self.roi_rect_actor.GetProperty().SetLineWidth(2.0)
        self.roi_rect_actor.GetProperty().LightingOff()
        self.roi_rect_actor.SetVisibility(False)
        self.renderer.AddActor(self.roi_rect_actor)

    def _init_mask_overlay(self):
        """2D 분할 마스크(피질골/수질골) 오버레이 파이프라인 초기화"""
        self.mask_lut = vtk.vtkLookupTable()
        self.mask_lut.SetNumberOfTableValues(3)
        self.mask_lut.SetRange(0, 2)
        # 0: 배경 (완전 투명)
        self.mask_lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
        # 1: 피질골 (Cortical: 빨강 1.0, 0.2, 0.2, alpha)
        self.mask_lut.SetTableValue(1, 1.0, 0.2, 0.2, 0.35)
        # 2: 수질골 (Cancellous: 파랑 0.2, 0.6, 1.0, alpha)
        self.mask_lut.SetTableValue(2, 0.2, 0.6, 1.0, 0.35)
        self.mask_lut.Build()

        self.mask_reslice_mapper = vtk.vtkImageResliceMapper()
        self.mask_reslice_mapper.SliceFacesCameraOff()
        self.mask_reslice_mapper.SetSlicePlane(self.slice_plane)  # 원본 영상 슬라이스 평면 공유

        self.mask_image_property = vtk.vtkImageProperty()
        self.mask_image_property.SetLookupTable(self.mask_lut)
        self.mask_image_property.UseLookupTableScalarRangeOn()
        self.mask_image_property.SetInterpolationTypeToNearest()  # 마스크 경계 정밀 표시

        self.mask_image_slice = vtk.vtkImageSlice()
        self.mask_image_slice.SetMapper(self.mask_reslice_mapper)
        self.mask_image_slice.SetProperty(self.mask_image_property)
        self.mask_image_slice.SetVisibility(False)
        self.mask_image_slice.PickableOff()

        self.renderer.AddViewProp(self.mask_image_slice)

    def _on_segmentation_result_changed(self, result):
        """분할 결과가 갱신되었을 때 마스크 리슬라이서에 볼륨 연결"""
        if self.state_mgr.combined_label_vtk is not None:
            self.mask_reslice_mapper.SetInputData(self.state_mgr.combined_label_vtk)
            self._apply_mask_visibility()
        else:
            self.mask_image_slice.SetVisibility(False)
        self.render()

    def _on_segmentation_visibility_changed(self, vis: dict):
        """피질골/수질골 가시성 및 불투명도 설정 반영"""
        self._apply_mask_visibility()
        self.render()

    def _apply_mask_visibility(self):
        if self.state_mgr.combined_label_vtk is None or self.state_mgr.segmentation_result is None:
            self.mask_image_slice.SetVisibility(False)
            return

        vis = self.state_mgr.segmentation_visibility
        opacity = vis.get("opacity", 0.35)
        show_cortical = vis.get("show_cortical", True)
        show_cancellous = vis.get("show_cancellous", True)

        alpha_c = opacity if show_cortical else 0.0
        alpha_m = opacity if show_cancellous else 0.0

        self.mask_lut.SetTableValue(1, 1.0, 0.2, 0.2, alpha_c)
        self.mask_lut.SetTableValue(2, 0.2, 0.6, 1.0, alpha_m)
        self.mask_lut.Modified()

        has_any = (alpha_c > 0.0 or alpha_m > 0.0)
        self.mask_image_slice.SetVisibility(has_any)

    def _init_implant_contour(self):
        """
        2D MPR 단면 상에 임플란트 단면 윤곽선(Cutter)을 노란색 실선(두께 2)으로 표시하는 파이프라인.
        """
        self.implant_cut_plane = vtk.vtkPlane()
        self.implant_cut_plane.SetNormal(self.normal)
        self.implant_cut_plane.SetOrigin(0.0, 0.0, 0.0)

        self.implant_cutter = vtk.vtkCutter()
        self.implant_cutter.SetCutFunction(self.implant_cut_plane)
        self.implant_cutter.GenerateCutScalarsOff()

        # 영상 위에 선명하게 보이도록 약간 앞쪽(카메라 방향)으로 오프셋을 주기 위한 Transform
        self.implant_contour_transform = vtk.vtkTransform()
        self.implant_contour_tf_filter = getattr(vtk, "vtkTransformFilter", vtk.vtkTransformPolyDataFilter)()
        self.implant_contour_tf_filter.SetInputConnection(self.implant_cutter.GetOutputPort())
        self.implant_contour_tf_filter.SetTransform(self.implant_contour_transform)

        self.implant_contour_mapper = vtk.vtkPolyDataMapper()
        self.implant_contour_mapper.SetInputConnection(self.implant_contour_tf_filter.GetOutputPort())
        self.implant_contour_mapper.ScalarVisibilityOff()

        self.implant_contour_actor = vtk.vtkActor()
        self.implant_contour_actor.SetMapper(self.implant_contour_mapper)
        # 눈에 띄는 밝은 노란색 (#FFE066)
        self.implant_contour_actor.GetProperty().SetColor(1.0, 0.88, 0.2)
        self.implant_contour_actor.GetProperty().SetLineWidth(2.0)
        self.implant_contour_actor.GetProperty().LightingOff()
        self.implant_contour_actor.SetVisibility(False)

        self.renderer.AddActor(self.implant_contour_actor)

    def _on_implant_model_changed(self, model):
        """임플란트 모델 객체 변경 시 단면 윤곽선 갱신"""
        self._request_implant_update()

    def _on_implant_pose_changed(self, model):
        """임플란트 포즈 변경 시 단면 윤곽선 갱신"""
        self._request_implant_update()

    def _request_implant_update(self):
        if not self._implant_render_timer.isActive():
            self._implant_render_timer.start()

    def _do_implant_render(self):
        self._update_implant_contour()
        self.render()

    def _update_implant_contour(self):
        """현재 슬라이스 평면과 임플란트 메쉬의 교차 단면 윤곽선을 계산하고 액터 위치를 갱신합니다."""
        model = self.state_mgr.implant_model
        if model is None or model.raw_polydata is None or self.state_mgr.vtk_image is None:
            self.implant_contour_actor.SetVisibility(False)
            return

        # 1. 커터 평면 위치를 현재 슬라이스 평면과 일치
        x, y, z = self.state_mgr.current_coords
        self.implant_cut_plane.SetNormal(self.normal)
        self.implant_cut_plane.SetOrigin(x, y, z)

        # 2. 임플란트 월드 변환 폴리데이터를 커터 입력으로 연결
        poly_w = model.get_transformed_polydata()
        self.implant_cutter.SetInputData(poly_w)
        self.implant_cutter.Update()

        cut_output = self.implant_cutter.GetOutput()
        has_lines = (cut_output is not None and cut_output.GetNumberOfPoints() > 0)
        self.implant_contour_actor.SetVisibility(has_lines)

        if has_lines:
            # 영상 위에 겹쳐 보이도록 카메라 방향으로 약간 앞쪽(0.25mm) 오프셋 적용
            fh = -1.0 if self._flip_h else 1.0
            fv = -1.0 if self._flip_v else 1.0
            norm_sign = fh * fv
            offset_dist = 0.25 * norm_sign

            dx = self.normal[0] * offset_dist
            dy = self.normal[1] * offset_dist
            dz = self.normal[2] * offset_dist

            self.implant_contour_transform.Identity()
            self.implant_contour_transform.Translate(dx, dy, dz)
            self.implant_contour_tf_filter.Update()

    def _init_drawing_and_measurement_actors(self):
        """임시 드래그 미리보기 및 계측 렌더링용 액터 풀 초기화"""
        # 1. 임시 미리보기 선분/도형
        self.temp_points = vtk.vtkPoints()
        self.temp_lines = vtk.vtkCellArray()
        self.temp_poly = vtk.vtkPolyData()
        self.temp_poly.SetPoints(self.temp_points)
        self.temp_poly.SetLines(self.temp_lines)

        temp_mapper = vtk.vtkPolyDataMapper()
        temp_mapper.SetInputData(self.temp_poly)
        self.temp_actor = vtk.vtkActor()
        self.temp_actor.SetMapper(temp_mapper)
        self.temp_actor.GetProperty().SetColor(1.0, 0.3, 0.3)
        self.temp_actor.GetProperty().SetLineWidth(2.0)
        self.temp_actor.GetProperty().LightingOff()
        self.temp_actor.SetVisibility(False)
        self.renderer.AddActor(self.temp_actor)

        # 2. 임시 텍스트 라벨 (길이 mm 또는 각도 deg)
        self.temp_text_actor = vtk.vtkTextActor()
        t_prop = self.temp_text_actor.GetTextProperty()
        t_prop.SetFontSize(12)
        t_prop.SetColor(0.2, 0.8, 1.0)
        t_prop.BoldOn()
        self.temp_text_actor.SetVisibility(False)
        self.renderer.AddActor(self.temp_text_actor)

        # 3. 확정된 도형 및 계측 목록 및 액터 보관
        self.drawn_shapes = []       # Dict list
        self.measure_items = []      # Dict list
        self.interaction_mode = "normal"  # 'normal', 'draw', 'draw_select', 'measure_length', 'measure_angle'
        self._temp_angle_pts = []    # 각도 측정을 위한 3개 점 버퍼
        self.draw_start_pt = None
        self.draw_pts = []           # 프리폼 등 연속 점 버퍼
        self.measure_start_pt = None
        self.measure_dialog = None
        self.draw_dialog = None
        self._shape_actors = []      # 렌더러에 추가된 확정 도형 액터들
        self._measure_actors = []    # 렌더러에 추가된 확정 계측 액터들 (선분 + 텍스트)

        # 가시성 및 하이라이트 제어
        self.show_measurements = True
        self.show_drawn_shapes = True
        self.selected_shape_idx = -1
        self.selected_measure_idx = -1

    def set_measurements_visible(self, visible: bool):
        self.show_measurements = visible
        self._update_overlays()
        self.render()

    def set_drawn_shapes_visible(self, visible: bool):
        self.show_drawn_shapes = visible
        self._update_overlays()
        self.render()

    def set_selected_shape(self, idx: int):
        self.selected_shape_idx = idx
        self._update_overlays()
        self.render()

    def set_selected_measure(self, idx: int):
        self.selected_measure_idx = idx
        self._update_overlays()
        self.render()

    def _on_interaction_mode_changed(self, mode: str):
        self.interaction_mode = mode
        self._temp_angle_pts.clear()
        self.draw_pts.clear()
        self.temp_actor.SetVisibility(False)
        self.temp_text_actor.SetVisibility(False)
        if mode != "normal":
            # 그리기나 측정 중일 때는 호버 십자선 비활성화
            self.hover_h_actor.SetVisibility(False)
            self.hover_v_actor.SetVisibility(False)
        self.render()

    def _on_roi_mode_toggled(self, enabled: bool):

        self._is_roi_setting_mode = enabled
        self._roi_rect_start = None
        if not enabled:
            # 설정 모드가 끝나면 그려진 2D 사각형은 사라짐
            self.roi_rect_actor.SetVisibility(False)
            self.render()

    def on_volume_loaded(self, vtk_image: vtk.vtkImageData):

        self.reslice_mapper.SetInputData(vtk_image)
        self.image_slice.SetVisibility(True)
        self._update_crosshair_visibility()
        self.reset_camera()
        self.update_slice()
        self.render()

    def reset_camera(self):
        """카메라를 초기화하고 회전/대칭 상태를 적용합니다."""
        bounds = self.state_mgr.bounds
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])

        dist = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2.0
        if dist <= 0:
            dist = 500.0

        focal_point = [cx, cy, cz]
        cam_pos = [
            cx + self.normal[0] * dist,
            cy + self.normal[1] * dist,
            cz + self.normal[2] * dist
        ]

        self.camera.SetFocalPoint(focal_point)
        self.camera.SetPosition(cam_pos)
        self.camera.SetViewUp(self.base_view_up)

        # 평행 투영 스케일
        if self.axis == AXIS_AXIAL:
            self.camera.SetParallelScale(0.55 * (bounds[3] - bounds[2]))
        elif self.axis == AXIS_CORONAL:
            self.camera.SetParallelScale(0.55 * (bounds[5] - bounds[4]))
        elif self.axis == AXIS_SAGITTAL:
            self.camera.SetParallelScale(0.55 * (bounds[5] - bounds[4]))

        # 회전/대칭 적용
        self._apply_transforms()
        self.update_slice()
        self.render()

    def _apply_transforms(self):
        """
        현재 회전각 및 상하/좌우 대칭 상태를 카메라에 반영.
        - 수평 대칭: fh = -1 if flip_h else +1
        - 수직 대칭: fv = -1 if flip_v else +1
        - 카메라 시선 방향 배수: dir_sign = fh * fv
        - 카메라 상향 벡터 배수: up_sign = fv
        이를 통해 상하 대칭 시 180도 회전(Roll) 없이 순수한 수직 대칭만 수행됨.
        """
        bounds = self.state_mgr.bounds
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])
        dist = 1000.0

        fh = -1.0 if self._flip_h else 1.0
        fv = -1.0 if self._flip_v else 1.0
        dir_sign = fh * fv
        up_sign = fv

        cam_pos = [
            cx + self.normal[0] * dist * dir_sign,
            cy + self.normal[1] * dist * dir_sign,
            cz + self.normal[2] * dist * dir_sign
        ]
        self.camera.SetFocalPoint(cx, cy, cz)
        self.camera.SetPosition(cam_pos)

        cam_up = [u * up_sign for u in self.base_view_up]
        self.camera.SetViewUp(cam_up)

        # 회전각 (Roll): 반대편 시선일 경우 시각적 회전 방향 일치
        rot = -self._rotation_angle if (dir_sign < 0) else self._rotation_angle
        if rot != 0:
            self.camera.Roll(rot)

        # 클리핑 레인지 안전화 (슬라이스 잘림 방지)
        self.camera.SetClippingRange(dist * 0.1, dist * 3.0)
        self.renderer.ResetCameraClippingRange()

        self._update_ruler()
        self.render()

    def _update_ruler(self):
        """현재 뷰포트 크기와 월드 범위에 맞춰 눈금자(Ruler) 업데이트"""
        if self.state_mgr.vtk_image is None or not self._show_ruler:
            self.ruler.set_visibility(False)
            return
        self.ruler.set_visibility(True)
        win_w = self.vtkWidget.width()
        win_h = self.vtkWidget.height()
        self.ruler.update(
            self.state_mgr.bounds,
            self.state_mgr.current_coords,
            win_w, win_h
        )

    # --- 회전 및 대칭 제어 메서드 ---
    def rotate_cw_90(self):
        """시계 방향 90도 회전"""
        self._rotation_angle = (self._rotation_angle + 90) % 360
        self._save_transform_settings()
        self._apply_transforms()

    def rotate_ccw_90(self):
        """반시계 방향 90도 (-90도) 회전"""
        self._rotation_angle = (self._rotation_angle - 90) % 360
        self._save_transform_settings()
        self._apply_transforms()

    def reset_view_transforms(self):
        """회전 및 대칭(flip)을 원래 로드 시의 초기 상태(0도, flip 없음)로 되돌리고 뷰에 즉시 반영"""
        self._rotation_angle = 0
        self._flip_h = False
        self._flip_v = False
        self._save_transform_settings()
        self._apply_transforms()

    def flip_vertical(self):
        """상하 대칭 토글 (독립적 동작)"""
        self._flip_v = not self._flip_v
        self._save_transform_settings()
        self._apply_transforms()

    def flip_horizontal(self):
        """좌우 대칭 토글 (독립적 동작)"""
        self._flip_h = not self._flip_h
        self._save_transform_settings()
        self._apply_transforms()

    def _save_transform_settings(self):
        prefix = f"mpr_views.{self.axis_name}"
        self.settings_mgr.set(f"{prefix}.rotation", self._rotation_angle, auto_save=False)
        self.settings_mgr.set(f"{prefix}.flip_v", self._flip_v, auto_save=False)
        self.settings_mgr.set(f"{prefix}.flip_h", self._flip_h, auto_save=True)

    # --- 십자선 & 눈금자 가시성 제어 ---
    def set_crosshair_visible(self, visible: bool):
        self._show_crosshair = visible
        self.settings_mgr.set(f"mpr_views.{self.axis_name}.show_crosshair", visible)
        self._update_crosshair_visibility()
        self.render()

    def is_crosshair_visible(self) -> bool:
        return self._show_crosshair

    def set_ruler_visible(self, visible: bool):
        self._show_ruler = visible
        self.settings_mgr.set(f"mpr_views.{self.axis_name}.show_ruler", visible)
        self.ruler.set_visibility(visible)
        self.render()

    def is_ruler_visible(self) -> bool:
        return self._show_ruler

    def _update_crosshair_visibility(self):
        has_volume = self.state_mgr.vtk_image is not None
        show = self._show_crosshair and has_volume
        self.line_h_actor.SetVisibility(show)
        self.line_v_actor.SetVisibility(show)
        if not show:
            self.hover_h_actor.SetVisibility(False)
            self.hover_v_actor.SetVisibility(False)

    def on_coordinates_changed(self, x: float, y: float, z: float):
        self.update_slice()
        self.render()

    def on_window_level_changed(self, window: float, level: float):
        self.image_property.SetColorWindow(window)
        self.image_property.SetColorLevel(level)
        self.render()

    def reset_view_state(self):
        """회전각, 대칭, 십자선, 눈금자를 초기화하고 카메라를 리셋합니다."""
        self._rotation_angle = 0
        self._flip_h = False
        self._flip_v = False
        self._show_crosshair = True
        self._show_ruler = True
        self.ruler.set_visibility(True)
        self._update_crosshair_visibility()
        self._save_transform_settings()
        self.reset_camera()

    def update_slice(self):
        """
        슬라이스 평면과 고정 십자선(회색) 좌표를 재계산합니다.
        카메라 방향(flip_h/flip_v)에 맞춰 전면 오프셋을 자동 설정합니다.
        """
        x, y, z = self.state_mgr.current_coords
        bx0, bx1, by0, by1, bz0, bz1 = self.state_mgr.bounds

        self.slice_plane.SetOrigin(x, y, z)

        focal = list(self.camera.GetFocalPoint())
        pos = list(self.camera.GetPosition())
        dist = 1000.0

        fh = -1.0 if self._flip_h else 1.0
        fv = -1.0 if self._flip_v else 1.0
        dir_sign = fh * fv
        norm_sign = dir_sign
        offset = 0.25 * norm_sign

        if self.axis == AXIS_AXIAL:
            focal[2] = z
            pos[2] = z + dist * norm_sign
            zo = z + offset
            self.line_h_source.SetPoint1(bx0, y, zo)
            self.line_h_source.SetPoint2(bx1, y, zo)
            self.line_v_source.SetPoint1(x, by0, zo)
            self.line_v_source.SetPoint2(x, by1, zo)

        elif self.axis == AXIS_CORONAL:
            focal[1] = y
            pos[1] = y - dist * norm_sign
            yo = y - offset
            self.line_h_source.SetPoint1(bx0, yo, z)
            self.line_h_source.SetPoint2(bx1, yo, z)
            self.line_v_source.SetPoint1(x, yo, bz0)
            self.line_v_source.SetPoint2(x, yo, bz1)

        elif self.axis == AXIS_SAGITTAL:
            focal[0] = x
            pos[0] = x + dist * norm_sign
            xo = x + offset
            self.line_h_source.SetPoint1(xo, by0, z)
            self.line_h_source.SetPoint2(xo, by1, z)
            self.line_v_source.SetPoint1(xo, y, bz0)
            self.line_v_source.SetPoint2(xo, y, bz1)

        self.camera.SetFocalPoint(focal)
        self.camera.SetPosition(pos)
        self.camera.SetClippingRange(dist * 0.1, dist * 3.0)
        self._update_overlays()
        self._update_implant_contour()
        if hasattr(self, "implant_gizmo_2d") and self.implant_gizmo_2d.is_attached:
            self.implant_gizmo_2d.update_geometry((x, y, z))

    def sync_implant_gizmo(self):
        """선택된 임플란트에 맞춰 2D 기즈모를 연결하거나 분리합니다."""
        if not hasattr(self, "implant_gizmo_2d"):
            return
        selected_imp = self.state_mgr.implant_placement_mgr.get_selected_implant()
        if selected_imp and selected_imp.is_visible and self.state_mgr.implant_placement_mgr.is_visible_all:
            self.implant_gizmo_2d.attach_implant(selected_imp, self.state_mgr.current_coords)
            self.implant_gizmo_2d.update_geometry(self.state_mgr.current_coords)
        else:
            self.implant_gizmo_2d.detach()
        self.render()

    def _update_hover_crosshair(self, wx: float, wy: float, wz: float):
        """마우스 이동에 따른 실시간 호버 십자선 (검은색) 업데이트"""
        if not self._show_crosshair or self.state_mgr.vtk_image is None:
            return

        bx0, bx1, by0, by1, bz0, bz1 = self.state_mgr.bounds
        fh = -1.0 if self._flip_h else 1.0
        fv = -1.0 if self._flip_v else 1.0
        norm_sign = fh * fv
        offset = 0.35 * norm_sign

        if self.axis == AXIS_AXIAL:
            zo = wz + offset
            self.hover_h_source.SetPoint1(bx0, wy, zo)
            self.hover_h_source.SetPoint2(bx1, wy, zo)
            self.hover_v_source.SetPoint1(wx, by0, zo)
            self.hover_v_source.SetPoint2(wx, by1, zo)
        elif self.axis == AXIS_CORONAL:
            yo = wy - offset
            self.hover_h_source.SetPoint1(bx0, yo, wz)
            self.hover_h_source.SetPoint2(bx1, yo, wz)
            self.hover_v_source.SetPoint1(wx, yo, bz0)
            self.hover_v_source.SetPoint2(wx, yo, bz1)
        elif self.axis == AXIS_SAGITTAL:
            xo = wx + offset
            self.hover_h_source.SetPoint1(xo, by0, wz)
            self.hover_h_source.SetPoint2(xo, by1, wz)
            self.hover_v_source.SetPoint1(xo, wy, bz0)
            self.hover_v_source.SetPoint2(xo, wy, bz1)

        self.hover_h_actor.SetVisibility(True)
        self.hover_v_actor.SetVisibility(True)
        self.render()

    def render(self):
        if not self.isVisible():
            return
        if self.state_mgr.vtk_image is not None:
            self._update_ruler()
        if self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()

    def _pan_camera(self, dx: float, dy: float):
        """마우스 휠(가운데 버튼) 드래그에 의한 2D 이미지 평행 이동(Pan)"""
        if self.state_mgr.vtk_image is None:
            return
        win_h = max(1, self.vtkWidget.height())
        scale = 2.0 * self.camera.GetParallelScale() / float(win_h)

        v_up = np.array(self.camera.GetViewUp(), dtype=float)
        v_dir = np.array(self.camera.GetDirectionOfProjection(), dtype=float)
        # 카메라 시선 방향과 상향 벡터의 외적 = 오른쪽 방향 벡터
        v_right = np.cross(v_dir, v_up)
        norm_r = np.linalg.norm(v_right)
        if norm_r > 0:
            v_right /= norm_r
        norm_u = np.linalg.norm(v_up)
        if norm_u > 0:
            v_up /= norm_u

        # 마우스 드래그 방향으로 이미지가 이동하도록 카메라 시점 조정 (마우스와 동일 방향 이동)
        shift = (-dx * scale) * v_right + (dy * scale) * v_up

        focal = np.array(self.camera.GetFocalPoint()) + shift
        pos = np.array(self.camera.GetPosition()) + shift
        self.camera.SetFocalPoint(focal.tolist())
        self.camera.SetPosition(pos.tolist())
        self.render()

    def showEvent(self, event):
        super().showEvent(event)
        try:
            if hasattr(self.vtkWidget, "_Iren") and self.vtkWidget._Iren:
                self.vtkWidget._Iren.Initialize()
        except Exception:
            pass
        self.render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.state_mgr.vtk_image is not None:
            self._resize_timer.start()

    def _on_resize_timeout(self):
        if self.state_mgr.vtk_image is not None and self.isVisible():
            self.render()

    # --- 마우스 및 이벤트 필터 ---
    def eventFilter(self, obj, event):
        if obj == self.vtkWidget:
            etype = event.type()

            # 더블클릭 스핀 회전 기능 완전 차단
            if etype == event.Type.MouseButtonDblClick:
                return True

            if etype == event.Type.MouseButtonPress:
                self._handle_mouse_press(event)
                return True
            elif etype == event.Type.MouseMove:
                self._handle_mouse_move(event)
                return True
            elif etype == event.Type.MouseButtonRelease:
                self._handle_mouse_release(event)
                return True
            elif etype == event.Type.Wheel:
                self._handle_wheel(event)
                return True
            elif etype == event.Type.Leave:
                # 마우스가 화면을 벗어나면 호버 십자선 감추기
                self.hover_h_actor.SetVisibility(False)
                self.hover_v_actor.SetVisibility(False)
                self.render()

        return super().eventFilter(obj, event)

    def _screen_to_world(self, qpos: QPoint) -> Optional[Tuple[float, float, float]]:
        if self.state_mgr.vtk_image is None:
            return None

        win_height = self.vtkWidget.height()
        vtk_x = qpos.x()
        vtk_y = win_height - qpos.y() - 1

        self.renderer.SetDisplayPoint(vtk_x, vtk_y, 0.0)
        self.renderer.DisplayToWorld()
        world_pt = list(self.renderer.GetWorldPoint())
        if world_pt[3] != 0.0:
            world_pt[0] /= world_pt[3]
            world_pt[1] /= world_pt[3]
            world_pt[2] /= world_pt[3]

        cur_x, cur_y, cur_z = self.state_mgr.current_coords
        if self.axis == AXIS_AXIAL:
            return (world_pt[0], world_pt[1], cur_z)
        elif self.axis == AXIS_CORONAL:
            return (world_pt[0], cur_y, world_pt[2])
        elif self.axis == AXIS_SAGITTAL:
            return (cur_x, world_pt[1], world_pt[2])
        return None

    def _get_current_slice_idx(self) -> int:
        """현재 단면의 슬라이스 정수 인덱스 반환"""
        if self.state_mgr.vtk_image is None:
            return 0
        coords = self.state_mgr.current_coords
        coord = coords[self.axis]
        origin = self.state_mgr.origin[self.axis]
        spacing = self.state_mgr.spacing[self.axis]
        dims = self.state_mgr.dimensions
        if spacing == 0:
            return 0
        idx = int(round((coord - origin) / spacing))
        return max(0, min(dims[self.axis] - 1, idx))

    def _get_slice_offset(self) -> float:
        fh = -1.0 if self._flip_h else 1.0
        fv = -1.0 if self._flip_v else 1.0
        norm_sign = fh * fv
        return 0.45 * norm_sign

    def _build_shape_polydata(self, shape_type: str, pt1: Tuple[float, float, float], pt2: Tuple[float, float, float],
                              end_cap: str = "none", is_dashed: bool = False, pts: Optional[List[Tuple[float, float, float]]] = None) -> vtk.vtkPolyData:
        """선분, 화살표, 사각형, 원, 프리폼, 끝단 캡(화살표, 원형, 양쪽 원형)을 vtkPolyData로 구성"""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        offset = self._get_slice_offset()

        x1, y1, z1 = pt1 if pt1 else (0, 0, 0)
        x2, y2, z2 = pt2 if pt2 else (0, 0, 0)

        # 2D 평면 투영 좌표 (u, v) 및 월드 좌표 변환기
        if self.axis == AXIS_AXIAL:
            ref_z = z1
            def to_uv(p): return (p[0], p[1])
            def to_w(u, v): return (u, v, ref_z + offset)
        elif self.axis == AXIS_CORONAL:
            ref_y = y1
            def to_uv(p): return (p[0], p[2])
            def to_w(u, v): return (u, ref_y - offset, v)
        else: # SAGITTAL
            ref_x = x1
            def to_uv(p): return (p[1], p[2])
            def to_w(u, v): return (ref_x + offset, u, v)

        u1, v1 = to_uv(pt1) if pt1 else (0, 0)
        u2, v2 = to_uv(pt2) if pt2 else (0, 0)

        du = u2 - u1
        dv = v2 - v1
        dist = math.hypot(du, dv)

        def add_segment(pa, pb):
            pid0 = points.InsertNextPoint(*pa)
            pid1 = points.InsertNextPoint(*pb)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pid0)
            line.GetPointIds().SetId(1, pid1)
            lines.InsertNextCell(line)

        def add_dashed_segment(pa, pb, seg_len=3.0, gap_len=2.0):
            (xa, ya, za), (xb, yb, zb) = pa, pb
            seg_dist = math.hypot(xb - xa, yb - ya, zb - za)
            if seg_dist == 0:
                return
            ux = (xb - xa) / seg_dist
            uy = (yb - ya) / seg_dist
            uz = (zb - za) / seg_dist
            cur = 0.0
            drawing = True
            while cur < seg_dist:
                step = seg_len if drawing else gap_len
                nxt = min(cur + step, seg_dist)
                if drawing:
                    p_start = (xa + ux * cur, ya + uy * cur, za + uz * cur)
                    p_end = (xa + ux * nxt, ya + uy * nxt, za + uz * nxt)
                    add_segment(p_start, p_end)
                cur = nxt
                drawing = not drawing

        draw_line_func = add_dashed_segment if is_dashed else add_segment

        if shape_type == "freeform":
            all_pts = pts if pts else ([pt1, pt2] if pt1 and pt2 else [])
            for i in range(len(all_pts) - 1):
                p_a, p_b = all_pts[i], all_pts[i + 1]
                ua, va = to_uv(p_a)
                ub, vb = to_uv(p_b)
                draw_line_func(to_w(ua, va), to_w(ub, vb))

        elif shape_type in ("line", "arrow"):
            draw_line_func(to_w(u1, v1), to_w(u2, v2))

            # 선분 끝 형태 / 화살표 처리
            cap_to_draw = end_cap
            if shape_type == "arrow" and cap_to_draw == "none":
                cap_to_draw = "single_arrow"

            if dist > 1.0:
                dir_u = du / dist
                dir_v = dv / dist
                perp_u = -dir_v
                perp_v = dir_u
                arrow_len = min(12.0, max(4.0, dist * 0.25))
                arrow_w = arrow_len * 0.45

                def draw_arrow_head(tip_u, tip_v, d_u, d_v):
                    b_u = tip_u - d_u * arrow_len
                    b_v = tip_v - d_v * arrow_len
                    w1 = (b_u + perp_u * arrow_w, b_v + perp_v * arrow_w)
                    w2 = (b_u - perp_u * arrow_w, b_v - perp_v * arrow_w)
                    add_segment(to_w(tip_u, tip_v), to_w(*w1))
                    add_segment(to_w(tip_u, tip_v), to_w(*w2))

                def draw_circle_cap(c_u, c_v, r=3.5):
                    n_segs = 12
                    c_pts = [
                        (c_u + r * math.cos(2 * math.pi * i / n_segs),
                         c_v + r * math.sin(2 * math.pi * i / n_segs))
                        for i in range(n_segs)
                    ]
                    for i in range(n_segs):
                        add_segment(to_w(*c_pts[i]), to_w(*c_pts[(i + 1) % n_segs]))

                if cap_to_draw in ("single_arrow", "arrow"):
                    draw_arrow_head(u2, v2, dir_u, dir_v)
                elif cap_to_draw == "double_arrow":
                    draw_arrow_head(u2, v2, dir_u, dir_v)
                    draw_arrow_head(u1, v1, -dir_u, -dir_v)
                elif cap_to_draw in ("circle", "single_circle"):
                    draw_circle_cap(u2, v2)
                elif cap_to_draw == "double_circle":
                    draw_circle_cap(u1, v1)
                    draw_circle_cap(u2, v2)

        elif shape_type == "rect":
            c_pts = [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]
            for i in range(4):
                draw_line_func(to_w(*c_pts[i]), to_w(*c_pts[(i + 1) % 4]))

        elif shape_type == "circle":
            r = dist
            n_segs = 32
            c_pts = [
                (u1 + r * math.cos(2 * math.pi * i / n_segs),
                 v1 + r * math.sin(2 * math.pi * i / n_segs))
                for i in range(n_segs)
            ]
            for i in range(n_segs):
                draw_line_func(to_w(*c_pts[i]), to_w(*c_pts[(i + 1) % n_segs]))

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        return poly

    def _build_measure_length_polydata(self, pt1: Tuple[float, float, float], pt2: Tuple[float, float, float]) -> Tuple[vtk.vtkPolyData, float, Tuple[float, float, float]]:
        """길이 측정용 두 점 사이 선분 및 양 끝단 T-cap PolyData 산출"""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        offset = self._get_slice_offset()

        x1, y1, z1 = pt1
        x2, y2, z2 = pt2

        if self.axis == AXIS_AXIAL:
            u1, v1 = x1, y1
            u2, v2 = x2, y2
            def to_w(u, v): return (u, v, z1 + offset)
        elif self.axis == AXIS_CORONAL:
            u1, v1 = x1, z1
            u2, v2 = x2, z2
            def to_w(u, v): return (u, y1 - offset, v)
        else: # SAGITTAL
            u1, v1 = y1, z1
            u2, v2 = y2, z2
            def to_w(u, v): return (x1 + offset, u, v)

        du = u2 - u1
        dv = v2 - v1
        dist_mm = math.hypot(du, dv)

        def add_segment(pa, pb):
            pid0 = points.InsertNextPoint(*pa)
            pid1 = points.InsertNextPoint(*pb)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pid0)
            line.GetPointIds().SetId(1, pid1)
            lines.InsertNextCell(line)

        # 주 선분
        add_segment(to_w(u1, v1), to_w(u2, v2))

        # 양 끝 T-cap
        if dist_mm > 0.1:
            perp_u = -dv / dist_mm
            perp_v = du / dist_mm
            cap_len = 5.0
            add_segment(to_w(u1 + perp_u * cap_len, v1 + perp_v * cap_len),
                        to_w(u1 - perp_u * cap_len, v1 - perp_v * cap_len))
            add_segment(to_w(u2 + perp_u * cap_len, v2 + perp_v * cap_len),
                        to_w(u2 - perp_u * cap_len, v2 - perp_v * cap_len))

        mid_pt = to_w((u1 + u2) * 0.5, (v1 + v2) * 0.5)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        return poly, dist_mm, mid_pt

    def _build_measure_angle_polydata(self, pt1: Tuple[float, float, float], pt2: Tuple[float, float, float], pt3: Tuple[float, float, float]) -> Tuple[vtk.vtkPolyData, float, Tuple[float, float, float]]:
        """각도 측정용 세 점(pt1 -> pt2(꼭짓점) -> pt3) 사이의 두 선분 및 각도 호 PolyData 산출"""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        offset = self._get_slice_offset()

        if self.axis == AXIS_AXIAL:
            def to_uv(p): return (p[0], p[1])
            def to_w(u, v): return (u, v, pt2[2] + offset)
        elif self.axis == AXIS_CORONAL:
            def to_uv(p): return (p[0], p[2])
            def to_w(u, v): return (u, pt2[1] - offset, v)
        else: # SAGITTAL
            def to_uv(p): return (p[1], p[2])
            def to_w(u, v): return (pt2[0] + offset, u, v)

        u1, v1 = to_uv(pt1)
        u2, v2 = to_uv(pt2)
        u3, v3 = to_uv(pt3)

        def add_segment(pa, pb):
            pid0 = points.InsertNextPoint(*pa)
            pid1 = points.InsertNextPoint(*pb)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pid0)
            line.GetPointIds().SetId(1, pid1)
            lines.InsertNextCell(line)

        # 두 선분: pt2 -> pt1, pt2 -> pt3
        add_segment(to_w(u2, v2), to_w(u1, v1))
        add_segment(to_w(u2, v2), to_w(u3, v3))

        # 각도 계산
        a1 = math.atan2(v1 - v2, u1 - u2)
        a2 = math.atan2(v3 - v2, u3 - u2)
        diff = math.degrees(abs(a1 - a2))
        if diff > 180.0:
            diff = 360.0 - diff

        # 호(Arc) 렌더링
        r_arc = min(20.0, max(5.0, min(math.hypot(u1 - u2, v1 - v2), math.hypot(u3 - u2, v3 - v2)) * 0.4))
        # start angle to end angle
        start_a = a1
        end_a = a2
        # 부호 있는 각도 차이
        da = (end_a - start_a) % (2 * math.pi)
        if da > math.pi:
            da -= 2 * math.pi

        n_steps = 12
        arc_pts = [
            (u2 + r_arc * math.cos(start_a + da * i / n_steps),
             v2 + r_arc * math.sin(start_a + da * i / n_steps))
            for i in range(n_steps + 1)
        ]
        for i in range(n_steps):
            add_segment(to_w(*arc_pts[i]), to_w(*arc_pts[i + 1]))

        mid_a = start_a + da * 0.5
        label_pos = to_w(u2 + (r_arc + 8.0) * math.cos(mid_a), v2 + (r_arc + 8.0) * math.sin(mid_a))

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        return poly, diff, label_pos

    def _point_to_segment_dist(self, p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """2D 평면에서 점 p와 선분 ab 사이의 최단 거리 계산"""
        px, py = p
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        l2 = dx * dx + dy * dy
        if l2 == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def _pick_shape_at_world(self, w_pt: Tuple[float, float, float]) -> int:
        """클릭된 월드 좌표에서 가장 가까운 현재 슬라이스의 도형 인덱스 반환 (없으면 -1)"""
        if not self.draw_dialog or not getattr(self.draw_dialog, "shapes", []):
            return -1

        cur_slice = self._get_current_slice_idx()
        if self.axis == AXIS_AXIAL:
            def to_uv(p): return (p[0], p[1])
        elif self.axis == AXIS_CORONAL:
            def to_uv(p): return (p[0], p[2])
        else:
            def to_uv(p): return (p[1], p[2])

        p_uv = to_uv(w_pt)
        min_dist = float("inf")
        best_idx = -1
        threshold_mm = 6.0  # 클릭 선택 허용 반경

        for idx, s in enumerate(self.draw_dialog.shapes):
            if s.get("view_name") != self.axis_name or s.get("slice_idx") != cur_slice:
                continue
            stype = s.get("shape_type", "line")
            pts = s.get("pts", [])
            if not pts:
                continue

            dist = float("inf")
            if stype in ("line", "arrow"):
                if len(pts) >= 2:
                    dist = self._point_to_segment_dist(p_uv, to_uv(pts[0]), to_uv(pts[1]))
            elif stype == "rect":
                if len(pts) >= 2:
                    u1, v1 = to_uv(pts[0])
                    u2, v2 = to_uv(pts[1])
                    c = [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]
                    for i in range(4):
                        d = self._point_to_segment_dist(p_uv, c[i], c[(i + 1) % 4])
                        if d < dist:
                            dist = d
            elif stype == "circle":
                if len(pts) >= 2:
                    u1, v1 = to_uv(pts[0])
                    u2, v2 = to_uv(pts[1])
                    r = math.hypot(u2 - u1, v2 - v1)
                    center_dist = math.hypot(p_uv[0] - u1, p_uv[1] - v1)
                    dist = abs(center_dist - r)
            elif stype == "freeform":
                for i in range(len(pts) - 1):
                    d = self._point_to_segment_dist(p_uv, to_uv(pts[i]), to_uv(pts[i + 1]))
                    if d < dist:
                        dist = d

            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        return best_idx if min_dist <= threshold_mm else -1

    def _pick_measure_at_world(self, w_pt: Tuple[float, float, float]) -> int:
        """클릭된 월드 좌표에서 가장 가까운 현재 슬라이스의 계측 항목 인덱스 반환 (없으면 -1)"""
        if not self.measure_dialog or not getattr(self.measure_dialog, "measurements", []):
            return -1

        cur_slice = self._get_current_slice_idx()
        if self.axis == AXIS_AXIAL:
            def to_uv(p): return (p[0], p[1])
        elif self.axis == AXIS_CORONAL:
            def to_uv(p): return (p[0], p[2])
        else:
            def to_uv(p): return (p[1], p[2])

        p_uv = to_uv(w_pt)
        min_dist = float("inf")
        best_idx = -1
        threshold_mm = 6.0

        for idx, m in enumerate(self.measure_dialog.measurements):
            if m.get("view_name") != self.axis_name or m.get("slice_idx") != cur_slice:
                continue
            mtype = m.get("type", "length")
            w_pts = m.get("world_pts", [])
            if not w_pts:
                continue

            dist = float("inf")
            if mtype == "length" and len(w_pts) >= 2:
                dist = self._point_to_segment_dist(p_uv, to_uv(w_pts[0]), to_uv(w_pts[1]))
            elif mtype == "angle" and len(w_pts) >= 3:
                d1 = self._point_to_segment_dist(p_uv, to_uv(w_pts[0]), to_uv(w_pts[1]))
                d2 = self._point_to_segment_dist(p_uv, to_uv(w_pts[2]), to_uv(w_pts[1]))
                dist = min(d1, d2)

            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        return best_idx if min_dist <= threshold_mm else -1

    def _update_overlays(self):
        """현재 슬라이스 인덱스에 맞춰 도형 및 계측 액터 가시성 필터링 및 렌더링"""
        cur_slice = self._get_current_slice_idx()

        # 1. 기존 액터들 제거
        for act in self._shape_actors:
            self.renderer.RemoveActor(act)
        self._shape_actors.clear()

        for act in self._measure_actors:
            self.renderer.RemoveActor(act)
        self._measure_actors.clear()

        # 2. DrawDialog 도형 렌더링 (창 닫혀도 보이기 옵션 반영)
        if self.draw_dialog and self.show_drawn_shapes:
            for idx, s in enumerate(getattr(self.draw_dialog, "shapes", [])):
                if s.get("view_name") == self.axis_name and s.get("slice_idx") == cur_slice:
                    stype = s.get("shape_type", "line")
                    pts = s.get("pts", [])
                    if len(pts) >= 2 or (stype == "freeform" and len(pts) >= 2):
                        color_val = s.get("color", "#ff3c3c")
                        qcol = QColor(color_val) if isinstance(color_val, str) else color_val
                        rgb = (qcol.redF(), qcol.greenF(), qcol.blueF()) if isinstance(qcol, QColor) else (1.0, 0.2, 0.2)
                        w = float(s.get("width", 1))
                        dashed = bool(s.get("is_dashed", False))
                        cap = s.get("end_cap", "none")

                        # 리스트 선택 하이라이트 (노란색 및 두께 강조)
                        if idx == self.selected_shape_idx:
                            rgb = (1.0, 1.0, 0.0)
                            w = max(w + 1.5, 3.0)

                        poly = self._build_shape_polydata(stype, pts[0], pts[1] if len(pts) > 1 else pts[0],
                                                          end_cap=cap, is_dashed=dashed, pts=pts)
                        m = vtk.vtkPolyDataMapper()
                        m.SetInputData(poly)
                        act = vtk.vtkActor()
                        act.SetMapper(m)
                        act.GetProperty().SetColor(*rgb)
                        act.GetProperty().SetLineWidth(w)
                        act.GetProperty().LightingOff()
                        self.renderer.AddActor(act)
                        self._shape_actors.append(act)

        # 3. MeasurementDialog 계측 렌더링 (선분 보이기 체크박스 반영, 측정값 라벨 항상 표시)
        if self.measure_dialog and self.show_measurements:
            for idx, m_item in enumerate(getattr(self.measure_dialog, "measurements", [])):
                if m_item.get("view_name") == self.axis_name and m_item.get("slice_idx") == cur_slice:
                    m_type = m_item.get("type", "length")
                    w_pts = m_item.get("world_pts", [])
                    is_selected = (idx == self.selected_measure_idx)
                    val_str = m_item.get("value_str", "")

                    if m_type == "length" and len(w_pts) >= 2:
                        poly, dist, mid_pt = self._build_measure_length_polydata(w_pts[0], w_pts[1])
                        m = vtk.vtkPolyDataMapper()
                        m.SetInputData(poly)
                        act = vtk.vtkActor()
                        act.SetMapper(m)
                        line_col = (1.0, 1.0, 0.0) if is_selected else (0.2, 0.8, 1.0)
                        line_w = 3.0 if is_selected else 1.0
                        act.GetProperty().SetColor(*line_col)
                        act.GetProperty().SetLineWidth(line_w)
                        act.GetProperty().LightingOff()
                        self.renderer.AddActor(act)
                        self._measure_actors.append(act)

                        # 측정값 라벨 영구 표시
                        if val_str:
                            text_act = vtk.vtkBillboardTextActor3D()
                            text_act.SetInput(val_str)
                            text_act.SetPosition(mid_pt)
                            tprop = text_act.GetTextProperty()
                            tprop.SetFontSize(12)
                            tprop.BoldOn()
                            if is_selected:
                                tprop.SetColor(1.0, 1.0, 0.0)
                            else:
                                tprop.SetColor(0.3, 0.9, 1.0)
                            self.renderer.AddActor(text_act)
                            self._measure_actors.append(text_act)

                    elif m_type == "angle" and len(w_pts) >= 3:
                        poly, angle_val, label_pt = self._build_measure_angle_polydata(w_pts[0], w_pts[1], w_pts[2])
                        m = vtk.vtkPolyDataMapper()
                        m.SetInputData(poly)
                        act = vtk.vtkActor()
                        act.SetMapper(m)
                        line_col = (1.0, 1.0, 0.0) if is_selected else (1.0, 0.8, 0.2)
                        line_w = 3.0 if is_selected else 1.0
                        act.GetProperty().SetColor(*line_col)
                        act.GetProperty().SetLineWidth(line_w)
                        act.GetProperty().LightingOff()
                        self.renderer.AddActor(act)
                        self._measure_actors.append(act)

                        # 각도 측정값 라벨 영구 표시
                        if val_str:
                            text_act = vtk.vtkBillboardTextActor3D()
                            text_act.SetInput(val_str)
                            text_act.SetPosition(label_pt)
                            tprop = text_act.GetTextProperty()
                            tprop.SetFontSize(12)
                            tprop.BoldOn()
                            if is_selected:
                                tprop.SetColor(1.0, 1.0, 0.0)
                            else:
                                tprop.SetColor(1.0, 0.85, 0.2)
                            self.renderer.AddActor(text_act)
                            self._measure_actors.append(text_act)

    def _handle_mouse_press(self, event):
        self._last_mouse_pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_pressed = True
            w_pt = self._screen_to_world(self._last_mouse_pos)
            if not w_pt:
                return

            if self._is_roi_setting_mode:
                # ROI boundary 설정 모드
                self._roi_rect_start = w_pt
                self.roi_rect_actor.SetVisibility(False)

            elif self.interaction_mode in ("draw", "draw_select"):
                style = self.draw_dialog.get_current_style() if self.draw_dialog else {}
                stype = style.get("shape_type", "line")

                # 기존 도형 클릭 여부 우선 검사
                matched = self._pick_shape_at_world(w_pt)

                if self.interaction_mode == "draw_select" or stype == "select":
                    # 선택 취소 모드: 화면에서 그린 도형 클릭 시 선택(하이라이트)
                    if self.draw_dialog:
                        if matched >= 0:
                            self.draw_dialog.table.selectRow(matched)
                            self.set_selected_shape(matched)
                        else:
                            self.draw_dialog.table.clearSelection()
                            self.set_selected_shape(-1)
                elif matched >= 0:
                    # 그리기 모드라도 기존 도형을 클릭한 경우 해당 도형 선택
                    if self.draw_dialog:
                        self.draw_dialog.table.selectRow(matched)
                    self.set_selected_shape(matched)
                elif stype == "freeform":
                    # 프리폼 그리기 시작
                    self.draw_start_pt = w_pt
                    self.draw_pts = [w_pt]
                    self.temp_actor.SetVisibility(False)
                else:
                    # 일반 도형 그리기 시작점 기록
                    self.draw_start_pt = w_pt
                    self.draw_pts.clear()
                    self.temp_actor.SetVisibility(False)

            elif self.interaction_mode == "measure_length":
                # 길이 측정 시작점 기록
                self.measure_start_pt = w_pt
                self.temp_actor.SetVisibility(False)
                self.temp_text_actor.SetVisibility(False)

            elif self.interaction_mode == "measure_angle":
                # 각도 측정 3점 순차 클릭
                self._temp_angle_pts.append(w_pt)
                if len(self._temp_angle_pts) == 3:
                    p1, p2, p3 = self._temp_angle_pts
                    poly, angle_val, label_pt = self._build_measure_angle_polydata(p1, p2, p3)
                    self._temp_angle_pts.clear()
                    self.temp_actor.SetVisibility(False)
                    self.temp_text_actor.SetVisibility(False)

                    if self.measure_dialog:
                        s_idx = self._get_current_slice_idx()
                        self.measure_dialog.add_measurement({
                            "type": "angle",
                            "value_str": f"{angle_val:.1f}°",
                            "view_name": self.axis_name,
                            "slice_idx": s_idx,
                            "world_pts": [p1, p2, p3]
                        })
                    self._update_overlays()
                    self.render()

            else:
                # 일반 모드: 2D 임플란트 기즈모 조작 및 선택 체크
                screen_x = self._last_mouse_pos.x()
                screen_y = self.vtkWidget.height() - self._last_mouse_pos.y()

                # 1) 기즈모 핸들 조작 체크
                if hasattr(self, "implant_gizmo_2d") and self.implant_gizmo_2d.is_attached:
                    handle = self.implant_gizmo_2d.find_handle_by_screen_pos(screen_x, screen_y)
                    if handle and self.implant_gizmo_2d.start_drag(handle, screen_x, screen_y):
                        self.render()
                        return

                # 2) 임플란트 단면 피킹(선택) 체크
                hit_imp = None
                picker = vtk.vtkPropPicker()
                if picker.Pick(screen_x, screen_y, 0, self.renderer):
                    prop = picker.GetViewProp()
                    hit_imp = self.state_mgr.implant_placement_mgr.find_implant_by_actor(prop)

                if not hit_imp:
                    hit_imp = self.state_mgr.implant_placement_mgr.find_implant_by_slice_pos(self.view_name, w_pt)

                if not hit_imp:
                    hit_imp = self.state_mgr.implant_placement_mgr.find_implant_by_screen_ray(self.renderer, screen_x, screen_y)

                if hit_imp:
                    self.state_mgr.implant_placement_mgr.select_implant(hit_imp.id, multi_select=False)
                    return

                # 3) 화면에서 그린 도형 피킹 체크
                matched_shape = self._pick_shape_at_world(w_pt)
                if matched_shape >= 0:
                    if self.draw_dialog:
                        self.draw_dialog.table.selectRow(matched_shape)
                    self.set_selected_shape(matched_shape)
                    return

                # 4) 화면에서 계측 항목 피킹 체크
                matched_measure = self._pick_measure_at_world(w_pt)
                if matched_measure >= 0:
                    if self.measure_dialog:
                        self.measure_dialog.table.selectRow(matched_measure)
                    self.set_selected_measurement(matched_measure)
                    return

                # 5) 빈 영역 클릭 시: 선택 해제 및 십자선 이동
                if hasattr(self, "implant_gizmo_2d") and self.implant_gizmo_2d.is_attached:
                    self.state_mgr.implant_placement_mgr.clear_selection()

                if getattr(self, "selected_measure_idx", -1) != -1:
                    self.selected_measure_idx = -1
                    if self.measure_dialog:
                        self.measure_dialog.clear_selection()
                    self._update_overlays()

                if getattr(self, "selected_shape_idx", -1) != -1:
                    self.selected_shape_idx = -1
                    if self.draw_dialog:
                        self.draw_dialog.clear_selection()
                    self._update_overlays()

                self.state_mgr.set_coordinates(w_pt[0], w_pt[1], w_pt[2])

        elif event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pressed = True
        elif event.button() == Qt.MouseButton.RightButton:
            self._right_pressed = True

    def _handle_mouse_move(self, event):
        pos = event.position().toPoint()
        delta = pos - self._last_mouse_pos
        w_pt = self._screen_to_world(pos)

        if self._middle_pressed:
            # 휠(가운데 버튼) 드래그: 이미지 이동 (Pan)
            self._pan_camera(delta.x(), delta.y())

        elif self._left_pressed:
            # 2D 기즈모 드래그 중인 경우 우선 처리
            if hasattr(self, "implant_gizmo_2d") and self.implant_gizmo_2d.is_dragging:
                screen_x = pos.x()
                screen_y = self.vtkWidget.height() - pos.y()
                if self.implant_gizmo_2d.update_drag(screen_x, screen_y):
                    if hasattr(self, "parent_panel") and hasattr(self.parent_panel, "container") and self.parent_panel.container:
                        self.parent_panel.container.render_all()
                    else:
                        self.render()
                self._last_mouse_pos = pos
                return

            if self._is_roi_setting_mode and self._roi_rect_start and w_pt:
                self._draw_roi_rect(self._roi_rect_start, w_pt)

            elif self.interaction_mode == "draw" and self.draw_start_pt and w_pt:
                style = self.draw_dialog.get_current_style() if self.draw_dialog else {}
                stype = style.get("shape_type", "line")
                col = style.get("color", QColor(255, 60, 60))
                rgb = (col.redF(), col.greenF(), col.blueF()) if isinstance(col, QColor) else (1.0, 0.2, 0.2)
                w = float(style.get("width", 1))

                if stype == "freeform":
                    # 프리폼 연속 좌표 수집 및 실시간 렌더링
                    last_pt = self.draw_pts[-1]
                    d_cur = math.hypot(w_pt[0] - last_pt[0], w_pt[1] - last_pt[1], w_pt[2] - last_pt[2])
                    if d_cur >= 0.6:
                        self.draw_pts.append(w_pt)
                    poly = self._build_shape_polydata("freeform", self.draw_start_pt, w_pt, pts=self.draw_pts)
                    self.temp_poly.ShallowCopy(poly)
                    self.temp_actor.GetProperty().SetColor(*rgb)
                    self.temp_actor.GetProperty().SetLineWidth(w)
                    self.temp_actor.SetVisibility(True)
                    self.render()

                elif stype != "select":
                    # 일반 도형 미리보기
                    cap = style.get("end_cap", "none")
                    dashed = style.get("is_dashed", False)
                    poly = self._build_shape_polydata(stype, self.draw_start_pt, w_pt, end_cap=cap, is_dashed=dashed)

                    self.temp_poly.ShallowCopy(poly)
                    self.temp_actor.GetProperty().SetColor(*rgb)
                    self.temp_actor.GetProperty().SetLineWidth(w)
                    self.temp_actor.SetVisibility(True)
                    self.render()

            elif self.interaction_mode == "measure_length" and self.measure_start_pt and w_pt:
                # 실시간 길이 계측 미리보기
                poly, dist_mm, mid_pt = self._build_measure_length_polydata(self.measure_start_pt, w_pt)
                self.temp_poly.ShallowCopy(poly)
                self.temp_actor.GetProperty().SetColor(0.2, 0.8, 1.0)
                self.temp_actor.GetProperty().SetLineWidth(1.0)
                self.temp_actor.SetVisibility(True)

                self.temp_text_actor.SetInput(f"{dist_mm:.2f} mm")
                self.temp_text_actor.SetDisplayPosition(pos.x() + 10, self.vtkWidget.height() - pos.y() + 10)
                self.temp_text_actor.SetVisibility(True)
                self.render()

            elif not self._is_roi_setting_mode and self.interaction_mode == "normal" and w_pt:
                # 좌클릭 드래그: 기준 십자선 이동 및 동조
                self.state_mgr.set_coordinates(w_pt[0], w_pt[1], w_pt[2])

        elif self._right_pressed:
            # 우클릭 드래그: Window / Level 조절
            cur_w = self.state_mgr.window
            cur_l = self.state_mgr.level
            new_w = max(1.0, cur_w + delta.x() * 4.0)
            new_l = cur_l - delta.y() * 4.0
            self.state_mgr.set_window_level(new_w, new_l)

        else:
            if self.interaction_mode == "measure_angle" and len(self._temp_angle_pts) in (1, 2) and w_pt:
                # 각도를 그리는 동안에는 점선(dashed)으로 표시
                if len(self._temp_angle_pts) == 1:
                    p1 = self._temp_angle_pts[0]
                    poly = self._build_shape_polydata("line", p1, w_pt, is_dashed=True)
                else:
                    # 2점 상태: p1 -> p2 (실선) + p2 -> w_pt (점선)
                    p1, p2 = self._temp_angle_pts
                    poly, angle_val, label_pt = self._build_measure_angle_polydata(p1, p2, w_pt)
                    self.temp_text_actor.SetInput(f"{angle_val:.1f}°")
                    self.temp_text_actor.SetDisplayPosition(pos.x() + 10, self.vtkWidget.height() - pos.y() + 10)
                    self.temp_text_actor.SetVisibility(True)

                self.temp_poly.ShallowCopy(poly)
                self.temp_actor.GetProperty().SetColor(1.0, 0.8, 0.2)
                self.temp_actor.GetProperty().SetLineWidth(1.0)
                self.temp_actor.SetVisibility(True)
                self.render()

            elif w_pt and not self._is_roi_setting_mode and self.interaction_mode == "normal":
                # 2D 기즈모 핸들 호버 하이라이트 체크
                if hasattr(self, "implant_gizmo_2d") and self.implant_gizmo_2d.is_attached:
                    screen_x = pos.x()
                    screen_y = self.vtkWidget.height() - pos.y()
                    handle = self.implant_gizmo_2d.find_handle_by_screen_pos(screen_x, screen_y)
                    if self.implant_gizmo_2d.highlight_handle(handle):
                        self.render()

                # 마우스 이동(Hover): 검은색 실선 십자선 실시간 표시
                self._update_hover_crosshair(w_pt[0], w_pt[1], w_pt[2])

        self._last_mouse_pos = pos

    def _handle_mouse_release(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_pressed = False
            # 2D 기즈모 드래그 종료 처리
            if hasattr(self, "implant_gizmo_2d") and self.implant_gizmo_2d.is_dragging:
                self.implant_gizmo_2d.end_drag()
                if hasattr(self, "parent_panel") and hasattr(self.parent_panel, "container") and self.parent_panel.container:
                    self.parent_panel.container.render_all()
                else:
                    self.render()
                return

            w_pt = self._screen_to_world(event.position().toPoint())

            if self._is_roi_setting_mode and self._roi_rect_start and w_pt:
                self._commit_roi_bounds(self._roi_rect_start, w_pt)
                self._roi_rect_start = None

            elif self.interaction_mode == "draw" and self.draw_start_pt and w_pt:
                style = self.draw_dialog.get_current_style() if self.draw_dialog else {}
                stype = style.get("shape_type", "line")
                s_idx = self._get_current_slice_idx()

                if stype == "freeform":
                    if len(self.draw_pts) >= 2 and self.draw_dialog:
                        self.draw_dialog.add_shape({
                            "shape_type": "freeform",
                            "view_name": self.axis_name,
                            "slice_idx": s_idx,
                            "pts": list(self.draw_pts),
                            "color": style.get("color", QColor(255, 60, 60)),
                            "width": style.get("width", 1),
                            "is_dashed": style.get("is_dashed", False),
                            "end_cap": "none"
                        })
                    self.draw_pts.clear()

                elif stype != "select":
                    # 드래그 거리가 1.5mm 이상일 때만 등록
                    dx = w_pt[0] - self.draw_start_pt[0]
                    dy = w_pt[1] - self.draw_start_pt[1]
                    dz = w_pt[2] - self.draw_start_pt[2]
                    if math.hypot(dx, dy, dz) > 1.5 and self.draw_dialog:
                        self.draw_dialog.add_shape({
                            "shape_type": stype,
                            "view_name": self.axis_name,
                            "slice_idx": s_idx,
                            "pts": [self.draw_start_pt, w_pt],
                            "color": style.get("color", QColor(255, 60, 60)),
                            "width": style.get("width", 1),
                            "is_dashed": style.get("is_dashed", False),
                            "end_cap": style.get("end_cap", "none")
                        })

                self.draw_start_pt = None
                if self.draw_dialog and self.draw_dialog.shapes:
                    new_idx = len(self.draw_dialog.shapes) - 1
                    self.selected_shape_idx = new_idx
                    self.draw_dialog.table.selectRow(new_idx)
                self.temp_actor.SetVisibility(False)
                self._update_overlays()
                self.render()

            elif self.interaction_mode == "measure_length" and self.measure_start_pt and w_pt:
                poly, dist_mm, mid_pt = self._build_measure_length_polydata(self.measure_start_pt, w_pt)
                if dist_mm > 0.5:
                    s_idx = self._get_current_slice_idx()
                    if self.measure_dialog:
                        self.measure_dialog.add_measurement({
                            "type": "length",
                            "value_str": f"{dist_mm:.2f} mm",
                            "view_name": self.axis_name,
                            "slice_idx": s_idx,
                            "world_pts": [self.measure_start_pt, w_pt]
                        })
                self.measure_start_pt = None
                if self.measure_dialog and self.measure_dialog.measurements:
                    new_idx = len(self.measure_dialog.measurements) - 1
                    self.selected_measure_idx = new_idx
                    self.measure_dialog.table.selectRow(new_idx)
                self.temp_actor.SetVisibility(False)
                self.temp_text_actor.SetVisibility(False)
                self._update_overlays()
                self.render()

        elif event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pressed = False
        elif event.button() == Qt.MouseButton.RightButton:
            self._right_pressed = False

    def _commit_roi_bounds(self, pt1: Tuple[float, float, float], pt2: Tuple[float, float, float]):
        """그린 2D 사각형을 기반으로 3D 육면체 Bounding Box 산출 및 StateManager 브로드캐스트"""
        bx0, bx1, by0, by1, bz0, bz1 = self.state_mgr.bounds
        x_min, x_max = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
        y_min, y_max = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
        z_min, z_max = min(pt1[2], pt2[2]), max(pt1[2], pt2[2])

        if self.axis == AXIS_AXIAL:
            new_bounds = (x_min, x_max, y_min, y_max, bz0, bz1)
        elif self.axis == AXIS_CORONAL:
            new_bounds = (x_min, x_max, by0, by1, z_min, z_max)
        elif self.axis == AXIS_SAGITTAL:
            new_bounds = (bx0, bx1, y_min, y_max, z_min, z_max)
        else:
            return

        if (new_bounds[1] - new_bounds[0] > 0.5) and (new_bounds[3] - new_bounds[2] > 0.5) and (new_bounds[5] - new_bounds[4] > 0.5):
            self.state_mgr.set_roi_bounds(new_bounds)


    def _handle_wheel(self, event):
        """마우스 휠: 화면 확대 / 축소 (Zoom In / Out)"""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self.camera.Zoom(factor)
        self.render()
