"""
result_view_controller.py
Controller for rendering FEA simulation results in a 3D viewer window or VTK renderer.

Manages VTK actors, mappers, colormaps, scalar bars, deformation scaling,
and cross-section clipping for postprocess simulation visualization.
"""

import os
from typing import Optional, Dict, Tuple, Any, List
import numpy as np
import vtk
from vtk.util import numpy_support
from PySide6.QtCore import QObject, Signal

from fea.models import MeshModel, MaterialAssignment
from fea.results import ResultSet
from postprocess.vtk_builder import build_part_grid, attach_results, set_deform_scale


FIELD_DISPLAY_CONFIG = {
    "von_mises": {
        "title": "von Mises Stress",
        "unit": "MPa",
        "is_cell": False,
        "default_cmap": "rainbow",
    },
    "u_mag": {
        "title": "Displacement Magnitude",
        "unit": "mm",
        "is_cell": False,
        "default_cmap": "rainbow",
    },
    "s_max_principal": {
        "title": "Max Principal Stress",
        "unit": "MPa",
        "is_cell": False,
        "default_cmap": "coolwarm",
    },
    "s_min_principal": {
        "title": "Min Principal Stress",
        "unit": "MPa",
        "is_cell": False,
        "default_cmap": "coolwarm",
    },
    "E_MPa": {
        "title": "Young's Modulus",
        "unit": "MPa",
        "is_cell": True,
        "default_cmap": "viridis",
    },
    "equiv_strain": {
        "title": "Equivalent Strain",
        "unit": "µε",
        "is_cell": False,
        "default_cmap": "rainbow",
    },
}


class ResultViewController(QObject):
    """
    Controller responsible for displaying simulation results in a 3D viewer.

    Operates on a VolumeViewerWindow or directly on a vtkRenderer.
    """
    info = Signal(str)

    def __init__(self, viewer_window_or_renderer: Any):
        super().__init__()
        if hasattr(viewer_window_or_renderer, "renderer"):
            self.window = viewer_window_or_renderer
            self.renderer: vtk.vtkRenderer = viewer_window_or_renderer.renderer
        elif isinstance(viewer_window_or_renderer, vtk.vtkRenderer):
            self.window = None
            self.renderer = viewer_window_or_renderer
        else:
            raise TypeError("viewer_window_or_renderer must be a VolumeViewerWindow or vtkRenderer instance.")

        # 모델 및 결과셋 참조
        self._model: Optional[MeshModel] = None
        self._rs: Optional[ResultSet] = None
        self._materials: Optional[MaterialAssignment] = None

        # 상태 기억 (기존 장면 액터들의 원래 가시성 백업)
        self._original_visibilities: Dict[vtk.vtkProp, bool] = {}

        # 파트별 시각화 객체 파이프라인
        self._grids: Dict[str, vtk.vtkUnstructuredGrid] = {}
        self._linear_grids: Dict[str, vtk.vtkUnstructuredGrid] = {}
        self._mappers: Dict[str, vtk.vtkDataSetMapper] = {}
        self._actors: Dict[str, vtk.vtkActor] = {}
        self._luts: Dict[str, vtk.vtkLookupTable] = {}
        self._clippers: Dict[str, vtk.vtkClipDataSet] = {}
        self._clip_surface_filters: Dict[str, vtk.vtkDataSetSurfaceFilter] = {}

        # 스칼라 바
        self._scalar_bar: Optional[vtk.vtkScalarBarActor] = None

        # 제어 파라미터 상태
        self._current_field: str = "von_mises"
        self._range_mode: str = "global"  # "global", "per_part", "manual"
        self._manual_range: Optional[Tuple[float, float]] = None
        self._active_scalar_part: Optional[str] = None
        self._colormap_name: str = "rainbow"
        self._deform_scale: float = 0.0
        self._clip_enabled: bool = False
        self._clip_origin: Optional[Tuple[float, float, float]] = None
        self._clip_normal: Optional[Tuple[float, float, float]] = None
        self._edges_visible: bool = False
        self._use_linear_default: bool = False
        self._load_arrow_actors: List[vtk.vtkProp] = []
        self._load_arrow_visible: bool = False

    # -------------------------------------------------------------------------
    # 렌더링 통합 진입점
    # -------------------------------------------------------------------------
    def _render(self) -> None:
        """단일 통합 렌더링 갱신 호출."""
        if self.window is not None and hasattr(self.window, "render"):
            self.window.render()
        elif self.renderer is not None:
            rw = self.renderer.GetRenderWindow()
            if rw is not None:
                rw.Render()

    # -------------------------------------------------------------------------
    # 결과 설정 및 제거 (Lifecycle)
    # -------------------------------------------------------------------------
    def set_results(
        self,
        model: MeshModel,
        rs: ResultSet,
        materials: Optional[MaterialAssignment] = None,
    ) -> None:
        """
        Build grids and actors for all parts in the MeshModel and display results.

        Args:
            model: MeshModel containing part geometry and assembly offsets.
            rs: ResultSet containing displacement, stress, and strain fields.
            materials: Optional MaterialAssignment for element Young's modulus display.
        """
        # 기존 결과 및 장면 정리 (현재 장면의 원래 가시성 백업)
        self.clear()

        self._model = model
        self._rs = rs
        self._materials = materials

        # 1. 기존 장면 액터 가시성 백업 및 숨김
        props = self.renderer.GetViewProps()
        props.InitTraversal()
        for _ in range(props.GetNumberOfItems()):
            prop = props.GetNextProp()
            if prop is not None:
                self._original_visibilities[prop] = bool(prop.GetVisibility())
                prop.SetVisibility(False)

        # 2. 메쉬 규모 검사 (전체 요소 수 30만 초과 시 선형 단순화)
        total_elements = sum(len(p.elements) for p in model.parts)
        if total_elements > 300_000:
            self._use_linear_default = True
            self.info.emit(
                f"대규모 메쉬 감지 ({total_elements:,} 요소): 렌더링 성능을 위해 선형 사면체로 단순화하여 표시합니다."
            )
        else:
            self._use_linear_default = False

        # 3. 파트별 그리드, 단면용 선형 그리드, 매퍼, 액터 생성
        for part in model.parts:
            offset = model.global_node_offset(part.name)

            # 주 렌더링 그리드
            grid = build_part_grid(part, node_offset=offset, linear_only=self._use_linear_default)
            attach_results(
                grid, rs, model, part.name,
                deform_scale=self._deform_scale,
                material_assignment=materials,
            )
            self._grids[part.name] = grid

            # 단면 클립용 선형 그리드 (단면은 항상 linear_only에서 수행)
            if self._use_linear_default:
                linear_grid = grid
            else:
                linear_grid = build_part_grid(part, node_offset=offset, linear_only=True)
                attach_results(
                    linear_grid, rs, model, part.name,
                    deform_scale=self._deform_scale,
                    material_assignment=materials,
                )
            self._linear_grids[part.name] = linear_grid

            # 클리퍼 파이프라인 준비
            clipper = vtk.vtkClipDataSet()
            clipper.SetInputData(linear_grid)
            clipper.InsideOutOn()
            self._clippers[part.name] = clipper

            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(clipper.GetOutputPort())
            self._clip_surface_filters[part.name] = surface_filter

            # 매퍼 및 액터
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(grid)

            lut = self._build_lookup_table(self._colormap_name)
            self._luts[part.name] = lut
            mapper.SetLookupTable(lut)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetEdgeVisibility(1 if self._edges_visible else 0)
            actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)
            actor.GetProperty().SetOpacity(1.0)

            self.renderer.AddActor(actor)
            self._mappers[part.name] = mapper
            self._actors[part.name] = actor

        # 기본 활성 스칼라 바 파트 설정 (첫 번째 파트 또는 골 파트)
        if model.parts:
            self._active_scalar_part = model.parts[0].name
            for p in model.parts:
                if "bone" in p.name.lower() or "mandible" in p.name.lower():
                    self._active_scalar_part = p.name
                    break

        # 4. 스칼라 바 액터 생성 및 등록
        self._init_scalar_bar()

        # 5. 기본 필드 및 범위 적용
        self.set_field(self._current_field)
        self._update_pipeline()

        # 6. 해석 결과 3D 모드 활성화 (임플란트 선택 기즈모 숨김)
        if self.window is not None and hasattr(self.window, "set_result_view_mode"):
            self.window.set_result_view_mode(True)

        self._render()

    def clear(self) -> None:
        """결과 액터를 모두 제거하고 원래 장면의 액터 가시성을 완벽히 복원."""
        # 1. 결과 액터 및 매퍼 제거
        for actor in self._actors.values():
            self.renderer.RemoveActor(actor)
        self._actors.clear()
        self._mappers.clear()
        self._grids.clear()
        self._linear_grids.clear()
        self._luts.clear()
        self._clippers.clear()
        self._clip_surface_filters.clear()

        # 2. 하중 화살표 액터 제거
        for prop in self._load_arrow_actors:
            self.renderer.RemoveViewProp(prop)
        self._load_arrow_actors.clear()

        # 3. 스칼라 바 제거
        if self._scalar_bar is not None:
            self.renderer.RemoveActor(self._scalar_bar)
            self._scalar_bar = None

        # 4. 원래 장면 액터 가시성 복원
        for prop, vis in self._original_visibilities.items():
            try:
                prop.SetVisibility(vis)
            except Exception:
                pass
        self._original_visibilities.clear()

        self._model = None
        self._rs = None
        self._materials = None

        # 5. 해석 결과 3D 모드 해제 (원래 상태 복구)
        if self.window is not None and hasattr(self.window, "set_result_view_mode"):
            self.window.set_result_view_mode(False)

        self._render()

    def set_load_arrow_visible(self, visible: bool, bc_set: Optional[Any] = None) -> None:
        """
        교합 하중 화살표 가시성 설정 (3D 조작 기즈모는 일체 미표시).
        화면에 단 1개의 대표 화살표만 관리하여 중복 생성을 방지합니다.
        """
        self._load_arrow_visible = visible
        for prop in self._load_arrow_actors:
            self.renderer.RemoveViewProp(prop)
        self._load_arrow_actors.clear()

        if not visible or self._model is None:
            self._render()
            return

        if bc_set is None:
            self._render()
            return

        loads = getattr(bc_set, "loads", [])
        if not loads:
            self._render()
            return

        for load in loads:
            force_N = np.asarray(load.get("force_N", (0.0, 0.0, 0.0)), dtype=np.float64)
            mag = float(np.linalg.norm(force_N))
            if mag < 1e-6:
                continue

            part_name = load.get("part", "")
            part = self._model.get_part(part_name)
            node_ids = load.get("node_ids", np.array([], dtype=np.int64))

            if part is not None and len(node_ids) > 0:
                load_pts = part.nodes[node_ids]
                center = np.mean(load_pts, axis=0)
            else:
                imp_part = self._model.get_part("implant")
                if imp_part is not None and len(imp_part.nodes) > 0:
                    center = imp_part.nodes[np.argmax(imp_part.nodes[:, 2])]
                else:
                    all_nodes = [p.nodes for p in self._model.parts if len(p.nodes) > 0]
                    if all_nodes:
                        stacked = np.vstack(all_nodes)
                        center = stacked[np.argmax(stacked[:, 2])]
                    else:
                        center = np.array([0.0, 0.0, 0.0])

            arrow = vtk.vtkArrowSource()
            arrow.SetTipResolution(16)
            arrow.SetShaftResolution(16)
            arrow.SetTipLength(0.3)
            arrow.SetTipRadius(0.12)
            arrow.SetShaftRadius(0.04)

            dir_vec = force_N / mag
            base_vec = np.array([1.0, 0.0, 0.0])
            axis = np.cross(base_vec, dir_vec)
            axis_norm = np.linalg.norm(axis)
            dot = np.dot(base_vec, dir_vec)
            angle_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

            arrow_len = min(10.0, max(4.0, mag * 0.04))

            transform = vtk.vtkTransform()
            start_pt = center - dir_vec * arrow_len
            transform.Translate(start_pt[0], start_pt[1], start_pt[2])
            if axis_norm > 1e-6:
                axis = axis / axis_norm
                transform.RotateWXYZ(angle_deg, axis[0], axis[1], axis[2])
            elif dot < -0.9999:
                transform.RotateWXYZ(180.0, 0.0, 1.0, 0.0)
            transform.Scale(arrow_len, arrow_len, arrow_len)

            tf_filter = vtk.vtkTransformPolyDataFilter()
            tf_filter.SetTransform(transform)
            tf_filter.SetInputConnection(arrow.GetOutputPort())
            tf_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tf_filter.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.95, 0.25, 0.25)
            actor.GetProperty().SetSpecular(0.4)
            self.renderer.AddActor(actor)
            self._load_arrow_actors.append(actor)

            # 하중 크기 및 경사 각도(tilt_deg) 반영
            # load에 명시적 크기/각도 메타데이터가 있으면 우선 사용, 없으면 방향 벡터에서 계산
            if "magnitude_N" in load and load["magnitude_N"] is not None:
                mag = float(load["magnitude_N"])

            if "tilt_deg" in load and load["tilt_deg"] is not None:
                tilt_deg = float(load["tilt_deg"])
            elif "angle_deg" in load and load["angle_deg"] is not None:
                tilt_deg = float(load["angle_deg"])
            else:
                ref_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                cos_tilt = np.clip(np.dot(dir_vec, ref_down), -1.0, 1.0)
                tilt_deg = float(np.degrees(np.arccos(cos_tilt)))

            text_actor = vtk.vtkBillboardTextActor3D()
            text_actor.SetInput(f"{mag:.1f} N, {tilt_deg:.1f}°")
            text_actor.SetPosition(start_pt[0], start_pt[1], start_pt[2] + 1.0)
            text_prop = text_actor.GetTextProperty()
            text_prop.SetFontSize(7)
            text_prop.SetColor(1.0, 0.3, 0.3)
            text_prop.BoldOn()
            self.renderer.AddActor(text_actor)
            self._load_arrow_actors.append(text_actor)

            # 단 1개만 생성
            break

        self._render()

    # -------------------------------------------------------------------------
    # 필드 및 범위 제어
    # -------------------------------------------------------------------------
    def set_field(self, name: str) -> None:
        """
        Switch active scalar field displayed on the 3D meshes.

        Args:
            name: One of 'von_mises', 'u_mag', 's_max_principal', 's_min_principal', 'E_MPa'.
        """
        if name not in FIELD_DISPLAY_CONFIG:
            return
        self._current_field = name
        cfg = FIELD_DISPLAY_CONFIG[name]
        is_cell = cfg["is_cell"]

        for part_name, mapper in self._mappers.items():
            grid = self._grids.get(part_name)
            if grid is None:
                continue

            if is_cell:
                mapper.SetScalarModeToUseCellFieldData()
                mapper.SelectColorArray(name)
            else:
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray(name)
            mapper.SetScalarVisibility(True)

        self._update_scalar_ranges()
        self._update_scalar_bar_title()
        self._render()

    def set_range_mode(self, mode: str, manual: Optional[Tuple[float, float]] = None) -> None:
        """
        Configure scalar color range mode.

        Args:
            mode: 'global' (all parts shared), 'per_part' (part-specific), 'manual' (user range).
            manual: Tuple of (min_val, max_val) when mode == 'manual'.
        """
        if mode not in ("global", "per_part", "manual"):
            return
        self._range_mode = mode
        if mode == "manual" and manual is not None:
            self._manual_range = manual

        self._update_scalar_ranges()
        self._update_scalar_bar_title()
        self._render()

    def set_active_scalar_part(self, part_name: str) -> None:
        """Set the active part for scalar bar display in 'per_part' mode."""
        if part_name in self._grids:
            self._active_scalar_part = part_name
            self._update_scalar_ranges()
            self._update_scalar_bar_title()
            self._render()

    def _extract_part_values(self, part_name: str) -> np.ndarray:
        """Extract array of active field values (excluding NaNs) for a part."""
        grid = self._grids.get(part_name)
        if grid is None:
            return np.array([], dtype=np.float64)

        is_cell = FIELD_DISPLAY_CONFIG[self._current_field]["is_cell"]
        data = grid.GetCellData() if is_cell else grid.GetPointData()
        arr = data.GetArray(self._current_field)
        if arr is None:
            return np.array([], dtype=np.float64)

        np_arr = numpy_support.vtk_to_numpy(arr)
        if np_arr.ndim > 1:
            np_arr = np.linalg.norm(np_arr, axis=-1)
        valid = np_arr[~np.isnan(np_arr)]
        return valid

    def _update_scalar_ranges(self) -> None:
        """Calculate and apply scalar range to mappers, lookup tables, and scalar bar."""
        if not self._grids:
            return

        part_ranges: Dict[str, Tuple[float, float]] = {}

        if self._range_mode == "manual" and self._manual_range is not None:
            cmin, cmax = self._manual_range
            for p in self._grids:
                part_ranges[p] = (cmin, cmax)
            active_range = (cmin, cmax)

        elif self._range_mode == "global":
            all_vals = []
            for p in self._grids:
                v = self._extract_part_values(p)
                if len(v) > 0:
                    all_vals.append(v)
            if all_vals:
                cat = np.concatenate(all_vals)
                cmin, cmax = float(np.min(cat)), float(np.max(cat))
                if cmin == cmax:
                    cmax = cmin + 1.0
            else:
                cmin, cmax = 0.0, 1.0

            for p in self._grids:
                part_ranges[p] = (cmin, cmax)
            active_range = (cmin, cmax)

        else:  # per_part
            active_range = (0.0, 1.0)
            for p in self._grids:
                v = self._extract_part_values(p)
                if len(v) > 0:
                    pmin, pmax = float(np.min(v)), float(np.max(v))
                    if pmin == pmax:
                        pmax = pmin + 1.0
                else:
                    pmin, pmax = 0.0, 1.0
                part_ranges[p] = (pmin, pmax)
                if p == self._active_scalar_part:
                    active_range = (pmin, pmax)

        # 각 파트별 LUT 및 Mapper에 범위 적용
        for p, (pmin, pmax) in part_ranges.items():
            if p in self._luts:
                self._luts[p].SetTableRange(pmin, pmax)
            if p in self._mappers:
                self._mappers[p].SetScalarRange(pmin, pmax)

        # 스칼라 바 범위 업데이트
        if self._scalar_bar is not None:
            active_lut = self._luts.get(self._active_scalar_part or "", None)
            if active_lut is not None:
                self._scalar_bar.SetLookupTable(active_lut)

    # -------------------------------------------------------------------------
    # 컬러맵 제어
    # -------------------------------------------------------------------------
    def set_colormap(self, name: str) -> None:
        """
        Set colormap style ('rainbow', 'viridis', 'coolwarm').

        Args:
            name: Color palette name.
        """
        if name not in ("rainbow", "viridis", "coolwarm"):
            return
        self._colormap_name = name

        for part_name, lut in self._luts.items():
            table_range = lut.GetTableRange()
            new_lut = self._build_lookup_table(name)
            new_lut.SetTableRange(table_range)
            self._luts[part_name] = new_lut
            if part_name in self._mappers:
                self._mappers[part_name].SetLookupTable(new_lut)

        if self._scalar_bar is not None and self._active_scalar_part in self._luts:
            self._scalar_bar.SetLookupTable(self._luts[self._active_scalar_part])

        self._render()

    def _build_lookup_table(self, name: str) -> vtk.vtkLookupTable:
        """Create a 256-color vtkLookupTable for the requested colormap."""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)

        if name == "rainbow":
            # FEA 관례: Blue(0.6667) -> Cyan -> Green -> Yellow -> Red(0.0)
            lut.SetHueRange(0.6667, 0.0)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.Build()

        elif name == "viridis":
            ctf = vtk.vtkColorTransferFunction()
            ctf.SetColorSpaceToRGB()
            # Viridis 제어점 보간
            viridis_pts = [
                (0.0, 0.267, 0.004, 0.329),
                (0.25, 0.191, 0.407, 0.556),
                (0.5, 0.128, 0.567, 0.551),
                (0.75, 0.369, 0.789, 0.383),
                (1.0, 0.993, 0.906, 0.144),
            ]
            for val, r, g, b in viridis_pts:
                ctf.AddRGBPoint(val, r, g, b)

            lut.Build()
            for i in range(256):
                val = i / 255.0
                rgb = ctf.GetColor(val)
                lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)

        elif name == "coolwarm":
            # 주응력 부호 구분: 음수(압축) 파랑 -> 0 연회색/흰색 -> 양수(인장) 빨강
            ctf = vtk.vtkColorTransferFunction()
            ctf.SetColorSpaceToDiverging()
            ctf.AddRGBPoint(0.0, 0.23, 0.299, 0.754)   # Blue
            ctf.AddRGBPoint(0.5, 0.865, 0.865, 0.865) # Neutral Gray
            ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150) # Red

            lut.Build()
            for i in range(256):
                val = i / 255.0
                rgb = ctf.GetColor(val)
                lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)

        return lut

    # -------------------------------------------------------------------------
    # 변형 스케일 제어
    # -------------------------------------------------------------------------
    def set_deform_scale(self, scale: float) -> None:
        """
        Update mesh deformation amplification factor.

        Recomputes nodal coordinates from preserved original coordinates (non-accumulating).
        """
        self._deform_scale = float(scale)
        for part_name, grid in self._grids.items():
            set_deform_scale(grid, self._deform_scale)
            if part_name in self._linear_grids and self._linear_grids[part_name] is not grid:
                set_deform_scale(self._linear_grids[part_name], self._deform_scale)

        if self._clip_enabled:
            for clipper in self._clippers.values():
                clipper.Modified()

        self._render()

    # -------------------------------------------------------------------------
    # 파트 가시성, 투명도 및 메쉬 모서리선 제어
    # -------------------------------------------------------------------------
    def set_part_visible(self, part_name: str, visible: bool) -> None:
        """Set visibility of an individual part actor."""
        actor = self._actors.get(part_name)
        if actor is not None:
            actor.SetVisibility(visible)
            self._render()

    def set_opacity(self, part_name: str, opacity: float) -> None:
        """Set opacity (0.0 ~ 1.0) of an individual part actor."""
        actor = self._actors.get(part_name)
        if actor is not None:
            actor.GetProperty().SetOpacity(float(opacity))
            self._render()

    def set_edges_visible(self, visible: bool) -> None:
        """Toggle mesh element wireframe edges visibility."""
        self._edges_visible = visible
        for actor in self._actors.values():
            actor.GetProperty().SetEdgeVisibility(1 if visible else 0)
        self._render()

    # -------------------------------------------------------------------------
    # 단면 클립 평면 제어
    # -------------------------------------------------------------------------
    def set_clip_plane(
        self,
        enabled: bool,
        origin: Optional[Tuple[float, float, float]] = None,
        normal: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Enable or disable section clipping on linear mesh grids.

        Args:
            enabled: If True, clips grids with a cutting plane.
            origin: Plane center point (defaults to implant center or mesh center).
            normal: Plane normal direction vector (defaults to (0.0, 1.0, 0.0)).
        """
        self._clip_enabled = enabled
        self._clip_origin = origin
        self._clip_normal = normal

        if not self._grids:
            return

        if enabled:
            # 기본 원점 및 법선 계산
            plane_origin = origin
            if plane_origin is None:
                # 임플란트 파트가 있으면 임플란트 중심, 없으면 전체 메쉬 바운즈 중심
                implant_grid = None
                for name, g in self._grids.items():
                    if "implant" in name.lower() or "fixture" in name.lower():
                        implant_grid = g
                        break
                target_g = implant_grid or next(iter(self._grids.values()))
                b = target_g.GetBounds()
                plane_origin = (
                    0.5 * (b[0] + b[1]),
                    0.5 * (b[2] + b[3]),
                    0.5 * (b[4] + b[5]),
                )

            plane_normal = normal if normal is not None else (0.0, 1.0, 0.0)

            clip_plane = vtk.vtkPlane()
            clip_plane.SetOrigin(*plane_origin)
            clip_plane.SetNormal(*plane_normal)

            for part_name, mapper in self._mappers.items():
                clipper = self._clippers[part_name]
                clipper.SetClipFunction(clip_plane)
                surface_filter = self._clip_surface_filters[part_name]
                surface_filter.Update()
                mapper.SetInputConnection(surface_filter.GetOutputPort())

        else:
            # 클립 해제: 원래 주 그리드로 매퍼 입력 복원
            for part_name, mapper in self._mappers.items():
                grid = self._grids[part_name]
                mapper.SetInputData(grid)

        self._render()

    def _update_pipeline(self) -> None:
        """Update clipping or full mesh pipeline based on current state."""
        if self._clip_enabled:
            self.set_clip_plane(True, self._clip_origin, self._clip_normal)
        else:
            for part_name, mapper in self._mappers.items():
                mapper.SetInputData(self._grids[part_name])

    # -------------------------------------------------------------------------
    # 스칼라 바 (Colorbar) 관리
    # -------------------------------------------------------------------------
    def _init_scalar_bar(self) -> None:
        """Initialize the 2D scalar color bar actor."""
        if self._scalar_bar is not None:
            self.renderer.RemoveActor(self._scalar_bar)

        self._scalar_bar = vtk.vtkScalarBarActor()
        self._scalar_bar.SetOrientationToVertical()
        self._scalar_bar.SetPosition(0.86, 0.15)
        self._scalar_bar.SetWidth(0.10)
        self._scalar_bar.SetHeight(0.70)
        self._scalar_bar.SetNumberOfLabels(5)

        # 텍스트 레이아웃 및 폰트 스타일
        title_prop = self._scalar_bar.GetTitleTextProperty()
        title_prop.SetFontSize(11)
        title_prop.SetBold(True)
        title_prop.SetColor(0.9, 0.9, 0.95)

        label_prop = self._scalar_bar.GetLabelTextProperty()
        label_prop.SetFontSize(10)
        label_prop.SetColor(0.85, 0.85, 0.85)

        self._scalar_bar.SetLabelFormat("%8.2f")

        active_lut = self._luts.get(self._active_scalar_part or "", None)
        if active_lut is not None:
            self._scalar_bar.SetLookupTable(active_lut)

        self.renderer.AddActor(self._scalar_bar)

    def _update_scalar_bar_title(self) -> None:
        """Update scalar bar title with active field name, unit, and part name."""
        if self._scalar_bar is None:
            return

        cfg = FIELD_DISPLAY_CONFIG.get(self._current_field, {"title": self._current_field, "unit": ""})
        title = cfg["title"]
        unit = f"({cfg['unit']})" if cfg["unit"] else ""

        if self._range_mode == "per_part" and self._active_scalar_part:
            full_title = f"[{self._active_scalar_part}]\n{title} {unit}"
        else:
            full_title = f"{title}\n{unit}"

        self._scalar_bar.SetTitle(full_title.strip())

    # -------------------------------------------------------------------------
    # 3D 화면 스크린샷 캡처
    # -------------------------------------------------------------------------
    def save_screenshot(self, path: str, scale: int = 1) -> bool:
        """
        Capture current 3D view including scalar bar and probe labels to PNG image.

        Args:
            path: Target file path (.png).
            scale: Resolution scaling factor (e.g. 1 for native, 2 for 2x resolution).

        Returns:
            bool: True if screenshot was captured and saved successfully, False otherwise.
        """
        if self.renderer is None:
            return False

        rw = self.renderer.GetRenderWindow()
        if rw is None:
            return False

        os_dir = os.path.dirname(os.path.abspath(path))
        if os_dir:
            os.makedirs(os_dir, exist_ok=True)

        try:
            rw.Render()

            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(rw)
            w2i.SetScale(max(1, int(scale)))
            w2i.SetInputBufferTypeToRGBA()
            w2i.ReadFrontBufferOff()
            w2i.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(path))
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            return True
        except Exception:
            return False

