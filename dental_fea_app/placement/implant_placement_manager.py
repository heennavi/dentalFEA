"""
implant_placement_manager.py
다중 임플란트 식립, 선택(1st: Blue, 2nd+: Green), 정렬(높이, 각도, 간격), 
2D MPR 단면 슬라이스(Cutter 채움/외곽선), Abutment 및 Prosthetic Crown 관리 엔진.
"""
import copy
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import vtk

from .implant_generator import (
    ImplantSpec, ImplantLibraryManager,
    AbutmentSpec, generate_abutment_polydata,
    ProstheticSpec, generate_prosthetic_polydata,
    get_material_color_rgb
)


def create_30mm_rgb_axes_actor() -> vtk.vtkAssembly:
    """30mm RGB 축 생성 (X: 빨강, Y: 초록, Z: 파랑)"""
    assembly = vtk.vtkAssembly()
    axis_length = 30.0
    radius = 0.45

    colors = [
        ((1.0, 0.15, 0.15), (axis_length, 0.0, 0.0)),
        ((0.15, 0.95, 0.15), (0.0, axis_length, 0.0)),
        ((0.20, 0.65, 1.00), (0.0, 0.0, axis_length)),
    ]

    for color, endpoint in colors:
        line = vtk.vtkLineSource()
        line.SetPoint1(0.0, 0.0, 0.0)
        line.SetPoint2(endpoint[0], endpoint[1], endpoint[2])
        line.Update()

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(radius)
        tube.SetNumberOfSides(16)
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetAmbient(0.8)
        actor.GetProperty().SetDiffuse(0.6)
        actor.GetProperty().SetLighting(False)
        assembly.AddPart(actor)

    return assembly


@dataclass
class PlacedImplant:
    """식립된 단일 임플란트 인스턴스"""
    id: str
    spec: ImplantSpec
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3)) # [yaw, pitch, roll] deg
    is_selected: bool = False
    sequence_idx: int = 0
    is_visible: bool = True

    # VTK 3D 액터
    actor_3d: Optional[vtk.vtkActor] = None
    axes_assembly: Optional[vtk.vtkAssembly] = None
    raw_polydata: Optional[vtk.vtkPolyData] = None

    # Abutment
    abutment: Optional[AbutmentSpec] = None
    abutment_actor: Optional[vtk.vtkActor] = None
    raw_abutment_polydata: Optional[vtk.vtkPolyData] = None

    # Prosthetic Crown
    prosthetic: Optional[ProstheticSpec] = None
    prosthetic_actor: Optional[vtk.vtkActor] = None
    raw_prosthetic_polydata: Optional[vtk.vtkPolyData] = None


class ImplantPlacementManager:
    """다중 임플란트 식립 및 3D/2D 연동 엔진"""

    def __init__(self, renderer_3d: Optional[vtk.vtkRenderer] = None, renderers_2d: Optional[Dict[str, vtk.vtkRenderer]] = None):
        self.renderer_3d: Optional[vtk.vtkRenderer] = renderer_3d
        self.renderers_2d: Dict[str, vtk.vtkRenderer] = renderers_2d or {}
        self.implants: Dict[str, PlacedImplant] = {}
        self.selected_ids: List[str] = []
        self.is_visible_all: bool = True
        self.gizmo_mode_enabled: bool = True

        # 2D 단면 액터 캐시: {view_name: {implant_id: vtkActor}}
        self.slice_actors_2d: Dict[str, Dict[str, vtk.vtkActor]] = {"axial": {}, "coronal": {}, "sagittal": {}}
        self.slice_fill_actors_2d: Dict[str, Dict[str, vtk.vtkActor]] = {"axial": {}, "coronal": {}, "sagittal": {}}
        self.slice_abutment_actors_2d: Dict[str, Dict[str, vtk.vtkActor]] = {"axial": {}, "coronal": {}, "sagittal": {}}
        self.slice_abutment_fill_actors_2d: Dict[str, Dict[str, vtk.vtkActor]] = {"axial": {}, "coronal": {}, "sagittal": {}}
        self.slice_prosthetic_actors_2d: Dict[str, Dict[str, vtk.vtkActor]] = {"axial": {}, "coronal": {}, "sagittal": {}}
        self.slice_prosthetic_fill_actors_2d: Dict[str, Dict[str, vtk.vtkActor]] = {"axial": {}, "coronal": {}, "sagittal": {}}

        self.current_cursor: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.history_stack: List[Dict[str, Any]] = []
        self.state_mgr: Optional[Any] = None

    def set_state_manager(self, state_mgr: Any):
        self.state_mgr = state_mgr

    def set_renderers_2d(self, renderers_2d: Dict[str, vtk.vtkRenderer]):
        self.renderers_2d = renderers_2d

    def set_renderer_3d(self, renderer_3d: Optional[vtk.vtkRenderer]):
        """3D 독립 뷰어가 열리거나 닫힐 때 3D 렌더러를 동적으로 바인딩/해제합니다."""
        if self.renderer_3d is not None and self.renderer_3d != renderer_3d:
            for imp in self.implants.values():
                if imp.actor_3d:
                    self.renderer_3d.RemoveActor(imp.actor_3d)
                if imp.axes_assembly:
                    self.renderer_3d.RemoveActor(imp.axes_assembly)
                if imp.abutment_actor:
                    self.renderer_3d.RemoveActor(imp.abutment_actor)
                if imp.prosthetic_actor:
                    self.renderer_3d.RemoveActor(imp.prosthetic_actor)

        self.renderer_3d = renderer_3d

        if self.renderer_3d is not None:
            for imp in self.implants.values():
                if imp.actor_3d:
                    self.renderer_3d.AddActor(imp.actor_3d)
                if imp.axes_assembly:
                    self.renderer_3d.AddActor(imp.axes_assembly)
                if imp.abutment_actor:
                    self.renderer_3d.AddActor(imp.abutment_actor)
                if imp.prosthetic_actor:
                    self.renderer_3d.AddActor(imp.prosthetic_actor)
            self._sync_colors_and_axes()

    def get_selected_implant(self) -> Optional[PlacedImplant]:
        if self.selected_ids and self.selected_ids[0] in self.implants:
            return self.implants[self.selected_ids[0]]
        return None

    def save_undo_state(self):
        state = {}
        for imp_id, imp in self.implants.items():
            state[imp_id] = {
                "position": np.copy(imp.position),
                "orientation": np.copy(imp.orientation),
                "spec": copy.deepcopy(imp.spec),
                "abutment": copy.deepcopy(imp.abutment) if imp.abutment else None,
                "prosthetic": copy.deepcopy(imp.prosthetic) if imp.prosthetic else None,
            }
        self.history_stack.append(state)
        if len(self.history_stack) > 25:
            self.history_stack.pop(0)

    def undo(self) -> bool:
        if not self.history_stack:
            return False
        state = self.history_stack.pop()
        for imp_id, imp_state in state.items():
            if imp_id in self.implants:
                imp = self.implants[imp_id]
                imp.position = np.copy(imp_state["position"])
                imp.orientation = np.copy(imp_state["orientation"])
                imp.spec = copy.deepcopy(imp_state["spec"])

                if imp_state.get("abutment"):
                    self.replace_abutment(imp_id, imp_state["abutment"])
                elif imp.abutment:
                    self.remove_abutment(imp_id)

                if imp_state.get("prosthetic"):
                    self.replace_prosthetic(imp_id, imp_state["prosthetic"])
                elif imp.prosthetic:
                    self.remove_prosthetic(imp_id)

                self._update_implant_transform(imp)

        self.update_2d_slice_contours(self.current_cursor)
        if self.state_mgr and hasattr(self.state_mgr, "implants_updated"):
            self.state_mgr.implants_updated.emit()
        return True

    def place_implant(self, spec: ImplantSpec, initial_position: Optional[Tuple[float, float, float]] = None, is_maxilla: bool = False, axial_pitch: float = 0.0, axial_roll: float = 0.0, custom_id: Optional[str] = None) -> str:
        self.save_undo_state()
        pos = initial_position if initial_position is not None else (0.0, 0.0, 0.0)

        imp_id = custom_id if custom_id else f"impl_{len(self.implants)+1}_{int(np.random.randint(1000, 9999))}"
        polydata = ImplantLibraryManager.load_implant_polydata(spec)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(0.72, 0.74, 0.77)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.85)
        prop.SetSpecular(0.4)
        prop.SetSpecularPower(25.0)
        prop.SetInterpolationToPhong()
        prop.BackfaceCullingOff()
        actor.SetVisibility(self.is_visible_all)
        if self.renderer_3d is not None:
            self.renderer_3d.AddActor(actor)

        axes_assembly = create_30mm_rgb_axes_actor()
        axes_assembly.SetVisibility(False)
        if self.renderer_3d is not None:
            self.renderer_3d.AddActor(axes_assembly)

        half_len = spec.length_mm / 2.0
        cur_pos = np.array(pos, dtype=float)

        trans_plane = vtk.vtkTransform()
        trans_plane.Identity()
        trans_plane.RotateX(axial_pitch)
        trans_plane.RotateY(axial_roll)
        norm_vec = np.array(trans_plane.TransformVector(0.0, 0.0, 1.0), dtype=float)
        norm_len = np.linalg.norm(norm_vec)
        norm_vec = (norm_vec / norm_len) if norm_len > 1e-6 else np.array([0.0, 0.0, 1.0])

        if is_maxilla:
            center_pos = cur_pos + (half_len * norm_vec)
            trans_imp = vtk.vtkTransform()
            trans_imp.Identity()
            trans_imp.RotateX(axial_pitch)
            trans_imp.RotateY(axial_roll)
            trans_imp.RotateY(180.0)
            orientation = np.array(trans_imp.GetOrientation(), dtype=float)
        else:
            center_pos = cur_pos - (half_len * norm_vec)
            trans_imp = vtk.vtkTransform()
            trans_imp.Identity()
            trans_imp.RotateX(axial_pitch)
            trans_imp.RotateY(axial_roll)
            orientation = np.array(trans_imp.GetOrientation(), dtype=float)

        placed = PlacedImplant(
            id=imp_id,
            spec=spec,
            position=center_pos,
            orientation=orientation,
            actor_3d=actor,
            axes_assembly=axes_assembly,
            is_selected=False,
            raw_polydata=polydata,
            sequence_idx=len(self.implants) + 1
        )
        self.implants[imp_id] = placed
        self._update_implant_transform(placed)
        self.select_implant(imp_id, multi_select=False)
        self.update_2d_slice_contours(self.current_cursor)
        if self.state_mgr and hasattr(self.state_mgr, "implants_updated"):
            self.state_mgr.implants_updated.emit()
        return imp_id

    def replace_implant(self, implant_id: str, new_spec: ImplantSpec) -> bool:
        if implant_id not in self.implants:
            return False
        self.save_undo_state()
        imp = self.implants[implant_id]
        imp.spec = copy.deepcopy(new_spec)
        poly = ImplantLibraryManager.load_implant_polydata(new_spec)
        imp.raw_polydata = poly
        if imp.actor_3d:
            imp.actor_3d.GetMapper().SetInputData(poly)
        self._update_implant_transform(imp)
        self.update_2d_slice_contours(self.current_cursor)
        if self.state_mgr and hasattr(self.state_mgr, "implants_updated"):
            self.state_mgr.implants_updated.emit()
        return True

    def delete_implant(self, implant_id: str):
        if implant_id in self.implants:
            self.save_undo_state()
            imp = self.implants.pop(implant_id)
            if imp.actor_3d and self.renderer_3d is not None:
                self.renderer_3d.RemoveActor(imp.actor_3d)
            if imp.axes_assembly and self.renderer_3d is not None:
                self.renderer_3d.RemoveActor(imp.axes_assembly)
            if imp.abutment_actor and self.renderer_3d is not None:
                self.renderer_3d.RemoveActor(imp.abutment_actor)
            if imp.prosthetic_actor and self.renderer_3d is not None:
                self.renderer_3d.RemoveActor(imp.prosthetic_actor)

            for view_name in ["axial", "coronal", "sagittal"]:
                ren = self.renderers_2d.get(view_name)
                for d in [self.slice_actors_2d, self.slice_fill_actors_2d,
                          self.slice_abutment_actors_2d, self.slice_abutment_fill_actors_2d,
                          self.slice_prosthetic_actors_2d, self.slice_prosthetic_fill_actors_2d]:
                    if implant_id in d[view_name]:
                        act = d[view_name].pop(implant_id)
                        if ren:
                            ren.RemoveActor(act)

            if implant_id in self.selected_ids:
                self.selected_ids.remove(implant_id)
            self._sync_colors_and_axes()
            self.update_2d_slice_contours(self.current_cursor)
            if self.renderer_3d is not None and self.renderer_3d.GetRenderWindow() is not None:
                self.renderer_3d.GetRenderWindow().Render()
            if self.state_mgr:
                self.state_mgr.request_render_all()
                if hasattr(self.state_mgr, "implants_updated"):
                    self.state_mgr.implants_updated.emit()
                if hasattr(self.state_mgr, "implant_selected"):
                    if not self.selected_ids:
                        self.state_mgr.implant_selected.emit("")
                    else:
                        self.state_mgr.implant_selected.emit(self.selected_ids[-1])

    def get_implant(self, implant_id: str) -> Optional[PlacedImplant]:
        """ID로 식립된 임플란트 인스턴스 반환"""
        return self.implants.get(implant_id)

    def delete_all_implants(self):
        ids = list(self.implants.keys())
        for i_id in ids:
            self.delete_implant(i_id)
        if self.state_mgr:
            if hasattr(self.state_mgr, "implant_selected"):
                self.state_mgr.implant_selected.emit("")
            if hasattr(self.state_mgr, "implants_updated"):
                self.state_mgr.implants_updated.emit()

    def to_dict(self) -> List[Dict[str, Any]]:
        """모든 식립된 임플란트 및 부착물(Abutment, Crown) 정보를 JSON 직렬화 가능한 형태로 반환"""
        res = []
        for imp in self.implants.values():
            item = {
                "id": imp.id,
                "sequence_idx": imp.sequence_idx,
                "is_visible": imp.is_visible,
                "position": [float(p) for p in imp.position],
                "orientation": [float(o) for o in imp.orientation],
                "spec": {
                    "id": imp.spec.id,
                    "name": imp.spec.name,
                    "diameter_mm": float(imp.spec.diameter_mm),
                    "length_mm": float(imp.spec.length_mm),
                    "taper_deg": float(imp.spec.taper_deg),
                    "top_type": imp.spec.top_type,
                    "thread_type": imp.spec.thread_type,
                    "micro_threads": bool(imp.spec.micro_threads),
                    "pitch_mm": float(getattr(imp.spec, "pitch_mm", 0.6)),
                    "depth_mm": float(getattr(imp.spec, "depth_mm", 0.3)),
                    "is_custom": bool(getattr(imp.spec, "is_custom", False)),
                    "file_name": getattr(imp.spec, "file_name", "")
                },
                "abutment": None,
                "prosthetic": None
            }
            if imp.abutment is not None:
                ab = imp.abutment
                item["abutment"] = {
                    "diameter_mm": float(getattr(ab, "diameter_mm", 3.0)),
                    "flare_deg": float(getattr(ab, "flare_deg", 30.0)),
                    "collar_len_mm": float(getattr(ab, "collar_len_mm", 4.0)),
                    "upper_len_mm": float(getattr(ab, "upper_len_mm", 5.0)),
                    "upper_angle_deg": float(getattr(ab, "upper_angle_deg", 0.0)),
                    "upper_dir_deg": float(getattr(ab, "upper_dir_deg", 0.0)),
                    "color_type": getattr(ab, "color_type", "gold"),
                    "hex_height_mm": float(getattr(ab, "hex_height_mm", 3.0))
                }
            if imp.prosthetic is not None:
                pr = imp.prosthetic
                item["prosthetic"] = {
                    "tooth_number": int(getattr(pr, "tooth_number", 16)),
                    "crown_len_mm": float(getattr(pr, "crown_len_mm", 7.5)),
                    "crown_width_mm": float(getattr(pr, "crown_width_mm", 10.5)),
                    "crown_depth_mm": float(getattr(pr, "crown_depth_mm", 11.0)),
                    "rotation_deg": float(getattr(pr, "rotation_deg", 0.0)),
                    "material": getattr(pr, "material", "zirconia"),
                    "crown_type": getattr(pr, "crown_type", "anatomical")
                }
            res.append(item)
        return res

    def from_dict(self, data: List[Dict[str, Any]]):
        """저장된 임플란트 및 부착물 데이터를 복원하여 재식립"""
        self.delete_all_implants()
        for item in data:
            spec_data = item.get("spec", {})
            spec = ImplantSpec(**spec_data)
            pos = np.array(item.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            orient = np.array(item.get("orientation", [0.0, 0.0, 0.0]), dtype=np.float64)
            is_vis = item.get("is_visible", True)

            imp_id = self.place_implant(
                spec,
                initial_position=(float(pos[0]), float(pos[1]), float(pos[2])),
                is_maxilla=False,
                custom_id=item.get("id")
            )
            imp = self.get_implant(imp_id)
            if imp:
                imp.position = pos
                imp.orientation = orient
                imp.is_visible = is_vis
                self._update_implant_transform(imp)

                # Abutment 복원
                ab_data = item.get("abutment")
                if ab_data:
                    ab_spec = AbutmentSpec(**ab_data)
                    self.attach_abutment(imp.id, ab_spec)

                # Prosthetic 복원
                pr_data = item.get("prosthetic")
                if pr_data:
                    pr_spec = ProstheticSpec(**pr_data)
                    self.attach_prosthetic(imp.id, pr_spec)

        self.update_2d_slice_contours(self.current_cursor)
        if self.renderer_3d and self.renderer_3d.GetRenderWindow():
            self.renderer_3d.GetRenderWindow().Render()
        if hasattr(self, "state_mgr") and self.state_mgr:
            self.state_mgr.request_render_all()
            if hasattr(self.state_mgr, "implants_updated"):
                self.state_mgr.implants_updated.emit()

    def select_implant(self, implant_id: Optional[str], multi_select: bool = False):
        if implant_id is None:
            self.clear_selection()
            return
        if implant_id not in self.implants:
            return

        if not multi_select:
            self.selected_ids = [implant_id]
        else:
            if implant_id in self.selected_ids:
                self.selected_ids.remove(implant_id)
            else:
                self.selected_ids.append(implant_id)

        for imp_id, imp in self.implants.items():
            imp.is_selected = (imp_id in self.selected_ids)

        self._sync_colors_and_axes()
        if hasattr(self, "state_mgr") and self.state_mgr is not None:
            self.state_mgr.implant_selected.emit(implant_id)

    def clear_selection(self):
        self.selected_ids.clear()
        for imp in self.implants.values():
            imp.is_selected = False
            self._apply_implant_color(imp, "default")
            if imp.axes_assembly:
                imp.axes_assembly.SetVisibility(False)
        if hasattr(self, "state_mgr") and self.state_mgr is not None:
            self.state_mgr.implant_selected.emit("")

    def _sync_colors_and_axes(self):
        for imp_id, imp in self.implants.items():
            if imp_id not in self.selected_ids:
                self._apply_implant_color(imp, "default")
                if imp.axes_assembly:
                    imp.axes_assembly.SetVisibility(False)
            else:
                idx = self.selected_ids.index(imp_id)
                if idx == 0:
                    self._apply_implant_color(imp, "blue")
                    if imp.axes_assembly:
                        imp.axes_assembly.SetVisibility(self.is_visible_all and imp.is_visible)
                else:
                    self._apply_implant_color(imp, "green")
                    if imp.axes_assembly:
                        imp.axes_assembly.SetVisibility(False)

    def _apply_implant_color(self, imp: PlacedImplant, color_mode: str):
        if not imp.actor_3d:
            return
        prop = imp.actor_3d.GetProperty()
        if color_mode == "blue":
            prop.SetColor(0.20, 0.65, 1.00)
            prop.SetAmbient(0.50)
            prop.SetDiffuse(0.90)
        elif color_mode == "green":
            prop.SetColor(0.20, 0.90, 0.35)
            prop.SetAmbient(0.50)
            prop.SetDiffuse(0.90)
        else:
            prop.SetColor(0.72, 0.74, 0.77)
            prop.SetAmbient(0.25)
            prop.SetDiffuse(0.85)

    def set_implant_visibility(self, implant_id: str, visible: bool):
        if implant_id in self.implants:
            imp = self.implants[implant_id]
            imp.is_visible = visible
            eff_vis = self.is_visible_all and visible
            if imp.actor_3d:
                imp.actor_3d.SetVisibility(eff_vis)
            if imp.axes_assembly:
                imp.axes_assembly.SetVisibility(eff_vis if imp.is_selected else False)
            if imp.abutment_actor:
                imp.abutment_actor.SetVisibility(eff_vis)
            if imp.prosthetic_actor:
                imp.prosthetic_actor.SetVisibility(eff_vis)

            for view_name in ["axial", "coronal", "sagittal"]:
                for d in [self.slice_actors_2d, self.slice_fill_actors_2d,
                          self.slice_abutment_actors_2d, self.slice_abutment_fill_actors_2d,
                          self.slice_prosthetic_actors_2d, self.slice_prosthetic_fill_actors_2d]:
                    if implant_id in d[view_name]:
                        act = d[view_name][implant_id]
                        has_cut = act.GetMapper().GetInput() and act.GetMapper().GetInput().GetNumberOfPoints() > 0
                        act.SetVisibility(eff_vis and has_cut)

    def set_visibility_all(self, visible: bool):
        self.is_visible_all = visible
        for imp_id in self.implants.keys():
            self.set_implant_visibility(imp_id, visible)

    def align_angles_to_reference(self):
        if not self.selected_ids:
            return
        ref_id = self.selected_ids[0]
        ref_orient = np.copy(self.implants[ref_id].orientation)
        self.save_undo_state()
        targets = [imp for i_id, imp in self.implants.items() if i_id != ref_id]
        for imp in targets:
            imp.orientation = np.copy(ref_orient)
            self._update_implant_transform(imp)
            if self.state_mgr:
                self.state_mgr.implant_transformed.emit(imp.id)
        self.update_2d_slice_contours(self.current_cursor)
        if self.state_mgr:
            self.state_mgr.request_render_all()

    def align_heights_to_reference(self):
        if not self.selected_ids:
            return
        ref_id = self.selected_ids[0]
        ref_imp = self.implants[ref_id]
        ref_top_z = ref_imp.position[2] + (ref_imp.spec.length_mm / 2.0)
        self.save_undo_state()
        targets = [imp for i_id, imp in self.implants.items() if i_id != ref_id]
        for imp in targets:
            imp.position[2] = ref_top_z - (imp.spec.length_mm / 2.0)
            self._update_implant_transform(imp)
            if self.state_mgr:
                self.state_mgr.implant_transformed.emit(imp.id)
        self.update_2d_slice_contours(self.current_cursor)
        if self.state_mgr:
            self.state_mgr.request_render_all()

    def align_spacing_sequence(self, spacing_mm: float):
        order = list(self.selected_ids) if len(self.selected_ids) >= 2 else list(self.implants.keys())
        if len(order) < 2:
            return
        self.save_undo_state()
        orig_pos = [np.copy(self.implants[i].position) for i in order]
        for i in range(len(order) - 1):
            curr_id = order[i]
            next_id = order[i + 1]
            dx = orig_pos[i + 1][0] - orig_pos[i][0]
            dy = orig_pos[i + 1][1] - orig_pos[i][1]
            dist_xy = math.hypot(dx, dy)
            dir_x, dir_y = (dx / dist_xy, dy / dist_xy) if dist_xy > 1e-4 else (1.0, 0.0)
            p_curr = self.implants[curr_id].position
            self.implants[next_id].position[0] = p_curr[0] + dir_x * spacing_mm
            self.implants[next_id].position[1] = p_curr[1] + dir_y * spacing_mm
            self._update_implant_transform(self.implants[next_id])
            if self.state_mgr:
                self.state_mgr.implant_transformed.emit(next_id)
        self.update_2d_slice_contours(self.current_cursor)
        if self.state_mgr:
            self.state_mgr.request_render_all()

    # Abutment 관리
    def attach_abutment(self, implant_id: str, spec: AbutmentSpec) -> bool:
        if implant_id not in self.implants:
            return False
        imp = self.implants[implant_id]
        self.save_undo_state()
        imp.abutment = copy.deepcopy(spec)
        abut_pd = generate_abutment_polydata(spec)
        imp.raw_abutment_polydata = abut_pd

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(abut_pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        ar, ag, ab = spec.color_rgb
        actor.GetProperty().SetColor(ar, ag, ab)
        actor.GetProperty().SetAmbient(0.35)
        actor.GetProperty().SetDiffuse(0.85)
        actor.GetProperty().SetSpecular(0.55)
        actor.GetProperty().SetSpecularPower(30.0)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().BackfaceCullingOff()
        actor.SetVisibility(self.is_visible_all and imp.is_visible)
        if self.renderer_3d is not None:
            self.renderer_3d.AddActor(actor)
        imp.abutment_actor = actor

        self._update_implant_transform(imp)
        self.update_2d_slice_contours(self.current_cursor)
        return True

    def remove_abutment(self, implant_id: str) -> bool:
        if implant_id not in self.implants:
            return False
        imp = self.implants[implant_id]
        if imp.abutment is None:
            return False
        self.save_undo_state()
        if imp.prosthetic:
            self.remove_prosthetic(implant_id)
        if imp.abutment_actor and self.renderer_3d is not None:
            self.renderer_3d.RemoveActor(imp.abutment_actor)
        imp.abutment_actor = None
        imp.abutment = None
        imp.raw_abutment_polydata = None

        for view_name in ["axial", "coronal", "sagittal"]:
            ren = self.renderers_2d.get(view_name)
            for d in [self.slice_abutment_actors_2d, self.slice_abutment_fill_actors_2d]:
                if implant_id in d[view_name]:
                    act = d[view_name].pop(implant_id)
                    if ren:
                        ren.RemoveActor(act)
        self.update_2d_slice_contours(self.current_cursor)
        return True

    def replace_abutment(self, implant_id: str, new_spec: AbutmentSpec) -> bool:
        if implant_id not in self.implants:
            return False
        imp = self.implants[implant_id]
        self.save_undo_state()
        imp.abutment = copy.deepcopy(new_spec)
        abut_pd = generate_abutment_polydata(new_spec)
        imp.raw_abutment_polydata = abut_pd
        if not imp.abutment_actor:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(abut_pd)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if self.renderer_3d is not None:
                self.renderer_3d.AddActor(actor)
            imp.abutment_actor = actor
        else:
            imp.abutment_actor.GetMapper().SetInputData(abut_pd)

        ar, ag, ab = new_spec.color_rgb
        imp.abutment_actor.GetProperty().SetColor(ar, ag, ab)
        imp.abutment_actor.SetVisibility(self.is_visible_all and imp.is_visible)

        if imp.prosthetic:
            imp.prosthetic.bottom_diameter_mm = max(1.5, new_spec.collar_top_diameter_mm - 1.0)
            p_pd = generate_prosthetic_polydata(imp.prosthetic)
            imp.raw_prosthetic_polydata = p_pd
            if imp.prosthetic_actor:
                imp.prosthetic_actor.GetMapper().SetInputData(p_pd)

        self._update_implant_transform(imp)
        self.update_2d_slice_contours(self.current_cursor)
        return True

    def remove_all_abutments(self):
        for imp_id in list(self.implants.keys()):
            self.remove_abutment(imp_id)

    # Prosthetic (Crown) 관리
    def attach_prosthetic(self, implant_id: str, spec: ProstheticSpec) -> bool:
        if implant_id not in self.implants:
            return False
        imp = self.implants[implant_id]
        if imp.abutment is None:
            return False
        self.save_undo_state()

        spec.bottom_diameter_mm = max(1.5, imp.abutment.collar_top_diameter_mm - 1.0)
        imp.prosthetic = copy.deepcopy(spec)
        prost_pd = generate_prosthetic_polydata(spec)
        imp.raw_prosthetic_polydata = prost_pd

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(prost_pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        pr, pg, pb = get_material_color_rgb(spec.material)
        actor.GetProperty().SetColor(pr, pg, pb)
        actor.GetProperty().SetAmbient(0.4)
        actor.GetProperty().SetDiffuse(0.85)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().BackfaceCullingOff()
        actor.SetVisibility(self.is_visible_all and imp.is_visible)
        if self.renderer_3d is not None:
            self.renderer_3d.AddActor(actor)
        imp.prosthetic_actor = actor

        self._update_implant_transform(imp)
        self.update_2d_slice_contours(self.current_cursor)
        return True

    def remove_prosthetic(self, implant_id: str) -> bool:
        if implant_id not in self.implants:
            return False
        imp = self.implants[implant_id]
        if imp.prosthetic is None:
            return False
        self.save_undo_state()
        if imp.prosthetic_actor and self.renderer_3d is not None:
            self.renderer_3d.RemoveActor(imp.prosthetic_actor)
        imp.prosthetic_actor = None
        imp.prosthetic = None
        imp.raw_prosthetic_polydata = None

        for view_name in ["axial", "coronal", "sagittal"]:
            ren = self.renderers_2d.get(view_name)
            for d in [self.slice_prosthetic_actors_2d, self.slice_prosthetic_fill_actors_2d]:
                if implant_id in d[view_name]:
                    act = d[view_name].pop(implant_id)
                    if ren:
                        ren.RemoveActor(act)
        self.update_2d_slice_contours(self.current_cursor)
        return True

    def replace_prosthetic(self, implant_id: str, new_spec: ProstheticSpec) -> bool:
        if implant_id not in self.implants:
            return False
        imp = self.implants[implant_id]
        if imp.abutment is None:
            return False
        self.save_undo_state()

        new_spec.bottom_diameter_mm = max(1.5, imp.abutment.collar_top_diameter_mm - 1.0)
        imp.prosthetic = copy.deepcopy(new_spec)
        prost_pd = generate_prosthetic_polydata(new_spec)
        imp.raw_prosthetic_polydata = prost_pd

        if not imp.prosthetic_actor:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(prost_pd)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if self.renderer_3d is not None:
                self.renderer_3d.AddActor(actor)
            imp.prosthetic_actor = actor
        else:
            imp.prosthetic_actor.GetMapper().SetInputData(prost_pd)

        pr, pg, pb = get_material_color_rgb(new_spec.material)
        imp.prosthetic_actor.GetProperty().SetColor(pr, pg, pb)
        imp.prosthetic_actor.SetVisibility(self.is_visible_all and imp.is_visible)

        self._update_implant_transform(imp)
        self.update_2d_slice_contours(self.current_cursor)
        return True

    def remove_all_prosthetics(self):
        for imp_id in list(self.implants.keys()):
            self.remove_prosthetic(imp_id)

    # 3D Transform 적용
    def _update_implant_transform(self, imp: PlacedImplant):
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        half_len = imp.spec.length_mm / 2.0
        transform.Translate(0.0, 0.0, -half_len)
        transform.RotateZ(imp.orientation[0])
        transform.RotateX(imp.orientation[1])
        transform.RotateY(imp.orientation[2])
        transform.Translate(imp.position[0], imp.position[1], imp.position[2])

        if imp.actor_3d:
            imp.actor_3d.SetUserTransform(transform)

        if imp.axes_assembly:
            axes_trans = vtk.vtkTransform()
            axes_trans.PostMultiply()
            axes_trans.RotateZ(imp.orientation[0])
            axes_trans.RotateX(imp.orientation[1])
            axes_trans.RotateY(imp.orientation[2])
            axes_trans.Translate(imp.position[0], imp.position[1], imp.position[2])
            imp.axes_assembly.SetUserTransform(axes_trans)

        if imp.abutment_actor:
            abut_trans = vtk.vtkTransform()
            abut_trans.PostMultiply()
            abut_trans.Translate(0.0, 0.0, half_len)
            abut_trans.RotateZ(imp.orientation[0])
            abut_trans.RotateX(imp.orientation[1])
            abut_trans.RotateY(imp.orientation[2])
            abut_trans.Translate(imp.position[0], imp.position[1], imp.position[2])
            imp.abutment_actor.SetUserTransform(abut_trans)

        if imp.prosthetic_actor and imp.abutment:
            prost_trans = vtk.vtkTransform()
            prost_trans.PostMultiply()
            prost_trans.RotateZ(getattr(imp.prosthetic, 'rotation_deg', 0.0))
            if getattr(imp.abutment, 'angulation_deg', 0.0) != 0.0:
                prost_trans.RotateZ(getattr(imp.abutment, 'angulation_dir_deg', 0.0))
                prost_trans.RotateY(getattr(imp.abutment, 'angulation_deg', 0.0))
                prost_trans.RotateZ(-getattr(imp.abutment, 'angulation_dir_deg', 0.0))
            prost_trans.Translate(0.0, 0.0, half_len + imp.abutment.collar_len_mm)
            prost_trans.RotateZ(imp.orientation[0])
            prost_trans.RotateX(imp.orientation[1])
            prost_trans.RotateY(imp.orientation[2])
            prost_trans.Translate(imp.position[0], imp.position[1], imp.position[2])
            imp.prosthetic_actor.SetUserTransform(prost_trans)

        if hasattr(self, "state_mgr") and self.state_mgr is not None:
            self.state_mgr.implant_transformed.emit(imp.id)

    def get_transformed_polydata(self, imp: PlacedImplant) -> vtk.vtkPolyData:
        poly = imp.raw_polydata or ImplantLibraryManager.load_implant_polydata(imp.spec)
        tf = vtk.vtkTransform()
        tf.PostMultiply()
        half_len = imp.spec.length_mm / 2.0
        tf.Translate(0.0, 0.0, -half_len)
        tf.RotateZ(imp.orientation[0])
        tf.RotateX(imp.orientation[1])
        tf.RotateY(imp.orientation[2])
        tf.Translate(imp.position[0], imp.position[1], imp.position[2])

        flt = vtk.vtkTransformPolyDataFilter()
        flt.SetInputData(poly)
        flt.SetTransform(tf)
        flt.Update()
        return flt.GetOutput()

    def get_transformed_abutment_polydata(self, imp: PlacedImplant) -> Optional[vtk.vtkPolyData]:
        if not imp.abutment:
            return None
        poly = imp.raw_abutment_polydata or generate_abutment_polydata(imp.abutment)
        half_len = imp.spec.length_mm / 2.0
        tf = vtk.vtkTransform()
        tf.PostMultiply()
        tf.Translate(0.0, 0.0, half_len)
        tf.RotateZ(imp.orientation[0])
        tf.RotateX(imp.orientation[1])
        tf.RotateY(imp.orientation[2])
        tf.Translate(imp.position[0], imp.position[1], imp.position[2])

        flt = vtk.vtkTransformPolyDataFilter()
        flt.SetInputData(poly)
        flt.SetTransform(tf)
        flt.Update()
        return flt.GetOutput()

    def get_transformed_prosthetic_polydata(self, imp: PlacedImplant) -> Optional[vtk.vtkPolyData]:
        if not imp.prosthetic or not imp.abutment:
            return None
        poly = imp.raw_prosthetic_polydata or generate_prosthetic_polydata(imp.prosthetic)
        half_len = imp.spec.length_mm / 2.0
        tf = vtk.vtkTransform()
        tf.PostMultiply()
        tf.RotateZ(getattr(imp.prosthetic, 'rotation_deg', 0.0))
        if getattr(imp.abutment, 'angulation_deg', 0.0) != 0.0:
            tf.RotateZ(getattr(imp.abutment, 'angulation_dir_deg', 0.0))
            tf.RotateY(getattr(imp.abutment, 'angulation_deg', 0.0))
            tf.RotateZ(-getattr(imp.abutment, 'angulation_dir_deg', 0.0))
        tf.Translate(0.0, 0.0, half_len + imp.abutment.collar_len_mm)
        tf.RotateZ(imp.orientation[0])
        tf.RotateX(imp.orientation[1])
        tf.RotateY(imp.orientation[2])
        tf.Translate(imp.position[0], imp.position[1], imp.position[2])

        flt = vtk.vtkTransformPolyDataFilter()
        flt.SetInputData(poly)
        flt.SetTransform(tf)
        flt.Update()
        return flt.GetOutput()

    def _create_2d_slice_polys(self, cut_pd: vtk.vtkPolyData, view_name: str, cursor: Tuple[float, float, float]) -> Tuple[vtk.vtkPolyData, vtk.vtkPolyData, bool]:
        """절단면 폴리데이터로부터 해당 2D 뷰의 슬라이스 평면에 맞춘 외곽선 및 채움 다각형 생성"""
        num_pts = cut_pd.GetNumberOfPoints()
        if num_pts == 0:
            return vtk.vtkPolyData(), vtk.vtkPolyData(), False

        cx, cy, cz = cursor
        pts = cut_pd.GetPoints()

        # 1. 2D 평면(u, v) 좌표계로 매핑하여 임시 2D 폴리데이터 생성
        pts_2d = vtk.vtkPoints()
        pts_2d.SetNumberOfPoints(num_pts)
        for i in range(num_pts):
            px, py, pz = pts.GetPoint(i)
            if view_name == "axial":
                pts_2d.SetPoint(i, px, py, 0.0)
            elif view_name == "coronal":
                pts_2d.SetPoint(i, px, pz, 0.0)
            else:  # sagittal
                pts_2d.SetPoint(i, py, pz, 0.0)

        poly_temp = vtk.vtkPolyData()
        poly_temp.SetPoints(pts_2d)
        poly_temp.SetLines(cut_pd.GetLines())
        poly_temp.SetPolys(cut_pd.GetPolys())

        # 2. vtkContourTriangulator로 2D 내부 면 삼각화 채움 생성
        stripper = vtk.vtkStripper()
        stripper.SetInputData(poly_temp)
        stripper.Update()
        triangulator = vtk.vtkContourTriangulator()
        triangulator.SetInputData(stripper.GetOutput())
        triangulator.Update()
        tri_output = triangulator.GetOutput()

        # 3. 채움 다각형 정점들을 해당 뷰의 3D 슬라이스 평면 월드 좌표로 복원 (약간 앞 오프셋)
        filled_pd = vtk.vtkPolyData()
        if tri_output.GetNumberOfPoints() > 0:
            tri_pts = tri_output.GetPoints()
            num_tri_pts = tri_pts.GetNumberOfPoints()
            filled_pts_3d = vtk.vtkPoints()
            filled_pts_3d.SetNumberOfPoints(num_tri_pts)
            for i in range(num_tri_pts):
                u, v, _ = tri_pts.GetPoint(i)
                if view_name == "axial":
                    filled_pts_3d.SetPoint(i, u, v, cz + 0.15)
                elif view_name == "coronal":
                    filled_pts_3d.SetPoint(i, u, cy - 0.15, v)
                else:  # sagittal
                    filled_pts_3d.SetPoint(i, cx + 0.15, u, v)
            filled_pd.SetPoints(filled_pts_3d)
            filled_pd.SetPolys(tri_output.GetPolys())

        # 4. 외곽선 다각형 정점들을 해당 뷰의 3D 슬라이스 평면 월드 좌표로 복원
        outline_pd = vtk.vtkPolyData()
        outline_pts_3d = vtk.vtkPoints()
        outline_pts_3d.SetNumberOfPoints(num_pts)
        for i in range(num_pts):
            u, v, _ = pts_2d.GetPoint(i)
            if view_name == "axial":
                outline_pts_3d.SetPoint(i, u, v, cz + 0.20)
            elif view_name == "coronal":
                outline_pts_3d.SetPoint(i, u, cy - 0.20, v)
            else:  # sagittal
                outline_pts_3d.SetPoint(i, cx + 0.20, u, v)
        outline_pd.SetPoints(outline_pts_3d)
        outline_pd.SetLines(cut_pd.GetLines())
        outline_pd.SetPolys(cut_pd.GetPolys())

        return outline_pd, filled_pd, True

    # 2D MPR 슬라이스 평면 윤곽선/채움
    def update_2d_slice_contours(self, cursor: Tuple[float, float, float]):
        self.current_cursor = cursor
        cx, cy, cz = cursor
        planes = {
            "axial": ((0.0, 0.0, cz), (0.0, 0.0, 1.0)),
            "coronal": ((0.0, cy, 0.0), (0.0, 1.0, 0.0)),
            "sagittal": ((cx, 0.0, 0.0), (1.0, 0.0, 0.0))
        }

        for view_name, (origin, normal) in planes.items():
            ren = self.renderers_2d.get(view_name)
            if not ren:
                continue

            plane = vtk.vtkPlane()
            plane.SetOrigin(origin[0], origin[1], origin[2])
            plane.SetNormal(normal[0], normal[1], normal[2])

            for imp_id, imp in self.implants.items():
                eff_vis = self.is_visible_all and imp.is_visible
                # 1. Implant Body
                world_pd = self.get_transformed_polydata(imp)
                cutter = vtk.vtkCutter()
                cutter.SetInputData(world_pd)
                cutter.SetCutFunction(plane)
                cutter.Update()
                poly_2d, filled_2d, has_cut = self._create_2d_slice_polys(cutter.GetOutput(), view_name, cursor)

                if imp_id not in self.slice_fill_actors_2d[view_name]:
                    m_f = vtk.vtkPolyDataMapper()
                    a_f = vtk.vtkActor()
                    a_f.SetMapper(m_f)
                    a_f.GetProperty().SetColor(1.0, 1.0, 1.0)
                    a_f.GetProperty().SetAmbient(1.0)
                    a_f.GetProperty().SetDiffuse(0.0)
                    a_f.GetProperty().SetLighting(False)
                    ren.AddActor(a_f)
                    self.slice_fill_actors_2d[view_name][imp_id] = a_f

                f_act = self.slice_fill_actors_2d[view_name][imp_id]
                f_act.GetMapper().SetInputData(filled_2d)
                f_act.SetVisibility(eff_vis and has_cut and filled_2d.GetNumberOfPolys() > 0)

                if imp_id not in self.slice_actors_2d[view_name]:
                    m_l = vtk.vtkPolyDataMapper()
                    a_l = vtk.vtkActor()
                    a_l.SetMapper(m_l)
                    a_l.GetProperty().SetColor(0.0, 0.0, 0.0)
                    a_l.GetProperty().SetLineWidth(1.8)
                    a_l.GetProperty().SetAmbient(1.0)
                    a_l.GetProperty().SetDiffuse(0.0)
                    a_l.GetProperty().SetLighting(False)
                    ren.AddActor(a_l)
                    self.slice_actors_2d[view_name][imp_id] = a_l

                l_act = self.slice_actors_2d[view_name][imp_id]
                l_act.GetMapper().SetInputData(poly_2d)
                l_act.SetVisibility(eff_vis and has_cut)

                # 2. Abutment
                if imp.abutment is not None:
                    a_world_pd = self.get_transformed_abutment_polydata(imp)
                    a_cut = vtk.vtkCutter()
                    a_cut.SetInputData(a_world_pd)
                    a_cut.SetCutFunction(plane)
                    a_cut.Update()
                    a_poly_2d, a_filled_2d, a_has_cut = self._create_2d_slice_polys(a_cut.GetOutput(), view_name, cursor)

                    ar, ag, ab = imp.abutment.color_rgb
                    if imp_id not in self.slice_abutment_fill_actors_2d[view_name]:
                        amf = vtk.vtkPolyDataMapper()
                        aaf = vtk.vtkActor()
                        aaf.SetMapper(amf)
                        aaf.GetProperty().SetColor(ar, ag, ab)
                        aaf.GetProperty().SetOpacity(0.8)
                        aaf.GetProperty().SetAmbient(1.0)
                        aaf.GetProperty().SetDiffuse(0.0)
                        aaf.GetProperty().SetLighting(False)
                        ren.AddActor(aaf)
                        self.slice_abutment_fill_actors_2d[view_name][imp_id] = aaf
                    aaf_act = self.slice_abutment_fill_actors_2d[view_name][imp_id]
                    aaf_act.GetProperty().SetColor(ar, ag, ab)
                    aaf_act.GetMapper().SetInputData(a_filled_2d)
                    aaf_act.SetVisibility(eff_vis and a_has_cut and a_filled_2d.GetNumberOfPolys() > 0)

                    if imp_id not in self.slice_abutment_actors_2d[view_name]:
                        aml = vtk.vtkPolyDataMapper()
                        aal = vtk.vtkActor()
                        aal.SetMapper(aml)
                        aal.GetProperty().SetColor(ar, ag, ab)
                        aal.GetProperty().SetLineWidth(2.0)
                        aal.GetProperty().SetAmbient(1.0)
                        aal.GetProperty().SetDiffuse(0.0)
                        aal.GetProperty().SetLighting(False)
                        ren.AddActor(aal)
                        self.slice_abutment_actors_2d[view_name][imp_id] = aal
                    aal_act = self.slice_abutment_actors_2d[view_name][imp_id]
                    aal_act.GetProperty().SetColor(ar, ag, ab)
                    aal_act.GetMapper().SetInputData(a_poly_2d)
                    aal_act.SetVisibility(eff_vis and a_has_cut)
                else:
                    if imp_id in self.slice_abutment_actors_2d[view_name]:
                        self.slice_abutment_actors_2d[view_name][imp_id].SetVisibility(False)
                    if imp_id in self.slice_abutment_fill_actors_2d[view_name]:
                        self.slice_abutment_fill_actors_2d[view_name][imp_id].SetVisibility(False)

                # 3. Prosthetic Crown
                if imp.prosthetic is not None and imp.abutment is not None:
                    p_world_pd = self.get_transformed_prosthetic_polydata(imp)
                    p_cut = vtk.vtkCutter()
                    p_cut.SetInputData(p_world_pd)
                    p_cut.SetCutFunction(plane)
                    p_cut.Update()
                    p_poly_2d, p_filled_2d, p_has_cut = self._create_2d_slice_polys(p_cut.GetOutput(), view_name, cursor)

                    pr, pg, pb = get_material_color_rgb(imp.prosthetic.material)
                    if imp_id not in self.slice_prosthetic_fill_actors_2d[view_name]:
                        pmf = vtk.vtkPolyDataMapper()
                        paf = vtk.vtkActor()
                        paf.SetMapper(pmf)
                        paf.GetProperty().SetColor(pr, pg, pb)
                        paf.GetProperty().SetOpacity(0.85)
                        paf.GetProperty().SetAmbient(1.0)
                        paf.GetProperty().SetDiffuse(0.0)
                        paf.GetProperty().SetLighting(False)
                        ren.AddActor(paf)
                        self.slice_prosthetic_fill_actors_2d[view_name][imp_id] = paf
                    paf_act = self.slice_prosthetic_fill_actors_2d[view_name][imp_id]
                    paf_act.GetProperty().SetColor(pr, pg, pb)
                    paf_act.GetMapper().SetInputData(p_filled_2d)
                    paf_act.SetVisibility(eff_vis and p_has_cut and p_filled_2d.GetNumberOfPolys() > 0)

                    if imp_id not in self.slice_prosthetic_actors_2d[view_name]:
                        pml = vtk.vtkPolyDataMapper()
                        pal = vtk.vtkActor()
                        pal.SetMapper(pml)
                        pal.GetProperty().SetColor(pr, pg, pb)
                        pal.GetProperty().SetLineWidth(2.0)
                        pal.GetProperty().SetAmbient(1.0)
                        pal.GetProperty().SetDiffuse(0.0)
                        pal.GetProperty().SetLighting(False)
                        ren.AddActor(pal)
                        self.slice_prosthetic_actors_2d[view_name][imp_id] = pal
                    pal_act = self.slice_prosthetic_actors_2d[view_name][imp_id]
                    pal_act.GetProperty().SetColor(pr, pg, pb)
                    pal_act.GetMapper().SetInputData(p_poly_2d)
                    pal_act.SetVisibility(eff_vis and p_has_cut)
                else:
                    if imp_id in self.slice_prosthetic_actors_2d[view_name]:
                        self.slice_prosthetic_actors_2d[view_name][imp_id].SetVisibility(False)
                    if imp_id in self.slice_prosthetic_fill_actors_2d[view_name]:
                        self.slice_prosthetic_fill_actors_2d[view_name][imp_id].SetVisibility(False)

    def find_implant_by_actor(self, actor: Optional[vtk.vtkProp]) -> Optional[PlacedImplant]:
        if not actor:
            return None
        for imp in self.implants.values():
            if imp.actor_3d == actor or imp.abutment_actor == actor or imp.prosthetic_actor == actor:
                return imp
            for d in [self.slice_actors_2d, self.slice_fill_actors_2d,
                      self.slice_abutment_actors_2d, self.slice_abutment_fill_actors_2d,
                      self.slice_prosthetic_actors_2d, self.slice_prosthetic_fill_actors_2d]:
                for view_dict in d.values():
                    if view_dict.get(imp.id) == actor:
                        return imp
        return None

    def find_implant_by_screen_ray(self, renderer: vtk.vtkRenderer, screen_x: int, screen_y: int, threshold_px: float = 24.0) -> Optional[PlacedImplant]:
        if not self.implants or not renderer:
            return None

        # 1. vtkPropPicker로 클릭된 액터 직접 검출 (가장 정확하고 즉각적)
        picker = vtk.vtkPropPicker()
        if picker.Pick(screen_x, screen_y, 0, renderer):
            picked_prop = picker.GetViewProp()
            hit = self.find_implant_by_actor(picked_prop)
            if hit:
                return hit

        # 2. 메쉬 바깥 약간의 여유 영역 클릭을 위한 화면 선분 거리 검출
        best_imp = None
        min_dist_px = float(threshold_px)

        for imp in self.implants.values():
            if not imp.is_visible or not self.is_visible_all:
                continue

            # 로컬 회전 행렬 계산 (World axis)
            yaw, pitch, roll = imp.orientation
            cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
            cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))

            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
            Ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
            R = Rz.dot(Rx).dot(Ry)
            axis_world = R.dot(np.array([0.0, 0.0, 1.0]))

            half_len = imp.spec.length_mm / 2.0
            p_top = imp.position + axis_world * half_len
            p_bot = imp.position - axis_world * half_len

            def _to_screen(pt):
                renderer.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
                renderer.WorldToDisplay()
                d = renderer.GetDisplayPoint()
                return d[0], d[1]

            s_top = _to_screen(p_top)
            s_bot = _to_screen(p_bot)

            # 점과 선분 사이의 2D 화면 거리
            dx = s_bot[0] - s_top[0]
            dy = s_bot[1] - s_top[1]
            l2 = dx * dx + dy * dy
            if l2 == 0:
                dist = math.hypot(screen_x - s_top[0], screen_y - s_top[1])
            else:
                t = max(0.0, min(1.0, ((screen_x - s_top[0]) * dx + (screen_y - s_top[1]) * dy) / l2))
                proj_x = s_top[0] + t * dx
                proj_y = s_top[1] + t * dy
                dist = math.hypot(screen_x - proj_x, screen_y - proj_y)

            if dist < min_dist_px:
                min_dist_px = dist
                best_imp = imp

        return best_imp

    def find_implant_by_slice_pos(self, view_name: str, world_pt: Tuple[float, float, float], threshold_mm: float = 6.0) -> Optional[PlacedImplant]:
        """2D 슬라이스 평면 상에서 클릭된 월드 좌표에 인접한 임플란트 검색 (임플란트 회전축 반영)"""
        if not self.implants:
            return None
        wx, wy, wz = world_pt
        p_click = np.array([wx, wy, wz], dtype=float)
        best_imp = None
        min_dist = float(threshold_mm)

        for imp in self.implants.values():
            if not imp.is_visible or not self.is_visible_all:
                continue

            # 로컬 회전 행렬 계산 (World axis)
            yaw, pitch, roll = imp.orientation
            cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
            cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))

            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
            Ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
            R = Rz.dot(Rx).dot(Ry)
            axis_world = R.dot(np.array([0.0, 0.0, 1.0]))

            half_len = imp.spec.length_mm / 2.0
            p_top = imp.position + axis_world * half_len
            p_bot = imp.position - axis_world * half_len

            # 선분 p_top <-> p_bot 과 점 p_click 사이의 최단 거리 계산
            seg_vec = p_bot - p_top
            seg_len2 = float(np.dot(seg_vec, seg_vec))
            if seg_len2 < 1e-6:
                dist_to_axis = float(np.linalg.norm(p_click - p_top))
            else:
                t = max(0.0, min(1.0, float(np.dot(p_click - p_top, seg_vec) / seg_len2)))
                closest_pt = p_top + t * seg_vec
                dist_to_axis = float(np.linalg.norm(p_click - closest_pt))

            radius = imp.spec.diameter_mm / 2.0
            effective_dist = max(0.0, dist_to_axis - radius)
            if effective_dist <= threshold_mm and effective_dist < min_dist:
                min_dist = effective_dist
                best_imp = imp

        return best_imp
