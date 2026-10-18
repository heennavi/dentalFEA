"""
2D Transform Gizmo for Axial, Coronal, and Sagittal MPR Viewports.
Renders 2D translation axes (thickness 2.0) and rotation ring arc (thickness 2.0),
supporting interactive dragging that converts 2D manipulation into precise 3D implant translation and rotation.
"""
import math
import numpy as np
import vtk
from typing import Optional, Tuple, Dict, Any
from .implant_placement_manager import ImplantPlacementManager, PlacedImplant


def get_tag_center(tag):
    return getattr(tag, 'position', np.zeros(3))


class Implant2DTransformGizmo:
    """
    Interactive 2D Transform Gizmo attached to a 2D MPR viewport (Axial, Coronal, or Sagittal).
    
    Coordinate Mappings:
    - Axial (view_name='axial'): 
        2D X = World X (Red), 2D Y = World Y (Green), Rotation = Yaw (Z-axis rotation, Blue ring)
    - Coronal (view_name='coronal'): 
        2D X = World X (Red), 2D Y = World Z (Blue), Rotation = Pitch/Roll (Y-axis rotation, Green ring)
    - Sagittal (view_name='sagittal'): 
        2D X = World Y (Green), 2D Y = World Z (Blue), Rotation = Pitch/Roll (X-axis rotation, Red ring)
    """

    def __init__(self, view_name: str, renderer: vtk.vtkRenderer, placement_mgr: ImplantPlacementManager):
        self.view_name = view_name
        self.renderer = renderer
        self.placement_mgr = placement_mgr

        self.attached_implant: Optional[PlacedImplant] = None
        self.attached_graft: Optional[Any] = None
        self.attached_tag: Optional[Any] = None
        self.bone_graft_mgr: Optional[Any] = None
        self.is_attached: bool = False
        self.is_dragging: bool = False

        self.active_handle: Optional[str] = None # 'axis_h', 'axis_v', 'center', 'rot_ring'
        self.drag_start_screen: Tuple[int, int] = (0, 0)
        self.drag_start_pos_3d: np.ndarray = np.zeros(3)
        self.drag_start_rot_3d: np.ndarray = np.zeros(3)
        self.drag_start_angle_deg: float = 0.0

        # Colors
        self.color_h = (0.20, 0.85, 0.30) if view_name == 'sagittal' else (0.95, 0.25, 0.25) # Green in Sagittal, Red otherwise
        self.color_v = (0.20, 0.85, 0.30) if view_name == 'axial' else (0.20, 0.60, 0.95) # Green / Blue
        self.color_ring = (0.20, 0.60, 0.95) if view_name == 'axial' else ((0.20, 0.85, 0.30) if view_name == 'coronal' else (0.95, 0.25, 0.25))
        self.color_center = (0.98, 0.98, 1.0)
        self.color_highlight = (1.0, 0.90, 0.20) # Bright yellow

        self.axis_length_mm = 14.0
        self.ring_radius_mm = 11.0
        self.slice_coord: Optional[float] = None

        self.actors: Dict[str, vtk.vtkActor] = {}
        self._build_actors()
        self.set_visible(False)

    def _build_actors(self):
        """Creates VTK 2D line and polygon actors for the gizmo handles."""
        self.actors['axis_h'] = self._create_line_actor(self.color_h)
        self.actors['axis_v'] = self._create_line_actor(self.color_v)
        self.actors['center'] = self._create_line_actor(self.color_center)
        self.actors['rot_ring'] = self._create_line_actor(self.color_ring)
        self.actors['rot_handle'] = self._create_line_actor(self.color_ring)

        for actor in self.actors.values():
            self.renderer.AddActor(actor)

    def _create_line_actor(self, color: Tuple[float, float, float]) -> vtk.vtkActor:
        pd = vtk.vtkPolyData()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(color[0], color[1], color[2])
        prop.SetLineWidth(2.0)
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetLighting(False)
        return actor

    def set_visible(self, visible: bool):
        for actor in self.actors.values():
            actor.SetVisibility(visible)

    def _to_3d_point(self, u: float, v: float) -> Tuple[float, float, float]:
        """2D 뷰포트 평면 좌표 (u, v)를 해당 뷰의 3D 월드 좌표로 정확히 변환"""
        if self.view_name == "axial":
            depth = self.slice_coord if self.slice_coord is not None else (self.attached_implant.position[2] if self.attached_implant else 0.0)
            return (u, v, depth + 0.60)
        elif self.view_name == "coronal":
            depth = self.slice_coord if self.slice_coord is not None else (self.attached_implant.position[1] if self.attached_implant else 0.0)
            return (u, depth - 0.60, v)
        else:  # sagittal
            depth = self.slice_coord if self.slice_coord is not None else (self.attached_implant.position[0] if self.attached_implant else 0.0)
            return (depth + 0.60, u, v)

    def attach_implant(self, imp: Optional[PlacedImplant], current_cursor: Optional[Tuple[float, float, float]] = None):
        """Attaches gizmo to an implant and updates geometry in 2D viewport coordinates."""
        if not imp:
            self.detach()
            return

        if current_cursor is not None:
            cx, cy, cz = current_cursor
            if self.view_name == "axial":
                self.slice_coord = cz
            elif self.view_name == "coronal":
                self.slice_coord = cy
            else:
                self.slice_coord = cx

        self.attached_graft = None
        self.attached_tag = None
        self.bone_graft_mgr = None
        self.attached_implant = imp
        self.is_attached = True
        self.update_geometry(current_cursor)
        self.set_visible(self.placement_mgr.is_visible_all)

    def attach_graft(self, spec: Optional[Any], bone_graft_mgr: Optional[Any] = None):
        """Attaches gizmo to a bone graft specification and updates geometry in 2D viewport coordinates."""
        if not spec:
            self.detach()
            return

        self.attached_implant = None
        self.attached_tag = None
        self.attached_graft = spec
        self.bone_graft_mgr = bone_graft_mgr
        self.is_attached = True
        self.update_geometry()
        self.set_visible(getattr(spec, 'is_visible', True))

    def attach_tag(self, spec: Optional[Any], bone_graft_mgr: Optional[Any] = None):
        """Attaches gizmo to a bone tag/screw specification and updates geometry in 2D viewport coordinates."""
        if not spec:
            self.detach()
            return

        self.attached_implant = None
        self.attached_graft = None
        self.attached_tag = spec
        self.bone_graft_mgr = bone_graft_mgr
        self.is_attached = True
        self.update_geometry()
        self.set_visible(getattr(spec, 'is_visible', True))

    def detach(self):
        """Detaches gizmo and hides actors."""
        self.attached_implant = None
        self.attached_graft = None
        self.attached_tag = None
        self.bone_graft_mgr = None
        self.is_attached = False
        self.is_dragging = False
        self.active_handle = None
        self.set_visible(False)

    def get_2d_center(self) -> Tuple[float, float]:
        """Returns the (u, v) center of the attached implant, graft, or tag in 2D viewport plane coordinates."""
        if self.attached_implant:
            pos = self.attached_implant.position
        elif self.attached_graft:
            pos = self.attached_graft.center
        elif self.attached_tag:
            pos = get_tag_center(self.attached_tag)
        else:
            return 0.0, 0.0

        if self.view_name == "axial":
            return float(pos[0]), float(pos[1])
        elif self.view_name == "coronal":
            return float(pos[0]), float(pos[2])
        elif self.view_name == "sagittal":
            return float(pos[1]), float(pos[2])
        return 0.0, 0.0

    def get_2d_angle(self) -> float:
        """Returns current 2D planar rotation angle in degrees."""
        if self.attached_implant:
            rot = self.attached_implant.orientation
            if self.view_name == "axial":
                return float(rot[0]) # Yaw (Z-axis)
            elif self.view_name == "coronal":
                return -float(rot[2]) # Roll (Y-axis): VTK RotateY is CW in XZ, so 2D angle is -rot[2]
            elif self.view_name == "sagittal":
                return float(rot[1]) # Pitch (X-axis)
        elif self.attached_graft:
            rot = getattr(self.attached_graft, 'rotation', np.zeros(3))
            if self.view_name == "axial":
                return float(rot[2]) # Z-axis rotation
            elif self.view_name == "coronal":
                return -float(rot[1]) # Y-axis rotation: CW in XZ
            elif self.view_name == "sagittal":
                return float(rot[0]) # X-axis rotation
        elif self.attached_tag:
            d = getattr(self.attached_tag, 'direction', np.array([0.0, 0.0, -1.0]))
            if self.view_name == "axial":
                return math.degrees(math.atan2(float(d[1]), float(d[0])))
            elif self.view_name == "coronal":
                return math.degrees(math.atan2(float(d[2]), float(d[0])))
            elif self.view_name == "sagittal":
                return math.degrees(math.atan2(float(d[2]), float(d[1])))
        return 0.0

    def update_geometry(self, current_slice_coords: Optional[Tuple[float, float, float]] = None):
        """Updates polydata geometry of the gizmo based on current implant, graft, or tag position & angle."""
        if not self.is_attached or (not self.attached_implant and not self.attached_graft and not self.attached_tag):
            return

        if current_slice_coords is not None:
            cx, cy, cz = current_slice_coords
            if self.view_name == "axial":
                self.slice_coord = cz
            elif self.view_name == "coronal":
                self.slice_coord = cy
            else:
                self.slice_coord = cx

        u0, v0 = self.get_2d_center()
        ang_rad = math.radians(self.get_2d_angle())
        cos_a = math.cos(ang_rad)
        sin_a = math.sin(ang_rad)

        L = self.axis_length_mm
        R = self.ring_radius_mm

        # 1. Horizontal Axis (Rotated along 2D angle)
        pts_h = vtk.vtkPoints()
        pts_h.InsertNextPoint(*self._to_3d_point(u0, v0))
        pts_h.InsertNextPoint(*self._to_3d_point(u0 + L * cos_a, v0 + L * sin_a))
        # Barb 1
        b1_u = u0 + (L - 2.5) * cos_a - 1.2 * sin_a
        b1_v = v0 + (L - 2.5) * sin_a + 1.2 * cos_a
        pts_h.InsertNextPoint(*self._to_3d_point(b1_u, b1_v))
        pts_h.InsertNextPoint(*self._to_3d_point(u0 + L * cos_a, v0 + L * sin_a))
        # Barb 2
        b2_u = u0 + (L - 2.5) * cos_a + 1.2 * sin_a
        b2_v = v0 + (L - 2.5) * sin_a - 1.2 * cos_a
        pts_h.InsertNextPoint(*self._to_3d_point(b2_u, b2_v))

        lines_h = vtk.vtkCellArray()
        lines_h.InsertNextCell(2)
        lines_h.InsertCellPoint(0)
        lines_h.InsertCellPoint(1)
        lines_h.InsertNextCell(3)
        lines_h.InsertCellPoint(2)
        lines_h.InsertCellPoint(3)
        lines_h.InsertCellPoint(4)

        pd_h = vtk.vtkPolyData()
        pd_h.SetPoints(pts_h)
        pd_h.SetLines(lines_h)
        self.actors['axis_h'].GetMapper().SetInputData(pd_h)

        # 2. Vertical Axis (Rotated along 2D angle + 90 deg)
        pts_v = vtk.vtkPoints()
        pts_v.InsertNextPoint(*self._to_3d_point(u0, v0))
        pts_v.InsertNextPoint(*self._to_3d_point(u0 - L * sin_a, v0 + L * cos_a))
        # Barb 1
        vb1_u = u0 - (L - 2.5) * sin_a - 1.2 * cos_a
        vb1_v = v0 + (L - 2.5) * cos_a - 1.2 * sin_a
        pts_v.InsertNextPoint(*self._to_3d_point(vb1_u, vb1_v))
        pts_v.InsertNextPoint(*self._to_3d_point(u0 - L * sin_a, v0 + L * cos_a))
        # Barb 2
        vb2_u = u0 - (L - 2.5) * sin_a + 1.2 * cos_a
        vb2_v = v0 + (L - 2.5) * cos_a + 1.2 * sin_a
        pts_v.InsertNextPoint(*self._to_3d_point(vb2_u, vb2_v))

        lines_v = vtk.vtkCellArray()
        lines_v.InsertNextCell(2)
        lines_v.InsertCellPoint(0)
        lines_v.InsertCellPoint(1)
        lines_v.InsertNextCell(3)
        lines_v.InsertCellPoint(2)
        lines_v.InsertCellPoint(3)
        lines_v.InsertCellPoint(4)

        pd_v = vtk.vtkPolyData()
        pd_v.SetPoints(pts_v)
        pd_v.SetLines(lines_v)
        self.actors['axis_v'].GetMapper().SetInputData(pd_v)

        # 3. Center square handle
        cw = 1.6
        pts_c = vtk.vtkPoints()
        pts_c.InsertNextPoint(*self._to_3d_point(u0 - cw * cos_a + cw * sin_a, v0 - cw * sin_a - cw * cos_a))
        pts_c.InsertNextPoint(*self._to_3d_point(u0 + cw * cos_a + cw * sin_a, v0 + cw * sin_a - cw * cos_a))
        pts_c.InsertNextPoint(*self._to_3d_point(u0 + cw * cos_a - cw * sin_a, v0 + cw * sin_a + cw * cos_a))
        pts_c.InsertNextPoint(*self._to_3d_point(u0 - cw * cos_a - cw * sin_a, v0 - cw * sin_a + cw * cos_a))
        lines_c = vtk.vtkCellArray()
        lines_c.InsertNextCell(5)
        lines_c.InsertCellPoint(0)
        lines_c.InsertCellPoint(1)
        lines_c.InsertCellPoint(2)
        lines_c.InsertCellPoint(3)
        lines_c.InsertCellPoint(0)
        pd_c = vtk.vtkPolyData()
        pd_c.SetPoints(pts_c)
        pd_c.SetLines(lines_c)
        self.actors['center'].GetMapper().SetInputData(pd_c)

        # 4. Rotation Ring (Full 360 circle arc)
        num_segments = 48
        pts_r = vtk.vtkPoints()
        for i in range(num_segments + 1):
            theta = 2.0 * math.pi * (i / float(num_segments))
            rx = u0 + R * math.cos(theta)
            ry = v0 + R * math.sin(theta)
            pts_r.InsertNextPoint(*self._to_3d_point(rx, ry))

        lines_r = vtk.vtkCellArray()
        lines_r.InsertNextCell(num_segments + 1)
        for i in range(num_segments + 1):
            lines_r.InsertCellPoint(i)

        pd_r = vtk.vtkPolyData()
        pd_r.SetPoints(pts_r)
        pd_r.SetLines(lines_r)
        self.actors['rot_ring'].GetMapper().SetInputData(pd_r)

        # 5. Rotation Knob Handle (Located on the ring at angle + 45 deg)
        phi = ang_rad + math.pi / 4.0
        knob_u = u0 + R * math.cos(phi)
        knob_v = v0 + R * math.sin(phi)
        kw = 1.6
        pts_k = vtk.vtkPoints()
        pts_k.InsertNextPoint(*self._to_3d_point(knob_u - kw, knob_v - kw))
        pts_k.InsertNextPoint(*self._to_3d_point(knob_u + kw, knob_v - kw))
        pts_k.InsertNextPoint(*self._to_3d_point(knob_u + kw, knob_v + kw))
        pts_k.InsertNextPoint(*self._to_3d_point(knob_u - kw, knob_v + kw))
        lines_k = vtk.vtkCellArray()
        lines_k.InsertNextCell(5)
        lines_k.InsertCellPoint(0)
        lines_k.InsertCellPoint(1)
        lines_k.InsertCellPoint(2)
        lines_k.InsertCellPoint(3)
        lines_k.InsertCellPoint(0)
        pd_k = vtk.vtkPolyData()
        pd_k.SetPoints(pts_k)
        pd_k.SetLines(lines_k)
        self.actors['rot_handle'].GetMapper().SetInputData(pd_k)

    def find_handle_by_screen_pos(self, screen_x: int, screen_y: int, threshold_px: float = 14.0) -> Optional[str]:
        """Finds closest gizmo handle to screen position in this 2D viewport."""
        if not self.is_attached or (not self.attached_implant and not self.attached_graft and not self.attached_tag):
            return None

        u0, v0 = self.get_2d_center()
        ang_rad = math.radians(self.get_2d_angle())
        cos_a = math.cos(ang_rad)
        sin_a = math.sin(ang_rad)

        L = self.axis_length_mm
        R = self.ring_radius_mm

        # Project 2D world points to display screen
        def _to_screen(u, v):
            pt = self._to_3d_point(u, v)
            self.renderer.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
            self.renderer.WorldToDisplay()
            disp = self.renderer.GetDisplayPoint()
            return disp[0], disp[1]

        # 1. Check center handle
        sc_cx, sc_cy = _to_screen(u0, v0)
        if math.hypot(screen_x - sc_cx, screen_y - sc_cy) < 12.0:
            return 'center'

        # 2. Check Rotation Knob
        phi = ang_rad + math.pi / 4.0
        sc_kx, sc_ky = _to_screen(u0 + R * math.cos(phi), v0 + R * math.sin(phi))
        if math.hypot(screen_x - sc_kx, screen_y - sc_ky) < 14.0:
            return 'rot_handle'

        # 3. Check Horizontal axis
        sc_hx, sc_hy = _to_screen(u0 + L * cos_a, v0 + L * sin_a)
        d_h = self._point_to_segment_dist(screen_x, screen_y, sc_cx, sc_cy, sc_hx, sc_hy)
        if d_h < threshold_px:
            return 'axis_h'

        # 4. Check Vertical axis
        sc_vx, sc_vy = _to_screen(u0 - L * sin_a, v0 + L * cos_a)
        d_v = self._point_to_segment_dist(screen_x, screen_y, sc_cx, sc_cy, sc_vx, sc_vy)
        if d_v < threshold_px:
            return 'axis_v'

        # 5. Check Rotation Ring
        sc_rx, sc_ry = _to_screen(u0 + R * cos_a, v0 + R * sin_a)
        rad_px = math.hypot(sc_rx - sc_cx, sc_ry - sc_cy)
        dist_from_center = math.hypot(screen_x - sc_cx, screen_y - sc_cy)
        if abs(dist_from_center - rad_px) < threshold_px:
            return 'rot_ring'

        return None

    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2) -> float:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def highlight_handle(self, handle: Optional[str]) -> bool:
        """Sets highlight color for hovered/active handle."""
        changed = False
        colors = {
            'axis_h': self.color_highlight if handle == 'axis_h' else self.color_h,
            'axis_v': self.color_highlight if handle == 'axis_v' else self.color_v,
            'center': self.color_highlight if handle == 'center' else self.color_center,
            'rot_ring': self.color_highlight if handle in ['rot_ring', 'rot_handle'] else self.color_ring,
            'rot_handle': self.color_highlight if handle in ['rot_ring', 'rot_handle'] else self.color_ring,
        }
        for name, col in colors.items():
            prop = self.actors[name].GetProperty()
            old_col = prop.GetColor()
            if old_col != col:
                prop.SetColor(col[0], col[1], col[2])
                changed = True
        return changed

    def start_drag(self, handle: str, screen_x: int, screen_y: int) -> bool:
        """Starts dragging a gizmo handle."""
        if not self.is_attached or (not self.attached_implant and not self.attached_graft and not self.attached_tag):
            return False

        self.is_dragging = True
        self.active_handle = handle
        self.drag_start_screen = (screen_x, screen_y)
        if self.attached_implant:
            self.drag_start_pos_3d = np.copy(self.attached_implant.position)
            self.drag_start_rot_3d = np.copy(self.attached_implant.orientation)
        elif self.attached_graft:
            self.drag_start_pos_3d = np.copy(self.attached_graft.center)
            self.drag_start_rot_3d = np.copy(getattr(self.attached_graft, 'rotation', np.zeros(3)))
        elif self.attached_tag:
            self.drag_start_pos_3d = np.copy(self.attached_tag.position)
            self.drag_start_rot_3d = np.copy(getattr(self.attached_tag, 'direction', np.array([0.0, 0.0, -1.0])))
            self.drag_start_center_3d = get_tag_center(self.attached_tag)

        u0, v0 = self.get_2d_center()
        pt0 = self._to_3d_point(u0, v0)
        self.renderer.SetWorldPoint(pt0[0], pt0[1], pt0[2], 1.0)
        self.renderer.WorldToDisplay()
        disp = self.renderer.GetDisplayPoint()
        self.drag_start_angle_deg = math.degrees(math.atan2(screen_y - disp[1], screen_x - disp[0]))

        self.highlight_handle(handle)
        return True

    def update_drag(self, screen_x: int, screen_y: int) -> bool:
        """Calculates 2D translation/rotation and updates 3D implant, graft, or tag position & orientation."""
        if not self.is_dragging or (not self.attached_implant and not self.attached_graft and not self.attached_tag) or not self.active_handle:
            return False

        # 1. Translation Dragging (Along Local Axes)
        if self.active_handle in ['axis_h', 'axis_v', 'center']:
            self.renderer.SetDisplayPoint(screen_x, screen_y, 0.0)
            self.renderer.DisplayToWorld()
            w_curr = self.renderer.GetWorldPoint()

            self.renderer.SetDisplayPoint(self.drag_start_screen[0], self.drag_start_screen[1], 0.0)
            self.renderer.DisplayToWorld()
            w_start = self.renderer.GetWorldPoint()

            if self.view_name == "axial":
                du_mm = w_curr[0] - w_start[0]
                dv_mm = w_curr[1] - w_start[1]
            elif self.view_name == "coronal":
                du_mm = w_curr[0] - w_start[0]
                dv_mm = w_curr[2] - w_start[2]
            else:  # sagittal
                du_mm = w_curr[1] - w_start[1]
                dv_mm = w_curr[2] - w_start[2]

            ang_rad = math.radians(self.get_2d_angle())
            cos_a = math.cos(ang_rad)
            sin_a = math.sin(ang_rad)

            if self.active_handle == 'axis_h':
                # Project mouse movement onto local horizontal axis vector (cos_a, sin_a)
                proj = du_mm * cos_a + dv_mm * sin_a
                move_2d_x = proj * cos_a
                move_2d_y = proj * sin_a
            elif self.active_handle == 'axis_v':
                # Project mouse movement onto local vertical axis vector (-sin_a, cos_a)
                proj = -du_mm * sin_a + dv_mm * cos_a
                move_2d_x = -proj * sin_a
                move_2d_y = proj * cos_a
            else: # 'center'
                move_2d_x = du_mm
                move_2d_y = dv_mm

            new_pos = np.copy(self.drag_start_pos_3d)

            if self.view_name == "axial":
                new_pos[0] += move_2d_x
                new_pos[1] += move_2d_y
            elif self.view_name == "coronal":
                new_pos[0] += move_2d_x
                new_pos[2] += move_2d_y
            elif self.view_name == "sagittal":
                new_pos[1] += move_2d_x
                new_pos[2] += move_2d_y

            if self.attached_implant:
                self.attached_implant.position = new_pos
                self.placement_mgr._update_implant_transform(self.attached_implant)
            elif self.attached_graft:
                self.attached_graft.center = new_pos
                if self.bone_graft_mgr:
                    self.bone_graft_mgr.current_roi_center = new_pos
                    self.bone_graft_mgr.update_roi_controls(
                        new_pos, self.attached_graft.size, getattr(self.attached_graft, 'rotation', np.zeros(3)),
                        visible=self.bone_graft_mgr.handles_visible
                    )
                    if self.bone_graft_mgr.on_geometry_changed_callback:
                        import inspect
                        sig = inspect.signature(self.bone_graft_mgr.on_geometry_changed_callback)
                        if len(sig.parameters) >= 3:
                            self.bone_graft_mgr.on_geometry_changed_callback(
                                new_pos, self.attached_graft.size, getattr(self.attached_graft, 'rotation', np.zeros(3))
                            )
                        else:
                            self.bone_graft_mgr.on_geometry_changed_callback(new_pos, self.attached_graft.size)
            elif self.attached_tag:
                new_center = np.copy(self.drag_start_center_3d)
                if self.view_name == "axial":
                    new_center[0] += move_2d_x
                    new_center[1] += move_2d_y
                elif self.view_name == "coronal":
                    new_center[0] += move_2d_x
                    new_center[2] += move_2d_y
                elif self.view_name == "sagittal":
                    new_center[1] += move_2d_x
                    new_center[2] += move_2d_y

                total_len = float(self.attached_tag.length_mm) + float(self.attached_tag.head_height_mm)
                ref_pt = getattr(self.attached_tag, 'reference_point', 'head').lower()
                if ref_pt == "head":
                    self.attached_tag.position = new_center - (total_len / 2.0) * self.attached_tag.direction
                else:
                    self.attached_tag.position = new_center + (total_len / 2.0) * self.attached_tag.direction

                if self.bone_graft_mgr:
                    self.bone_graft_mgr._update_tag_3d_actor(self.attached_tag)
                    if hasattr(self.bone_graft_mgr, 'tag_gizmo') and self.bone_graft_mgr.tag_gizmo:
                        self.bone_graft_mgr.tag_gizmo.update_transform()
                    self.bone_graft_mgr.update_slice_views(self.bone_graft_mgr.current_cursor)

            self.update_geometry()
            return True

        # 2. Rotation Dragging (Intuitive Planar Angular Motion)
        elif self.active_handle in ['rot_ring', 'rot_handle']:
            u0, v0 = self.get_2d_center()
            pt3 = self._to_3d_point(u0, v0)
            self.renderer.SetWorldPoint(pt3[0], pt3[1], pt3[2], 1.0)
            self.renderer.WorldToDisplay()
            disp = self.renderer.GetDisplayPoint()
            curr_angle_deg = math.degrees(math.atan2(screen_y - disp[1], screen_x - disp[0]))

            d_theta = curr_angle_deg - self.drag_start_angle_deg

            if self.attached_implant:
                new_rot = np.copy(self.drag_start_rot_3d)
                if self.view_name == "axial":
                    new_rot[0] = (self.drag_start_rot_3d[0] + d_theta) % 360.0
                elif self.view_name == "coronal":
                    new_rot[2] = (self.drag_start_rot_3d[2] - d_theta) % 360.0
                elif self.view_name == "sagittal":
                    new_rot[1] = (self.drag_start_rot_3d[1] + d_theta) % 360.0

                self.attached_implant.orientation = new_rot
                self.placement_mgr._update_implant_transform(self.attached_implant)
            elif self.attached_graft:
                new_rot = np.copy(self.drag_start_rot_3d)
                if self.view_name == "axial":
                    new_rot[2] = ((self.drag_start_rot_3d[2] - d_theta + 180.0) % 360.0) - 180.0
                elif self.view_name == "coronal":
                    new_rot[1] = ((self.drag_start_rot_3d[1] - d_theta + 180.0) % 360.0) - 180.0
                elif self.view_name == "sagittal":
                    new_rot[0] = ((self.drag_start_rot_3d[0] + d_theta + 180.0) % 360.0) - 180.0

                self.attached_graft.rotation = new_rot
                if self.bone_graft_mgr:
                    self.bone_graft_mgr.current_roi_rotation = new_rot
                    self.bone_graft_mgr.update_roi_controls(
                        self.attached_graft.center, self.attached_graft.size, new_rot,
                        visible=self.bone_graft_mgr.handles_visible
                    )
                    if self.bone_graft_mgr.on_geometry_changed_callback:
                        import inspect
                        sig = inspect.signature(self.bone_graft_mgr.on_geometry_changed_callback)
                        if len(sig.parameters) >= 3:
                            self.bone_graft_mgr.on_geometry_changed_callback(
                                self.attached_graft.center, self.attached_graft.size, new_rot
                            )
                        else:
                            self.bone_graft_mgr.on_geometry_changed_callback(self.attached_graft.center, self.attached_graft.size)
            elif self.attached_tag:
                curr_dir = np.copy(self.drag_start_rot_3d)
                rad = math.radians(-d_theta if self.view_name == "axial" else d_theta)
                cos_t, sin_t = math.cos(rad), math.sin(rad)
                new_dir = np.copy(curr_dir)
                if self.view_name == "axial":
                    new_dir[0] = curr_dir[0] * cos_t - curr_dir[1] * sin_t
                    new_dir[1] = curr_dir[0] * sin_t + curr_dir[1] * cos_t
                elif self.view_name == "coronal":
                    new_dir[0] = curr_dir[0] * cos_t - curr_dir[2] * sin_t
                    new_dir[2] = curr_dir[0] * sin_t + curr_dir[2] * cos_t
                elif self.view_name == "sagittal":
                    new_dir[1] = curr_dir[1] * cos_t - curr_dir[2] * sin_t
                    new_dir[2] = curr_dir[1] * sin_t + curr_dir[2] * cos_t

                norm = np.linalg.norm(new_dir)
                if norm > 1e-6:
                    new_dir = new_dir / norm
                self.attached_tag.direction = new_dir

                total_len = float(self.attached_tag.length_mm) + float(self.attached_tag.head_height_mm)
                ref_pt = getattr(self.attached_tag, 'reference_point', 'head').lower()
                if ref_pt == "head":
                    self.attached_tag.position = self.drag_start_center_3d - (total_len / 2.0) * new_dir
                else:
                    self.attached_tag.position = self.drag_start_center_3d + (total_len / 2.0) * new_dir

                if self.bone_graft_mgr:
                    self.bone_graft_mgr._update_tag_3d_actor(self.attached_tag)
                    if hasattr(self.bone_graft_mgr, 'tag_gizmo') and self.bone_graft_mgr.tag_gizmo:
                        self.bone_graft_mgr.tag_gizmo.update_transform()
                    self.bone_graft_mgr.update_slice_views(self.bone_graft_mgr.current_cursor)

            self.update_geometry()
            return True

        return False

    def end_drag(self):
        """Finalizes dragging and resets active state."""
        self.is_dragging = False
        self.active_handle = None
        self.highlight_handle(None)
        self.update_geometry()
        if self.attached_graft and self.bone_graft_mgr:
            if self.attached_graft.id in self.bone_graft_mgr.grafts_history:
                self.bone_graft_mgr.apply_bone_graft(self.attached_graft)
