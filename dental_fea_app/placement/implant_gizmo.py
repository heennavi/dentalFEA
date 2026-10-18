"""
Interactive 3D Transform Gizmo for Dental Implant Placement.
Provides visual Translation (bidirectional X/Y/Z arrows) and Rotation (Yaw/Pitch/Roll perpendicular rings) handles
with hover brightness highlighting and live drag transformation.
"""
import math
import numpy as np
import vtk
from typing import Optional, Tuple, Dict, Any

from .implant_placement_manager import ImplantPlacementManager, PlacedImplant


class ImplantTransformGizmo:
    """3D In-Scene Interactive Gizmo for translating and rotating placed implants."""

    def __init__(self, renderer: vtk.vtkRenderer, placement_mgr: ImplantPlacementManager):
        self.renderer = renderer
        self.placement_mgr = placement_mgr
        self.active_implant_id: Optional[str] = None
        self.mode: str = "translate" # "translate" or "rotate"
        self.is_dragging: bool = False
        self.hovered_axis: Optional[str] = None # 'x', 'y', 'z', etc.

        # Gizmo assembly & actor dictionaries
        self.assembly = vtk.vtkAssembly()
        self.handle_actors: Dict[str, vtk.vtkActor] = {}
        self.base_colors = {
            "x": (1.0, 0.22, 0.22),
            "y": (0.22, 0.95, 0.28),
            "z": (0.25, 0.65, 1.00),
        }
        self.highlight_colors = {
            "x": (1.0, 0.70, 0.70),
            "y": (0.70, 1.00, 0.75),
            "z": (0.70, 0.90, 1.00),
        }

        self._build_gizmo_geometry()
        self.assembly.SetVisibility(False)
        self.renderer.AddActor(self.assembly)

        # Interaction tracking
        self.drag_start_screen = (0, 0)
        self.drag_start_pos = np.zeros(3)
        self.drag_start_orient = np.zeros(3)

    @property
    def is_attached(self) -> bool:
        """Returns True if the gizmo is actively attached to a valid implant and visible."""
        return (
            self.active_implant_id is not None
            and self.active_implant_id in self.placement_mgr.implants
            and (self.assembly.GetVisibility() == 1)
        )

    def _build_gizmo_geometry(self):
        """Builds bidirectional translation arrows and perpendicular rotation rings."""
        axis_len = 18.0
        cone_len = 4.5
        tube_r = 0.55
        cone_r = 1.35

        # --- 1. Translation: X, Y, Z Bidirectional Arrows ---
        # Each axis has arrows on BOTH ends (+ and - directions)
        axes_info = [
            ("tx", "x", (1.0, 0.0, 0.0)),
            ("ty", "y", (0.0, 1.0, 0.0)),
            ("tz", "z", (0.0, 0.0, 1.0)),
        ]

        for handle_id, axis_key, direction in axes_info:
            dx, dy, dz = direction
            
            # Central through-shaft from -axis_len to +axis_len
            line = vtk.vtkLineSource()
            line.SetPoint1(-dx * (axis_len - cone_len), -dy * (axis_len - cone_len), -dz * (axis_len - cone_len))
            line.SetPoint2(dx * (axis_len - cone_len), dy * (axis_len - cone_len), dz * (axis_len - cone_len))
            line.Update()

            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(tube_r)
            tube.SetNumberOfSides(16)
            tube.Update()

            # Positive cone tip
            cone_pos = vtk.vtkConeSource()
            cone_pos.SetHeight(cone_len)
            cone_pos.SetRadius(cone_r)
            cone_pos.SetResolution(20)
            cone_pos.SetDirection(dx, dy, dz)
            cone_pos.SetCenter(dx * (axis_len - cone_len / 2.0), dy * (axis_len - cone_len / 2.0), dz * (axis_len - cone_len / 2.0))
            cone_pos.Update()

            # Negative cone tip (opposite end)
            cone_neg = vtk.vtkConeSource()
            cone_neg.SetHeight(cone_len)
            cone_neg.SetRadius(cone_r)
            cone_neg.SetResolution(20)
            cone_neg.SetDirection(-dx, -dy, -dz)
            cone_neg.SetCenter(-dx * (axis_len - cone_len / 2.0), -dy * (axis_len - cone_len / 2.0), -dz * (axis_len - cone_len / 2.0))
            cone_neg.Update()

            append = vtk.vtkAppendPolyData()
            append.AddInputData(tube.GetOutput())
            append.AddInputData(cone_pos.GetOutput())
            append.AddInputData(cone_neg.GetOutput())
            append.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(append.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            c = self.base_colors[axis_key]
            actor.GetProperty().SetColor(c[0], c[1], c[2])
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            actor.GetProperty().SetLighting(False)

            self.handle_actors[handle_id] = actor
            self.assembly.AddPart(actor)

        # --- 2. Rotation: Perpendicular Rings for each axis ---
        # Z-rotation (Yaw): Ring on XY plane (perpendicular to Z axis)
        # X-rotation (Pitch): Ring on YZ plane (perpendicular to X axis)
        # Y-rotation (Roll): Ring on XZ plane (perpendicular to Y axis)
        ring_radius = 16.0
        ring_r_tube = 0.50
        rings_info = [
            ("rz", "z", 0.0, 0.0, 0.0),       # Yaw (XY plane, perpendicular to Z)
            ("rx", "x", 0.0, 90.0, 0.0),      # Pitch (YZ plane, perpendicular to X)
            ("ry", "y", 90.0, 0.0, 0.0),      # Roll (XZ plane, perpendicular to Y)
        ]

        for handle_id, axis_key, rx, ry, rz in rings_info:
            circle = vtk.vtkRegularPolygonSource()
            circle.SetNumberOfSides(64)
            circle.SetRadius(ring_radius)
            circle.SetCenter(0, 0, 0)
            circle.GeneratePolygonOff() # Wire loop only
            circle.Update()

            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(circle.GetOutputPort())
            tube.SetRadius(ring_r_tube)
            tube.SetNumberOfSides(16)
            tube.Update()

            trans = vtk.vtkTransform()
            trans.RotateX(rx)
            trans.RotateY(ry)
            trans.RotateZ(rz)

            tf_poly = vtk.vtkTransformPolyDataFilter()
            tf_poly.SetInputConnection(tube.GetOutputPort())
            tf_poly.SetTransform(trans)
            tf_poly.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tf_poly.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            c = self.base_colors[axis_key]
            actor.GetProperty().SetColor(c[0], c[1], c[2])
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            actor.GetProperty().SetLighting(False)

            self.handle_actors[handle_id] = actor
            self.assembly.AddPart(actor)

    def attach_implant(self, imp: PlacedImplant, mode: Optional[str] = None):
        """Attaches gizmo to the given placed implant."""
        if imp is None:
            self.detach()
            return

        self.active_implant_id = imp.id
        self._update_handles_visibility()
        self._sync_gizmo_transform(imp)
        self.assembly.SetVisibility(1)

    def attach_to_implant(self, implant_id: Optional[str], mode: str = "all"):
        """Compatibility wrapper for attach_implant."""
        if not implant_id or implant_id not in self.placement_mgr.implants:
            self.detach()
            return
        imp = self.placement_mgr.implants[implant_id]
        self.attach_implant(imp, mode=mode)

    def detach(self):
        """Detaches gizmo and hides it from the scene."""
        self.active_implant_id = None
        self.is_dragging = False
        self.hovered_axis = None
        self.assembly.SetVisibility(0)

    def set_mode(self, mode: str):
        """Sets gizmo mode (displays all handles simultaneously)."""
        self.mode = "all"
        self._update_handles_visibility()

    def _update_handles_visibility(self):
        """Shows all translation arrows and rotation rings simultaneously."""
        for actor in self.handle_actors.values():
            actor.SetVisibility(1)

    def update_geometry(self):
        """Updates gizmo transformation to follow the active implant."""
        if self.active_implant_id and self.active_implant_id in self.placement_mgr.implants:
            imp = self.placement_mgr.implants[self.active_implant_id]
            self._sync_gizmo_transform(imp)

    def update_transform(self):
        """Alias for update_geometry."""
        self.update_geometry()

    def _sync_gizmo_transform(self, imp: PlacedImplant):
        """Moves gizmo center to implant center."""
        trans = vtk.vtkTransform()
        trans.PostMultiply()
        trans.RotateZ(imp.orientation[0])
        trans.RotateX(imp.orientation[1])
        trans.RotateY(imp.orientation[2])
        trans.Translate(imp.position[0], imp.position[1], imp.position[2])
        self.assembly.SetUserTransform(trans)

    def find_handle_by_actor(self, actor: Optional[vtk.vtkProp]) -> Optional[str]:
        """Returns handle_id if actor belongs to the gizmo."""
        if not actor:
            return None
        for handle_id, act in self.handle_actors.items():
            if act == actor:
                return handle_id
        return None

    def _get_rotation_matrix(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """Returns 3x3 rotation matrix matching VTK PostMultiply RotateZ(yaw)->RotateX(pitch)->RotateY(roll)."""
        trans = vtk.vtkTransform()
        trans.PostMultiply()
        trans.RotateZ(yaw)
        trans.RotateX(pitch)
        trans.RotateY(roll)
        m = trans.GetMatrix()
        mat = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                mat[i, j] = m.GetElement(i, j)
        return mat

    def _matrix_to_euler_angles(self, mat: np.ndarray) -> np.ndarray:
        """Extracts (yaw, pitch, roll) in degrees from 3x3 rotation matrix."""
        # mat = Ry(roll) * Rx(pitch) * Rz(yaw)
        # sy_cx = mat[0, 2], -sx = mat[1, 2], cy_cx = mat[2, 2]
        # cx_sz = mat[1, 0],  cx_cz = mat[1, 1]
        sin_pitch = -mat[1, 2]
        sin_pitch = max(-1.0, min(1.0, sin_pitch))
        pitch = math.degrees(math.asin(sin_pitch))

        cos_pitch = math.sqrt(max(0.0, mat[0, 2]**2 + mat[2, 2]**2))
        if cos_pitch > 1e-5:
            yaw = math.degrees(math.atan2(mat[1, 0], mat[1, 1]))
            roll = math.degrees(math.atan2(mat[0, 2], mat[2, 2]))
        else:
            # Gimbal lock at pitch = +-90 deg
            yaw = math.degrees(math.atan2(-mat[0, 1], mat[0, 0]))
            roll = 0.0

        return np.array([yaw % 360.0, pitch % 360.0, roll % 360.0])

    def find_handle_by_screen_pos(self, screen_x: int, screen_y: int, renderer: vtk.vtkRenderer, threshold_px: float = 24.0) -> Optional[str]:
        """Finds closest gizmo handle within screen distance threshold using local rotated handle coordinates."""
        if not self.is_attached or not self.active_implant_id:
            return None

        imp = self.placement_mgr.implants.get(self.active_implant_id)
        if not imp:
            return None

        best_handle = None
        min_dist = threshold_px
        center = imp.position
        R = self._get_rotation_matrix(imp.orientation[0], imp.orientation[1], imp.orientation[2])

        # Test local translation handle points
        t_local = {
            "tx": [np.array([18.0, 0, 0]), np.array([-18.0, 0, 0]), np.array([10.0, 0, 0]), np.array([-10.0, 0, 0])],
            "ty": [np.array([0, 18.0, 0]), np.array([0, -18.0, 0]), np.array([0, 10.0, 0]), np.array([0, -10.0, 0])],
            "tz": [np.array([0, 0, 18.0]), np.array([0, 0, -18.0]), np.array([0, 0, 10.0]), np.array([0, 0, -10.0])],
        }

        # Test local rotation ring circle points (radius 16mm)
        r_local = {"rx": [], "ry": [], "rz": []}
        r_rad = 16.0
        for ang_deg in range(0, 360, 20):
            rad = math.radians(ang_deg)
            # rz: XY plane (perpendicular to local Z)
            r_local["rz"].append(np.array([r_rad * math.cos(rad), r_rad * math.sin(rad), 0.0]))
            # rx: YZ plane (perpendicular to local X)
            r_local["rx"].append(np.array([0.0, r_rad * math.cos(rad), r_rad * math.sin(rad)]))
            # ry: XZ plane (perpendicular to local Y)
            r_local["ry"].append(np.array([r_rad * math.cos(rad), 0.0, r_rad * math.sin(rad)]))

        all_local = {**t_local, **r_local}

        for handle_id, local_pts in all_local.items():
            for l_pt in local_pts:
                world_pt = center + R.dot(l_pt)
                renderer.SetWorldPoint(world_pt[0], world_pt[1], world_pt[2], 1.0)
                renderer.WorldToDisplay()
                disp = renderer.GetDisplayPoint()
                dist = math.hypot(disp[0] - screen_x, disp[1] - screen_y)
                if dist < min_dist:
                    min_dist = dist
                    best_handle = handle_id

        return best_handle

    def highlight_handle(self, hover_actor: Optional[vtk.vtkActor], screen_x: Optional[int] = None, screen_y: Optional[int] = None, renderer: Optional[vtk.vtkRenderer] = None) -> bool:
        """Slightly brightens the hovered handle. Returns True if visual state changed."""
        if not self.is_attached:
            return False

        hovered_handle_id = self.find_handle_by_actor(hover_actor)
        if not hovered_handle_id and screen_x is not None and screen_y is not None and renderer is not None:
            hovered_handle_id = self.find_handle_by_screen_pos(screen_x, screen_y, renderer, threshold_px=22.0)

        if hovered_handle_id == self.hovered_axis:
            return False

        self.hovered_axis = hovered_handle_id
        for handle_id, actor in self.handle_actors.items():
            axis_key = handle_id[1] # 'x', 'y', 'z'
            if handle_id == hovered_handle_id:
                c = self.highlight_colors[axis_key]
                actor.GetProperty().SetColor(c[0], c[1], c[2])
            else:
                c = self.base_colors[axis_key]
                actor.GetProperty().SetColor(c[0], c[1], c[2])

        return True

    def start_drag(self, actor: Optional[vtk.vtkProp], screen_x: int, screen_y: int, renderer: vtk.vtkRenderer) -> bool:
        """Begins interactive gizmo drag manipulation if a gizmo handle was picked."""
        if not self.is_attached:
            return False

        handle_id = self.find_handle_by_actor(actor)
        if not handle_id:
            handle_id = self.find_handle_by_screen_pos(screen_x, screen_y, renderer, threshold_px=24.0)

        if not handle_id or not self.active_implant_id:
            return False

        self.is_dragging = True
        self.hovered_axis = handle_id
        self.drag_start_screen = (screen_x, screen_y)
        imp = self.placement_mgr.implants[self.active_implant_id]
        self.drag_start_pos = np.copy(imp.position)
        self.drag_start_orient = np.copy(imp.orientation)
        self.placement_mgr.save_undo_state()
        return True

    def update_drag(self, screen_x: int, screen_y: int, renderer: vtk.vtkRenderer) -> bool:
        """Updates implant translation or rotation based on local coordinate axes of the implant."""
        if not self.is_dragging or not self.active_implant_id or not self.hovered_axis:
            return False

        imp = self.placement_mgr.implants.get(self.active_implant_id)
        if not imp:
            return False

        dx_screen = screen_x - self.drag_start_screen[0]
        dy_screen = screen_y - self.drag_start_screen[1]

        camera = renderer.GetActiveCamera()
        view_up = np.array(camera.GetViewUp(), dtype=float)
        view_dir = np.array(camera.GetDirectionOfProjection(), dtype=float)
        view_right = np.cross(view_dir, view_up)
        norm_r = np.linalg.norm(view_right)
        if norm_r > 1e-6:
            view_right /= norm_r
        norm_u = np.linalg.norm(view_up)
        if norm_u > 1e-6:
            view_up /= norm_u

        # Compute starting rotation matrix for local axes
        R_start = self._get_rotation_matrix(
            self.drag_start_orient[0], self.drag_start_orient[1], self.drag_start_orient[2]
        )

        # --- 1. Local Translation: Pulling along the implant's LOCAL axis ---
        if self.hovered_axis.startswith("t"):
            axis_key = self.hovered_axis[1] # 'x', 'y', 'z'
            local_unit = np.zeros(3)
            if axis_key == "x":
                local_unit[0] = 1.0
            elif axis_key == "y":
                local_unit[1] = 1.0
            elif axis_key == "z":
                local_unit[2] = 1.0

            # World direction of this local axis
            world_axis_vec = R_start.dot(local_unit)
            norm_w = np.linalg.norm(world_axis_vec)
            if norm_w > 1e-6:
                world_axis_vec /= norm_w

            # Project local axis onto screen right and up vectors
            proj_x = np.dot(world_axis_vec, view_right)
            proj_y = np.dot(world_axis_vec, view_up)

            # Delta pulled in mm along this local axis
            scale = 0.22 # mm per screen pixel
            delta_dist = (dx_screen * proj_x + dy_screen * proj_y) * scale

            new_pos = np.copy(self.drag_start_pos) + world_axis_vec * delta_dist
            imp.position = new_pos
            self.placement_mgr._update_implant_transform(imp)
            self._sync_gizmo_transform(imp)
            self.placement_mgr.update_2d_slice_contours(self.placement_mgr.current_cursor)
            return True

        # --- 2. Local Rotation: Rotating around the implant's LOCAL axis ---
        elif self.hovered_axis.startswith("r"):
            axis_key = self.hovered_axis[1] # 'x', 'y', 'z'

            # Local rotation axis vector
            local_axis_vec = np.zeros(3)
            if axis_key == "x":
                local_axis_vec[0] = 1.0
            elif axis_key == "y":
                local_axis_vec[1] = 1.0
            elif axis_key == "z":
                local_axis_vec[2] = 1.0

            world_axis_vec = R_start.dot(local_axis_vec)
            norm_w = np.linalg.norm(world_axis_vec)
            if norm_w > 1e-6:
                world_axis_vec /= norm_w

            # Compute screen center of the implant/gizmo
            renderer.SetWorldPoint(imp.position[0], imp.position[1], imp.position[2], 1.0)
            renderer.WorldToDisplay()
            center_disp = renderer.GetDisplayPoint()
            cx, cy = center_disp[0], center_disp[1]

            # Angle difference from drag start around center
            start_vec = (self.drag_start_screen[0] - cx, self.drag_start_screen[1] - cy)
            curr_vec = (screen_x - cx, screen_y - cy)

            ang_start = math.atan2(start_vec[1], start_vec[0])
            ang_curr = math.atan2(curr_vec[1], curr_vec[0])
            angle_delta_deg = math.degrees(ang_curr - ang_start)

            if math.hypot(start_vec[0], start_vec[1]) < 5.0:
                angle_delta_deg = dx_screen * 0.8

            # Sign based on whether local axis points towards or away from camera
            facing_sign = 1.0 if np.dot(world_axis_vec, -view_dir) >= 0 else -1.0
            eff_angle_deg = angle_delta_deg * facing_sign

            # Apply delta rotation in local coordinate system: R_new = R_start * R_local(eff_angle_deg)
            rad = math.radians(eff_angle_deg)
            c, s = math.cos(rad), math.sin(rad)
            if axis_key == "z": # Local Yaw
                R_delta = np.array([
                    [c, -s, 0.0],
                    [s,  c, 0.0],
                    [0.0, 0.0, 1.0]
                ])
            elif axis_key == "x": # Local Pitch
                R_delta = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, c, -s],
                    [0.0, s,  c]
                ])
            elif axis_key == "y": # Local Roll
                R_delta = np.array([
                    [c, 0.0, s],
                    [0.0, 1.0, 0.0],
                    [-s, 0.0, c]
                ])

            R_new = R_start.dot(R_delta)
            new_orient = self._matrix_to_euler_angles(R_new)

            imp.orientation = new_orient
            self.placement_mgr._update_implant_transform(imp)
            self._sync_gizmo_transform(imp)
            self.placement_mgr.update_2d_slice_contours(self.placement_mgr.current_cursor)
            return True

        return False

    def end_drag(self):
        """Ends drag operation."""
        self.is_dragging = False
        self.hovered_axis = None
