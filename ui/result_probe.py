"""
result_probe.py
Interactive 3D probe and peak stress marker for FEA simulation results.

Provides node probing via 3D click (with threshold distance check to preserve mouse rotation),
closest node lookup, peak stress localization, and VTK marker/caption management.

[Units]
- Length / Displacement: mm
- Stress: MPa
- Modulus: MPa
"""

import time
from typing import Optional, Dict, Any, Sequence, Tuple
import numpy as np
import vtk
from vtk.util import numpy_support
from PySide6.QtCore import QObject, Signal, QEvent, Qt
from PySide6.QtGui import QCursor

from ui.result_view_controller import ResultViewController, FIELD_DISPLAY_CONFIG
from postprocess.peri_implant import max_stress_location


class ResultProbe(QObject):
    """
    Interactive 3D inspection tool for probing nodal FEA results and marking peak stress.

    Signals:
        probed (dict): Emitted when a node is probed via click or direct coordinate lookup.
            Contains {part, node, xyz, u, u_mag, von_mises, s_max_principal, s_min_principal, E_MPa}.
        max_marked (dict): Emitted when peak stress is marked.
            Contains {node_idx, part_name, coord, value, field}.
    """
    probed = Signal(dict)
    max_marked = Signal(dict)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._controller: Optional[ResultViewController] = None
        self._renderer: Optional[vtk.vtkRenderer] = None
        self._interactor: Optional[vtk.vtkRenderWindowInteractor] = None
        self._vtk_widget: Optional[QObject] = None

        # Observer 태그 관리
        self._observer_tags: list = []

        # 마우스 드래그 감지 상태 (Qt 논리 좌표와 VTK 픽셀 좌표 분리)
        self._press_pos: Tuple[int, int] = (0, 0)
        self._qt_press_pos: Optional[Tuple[int, int]] = None
        self._vtk_press_pos: Optional[Tuple[int, int]] = None
        self._press_ctrl: bool = False
        self._probe_enabled: bool = False
        self._last_pick_timestamp: float = 0.0

        # VTK 액터 관리
        self._probe_sphere_actor: Optional[vtk.vtkActor] = None
        self._probe_caption_actor: Optional[vtk.vtkCaptionActor2D] = None
        self._max_sphere_actor: Optional[vtk.vtkActor] = None
        self._max_caption_actor: Optional[vtk.vtkCaptionActor2D] = None

    # -------------------------------------------------------------------------
    # 연결 및 수명주기
    # -------------------------------------------------------------------------
    def attach(self, controller: ResultViewController) -> None:
        """
        Attach the probe to an active ResultViewController and bind VTK interactor events.

        Args:
            controller: Target ResultViewController instance.
        """
        self.detach()

        self._controller = controller
        self._renderer = controller.renderer

        # Qt 위젯 eventFilter 설치 (Qt 레벨 마우스 클릭 감지)
        if controller.window is not None and hasattr(controller.window, "vtkWidget"):
            self._vtk_widget = controller.window.vtkWidget
            try:
                self._vtk_widget.installEventFilter(self)
            except Exception:
                pass

        # VTK Interactor 탐색 및 옵저버 바인딩
        iren = None
        if controller.window is not None and hasattr(controller.window, "vtkWidget"):
            vtk_w = controller.window.vtkWidget
            if hasattr(vtk_w, "GetRenderWindow"):
                rw = vtk_w.GetRenderWindow()
                if rw is not None:
                    iren = rw.GetInteractor()
        if iren is None and self._renderer is not None:
            rw = self._renderer.GetRenderWindow()
            if rw is not None:
                iren = rw.GetInteractor()

        self._interactor = iren
        if self._interactor is not None:
            try:
                if not self._interactor.GetEnabled():
                    self._interactor.Enable()
            except Exception:
                pass
            tag1 = self._interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press, 1.0)
            tag2 = self._interactor.AddObserver("LeftButtonReleaseEvent", self._on_left_button_release, 1.0)
            self._observer_tags.extend([tag1, tag2])

        # 프로브 모드가 활성화되어 있다면 커서 및 상태바 즉시 갱신
        if self._probe_enabled:
            self.enable_click_probe(True)

    def detach(self) -> None:
        """Detach from controller, remove markers, and unbind interactor observers."""
        self.clear_markers()

        if self._vtk_widget is not None:
            try:
                self._vtk_widget.removeEventFilter(self)
                self._vtk_widget.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception:
                pass
            self._vtk_widget = None

        if self._interactor is not None:
            for tag in self._observer_tags:
                self._interactor.RemoveObserver(tag)
        self._observer_tags.clear()

        self._controller = None
        self._renderer = None
        self._interactor = None

    def enable_click_probe(self, enabled: bool) -> None:
        """Enable or disable single-click probing mode and update cursor and status bar."""
        self._probe_enabled = bool(enabled)

        # 프로브 모드 비활성화 시 화면에 표시된 값(캡션)과 노란 구 즉시 숨김
        if not self._probe_enabled:
            self.clear_probe_marker()

        # 3D 위젯 마우스 커서 갱신
        if self._vtk_widget is not None:
            try:
                if self._probe_enabled:
                    self._vtk_widget.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self._vtk_widget.setCursor(Qt.CursorShape.ArrowCursor)
            except Exception:
                pass

        # 3D 뷰어 창 상태바 안내
        if self._controller is not None and getattr(self._controller, "window", None) is not None:
            win = self._controller.window
            if hasattr(win, "statusBar") and win.statusBar() is not None:
                try:
                    if self._probe_enabled:
                        win.statusBar().showMessage("🎯 3D 클릭 프로브 모드 활성화됨 (메쉬 표면을 클릭하여 절점 응력/변위 확인)")
                    else:
                        win.statusBar().clearMessage()
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # 마우스 이벤트 핸들러 (드래그 회전 유지 및 클릭 판별)
    # -------------------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Intercept Qt mouse events on vtkWidget to reliably detect clicks for probing,
        bypassing VTK C++ interactor event delivery discrepancies.
        """
        if watched == self._vtk_widget and self._vtk_widget is not None:
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position().toPoint()
                self._qt_press_pos = (pos.x(), pos.y())
                self._press_pos = self._qt_press_pos  # 하위 호환성 유지
                self._press_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

            elif etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position().toPoint()
                press_pos = self._qt_press_pos or self._press_pos
                self._qt_press_pos = None

                dx = pos.x() - press_pos[0]
                dy = pos.y() - press_pos[1]
                dist = (dx * dx + dy * dy) ** 0.5

                if dist <= 6.0 and (self._probe_enabled or self._press_ctrl):
                    scale = getattr(self._vtk_widget, "_getPixelRatio", lambda: 1.0)()
                    h = self._vtk_widget.height()
                    vtk_x = int(round(pos.x() * scale))
                    vtk_y = int(round((h - pos.y() - 1) * scale))
                    self._perform_pick(vtk_x, vtk_y)

        return super().eventFilter(watched, event)

    def _on_left_button_press(self, caller: Any, event: str) -> None:
        """Record VTK press position and control key state; do not abort to preserve mouse drag."""
        if caller is not None and hasattr(caller, "GetEventPosition"):
            self._vtk_press_pos = caller.GetEventPosition()
            self._press_ctrl = bool(caller.GetControlKey())

    def _on_left_button_release(self, caller: Any, event: str) -> None:
        """Evaluate click displacement; trigger probing if displacement <= 6 pixels."""
        if caller is None or not hasattr(caller, "GetEventPosition") or self._vtk_press_pos is None:
            return

        rel_pos = caller.GetEventPosition()
        dx = rel_pos[0] - self._vtk_press_pos[0]
        dy = rel_pos[1] - self._vtk_press_pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        self._vtk_press_pos = None

        # 드래그 거리 6px 이하이고 (프로브 모드 ON 또는 Ctrl 키 눌림) 상태일 때 실행
        if dist <= 6.0 and (self._probe_enabled or self._press_ctrl):
            self._perform_pick(rel_pos[0], rel_pos[1])

    def _perform_pick(self, x: int, y: int) -> None:
        """Execute VTK picker at screen pixel (x, y) and update probe."""
        if self._renderer is None or self._controller is None:
            return

        # 중복 피킹 방지 (동일 클릭에 대한 짧은 간격 중복 호출 방지)
        now = time.time()
        if now - self._last_pick_timestamp < 0.08:
            return
        self._last_pick_timestamp = now

        # 현재 렌더러에 표시 중인 활성 결과 액터 수집
        actors = self._controller._actors if self._controller else {}
        active_actors = {name: act for name, act in actors.items() if act.GetVisibility()}
        if not active_actors:
            return

        hit_part = None
        pick_pos = None

        # 1. 메쉬 표면(Cell) 레이캐스팅 피킹 (허용오차 0.015, 파트 액터 목록 우선 한정)
        cell_picker = vtk.vtkCellPicker()
        cell_picker.SetTolerance(0.015)
        cell_picker.InitializePickList()
        for act in active_actors.values():
            cell_picker.AddPickList(act)
        cell_picker.PickFromListOn()

        if cell_picker.Pick(float(x), float(y), 0.0, self._renderer):
            hit_act = cell_picker.GetActor()
            for name, act in active_actors.items():
                if act == hit_act:
                    hit_part = name
                    break
            pick_pos = cell_picker.GetPickPosition()

        # 2. CellPicker 실패 시 PropPicker + WorldPointPicker fallback (깊이 버퍼 기반 3D 좌표)
        if pick_pos is None:
            prop_picker = vtk.vtkPropPicker()
            prop_picker.InitializePickList()
            for act in active_actors.values():
                prop_picker.AddPickList(act)
            prop_picker.PickFromListOn()

            if prop_picker.Pick(float(x), float(y), 0.0, self._renderer):
                hit_act = prop_picker.GetViewProp()
                for name, act in active_actors.items():
                    if act == hit_act:
                        hit_part = name
                        break
                wp = vtk.vtkWorldPointPicker()
                wp.Pick(float(x), float(y), 0.0, self._renderer)
                pick_pos = wp.GetPickPosition()

        # 3. 추가 fallback: vtkPointPicker (허용오차 0.03)
        if pick_pos is None:
            point_picker = vtk.vtkPointPicker()
            point_picker.SetTolerance(0.03)
            point_picker.InitializePickList()
            for act in active_actors.values():
                point_picker.AddPickList(act)
            point_picker.PickFromListOn()
            if point_picker.Pick(float(x), float(y), 0.0, self._renderer):
                hit_act = point_picker.GetActor()
                for name, act in active_actors.items():
                    if act == hit_act:
                        hit_part = name
                        break
                pick_pos = point_picker.GetPickPosition()

        if pick_pos is not None:
            probe_data = self.find_nearest_node(pick_pos, part_name=hit_part)
            if probe_data is None and hit_part is not None:
                probe_data = self.find_nearest_node(pick_pos, part_name=None)

            if probe_data is not None:
                self._update_probe_marker(probe_data)
                self.probed.emit(probe_data)
                self._notify_status_bar(probe_data)

    def _notify_status_bar(self, probe_data: dict) -> None:
        """Update 3D viewer window status bar with probed node info."""
        if self._controller is not None and getattr(self._controller, "window", None) is not None:
            win = self._controller.window
            if hasattr(win, "statusBar") and win.statusBar() is not None:
                try:
                    p_name = probe_data.get("part", "")
                    node = probe_data.get("node", -1)
                    vm = probe_data.get("von_mises")
                    vm_str = f"{vm:.2f} MPa" if vm is not None else "--"
                    u_mag = probe_data.get("u_mag", 0.0)
                    win.statusBar().showMessage(
                        f"🎯 [{p_name}] 절점 #{node} | von Mises: {vm_str} | 변위: {u_mag:.4f} mm",
                        5000,
                    )
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # 최근접 절점 검색 및 데이터 수집 (오프스크린 단위 테스트 핵심 함수)
    # -------------------------------------------------------------------------
    def find_nearest_node(
        self,
        pick_pos: Sequence[float],
        part_name: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Locate the nearest node to pick_pos (in 3D coordinates) and collect full nodal results.

        This pure computation method operates independently of GUI picking, allowing
        headless unit testing with arbitrary 3D coordinates.

        Args:
            pick_pos: (x, y, z) 3D coordinate sequence.
            part_name: Optional part filter. If None, queries all parts.

        Returns:
            dict: {
                "part": str,
                "node": int (global 0-based index),
                "xyz": list[float] (undeformed coordinates),
                "u": list[float] (displacement vector in mm),
                "u_mag": float (displacement magnitude in mm),
                "von_mises": float | None (MPa),
                "s_max_principal": float | None (MPa),
                "s_min_principal": float | None (MPa),
                "E_MPa": float | None (MPa),
            } or None if model or result set is missing.
        """
        if self._controller is None:
            return None

        model = self._controller._model
        rs = self._controller._rs
        if model is None or rs is None:
            return None

        pick_vec = np.asarray(pick_pos, dtype=np.float64)[:3]

        best_part = None
        best_loc_idx = -1
        best_dist = float("inf")

        target_parts = [model.get_part(part_name)] if part_name else model.parts
        target_parts = [p for p in target_parts if p is not None]

        for part in target_parts:
            # 변형 스케일이 적용된 경우 현재 표시 그리드 좌표 우선 탐색
            coords = part.nodes
            if self._controller._grids and part.name in self._controller._grids:
                grid = self._controller._grids[part.name]
                if grid.GetPoints() is not None:
                    coords = numpy_support.vtk_to_numpy(grid.GetPoints().GetData())

            if len(coords) == 0:
                continue

            dists = np.linalg.norm(coords - pick_vec, axis=1)
            min_i = int(np.argmin(dists))
            min_d = float(dists[min_i])

            if min_d < best_dist:
                best_dist = min_d
                best_part = part
                best_loc_idx = min_i

        if best_part is None or best_loc_idx < 0:
            return None

        offset = model.global_node_offset(best_part.name)
        global_idx = offset + best_loc_idx

        # 1. 원본 비변형 좌표
        xyz = [float(c) for c in best_part.nodes[best_loc_idx]]

        # 2. 변위
        if rs.u is not None and global_idx < len(rs.u):
            u_vec = [float(x) for x in rs.u[global_idx]]
        else:
            u_vec = [0.0, 0.0, 0.0]

        if rs.u_mag is not None and global_idx < len(rs.u_mag):
            u_mag_val = float(rs.u_mag[global_idx])
        else:
            u_mag_val = float(np.linalg.norm(u_vec))

        # 3. 응력 필드
        vm_val = None
        if rs.von_mises is not None and global_idx < len(rs.von_mises):
            raw_vm = rs.von_mises[global_idx]
            if not np.isnan(raw_vm):
                vm_val = float(raw_vm)

        smax_val = None
        if rs.s_max_principal is not None and global_idx < len(rs.s_max_principal):
            raw_smax = rs.s_max_principal[global_idx]
            if not np.isnan(raw_smax):
                smax_val = float(raw_smax)

        smin_val = None
        if rs.s_min_principal is not None and global_idx < len(rs.s_min_principal):
            raw_smin = rs.s_min_principal[global_idx]
            if not np.isnan(raw_smin):
                smin_val = float(raw_smin)

        # 4. 탄성계수 (E_MPa) - 인접 요소들의 대표값
        e_mpa_val = None
        materials = self._controller._materials
        if materials is not None and best_part.name in materials.element_material:
            mat_indices = materials.element_material[best_part.name]
            elems = np.asarray(best_part.elements, dtype=np.int64)

            # 전역 인덱스로 저장되어 있을 경우 로컬로 변환
            if len(elems) > 0 and np.all(elems >= offset) and np.all(elems < offset + len(best_part.nodes)):
                elems = elems - offset

            # best_loc_idx를 포함하는 요소 검색
            adjacent_elems = [e_i for e_i, el in enumerate(elems) if best_loc_idx in el]
            if adjacent_elems:
                adjacent_mat_ids = mat_indices[adjacent_elems]
                e_vals = [materials.materials[m_id].E for m_id in adjacent_mat_ids if m_id < len(materials.materials)]
                if e_vals:
                    e_mpa_val = float(np.mean(e_vals))

        return {
            "part": best_part.name,
            "node": int(global_idx),
            "xyz": xyz,
            "u": u_vec,
            "u_mag": u_mag_val,
            "von_mises": vm_val,
            "s_max_principal": smax_val,
            "s_min_principal": smin_val,
            "E_MPa": e_mpa_val,
        }

    # -------------------------------------------------------------------------
    # 마커 및 라벨 시각화
    # -------------------------------------------------------------------------
    def _update_probe_marker(self, probe_data: dict) -> None:
        """Create or update 3D sphere marker and 2D caption for probed node."""
        if self._renderer is None:
            return

        # 이전 프로브 마커 제거
        if self._probe_sphere_actor is not None:
            self._renderer.RemoveActor(self._probe_sphere_actor)
            self._probe_sphere_actor = None
        if self._probe_caption_actor is not None:
            self._renderer.RemoveActor(self._probe_caption_actor)
            self._probe_caption_actor = None

        xyz = probe_data["xyz"]
        part = probe_data["part"]
        node = probe_data["node"]
        vm = probe_data["von_mises"]
        u_mag = probe_data["u_mag"]
        e_val = probe_data["E_MPa"]

        # 구 마커 (노란색, 반경 0.3mm)
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*xyz)
        sphere.SetRadius(0.35)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.9, 0.1)  # Yellow
        actor.GetProperty().SetAmbient(0.8)

        # 캡션 액터
        lines = [f"[{part}] Node #{node}"]
        if vm is not None:
            lines.append(f"von Mises: {vm:.2f} MPa")
        lines.append(f"U: {u_mag:.4f} mm")
        if e_val is not None:
            lines.append(f"E: {e_val:.0f} MPa")
        caption_text = "\n".join(lines)

        caption = vtk.vtkCaptionActor2D()
        caption.SetAttachmentPoint(*xyz)
        caption.SetCaption(caption_text)
        caption.BorderOff()
        caption.GetCaptionTextProperty().SetFontSize(10)
        caption.GetCaptionTextProperty().SetColor(1.0, 1.0, 1.0)
        caption.GetCaptionTextProperty().SetBold(True)
        caption.GetCaptionTextProperty().SetShadow(True)

        self._renderer.AddActor(actor)
        self._renderer.AddActor(caption)
        self._probe_sphere_actor = actor
        self._probe_caption_actor = caption

        if self._controller is not None:
            self._controller._render()

    def mark_maximum(
        self,
        field: str = "von_mises",
        part_name: Optional[str] = None,
        focus_camera: bool = False,
    ) -> Optional[dict]:
        """
        Mark the maximum value location with a sphere marker and label, and emit max_marked.

        Args:
            field: Target scalar field ('von_mises', 's_max_principal', 's_min_principal', 'u_mag').
            part_name: Optional part name filter.
            focus_camera: If True, moves the camera focal point to the peak location.

        Returns:
            dict: Max location metadata or None if no valid data.
        """
        if self._controller is not None or self._controller._model is None or self._controller._rs is None:
            pass

        # "보이지 않기" (NONE) 선택 시 마커 제거 후 종료
        if part_name in ("NONE", "none", "보이지 않기", "hide"):
            self.clear_max_marker()
            return None

        if self._controller is None or self._controller._model is None or self._controller._rs is None:
            return None

        max_info = max_stress_location(
            rs=self._controller._rs,
            model=self._controller._model,
            field=field,
            part_name=part_name,
        )

        if max_info.get("value") is None or max_info.get("coord") is None:
            return None

        # 이전 최대값 마커 제거
        self.clear_max_marker()

        coord = max_info["coord"]
        val = max_info["value"]
        p_name = max_info["part_name"]

        # 필드 표시 정보
        cfg = FIELD_DISPLAY_CONFIG.get(field, {"title": field, "unit": ""})
        title = cfg["title"]
        unit = cfg["unit"]

        # 구 마커 (빨간색, 반경 0.5mm)
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*coord)
        sphere.SetRadius(0.5)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.2, 0.2)  # Red
        actor.GetProperty().SetAmbient(0.8)

        # 캡션 액터
        caption_text = f"최대 {title} {val:.1f} {unit} @ {p_name}"
        caption = vtk.vtkCaptionActor2D()
        caption.SetAttachmentPoint(*coord)
        caption.SetCaption(caption_text)
        caption.BorderOff()
        caption.GetCaptionTextProperty().SetFontSize(11)
        caption.GetCaptionTextProperty().SetColor(1.0, 0.85, 0.85)
        caption.GetCaptionTextProperty().SetBold(True)
        caption.GetCaptionTextProperty().SetShadow(True)

        if self._renderer is not None:
            self._renderer.AddActor(actor)
            self._renderer.AddActor(caption)
            self._max_sphere_actor = actor
            self._max_caption_actor = caption

            # 카메라 초점 이동 옵션
            if focus_camera:
                cam = self._renderer.GetActiveCamera()
                if cam is not None:
                    cam.SetFocalPoint(*coord)

        self.max_marked.emit(max_info)

        if self._controller is not None:
            self._controller._render()

        return max_info

    def mark_max_stress(
        self,
        field: str = "von_mises",
        part_name: Optional[str] = None,
        focus_camera: bool = False,
    ) -> Optional[dict]:
        """Alias for mark_maximum for compatibility with ResultController."""
        return self.mark_maximum(field=field, part_name=part_name, focus_camera=focus_camera)

    def clear_probe_marker(self) -> None:
        """Remove probe sphere marker (yellow) and caption from the renderer."""
        changed = False
        if self._renderer is not None:
            if self._probe_sphere_actor is not None:
                self._renderer.RemoveActor(self._probe_sphere_actor)
                self._probe_sphere_actor = None
                changed = True
            if self._probe_caption_actor is not None:
                self._renderer.RemoveActor(self._probe_caption_actor)
                self._probe_caption_actor = None
                changed = True

        if changed and self._controller is not None:
            self._controller._render()

    def clear_max_marker(self) -> None:
        """Remove maximum stress sphere marker (red) and caption from the renderer."""
        changed = False
        if self._renderer is not None:
            if self._max_sphere_actor is not None:
                self._renderer.RemoveActor(self._max_sphere_actor)
                self._max_sphere_actor = None
                changed = True
            if self._max_caption_actor is not None:
                self._renderer.RemoveActor(self._max_caption_actor)
                self._max_caption_actor = None
                changed = True

        if changed and self._controller is not None:
            self._controller._render()

    def clear_markers(self) -> None:
        """Remove all probe and maximum markers and labels from the renderer."""
        self.clear_probe_marker()
        self.clear_max_marker()
