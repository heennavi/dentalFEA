"""
occlusal_load_gizmo.py
3D FEA Geometry 뷰어 전용 대화형 교합 하중 화살표 및 3D 이동/회전 인터랙션 기즈모.
- 굵기 1mm (직경 1.0mm, 이전의 1/5) 및 콘 직경 2.0mm의 일체형 정밀 화살표
- 화살표 끝(Tip)을 중심 원점으로 하는 3축 이동(X 빨강, Y 초록, Z 파랑) + 회전 링 3D 기즈모
- 빈 곳 클릭 시 기즈모 숨김, 화살표 클릭 시 기즈모 활성화
- 노란색 텍스트 빌보드 라벨 (100N, 0.0°)
- 노란색 구체 제거
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import vtk


class OcclusalLoadGizmo:
    """
    3D FEA Geometry 뷰어 내 교합 하중 화살표 및 조작 기즈모.
    """

    def __init__(self, renderer: Optional[vtk.vtkRenderer] = None):
        self.renderer = renderer
        self.is_visible = False
        self.is_selected = False  # 기즈모 핸들 표시 여부 (선택 상태)
        self.attach_to_implant = True  # 임플란트에 부착 체크 여부

        # 하중 물리량 및 방향 (World LPS 좌표계)
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 하중 작용점 (화살표 팁)
        self.direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # 하중 방향 단위 벡터
        self.magnitude_N = 100.0
        self.tilt_deg = 0.0
        self.azimuth_deg = 0.0

        # 기즈모 드래그 인터랙션 상태
        self.is_dragging = False
        self.active_handle: Optional[str] = None  # 'tx', 'ty', 'tz', 'rot'
        self.drag_start_screen = (0, 0)
        self.drag_start_pos = np.zeros(3)
        self.drag_start_dir = np.array([0.0, 0.0, -1.0])
        self.drag_start_tilt = 0.0
        self.rot_matrix = np.eye(3, dtype=np.float64)  # 기즈모 로컬 3축 회전 행렬
        self.drag_start_rot_matrix = np.eye(3, dtype=np.float64)

        # 기본 색상 및 하이라이트 색상 (빨간선-빨간원 rx, 녹색선-녹색원 ry, 파란선-파란원 rz)
        self.base_colors = {
            "tx": (1.0, 0.22, 0.22),   # X축 빨강
            "ty": (0.22, 0.95, 0.28),  # Y축 초록
            "tz": (0.25, 0.65, 1.00),  # Z축 파랑
            "rx": (1.0, 0.22, 0.22),   # 빨간 선(X축)에 수직인 빨간색 원
            "ry": (0.22, 0.95, 0.28),  # 녹색 선(Y축)에 수직인 녹색 원
            "rz": (0.25, 0.65, 1.00),  # 파란 선(Z축)에 수직인 파란색 원
            "rot": (0.22, 0.95, 0.28), # 호환 키
        }
        self.highlight_colors = {
            "tx": (1.0, 0.65, 0.65),
            "ty": (0.65, 1.0, 0.70),
            "tz": (0.70, 0.90, 1.0),
            "rx": (1.0, 0.70, 0.70),
            "ry": (0.70, 1.0, 0.75),
            "rz": (0.70, 0.90, 1.0),
            "rot": (0.70, 1.0, 0.75),
        }

        # 액터 및 어셈블리 분리
        # 1) 화살표 본체 어셈블리 (항상 표시)
        self.arrow_assembly = vtk.vtkAssembly()
        self.arrow_actor: Optional[vtk.vtkActor] = None
        self.cone_actor: Optional[vtk.vtkActor] = None

        # 2) 3D 조작 기즈모 어셈블리 (양방향 화살표 + 회전 링)
        self.gizmo_assembly = vtk.vtkAssembly()
        self.handle_actors: Dict[str, vtk.vtkActor] = {}
        self.rot_handle: Optional[vtk.vtkActor] = None

        # 3) 노란색 텍스트 라벨 (카메라 빌보드)
        self.text_actor: Optional[vtk.vtkBillboardTextActor3D] = None

        self._build_arrow_geometry()
        self._build_gizmo_handles()
        self._build_text_label()

        self.arrow_assembly.SetVisibility(False)
        self.gizmo_assembly.SetVisibility(False)
        if self.renderer is not None:
            self.renderer.AddActor(self.arrow_assembly)
            self.renderer.AddActor(self.gizmo_assembly)
            if self.text_actor is not None:
                self.text_actor.SetVisibility(False)
                self.renderer.AddActor(self.text_actor)

    @property
    def origin(self) -> np.ndarray:
        return self.position

    @origin.setter
    def origin(self, val: np.ndarray):
        if val is not None:
            self.position = np.asarray(val, dtype=np.float64)

    @property
    def snap_to_implant(self) -> bool:
        return self.attach_to_implant

    @snap_to_implant.setter
    def snap_to_implant(self, val: bool):
        self.attach_to_implant = bool(val)

    def _build_arrow_geometry(self):
        """굵기 1.0mm(기존 0.5mm의 2배) 및 콘 직경 2.4mm(반경 1.2mm)의 교합 하중 화살표 생성 (길이는 10mm로 유지)"""
        cone_len = 2.0
        cone_r = 1.20   # 현재 0.60mm의 2배
        shaft_len = 8.0
        tube_r = 0.50   # 현재 0.25mm의 2배 (굵기/직경 1.0mm)
        total_len = cone_len + shaft_len

        # 1) 콘 헤드 (팁이 (0,0,0)에 오고, 방향은 (0,0,-1)을 가리킴)
        cone = vtk.vtkConeSource()
        cone.SetRadius(cone_r)
        cone.SetHeight(cone_len)
        cone.SetResolution(24)
        cone.SetDirection(0.0, 0.0, -1.0)
        cone.SetCenter(0.0, 0.0, cone_len / 2.0)
        cone.Update()

        # 2) 샤프트 실린더 (콘 밑면 z=cone_len 에서 꼬리 z=total_len 까지)
        line = vtk.vtkLineSource()
        line.SetPoint1(0.0, 0.0, cone_len)
        line.SetPoint2(0.0, 0.0, total_len)
        line.Update()

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(tube_r)
        tube.SetNumberOfSides(24)
        tube.CappingOn()
        tube.Update()

        append = vtk.vtkAppendPolyData()
        append.AddInputData(cone.GetOutput())
        append.AddInputData(tube.GetOutput())
        append.Update()

        arrow_mapper = vtk.vtkPolyDataMapper()
        arrow_mapper.SetInputConnection(append.GetOutputPort())

        self.arrow_actor = vtk.vtkActor()
        self.arrow_actor.SetMapper(arrow_mapper)
        self.arrow_actor.GetProperty().SetColor(0.95, 0.20, 0.20)  # 빨간색
        self.arrow_actor.GetProperty().SetSpecular(0.4)
        self.arrow_actor.GetProperty().SetSpecularPower(30.0)
        self.arrow_assembly.AddPart(self.arrow_actor)

    def _build_gizmo_handles(self):
        """화살표 끝(Tip)을 중심 원점으로 하는 3축 이동 화살표와 회전 원의 크기를 현재의 2/3로 축소"""
        axis_len = 12.0     # 18.0 * (2/3)
        cone_len = 2.0      # 3.0 * (2/3)
        cone_r = 0.50       # 적정 비례 슬림 콘
        tube_r = 0.20       # 0.25 * (2/3) ~ 0.20
        ring_rad = 10.7     # 16.0 * (2/3) ~ 10.67mm
        ring_r_tube = 0.20

        # 1. 3축 양방향 이동 화살표 (X 빨강, Y 초록, Z 파랑) - 양 끝 모두에 화살표 콘 배치
        axes_info = [
            ("tx", (1.0, 0.0, 0.0), self.base_colors["tx"]),
            ("ty", (0.0, 1.0, 0.0), self.base_colors["ty"]),
            ("tz", (0.0, 0.0, 1.0), self.base_colors["tz"]),
        ]

        for handle_id, (dx, dy, dz), color in axes_info:
            # -axis_len 에서 +axis_len 까지 관통하는 메인 샤프트
            line = vtk.vtkLineSource()
            line.SetPoint1(-dx * (axis_len - cone_len), -dy * (axis_len - cone_len), -dz * (axis_len - cone_len))
            line.SetPoint2(dx * (axis_len - cone_len), dy * (axis_len - cone_len), dz * (axis_len - cone_len))
            line.Update()

            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(tube_r)
            tube.SetNumberOfSides(16)
            tube.CappingOn()
            tube.Update()

            # 양의 방향 콘 헤드 (+방향)
            cone_pos = vtk.vtkConeSource()
            cone_pos.SetHeight(cone_len)
            cone_pos.SetRadius(cone_r)
            cone_pos.SetResolution(18)
            cone_pos.SetDirection(dx, dy, dz)
            cone_pos.SetCenter(dx * (axis_len - cone_len / 2.0), dy * (axis_len - cone_len / 2.0), dz * (axis_len - cone_len / 2.0))
            cone_pos.Update()

            # 음의 방향 콘 헤드 (-방향, 뼈 반대편으로 돌출되어 항상 보임)
            cone_neg = vtk.vtkConeSource()
            cone_neg.SetHeight(cone_len)
            cone_neg.SetRadius(cone_r)
            cone_neg.SetResolution(18)
            cone_neg.SetDirection(-dx, -dy, -dz)
            cone_neg.SetCenter(-dx * (axis_len - cone_len / 2.0), -dy * (axis_len - cone_len / 2.0), -dz * (axis_len - cone_len / 2.0))
            cone_neg.Update()

            app = vtk.vtkAppendPolyData()
            app.AddInputData(tube.GetOutput())
            app.AddInputData(cone_pos.GetOutput())
            app.AddInputData(cone_neg.GetOutput())
            app.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(app.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            actor.GetProperty().SetLighting(False)

            self.handle_actors[handle_id] = actor
            self.gizmo_assembly.AddPart(actor)

        # 2. 회전 링 핸들 3개 생성 (반경 10.7mm, 16.0mm의 2/3)
        # (1) 빨간 선(X축)에 수직인 빨간색 원 (rx: YZ 평면, 법선 = X축)
        circle_rx = vtk.vtkRegularPolygonSource()
        circle_rx.SetNumberOfSides(64)
        circle_rx.SetRadius(ring_rad)
        circle_rx.GeneratePolygonOff()
        circle_rx.Update()

        tube_rx = vtk.vtkTubeFilter()
        tube_rx.SetInputConnection(circle_rx.GetOutputPort())
        tube_rx.SetRadius(ring_r_tube)
        tube_rx.SetNumberOfSides(16)
        tube_rx.CappingOn()
        tube_rx.Update()

        trans_rx = vtk.vtkTransform()
        trans_rx.RotateY(90.0)  # XY 평면 -> YZ 평면 (법선이 X축)

        tf_rx = vtk.vtkTransformFilter()
        tf_rx.SetTransform(trans_rx)
        tf_rx.SetInputConnection(tube_rx.GetOutputPort())
        tf_rx.Update()

        mapper_rx = vtk.vtkPolyDataMapper()
        mapper_rx.SetInputConnection(tf_rx.GetOutputPort())

        self.rx_handle = vtk.vtkActor()
        self.rx_handle.SetMapper(mapper_rx)
        c_rx = self.base_colors["rx"]
        self.rx_handle.GetProperty().SetColor(c_rx[0], c_rx[1], c_rx[2])
        self.rx_handle.GetProperty().SetAmbient(1.0)
        self.rx_handle.GetProperty().SetDiffuse(0.0)
        self.rx_handle.GetProperty().SetLighting(False)

        self.handle_actors["rx"] = self.rx_handle
        self.gizmo_assembly.AddPart(self.rx_handle)

        # (2) 녹색 선(Y축)에 수직인 녹색 원 (ry: XZ 평면, 법선 = Y축)
        circle_ry = vtk.vtkRegularPolygonSource()
        circle_ry.SetNumberOfSides(64)
        circle_ry.SetRadius(ring_rad)
        circle_ry.GeneratePolygonOff()
        circle_ry.Update()

        tube_ry = vtk.vtkTubeFilter()
        tube_ry.SetInputConnection(circle_ry.GetOutputPort())
        tube_ry.SetRadius(ring_r_tube)
        tube_ry.SetNumberOfSides(16)
        tube_ry.CappingOn()
        tube_ry.Update()

        trans_ry = vtk.vtkTransform()
        trans_ry.RotateX(90.0)  # XY 평면 -> XZ 평면 (법선이 Y축)

        tf_ry = vtk.vtkTransformFilter()
        tf_ry.SetTransform(trans_ry)
        tf_ry.SetInputConnection(tube_ry.GetOutputPort())
        tf_ry.Update()

        mapper_ry = vtk.vtkPolyDataMapper()
        mapper_ry.SetInputConnection(tf_ry.GetOutputPort())

        self.ry_handle = vtk.vtkActor()
        self.ry_handle.SetMapper(mapper_ry)
        c_ry = self.base_colors["ry"]
        self.ry_handle.GetProperty().SetColor(c_ry[0], c_ry[1], c_ry[2])
        self.ry_handle.GetProperty().SetAmbient(1.0)
        self.ry_handle.GetProperty().SetDiffuse(0.0)
        self.ry_handle.GetProperty().SetLighting(False)

        self.handle_actors["ry"] = self.ry_handle
        self.rot_handle = self.ry_handle  # 호환성 유지
        self.gizmo_assembly.AddPart(self.ry_handle)

        # (3) 파란 선(Z축)에 수직인 파란색 원 (rz: XY 평면, 법선 = Z축)
        circle_rz = vtk.vtkRegularPolygonSource()
        circle_rz.SetNumberOfSides(64)
        circle_rz.SetRadius(ring_rad)
        circle_rz.GeneratePolygonOff()
        circle_rz.Update()

        tube_rz = vtk.vtkTubeFilter()
        tube_rz.SetInputConnection(circle_rz.GetOutputPort())
        tube_rz.SetRadius(ring_r_tube)
        tube_rz.SetNumberOfSides(16)
        tube_rz.CappingOn()
        tube_rz.Update()

        mapper_rz = vtk.vtkPolyDataMapper()
        mapper_rz.SetInputConnection(tube_rz.GetOutputPort())

        self.rz_handle = vtk.vtkActor()
        self.rz_handle.SetMapper(mapper_rz)
        c_rz = self.base_colors["rz"]
        self.rz_handle.GetProperty().SetColor(c_rz[0], c_rz[1], c_rz[2])
        self.rz_handle.GetProperty().SetAmbient(1.0)
        self.rz_handle.GetProperty().SetDiffuse(0.0)
        self.rz_handle.GetProperty().SetLighting(False)

        self.handle_actors["rz"] = self.rz_handle
        self.gizmo_assembly.AddPart(self.rz_handle)

    def _build_text_label(self):
        """노란색 텍스트 빌보드 라벨 생성 (100N, 0.0°)"""
        self.text_actor = vtk.vtkBillboardTextActor3D()
        self.text_actor.SetInput(f"{self.magnitude_N:.0f}N, {self.tilt_deg:.1f}°")
        tprop = self.text_actor.GetTextProperty()
        tprop.SetColor(1.0, 0.95, 0.15)  # 노란색
        tprop.SetFontSize(16)
        tprop.BoldOn()
        tprop.SetJustificationToCentered()
        tprop.SetVerticalJustificationToBottom()

    def set_visibility(self, visible: bool):
        """교합 하중 전체 표시 여부 설정"""
        self.is_visible = bool(visible)
        self.arrow_assembly.SetVisibility(1 if self.is_visible else 0)
        self.set_gizmo_handles_visible(self.is_visible)
        if self.text_actor is not None:
            self.text_actor.SetVisibility(1 if self.is_visible else 0)

    def set_gizmo_handles_visible(self, visible: bool):
        """3D 조작 기즈모 핸들 표시/숨김 설정 (화살표 클릭 시 표시, 빈 곳 클릭 시 숨김)"""
        self.is_selected = bool(visible) and self.is_visible
        self.gizmo_assembly.SetVisibility(1 if self.is_selected else 0)

    def _compute_rotation_matrix_from_dir(self, target_dir: np.ndarray) -> np.ndarray:
        """기본 화살표 방향 (0,0,-1)을 target_dir로 회전시키는 3x3 회전 행렬 계산"""
        base = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        norm_t = np.linalg.norm(target_dir)
        curr = target_dir / norm_t if norm_t > 1e-6 else base
        dot = float(np.dot(base, curr))
        if dot > 0.999999:
            return np.eye(3, dtype=np.float64)
        elif dot < -0.999999:
            return np.diag([1.0, -1.0, -1.0])
        axis = np.cross(base, curr)
        axis /= np.linalg.norm(axis)
        angle = math.acos(np.clip(dot, -1.0, 1.0))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ], dtype=np.float64)
        R = np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K.dot(K)
        return R

    def update_parameters(
        self,
        origin: Optional[np.ndarray] = None,
        direction: Optional[np.ndarray] = None,
        magnitude_N: Optional[float] = None,
        tilt_deg: Optional[float] = None,
        azimuth_deg: Optional[float] = None,
    ):
        """하중 작용점, 방향, 크기, 각도 갱신 및 3D 변환 동기화"""
        if origin is not None:
            self.position = np.asarray(origin, dtype=np.float64)
        if direction is not None:
            d = np.asarray(direction, dtype=np.float64)
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-6:
                self.direction = d / norm_d
                self.rot_matrix = self._compute_rotation_matrix_from_dir(self.direction)
        if magnitude_N is not None:
            self.magnitude_N = float(magnitude_N)
        if tilt_deg is not None:
            self.tilt_deg = float(tilt_deg)
        if azimuth_deg is not None:
            self.azimuth_deg = float(azimuth_deg)

        if self.text_actor:
            self.text_actor.SetInput(f"{self.magnitude_N:.0f}N, {self.tilt_deg:.1f}°")

        self.sync_transform()

    def set_load_parameters(
        self,
        magnitude_N: Optional[float] = None,
        tilt_deg: Optional[float] = None,
        azimuth_deg: Optional[float] = None,
    ):
        """하중 크기 및 각도 갱신 (호환성 메서드)"""
        self.update_parameters(
            magnitude_N=magnitude_N,
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
        )

    def update_orientation_from_angles(self, reference_axis: np.ndarray = np.array([0.0, 0.0, -1.0])):
        """기준 축(임플란트 장축)으로부터 tilt_deg만큼 회전된 방향 벡터 및 rot_matrix 동기화"""
        ref = np.asarray(reference_axis, dtype=np.float64)
        norm_r = np.linalg.norm(ref)
        ref = ref / norm_r if norm_r > 1e-6 else np.array([0.0, 0.0, -1.0])

        if abs(ref[1]) < 0.9:
            up = np.array([0.0, 1.0, 0.0])
        elif abs(ref[0]) < 0.9:
            up = np.array([1.0, 0.0, 0.0])
        else:
            up = np.array([0.0, 0.0, 1.0])

        side = np.cross(ref, up)
        norm_s = np.linalg.norm(side)
        side = side / norm_s if norm_s > 1e-6 else np.array([1.0, 0.0, 0.0])

        rad = math.radians(self.tilt_deg)
        d = ref * math.cos(rad) + np.cross(side, ref) * math.sin(rad)
        norm_d = np.linalg.norm(d)
        if norm_d > 1e-6:
            self.direction = d / norm_d
        else:
            self.direction = ref.copy()

        self.rot_matrix = self._compute_rotation_matrix_from_dir(self.direction)
        self.sync_transform()

    def sync_transform(self):
        """기즈모 전체(3개 축 화살표와 3개 회전 링) 및 화살표 본체를 rot_matrix로 일체화 동기화"""
        m = vtk.vtkMatrix4x4()
        for r in range(3):
            for c in range(3):
                m.SetElement(r, c, float(self.rot_matrix[r, c]))
            m.SetElement(r, 3, float(self.position[r]))
        m.SetElement(3, 3, 1.0)

        trans = vtk.vtkTransform()
        trans.SetMatrix(m)

        # 1. 화살표 본체 변환
        self.arrow_assembly.SetUserTransform(trans)

        # 2. 3D 조작 기즈모 전체 변환 (어셈블리 전체가 회전하므로 모든 원과 축이 일체로 회전함)
        self.gizmo_assembly.SetUserTransform(trans)

        # 3. 노란색 텍스트 라벨 위치 (화살표 꼬리 z=12.0mm 위쪽)
        if self.text_actor is not None:
            tail_world = trans.TransformPoint(0.0, 0.0, 12.0)
            self.text_actor.SetPosition(tail_world[0], tail_world[1], tail_world[2])

    def pick(self, screen_x: int, screen_y: int, renderer: vtk.vtkRenderer) -> Optional[str]:
        """
        화면 클릭 위치 피킹 (28px 고감도 다중 점 투영 검사):
        - 'arrow': 화살표 본체 또는 라벨 클릭 -> 기즈모 핸들 표시
        - 'tx', 'ty', 'tz', 'rx', 'ry', 'rot': 기즈모 핸들 클릭 -> 드래그 모드 진입
        - None: 빈 곳 클릭 -> 기즈모 핸들 숨김
        """
        if not self.is_visible or renderer is None:
            return None

        p_screen = np.array([float(screen_x), float(screen_y)], dtype=np.float64)

        def _to_display(world_pt: np.ndarray) -> np.ndarray:
            renderer.SetWorldPoint(float(world_pt[0]), float(world_pt[1]), float(world_pt[2]), 1.0)
            renderer.WorldToDisplay()
            dp = renderer.GetDisplayPoint()
            return np.array([dp[0], dp[1]], dtype=np.float64)

        def _dist_to_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            ab = b - a
            l2 = float(np.dot(ab, ab))
            if l2 < 1e-6:
                return float(np.linalg.norm(p - a))
            t = float(np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0))
            proj = a + t * ab
            return float(np.linalg.norm(p - proj))

        tip_w = self.position
        center = tip_w

        # 1. 3D 조작 기즈모 핸들이 켜져 있을 때 각 축 및 회전 링 피킹 우선 확인
        if self.is_selected:
            min_dist = 28.0  # 28px 여유로운 tolerance
            best_handle = None

            # 3축 이동 핸들: 각 축마다 -12mm, -6mm, 0, +6mm, +12mm 점들 및 선분 검사 (2/3 크기)
            axes_pts = {
                "tx": [np.array([s, 0.0, 0.0]) for s in (-12.0, -6.0, 0.0, 6.0, 12.0)],
                "ty": [np.array([0.0, s, 0.0]) for s in (-12.0, -6.0, 0.0, 6.0, 12.0)],
                "tz": [np.array([0.0, 0.0, s]) for s in (-12.0, -6.0, 0.0, 6.0, 12.0)],
            }

            R = self.rot_matrix
            for h_id, local_pts in axes_pts.items():
                p_disp_list = [_to_display(center + R.dot(l_pt)) for l_pt in local_pts]
                for i in range(len(p_disp_list) - 1):
                    d = _dist_to_seg(p_screen, p_disp_list[i], p_disp_list[i + 1])
                    if d < min_dist:
                        min_dist = d
                        best_handle = h_id

            # 회전 링 3개 (반경 10.7mm, 16.0mm의 2/3):
            # (1) 빨간색 원 rx: YZ 평면 (빨간 선 X축에 수직)
            # (2) 녹색 원 ry: XZ 평면 (녹색 선 Y축에 수직)
            # (3) 파란색 원 rz: XY 평면 (파란 선 Z축에 수직)
            ring_rad = 10.7
            deg_samples = list(range(0, 360, 15))

            rings = {
                "rx": [center + R.dot(np.array([0.0, ring_rad * math.cos(math.radians(deg)), ring_rad * math.sin(math.radians(deg))])) for deg in deg_samples],
                "ry": [center + R.dot(np.array([ring_rad * math.cos(math.radians(deg)), 0.0, ring_rad * math.sin(math.radians(deg))])) for deg in deg_samples],
                "rz": [center + R.dot(np.array([ring_rad * math.cos(math.radians(deg)), ring_rad * math.sin(math.radians(deg)), 0.0])) for deg in deg_samples],
            }

            for r_id, r_pts in rings.items():
                for pt in r_pts:
                    d_pt = _to_display(pt)
                    dist = float(np.linalg.norm(p_screen - d_pt))
                    if dist < min_dist:
                        min_dist = dist
                        best_handle = r_id

            if best_handle is not None:
                return best_handle

        # 2. 화살표 본체 또는 라벨 피킹 확인 (선분/원점/라벨 화면 거리 26px tolerance)
        tail_w = self.position - self.direction * 11.0
        label_w = self.position - self.direction * 13.0

        tip_d = _to_display(tip_w)
        tail_d = _to_display(tail_w)
        label_d = _to_display(label_w)

        dist_arrow = _dist_to_seg(p_screen, tip_d, tail_d)
        dist_tip = float(np.linalg.norm(p_screen - tip_d))
        dist_tail = float(np.linalg.norm(p_screen - tail_d))
        dist_label = float(np.linalg.norm(p_screen - label_d))

        if min(dist_arrow, dist_tip, dist_tail) <= 26.0 or dist_label <= 32.0:
            return "arrow"

        # 3. VTK PropPicker 보조 검사
        picker = vtk.vtkPropPicker()
        if picker.Pick(screen_x, screen_y, 0, renderer):
            prop = picker.GetViewProp() or picker.GetProp3D() or picker.GetAssembly()
            if prop in (self.arrow_assembly, self.arrow_actor):
                return "arrow"
            if self.is_selected:
                if prop == getattr(self, "rx_handle", None):
                    return "rx"
                if prop == getattr(self, "ry_handle", None) or prop == getattr(self, "rot_handle", None):
                    return "ry"
                if prop == getattr(self, "rz_handle", None):
                    return "rz"

        return None

    def highlight_handle(self, handle_id: Optional[str]):
        """마우스 호버 시 해당 핸들을 밝게 강조 표시"""
        target_id = "ry" if handle_id == "rot" else handle_id
        for h_id, actor in self.handle_actors.items():
            if h_id == target_id and h_id in self.highlight_colors:
                c = self.highlight_colors[h_id]
                actor.GetProperty().SetColor(c[0], c[1], c[2])
            elif h_id in self.base_colors:
                c = self.base_colors[h_id]
                actor.GetProperty().SetColor(c[0], c[1], c[2])

    def start_drag(self, handle_id: str, screen_x: int, screen_y: int):
        """기즈모 핸들 드래그 시작"""
        self.is_dragging = True
        self.active_handle = handle_id
        self.drag_start_screen = (screen_x, screen_y)
        self.drag_start_pos = np.copy(self.position)
        self.drag_start_dir = np.copy(self.direction)
        self.drag_start_rot_matrix = np.copy(self.rot_matrix)
        self.drag_start_tilt = float(self.tilt_deg)
        self.highlight_handle(handle_id)

    def update_drag(
        self,
        screen_x: int,
        screen_y: int,
        renderer: vtk.vtkRenderer,
        snap_nodes: Optional[np.ndarray] = None,
        reference_axis: np.ndarray = np.array([0.0, 0.0, -1.0])
    ) -> bool:
        """기즈모 핸들 드래그 이동/360도 회전 수행 (마우스 단순 이동 시에는 절대 실행되지 않음)"""
        if not self.is_dragging or not self.active_handle:
            return False

        dx = screen_x - self.drag_start_screen[0]
        dy = screen_y - self.drag_start_screen[1]

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

        # 1. 3축 이동 (기즈모의 회전된 축 방향으로 화면 투영 구속 이동)
        if self.active_handle in ("tx", "ty", "tz"):
            local_unit = np.zeros(3, dtype=np.float64)
            if self.active_handle == "tx":
                local_unit[0] = 1.0
            elif self.active_handle == "ty":
                local_unit[1] = 1.0
            elif self.active_handle == "tz":
                local_unit[2] = 1.0

            # 기즈모가 회전된 상태에서의 월드 축 단위 벡터
            axis_unit = self.rot_matrix.dot(local_unit)
            norm_a = np.linalg.norm(axis_unit)
            if norm_a > 1e-6:
                axis_unit /= norm_a

            # 3D 축 단위 벡터를 카메라 화면 오른쪽 및 위쪽 벡터로 정사영
            proj_x = float(np.dot(axis_unit, view_right))
            proj_y = float(np.dot(axis_unit, view_up))
            proj_len = math.hypot(proj_x, proj_y)

            scale = 0.20  # mm per pixel
            if proj_len < 0.15:
                # 축이 화면 깊이(시선) 방향과 거의 평행할 때: 상하 드래그(dy)로 직관적 이동
                sign = 1.0 if np.dot(axis_unit, -view_dir) >= 0 else -1.0
                delta_dist = dy * scale * sign
            else:
                # 화면 마우스 변위 (dx, dy)를 축 방향으로 정사영하여 이동 거리(mm) 계산
                delta_dist = (dx * proj_x + dy * proj_y) * scale

            new_pos = self.drag_start_pos + axis_unit * delta_dist

            if self.attach_to_implant and snap_nodes is not None and len(snap_nodes) > 0:
                # 임플란트 표면에 부착 (가장 가까운 절점으로 스냅)
                dists = np.linalg.norm(snap_nodes - new_pos, axis=1)
                self.position = np.copy(snap_nodes[int(np.argmin(dists))])
            else:
                self.position = new_pos

            self.sync_transform()
            return True

        # 2. 회전 링:
        # - 빨간 원 rx (X축에 수직인 원 -> X축 기준 360도 회전)
        # - 녹색 원 ry (Y축에 수직인 원 -> Y축 기준 360도 회전)
        # - 파란 원 rz (Z축에 수직인 원 -> Z축 기준 360도 회전)
        elif self.active_handle in ("rx", "ry", "rz", "rot"):
            # 기즈모 중심의 화면 2D 좌표
            renderer.SetWorldPoint(self.position[0], self.position[1], self.position[2], 1.0)
            renderer.WorldToDisplay()
            disp = renderer.GetDisplayPoint()
            cx, cy = disp[0], disp[1]

            # 드래그 시작점 및 현재 마우스 위치의 중심 대비 각도
            start_vec = (self.drag_start_screen[0] - cx, self.drag_start_screen[1] - cy)
            curr_vec = (screen_x - cx, screen_y - cy)

            ang_start = math.atan2(start_vec[1], start_vec[0])
            ang_curr = math.atan2(curr_vec[1], curr_vec[0])
            delta_deg = math.degrees(ang_curr - ang_start)

            if math.hypot(start_vec[0], start_vec[1]) < 5.0:
                delta_deg = dx * 0.5

            # 회전축 선택 (기즈모의 drag_start_rot_matrix 기준 로컬 축 -> 월드 축)
            if self.active_handle == "rx":
                local_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            elif self.active_handle in ("ry", "rot"):
                local_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:  # "rz"
                local_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # 월드 회전축
            world_axis = self.drag_start_rot_matrix.dot(local_axis)
            norm_w = np.linalg.norm(world_axis)
            if norm_w > 1e-6:
                world_axis /= norm_w

            # 카메라 시선 방향(-view_dir)과 회전축의 내적으로 회전 방향 부호 일치화
            facing_sign = 1.0 if np.dot(world_axis, -view_dir) >= 0 else -1.0
            eff_angle_deg = delta_deg * facing_sign
            phi = math.radians(eff_angle_deg)

            # 로컬 축 기준 증분 회전 행렬 R_delta (3x3)
            # 로드리게스 공식: R_delta = I + sin(phi)*K + (1-cos(phi))*K^2
            kx, ky, kz = local_axis[0], local_axis[1], local_axis[2]
            K = np.array([
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0]
            ], dtype=np.float64)
            R_delta = np.eye(3, dtype=np.float64) + math.sin(phi) * K + (1.0 - math.cos(phi)) * K.dot(K)

            # 기즈모 전체의 회전 행렬을 갱신 (3개의 원과 모든 축이 함께 회전함!)
            self.rot_matrix = self.drag_start_rot_matrix.dot(R_delta)

            # 화살표의 방향은 기본 로컬 방향 [0, 0, -1]에 전체 회전 행렬을 적용한 것
            new_dir = self.rot_matrix.dot(np.array([0.0, 0.0, -1.0], dtype=np.float64))
            norm_d = np.linalg.norm(new_dir)
            if norm_d > 1e-6:
                self.direction = new_dir / norm_d

            # 기준 축(임플란트 장축 reference_axis)과의 사잇각(경사각, 0~180도) 계산
            ref = np.asarray(reference_axis, dtype=np.float64)
            norm_r = np.linalg.norm(ref)
            ref = ref / norm_r if norm_r > 1e-6 else np.array([0.0, 0.0, -1.0])
            dot_ref = float(np.dot(ref, self.direction))
            self.tilt_deg = float(math.degrees(math.acos(np.clip(dot_ref, -1.0, 1.0))))

            self.sync_transform()
            if self.text_actor:
                self.text_actor.SetInput(f"{self.magnitude_N:.0f}N, {self.tilt_deg:.1f}°")
            return True

        return False

    def end_drag(self):
        """드래그 종료"""
        self.is_dragging = False
        self.active_handle = None
        self.highlight_handle(None)

    def cleanup(self):
        """렌더러에서 모든 액터 제거 및 정리"""
        if self.renderer is not None:
            if self.arrow_assembly is not None:
                self.renderer.RemoveActor(self.arrow_assembly)
            if self.gizmo_assembly is not None:
                self.renderer.RemoveActor(self.gizmo_assembly)
            if self.text_actor is not None:
                self.renderer.RemoveActor(self.text_actor)
        self.is_visible = False
        self.is_selected = False
        self.is_dragging = False
