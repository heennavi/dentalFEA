"""
fea_geometry_viewer.py
메인 화면 우측 하단 창에 FEA 기하(피질골, 수질골, 임플란트) 3D 솔리드를 렌더링하고,
가시성 토글 및 단면 컷팅(클리핑 평면)을 실시간으로 반영하는 독립 3D 뷰어 위젯.
"""

from typing import Optional, Dict, Any
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QToolButton, QStackedLayout
)
from PySide6.QtCore import Qt, QSize

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.geometry.surface_extract import trimesh_to_vtk_polydata
from dental_fea_app.geometry.boolean_ops import FeaGeometry
from dental_fea_app.ui.icons import IconFactory


class FeaGeometryViewWidget(QWidget):
    """
    메인 화면 우측 하단(사분면 4번) 전용 3D FEA 지오메트리 뷰어 위젯.
    """

    def __init__(self, state_manager: StateManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_mgr = state_manager
        self._fea_bounds = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 1. 상단 미니멀 헤더
        header = QFrame(self)
        header.setFixedHeight(30)
        header.setStyleSheet("""
            QFrame {
                background-color: #222225;
                border: 1px solid #333338;
                border-radius: 3px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 4, 0)
        header_layout.setSpacing(4)

        self.title_label = QLabel("3D FEA Geometry (기하 형상)", header)
        self.title_label.setStyleSheet("font-weight: bold; color: #61afef; font-size: 11px;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        btn_qss = """
            QToolButton {
                background-color: #2c2c32;
                border: 1px solid #444450;
                border-radius: 3px;
                padding: 2px;
            }
            QToolButton:hover {
                background-color: #3e3e48;
                border-color: #61afef;
            }
        """
        self.btn_reset_cam = QToolButton(header)
        self.btn_reset_cam.setIcon(IconFactory.reset_view(22))
        self.btn_reset_cam.setIconSize(QSize(18, 18))
        self.btn_reset_cam.setFixedSize(26, 24)
        self.btn_reset_cam.setToolTip("3D 시점 초기화 (카메라 리셋)")
        self.btn_reset_cam.setStyleSheet(btn_qss)
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        header_layout.addWidget(self.btn_reset_cam)

        layout.addWidget(header)

        # 2. 3D VTK 렌더링 위젯 및 오버레이 스택
        self.render_container = QFrame(self)
        self.render_container.setStyleSheet("background-color: #121214; border: 1px solid #282830; border-radius: 4px;")
        self.stack = QStackedLayout(self.render_container)
        self.stack.setContentsMargins(0, 0, 0, 0)

        # 안내 문구 오버레이 라벨
        self.lbl_placeholder = QLabel(
            "기하 생성 후 3D 형상(피질골, 수질골, 임플란트)이\n여기에 표시됩니다.\n\n좌측 '4. 기하 처리' 탭에서 [기하 생성]을 실행하십시오.",
            self.render_container
        )
        self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_placeholder.setStyleSheet("color: #666675; font-size: 12px; font-weight: bold; background: transparent;")
        self.stack.addWidget(self.lbl_placeholder)

        self.vtkWidget = QVTKRenderWindowInteractor(self.render_container)
        self.stack.addWidget(self.vtkWidget)

        self.stack.setCurrentWidget(self.lbl_placeholder)

        layout.addWidget(self.render_container, stretch=1)

        # 3. VTK 렌더러 및 카메라 인터랙터 설정
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.08, 0.08, 0.10)
        self.renderer.SetBackground2(0.14, 0.14, 0.18)
        self.renderer.SetGradientBackground(True)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidget.SetInteractorStyle(style)

        # 4. 방향 마커 (Orientation Marker)
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(15.0, 15.0, 15.0)
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 0.3, 0.3)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.3, 1, 0.3)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0.3, 0.6, 1)

        self.marker_widget = vtk.vtkOrientationMarkerWidget()
        self.marker_widget.SetOrientationMarker(axes)
        self.marker_widget.SetInteractor(self.vtkWidget._Iren)
        self.marker_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
        self.marker_widget.SetEnabled(1)
        self.marker_widget.InteractiveOff()

        # 5. FEA 액터 및 클리핑 평면 초기화
        self._init_actors()

        # 6. StateManager 신호 연결
        self.state_mgr.fea_geometry_changed.connect(self._on_fea_geometry_changed)
        self.state_mgr.fea_visibility_changed.connect(self._on_fea_visibility_changed)
        self.state_mgr.fea_clipping_changed.connect(self._on_fea_clipping_changed)
        self.state_mgr.mesh_model_changed.connect(self._on_mesh_model_changed)
        self.state_mgr.material_assignment_changed.connect(self._on_material_assignment_changed)
        self.state_mgr.bc_set_changed.connect(self._on_bc_set_changed)

        # 이미 생성된 기하가 있을 경우 즉시 표시
        geom = getattr(self.state_mgr, "fea_geometry", None)
        if geom is not None:
            self._on_fea_geometry_changed(geom)

        mesh_model = getattr(self.state_mgr, "mesh_model", None)
        if mesh_model is not None:
            self._on_mesh_model_changed(mesh_model)

        self.vtkWidget.Initialize()
        self.vtkWidget.Start()


    def _init_actors(self):
        """피질골, 수질골, 임플란트 3개 액터 및 단면 클리핑 평면 생성"""
        self.fea_clip_plane = vtk.vtkPlane()
        self.fea_clip_plane.SetOrigin(0.0, 0.0, 0.0)
        self.fea_clip_plane.SetNormal(0.0, 0.0, 1.0)

        # 1. 피질골 솔리드 (좌측 패널 #e06c75 색상과 일치화)
        self.cortical_mapper = vtk.vtkPolyDataMapper()
        self.cortical_actor = vtk.vtkActor()
        self.cortical_actor.SetMapper(self.cortical_mapper)
        self.cortical_actor.GetProperty().SetColor(0.88, 0.42, 0.46)
        self.cortical_actor.GetProperty().SetOpacity(0.85)
        self.cortical_actor.GetProperty().SetSpecular(0.2)
        self.cortical_actor.GetProperty().SetSpecularPower(20.0)
        self.cortical_actor.SetVisibility(False)
        self.renderer.AddActor(self.cortical_actor)

        # 2. 수질골 솔리드 (수질골 연파랑/산호)
        self.cancellous_mapper = vtk.vtkPolyDataMapper()
        self.cancellous_actor = vtk.vtkActor()
        self.cancellous_actor.SetMapper(self.cancellous_mapper)
        self.cancellous_actor.GetProperty().SetColor(0.38, 0.69, 0.94)
        self.cancellous_actor.GetProperty().SetOpacity(0.85)
        self.cancellous_actor.GetProperty().SetSpecular(0.2)
        self.cancellous_actor.GetProperty().SetSpecularPower(20.0)
        self.cancellous_actor.SetVisibility(False)
        self.renderer.AddActor(self.cancellous_actor)

        # 3. 임플란트 솔리드 (골드/티타늄)
        self.implant_mapper = vtk.vtkPolyDataMapper()
        self.implant_actor = vtk.vtkActor()
        self.implant_actor.SetMapper(self.implant_mapper)
        self.implant_actor.GetProperty().SetColor(0.90, 0.75, 0.40)
        self.implant_actor.GetProperty().SetOpacity(1.0)
        self.implant_actor.GetProperty().SetSpecular(0.6)
        self.implant_actor.GetProperty().SetSpecularPower(40.0)
        self.implant_actor.SetVisibility(False)
        self.renderer.AddActor(self.implant_actor)

        # 4. 체적 메쉬 표면 액터 및 경계조건 글리프 액터
        self.mesh_actors = {}
        self.mesh_mappers = {}
        self.bc_actors = []


    def _on_fea_geometry_changed(self, fea_geom: Optional[FeaGeometry]):
        """FEA 기하 솔리드가 생성/갱신되었을 때 매핑 및 카메라 리셋"""
        if fea_geom is None:
            self.cortical_actor.SetVisibility(False)
            self.cancellous_actor.SetVisibility(False)
            self.implant_actor.SetVisibility(False)
            self.stack.setCurrentWidget(self.lbl_placeholder)
            self.render()
            return

        self.stack.setCurrentWidget(self.vtkWidget)

        poly_c = trimesh_to_vtk_polydata(fea_geom.cortical)
        poly_cn = trimesh_to_vtk_polydata(fea_geom.cancellous)
        poly_i = trimesh_to_vtk_polydata(fea_geom.implant)

        self.cortical_mapper.SetInputData(poly_c)
        self.cancellous_mapper.SetInputData(poly_cn)
        self.implant_mapper.SetInputData(poly_i)

        all_bounds = []
        for poly in [poly_c, poly_cn, poly_i]:
            if poly.GetNumberOfPoints() > 0:
                all_bounds.append(poly.GetBounds())
        if all_bounds:
            b_arr = np.array(all_bounds)
            self._fea_bounds = (
                float(b_arr[:, 0].min()), float(b_arr[:, 1].max()),
                float(b_arr[:, 2].min()), float(b_arr[:, 2].max()),
                float(b_arr[:, 4].min()), float(b_arr[:, 5].max()),
            )

        self._apply_fea_visibility()
        self._apply_fea_clipping()
        self.reset_camera()

    def _on_fea_visibility_changed(self, vis: dict):
        self._apply_fea_visibility()
        self.render()

    def _apply_fea_visibility(self):
        geom = getattr(self.state_mgr, "fea_geometry", None)
        if geom is None:
            self.cortical_actor.SetVisibility(False)
            self.cancellous_actor.SetVisibility(False)
            self.implant_actor.SetVisibility(False)
            return

        vis = self.state_mgr.fea_visibility
        opacity = float(vis.get("opacity", 0.85))

        self.cortical_actor.SetVisibility(bool(vis.get("cortical", True)))
        self.cortical_actor.GetProperty().SetOpacity(opacity)

        self.cancellous_actor.SetVisibility(bool(vis.get("cancellous", True)))
        self.cancellous_actor.GetProperty().SetOpacity(opacity)

        self.implant_actor.SetVisibility(bool(vis.get("implant", True)))
        self.implant_actor.GetProperty().SetOpacity(1.0)

    def _on_fea_clipping_changed(self, clip: dict):
        self._apply_fea_clipping()
        self.render()

    def _apply_fea_clipping(self):
        clip = getattr(self.state_mgr, "fea_clipping", None)
        if clip is None or not clip.get("enabled", False):
            self.cortical_mapper.RemoveAllClippingPlanes()
            self.cancellous_mapper.RemoveAllClippingPlanes()
            self.implant_mapper.RemoveAllClippingPlanes()
            self.render()
            return

        axis = clip.get("axis", "Z")
        pct = float(clip.get("value_pct", 50.0)) / 100.0
        inverted = bool(clip.get("inverted", False))

        bounds = self._fea_bounds or self.state_mgr.bounds
        normal = [0.0, 0.0, 0.0]
        origin = [0.0, 0.0, 0.0]

        if axis == "X":
            x_min, x_max = bounds[0], bounds[1]
            origin[0] = x_min + (x_max - x_min) * pct
            origin[1] = (bounds[2] + bounds[3]) / 2.0
            origin[2] = (bounds[4] + bounds[5]) / 2.0
            normal[0] = -1.0 if inverted else 1.0
        elif axis == "Y":
            y_min, y_max = bounds[2], bounds[3]
            origin[0] = (bounds[0] + bounds[1]) / 2.0
            origin[1] = y_min + (y_max - y_min) * pct
            origin[2] = (bounds[4] + bounds[5]) / 2.0
            normal[1] = -1.0 if inverted else 1.0
        else:
            z_min, z_max = bounds[4], bounds[5]
            origin[0] = (bounds[0] + bounds[1]) / 2.0
            origin[1] = (bounds[2] + bounds[3]) / 2.0
            origin[2] = z_min + (z_max - z_min) * pct
            normal[2] = -1.0 if inverted else 1.0

        self.fea_clip_plane.SetOrigin(origin[0], origin[1], origin[2])
        self.fea_clip_plane.SetNormal(normal[0], normal[1], normal[2])

        for mapper in [self.cortical_mapper, self.cancellous_mapper, self.implant_mapper]:
            mapper.RemoveAllClippingPlanes()
            mapper.AddClippingPlane(self.fea_clip_plane)

        self.render()

    def reset_camera(self):
        """카메라를 메쉬 전체가 보이도록 리셋"""
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.Elevation(20.0)
        cam.Azimuth(30.0)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def render(self):
        rw = self.vtkWidget.GetRenderWindow()
        if rw:
            rw.Render()

    # --- Phase 3 FEA 파이프라인 시각화 슬롯 ---

    def _on_mesh_model_changed(self, mesh_model):
        """체적 메쉬 생성 시 표면 추출 및 메쉬 와이어프레임 액터 표시"""
        # 기존 메쉬 액터 제거
        for act in self.mesh_actors.values():
            self.renderer.RemoveActor(act)
        self.mesh_actors.clear()
        self.mesh_mappers.clear()

        if mesh_model is None:
            # 기하 형상 복원
            self._apply_fea_visibility()
            self.render()
            return

        from vtk.util import numpy_support
        from meshing.surface_extractor import boundary_faces

        # 체적 메쉬가 생성되면 이전 불리언 솔리드 액터는 숨기고 체적 메쉬 표면을 표시
        self.cortical_actor.SetVisibility(False)
        self.cancellous_actor.SetVisibility(False)
        self.implant_actor.SetVisibility(False)

        for part in mesh_model.parts:
            faces_nodes, _ = boundary_faces(part.elements, nodes=part.nodes)
            if len(faces_nodes) == 0:
                continue

            poly = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(part.nodes, dtype=np.float32), deep=True))
            poly.SetPoints(pts)

            cells = vtk.vtkCellArray()
            n_faces = len(faces_nodes)
            cells_arr = np.empty((n_faces, 4), dtype=np.int64)
            cells_arr[:, 0] = 3
            cells_arr[:, 1:] = faces_nodes
            vtk_id_arr = numpy_support.numpy_to_vtkIdTypeArray(np.ascontiguousarray(cells_arr.ravel()), deep=True)
            cells.SetCells(n_faces, vtk_id_arr)
            poly.SetPolys(cells)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            if hasattr(self, "fea_clip_plane") and getattr(self.state_mgr, "fea_clipping", {}).get("enabled", False):
                mapper.AddClippingPlane(self.fea_clip_plane)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(0.25, 0.25, 0.30)

            if part.name == "implant":
                actor.GetProperty().SetColor(0.92, 0.76, 0.38)
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetSpecular(0.5)
            else:
                actor.GetProperty().SetColor(0.85, 0.82, 0.78)
                actor.GetProperty().SetOpacity(0.85)

            self.renderer.AddActor(actor)
            self.mesh_actors[part.name] = actor
            self.mesh_mappers[part.name] = mapper

        self.stack.setCurrentWidget(self.vtkWidget)
        self.render()

    def _on_material_assignment_changed(self, mat_assign):
        """물성 할당 시 요소 탄성계수(E)를 컬러맵으로 표시"""
        if mat_assign is None or not self.mesh_mappers:
            return

        from vtk.util import numpy_support
        from meshing.surface_extractor import boundary_faces

        mesh_model = getattr(self.state_mgr, "mesh_model", None)
        if mesh_model is None:
            return

        bone_part = mesh_model.get_part("bone")
        if bone_part is not None and "bone" in self.mesh_mappers:
            elem_mat_indices = mat_assign.element_material.get("bone")
            if elem_mat_indices is not None:
                faces_nodes, faces_info = boundary_faces(bone_part.elements, nodes=bone_part.nodes)
                elem_indices = faces_info[:, 0]
                face_mat_idx = elem_mat_indices[elem_indices]
                e_vals = np.array([mat_assign.materials[idx].E for idx in face_mat_idx], dtype=np.float32)

                vtk_scalars = numpy_support.numpy_to_vtk(e_vals, deep=True)
                vtk_scalars.SetName("YoungsModulus_E")

                mapper = self.mesh_mappers["bone"]
                poly = mapper.GetInput()
                poly.GetCellData().SetScalars(vtk_scalars)

                # Cold-to-Hot 룩업 테이블
                e_min = float(np.min(e_vals))
                e_max = float(np.max(e_vals))
                lut = vtk.vtkLookupTable()
                lut.SetNumberOfTableValues(64)
                lut.SetHueRange(0.667, 0.0)  # Blue to Red
                lut.SetTableRange(e_min, max(e_min + 1.0, e_max))
                lut.Build()

                mapper.SetLookupTable(lut)
                mapper.SetScalarRange(e_min, max(e_min + 1.0, e_max))
                mapper.SetScalarModeToUseCellData()
                mapper.ScalarVisibilityOn()

        self.render()

    def _on_bc_set_changed(self, bc_set):
        """경계조건 변경 시 고정 절점(파란 점)과 하중 벡터(빨간 화살표) 표시"""
        for act in self.bc_actors:
            self.renderer.RemoveActor(act)
        self.bc_actors.clear()

        if bc_set is None:
            self.render()
            return

        from vtk.util import numpy_support
        mesh_model = getattr(self.state_mgr, "mesh_model", None)
        if mesh_model is None:
            return

        # 1. 고정 절점(Fixed Nodes) 시각화: 파란색 포인트 글리프
        all_fixed_coords = []
        for fix in bc_set.fixed_node_sets:
            p_name = fix.get("part", "")
            part = mesh_model.get_part(p_name)
            node_ids = fix.get("node_ids", np.array([], dtype=np.int64))
            if part is not None and len(node_ids) > 0:
                all_fixed_coords.append(part.nodes[node_ids])

        if all_fixed_coords:
            fixed_pts = np.vstack(all_fixed_coords)
            poly_fixed = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(fixed_pts, dtype=np.float32), deep=True))
            poly_fixed.SetPoints(pts)

            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(poly_fixed)
            glyph.Update()

            mapper_fix = vtk.vtkPolyDataMapper()
            mapper_fix.SetInputConnection(glyph.GetOutputPort())

            actor_fix = vtk.vtkActor()
            actor_fix.SetMapper(mapper_fix)
            actor_fix.GetProperty().SetColor(0.2, 0.55, 1.0)  # 파란색
            actor_fix.GetProperty().SetPointSize(6.0)
            self.renderer.AddActor(actor_fix)
            self.bc_actors.append(actor_fix)

        # 2. 하중 벡터(Loads) 시각화: 빨간색 화살표 글리프
        for load in bc_set.loads:
            p_name = load.get("part", "")
            part = mesh_model.get_part(p_name)
            node_ids = load.get("node_ids", np.array([], dtype=np.int64))
            force_N = np.asarray(load.get("force_N", (0.0, 0.0, 0.0)), dtype=np.float64)
            mag = float(np.linalg.norm(force_N))

            if part is not None and len(node_ids) > 0 and mag > 1e-6:
                load_pts = part.nodes[node_ids]
                center = np.mean(load_pts, axis=0)

                # 화살표 소스 생성
                arrow = vtk.vtkArrowSource()
                arrow.SetTipResolution(16)
                arrow.SetShaftResolution(16)
                arrow.SetTipLength(0.3)
                arrow.SetTipRadius(0.12)
                arrow.SetShaftRadius(0.04)

                # 정규화 방향 벡터
                dir_vec = force_N / mag

                # 기본 방향 (1, 0, 0)에서 dir_vec로의 회전 축 및 각도
                base_vec = np.array([1.0, 0.0, 0.0])
                axis = np.cross(base_vec, dir_vec)
                axis_norm = np.linalg.norm(axis)
                dot = np.dot(base_vec, dir_vec)
                angle_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

                arrow_len = min(15.0, max(5.0, mag * 0.08))

                transform = vtk.vtkTransform()
                # 화살표 팁이 작용점(center)을 가리키도록 배치
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

                mapper_load = vtk.vtkPolyDataMapper()
                mapper_load.SetInputConnection(tf_filter.GetOutputPort())

                actor_load = vtk.vtkActor()
                actor_load.SetMapper(mapper_load)
                actor_load.GetProperty().SetColor(0.95, 0.25, 0.25)  # 빨간색
                self.renderer.AddActor(actor_load)
                self.bc_actors.append(actor_load)

        self.render()

