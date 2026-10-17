"""
mpr_ruler.py
VTK 2D 네이티브 오버레이 기반의 고정밀 의료 영상 눈금자 (Ruler).
- 1mm 간격: 미세 틱 (길이 3px)
- 5mm 간격: 중간 틱 (길이 6px)
- 10mm 간격: 대형 틱 (길이 10px) 및 숫자 라벨
- 하단(Bottom) 및 우측(Right) 창 테두리에 밀착
- CT 이미지 실제 크기에 정렬 및 Zoom/Pan 실시간 연동
"""

from typing import Optional, List, Tuple
import vtk

from dental_fea_app.config import AXIS_AXIAL, AXIS_CORONAL, AXIS_SAGITTAL


class MPRRuler:
    """
    VTK Renderer에 2D 선분과 텍스트로 눈금자를 직접 렌더링하는 네이티브 눈금자 컨트롤러.
    """

    def __init__(self, renderer: vtk.vtkRenderer, axis: int):
        self.renderer = renderer
        self.axis = axis
        self._visible = True

        # 선분용 PolyData 파이프라인
        self.lines_poly = vtk.vtkPolyData()
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.lines_poly.SetPoints(self.points)
        self.lines_poly.SetLines(self.lines)

        self.mapper = vtk.vtkPolyDataMapper2D()
        self.mapper.SetInputData(self.lines_poly)

        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(self.mapper)
        # 밝은 회색 계열 (가시성 우수)
        self.actor.GetProperty().SetColor(0.85, 0.85, 0.90)
        self.actor.GetProperty().SetLineWidth(1.0)
        self.actor.SetVisibility(True)
        self.renderer.AddActor(self.actor)

        # 10mm 텍스트 라벨 풀 (재사용)
        self.label_actors: List[vtk.vtkTextActor] = []
        self._max_labels = 60
        for _ in range(self._max_labels):
            t_actor = vtk.vtkTextActor()
            t_prop = t_actor.GetTextProperty()
            t_prop.SetFontSize(10)
            t_prop.SetFontFamilyToArial()
            t_prop.SetColor(0.85, 0.85, 0.90)
            t_prop.BoldOff()
            t_actor.SetVisibility(False)
            self.renderer.AddActor(t_actor)
            self.label_actors.append(t_actor)

    def set_visibility(self, visible: bool):
        self._visible = visible
        self.actor.SetVisibility(visible)
        if not visible:
            for la in self.label_actors:
                la.SetVisibility(False)

    def is_visible(self) -> bool:
        return self._visible

    def update(self, bounds: Tuple[float, float, float, float, float, float],
               current_coords: Tuple[float, float, float],
               win_w: int, win_h: int):
        """
        현재 뷰포트 크기와 CT 이미지 월드 bounds에 맞춰 눈금자 선분 및 라벨 업데이트
        """
        if not self._visible or win_w <= 10 or win_h <= 10:
            self.actor.SetVisibility(False)
            for la in self.label_actors:
                la.SetVisibility(False)
            return

        self.actor.SetVisibility(True)

        bx0, bx1, by0, by1, bz0, bz1 = bounds
        cur_x, cur_y, cur_z = current_coords

        # 축별 가로/세로 월드 범위 및 좌표 매핑
        if self.axis == AXIS_AXIAL:
            # 가로: X [bx0, bx1], 세로: Y [by0, by1], 깊이: Z = cur_z
            h_min, h_max = bx0, bx1
            v_min, v_max = by0, by1
            def pt_h(val): return (val, cur_y, cur_z)
            def pt_v(val): return (cur_x, val, cur_z)
        elif self.axis == AXIS_CORONAL:
            # 가로: X [bx0, bx1], 세로: Z [bz0, bz1], 깊이: Y = cur_y
            h_min, h_max = bx0, bx1
            v_min, v_max = bz0, bz1
            def pt_h(val): return (val, cur_y, cur_z)
            def pt_v(val): return (cur_x, cur_y, val)
        elif self.axis == AXIS_SAGITTAL:
            # 가로: Y [by0, by1], 세로: Z [bz0, bz1], 깊이: X = cur_x
            h_min, h_max = by0, by1
            v_min, v_max = bz0, bz1
            def pt_h(val): return (cur_x, val, cur_z)
            def pt_v(val): return (cur_x, cur_y, val)
        else:
            return

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        label_idx = 0

        # --- 1. 하단 눈금자 (Bottom: Y = 2) ---
        base_y = 2.0
        w_p0 = pt_h(h_min)
        w_p1 = pt_h(h_max)
        self.renderer.SetWorldPoint(w_p0[0], w_p0[1], w_p0[2], 1.0)
        self.renderer.WorldToDisplay()
        d_x0 = self.renderer.GetDisplayPoint()[0]

        self.renderer.SetWorldPoint(w_p1[0], w_p1[1], w_p1[2], 1.0)
        self.renderer.WorldToDisplay()
        d_x1 = self.renderer.GetDisplayPoint()[0]

        d_xmin = min(d_x0, d_x1)
        d_xmax = max(d_x0, d_x1)

        draw_x0 = max(0.0, d_xmin)
        draw_x1 = min(float(win_w - 1), d_xmax)

        if draw_x1 > draw_x0:
            # 하단 기준선
            p1_id = points.InsertNextPoint(draw_x0, base_y, 0.0)
            p2_id = points.InsertNextPoint(draw_x1, base_y, 0.0)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, p1_id)
            line.GetPointIds().SetId(1, p2_id)
            lines.InsertNextCell(line)

            # 1mm 간격으로 틱 생성
            total_mm = int(round(h_max - h_min))
            px_per_mm = abs(d_x1 - d_x0) / max(1.0, float(total_mm))
            step_stride = 1
            if px_per_mm < 1.5:
                step_stride = 5 if px_per_mm < 0.6 else 2

            for mm in range(0, total_mm + 1, step_stride):
                val = h_min + mm
                w_pt = pt_h(val)
                self.renderer.SetWorldPoint(w_pt[0], w_pt[1], w_pt[2], 1.0)
                self.renderer.WorldToDisplay()
                dx, _, _ = self.renderer.GetDisplayPoint()

                if 0 <= dx < win_w - 5:
                    if mm % 10 == 0:
                        tick_h = 10.0
                    elif mm % 5 == 0:
                        tick_h = 6.0
                    else:
                        tick_h = 3.0

                    pid0 = points.InsertNextPoint(dx, base_y, 0.0)
                    pid1 = points.InsertNextPoint(dx, base_y + tick_h, 0.0)
                    t_line = vtk.vtkLine()
                    t_line.GetPointIds().SetId(0, pid0)
                    t_line.GetPointIds().SetId(1, pid1)
                    lines.InsertNextCell(t_line)

        # --- 2. 우측 눈금자 (Right: X = win_w - 2) ---
        base_x = float(win_w - 2)
        w_v0 = pt_v(v_min)
        w_v1 = pt_v(v_max)
        self.renderer.SetWorldPoint(w_v0[0], w_v0[1], w_v0[2], 1.0)
        self.renderer.WorldToDisplay()
        d_y0 = self.renderer.GetDisplayPoint()[1]

        self.renderer.SetWorldPoint(w_v1[0], w_v1[1], w_v1[2], 1.0)
        self.renderer.WorldToDisplay()
        d_y1 = self.renderer.GetDisplayPoint()[1]

        d_ymin = min(d_y0, d_y1)
        d_ymax = max(d_y0, d_y1)

        draw_y0 = max(0.0, d_ymin)
        draw_y1 = min(float(win_h - 1), d_ymax)

        if draw_y1 > draw_y0:
            # 우측 기준선
            p1_id = points.InsertNextPoint(base_x, draw_y0, 0.0)
            p2_id = points.InsertNextPoint(base_x, draw_y1, 0.0)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, p1_id)
            line.GetPointIds().SetId(1, p2_id)
            lines.InsertNextCell(line)

            total_v_mm = int(round(v_max - v_min))
            px_per_v = abs(d_y1 - d_y0) / max(1.0, float(total_v_mm))
            step_v = 1
            if px_per_v < 1.5:
                step_v = 5 if px_per_v < 0.6 else 2

            for mm in range(0, total_v_mm + 1, step_v):
                val = v_min + mm
                w_pt = pt_v(val)
                self.renderer.SetWorldPoint(w_pt[0], w_pt[1], w_pt[2], 1.0)
                self.renderer.WorldToDisplay()
                _, dy, _ = self.renderer.GetDisplayPoint()

                if 0 <= dy < win_h - 5:
                    if mm % 10 == 0:
                        tick_w = 10.0
                    elif mm % 5 == 0:
                        tick_w = 6.0
                    else:
                        tick_w = 3.0

                    pid0 = points.InsertNextPoint(base_x, dy, 0.0)
                    pid1 = points.InsertNextPoint(base_x - tick_w, dy, 0.0)
                    t_line = vtk.vtkLine()
                    t_line.GetPointIds().SetId(0, pid0)
                    t_line.GetPointIds().SetId(1, pid1)
                    lines.InsertNextCell(t_line)

        # 모든 텍스트 라벨 비활성화 (숫자 없이 눈금만 표시)
        for act in self.label_actors:
            act.SetVisibility(False)

        self.lines_poly.SetPoints(points)
        self.lines_poly.SetLines(lines)
        self.lines_poly.Modified()
