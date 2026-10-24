"""
implant_generator.py
치과 임플란트 3D 형상 생성기, Abutment(어버트먼트), Prosthetic(치관 보철물) 및 라이브러리 규격 관리자.
(dentalSeeTee의 핵심 파라메트릭 생성기 및 프리셋 라이브러리 전체 로직)
"""
import os
import sys
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import vtk


@dataclass
class ImplantSpec:
    """치과 임플란트 규격 사양"""
    id: str
    name: str
    diameter_mm: float      # 2.0 ~ 7.0 mm (예: 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)
    length_mm: float        # 5.0, 6.0, 7.0, 8.5, 10.0, 11.5 등
    taper_deg: float        # 0.0 ~ 15.0 deg (예: 0.0, 2.0, 4.0)
    top_type: str           # "flat" 또는 "collar"
    thread_type: str        # "triangular" 또는 "square"
    micro_threads: bool     # 상부 1.5mm 미세 나사산 유무
    pitch_mm: float = 0.6   # 나사산 피치 (기본 0.6mm)
    depth_mm: float = 0.3   # 나사산 깊이 (기본 0.3mm)
    is_custom: bool = False
    created_at: str = ""
    file_name: str = ""


class AbutmentSpec:
    """치과 어버트먼트(지대주) 규격 사양"""
    def __init__(
        self,
        diameter_mm: float = 3.0,
        flare_deg: float = 30.0,
        collar_len_mm: float = 4.0,
        upper_len_mm: float = 5.0,
        upper_angle_deg: float = 0.0,
        upper_dir_deg: float = 0.0,
        color_type: str = "gold",
        color_rgb: Tuple[float, float, float] = (0.92, 0.75, 0.20),
        hex_height_mm: float = 3.0,
        bottom_diameter_mm: Optional[float] = None,
        collar_flare_angle_deg: Optional[float] = None,
        angulation_deg: Optional[float] = None,
        angulation_dir_deg: Optional[float] = None,
        color_name: Optional[str] = None,
        **kwargs
    ):
        self.diameter_mm = float(bottom_diameter_mm if bottom_diameter_mm is not None else diameter_mm)
        self.flare_deg = float(collar_flare_angle_deg if collar_flare_angle_deg is not None else flare_deg)
        self.collar_len_mm = float(collar_len_mm)
        self.upper_len_mm = float(upper_len_mm)
        self.upper_angle_deg = float(angulation_deg if angulation_deg is not None else upper_angle_deg)
        self.upper_dir_deg = float(angulation_dir_deg if angulation_dir_deg is not None else upper_dir_deg)
        self.color_type = str(color_name if color_name is not None else color_type)
        self.color_rgb = tuple(color_rgb)
        self.hex_height_mm = float(hex_height_mm)

    @property
    def bottom_diameter_mm(self) -> float:
        return self.diameter_mm

    @property
    def collar_flare_angle_deg(self) -> float:
        return self.flare_deg

    @property
    def angulation_deg(self) -> float:
        return self.upper_angle_deg

    @property
    def angulation_dir_deg(self) -> float:
        return self.upper_dir_deg

    @property
    def color_name(self) -> str:
        return self.color_type

    @property
    def collar_top_diameter_mm(self) -> float:
        r_bottom = self.diameter_mm / 2.0
        flare_rad = math.radians(self.flare_deg)
        r_collar_top = r_bottom + self.collar_len_mm * math.tan(flare_rad)
        return r_collar_top * 2.0


class ProstheticSpec:
    """치아 보철물(Crown) 규격 사양"""
    def __init__(
        self,
        tooth_number: int = 11,
        base_diameter_mm: float = 3.5,
        length_mm: float = 9.0,
        width_mm: float = 8.5,
        depth_mm: float = 7.0,
        material: str = "Ceramic",
        rotation_deg: float = 0.0,
        color_rgb: Tuple[float, float, float] = (0.95, 0.95, 0.92),
        bottom_diameter_mm: Optional[float] = None,
        **kwargs
    ):
        self.tooth_number = int(tooth_number)
        self.base_diameter_mm = float(bottom_diameter_mm if bottom_diameter_mm is not None else base_diameter_mm)
        self.length_mm = float(length_mm)
        self.width_mm = float(width_mm)
        self.depth_mm = float(depth_mm)
        self.material = str(material)
        self.rotation_deg = float(rotation_deg)
        self.color_rgb = tuple(color_rgb)

    @property
    def bottom_diameter_mm(self) -> float:
        return self.base_diameter_mm

    @bottom_diameter_mm.setter
    def bottom_diameter_mm(self, val: float):
        self.base_diameter_mm = float(val)


def get_default_tooth_dimensions(tooth_number: int) -> Tuple[float, float, float]:
    """각 FDI 치아 번호별 표준 (길이mm, 협설폭mm, 근원심폭mm) 반환"""
    t_str = str(tooth_number)
    pos = int(t_str[1]) if len(t_str) == 2 else 1
    is_upper = (int(t_str[0]) in (1, 2)) if len(t_str) == 2 else True

    if pos == 1:
        return (10.0 if is_upper else 9.0, 8.5 if is_upper else 5.5, 7.0 if is_upper else 6.0)
    elif pos == 2:
        return (9.0 if is_upper else 8.5, 7.0 if is_upper else 6.0, 6.5 if is_upper else 6.2)
    elif pos == 3:
        return (10.5 if is_upper else 10.0, 7.5 if is_upper else 7.0, 8.0 if is_upper else 7.8)
    elif pos in (4, 5):
        return (8.5 if is_upper else 8.0, 7.0 if is_upper else 7.2, 9.0 if is_upper else 8.5)
    else:
        return (7.5 if is_upper else 7.0, 10.5 if is_upper else 11.0, 11.0 if is_upper else 10.5)


def get_material_color_rgb(material: str) -> Tuple[float, float, float]:
    """보철물 재료별 RGB 색상 반환"""
    mat_lower = material.lower()
    if "ceramic" in mat_lower:
        return (0.96, 0.96, 0.94)
    elif "zirconia" in mat_lower:
        return (0.93, 0.89, 0.82)
    elif "gold" in mat_lower:
        return (0.94, 0.76, 0.18)
    elif "metal" in mat_lower:
        return (0.35, 0.38, 0.40)
    return (0.95, 0.95, 0.92)


class Implant3DGenerator:
    """파라메트릭 치과 임플란트 3D PolyData 생성기"""

    @staticmethod
    def generate_implant_polydata(spec: ImplantSpec) -> vtk.vtkPolyData:
        diameter = spec.diameter_mm
        length = spec.length_mm
        taper_rad = math.radians(spec.taper_deg)
        top_type = spec.top_type.lower()
        thread_type = spec.thread_type.lower()
        has_micro = spec.micro_threads
        pitch = spec.pitch_mm if spec.pitch_mm > 0.1 else 0.6
        depth = spec.depth_mm if spec.depth_mm > 0.05 else 0.3

        n_radial = 48
        is_collar = (top_type == "collar")
        collar_len = 2.0 if is_collar else 0.0
        micro_len = 1.5

        apex_len = min(1.2, length * 0.15)
        threaded_len = max(0.5, length - collar_len - apex_len)

        hole_radius = diameter * 0.30
        hole_depth = min(2.5, length * 0.35)
        hole_bottom_z = length - hole_depth

        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()

        z_levels = []

        if is_collar:
            n_collar_steps = 10 if has_micro else 6
            for i in range(n_collar_steps + 1):
                t = i / float(n_collar_steps)
                z = length - t * collar_len
                dist_from_collar_base = (1.0 - t) * collar_len
                flare_rad = math.radians(5.0)
                r_top = (diameter / 2.0) + dist_from_collar_base * math.tan(flare_rad)
                is_micro_zone = has_micro and ((length - z) <= micro_len)
                z_levels.append((z, r_top, "collar", (length - z), is_micro_zone))

        body_top_z = (length - collar_len) if is_collar else length
        total_body_len = body_top_z - apex_len

        n_thread_turns = int(total_body_len / pitch)
        steps_per_turn = 10
        n_body_steps = max(25, n_thread_turns * steps_per_turn)

        for i in range(0 if not is_collar else 1, n_body_steps + 1):
            t = i / float(n_body_steps)
            z = body_top_z - t * total_body_len
            dist_from_top = (length - z)
            r_base = (diameter / 2.0) - dist_from_top * math.tan(taper_rad)
            r_base = max(diameter * 0.22, r_base)

            phase = (dist_from_top / pitch) % 1.0
            if thread_type == "square":
                thread_factor = 1.0 if phase < 0.45 else 0.0
            else:
                thread_factor = math.sin(phase * 2.0 * math.pi)
                thread_factor = max(0.0, thread_factor)

            curr_depth = depth * (1.0 - 0.4 * t)
            r = r_base + thread_factor * curr_depth
            is_micro_zone = has_micro and (dist_from_top <= micro_len)
            z_levels.append((z, r, "body", phase, is_micro_zone))

        n_apex_steps = 6
        r_apex_base = z_levels[-1][1] if z_levels else (diameter * 0.25)
        for i in range(1, n_apex_steps + 1):
            t = i / float(n_apex_steps)
            z = apex_len * (1.0 - t)
            dome_factor = math.sqrt(max(0.0, 1.0 - (t ** 2)))
            r = r_apex_base * dome_factor
            z_levels.append((z, max(0.05, r), "apex", t, False))

        bottom_center_idx = points.InsertNextPoint(0.0, 0.0, 0.0)

        outer_ring_point_indices = []
        for ring_idx, (z, r_base, zone, param, is_micro) in enumerate(z_levels):
            ring_pt_ids = []
            for j in range(n_radial):
                angle = 2.0 * math.pi * j / float(n_radial)
                if zone == "body":
                    spiral_shift = math.sin(angle + (length - z) / pitch * 2.0 * math.pi)
                    r = r_base + 0.15 * depth * spiral_shift
                elif zone == "collar":
                    r = r_base
                else:
                    r = r_base

                if is_micro:
                    micro_phase = ((length - z) / 0.3) * 2.0 * math.pi
                    r += 0.05 * math.sin(micro_phase)

                x = r * math.cos(angle)
                y = r * math.sin(angle)
                pid = points.InsertNextPoint(x, y, z)
                ring_pt_ids.append(pid)
            outer_ring_point_indices.append(ring_pt_ids)

        inner_z_levels = [
            length,
            length - 0.2 * hole_depth,
            length - 0.8 * hole_depth,
            hole_bottom_z
        ]
        inner_ring_point_indices = []
        for z in inner_z_levels:
            ring_pt_ids = []
            for j in range(n_radial):
                angle = 2.0 * math.pi * j / float(n_radial)
                x = hole_radius * math.cos(angle)
                y = hole_radius * math.sin(angle)
                pid = points.InsertNextPoint(x, y, z)
                ring_pt_ids.append(pid)
            inner_ring_point_indices.append(ring_pt_ids)

        hole_bottom_center_idx = points.InsertNextPoint(0.0, 0.0, hole_bottom_z)

        top_outer_ring = outer_ring_point_indices[0]
        top_inner_ring = inner_ring_point_indices[0]

        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            p_out0 = top_outer_ring[j]
            p_out1 = top_outer_ring[next_j]
            p_in0 = top_inner_ring[j]
            p_in1 = top_inner_ring[next_j]

            t1 = vtk.vtkTriangle()
            t1.GetPointIds().SetId(0, p_out0)
            t1.GetPointIds().SetId(1, p_in0)
            t1.GetPointIds().SetId(2, p_out1)
            polys.InsertNextCell(t1)

            t2 = vtk.vtkTriangle()
            t2.GetPointIds().SetId(0, p_out1)
            t2.GetPointIds().SetId(1, p_in0)
            t2.GetPointIds().SetId(2, p_in1)
            polys.InsertNextCell(t2)

        for r_idx in range(len(inner_ring_point_indices) - 1):
            r1 = inner_ring_point_indices[r_idx]
            r2 = inner_ring_point_indices[r_idx + 1]
            for j in range(n_radial):
                next_j = (j + 1) % n_radial
                p0 = r1[j]
                p1 = r1[next_j]
                p2 = r2[next_j]
                p3 = r2[j]

                t1 = vtk.vtkTriangle()
                t1.GetPointIds().SetId(0, p0)
                t1.GetPointIds().SetId(1, p1)
                t1.GetPointIds().SetId(2, p3)
                polys.InsertNextCell(t1)

                t2 = vtk.vtkTriangle()
                t2.GetPointIds().SetId(0, p1)
                t2.GetPointIds().SetId(1, p2)
                t2.GetPointIds().SetId(2, p3)
                polys.InsertNextCell(t2)

        bottom_inner_ring = inner_ring_point_indices[-1]
        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, hole_bottom_center_idx)
            tri.GetPointIds().SetId(1, bottom_inner_ring[j])
            tri.GetPointIds().SetId(2, bottom_inner_ring[next_j])
            polys.InsertNextCell(tri)

        for r_idx in range(len(outer_ring_point_indices) - 1):
            r1 = outer_ring_point_indices[r_idx]
            r2 = outer_ring_point_indices[r_idx + 1]
            for j in range(n_radial):
                next_j = (j + 1) % n_radial
                p0 = r1[j]
                p1 = r1[next_j]
                p2 = r2[next_j]
                p3 = r2[j]

                t1 = vtk.vtkTriangle()
                t1.GetPointIds().SetId(0, p0)
                t1.GetPointIds().SetId(1, p3)
                t1.GetPointIds().SetId(2, p1)
                polys.InsertNextCell(t1)

                t2 = vtk.vtkTriangle()
                t2.GetPointIds().SetId(0, p1)
                t2.GetPointIds().SetId(1, p3)
                t2.GetPointIds().SetId(2, p2)
                polys.InsertNextCell(t2)

        bottom_outer_ring = outer_ring_point_indices[-1]
        for j in range(n_radial):
            next_j = (j + 1) % n_radial
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, bottom_center_idx)
            tri.GetPointIds().SetId(1, bottom_outer_ring[next_j])
            tri.GetPointIds().SetId(2, bottom_outer_ring[j])
            polys.InsertNextCell(tri)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(cleaner.GetOutputPort())
        normals.SetFeatureAngle(45.0)
        normals.SplittingOn()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.ComputePointNormalsOn()
        normals.Update()

        return normals.GetOutput()


def generate_abutment_polydata(
    spec: AbutmentSpec,
    implant_spec: Optional[ImplantSpec] = None,
) -> vtk.vtkPolyData:
    """
    치과 어버트먼트 3D 메쉬 생성 (임플란트 홀 맞춤 Hex 돌기, Collar, Morse Taper 상부).
    임플란트 홀에 최적화된 하부 돌출부와 자가교차 없는 완전한 닫힌 2-매니폴드 표면을 생성합니다.
    """
    diameter_mm = float(spec.diameter_mm)
    flare_deg = float(spec.flare_deg)
    collar_len_mm = float(spec.collar_len_mm)
    upper_len_mm = float(spec.upper_len_mm)
    upper_angle_deg = float(spec.upper_angle_deg)
    upper_dir_deg = float(spec.upper_dir_deg)

    r_bottom = diameter_mm / 2.0
    flare_rad = math.radians(flare_deg)
    r_collar_top = r_bottom + collar_len_mm * math.tan(flare_rad)
    r_upper_bottom = max(0.6, r_collar_top - 0.5)
    taper_rad = math.radians(2.0)
    r_upper_top = max(0.5, r_upper_bottom - upper_len_mm * math.tan(taper_rad))

    # 임플란트 홀에 맞춘 Hex 하부 돌출부 크기 및 깊이 자동 변형
    if implant_spec is not None:
        hex_r = float(implant_spec.diameter_mm) * 0.30
        hex_depth = min(2.5, float(implant_spec.length_mm) * 0.35)
    else:
        hex_r = min(1.2, r_bottom * 0.60)
        hex_depth = float(spec.hex_height_mm) if getattr(spec, "hex_height_mm", 0) > 0.5 else 2.5

    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    n_radial = 36

    def add_quad(p1, p2, p3, p4):
        tri1 = vtk.vtkTriangle()
        tri1.GetPointIds().SetId(0, p1)
        tri1.GetPointIds().SetId(1, p2)
        tri1.GetPointIds().SetId(2, p3)
        polys.InsertNextCell(tri1)

        tri2 = vtk.vtkTriangle()
        tri2.GetPointIds().SetId(0, p1)
        tri2.GetPointIds().SetId(1, p3)
        tri2.GetPointIds().SetId(2, p4)
        polys.InsertNextCell(tri2)

    def add_tri(p1, p2, p3):
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, p1)
        tri.GetPointIds().SetId(1, p2)
        tri.GetPointIds().SetId(2, p3)
        polys.InsertNextCell(tri)

    # 1. Hex 하부 돌출부 바닥 캡 및 측면 기둥 (z = -hex_depth ~ 0.0)
    hex_bottom_pts = []
    hex_top_pts = []
    for i in range(6):
        ang = i * (2.0 * math.pi / 6.0)
        hex_bottom_pts.append(points.InsertNextPoint(hex_r * math.cos(ang), hex_r * math.sin(ang), -hex_depth))
        hex_top_pts.append(points.InsertNextPoint(hex_r * math.cos(ang), hex_r * math.sin(ang), 0.0))

    hex_bottom_center = points.InsertNextPoint(0.0, 0.0, -hex_depth)
    for i in range(6):
        inxt = (i + 1) % 6
        # 바닥면 법선: -Z 방향
        add_tri(hex_bottom_center, hex_bottom_pts[inxt], hex_bottom_pts[i])
        # 기둥 옆면: 외측 방향 법선
        add_quad(hex_bottom_pts[i], hex_bottom_pts[inxt], hex_top_pts[inxt], hex_top_pts[i])

    # 2. Collar (z = 0.0 ~ collar_len_mm)
    n_collar_steps = max(6, int(collar_len_mm * 2))
    collar_rings = []
    for s in range(n_collar_steps + 1):
        t = s / float(n_collar_steps)
        cz = t * collar_len_mm
        cr = r_bottom + cz * math.tan(flare_rad)
        ring = []
        for j in range(n_radial):
            ang = j * (2.0 * math.pi / float(n_radial))
            ring.append(points.InsertNextPoint(cr * math.cos(ang), cr * math.sin(ang), cz))
        collar_rings.append(ring)

    base_ring = collar_rings[0]

    # Hex 상단(6정점)과 Collar 바닥 원주(36정점) 간의 일체형 환형 전이면(Annular Transition)
    for i in range(6):
        h0 = hex_top_pts[i]
        h1 = hex_top_pts[(i + 1) % 6]
        j_start = i * 6
        for k in range(6):
            b0 = base_ring[(j_start + k) % n_radial]
            b1 = base_ring[(j_start + k + 1) % n_radial]
            if k < 3:
                add_tri(h0, b1, b0)
            elif k == 3:
                add_tri(h0, b1, b0)
                add_tri(h0, h1, b1)
            else:
                add_tri(h1, b1, b0)

    # Collar 옆면 쿼드
    for s in range(len(collar_rings) - 1):
        r1 = collar_rings[s]
        r2 = collar_rings[s + 1]
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_quad(r1[j], r1[jnxt], r2[jnxt], r2[j])

    # 3. Upper Post (지대주 상부 기둥)
    rot_trans = vtk.vtkTransform()
    rot_trans.Identity()
    rot_trans.RotateZ(upper_dir_deg)
    rot_trans.RotateX(upper_angle_deg)
    rot_trans.RotateZ(-upper_dir_deg)

    n_upper_steps = max(6, int(upper_len_mm * 2))
    upper_rings = []
    for s in range(n_upper_steps + 1):
        t = s / float(n_upper_steps)
        uz_local = t * upper_len_mm
        ur = r_upper_bottom - uz_local * math.tan(taper_rad)
        ring = []
        for j in range(n_radial):
            ang = j * (2.0 * math.pi / float(n_radial))
            lx = ur * math.cos(ang)
            ly = ur * math.sin(ang)
            lz = uz_local
            wx, wy, wz = rot_trans.TransformPoint(lx, ly, lz)
            ring.append(points.InsertNextPoint(wx, wy, collar_len_mm + wz))
        upper_rings.append(ring)

    collar_top = collar_rings[-1]
    upper_bot = upper_rings[0]
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_quad(collar_top[j], collar_top[jnxt], upper_bot[jnxt], upper_bot[j])

    for s in range(len(upper_rings) - 1):
        r1 = upper_rings[s]
        r2 = upper_rings[s + 1]
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_quad(r1[j], r1[jnxt], r2[jnxt], r2[j])

    # 4. Upper Post 상단 캡 (단일 폐곡면)
    upper_top_ring = upper_rings[-1]
    top_center_local = rot_trans.TransformPoint(0.0, 0.0, upper_len_mm)
    top_center_pid = points.InsertNextPoint(top_center_local[0], top_center_local[1], collar_len_mm + top_center_local[2])
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_tri(top_center_pid, upper_top_ring[j], upper_top_ring[jnxt])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def generate_prosthetic_polydata(
    spec: ProstheticSpec,
    abutment_spec: Optional[AbutmentSpec] = None,
) -> vtk.vtkPolyData:
    """
    해부학적 치관(Crown) 3D 메쉬 생성.
    abutment_spec이 주어지면 하부 Abutment 상부 기둥에 완벽히 맞닿는 오목한 내부 홀(Internal Cavity)을
    자동으로 형성하여 간섭 없는 일체형 닫힌 2-매니폴드 솔리드를 생성합니다.
    """
    tooth_num = int(spec.tooth_number)
    base_diameter_mm = float(abutment_spec.collar_top_diameter_mm) if abutment_spec is not None else float(spec.base_diameter_mm)
    length_mm = float(spec.length_mm)
    width_mm = float(spec.width_mm)
    depth_mm = float(spec.depth_mm)
    rot_deg = float(spec.rotation_deg)

    t_str = str(tooth_num)
    pos = int(t_str[1]) if len(t_str) == 2 else 1
    if pos in (1, 2):
        t_type = "incisor"
    elif pos == 3:
        t_type = "canine"
    elif pos in (4, 5):
        t_type = "premolar"
    else:
        t_type = "molar"

    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    n_radial = 36
    n_height_steps = 14
    r_base = base_diameter_mm / 2.0
    rot_rad = math.radians(rot_deg)

    def add_quad(p1, p2, p3, p4):
        tri1 = vtk.vtkTriangle()
        tri1.GetPointIds().SetId(0, p1)
        tri1.GetPointIds().SetId(1, p2)
        tri1.GetPointIds().SetId(2, p3)
        polys.InsertNextCell(tri1)

        tri2 = vtk.vtkTriangle()
        tri2.GetPointIds().SetId(0, p1)
        tri2.GetPointIds().SetId(1, p3)
        tri2.GetPointIds().SetId(2, p4)
        polys.InsertNextCell(tri2)

    def add_tri(p1, p2, p3):
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, p1)
        tri.GetPointIds().SetId(1, p2)
        tri.GetPointIds().SetId(2, p3)
        polys.InsertNextCell(tri)

    rings = []
    for s in range(n_height_steps + 1):
        t = s / float(n_height_steps)
        z = t * length_mm
        w_factor = math.sin((t / 0.45) * (math.pi / 2.0)) if t < 0.45 else 1.0

        if t_type == "incisor":
            rx = r_base + (width_mm / 2.0 - r_base) * w_factor
            ry_mid = r_base + (depth_mm / 2.0 - r_base) * w_factor
            ry = ry_mid * max(0.25, 1.0 - ((t - 0.75) / 0.25) * 0.7) if t > 0.75 else ry_mid
            p_exp = 2.0 + t * 1.5
        elif t_type == "canine":
            rx_mid = r_base + (width_mm / 2.0 - r_base) * w_factor
            ry_mid = r_base + (depth_mm / 2.0 - r_base) * w_factor
            taper = 1.0 if t <= 0.75 else max(0.5, 1.0 - ((t - 0.75) / 0.25) * 0.4)
            rx = rx_mid * taper
            ry = ry_mid * taper
            p_exp = 2.0 - t * 0.3
        elif t_type == "premolar":
            rx = r_base + (width_mm / 2.0 - r_base) * w_factor
            ry = r_base + (depth_mm / 2.0 - r_base) * w_factor
            p_exp = 2.2 + t * 0.8
        else:
            rx = r_base + (width_mm / 2.0 - r_base) * w_factor
            ry = r_base + (depth_mm / 2.0 - r_base) * w_factor
            p_exp = 2.5 + t * 1.5

        ring = []
        for j in range(n_radial):
            theta = j * (2.0 * math.pi / float(n_radial))
            c = math.cos(theta)
            s_ang = math.sin(theta)

            cos_p = abs(c) ** p_exp
            sin_p = abs(s_ang) ** p_exp
            factor = (cos_p + sin_p) ** (-1.0 / p_exp) if (cos_p + sin_p) > 0 else 1.0

            lx = rx * c * factor
            ly = ry * s_ang * factor

            if t > 0.85:
                top_frac = (t - 0.85) / 0.15
                if t_type == "molar":
                    cusp_mod = 0.8 * math.sin(2.0 * theta) * top_frac
                elif t_type == "premolar":
                    cusp_mod = 0.7 * abs(math.sin(theta)) * top_frac
                elif t_type == "canine":
                    cusp_mod = (1.0 - math.hypot(lx / max(1e-3, rx), ly / max(1e-3, ry))) * 1.2 * top_frac
                else:
                    cusp_mod = 0.0
            else:
                cusp_mod = 0.0

            wx = lx * math.cos(rot_rad) - ly * math.sin(rot_rad)
            wy = lx * math.sin(rot_rad) + ly * math.cos(rot_rad)
            pid = points.InsertNextPoint(wx, wy, z + cusp_mod)
            ring.append(pid)
        rings.append(ring)

    # 1. 외측 측면 벽 (Outer Side walls)
    for s in range(len(rings) - 1):
        r1 = rings[s]
        r2 = rings[s + 1]
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_quad(r1[j], r1[jnxt], r2[jnxt], r2[j])

    # 2. 외측 상단 캡 (Outer Top Cap)
    top_center = points.InsertNextPoint(0.0, 0.0, length_mm)
    t_ring = rings[-1]
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_tri(top_center, t_ring[j], t_ring[jnxt])

    # 3. 바닥 및 내부 오목 홀 (Bottom & Inner Cavity)
    b_ring = rings[0]
    if abutment_spec is not None:
        u_len = float(abutment_spec.upper_len_mm)
        u_ang = float(abutment_spec.upper_angle_deg)
        u_dir = float(abutment_spec.upper_dir_deg)
        flare_rad = math.radians(abutment_spec.flare_deg)
        r_collar_top_val = (abutment_spec.diameter_mm / 2.0) + abutment_spec.collar_len_mm * math.tan(flare_rad)
        r_upper_bottom = max(0.6, r_collar_top_val - 0.5)
        taper_rad = math.radians(2.0)

        rot_trans = vtk.vtkTransform()
        rot_trans.Identity()
        rot_trans.RotateZ(u_dir)
        rot_trans.RotateX(u_ang)
        rot_trans.RotateZ(-u_dir)

        n_upper_steps = max(6, int(u_len * 2))
        cav_rings = []
        for s in range(n_upper_steps + 1):
            t = s / float(n_upper_steps)
            uz_local = t * u_len
            ur = r_upper_bottom - uz_local * math.tan(taper_rad)
            ring = []
            for j in range(n_radial):
                ang = j * (2.0 * math.pi / float(n_radial))
                lx = ur * math.cos(ang)
                ly = ur * math.sin(ang)
                lz = uz_local
                wx, wy, wz = rot_trans.TransformPoint(lx, ly, lz)
                ring.append(points.InsertNextPoint(wx, wy, wz))
            cav_rings.append(ring)

        # 바닥 환형 마진 (외측 링과 내부 홀 입구 링 사이를 연결, 법선 -Z)
        c_bot_ring = cav_rings[0]
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_quad(c_bot_ring[j], c_bot_ring[jnxt], b_ring[jnxt], b_ring[j])

        # 내부 공동 내벽 (법선은 솔리드 내부를 향함)
        for s in range(len(cav_rings) - 1):
            r1 = cav_rings[s]
            r2 = cav_rings[s + 1]
            for j in range(n_radial):
                jnxt = (j + 1) % n_radial
                add_quad(r1[jnxt], r1[j], r2[j], r2[jnxt])

        # 내부 공동 천장 (내부 캡)
        cav_top_ring = cav_rings[-1]
        cav_top_center = rot_trans.TransformPoint(0.0, 0.0, u_len)
        cav_top_pid = points.InsertNextPoint(cav_top_center[0], cav_top_center[1], cav_top_center[2])
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_tri(cav_top_pid, cav_top_ring[jnxt], cav_top_ring[j])
    else:
        # 단독 사용 시 평평한 바닥 캡
        base_center = points.InsertNextPoint(0.0, 0.0, 0.0)
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_tri(base_center, b_ring[jnxt], b_ring[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


class ImplantLibraryManager:
    """임플란트 프리셋 라이브러리 파일 로드 및 관리자"""

    @classmethod
    def get_default_dir(cls) -> str:
        candidates = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "asset", "implant")),
            os.path.abspath("asset/implant"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    @classmethod
    def get_custom_dir(cls) -> str:
        d = os.path.join(cls.get_default_dir(), "custom")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def list_all_implants(cls) -> List[ImplantSpec]:
        implants = []
        default_dir = cls.get_default_dir()
        custom_dir = cls.get_custom_dir()

        if os.path.exists(default_dir):
            for fname in sorted(os.listdir(default_dir)):
                if fname.endswith(".json"):
                    fpath = os.path.join(default_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            data["is_custom"] = False
                            implants.append(ImplantSpec(**data))
                    except Exception:
                        pass

        if os.path.exists(custom_dir):
            for fname in sorted(os.listdir(custom_dir)):
                if fname.endswith(".json"):
                    fpath = os.path.join(custom_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            data["is_custom"] = True
                            implants.append(ImplantSpec(**data))
                    except Exception:
                        pass

        return implants

    @classmethod
    def load_implant_polydata(cls, spec: ImplantSpec) -> vtk.vtkPolyData:
        dir_path = cls.get_custom_dir() if spec.is_custom else cls.get_default_dir()
        vtp_path = os.path.join(dir_path, spec.file_name or f"{spec.id}.vtp")

        if os.path.exists(vtp_path):
            try:
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(vtp_path)
                reader.Update()
                output = reader.GetOutput()
                if output and output.GetNumberOfPoints() > 0:
                    return output
            except Exception:
                pass

        return Implant3DGenerator.generate_implant_polydata(spec)

    @classmethod
    def save_custom_implant(cls, spec: ImplantSpec) -> Tuple[bool, str]:
        """커스텀 임플란트를 asset/implant/custom/ 에 저장"""
        custom_dir = cls.get_custom_dir()
        spec.is_custom = True
        spec.pitch_mm = 0.6
        spec.depth_mm = 0.3
        import datetime
        spec.created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        safe_id = f"custom_d{int(spec.diameter_mm*10)}_l{int(spec.length_mm*10)}_t{int(spec.taper_deg)}_{spec.top_type}_{int(datetime.datetime.now().timestamp())}"
        spec.id = safe_id
        spec.file_name = f"{safe_id}.vtp"

        json_path = os.path.join(custom_dir, f"{safe_id}.json")
        vtp_path = os.path.join(custom_dir, f"{safe_id}.vtp")

        try:
            polydata = Implant3DGenerator.generate_implant_polydata(spec)
            if os.path.exists(vtp_path):
                try:
                    os.remove(vtp_path)
                except Exception:
                    pass

            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(vtp_path)
            writer.SetInputData(polydata)
            writer.SetDataModeToBinary()
            writer.Write()

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(spec), f, indent=2, ensure_ascii=False)
            return True, safe_id
        except Exception as e:
            return False, str(e)

    @classmethod
    def delete_custom_implant(cls, implant_id: str) -> bool:
        """커스텀 임플란트 삭제"""
        custom_dir = cls.get_custom_dir()
        json_path = os.path.join(custom_dir, f"{implant_id}.json")
        vtp_path = os.path.join(custom_dir, f"{implant_id}.vtp")

        deleted = False
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
                deleted = True
            except Exception:
                pass
        if os.path.exists(vtp_path):
            try:
                os.remove(vtp_path)
                deleted = True
            except Exception:
                pass
        return deleted
