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


@dataclass
class ScrewSpec:
    """치과 어버트먼트 스크류(Abutment Screw) 규격 사양"""
    thread_name: str = "M1.8"           # M1.6, M1.8, M2.0
    thread_major_mm: float = 1.80       # 나사 외경
    thread_pitch_mm: float = 0.35       # 나사 피치
    head_diameter_mm: float = 2.20      # 스크류 헤드 외경
    head_length_mm: float = 1.80        # 스크류 헤드 길이
    driver_hex_mm: float = 1.25         # 내부 헥스 드라이버 홈 크기 (대변거리)
    shaft_diameter_mm: float = 1.60     # 중간 샤프트 외경
    total_length_mm: float = 7.10       # 전체 길이
    thread_length_mm: float = 3.00      # 결합 나사산 길이
    seat_z_offset_mm: float = 0.80      # Abutment 내부 턱 안착 높이 (Abutment 로컬 z 좌표)
    color_rgb: Tuple[float, float, float] = (0.78, 0.80, 0.83) # 티타늄 메탈릭 실버


def get_screw_hole_params(implant_diameter: float, implant_length: float) -> Dict[str, Any]:
    """임플란트 직경과 길이에 따른 표준 내부 헥사 및 암나사 스크류홀 규격 반환"""
    d = float(implant_diameter)
    l = float(implant_length)

    if d < 3.4:
        hex_across_flats = 2.10
        thread_name = "M1.6"
        thread_pitch = 0.35
        thread_minor_r = 1.22 / 2.0  # 0.61 mm
        thread_major_r = 1.60 / 2.0  # 0.80 mm
        head_diam = 2.00
    elif d < 4.4:
        hex_across_flats = 2.44
        thread_name = "M1.8"
        thread_pitch = 0.35
        thread_minor_r = 1.42 / 2.0  # 0.71 mm
        thread_major_r = 1.80 / 2.0  # 0.90 mm
        head_diam = 2.20
    else:
        hex_across_flats = 2.50
        thread_name = "M2.0"
        thread_pitch = 0.40
        thread_minor_r = 1.57 / 2.0  # 0.785 mm
        thread_major_r = 2.00 / 2.0  # 1.00 mm
        head_diam = 2.30

    hex_depth = 1.50
    # 최소 유효 결합 길이 1.8mm, 권장 2.5~3.5mm
    thread_len = max(1.8, min(3.5, l * 0.45 - hex_depth))

    return {
        "hex_across_flats": hex_across_flats,
        "hex_depth": hex_depth,
        "thread_name": thread_name,
        "thread_pitch": thread_pitch,
        "thread_minor_r": thread_minor_r,
        "thread_major_r": thread_major_r,
        "thread_len": thread_len,
        "head_diam": head_diam,
    }


def get_hex_radius_at_angle(angle: float, across_flats: float) -> float:
    """정육각형의 중심에서 특정 각도(angle, rad) 방향 변까지의 거리(반지름) 반환"""
    half_flat = float(across_flats) / 2.0
    seg = math.pi / 3.0
    mod_ang = ((angle + (math.pi / 6.0)) % seg) - (math.pi / 6.0)
    return half_flat / max(1e-4, math.cos(mod_ang))


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

        # 규격 스크류 홀 파라미터 (내부 헥사 + 암나사)
        hole_params = get_screw_hole_params(diameter, length)
        hex_af = hole_params["hex_across_flats"]
        hex_depth = hole_params["hex_depth"] # 1.5mm
        t_pitch = hole_params["thread_pitch"]
        t_minor_r = hole_params["thread_minor_r"]
        t_major_r = hole_params["thread_major_r"]
        t_len = hole_params["thread_len"]

        # 홀 내부 Z 레벨 및 프로파일 정의
        hole_bottom_z = length - 1.65 - t_len
        n_thread_steps = max(8, int(t_len / max(0.1, t_pitch) * 2))

        inner_ring_point_indices = []

        # 1. 상단 리드인 챔퍼 입구 (z = length)
        ring_0 = []
        for j in range(n_radial):
            ang = 2.0 * math.pi * j / float(n_radial)
            r = get_hex_radius_at_angle(ang, hex_af) + 0.15
            ring_0.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), length))
        inner_ring_point_indices.append(ring_0)

        # 2. 내부 헥사 상단 (z = length - 0.2, 45도 리드인 챔퍼 바닥)
        ring_hex_top = []
        for j in range(n_radial):
            ang = 2.0 * math.pi * j / float(n_radial)
            r = get_hex_radius_at_angle(ang, hex_af)
            ring_hex_top.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), length - 0.2))
        inner_ring_point_indices.append(ring_hex_top)

        # 3. 내부 헥사 중단 (z = length - 0.8)
        ring_hex_mid = []
        for j in range(n_radial):
            ang = 2.0 * math.pi * j / float(n_radial)
            r = get_hex_radius_at_angle(ang, hex_af)
            ring_hex_mid.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), length - 0.8))
        inner_ring_point_indices.append(ring_hex_mid)

        # 4. 내부 헥사 하단 (z = length - 1.5, 헥사 깊이 1.5mm)
        ring_hex_bot = []
        for j in range(n_radial):
            ang = 2.0 * math.pi * j / float(n_radial)
            r = get_hex_radius_at_angle(ang, hex_af)
            ring_hex_bot.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), length - hex_depth))
        inner_ring_point_indices.append(ring_hex_bot)

        # 5. 암나사 전이부 입구 (z = length - 1.65)
        ring_trans = []
        for j in range(n_radial):
            ang = 2.0 * math.pi * j / float(n_radial)
            ring_trans.append(points.InsertNextPoint(t_major_r * math.cos(ang), t_major_r * math.sin(ang), length - 1.65))
        inner_ring_point_indices.append(ring_trans)

        # 6. 암나사 나선부 (z = length - 1.65 ~ hole_bottom_z)
        for s in range(1, n_thread_steps + 1):
            t_frac = s / float(n_thread_steps)
            z_curr = (length - 1.65) - t_frac * t_len
            ring_thread = []
            phase = ((length - 1.65 - z_curr) / t_pitch) % 1.0
            for j in range(n_radial):
                ang = 2.0 * math.pi * j / float(n_radial)
                # 나선형 나사산 형성
                thread_wave = 0.5 * (1.0 + math.sin(2.0 * math.pi * (phase + j / float(n_radial))))
                r = t_minor_r + (t_major_r - t_minor_r) * thread_wave
                ring_thread.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), z_curr))
            inner_ring_point_indices.append(ring_thread)

        # 7. 바닥 테이퍼 드릴 팁
        hole_bottom_center_idx = points.InsertNextPoint(0.0, 0.0, hole_bottom_z - 0.4 * t_minor_r)

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
    치과 어버트먼트 3D 메쉬 생성.
    - 하부: 임플란트 내부 헥사에 정밀 대응하는 수(Male) 헥사 돌출부 (결합 높이 1.5mm, across flats 2.44mm)
    - 외측: Collar(플레어 원추대) 및 Morse Taper 상부 기둥
    - 내부: 스크류 체결 채널 및 스크류 헤드 안착용 내부 턱(Shoulder / Seat) 구조 구비
    - 상하 관통 닫힌 2-매니폴드(watertight) 표면 형성
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

    # Fixture 내부 헥사 규격에 맞춘 수 헥사 규격 (공차 0.02mm 적용)
    if implant_spec is not None:
        hole_params = get_screw_hole_params(implant_spec.diameter_mm, implant_spec.length_mm)
        male_hex_af = hole_params["hex_across_flats"] - 0.02
        hex_depth = hole_params["hex_depth"] # 1.5mm
    else:
        male_hex_af = 2.42
        hex_depth = 1.50

    # 스크류 채널 규격: 상부 헤드 채널 및 내부 턱, 하부 관통 홀
    seat_z = 0.80 # Collar 하단 내부 턱 높이
    r_ch_head = min(1.15, max(0.8, r_upper_top - 0.35)) # 헤드 통과 반경 (직경 약 2.2~2.3mm)
    r_ch_shaft = 0.90 # 샤프트 관통 반경 (직경 1.8mm, M1.6/M1.8 스크류 샤프트 무간섭 통과)

    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    n_radial = 48 # 6의 배수로 헥사와 원형 단면이 1:1 완벽 정렬

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

    # ==========================================================
    # 1. 외벽 링 생성 (Outer Surface)
    # ==========================================================
    outer_rings = []

    # 1-1. 수 헥사 바닥 (z = -hex_depth) 및 상단 (z = 0.0)
    hex_bot_ring = []
    hex_top_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        r = get_hex_radius_at_angle(ang, male_hex_af)
        hex_bot_ring.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), -hex_depth))
        hex_top_ring.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), 0.0))
    outer_rings.append(hex_bot_ring)
    outer_rings.append(hex_top_ring)

    # 1-2. Collar 바닥 (z = 0.0, 외경 r_bottom)
    # 수 헥사 외경과 Collar 바닥 사이의 수평 환형 림(Shoulder) 형성용 링
    collar_bot_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        collar_bot_ring.append(points.InsertNextPoint(r_bottom * math.cos(ang), r_bottom * math.sin(ang), 0.0))
    outer_rings.append(collar_bot_ring)

    # 1-3. Collar 상승 구간 (z = 0.0 ~ collar_len_mm)
    n_collar_steps = max(4, int(collar_len_mm * 2))
    for s in range(1, n_collar_steps + 1):
        t = s / float(n_collar_steps)
        cz = t * collar_len_mm
        cr = r_bottom + cz * math.tan(flare_rad)
        c_ring = []
        for j in range(n_radial):
            ang = 2.0 * math.pi * j / float(n_radial)
            c_ring.append(points.InsertNextPoint(cr * math.cos(ang), cr * math.sin(ang), cz))
        outer_rings.append(c_ring)

    # 1-4. Upper Post 기둥 (z = collar_len_mm ~ collar_len_mm + upper_len_mm)
    rot_trans = vtk.vtkTransform()
    rot_trans.Identity()
    rot_trans.RotateZ(upper_dir_deg)
    rot_trans.RotateX(upper_angle_deg)
    rot_trans.RotateZ(-upper_dir_deg)

    # Collar 상단에서 Upper 기둥 기저부로의 연결 링 (r_upper_bottom)
    upper_base_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        upper_base_ring.append(points.InsertNextPoint(r_upper_bottom * math.cos(ang), r_upper_bottom * math.sin(ang), collar_len_mm))
    outer_rings.append(upper_base_ring)

    n_upper_steps = max(6, int(upper_len_mm * 2))
    for s in range(1, n_upper_steps + 1):
        t = s / float(n_upper_steps)
        uz_local = t * upper_len_mm
        ur = r_upper_bottom - uz_local * math.tan(taper_rad)
        u_ring = []
        for j in range(n_radial):
            ang = 2.0 * math.pi * j / float(n_radial)
            lx = ur * math.cos(ang)
            ly = ur * math.sin(ang)
            lz = uz_local
            wx, wy, wz = rot_trans.TransformPoint(lx, ly, lz)
            u_ring.append(points.InsertNextPoint(wx, wy, collar_len_mm + wz))
        outer_rings.append(u_ring)

    # 외벽 쿼드 연결
    for s in range(len(outer_rings) - 1):
        r1 = outer_rings[s]
        r2 = outer_rings[s + 1]
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_quad(r1[j], r1[jnxt], r2[jnxt], r2[j])

    # ==========================================================
    # 2. 내부 스크류 채널 및 내부 턱(Shoulder / Seat) 생성
    # ==========================================================
    inner_rings = []

    # 2-1. 상단 스크류 채널 입구 링 (Upper post 상단 높이, 반경 r_ch_head)
    ch_top_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        lx = r_ch_head * math.cos(ang)
        ly = r_ch_head * math.sin(ang)
        lz = upper_len_mm
        wx, wy, wz = rot_trans.TransformPoint(lx, ly, lz)
        ch_top_ring.append(points.InsertNextPoint(wx, wy, collar_len_mm + wz))
    inner_rings.append(ch_top_ring)

    # 2-2. Upper 기둥 하단 부근 스크류 채널
    ch_mid_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        ch_mid_ring.append(points.InsertNextPoint(r_ch_head * math.cos(ang), r_ch_head * math.sin(ang), collar_len_mm))
    inner_rings.append(ch_mid_ring)

    # 2-3. 내부 안착 턱(Seat) 상단 (z = seat_z, 반경 r_ch_head)
    seat_top_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        seat_top_ring.append(points.InsertNextPoint(r_ch_head * math.cos(ang), r_ch_head * math.sin(ang), seat_z))
    inner_rings.append(seat_top_ring)

    # 2-4. 내부 안착 턱(Seat) 하단 (z = seat_z, 반경 r_ch_shaft) -> 스크류 헤드 걸림 턱
    seat_bot_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        seat_bot_ring.append(points.InsertNextPoint(r_ch_shaft * math.cos(ang), r_ch_shaft * math.sin(ang), seat_z))
    inner_rings.append(seat_bot_ring)

    # 2-5. 하부 관통 홀 중간 (z = 0.0, 반경 r_ch_shaft)
    shaft_mid_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        shaft_mid_ring.append(points.InsertNextPoint(r_ch_shaft * math.cos(ang), r_ch_shaft * math.sin(ang), 0.0))
    inner_rings.append(shaft_mid_ring)

    # 2-6. 하부 관통 홀 바닥 (z = -hex_depth, 반경 r_ch_shaft)
    shaft_bot_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        shaft_bot_ring.append(points.InsertNextPoint(r_ch_shaft * math.cos(ang), r_ch_shaft * math.sin(ang), -hex_depth))
    inner_rings.append(shaft_bot_ring)

    # 내벽 쿼드 연결 (법선이 솔리드 외측, 즉 중공 내부 중심을 향하도록 와인딩)
    for s in range(len(inner_rings) - 1):
        r1 = inner_rings[s]
        r2 = inner_rings[s + 1]
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_quad(r1[jnxt], r1[j], r2[j], r2[jnxt])

    # ==========================================================
    # 3. 상단 및 하단 환형 캡 연결 (닫힌 솔리드 완성)
    # ==========================================================
    # 상단 캡: Upper post 상단 외경 링과 스크류 채널 상단 입구 링 연결 (+Z 방향 법선)
    outer_top_ring = outer_rings[-1]
    inner_top_ring = inner_rings[0]
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_quad(outer_top_ring[j], outer_top_ring[jnxt], inner_top_ring[jnxt], inner_top_ring[j])

    # 하단 캡: 수 헥사 바닥 링과 하부 관통 홀 바닥 링 연결 (-Z 방향 법선)
    outer_bot_ring = outer_rings[0]
    inner_bot_ring = inner_rings[-1]
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_quad(inner_bot_ring[j], inner_bot_ring[jnxt], outer_bot_ring[jnxt], outer_bot_ring[j])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(cleaner.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def generate_screw_polydata(
    spec: Optional[ScrewSpec] = None,
    implant_spec: Optional[ImplantSpec] = None,
    abutment_spec: Optional[AbutmentSpec] = None,
) -> vtk.vtkPolyData:
    """
    치과 어버트먼트 스크류(Abutment Screw) 3D 메쉬 생성.
    - 헤드: Abutment 내부 턱(Seat)에 밀착 안착, 상부에 1.25mm 헥스 드라이버 홈 구비
    - 샤프트: Abutment 관통부를 지나 Fixture 스크류 홀로 진입
    - 수나사산부: Fixture 암나사와 정확히 맞물려 체결
    - Abutment 로컬 좌표계 기준으로 생성되어 Abutment 및 Fixture에 정확히 안착됨.
    """
    if spec is None:
        if implant_spec is not None:
            p = get_screw_hole_params(implant_spec.diameter_mm, implant_spec.length_mm)
            spec = ScrewSpec(
                thread_name=p["thread_name"],
                thread_major_mm=p["thread_major_r"] * 2.0,
                thread_pitch_mm=p["thread_pitch"],
                head_diameter_mm=p["head_diam"],
                thread_length_mm=p["thread_len"],
            )
        else:
            spec = ScrewSpec()

    seat_z = float(spec.seat_z_offset_mm) # 0.80 mm
    head_len = float(spec.head_length_mm) # 1.80 mm
    head_r = float(spec.head_diameter_mm) / 2.0 # 1.10 mm
    shaft_r = float(spec.shaft_diameter_mm) / 2.0 # 0.80 mm
    t_major_r = float(spec.thread_major_mm) / 2.0 # 0.90 mm
    t_minor_r = max(0.5, t_major_r - 0.19)
    t_pitch = float(spec.thread_pitch_mm) # 0.35 mm
    t_len = float(spec.thread_length_mm) # 3.0 mm
    driver_af = float(spec.driver_hex_mm) # 1.25 mm
    driver_depth = 0.80 # mm

    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    n_radial = 48

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

    head_top_z = seat_z + head_len
    driver_bot_z = head_top_z - driver_depth
    screw_bot_z = -1.50 - t_len

    # ==========================================================
    # 1. 스크류 헤드 상단 헥스 드라이버 홈
    # ==========================================================
    # 1-1. 헤드 상단 외경 링 (z = head_top_z, 반경 head_r)
    head_top_outer = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        head_top_outer.append(points.InsertNextPoint(head_r * math.cos(ang), head_r * math.sin(ang), head_top_z))

    # 1-2. 드라이버 홈 상단 링 (z = head_top_z, 헥스 반경)
    driver_top_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        r = get_hex_radius_at_angle(ang, driver_af)
        driver_top_ring.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), head_top_z))

    # 1-3. 드라이버 홈 바닥 링 (z = driver_bot_z, 헥스 반경)
    driver_bot_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        r = get_hex_radius_at_angle(ang, driver_af)
        driver_bot_ring.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), driver_bot_z))

    driver_bottom_center = points.InsertNextPoint(0.0, 0.0, driver_bot_z)

    # 헤드 상단면 연결 (외경과 헥스 홈 사이 환형 상단면)
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_quad(head_top_outer[j], head_top_outer[jnxt], driver_top_ring[jnxt], driver_top_ring[j])

    # 드라이버 홈 내벽 연결
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_quad(driver_top_ring[jnxt], driver_top_ring[j], driver_bot_ring[j], driver_bot_ring[jnxt])

    # 드라이버 홈 바닥면 닫기
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_tri(driver_bottom_center, driver_bot_ring[jnxt], driver_bot_ring[j])

    # ==========================================================
    # 2. 스크류 외벽 기둥들 (헤드, 안착면, 샤프트, 수나사산)
    # ==========================================================
    body_rings = [head_top_outer]

    # 2-1. 헤드 하단 링 (z = seat_z, 반경 head_r)
    head_bot_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        head_bot_ring.append(points.InsertNextPoint(head_r * math.cos(ang), head_r * math.sin(ang), seat_z))
    body_rings.append(head_bot_ring)

    # 2-2. 안착면 림 (z = seat_z, 반경 shaft_r) -> Abutment 턱에 안착되는 환형 면
    seat_rim_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        seat_rim_ring.append(points.InsertNextPoint(shaft_r * math.cos(ang), shaft_r * math.sin(ang), seat_z))
    body_rings.append(seat_rim_ring)

    # 2-3. 무나사 샤프트 하단 (z = -1.50, 반경 shaft_r)
    shaft_bot_ring = []
    for j in range(n_radial):
        ang = 2.0 * math.pi * j / float(n_radial)
        shaft_bot_ring.append(points.InsertNextPoint(shaft_r * math.cos(ang), shaft_r * math.sin(ang), -1.50))
    body_rings.append(shaft_bot_ring)

    # 2-4. 수나사산 구간 (z = -1.50 ~ screw_bot_z)
    n_thread_steps = max(10, int(t_len / max(0.1, t_pitch) * 2))
    for s in range(1, n_thread_steps + 1):
        t_frac = s / float(n_thread_steps)
        z_curr = -1.50 - t_frac * t_len
        th_ring = []
        phase = ((-1.50 - z_curr) / t_pitch) % 1.0
        for j in range(n_radial):
            ang = 2.0 * math.pi * j / float(n_radial)
            thread_wave = 0.5 * (1.0 + math.sin(2.0 * math.pi * (phase + j / float(n_radial))))
            r = t_minor_r + (t_major_r - t_minor_r) * thread_wave
            th_ring.append(points.InsertNextPoint(r * math.cos(ang), r * math.sin(ang), z_curr))
        body_rings.append(th_ring)

    # 외벽 쿼드 연결
    for s in range(len(body_rings) - 1):
        r1 = body_rings[s]
        r2 = body_rings[s + 1]
        for j in range(n_radial):
            jnxt = (j + 1) % n_radial
            add_quad(r1[j], r1[jnxt], r2[jnxt], r2[j])

    # 2-5. 스크류 바닥 캡
    screw_bottom_center = points.InsertNextPoint(0.0, 0.0, screw_bot_z - 0.3 * t_minor_r)
    last_ring = body_rings[-1]
    for j in range(n_radial):
        jnxt = (j + 1) % n_radial
        add_tri(screw_bottom_center, last_ring[j], last_ring[jnxt])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(cleaner.GetOutputPort())
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
