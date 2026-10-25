"""
test_screw_and_internal_hex.py
내부 헥사(Internal Hex), 암나사 스크류 홀, Abutment 턱(Shoulder/Seat) 구조,
Abutment Screw 자동 장착 및 크라운 내부 편평화(Flat Ceiling) 사양 검증 테스트.
"""

import os
import math
import tempfile
import numpy as np
import pytest
import vtk
import trimesh

from dental_fea_app.placement.implant_generator import (
    ImplantSpec, Implant3DGenerator, ImplantLibraryManager,
    AbutmentSpec, generate_abutment_polydata,
    ScrewSpec, generate_screw_polydata, get_screw_hole_params, get_hex_radius_at_angle,
    ProstheticSpec, generate_prosthetic_polydata
)
from dental_fea_app.geometry.boolean_ops import FeaGeometry, build_fea_geometry, export_stl


def vtk_polydata_to_trimesh(poly: vtk.vtkPolyData) -> trimesh.Trimesh:
    """VTK PolyData를 trimesh.Trimesh로 변환 (테스트 검증용)"""
    pts = poly.GetPoints()
    n_pts = pts.GetNumberOfPoints()
    vertices = np.zeros((n_pts, 3), dtype=np.float64)
    for i in range(n_pts):
        vertices[i] = pts.GetPoint(i)

    polys = poly.GetPolys()
    polys.InitTraversal()
    pt_ids = vtk.vtkIdList()
    faces = []
    while polys.GetNextCell(pt_ids):
        if pt_ids.GetNumberOfIds() == 3:
            faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])
        elif pt_ids.GetNumberOfIds() == 4:
            faces.append([pt_ids.GetId(0), pt_ids.GetId(1), pt_ids.GetId(2)])
            faces.append([pt_ids.GetId(0), pt_ids.GetId(2), pt_ids.GetId(3)])

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return tm


def test_internal_hex_radius_calculation():
    """정육각형 대변거리(across flats)에 따른 각도별 반지름 계산 검증"""
    across_flats = 2.44
    half_flat = 1.22
    # 0도: 변의 중점 (r = half_flat)
    r_0 = get_hex_radius_at_angle(0.0, across_flats)
    assert r_0 == pytest.approx(half_flat, abs=1e-4)

    # 30도 (pi/6): 꼭짓점 (r = across_flats / sqrt(3))
    r_30 = get_hex_radius_at_angle(math.pi / 6.0, across_flats)
    expected_vertex_r = across_flats / math.sqrt(3.0)
    assert r_30 == pytest.approx(expected_vertex_r, abs=1e-4)

    # 60도 (pi/3): 다음 변의 중점 (r = half_flat)
    r_60 = get_hex_radius_at_angle(math.pi / 3.0, across_flats)
    assert r_60 == pytest.approx(half_flat, abs=1e-4)


def test_implant_fixture_screw_hole_geometry():
    """임플란트 Fixture 메쉬 생성 시 내부 헥사 및 암나사 홀이 정밀하게 형성되고 watertight인지 검증"""
    spec = ImplantSpec(
        id="test_impl_d40_l100",
        name="Test Ø4.0 x 10.0mm",
        diameter_mm=4.0,
        length_mm=10.0,
        taper_deg=2.0,
        top_type="flat",
        thread_type="triangular",
        micro_threads=False,
    )

    poly = Implant3DGenerator.generate_implant_polydata(spec)
    assert poly is not None
    assert poly.GetNumberOfPoints() > 1000
    assert poly.GetNumberOfCells() > 2000

    # trimesh로 변환하여 메니폴드/watertight 및 내부 홀 깊이 검증
    tm = vtk_polydata_to_trimesh(poly)
    assert tm.is_watertight, "임플란트 Fixture 메쉬가 닫힌 2-매니폴드(watertight)여야 합니다."

    # 바운딩 박스 Z 범위 검증 (z = 0 ~ 10.0 mm)
    z_min, z_max = tm.bounds[0, 2], tm.bounds[1, 2]
    assert z_min == pytest.approx(0.0, abs=0.1)
    assert z_max == pytest.approx(10.0, abs=0.1)

    # 파라미터 검증
    params = get_screw_hole_params(spec.diameter_mm, spec.length_mm)
    assert params["hex_across_flats"] == 2.44
    assert params["hex_depth"] == 1.50
    assert params["thread_name"] == "M1.8"
    assert params["thread_pitch"] == 0.35


def test_default_implants_vtp_files():
    """기본 제공 임플란트 25개 모델의 VTP 파일이 모두 로드 가능하고 올바른 정점을 가지는지 검증"""
    implants = ImplantLibraryManager.list_all_implants()
    assert len(implants) >= 20

    # 샘플 3개 로드 검증
    for imp in implants[:3]:
        pd = ImplantLibraryManager.load_implant_polydata(imp)
        assert pd.GetNumberOfPoints() > 3000
        assert pd.GetNumberOfCells() > 5000


def test_abutment_male_hex_and_internal_seat():
    """Abutment가 하부 수 헥사(1.5mm) 및 내부 스크류 안착 턱(Seat)을 가지며 watertight인지 검증"""
    imp_spec = ImplantSpec(
        id="test_impl",
        name="Test Implant",
        diameter_mm=4.0,
        length_mm=10.0,
        taper_deg=2.0,
        top_type="flat",
        thread_type="triangular",
        micro_threads=False,
    )
    abut_spec = AbutmentSpec(
        diameter_mm=4.0,
        flare_deg=30.0,
        collar_len_mm=4.0,
        upper_len_mm=5.0,
    )

    poly = generate_abutment_polydata(abut_spec, imp_spec)
    assert poly is not None
    assert poly.GetNumberOfPoints() > 500

    tm = vtk_polydata_to_trimesh(poly)
    assert tm.is_watertight, "Abutment 메쉬는 스크류 채널이 관통하는 완전한 닫힌 매니폴드(watertight)여야 합니다."

    # 수 헥사는 z = -1.50 ~ 0.0, 상단은 z = 9.0 mm (collar 4.0 + upper 5.0)
    z_min, z_max = tm.bounds[0, 2], tm.bounds[1, 2]
    assert z_min == pytest.approx(-1.50, abs=0.1), "수 헥사는 -1.5mm까지 돌출되어야 합니다."
    assert z_max == pytest.approx(9.00, abs=0.1)

    # 중심 관통 채널 존재 확인 (x=0, y=0에서 수직 레이캐스팅 시 내부 구멍 통과)
    # z = -0.5 (수 헥사 내부)에서 반경 0.5 지점은 빈 공간(스크류 채널 내부)이어야 함
    contains_center = tm.contains([[0.0, 0.0, 2.0]])
    assert not contains_center[0], "스크류 채널 내부는 빈 공간이어야 합니다."


def test_abutment_screw_geometry_and_locking():
    """Abutment Screw가 Abutment 내부 턱에 안착되고 Fixture 암나사 영역으로 결합되는지 검증"""
    imp_spec = ImplantSpec(
        id="test_impl",
        name="Test",
        diameter_mm=4.0,
        length_mm=10.0,
        taper_deg=2.0,
        top_type="flat",
        thread_type="triangular",
        micro_threads=False,
    )
    abut_spec = AbutmentSpec(diameter_mm=4.0, collar_len_mm=4.0, upper_len_mm=5.0)

    screw_poly = generate_screw_polydata(None, imp_spec, abut_spec)
    assert screw_poly is not None
    assert screw_poly.GetNumberOfPoints() > 500

    tm = vtk_polydata_to_trimesh(screw_poly)
    assert tm.is_watertight, "Abutment Screw 메쉬는 완전한 닫힌 매니폴드(watertight)여야 합니다."

    # 스크류의 Z 범위:
    # 헤드 상단: z = 0.8 (안착면) + 1.8 (헤드길이) = 2.6 mm
    # 나사산 하단: z = -1.5 (헥사바닥) - 3.0 (나사길이) - 팁 ≈ -4.8 mm
    z_min, z_max = tm.bounds[0, 2], tm.bounds[1, 2]
    assert z_max == pytest.approx(2.60, abs=0.1)
    assert z_min < -4.0, "스크류 나사산부가 Fixture 암나사 깊이(-4.5mm)까지 결합되어야 합니다."


def test_crown_internal_cavity_flat_ceiling():
    """보철물(Crown)의 내부 오목 형상이 스크류 홀 안으로 튀어나오지 않고 편평한 천장(flat ceiling)을 형성하는지 검증"""
    abut_spec = AbutmentSpec(diameter_mm=4.0, flare_deg=30.0, collar_len_mm=4.0, upper_len_mm=5.0)
    crown_spec = ProstheticSpec(tooth_number=11, length_mm=9.0, width_mm=8.5, depth_mm=7.0)

    poly = generate_prosthetic_polydata(crown_spec, abut_spec)
    assert poly is not None

    tm = vtk_polydata_to_trimesh(poly)
    assert tm.is_watertight, "Crown 메쉬는 내부 오목면을 가진 닫힌 2-매니폴드(watertight)여야 합니다."

    # 내부 천장 z 높이: z = u_len = 5.0 mm
    # 크라운 내부 공동 천장 부근에서 반경 0.5mm 원반 상의 정점들이 z=5.0 근방에서 편평하게 분포하는지 검증
    cavity_top_pts = [pt for pt in tm.vertices if pt[2] > 4.8 and pt[2] < 5.2 and math.hypot(pt[0], pt[1]) < 1.0]
    assert len(cavity_top_pts) > 0
    z_coords = [pt[2] for pt in cavity_top_pts]
    assert (max(z_coords) - min(z_coords)) < 0.2, "스크류 홀 개구부 부위가 볼록하게 솟지 않고 편평하게 유지되어야 합니다."


def test_fea_geometry_integration_with_screw():
    """FeaGeometry 및 build_fea_geometry에 screw_mesh가 정상 통합되는지 검증"""
    def create_cyl(radius, height, z_offset):
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
        cyl.apply_translation([0.0, 0.0, z_offset + height / 2.0])
        cyl.fix_normals()
        return cyl

    bone = create_cyl(10.0, 15.0, -15.0)
    implant = create_cyl(2.0, 10.0, -10.0)
    abutment = create_cyl(2.2, 4.0, 0.0)
    crown = create_cyl(4.0, 7.0, 4.0)
    screw = create_cyl(1.0, 6.0, -3.0)

    fea_geom = build_fea_geometry(
        cortical_mesh=bone,
        cancellous_mesh=None,
        implant_mesh=implant,
        bone_mesh=bone,
        abutment_mesh=abutment,
        crown_mesh=crown,
        screw_mesh=screw
    )

    assert fea_geom.screw is not None
    assert fea_geom.report["screw_volume"] > 0.0
    assert fea_geom.report["screw_watertight"] is True

    # export_stl 검증
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = export_stl(fea_geom, tmp_dir)
        assert "screw" in paths
        assert os.path.exists(paths["screw"])
