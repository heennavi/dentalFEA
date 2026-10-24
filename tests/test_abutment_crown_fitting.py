"""
tests/test_abutment_crown_fitting.py
Unit tests for:
1. Abutment watertight 2-manifold surface and parametric adaptation to implant hole dimensions.
2. Crown internal cavity creation matching Abutment upper post geometry.
3. Gmsh volume tetrahedral meshing for Abutment and Crown without self-intersection PLC errors.
4. VolumeViewerWindow implant axes_assembly visibility suppression when implant selection is disabled.
5. Implant simulation dialog color option 'Titanium(TiN3)'.
"""

import os
import sys
import math
import numpy as np
import pytest
import vtk

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.geometry.surface_extract import _vtk_polydata_to_trimesh
from dental_fea_app.placement.implant_generator import (
    ImplantSpec,
    AbutmentSpec,
    ProstheticSpec,
    generate_abutment_polydata,
    generate_prosthetic_polydata,
)
from dental_fea_app.placement.implant_placement_manager import ImplantPlacementManager
from meshing.gmsh_mesher import build_part_mesh


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_abutment_watertight_and_implant_fitting():
    """Abutment 메쉬가 임플란트 홀 규격에 맞게 변형되며, 단일 watertight 2-manifold 솔리드를 형성하는지 검증"""
    imp_spec = ImplantSpec(
        id="test_imp",
        name="Test Implant",
        diameter_mm=4.5,
        length_mm=10.0,
        taper_deg=2.0,
        top_type="collar",
        thread_type="triangular",
        micro_threads=True,
    )
    abut_spec = AbutmentSpec(
        diameter_mm=4.5,
        flare_deg=25.0,
        collar_len_mm=3.0,
        upper_len_mm=4.5,
        upper_angle_deg=0.0,
    )

    poly = generate_abutment_polydata(abut_spec, imp_spec)
    assert poly.GetNumberOfPoints() > 0

    mesh = _vtk_polydata_to_trimesh(poly)
    assert mesh.is_watertight, "Abutment 메쉬는 반드시 watertight여야 합니다."
    assert mesh.is_volume, "Abutment 메쉬는 유효한 양의 체적을 가져야 합니다."
    assert mesh.euler_number == 2, f"Abutment Euler 수는 2(구상동형)여야 하나 {mesh.euler_number}입니다."

    # Hex 하부 돌출부 깊이 검증 (z min이 임플란트 홀 깊이와 일치)
    expected_depth = min(2.5, 10.0 * 0.35)
    z_min = mesh.vertices[:, 2].min()
    assert math.isclose(z_min, -expected_depth, abs_tol=1e-2)


def test_crown_watertight_and_abutment_cavity_fitting():
    """Crown 메쉬가 Abutment 상부 기둥에 맞춘 오목한 내부 홀(Cavity)을 형성하며 watertight를 유지하는지 검증"""
    abut_spec = AbutmentSpec(
        diameter_mm=4.0,
        flare_deg=30.0,
        collar_len_mm=2.5,
        upper_len_mm=4.0,
        upper_angle_deg=0.0,
    )
    crown_spec = ProstheticSpec(
        tooth_number=46,
        length_mm=7.5,
        width_mm=10.5,
        depth_mm=11.0,
    )

    poly = generate_prosthetic_polydata(crown_spec, abut_spec)
    assert poly.GetNumberOfPoints() > 0

    mesh = _vtk_polydata_to_trimesh(poly)
    assert mesh.is_watertight, "Crown 메쉬는 반드시 watertight여야 합니다."
    assert mesh.is_volume, "Crown 메쉬는 유효한 양의 체적을 가져야 합니다."
    assert mesh.euler_number == 2, f"Crown Euler 수는 2여야 하나 {mesh.euler_number}입니다."

    # Crown 바닥이 열린 구멍이 아니라 내부 오목면으로 닫혀 있는지 검증 (z=0 근처 및 상부 공동 존재)
    # 바닥 마진 z는 0, 내부 공동 천장은 z=4.0 부근
    assert mesh.vertices[:, 2].min() >= -1e-4
    assert mesh.volume > 200.0, "치관 체적이 정상 범위여야 합니다."


def test_abutment_and_crown_volume_meshing():
    """Abutment 및 Crown 표면 메쉬에 대해 Gmsh 체적 메싱이 self-intersection 오류 없이 성공하는지 검증"""
    abut_spec = AbutmentSpec(diameter_mm=4.0, flare_deg=30.0, collar_len_mm=2.5, upper_len_mm=4.0)
    crown_spec = ProstheticSpec(tooth_number=46, length_mm=7.5, width_mm=10.5, depth_mm=11.0)

    poly_abut = generate_abutment_polydata(abut_spec)
    poly_crown = generate_prosthetic_polydata(crown_spec, abut_spec)

    mesh_abut = _vtk_polydata_to_trimesh(poly_abut)
    mesh_crown = _vtk_polydata_to_trimesh(poly_crown)

    # Gmsh 3D 체적 메싱
    part_abut = build_part_mesh(name="abutment", surface=mesh_abut, size_far=0.5, size_near=0.5, order=2)
    assert len(part_abut.nodes) > 0
    assert len(part_abut.elements) > 0
    assert part_abut.element_type == "C3D10"

    part_crown = build_part_mesh(name="crown", surface=mesh_crown, size_far=0.5, size_near=0.5, order=2)
    assert len(part_crown.nodes) > 0
    assert len(part_crown.elements) > 0
    assert part_crown.element_type == "C3D10"


def test_axes_assembly_hidden_on_selectable_unchecked(qapp):
    """3D 볼륨 뷰어에서 임플란트 선택 해제 시 RGB 3차원 축선(axes_assembly)이 기즈모와 함께 감춰지는지 검증"""
    from dental_fea_app.ui.volume_viewer_window import VolumeViewerWindow

    sm = StateManager()
    mgr = ImplantPlacementManager()
    sm.implant_placement_mgr = mgr

    viewer = VolumeViewerWindow(sm)

    spec = ImplantSpec(
        id="imp_axes_test",
        name="Test",
        diameter_mm=4.0,
        length_mm=10.0,
        taper_deg=2.0,
        top_type="collar",
        thread_type="triangular",
        micro_threads=True,
    )
    imp_id = mgr.place_implant(spec, (10.0, 20.0, 30.0))
    mgr.select_implant(imp_id)
    imp = mgr.implants[imp_id]

    assert imp.axes_assembly is not None
    assert imp.axes_assembly.GetVisibility() == 1

    # 1. 체크 해제 시 축선도 즉시 숨겨짐
    viewer.chk_implant_selectable.setChecked(False)
    assert imp.axes_assembly.GetVisibility() == 0

    # 2. 체크 다시 활성화 시 축선 복구
    viewer.chk_implant_selectable.setChecked(True)
    assert imp.axes_assembly.GetVisibility() == 1


def test_dialog_color_option_tin3(qapp):
    """시뮬레이션 대화상자의 색상 옵션 라디오버튼 텍스트가 'Titanium(TiN3)'인지 검증"""
    from dental_fea_app.ui.placement.implant_simulation_dialog import ImplantSimulationDialog

    mgr = ImplantPlacementManager()
    dlg = ImplantSimulationDialog(mgr)
    assert hasattr(dlg, "rb_abut_gold")
    assert dlg.rb_abut_gold.text() == "Titanium(TiN3)"
