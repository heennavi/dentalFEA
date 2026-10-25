"""
test_screw_fea_and_preload.py
Unit and integration tests for screw inclusion in FEA, thread tolerance contact, and bolt preload.
"""

import os
import json
import numpy as np
import pytest
import trimesh

from dental_fea_app.core.state_manager import StateManager
from dental_fea_app.placement.implant_generator import (
    ImplantLibraryManager, ImplantSpec, AbutmentSpec, ProstheticSpec, ScrewSpec,
    Implant3DGenerator, generate_abutment_polydata, generate_screw_polydata,
)
from dental_fea_app.placement.implant_placement_manager import ImplantPlacementManager
from dental_fea_app.geometry.boolean_ops import FeaGeometry, build_fea_geometry
from dental_fea_app.ui.geometry.geometry_worker import (
    get_implant_world_mesh, get_abutment_world_mesh, get_screw_world_mesh, get_crown_world_mesh,
)
from fea.models import MeshModel, MeshPart, AnalysisConfig, MaterialAssignment, BoundaryConditionSet
from meshing.gmsh_mesher import build_part_mesh
from meshing.surface_extractor import contact_pair_from_parts
from solver.calculix_writer import write_inp


USER_INFO_PATH = "/home/kim/Documents/project/dental_FEA/saved/user_info_20261025_021845.json"


@pytest.fixture
def sample_user_info_mgr():
    """사용자 정보 JSON 파일을 로드하여 초기화된 ImplantPlacementManager 반환"""
    assert os.path.exists(USER_INFO_PATH), f"사용자 정보 파일이 없습니다: {USER_INFO_PATH}"
    with open(USER_INFO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    state_mgr = StateManager()
    mgr = ImplantPlacementManager()
    mgr.set_state_manager(state_mgr)
    state_mgr.implant_placement_mgr = mgr

    implants_data = data.get("implants", [])
    mgr.from_dict(implants_data)
    return state_mgr, mgr


def test_01_user_info_load_and_screw_restoration(sample_user_info_mgr):
    """user_info_20261025_021845.json 로드 시 스크류가 자동 장착되고 월드 메쉬가 정상 추출되는지 검증"""
    state_mgr, mgr = sample_user_info_mgr
    assert len(mgr.implants) >= 1
    imp = next(iter(mgr.implants.values()))

    # 1. Abutment와 Screw가 모두 존재해야 함
    assert imp.abutment is not None
    assert imp.screw is not None
    assert imp.raw_screw_polydata is not None

    # 2. 월드 메쉬 추출 검증
    screw_world = get_screw_world_mesh(state_mgr)
    assert screw_world is not None
    assert len(screw_world.faces) > 0
    assert screw_world.is_watertight

    # 3. 임플란트 및 어버트먼트 월드 메쉬 검증
    imp_world = get_implant_world_mesh(state_mgr)
    abut_world = get_abutment_world_mesh(state_mgr)
    assert imp_world is not None and imp_world.is_watertight
    assert abut_world is not None and abut_world.is_watertight


def test_02_screw_meshing_and_mesh_part(sample_user_info_mgr):
    """Gmsh를 통해 screw_part가 10절점 C3D10 2차 사면체로 정상 메싱되는지 검증"""
    state_mgr, mgr = sample_user_info_mgr
    screw_world = get_screw_world_mesh(state_mgr)

    part_screw = build_part_mesh(
        name="screw",
        surface=screw_world,
        size_far=0.5,
        size_near=0.5,
        order=2,
    )

    assert isinstance(part_screw, MeshPart)
    assert part_screw.name == "screw"
    assert part_screw.element_type == "C3D10"
    assert len(part_screw.nodes) > 0
    assert len(part_screw.elements) > 0
    assert part_screw.elements.shape[1] == 10


def test_03_thread_contact_extraction_with_tolerance(sample_user_info_mgr):
    """임플란트 암나사와 스크류 수나사 사이의 일반 공차(0.20mm) 기반 접촉(*TIE) 추출 검증"""
    state_mgr, mgr = sample_user_info_mgr
    imp_world = get_implant_world_mesh(state_mgr)
    abut_world = get_abutment_world_mesh(state_mgr)
    screw_world = get_screw_world_mesh(state_mgr)

    part_imp = build_part_mesh("implant", imp_world, size_far=0.6, size_near=0.6, order=2)
    part_abut = build_part_mesh("abutment", abut_world, size_far=0.6, size_near=0.6, order=2)
    part_screw = build_part_mesh("screw", screw_world, size_far=0.5, size_near=0.5, order=2)

    # 1. 스크류 헤드와 어버트먼트 턱 접촉 (TIE_SCREW_ABUTMENT)
    tie_sc_abut = contact_pair_from_parts(
        part_a=part_abut,
        part_b=part_screw,
        tol_mm=0.20,
        master="auto",
        name="TIE_SCREW_ABUTMENT"
    )
    assert len(tie_sc_abut.master_faces) > 0
    assert len(tie_sc_abut.slave_nodes) > 0

    # 2. 임플란트 암나사와 스크류 수나사 접촉 (TIE_IMPLANT_SCREW, 일반 공차 0.20mm)
    tie_imp_sc = contact_pair_from_parts(
        part_a=part_imp,
        part_b=part_screw,
        tol_mm=0.20,
        master="a",
        name="TIE_IMPLANT_SCREW"
    )
    assert len(tie_imp_sc.master_faces) > 0
    assert len(tie_imp_sc.slave_nodes) > 0


def test_04_screw_material_assignment_and_preload_bc(sample_user_info_mgr):
    """FEAController를 통한 스크류 물성 매핑 및 Preload 하중 세트 생성 검증"""
    state_mgr, mgr = sample_user_info_mgr
    imp_world = get_implant_world_mesh(state_mgr)
    screw_world = get_screw_world_mesh(state_mgr)

    # 가상 뼈 박스 메쉬
    box_bone = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    box_bone.apply_translation([78.5, 31.2, -120.0])

    part_bone = build_part_mesh("bone", box_bone, size_far=2.0, size_near=2.0, order=2)
    part_imp = build_part_mesh("implant", imp_world, size_far=0.8, size_near=0.8, order=2)
    part_screw = build_part_mesh("screw", screw_world, size_far=0.6, size_near=0.6, order=2)

    mesh_model = MeshModel(parts=[part_bone, part_imp, part_screw])
    state_mgr.set_mesh_model(mesh_model)

    from dental_fea_app.ui.fea_controller import FeaController
    controller = FeaController(state_manager=state_mgr)

    # 1. Preload 활성화 시 경계조건 설정
    issues = controller.setup_boundary_conditions(
        bottom_z=-129.0,
        load_mag=100.0,
        load_angle=0.0,
        preload_enabled=True,
        preload_force_n=200.0,
    )
    bc_set = state_mgr.bc_set
    assert bc_set is not None

    # OCCLUSAL_FORCE 1개 + PRELOAD_SCREW_HEAD 1개 + PRELOAD_SCREW_THREAD 1개 = 총 3개 하중
    assert len(bc_set.loads) == 3
    load_names = [ld["name"] for ld in bc_set.loads]
    assert "OCCLUSAL_FORCE" in load_names
    assert "PRELOAD_SCREW_HEAD" in load_names
    assert "PRELOAD_SCREW_THREAD" in load_names

    head_load = next(ld for ld in bc_set.loads if ld["name"] == "PRELOAD_SCREW_HEAD")
    thread_load = next(ld for ld in bc_set.loads if ld["name"] == "PRELOAD_SCREW_THREAD")

    # 두 하중의 크기는 200 N이고 힘 벡터의 방향은 서로 반대 (합력이 0)
    assert np.isclose(head_load["magnitude_N"], 200.0)
    assert np.isclose(thread_load["magnitude_N"], 200.0)
    force_sum = np.array(head_load["force_N"]) + np.array(thread_load["force_N"])
    assert np.allclose(force_sum, [0.0, 0.0, 0.0], atol=1e-5)

    # 2. Preload 비활성화 시 경계조건 설정
    controller.setup_boundary_conditions(
        bottom_z=-129.0,
        load_mag=100.0,
        preload_enabled=False,
    )
    bc_no_pre = state_mgr.bc_set
    assert len(bc_no_pre.loads) == 1
    assert bc_no_pre.loads[0]["name"] == "OCCLUSAL_FORCE"


def test_05_calculix_inp_generation_with_screw_and_preload(tmp_path, sample_user_info_mgr):
    """스크류 파트, *TIE, 물성, Preload 하중이 포함된 CalculiX .inp 파일 작성 검증"""
    state_mgr, mgr = sample_user_info_mgr
    imp_world = get_implant_world_mesh(state_mgr)
    screw_world = get_screw_world_mesh(state_mgr)

    box_bone = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    box_bone.apply_translation([78.5, 31.2, -120.0])

    part_bone = build_part_mesh("bone", box_bone, size_far=2.5, size_near=2.5, order=2)
    part_imp = build_part_mesh("implant", imp_world, size_far=1.0, size_near=1.0, order=2)
    part_screw = build_part_mesh("screw", screw_world, size_far=0.8, size_near=0.8, order=2)

    mesh_model = MeshModel(parts=[part_bone, part_imp, part_screw])
    state_mgr.set_mesh_model(mesh_model)

    # 접촉면(*TIE)
    tie_imp_sc = contact_pair_from_parts(part_imp, part_screw, tol_mm=0.20, master="a", name="TIE_IMPLANT_SCREW")

    # 경계조건 및 Preload
    from dental_fea_app.ui.fea_controller import FeaController
    controller = FeaController(state_manager=state_mgr)
    controller.setup_boundary_conditions(
        bottom_z=-129.0,
        load_mag=100.0,
        preload_enabled=True,
        preload_force_n=200.0,
    )
    bc_set = state_mgr.bc_set

    # 물성 정의
    from fea.models import MaterialDef
    mat_bone = MaterialDef("bone_mat", E=15000.0, nu=0.3)
    mat_ti = MaterialDef("titanium", E=110000.0, nu=0.34)
    mat_assign = MaterialAssignment(
        element_material={
            "bone": np.zeros(len(part_bone.elements), dtype=np.int64),
            "implant": np.ones(len(part_imp.elements), dtype=np.int64),
            "screw": np.ones(len(part_screw.elements), dtype=np.int64),
        },
        materials=[mat_bone, mat_ti]
    )

    inp_path = str(tmp_path / "dental_test_screw.inp")
    write_inp(
        path=inp_path,
        model=mesh_model,
        materials={"bone": mat_assign, "implant": mat_assign, "screw": mat_assign},
        bc=bc_set,
        ties=[tie_imp_sc],
        title="dental_screw_test",
    )

    assert os.path.exists(inp_path)
    with open(inp_path, "r", encoding="ascii") as f:
        content = f.read()

    # .inp 필수 키워드 및 파트 검증
    assert "*ELEMENT, TYPE=C3D10, ELSET=EALL_SCREW" in content
    assert "*TIE, NAME=TIE_IMPLANT_SCREW" in content
    assert "*CLOAD" in content
    assert "SURF_M_TIE_IMPLANT_SCREW" in content
    assert "SURF_S_TIE_IMPLANT_SCREW" in content
