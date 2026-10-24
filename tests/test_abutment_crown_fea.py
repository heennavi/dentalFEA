"""
test_abutment_crown_fea.py
Abutment 및 Crown 기하 생성, 체적 메싱, TIE 완전결합, Gold/Zirconia 물성 할당,
하중 부가 위치 우선순위 검증을 위한 테스트 스위트.
"""

import os
import json
import tempfile
import numpy as np
import pytest
import trimesh

from fea.models import MeshModel, MeshPart, AnalysisConfig, MaterialDef, MaterialAssignment, BoundaryConditionSet, TieDef
from dental_fea_app.geometry.boolean_ops import FeaGeometry, build_fea_geometry, export_stl
from material.bone_mapping import load_preset
from solver.calculix_writer import write_inp


def create_cylinder_trimesh(radius: float, height: float, z_offset: float = 0.0) -> trimesh.Trimesh:
    """테스트용 실린더 매니폴드 메쉬 생성"""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    cyl.apply_translation([0.0, 0.0, z_offset + height / 2.0])
    cyl.fix_normals()
    return cyl


def test_presets_gold_and_zirconia():
    """presets.json에 Gold 및 Zirconia 물성이 정의되어 있고 올바른 물성치를 가지는지 검증"""
    preset_path = os.path.join(os.path.dirname(__file__), "..", "material", "presets.json")
    preset = load_preset(preset_path)

    # 1. Titanium 검증
    assert "titanium" in preset
    assert preset["titanium"]["E"] == pytest.approx(110000.0)
    assert preset["titanium"]["nu"] == pytest.approx(0.34)

    # 2. Gold 검증 (90 ~ 100 GPa 범위, Type IV 금합금)
    assert "gold" in preset
    assert preset["gold"]["E"] == pytest.approx(95000.0)
    assert preset["gold"]["nu"] == pytest.approx(0.33)
    assert preset["gold"]["density"] == pytest.approx(1.55e-8)

    # 3. Zirconia 검증 (200 ~ 210 GPa 범위, Y-TZP 지르코니아)
    assert "zirconia" in preset
    assert preset["zirconia"]["E"] == pytest.approx(210000.0)
    assert preset["zirconia"]["nu"] == pytest.approx(0.30)
    assert preset["zirconia"]["density"] == pytest.approx(6.05e-9)


def test_analysis_config_abutment_and_crown_mesh_size():
    """AnalysisConfig에 mesh_size_abutment_mm, mesh_size_crown_mm 추가 및 직렬화/역직렬화 검증"""
    cfg = AnalysisConfig(
        mesh_size_far_mm=2.0,
        mesh_size_near_mm=0.4,
        element_type="C3D10",
        ccx_path=None,
        n_threads=4,
        mesh_size_abutment_mm=0.35,
        mesh_size_crown_mm=0.45,
    )
    assert cfg.mesh_size_abutment_mm == 0.35
    assert cfg.mesh_size_crown_mm == 0.45

    d = cfg.to_dict()
    assert d["mesh_size_abutment_mm"] == 0.35
    assert d["mesh_size_crown_mm"] == 0.45

    cfg2 = AnalysisConfig.from_dict(d)
    assert cfg2.mesh_size_abutment_mm == 0.35
    assert cfg2.mesh_size_crown_mm == 0.45

    # 기본값 None 호환성 검증
    cfg_default = AnalysisConfig.from_dict({
        "mesh_size_far_mm": 2.0,
        "mesh_size_near_mm": 0.4,
        "element_type": "C3D10",
    })
    assert cfg_default.mesh_size_abutment_mm is None
    assert cfg_default.mesh_size_crown_mm is None


def test_fea_geometry_and_build_with_abutment_and_crown():
    """FeaGeometry 및 build_fea_geometry에 abutment와 crown이 정상 통합되는지 검증"""
    # 원기둥 메쉬로 뼈, 임플란트, 어버트먼트, 크라운 모사
    # 뼈: z = -15 ~ 0
    bone = create_cylinder_trimesh(radius=10.0, height=15.0, z_offset=-15.0)
    # 임플란트: z = -10 ~ 0
    implant = create_cylinder_trimesh(radius=2.0, height=10.0, z_offset=-10.0)
    # 어버트먼트: z = 0 ~ 4
    abutment = create_cylinder_trimesh(radius=2.2, height=4.0, z_offset=0.0)
    # 크라운: z = 4 ~ 11
    crown = create_cylinder_trimesh(radius=4.5, height=7.0, z_offset=4.0)

    fea_geom = build_fea_geometry(
        cortical_mesh=bone,
        cancellous_mesh=None,
        implant_mesh=implant,
        bone_mesh=bone,
        abutment_mesh=abutment,
        crown_mesh=crown,
    )

    assert fea_geom.abutment is not None
    assert fea_geom.crown is not None
    assert fea_geom.abutment.is_watertight
    assert fea_geom.crown.is_watertight
    assert fea_geom.report["abutment_volume"] > 0.0
    assert fea_geom.report["crown_volume"] > 0.0
    assert fea_geom.report["abutment_watertight"] is True
    assert fea_geom.report["crown_watertight"] is True

    # export_stl 검증
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_stl(fea_geom, tmpdir)
        assert os.path.exists(paths["cortical"])
        assert os.path.exists(paths["implant"])
        assert "abutment" in paths and os.path.exists(paths["abutment"])
        assert "crown" in paths and os.path.exists(paths["crown"])


def test_write_inp_with_four_parts_and_ties():
    """CalculiX write_inp가 bone, implant, abutment, crown 4개 파트와 TIE 구속조건을 정상 생성하는지 검증"""
    # 1. 4개 파트 생성 (단순 사면체 메쉬)
    nodes_b = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,-1]], dtype=np.float64)
    elems_b = np.array([[0, 1, 2, 3]], dtype=np.int64)
    bone_part = MeshPart(name="bone", nodes=nodes_b, elements=elems_b, element_type="C3D4")

    nodes_i = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
    elems_i = np.array([[0, 1, 2, 3]], dtype=np.int64)
    implant_part = MeshPart(name="implant", nodes=nodes_i, elements=elems_i, element_type="C3D4")

    nodes_a = np.array([[0,0,1], [1,0,1], [0,1,1], [0,0,2]], dtype=np.float64)
    elems_a = np.array([[0, 1, 2, 3]], dtype=np.int64)
    abutment_part = MeshPart(name="abutment", nodes=nodes_a, elements=elems_a, element_type="C3D4")

    nodes_c = np.array([[0,0,2], [1,0,2], [0,1,2], [0,0,3]], dtype=np.float64)
    elems_c = np.array([[0, 1, 2, 3]], dtype=np.int64)
    crown_part = MeshPart(name="crown", nodes=nodes_c, elements=elems_c, element_type="C3D4")

    model = MeshModel(parts=[bone_part, implant_part, abutment_part, crown_part])
    assert model.total_nodes() == 16
    assert model.total_elements() == 4

    # 2. 재질 할당
    mat_bone = MaterialDef(name="mat_bone", E=13700.0, nu=0.3)
    mat_ti = MaterialDef(name="mat_titanium", E=110000.0, nu=0.34, density=4.5e-9)
    mat_gold = MaterialDef(name="mat_gold", E=95000.0, nu=0.33, density=1.55e-8)

    materials = {
        "bone": MaterialAssignment(element_material={"bone": np.array([0])}, materials=[mat_bone]),
        "implant": MaterialAssignment(element_material={"implant": np.array([0])}, materials=[mat_ti]),
        "abutment": MaterialAssignment(element_material={"abutment": np.array([0])}, materials=[mat_ti]),
        "crown": MaterialAssignment(element_material={"crown": np.array([0])}, materials=[mat_gold]),
    }

    # 3. 경계조건 및 TIE 정의
    fix_dict = {"name": "FIX_BASE", "part": "bone", "node_ids": np.array([3]), "dofs": (1, 3)}
    # Crown 최상단 절점(로컬 3번)에 하중 부여
    load_dict = {"name": "OCCLUSAL_FORCE", "part": "crown", "node_ids": np.array([3]), "force_N": (0.0, 0.0, -100.0)}
    bc_set = BoundaryConditionSet(fixed_node_sets=[fix_dict], loads=[load_dict])

    tie_bi = TieDef(
        name="TIE_BONE_IMPLANT",
        master_part="implant",
        slave_part="bone",
        master_faces=np.array([[0, 1]]),
        slave_nodes=np.array([0, 1, 2]),
        position_tolerance_mm=0.3
    )
    tie_ia = TieDef(
        name="TIE_IMPLANT_ABUTMENT",
        master_part="implant",
        slave_part="abutment",
        master_faces=np.array([[0, 2]]),
        slave_nodes=np.array([0, 1, 2]),
        position_tolerance_mm=0.3
    )
    tie_ac = TieDef(
        name="TIE_ABUTMENT_CROWN",
        master_part="abutment",
        slave_part="crown",
        master_faces=np.array([[0, 2]]),
        slave_nodes=np.array([0, 1, 2]),
        position_tolerance_mm=0.3
    )
    ties = [tie_bi, tie_ia, tie_ac]

    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path = os.path.join(tmpdir, "test_job.inp")
        write_inp(path=inp_path, model=model, materials=materials, bc=bc_set, ties=ties)
        assert os.path.exists(inp_path)

        with open(inp_path, "r", encoding="ascii") as f:
            content = f.read()

        # 파트 및 키워드 검증
        assert "*ELEMENT, TYPE=C3D4, ELSET=EALL_BONE" in content
        assert "*ELEMENT, TYPE=C3D4, ELSET=EALL_IMPLANT" in content
        assert "*ELEMENT, TYPE=C3D4, ELSET=EALL_ABUTMENT" in content
        assert "*ELEMENT, TYPE=C3D4, ELSET=EALL_CROWN" in content
        assert "*TIE, NAME=TIE_BONE_IMPLANT" in content
        assert "*TIE, NAME=TIE_IMPLANT_ABUTMENT" in content
        assert "*TIE, NAME=TIE_ABUTMENT_CROWN" in content
        assert "*CLOAD" in content
        assert "95000.000000, 0.330000" in content  # Gold 탄성계수
        assert "110000.000000, 0.340000" in content  # Titanium 탄성계수
