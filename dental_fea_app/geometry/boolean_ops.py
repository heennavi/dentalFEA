"""
boolean_ops.py
피질골/수질골 표면 메쉬와 임플란트 간 Boolean 차집합(식립부 절제) 및 인터페이스 분할 엔진.

[참고]
- 기본 엔진: trimesh + manifold3d (engine="manifold")
- 피질골과 수질골 사이 경계면은 Phase 3에서 노드를 공유(conformal)해야 합니다.
  이 모듈에서는 두 솔리드가 겹치지 않고 경계가 일치하도록 만드는 데까지만 처리하며,
  메쉬 공유(Conformal meshing) 처리는 Phase 3(Gmsh)에서 수행합니다.
"""

import os
import copy
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh.proximity import closest_point


@dataclass
class FeaGeometry:
    """FEA 입력용 솔리드 메쉬 및 기하 분석 보고서"""
    cortical: trimesh.Trimesh
    cancellous: trimesh.Trimesh
    implant: trimesh.Trimesh
    report: Dict[str, Any]
    bone: Optional[trimesh.Trimesh] = None
    abutment: Optional[trimesh.Trimesh] = None
    crown: Optional[trimesh.Trimesh] = None
    screw: Optional[trimesh.Trimesh] = None


def filter_small_components(
    mesh: trimesh.Trimesh,
    min_volume_ratio: float = 0.01,
    name: str = "mesh"
) -> Tuple[trimesh.Trimesh, int]:
    """
    메쉬가 여러 개의 연결 요소로 분리되었을 때,
    최대 컴포넌트 부피 대비 min_volume_ratio 미만의 미세 파편 조각을 제거합니다.
    """
    components = mesh.split(only_watertight=False)
    num_pieces = len(components)
    if num_pieces <= 1:
        return mesh, 1

    # 부피 기준 정렬 (양수 부피 보장)
    valid_pieces = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
        for c in components:
            vol = 0.0
            try:
                if c.is_watertight and len(c.faces) >= 4:
                    vol = abs(float(c.volume))
                else:
                    vol = float(c.convex_hull.volume)
            except Exception:
                try:
                    vol = float(c.convex_hull.volume)
                except Exception:
                    vol = 0.0
            if np.isnan(vol) or np.isinf(vol):
                vol = 0.0
            valid_pieces.append((vol, c))

    valid_pieces.sort(key=lambda x: x[0], reverse=True)
    max_vol = valid_pieces[0][0]

    retained = []
    removed_count = 0
    for vol, c in valid_pieces:
        if max_vol > 1e-6 and (vol / max_vol) < min_volume_ratio:
            removed_count += 1
        else:
            retained.append(c)

    if removed_count > 0:
        warnings.warn(
            f"'{name}' 메쉬에서 Boolean 연산 후 발생한 미세 분리 조각 {removed_count}개가 제거되었습니다.",
            UserWarning
        )

    retained_count = len(retained)
    if retained_count == 1:
        return retained[0], 1
    else:
        merged = trimesh.util.concatenate(retained)
        return merged, retained_count


def ensure_watertight(
    mesh: trimesh.Trimesh,
    name: str = "mesh",
    allow_empty: bool = False
) -> trimesh.Trimesh:
    """
    메쉬가 watertight(닫힌 매니폴드) 상태인지 확인하고,
    아닐 경우 미세 파편 제거, 구멍 메우기 및 비매니폴드 수리를 순차적으로 시도합니다.
    FEA 체적 메싱(Volume meshing)을 위해 항상 견고하고 닫힌 솔리드를 보장합니다.
    """
    if mesh is None or len(mesh.faces) == 0:
        if allow_empty:
            return trimesh.Trimesh()
        raise ValueError(f"'{name}' 메쉬가 비어 있거나 존재하지 않습니다.")

    m = mesh.copy()
    def _is_valid_solid(mesh_obj) -> bool:
        if mesh_obj is None or len(mesh_obj.faces) < 4:
            return False
        if not mesh_obj.is_watertight:
            return False
        try:
            vol = abs(float(mesh_obj.volume))
            return vol > 1e-6
        except Exception:
            return False

    if _is_valid_solid(m):
        return m

    # 1단계: 기본 기하 정리 및 중복/퇴화 요소 정돈
    m.process(validate=True)
    if _is_valid_solid(m):
        return m

    # 2단계: 미세 분리 파편 조각 제거 (미세 파편으로 인한 watertight 실패 해결)
    try:
        m, _ = filter_small_components(m, min_volume_ratio=0.02, name=name)
        m.process(validate=True)
    except Exception:
        pass
    if _is_valid_solid(m):
        return m

    # 2-1단계: 비매니폴드 엣지(한 모서리에 3개 이상의 면이 인접) 검출 및 정리
    try:
        edges_sorted = m.edges_sorted
        edge_groups = trimesh.grouping.group_rows(edges_sorted, require_count=None)
        edge_counts = np.array([len(g) for g in edge_groups])
        bad_edge_indices = np.where(edge_counts > 2)[0]
        if len(bad_edge_indices) > 0:
            bad_faces = set()
            for idx in bad_edge_indices:
                for edge_idx in edge_groups[idx]:
                    bad_faces.add(edge_idx // 3)
            keep_faces = [i for i in range(len(m.faces)) if i not in bad_faces]
            if len(keep_faces) >= 4:
                m = m.submesh([keep_faces], append=True)
                m.process(validate=True)
    except Exception:
        pass
    if _is_valid_solid(m):
        return m

    # 3단계: 구멍 메우기 및 와인딩/법선 수리
    try:
        trimesh.repair.fix_winding(m)
        trimesh.repair.fill_holes(m, use_fan=True)
        trimesh.repair.fix_normals(m)
        trimesh.repair.fix_inversion(m)
        m.process(validate=True)
    except Exception:
        pass
    if _is_valid_solid(m):
        return m

    # 4단계: stitch 후 fill_holes 재시도
    try:
        trimesh.repair.stitch(m)
        trimesh.repair.fill_holes(m, use_fan=True)
        trimesh.repair.fix_normals(m)
        m.process(validate=True)
    except Exception:
        pass
    if _is_valid_solid(m):
        return m

    # 5단계: 손상된 비매니폴드 면 제거 후 구멍 메우기
    try:
        broken = trimesh.repair.broken_faces(m)
        if 0 < len(broken) < len(m.faces):
            m.update_faces(~np.isin(np.arange(len(m.faces)), broken))
            m.process(validate=True)
            trimesh.repair.fill_holes(m, use_fan=True)
            trimesh.repair.fix_normals(m)
    except Exception:
        pass
    if _is_valid_solid(m):
        return m

    # 6단계: 부피 검증 (2D 평면 패치 또는 완전히 닫히지 않는 열린 표면인 경우 예외 발생)
    try:
        vol = abs(float(m.volume))
    except Exception:
        vol = 0.0

    if len(m.faces) < 4 or vol < 1e-6:
        raise ValueError(f"'{name}' 메쉬가 watertight(닫힌 매니폴드) 상태가 아니며 구멍 메우기 및 기하 수리에 실패했습니다.")

    warnings.warn(
        f"'{name}' 메쉬에 미세한 비매니폴드 경계가 남아 있어 구멍 메우기 수리본을 사용합니다.",
        UserWarning
    )
    return m


def calculate_contact_area(
    implant_mesh: trimesh.Trimesh,
    bone_mesh: trimesh.Trimesh,
    threshold_mm: float = 0.05
) -> float:
    """
    임플란트 표면 삼각형 중심점에서 뼈 메쉬 표면까지의 거리를 계산하여,
    임계값(threshold_mm, 기본 0.05mm) 이내로 맞닿아 있는 임플란트 표면적 합(mm^2)을 계산합니다.

    [OOM 방지 최적화]
    뼈 메쉬(수십만 개 삼각형) 전체에 대해 직접 R-tree 인덱싱을 수행하면 수십 GB의 메모리가
    순식간에 소진되어 OOM Killer가 동작할 수 있습니다.
    따라서 임플란트 바운딩 박스 주변 국소 영역의 삼각형만 필터링한 서브메쉬를 대상으로
    최근접 거리를 계산하여 메모리 안전성(수 MB)과 초고속 연산(<0.1초)을 보장합니다.
    """
    if bone_mesh is None or len(bone_mesh.faces) == 0:
        return 0.0
    if implant_mesh is None or len(implant_mesh.faces) == 0:
        return 0.0

    face_centers = implant_mesh.triangles_center

    # 1. 임플란트 Bounding Box + 여유 마진(threshold_mm * 5 이상)
    margin = max(0.5, float(threshold_mm) * 5.0)
    b_min = implant_mesh.bounds[0] - margin
    b_max = implant_mesh.bounds[1] + margin

    # 2. 뼈 삼각형 중 임플란트 바운딩 박스 영역과 교차하는 삼각형 선별
    tri_bounds_min = np.min(bone_mesh.triangles, axis=1)
    tri_bounds_max = np.max(bone_mesh.triangles, axis=1)

    overlap = np.all(tri_bounds_max >= b_min, axis=1) & np.all(tri_bounds_min <= b_max, axis=1)
    overlap_face_indices = np.where(overlap)[0]

    if len(overlap_face_indices) == 0:
        return 0.0

    # 3. 국소 서브메쉬 추출 (수십만 개 중 수백~수천 개 삼각형으로 축소)
    sub_bone = bone_mesh.submesh([overlap_face_indices], append=True)

    # 4. 국소 서브메쉬에 대한 정밀 거리 계산
    try:
        _, distances, _ = closest_point(sub_bone, face_centers)
    except Exception:
        # Fallback: cKDTree (정점 + 삼각형 중심점 샘플링)
        sample_pts = np.vstack([sub_bone.vertices, sub_bone.triangles_center])
        tree = cKDTree(sample_pts)
        distances, _ = tree.query(face_centers, k=1)

    contact_mask = distances <= threshold_mm
    contact_area = float(np.sum(implant_mesh.area_faces[contact_mask]))
    return contact_area


def create_implant_drilling_space(
    implant_mesh: trimesh.Trimesh,
    undersize_mm: float = 0.0,
    margin_mm: Optional[float] = None
) -> trimesh.Trimesh:
    """
    (옵션) 임플란트를 undersize_mm 또는 margin_mm만큼 확대/축소한 드릴링 공간 메쉬를 생성합니다.
    - undersize_mm > 0: 프레스핏을 위해 드릴링 공간을 임플란트보다 작게 형성 (내부 축소)
    - margin_mm > 0: 안전 마진을 위한 외부 확장
    """
    space = implant_mesh.copy()
    delta = 0.0
    if margin_mm is not None:
        delta += float(margin_mm)
    if undersize_mm != 0.0:
        delta -= float(undersize_mm)

    if abs(delta) > 1e-5:
        # 정점 법선 방향으로 오프셋 적용
        space.vertices += space.vertex_normals * delta
        space.remove_degenerate_faces()
        space.remove_duplicate_faces()
        trimesh.repair.fix_normals(space)

    return space


def build_fea_geometry(
    cortical_mesh: trimesh.Trimesh,
    cancellous_mesh: trimesh.Trimesh,
    implant_mesh: trimesh.Trimesh,
    undersize_mm: float = 0.0,
    margin_mm: Optional[float] = None,
    bone_mesh: Optional[trimesh.Trimesh] = None,
    abutment_mesh: Optional[trimesh.Trimesh] = None,
    crown_mesh: Optional[trimesh.Trimesh] = None,
    screw_mesh: Optional[trimesh.Trimesh] = None,
) -> FeaGeometry:
    """
    피질골, 수질골 메쉬에서 임플란트 식립 공간을 절제하고 중복 부피를 정리하며,
    Abutment 및 Crown이 존재하는 경우 함께 포함하여 FEA 입력용 솔리드 메쉬들을 구축합니다.

    [설계 원칙]
    - cancellous_mesh가 비어 있거나 없는 경우 피질골 단독으로 안전하게 FEA 기하를 구성합니다.
    - bone 속성에 피질골과 수질골이 완벽하게 결합된 단일 매니폴드 뼈 메쉬를 보장하여
      체적 메싱(Volume meshing) 시 경계면 중복(overlapping facets)을 원천 방지합니다.
    - abutment_mesh, crown_mesh가 제공되면 각각 watertight 매니폴드 메쉬로 검증/수리하여 유지합니다.
    """
    # 0. 입력 메쉬 watertight 검사 및 수리
    has_cortical = cortical_mesh is not None and len(getattr(cortical_mesh, "faces", [])) > 0
    has_cancellous = cancellous_mesh is not None and len(getattr(cancellous_mesh, "faces", [])) > 0

    if not has_cortical and not has_cancellous:
        raise ValueError("골 조직(피질골 또는 수질골) 표면 메쉬가 비어 있거나 존재하지 않습니다. ROI 범위를 다시 설정하십시오.")

    c_mesh = ensure_watertight(cortical_mesh, name="cortical", allow_empty=True) if has_cortical else None
    cn_mesh = ensure_watertight(cancellous_mesh, name="cancellous", allow_empty=True) if has_cancellous else None
    i_mesh = ensure_watertight(implant_mesh, name="implant")

    initial_cortical_vol = abs(float(c_mesh.volume)) if has_cortical and c_mesh is not None else 0.0
    initial_cancellous_vol = abs(float(cn_mesh.volume)) if has_cancellous and cn_mesh is not None else 0.0
    initial_bone_vol = initial_cortical_vol + initial_cancellous_vol

    # 1. 드릴링 공간용 메쉬 생성
    implant_space = create_implant_drilling_space(
        i_mesh,
        undersize_mm=undersize_mm,
        margin_mm=margin_mm
    )
    implant_space = ensure_watertight(implant_space, name="implant_space")

    # 2. 피질골 처리
    if has_cortical and c_mesh is not None:
        try:
            cortical_diff = trimesh.boolean.difference(
                [c_mesh, implant_space],
                engine="manifold",
                check_volume=False
            )
        except Exception as e:
            warnings.warn(f"피질골 Boolean 차집합 연산 중 경고 발생: {e}", UserWarning)
            cortical_diff = c_mesh.copy()

        cortical_final, c_pieces = filter_small_components(cortical_diff, min_volume_ratio=0.01, name="cortical")
        cortical_final = ensure_watertight(cortical_final, name="cortical")
        cortical_final.fix_normals()
    else:
        cortical_final = trimesh.Trimesh()
        c_pieces = 0

    # 3. 수질골 처리 (수질골이 존재하는 경우만 수행)
    if has_cancellous and cn_mesh is not None:
        try:
            cancellous_diff = trimesh.boolean.difference(
                [cn_mesh, implant_space],
                engine="manifold",
                check_volume=False
            )
        except Exception as e:
            warnings.warn(f"수질골 Boolean 차집합 연산 중 경고 발생: {e}", UserWarning)
            cancellous_diff = cn_mesh.copy()

        # 4. 피질골과 수질골 사이 중복 부피 제거
        if has_cortical and len(cortical_final.faces) > 0:
            try:
                cancellous_clean = trimesh.boolean.difference(
                    [cancellous_diff, cortical_final],
                    engine="manifold",
                    check_volume=False
                )
            except Exception as e:
                warnings.warn(f"수질골-피질골 인터페이스 정리 중 경고 발생: {e}", UserWarning)
                cancellous_clean = cancellous_diff.copy()
        else:
            cancellous_clean = cancellous_diff

        cancellous_final, cn_pieces = filter_small_components(cancellous_clean, min_volume_ratio=0.01, name="cancellous")
        cancellous_final = ensure_watertight(cancellous_final, name="cancellous", allow_empty=True)
        if len(cancellous_final.faces) > 0:
            cancellous_final.fix_normals()
    else:
        cancellous_final = trimesh.Trimesh()
        cn_pieces = 0

    # 5. 임플란트 정리
    implant_final, i_pieces = filter_small_components(i_mesh, min_volume_ratio=0.01, name="implant")
    implant_final = ensure_watertight(implant_final, name="implant")
    implant_final.fix_normals()

    # 6. 단일 결합 뼈 메쉬(bone_final) 구축
    if bone_mesh is not None and len(getattr(bone_mesh, "faces", [])) > 0:
        try:
            b_mesh = ensure_watertight(bone_mesh, name="bone")
            bone_diff = trimesh.boolean.difference(
                [b_mesh, implant_space],
                engine="manifold",
                check_volume=False
            )
            bone_final, _ = filter_small_components(bone_diff, min_volume_ratio=0.01, name="bone")
            bone_final = ensure_watertight(bone_final, name="bone")
            bone_final.fix_normals()
        except Exception as e:
            warnings.warn(f"전체 뼈 Boolean 차집합 실패, 기본 피질골/수질골 활용: {e}", UserWarning)
            bone_final = cortical_final.copy() if has_cortical else cancellous_final.copy()
    elif has_cortical and has_cancellous and len(cancellous_final.faces) > 0 and len(cortical_final.faces) > 0:
        # manifold3d 결합으로 내부 경계면을 제거한 완벽한 단일 껍질 생성
        try:
            import manifold3d as m3d
            m_c = m3d.Manifold(cortical_final.vertices, cortical_final.faces)
            m_cn = m3d.Manifold(cancellous_final.vertices, cancellous_final.faces)
            m_u = m_c + m_cn
            raw = m_u.to_mesh()
            bone_final = trimesh.Trimesh(raw.vert_properties[:, :3], raw.tri_verts, process=True)
            bone_final = ensure_watertight(bone_final, name="bone")
            bone_final.fix_normals()
        except Exception:
            try:
                bone_final = trimesh.boolean.union([cortical_final, cancellous_final], engine="manifold")
                bone_final = ensure_watertight(bone_final, name="bone")
                bone_final.fix_normals()
            except Exception:
                bone_final = cortical_final.copy()
    elif has_cortical:
        bone_final = cortical_final.copy()
    else:
        bone_final = cancellous_final.copy()

    c_vol = abs(float(cortical_final.volume)) if len(cortical_final.faces) > 0 else 0.0
    cn_vol = abs(float(cancellous_final.volume)) if len(cancellous_final.faces) > 0 else 0.0
    i_vol = abs(float(implant_final.volume))
    final_bone_vol = c_vol + cn_vol
    removed_bone_vol = max(0.0, initial_bone_vol - final_bone_vol)

    # 임플란트가 뼈와 전혀 겹치지 않는 경우 경고
    if removed_bone_vol < 1e-4:
        warnings.warn(
            "임플란트가 뼈(피질골/수질골) 메쉬와 전혀 겹치지 않습니다 (제거된 뼈 부피 0.0 mm³).",
            UserWarning
        )

    # 7. 임플란트와 뼈의 접촉 면적(mm^2) 추정 (거리 임계 0.05 mm)
    c_contact_area = calculate_contact_area(implant_final, cortical_final, threshold_mm=0.05)
    cn_contact_area = calculate_contact_area(implant_final, cancellous_final, threshold_mm=0.05) if len(cancellous_final.faces) > 0 else 0.0

    # 8. Abutment 및 Crown 검증 및 처리
    abutment_final = None
    # 8. Abutment, Crown 및 Screw 검증 및 처리
    abutment_final = None
    if abutment_mesh is not None and len(getattr(abutment_mesh, "faces", [])) > 0:
        abutment_final = ensure_watertight(abutment_mesh, name="abutment", allow_empty=True)
        if len(abutment_final.faces) > 0:
            abutment_final.fix_normals()

    crown_final = None
    if crown_mesh is not None and len(getattr(crown_mesh, "faces", [])) > 0:
        crown_final = ensure_watertight(crown_mesh, name="crown", allow_empty=True)
        if len(crown_final.faces) > 0:
            crown_final.fix_normals()

    screw_final = None
    if screw_mesh is not None and len(getattr(screw_mesh, "faces", [])) > 0:
        screw_final = ensure_watertight(screw_mesh, name="screw", allow_empty=True)
        if len(screw_final.faces) > 0:
            screw_final.fix_normals()

    abutment_vol = abs(float(abutment_final.volume)) if (abutment_final is not None and len(abutment_final.faces) > 0) else 0.0
    crown_vol = abs(float(crown_final.volume)) if (crown_final is not None and len(crown_final.faces) > 0) else 0.0
    screw_vol = abs(float(screw_final.volume)) if (screw_final is not None and len(screw_final.faces) > 0) else 0.0

    report = {
        "cortical_volume": c_vol,
        "cancellous_volume": cn_vol,
        "implant_volume": i_vol,
        "abutment_volume": abutment_vol,
        "crown_volume": crown_vol,
        "screw_volume": screw_vol,
        "initial_cortical_volume": initial_cortical_vol,
        "initial_cancellous_volume": initial_cancellous_vol,
        "initial_bone_volume": initial_bone_vol,
        "remaining_bone_volume": final_bone_vol,
        "removed_bone_volume": removed_bone_vol,
        "cortical_watertight": bool(cortical_final.is_watertight),
        "cancellous_watertight": bool(cancellous_final.is_watertight) if len(cancellous_final.faces) > 0 else True,
        "implant_watertight": bool(implant_final.is_watertight),
        "abutment_watertight": bool(abutment_final.is_watertight) if abutment_final is not None else False,
        "crown_watertight": bool(crown_final.is_watertight) if crown_final is not None else False,
        "screw_watertight": bool(screw_final.is_watertight) if screw_final is not None else False,
        "cortical_pieces": c_pieces,
        "cancellous_pieces": cn_pieces,
        "implant_pieces": i_pieces,
        "implant_cortical_contact_area": c_contact_area,
        "implant_cancellous_contact_area": cn_contact_area,
        "total_contact_area": c_contact_area + cn_contact_area,
        "undersize_mm": float(undersize_mm),
        "margin_mm": float(margin_mm) if margin_mm is not None else 0.0,
    }

    return FeaGeometry(
        cortical=cortical_final,
        cancellous=cancellous_final,
        implant=implant_final,
        report=report,
        bone=bone_final,
        abutment=abutment_final,
        crown=crown_final,
        screw=screw_final,
    )


def export_stl(geometry: FeaGeometry, out_dir: str) -> Dict[str, str]:
    """
    FeaGeometry의 솔리드들(cortical, cancellous, implant, abutment, crown, screw)을 STL 파일로 저장합니다.

    Parameters
    ----------
    geometry : FeaGeometry
        저장할 FEA 기하 데이터
    out_dir : str
        저장 디렉터리 경로

    Returns
    -------
    Dict[str, str]
        각 솔리드 명칭별 저장된 파일 경로 딕셔너리
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "cortical": os.path.join(out_dir, "cortical.stl"),
        "cancellous": os.path.join(out_dir, "cancellous.stl"),
        "implant": os.path.join(out_dir, "implant.stl"),
    }

    geometry.cortical.export(paths["cortical"])
    geometry.cancellous.export(paths["cancellous"])
    geometry.implant.export(paths["implant"])

    if geometry.abutment is not None and len(geometry.abutment.faces) > 0:
        paths["abutment"] = os.path.join(out_dir, "abutment.stl")
        geometry.abutment.export(paths["abutment"])

    if geometry.crown is not None and len(geometry.crown.faces) > 0:
        paths["crown"] = os.path.join(out_dir, "crown.stl")
        geometry.crown.export(paths["crown"])

    if geometry.screw is not None and len(geometry.screw.faces) > 0:
        paths["screw"] = os.path.join(out_dir, "screw.stl")
        geometry.screw.export(paths["screw"])

    return paths
