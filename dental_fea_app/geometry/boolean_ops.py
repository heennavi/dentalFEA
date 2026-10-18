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
    """FEA 입력용 3개 솔리드 메쉬 및 기하 분석 보고서"""
    cortical: trimesh.Trimesh
    cancellous: trimesh.Trimesh
    implant: trimesh.Trimesh
    report: Dict[str, Any]


def ensure_watertight(mesh: trimesh.Trimesh, name: str = "mesh") -> trimesh.Trimesh:
    """
    메쉬가 watertight(닫힌 매니폴드) 상태인지 확인하고,
    아닐 경우 구멍 메우기 및 법선 정렬 수리를 시도합니다.
    수리 실패 시 명확한 ValueError를 발생시킵니다.
    """
    if mesh is None or len(mesh.faces) == 0:
        raise ValueError(f"'{name}' 메쉬가 비어 있거나 존재하지 않습니다.")

    m = mesh.copy()
    if m.is_watertight:
        return m

    # 1. 기하 정리 및 중복/퇴화 요소 정돈
    m.process(validate=True)

    # 2. 구멍 메우기 및 와인딩/법선 수리
    trimesh.repair.fix_winding(m)
    trimesh.repair.fill_holes(m, use_fan=True)
    trimesh.repair.fix_normals(m)
    trimesh.repair.fix_inversion(m)
    m.process(validate=True)

    if not m.is_watertight:
        # 3. 2차 수리 시도: stitch 후 fill_holes 재시도
        try:
            trimesh.repair.stitch(m)
            trimesh.repair.fill_holes(m, use_fan=True)
            trimesh.repair.fix_normals(m)
            m.process(validate=True)
        except Exception:
            pass

    if not m.is_watertight:
        raise ValueError(
            f"'{name}' 메쉬가 watertight(닫힌 매니폴드) 상태가 아니며 구멍 메우기 및 기하 수리에 실패했습니다."
        )

    return m


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

    if len(retained) == 1:
        return retained[0], num_pieces
    else:
        merged = trimesh.util.concatenate(retained)
        return merged, num_pieces


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
    margin_mm: Optional[float] = None
) -> FeaGeometry:
    """
    피질골, 수질골 메쉬에서 임플란트 식립 공간을 절제하고 중복 부피를 정리하여
    FEA 입력용 3개 솔리드(피질골, 수질골, 임플란트)를 구축합니다.

    [주의]
    피질골과 수질골 사이 경계면은 Phase 3에서 노드를 공유(conformal)해야 합니다.
    이 함수에서는 두 솔리드가 겹치지 않고 경계가 일치하도록 만드는 데까지만 처리하며,
    메쉬 공유(Conformal meshing) 처리는 Phase 3(Gmsh)에서 수행합니다.

    Parameters
    ----------
    cortical_mesh : trimesh.Trimesh
        피질골 표면 메쉬 (LPS mm)
    cancellous_mesh : trimesh.Trimesh
        수질골 표면 메쉬 (LPS mm)
    implant_mesh : trimesh.Trimesh
        식립된 위치의 임플란트 표면 메쉬 (LPS mm)
    undersize_mm : float, optional
        언더사이즈 드릴링 크기 (기본값 0.0 = 정확한 맞닿음)
    margin_mm : float, optional
        추가 안전 여유 마진 (기본 None)

    Returns
    -------
    FeaGeometry
        정리된 3개 솔리드 메쉬(cortical, cancellous, implant) 및 기하 분석 보고서(report)
    """
    # 0. 입력 메쉬 watertight 검사 및 수리
    c_mesh = ensure_watertight(cortical_mesh, name="cortical")
    cn_mesh = ensure_watertight(cancellous_mesh, name="cancellous")
    i_mesh = ensure_watertight(implant_mesh, name="implant")

    initial_cortical_vol = abs(float(c_mesh.volume))
    initial_cancellous_vol = abs(float(cn_mesh.volume))
    initial_bone_vol = initial_cortical_vol + initial_cancellous_vol

    # 1. 드릴링 공간용 메쉬 생성
    implant_space = create_implant_drilling_space(
        i_mesh,
        undersize_mm=undersize_mm,
        margin_mm=margin_mm
    )
    implant_space = ensure_watertight(implant_space, name="implant_space")

    # 2. 피질골' = 피질골 - 임플란트공간
    try:
        cortical_diff = trimesh.boolean.difference(
            [c_mesh, implant_space],
            engine="manifold"
        )
    except Exception as e:
        warnings.warn(f"피질골 Boolean 차집합 연산 중 경고 발생: {e}", UserWarning)
        cortical_diff = c_mesh.copy()

    # 3. 수질골' = 수질골 - 임플란트공간
    try:
        cancellous_diff = trimesh.boolean.difference(
            [cn_mesh, implant_space],
            engine="manifold"
        )
    except Exception as e:
        warnings.warn(f"수질골 Boolean 차집합 연산 중 경고 발생: {e}", UserWarning)
        cancellous_diff = cn_mesh.copy()

    # 4. 피질골'와 수질골'의 중복 부피 제거: 수질골' = 수질골' - 피질골'
    # (피질골과 수질골 사이 경계면 일치화)
    try:
        cancellous_clean = trimesh.boolean.difference(
            [cancellous_diff, cortical_diff],
            engine="manifold"
        )
    except Exception as e:
        warnings.warn(f"수질골-피질골 인터페이스 정리 중 경고 발생: {e}", UserWarning)
        cancellous_clean = cancellous_diff.copy()

    # 5. 작은 파편 조각 정리 및 분리 경고
    cortical_final, c_pieces = filter_small_components(cortical_diff, min_volume_ratio=0.01, name="cortical")
    cancellous_final, cn_pieces = filter_small_components(cancellous_clean, min_volume_ratio=0.01, name="cancellous")
    implant_final, i_pieces = filter_small_components(i_mesh, min_volume_ratio=0.01, name="implant")

    # 6. 각 결과 watertight 검증 및 부피 계산
    cortical_final.fix_normals()
    cancellous_final.fix_normals()
    implant_final.fix_normals()

    c_vol = abs(float(cortical_final.volume))
    cn_vol = abs(float(cancellous_final.volume))
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
    cn_contact_area = calculate_contact_area(implant_final, cancellous_final, threshold_mm=0.05)

    report = {
        "cortical_volume": c_vol,
        "cancellous_volume": cn_vol,
        "implant_volume": i_vol,
        "initial_cortical_volume": initial_cortical_vol,
        "initial_cancellous_volume": initial_cancellous_vol,
        "initial_bone_volume": initial_bone_vol,
        "remaining_bone_volume": final_bone_vol,
        "removed_bone_volume": removed_bone_vol,
        "cortical_watertight": bool(cortical_final.is_watertight),
        "cancellous_watertight": bool(cancellous_final.is_watertight),
        "implant_watertight": bool(implant_final.is_watertight),
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
        report=report
    )


def export_stl(geometry: FeaGeometry, out_dir: str) -> Dict[str, str]:
    """
    FeaGeometry의 3개 솔리드(cortical, cancellous, implant)를 STL 파일로 저장합니다.

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

    return paths
