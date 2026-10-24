"""
peri_implant.py
Peri-implant bone stress statistics and peak stress localization for dental FEA.

Provides headless post-processing functions to evaluate mechanical stimuli
around the implant-bone interface, including cortical/cancellous bone stratification
and peak stress identification.

[Units]
- Length / Distance: mm
- Stress: MPa (N/mm^2)
- Displacement: mm
"""

from typing import Optional, Dict, Any, Sequence, Union
import numpy as np
from scipy.spatial import cKDTree

from fea.models import MeshModel, MeshPart, MaterialAssignment, TieDef
from fea.results import ResultSet


def extract_bone_interface_nodes(
    tie: TieDef,
    model: MeshModel,
    bone_part_name: Optional[str] = None,
) -> np.ndarray:
    """
    Extract bone-side local node indices participating in an implant-bone tie constraint.

    Inspects master and slave part assignments in the TieDef to distinguish the bone part.
    If the bone part is the slave, returns slave_nodes directly.
    If the bone part is the master, reconstructs unique node indices from master_faces.

    Args:
        tie: TieDef constraint connecting implant and bone.
        model: MeshModel containing assembly parts.
        bone_part_name: Optional explicit bone part identifier. If None, infers from part names.

    Returns:
        np.ndarray: 1D int64 array of bone part local 0-based node indices.
    """
    if bone_part_name is None:
        # master와 slave 이름 중 "bone" 또는 "mandible"을 포함하는 쪽을 뼈로 추정
        if "bone" in tie.slave_part.lower() or "mandible" in tie.slave_part.lower():
            bone_part_name = tie.slave_part
        elif "bone" in tie.master_part.lower() or "mandible" in tie.master_part.lower():
            bone_part_name = tie.master_part
        else:
            # 기본값으로 slave_part 채택
            bone_part_name = tie.slave_part

    part = model.get_part(bone_part_name)
    if part is None:
        return np.array([], dtype=np.int64)

    offset = model.global_node_offset(part.name)
    n_nodes = len(part.nodes)

    if tie.slave_part == bone_part_name:
        s_nodes = np.asarray(tie.slave_nodes, dtype=np.int64)
        if len(s_nodes) == 0:
            return np.array([], dtype=np.int64)
        # 전역 번호로 저장되어 있을 경우 로컬로 변환
        if np.all(s_nodes >= offset) and np.all(s_nodes < offset + n_nodes):
            s_nodes = s_nodes - offset
        return np.unique(s_nodes[ (s_nodes >= 0) & (s_nodes < n_nodes) ])

    elif tie.master_part == bone_part_name:
        if len(tie.master_faces) == 0:
            return np.array([], dtype=np.int64)

        # CalculiX 1-based tetrahedron face connectivity (0-based local indices)
        c3d10_face_nodes_map = {
            1: [0, 1, 2, 4, 5, 6],
            2: [0, 3, 1, 7, 9, 4],
            3: [1, 3, 2, 8, 9, 5],
            4: [2, 3, 0, 6, 9, 7],
        }
        c3d4_face_nodes_map = {
            1: [0, 1, 2],
            2: [0, 3, 1],
            3: [1, 3, 2],
            4: [2, 3, 0],
        }

        elems = np.asarray(part.elements, dtype=np.int64)
        if len(elems) > 0 and np.all(elems >= offset) and np.all(elems < offset + n_nodes):
            elems = elems - offset

        is_c3d10 = (elems.shape[1] == 10)
        face_map = c3d10_face_nodes_map if is_c3d10 else c3d4_face_nodes_map

        master_node_set = set()
        for elem_idx, fid in tie.master_faces:
            if 0 <= elem_idx < len(elems):
                el = elems[elem_idx]
                local_f_nids = face_map.get(int(fid), [0, 1, 2])
                for ln in local_f_nids:
                    if ln < len(el):
                        master_node_set.add(int(el[ln]))

        arr = np.array(sorted(list(master_node_set)), dtype=np.int64)
        return arr[(arr >= 0) & (arr < n_nodes)]

    return np.array([], dtype=np.int64)


def peri_implant_bone_stats(
    rs: ResultSet,
    model: MeshModel,
    interface_nodes_bone: np.ndarray,
    radius_mm: float,
    materials: Optional[MaterialAssignment] = None,
    cortical_name: str = "CORTICAL",
    bone_part_name: Optional[str] = None,
) -> dict:
    """
    Compute stress statistics for bone nodes located within radius_mm from the implant interface.

    Selects bone nodes within radius_mm from interface_nodes_bone using scipy.spatial.cKDTree.
    Computes von Mises stress (max, mean, 95th, 99th percentile), max principal stress (peak tensile),
    and min principal stress (peak compressive).

    Bone stratification rule:
        If materials is provided, nodes are divided into cortical and cancellous bone.
        Classification rule: If a node belongs to at least one element assigned with the cortical material,
        it is classified as cortical bone; all remaining nodes are classified as cancellous bone.

    Args:
        rs: Computed ResultSet with nodal stress fields.
        model: MeshModel containing bone and implant parts.
        interface_nodes_bone: 1D array of bone interface node indices (local or global).
        radius_mm: Proximity search radius from the interface in mm.
        materials: Optional MaterialAssignment for cortical/cancellous bone stratification.
        cortical_name: Substring identifying cortical bone in MaterialDef.name (default "CORTICAL").
        bone_part_name: Optional explicit bone part identifier. If None, auto-detected.

    Returns:
        dict: Summary statistics with float/int values and optional 'cortical'/'cancellous' sub-dicts.
              If no nodes are found, returns 0 count and a Korean warning message.
    """
    # 1. 뼈 파트 식별
    bone_part = None
    if bone_part_name is not None:
        bone_part = model.get_part(bone_part_name)
    else:
        for p in model.parts:
            if "bone" in p.name.lower() or "mandible" in p.name.lower():
                bone_part = p
                break
        if bone_part is None and len(model.parts) > 0:
            bone_part = model.parts[0]

    if bone_part is None or len(bone_part.nodes) == 0:
        return {
            "num_nodes": 0,
            "radius_mm": float(radius_mm),
            "warning": "유효한 뼈 파트를 찾을 수 없습니다.",
            "max_von_mises": None,
            "mean_von_mises": None,
            "p95_von_mises": None,
            "p99_von_mises": None,
            "max_s_max_principal": None,
            "min_s_min_principal": None,
        }

    bone_offset = model.global_node_offset(bone_part.name)
    n_bone_nodes = len(bone_part.nodes)

    # 2. interface_nodes_bone 전역/로컬 인덱스 정규화
    iface_arr = np.asarray(interface_nodes_bone, dtype=np.int64).ravel()
    if len(iface_arr) == 0:
        return {
            "num_nodes": 0,
            "radius_mm": float(radius_mm),
            "warning": "지정된 뼈-임플란트 접촉면 절점이 없습니다.",
            "max_von_mises": None,
            "mean_von_mises": None,
            "p95_von_mises": None,
            "p99_von_mises": None,
            "max_s_max_principal": None,
            "min_s_min_principal": None,
        }

    # 전역 0-based 인덱스로 들어온 경우 로컬로 변환
    if np.all(iface_arr >= bone_offset) and np.all(iface_arr < bone_offset + n_bone_nodes):
        iface_local = iface_arr - bone_offset
    else:
        iface_local = iface_arr[(iface_arr >= 0) & (iface_arr < n_bone_nodes)]

    if len(iface_local) == 0:
        return {
            "num_nodes": 0,
            "radius_mm": float(radius_mm),
            "warning": "뼈 파트 범위에 부합하는 접촉면 절점이 없습니다.",
            "max_von_mises": None,
            "mean_von_mises": None,
            "p95_von_mises": None,
            "p99_von_mises": None,
            "max_s_max_principal": None,
            "min_s_min_principal": None,
        }

    # 3. cKDTree 근접 절점 탐색 (radius_mm 이내)
    tree = cKDTree(bone_part.nodes)
    iface_coords = bone_part.nodes[iface_local]
    neighbor_lists = tree.query_ball_point(iface_coords, r=float(radius_mm))

    flat_neighbors = [idx for sublist in neighbor_lists for idx in sublist]
    if len(flat_neighbors) == 0:
        return {
            "num_nodes": 0,
            "radius_mm": float(radius_mm),
            "warning": f"접촉면으로부터 반경 {radius_mm:.2f} mm 이내에 속한 뼈 절점이 없습니다.",
            "max_von_mises": None,
            "mean_von_mises": None,
            "p95_von_mises": None,
            "p99_von_mises": None,
            "max_s_max_principal": None,
            "min_s_min_principal": None,
        }

    selected_local_nodes = np.unique(np.array(flat_neighbors, dtype=np.int64))

    # 4. 통계 계산 내부 헬퍼
    def _calc_stats(loc_nodes: np.ndarray) -> dict:
        if len(loc_nodes) == 0:
            return {
                "num_nodes": 0,
                "max_von_mises": None,
                "mean_von_mises": None,
                "p95_von_mises": None,
                "p99_von_mises": None,
                "max_s_max_principal": None,
                "min_s_min_principal": None,
            }

        gnodes = bone_offset + loc_nodes

        # von Mises 통계
        vm_sub = rs.von_mises[gnodes] if rs.von_mises is not None else None
        if vm_sub is not None:
            vm_valid = vm_sub[~np.isnan(vm_sub)]
            if len(vm_valid) > 0:
                max_vm = float(np.max(vm_valid))
                mean_vm = float(np.mean(vm_valid))
                p95_vm = float(np.percentile(vm_valid, 95))
                p99_vm = float(np.percentile(vm_valid, 99))
            else:
                max_vm = mean_vm = p95_vm = p99_vm = None
        else:
            max_vm = mean_vm = p95_vm = p99_vm = None

        # 최대 주응력 (인장 응력)
        smax_sub = rs.s_max_principal[gnodes] if rs.s_max_principal is not None else None
        if smax_sub is not None:
            smax_valid = smax_sub[~np.isnan(smax_sub)]
            max_smax = float(np.max(smax_valid)) if len(smax_valid) > 0 else None
        else:
            max_smax = None

        # 최소 주응력 (압축 응력 - 최소값)
        smin_sub = rs.s_min_principal[gnodes] if rs.s_min_principal is not None else None
        if smin_sub is not None:
            smin_valid = smin_sub[~np.isnan(smin_sub)]
            min_smin = float(np.min(smin_valid)) if len(smin_valid) > 0 else None
        else:
            min_smin = None

        return {
            "num_nodes": int(len(loc_nodes)),
            "max_von_mises": max_vm,
            "mean_von_mises": mean_vm,
            "p95_von_mises": p95_vm,
            "p99_von_mises": p99_vm,
            "max_s_max_principal": max_smax,
            "min_s_min_principal": min_smin,
        }

    # 전체 반경 영역 통계
    overall_stats = _calc_stats(selected_local_nodes)
    result: Dict[str, Any] = {
        "part_name": bone_part.name,
        "radius_mm": float(radius_mm),
        **overall_stats,
    }

    # 5. 피질골 / 해면골 분리 통계 계산 (materials 제공 시)
    if materials is not None and bone_part.name in materials.element_material:
        mat_indices = materials.element_material[bone_part.name]
        cortical_target = cortical_name.strip().lower()

        # cortical 재료 ID 식별
        cortical_mat_ids = set()
        for idx, m_def in enumerate(materials.materials):
            if cortical_target in m_def.name.lower():
                cortical_mat_ids.add(idx)

        elems = np.asarray(bone_part.elements, dtype=np.int64)
        if len(elems) > 0 and np.all(elems >= bone_offset) and np.all(elems < bone_offset + n_bone_nodes):
            elems = elems - bone_offset

        # 각 절점별 인접 요소 중 cortical 요소 포함 여부 검사
        cortical_nodes_set = set()
        for e_idx, el in enumerate(elems):
            if e_idx < len(mat_indices) and mat_indices[e_idx] in cortical_mat_ids:
                for nid in el:
                    cortical_nodes_set.add(int(nid))

        is_cortical = np.array([nid in cortical_nodes_set for nid in selected_local_nodes], dtype=bool)
        cortical_nodes = selected_local_nodes[is_cortical]
        cancellous_nodes = selected_local_nodes[~is_cortical]

        result["cortical"] = _calc_stats(cortical_nodes)
        result["cancellous"] = _calc_stats(cancellous_nodes)

    return result


def max_stress_location(
    rs: ResultSet,
    model: MeshModel,
    field: str = "von_mises",
    part_name: Optional[str] = None,
) -> dict:
    """
    Locate peak stress coordinate, global node index, and value across the model or a specific part.

    Filters out NaN values. Returns dictionary with global node index, part name,
    undeformed 3D coordinates, and scalar field value.

    Args:
        rs: Computed ResultSet.
        model: MeshModel containing assembly parts.
        field: Target field name ('von_mises', 's_max_principal', 's_min_principal', 'u_mag').
        part_name: Optional part name filter. If None, searches across all parts.

    Returns:
        dict: {
            "node_idx": int | None,
            "part_name": str | None,
            "coord": list[float] | None,
            "value": float | None,
            "field": str,
        }
    """
    val_arr = getattr(rs, field, None)
    if val_arr is None or len(val_arr) == 0:
        return {
            "node_idx": None,
            "part_name": None,
            "coord": None,
            "value": None,
            "field": field,
        }

    val_arr = np.asarray(val_arr, dtype=np.float64)

    if part_name is not None:
        part = model.get_part(part_name)
        if part is None:
            return {
                "node_idx": None,
                "part_name": None,
                "coord": None,
                "value": None,
                "field": field,
            }

        offset = model.global_node_offset(part_name)
        n_nodes = len(part.nodes)
        part_vals = val_arr[offset : offset + n_nodes]
        valid_mask = ~np.isnan(part_vals)

        if not np.any(valid_mask):
            return {
                "node_idx": None,
                "part_name": part_name,
                "coord": None,
                "value": None,
                "field": field,
            }

        loc_indices = np.where(valid_mask)[0]
        best_i = int(np.argmax(part_vals[valid_mask]))
        best_loc_idx = int(loc_indices[best_i])
        best_global_idx = int(offset + best_loc_idx)
        best_val = float(part_vals[best_loc_idx])
        coord = [float(x) for x in part.nodes[best_loc_idx]]

        return {
            "node_idx": best_global_idx,
            "part_name": part_name,
            "coord": coord,
            "value": best_val,
            "field": field,
        }

    else:
        # 전체 모델 대상 탐색
        total_nodes = model.total_nodes()
        search_vals = val_arr[:total_nodes]
        valid_mask = ~np.isnan(search_vals)

        if not np.any(valid_mask):
            return {
                "node_idx": None,
                "part_name": None,
                "coord": None,
                "value": None,
                "field": field,
            }

        valid_indices = np.where(valid_mask)[0]
        best_i = int(np.argmax(search_vals[valid_mask]))
        best_global_idx = int(valid_indices[best_i])
        best_val = float(search_vals[best_global_idx])

        # 해당 전역 절점 인덱스가 속한 파트 및 로컬 좌표 검색
        found_part_name = None
        found_coord = None
        curr_offset = 0

        for p in model.parts:
            p_len = len(p.nodes)
            if curr_offset <= best_global_idx < curr_offset + p_len:
                found_part_name = p.name
                loc_idx = best_global_idx - curr_offset
                found_coord = [float(x) for x in p.nodes[loc_idx]]
                break
            curr_offset += p_len

        return {
            "node_idx": best_global_idx,
            "part_name": found_part_name,
            "coord": found_coord,
            "value": best_val,
            "field": field,
        }


def peri_implant_stats(
    rs: ResultSet,
    model: MeshModel,
    radius_mm: float = 1.0,
    materials: Optional[MaterialAssignment] = None,
    ties: Optional[Sequence[TieDef]] = None,
    bone_part_name: Optional[str] = None,
) -> dict:
    """
    Convenience wrapper to compute peri-implant bone stress statistics.

    Automatically resolves bone interface nodes from TieDef constraints if provided,
    or identifies contact boundary nodes via proximity search between implant and bone parts.

    Args:
        rs: Computed ResultSet.
        model: MeshModel containing assembly parts.
        radius_mm: Distance from interface to analyze (mm).
        materials: Optional MaterialAssignment for cortical/cancellous stratification.
        ties: Optional list of TieDef constraints.
        bone_part_name: Optional bone part name override.

    Returns:
        dict: Summary statistics including node count and stress percentiles.
    """
    if not model.parts:
        return {
            "num_nodes": 0,
            "radius_mm": float(radius_mm),
            "max_von_mises": None,
            "mean_von_mises": None,
            "p95_von_mises": None,
            "p99_von_mises": None,
            "max_s_max_principal": None,
            "min_s_min_principal": None,
        }

    # 1. 뼈 파트 식별
    target_bone = None
    if bone_part_name is not None:
        target_bone = model.get_part(bone_part_name)
    if target_bone is None:
        for p in model.parts:
            if "bone" in p.name.lower() or "mandible" in p.name.lower():
                target_bone = p
                break
    if target_bone is None:
        target_bone = model.parts[0]

    # 2. 임플란트 파트 식별
    target_implant = None
    for p in model.parts:
        if p.name != target_bone.name and ("implant" in p.name.lower() or "fixture" in p.name.lower()):
            target_implant = p
            break
    if target_implant is None and len(model.parts) > 1:
        for p in model.parts:
            if p.name != target_bone.name:
                target_implant = p
                break

    if target_implant is None:
        # 단일 파트인 경우 인터페이스가 없으므로 빈 통계 반환
        return {
            "num_nodes": 0,
            "radius_mm": float(radius_mm),
            "max_von_mises": None,
            "mean_von_mises": None,
            "p95_von_mises": None,
            "p99_von_mises": None,
            "max_s_max_principal": None,
            "min_s_min_principal": None,
        }

    # 3. 접촉면 절점 추출
    iface_nodes = np.array([], dtype=np.int64)
    if ties:
        for tie in ties:
            sub_iface = extract_bone_interface_nodes(tie, model, bone_part_name=target_bone.name)
            if len(sub_iface) > 0:
                iface_nodes = np.union1d(iface_nodes, sub_iface)

    if len(iface_nodes) == 0:
        # 최근접 기하 거리 기반 인터페이스 추정
        tree_bone = cKDTree(target_bone.nodes)
        distances, nearest_idx = tree_bone.query(target_implant.nodes, k=1)
        min_d = np.min(distances) if len(distances) > 0 else 0.0
        # 최근접 거리 + 0.5mm 이내의 절점들을 인터페이스로 채택
        threshold = max(min_d + 0.5, 0.5)
        contact_mask = distances <= threshold
        if np.any(contact_mask):
            iface_nodes = np.unique(nearest_idx[contact_mask].astype(np.int64))
        elif len(nearest_idx) > 0:
            iface_nodes = np.unique(nearest_idx.astype(np.int64))

    return peri_implant_bone_stats(
        rs=rs,
        model=model,
        interface_nodes_bone=iface_nodes,
        radius_mm=radius_mm,
        materials=materials,
        bone_part_name=target_bone.name,
    )
