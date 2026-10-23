"""
result_fields.py
Tensor transformations, stress calculations, and ResultSet construction from CalculiX .frd.

[Units]
- Length / Displacement (u): mm
- Stress: MPa (N/mm^2)
- Strain: Dimensionless
- Force / Reaction Force (rf): N
"""

from typing import Optional, Dict, Any, Union
import numpy as np

from fea.models import MeshModel, BoundaryConditionSet
from fea.results import FrdData, ResultSet
from solver.calculix_writer import GlobalIndexMapper


def principal_stresses(stress6: np.ndarray) -> np.ndarray:
    """
    Calculate sorted principal stresses (s1 >= s2 >= s3) from Cauchy stress 6-components.

    Vectorized implementation using batch np.linalg.eigvalsh over (N, 3, 3) tensors.
    No Python loops are used for tensor eigenvalue decomposition.
    Rows containing NaN values in stress6 will retain NaN in the output.

    Args:
        stress6: (N, 6) float array ordered as [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.

    Returns:
        np.ndarray: (N, 3) float64 array of principal stresses descending (s1 >= s2 >= s3).
    """
    stress6 = np.asarray(stress6, dtype=np.float64)
    if stress6.ndim == 1:
        stress6 = stress6.reshape(1, -1)

    n_nodes = stress6.shape[0]
    out = np.full((n_nodes, 3), np.nan, dtype=np.float64)
    if n_nodes == 0:
        return out

    # NaN 행 검출
    nan_mask = np.isnan(stress6).any(axis=1)
    valid_mask = ~nan_mask

    if not np.any(valid_mask):
        return out

    valid_stress = stress6[valid_mask]
    n_valid = valid_stress.shape[0]

    # (N, 3, 3) 대칭 응력 텐서 배치 구성
    # [ SXX  SXY  SZX ]
    # [ SXY  SYY  SYZ ]
    # [ SZX  SYZ  SZZ ]
    tensors = np.empty((n_valid, 3, 3), dtype=np.float64)
    tensors[:, 0, 0] = valid_stress[:, 0]  # SXX
    tensors[:, 1, 1] = valid_stress[:, 1]  # SYY
    tensors[:, 2, 2] = valid_stress[:, 2]  # SZZ

    tensors[:, 0, 1] = valid_stress[:, 3]  # SXY
    tensors[:, 1, 0] = valid_stress[:, 3]

    tensors[:, 1, 2] = valid_stress[:, 4]  # SYZ
    tensors[:, 2, 1] = valid_stress[:, 4]

    tensors[:, 0, 2] = valid_stress[:, 5]  # SZX
    tensors[:, 2, 0] = valid_stress[:, 5]

    # np.linalg.eigvalsh는 대칭/에르미트 행렬의 고유치를 오름차순으로 반환
    eigenvals = np.linalg.eigvalsh(tensors)  # (n_valid, 3)

    # 내림차순 정렬 (s1 >= s2 >= s3)
    out[valid_mask] = eigenvals[:, ::-1]
    return out


def von_mises(stress6: np.ndarray) -> np.ndarray:
    """
    Calculate von Mises equivalent stress from Cauchy stress 6-components.

    Formula:
        sqrt(0.5 * ((sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2) + 3 * (sxy^2 + syz^2 + szx^2))

    Args:
        stress6: (N, 6) float array ordered as [SXX, SYY, SZZ, SXY, SYZ, SZX] in MPa.

    Returns:
        np.ndarray: (N,) float64 array of von Mises stress in MPa.
    """
    stress6 = np.asarray(stress6, dtype=np.float64)
    if stress6.ndim == 1:
        stress6 = stress6.reshape(1, -1)

    sxx = stress6[:, 0]
    syy = stress6[:, 1]
    szz = stress6[:, 2]
    sxy = stress6[:, 3]
    syz = stress6[:, 4]
    szx = stress6[:, 5]

    diff_sq = (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
    shear_sq = sxy ** 2 + syz ** 2 + szx ** 2
    vm_sq = 0.5 * diff_sq + 3.0 * shear_sq

    # 수치 오차로 인한 미세 음수 방지
    vm_sq = np.maximum(vm_sq, 0.0)
    return np.sqrt(vm_sq)


def build_result_set(
    frd: FrdData,
    model: MeshModel,
    step: Union[int, str] = "last",
) -> ResultSet:
    """
    Convert raw FrdData into a unified ResultSet aligned with MeshModel global node ordering.

    Uses Phase 3 GlobalIndexMapper to translate 1-based .frd node numbers to 0-based
    MeshModel assembly node indices. Nodal arrays are sized to total_nodes of model;
    nodes omitted from .frd result blocks are populated with NaN.

    Args:
        frd: Parsed FrdData container.
        model: MeshModel containing assembly mesh parts.
        step: Step increment selector ('last', 'first', or integer index).

    Returns:
        ResultSet: Standardized result fields aligned to MeshModel global 0-based node indices.

    Raises:
        ValueError: If node counts differ or undeformed coordinates mismatch (> 1e-4 mm).
    """
    # 1. Phase 3 GlobalIndexMapper 재사용
    mapper = GlobalIndexMapper(model)
    total_nodes = mapper.total_nodes

    # 메쉬 모델 전체 절점 비변형 좌표 배열 생성
    if model.parts:
        model_coords = np.vstack([p.nodes for p in model.parts])
    else:
        model_coords = np.zeros((0, 3), dtype=np.float64)

    # 2. 절점 수 불일치 검증
    if len(frd.node_ids) != total_nodes:
        raise ValueError(
            f"CalculiX 결과 파일(.frd)의 절점 수({len(frd.node_ids)})와 "
            f"메쉬 모델의 절점 수({total_nodes})가 일치하지 않습니다."
        )

    # 3. 비변형 좌표 일치 검증 (허용 오차 1e-4 mm)
    # frd.node_ids는 1-based 전역 번호이므로 전역 0-based 인덱스는 node_ids - 1
    frd_global_0based = frd.node_ids - 1
    if np.any(frd_global_0based < 0) or np.any(frd_global_0based >= total_nodes):
        raise ValueError("CalculiX 결과 파일의 절점 번호가 메쉬 모델의 절점 범위를 벗어났습니다.")

    coord_diff = np.abs(frd.node_coords - model_coords[frd_global_0based])
    max_coord_diff = float(np.max(coord_diff)) if len(coord_diff) > 0 else 0.0
    if max_coord_diff > 1e-4:
        raise ValueError(
            f"CalculiX 결과 파일(.frd)의 절점 좌표가 메쉬 모델과 일치하지 않습니다 "
            f"(최대 오차: {max_coord_diff:.6f} mm > 허용 오차 1e-4 mm)."
        )

    sim_time = 0.0

    # 4. DISP 블록 처리 -> u, u_mag
    disp_block = frd.get_block("DISP", which=step)
    if disp_block is not None and len(disp_block.node_ids) > 0:
        u = np.full((total_nodes, 3), np.nan, dtype=np.float64)
        node_indices = disp_block.node_ids - 1
        valid_range = (node_indices >= 0) & (node_indices < total_nodes)
        u[node_indices[valid_range]] = disp_block.values[valid_range, :3]
        u_mag = np.linalg.norm(u, axis=1)
        sim_time = disp_block.time
    else:
        u = None
        u_mag = None

    # 5. STRESS 블록 처리 -> stress6, von_mises, s_max_principal, s_min_principal
    stress_block = frd.get_block("STRESS", which=step)
    if stress_block is not None and len(stress_block.node_ids) > 0:
        stress6 = np.full((total_nodes, 6), np.nan, dtype=np.float64)
        node_indices = stress_block.node_ids - 1
        valid_range = (node_indices >= 0) & (node_indices < total_nodes)
        stress6[node_indices[valid_range]] = stress_block.values[valid_range, :6]

        vm = von_mises(stress6)
        ps = principal_stresses(stress6)
        s_max = ps[:, 0]
        s_min = ps[:, 2]
        if sim_time == 0.0:
            sim_time = stress_block.time
    else:
        stress6 = None
        vm = None
        s_max = None
        s_min = None

    # 6. TOSTRAIN (또는 E) 블록 처리 -> strain6
    strain_block = frd.get_block("TOSTRAIN", which=step) or frd.get_block("E", which=step)
    if strain_block is not None and len(strain_block.node_ids) > 0:
        strain6 = np.full((total_nodes, 6), np.nan, dtype=np.float64)
        node_indices = strain_block.node_ids - 1
        valid_range = (node_indices >= 0) & (node_indices < total_nodes)
        strain6[node_indices[valid_range]] = strain_block.values[valid_range, :6]
        if sim_time == 0.0:
            sim_time = strain_block.time
    else:
        strain6 = None

    # 7. FORC (또는 RF) 블록 처리 -> rf
    forc_block = frd.get_block("FORC", which=step) or frd.get_block("RF", which=step)
    if forc_block is not None and len(forc_block.node_ids) > 0:
        rf = np.full((total_nodes, 3), np.nan, dtype=np.float64)
        node_indices = forc_block.node_ids - 1
        valid_range = (node_indices >= 0) & (node_indices < total_nodes)
        rf[node_indices[valid_range]] = forc_block.values[valid_range, :3]
        if sim_time == 0.0:
            sim_time = forc_block.time
    else:
        rf = None

    source_frd = getattr(frd, "source_frd", "")

    return ResultSet(
        time=float(sim_time),
        u=u,
        u_mag=u_mag,
        stress6=stress6,
        von_mises=vm,
        s_max_principal=s_max,
        s_min_principal=s_min,
        strain6=strain6,
        rf=rf,
        source_frd=source_frd,
    )


def part_statistics(rs: ResultSet, model: MeshModel, part_name: str) -> dict:
    """
    Calculate summary statistics for a specific mesh part, filtering out NaNs.

    Computes node count, maximum u_mag, von Mises (max, mean, p95, p99),
    s_max_principal (max), and s_min_principal (min). Also provides the global node
    indices and undeformed 3D coordinates of peak locations.

    Args:
        rs: Computed ResultSet.
        model: MeshModel containing the part.
        part_name: Identifier name of the part.

    Returns:
        dict: Summary statistics dictionary.

    Raises:
        KeyError: If part_name is not found in MeshModel.
    """
    part = model.get_part(part_name)
    if part is None:
        raise KeyError(f"Part '{part_name}' not found in MeshModel.")

    offset = model.global_node_offset(part_name)
    n_nodes = len(part.nodes)
    sl = slice(offset, offset + n_nodes)

    stats: Dict[str, Any] = {
        "part_name": part_name,
        "num_nodes": int(n_nodes),
        "max_u_mag": None,
        "max_u_mag_node_idx": None,
        "max_u_mag_node_coord": None,
        "max_von_mises": None,
        "mean_von_mises": None,
        "p95_von_mises": None,
        "p99_von_mises": None,
        "max_von_mises_node_idx": None,
        "max_von_mises_node_coord": None,
        "max_s_max_principal": None,
        "max_s_max_node_idx": None,
        "max_s_max_node_coord": None,
        "min_s_min_principal": None,
        "min_s_min_node_idx": None,
        "min_s_min_node_coord": None,
    }

    # 1. Displacement magnitude (u_mag)
    if rs.u_mag is not None and len(rs.u_mag) > 0:
        part_u = rs.u_mag[sl]
        valid_mask = ~np.isnan(part_u)
        if np.any(valid_mask):
            valid_vals = part_u[valid_mask]
            loc_indices = np.where(valid_mask)[0]
            best_i = int(np.argmax(valid_vals))
            loc_idx = loc_indices[best_i]
            stats["max_u_mag"] = float(valid_vals[best_i])
            stats["max_u_mag_node_idx"] = int(offset + loc_idx)
            stats["max_u_mag_node_coord"] = [float(x) for x in part.nodes[loc_idx]]

    # 2. Von Mises stress
    if rs.von_mises is not None and len(rs.von_mises) > 0:
        part_vm = rs.von_mises[sl]
        valid_mask = ~np.isnan(part_vm)
        if np.any(valid_mask):
            valid_vals = part_vm[valid_mask]
            loc_indices = np.where(valid_mask)[0]
            best_i = int(np.argmax(valid_vals))
            loc_idx = loc_indices[best_i]
            stats["max_von_mises"] = float(valid_vals[best_i])
            stats["mean_von_mises"] = float(np.mean(valid_vals))
            stats["p95_von_mises"] = float(np.percentile(valid_vals, 95))
            stats["p99_von_mises"] = float(np.percentile(valid_vals, 99))
            stats["max_von_mises_node_idx"] = int(offset + loc_idx)
            stats["max_von_mises_node_coord"] = [float(x) for x in part.nodes[loc_idx]]

    # 3. Maximum principal stress (s_max_principal)
    if rs.s_max_principal is not None and len(rs.s_max_principal) > 0:
        part_smax = rs.s_max_principal[sl]
        valid_mask = ~np.isnan(part_smax)
        if np.any(valid_mask):
            valid_vals = part_smax[valid_mask]
            loc_indices = np.where(valid_mask)[0]
            best_i = int(np.argmax(valid_vals))
            loc_idx = loc_indices[best_i]
            stats["max_s_max_principal"] = float(valid_vals[best_i])
            stats["max_s_max_node_idx"] = int(offset + loc_idx)
            stats["max_s_max_node_coord"] = [float(x) for x in part.nodes[loc_idx]]

    # 4. Minimum principal stress (s_min_principal)
    if rs.s_min_principal is not None and len(rs.s_min_principal) > 0:
        part_smin = rs.s_min_principal[sl]
        valid_mask = ~np.isnan(part_smin)
        if np.any(valid_mask):
            valid_vals = part_smin[valid_mask]
            loc_indices = np.where(valid_mask)[0]
            best_i = int(np.argmin(valid_vals))  # 최소값
            loc_idx = loc_indices[best_i]
            stats["min_s_min_principal"] = float(valid_vals[best_i])
            stats["min_s_min_node_idx"] = int(offset + loc_idx)
            stats["min_s_min_node_coord"] = [float(x) for x in part.nodes[loc_idx]]

    return stats


def reaction_balance(
    rs: ResultSet,
    bc: BoundaryConditionSet,
    model: MeshModel,
) -> Optional[dict]:
    """
    Calculate static reaction equilibrium balance between constraints and applied forces.

    Checks sum of nodal reaction forces at fixed boundary condition sets against
    the sum of applied external force vectors.

    Args:
        rs: ResultSet containing reaction force field 'rf'.
        bc: BoundaryConditionSet containing fixed sets and applied loads.
        model: MeshModel for indexing.

    Returns:
        dict: Reaction balance summary including reaction sum, applied sum, residual,
              and relative error, or None if rs.rf is None.
    """
    if rs.rf is None:
        return None

    mapper = GlobalIndexMapper(model)

    # 1. 구속 절점의 전역 0-based 인덱스 수집
    fixed_global_0based = []
    for fix_item in bc.fixed_node_sets:
        part_name = fix_item.get("part", "")
        nids = fix_item.get("node_ids", np.array([], dtype=np.int64))
        if len(nids) > 0:
            gnids = mapper.node_ids(part_name, nids)
            fixed_global_0based.extend((gnids - 1).tolist())

    if len(fixed_global_0based) > 0:
        unique_fixed = np.unique(fixed_global_0based)
        valid_fixed = unique_fixed[(unique_fixed >= 0) & (unique_fixed < len(rs.rf))]
        if len(valid_fixed) > 0:
            total_reaction = np.nansum(rs.rf[valid_fixed], axis=0)
        else:
            total_reaction = np.zeros(3, dtype=np.float64)
    else:
        total_reaction = np.zeros(3, dtype=np.float64)

    # 2. 적용 하중 벡터 합력 수집
    total_applied = np.zeros(3, dtype=np.float64)
    for load_item in bc.loads:
        f_vec = np.asarray(load_item.get("force_N", (0.0, 0.0, 0.0)), dtype=np.float64)
        total_applied += f_vec

    reaction_mag = float(np.linalg.norm(total_reaction))
    applied_mag = float(np.linalg.norm(total_applied))

    # 잔차 (정적 평형: Total Reaction + Total Applied = 0)
    residual = total_reaction + total_applied
    residual_mag = float(np.linalg.norm(residual))

    if applied_mag > 1e-12:
        relative_error = float(residual_mag / applied_mag)
    else:
        relative_error = float(residual_mag)

    return {
        "reaction_force_sum": [float(x) for x in total_reaction],
        "applied_force_sum": [float(x) for x in total_applied],
        "reaction_mag": reaction_mag,
        "applied_mag": applied_mag,
        "residual_force": [float(x) for x in residual],
        "residual_mag": residual_mag,
        "relative_error": relative_error,
    }
