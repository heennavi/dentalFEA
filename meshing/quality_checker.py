"""
quality_checker.py
Mesh quality inspection and validation metrics for tetrahedral solid elements.

[Quality Metrics]
- Volume: Signed volume (V = 1/6 * det(J)) in mm^3.
- Aspect Ratio: Ratio of maximum edge length to minimum edge length (L_max / L_min).
  Normalized to 1.0 for regular tetrahedra; higher values indicate sliver or needle distortion.
- Scaled Jacobian: Normalized determinant of the Jacobian matrix across all 4 vertices.
  Value of 1.0 represents an ideal equilateral tetrahedron; <= 0 represents an inverted or collapsed element.
"""

from typing import Dict, Any, List
import numpy as np


def tet_quality(nodes: np.ndarray, elements: np.ndarray) -> Dict[str, Any]:
    """
    Compute vectorized quality metrics for 4-node or 10-node tetrahedral elements.

    For quadratic (C3D10) elements, the first 4 corner vertices are evaluated.

    Aspect Ratio Definition:
        L_max / L_min, where L_max and L_min are the maximum and minimum lengths
        among the 6 tetrahedral edges. (Ideal equilateral tetrahedron = 1.0).

    Args:
        nodes: (N, 3) float64 array of vertex coordinates in mm.
        elements: (M, 4 or 10) int64 array of tetrahedral nodal connectivity.

    Returns:
        dict:
            - n_elements (int): Total number of solid elements.
            - n_negative (int): Number of inverted elements with non-positive volume (V <= 0).
            - min_volume (float): Minimum element volume in mm^3.
            - aspect_max (float): Maximum aspect ratio across all elements.
            - aspect_mean (float): Mean aspect ratio across all elements.
            - min_scaled_jacobian (float): Minimum scaled Jacobian across all elements.
            - bad_element_ids (np.ndarray): (B,) int64 array of element indices exceeding defect thresholds
                                            (V <= 0, aspect ratio > 15.0, or scaled Jacobian < 0.05).
    """
    n_elems = len(elements)
    if n_elems == 0 or len(nodes) == 0:
        return {
            "n_elements": 0,
            "n_negative": 0,
            "min_volume": 0.0,
            "aspect_max": 1.0,
            "aspect_mean": 1.0,
            "min_scaled_jacobian": 1.0,
            "bad_element_ids": np.array([], dtype=np.int64),
        }

    # First 4 corner vertices
    v0 = nodes[elements[:, 0]]
    v1 = nodes[elements[:, 1]]
    v2 = nodes[elements[:, 2]]
    v3 = nodes[elements[:, 3]]

    # 1. Signed volume calculation
    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0

    cross0 = np.cross(d1, d2)
    det0 = np.sum(cross0 * d3, axis=1)
    volumes = det0 / 6.0

    neg_mask = volumes <= 0.0
    n_negative = int(np.sum(neg_mask))
    min_volume = float(np.min(volumes))

    # 2. 6 edge lengths
    e01 = np.linalg.norm(v1 - v0, axis=1)
    e02 = np.linalg.norm(v2 - v0, axis=1)
    e03 = np.linalg.norm(v3 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e23 = np.linalg.norm(v3 - v2, axis=1)
    e31 = np.linalg.norm(v1 - v3, axis=1)

    edges_all = np.column_stack([e01, e02, e03, e12, e23, e31])
    l_max = np.max(edges_all, axis=1)
    l_min = np.min(edges_all, axis=1)
    # Avoid zero division
    aspect_ratios = l_max / np.maximum(l_min, 1e-12)

    aspect_max = float(np.max(aspect_ratios))
    aspect_mean = float(np.mean(aspect_ratios))

    # 3. Scaled Jacobian calculation at 4 vertices
    # J_vi = (6 * V) / (e_a * e_b * e_c) * sqrt(2)
    # For an equilateral regular tet, 6V / (e^3) = sqrt(2)/2, so scaled Jacobian is exactly 1.0.
    sqrt2 = np.sqrt(2.0)
    vol6 = 6.0 * volumes

    denom0 = np.maximum(e01 * e02 * e03, 1e-12)
    sj0 = (vol6 / denom0) * sqrt2

    denom1 = np.maximum(e01 * e12 * e31, 1e-12)
    sj1 = (vol6 / denom1) * sqrt2

    denom2 = np.maximum(e02 * e12 * e23, 1e-12)
    sj2 = (vol6 / denom2) * sqrt2

    denom3 = np.maximum(e03 * e31 * e23, 1e-12)
    sj3 = (vol6 / denom3) * sqrt2

    sj_min_per_elem = np.minimum(np.minimum(sj0, sj1), np.minimum(sj2, sj3))
    min_scaled_jacobian = float(np.min(sj_min_per_elem))

    # 4. Bad elements identification
    # Defect criteria: V <= 0, Aspect Ratio > 15.0, or Scaled Jacobian < 0.05
    bad_mask = neg_mask | (aspect_ratios > 15.0) | (sj_min_per_elem < 0.05)
    bad_element_ids = np.where(bad_mask)[0]

    return {
        "n_elements": n_elems,
        "n_negative": n_negative,
        "min_volume": min_volume,
        "aspect_max": aspect_max,
        "aspect_mean": aspect_mean,
        "min_scaled_jacobian": min_scaled_jacobian,
        "bad_element_ids": bad_element_ids,
    }


def quality_report_text(report: Dict[str, Any]) -> str:
    """
    Format a user-friendly Korean summary text from quality inspection report.

    Args:
        report: Dictionary returned by tet_quality().

    Returns:
        str: Multiline formatted report text.
    """
    n_elems = report.get("n_elements", 0)
    n_neg = report.get("n_negative", 0)
    min_vol = report.get("min_volume", 0.0)
    ar_max = report.get("aspect_max", 0.0)
    ar_mean = report.get("aspect_mean", 0.0)
    min_sj = report.get("min_scaled_jacobian", 1.0)
    n_bad = len(report.get("bad_element_ids", []))

    neg_status = "정상 (0개)" if n_neg == 0 else f"⚠️ 위험 ({n_neg}개 반전 요소 검출)"
    qual_status = "양호 (우수)" if (ar_max < 10.0 and min_sj > 0.1 and n_neg == 0) else "주의 요망 (요소 크기 조정 권장)"

    lines = [
        "========================================",
        "        3D 사면체 메쉬 품질 검사 보고서      ",
        "========================================",
        f"• 총 요소 수: {n_elems:,} 개",
        f"• 음의 부피(반전) 요소: {neg_status}",
        f"• 최소 요소 체적: {min_vol:.6f} mm³",
        f"• 형상비(Aspect Ratio): 최대 {ar_max:.2f} / 평균 {ar_mean:.2f}",
        f"• 최소 Scaled Jacobian: {min_sj:.4f}",
        f"• 불량/왜곡 요소 수: {n_bad:,} 개 (전체의 { (n_bad / max(1, n_elems)) * 100:.2f}%)",
        f"• 전반적 품질 판정: {qual_status}",
        "========================================",
    ]
    return "\n".join(lines)


def assert_mesh_ok(report: Dict[str, Any], max_negative: int = 0) -> None:
    """
    Assert that mesh contains no more than max_negative inverted elements.

    Args:
        report: Dictionary returned by tet_quality().
        max_negative: Maximum tolerable inverted elements (default: 0).

    Raises:
        ValueError: If inverted element count exceeds max_negative.
    """
    n_neg = report.get("n_negative", 0)
    if n_neg > max_negative:
        raise ValueError(
            f"메쉬 품질 검사 실패: 음의 부피를 가진 반전 사면체 요소가 {n_neg}개 "
            f"검출되었습니다 (허용치: {max_negative}개)."
        )
