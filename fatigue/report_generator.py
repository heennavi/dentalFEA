"""
report_generator.py
Comprehensive fatigue analysis report generator in HTML and plain text formats.
Includes mandatory clinical disclaimer and explicit assumption statements.
Pure logic module with NO PySide6 or rendering dependencies.
"""

import datetime
from typing import Optional, Dict, Any, List
import numpy as np

from fea.models import MeshModel
from fea.results import ResultSet
from .models import FatigueMaterial, FatigueLoadCase


DISCLAIMER_TEXT = "⚠️ [주의] 임상 판단 보조용, 시험 대체 불가 (For Clinical Decision Support Only - Cannot Replace Physical Testing)"


def generate_fatigue_report_html(
    rs: ResultSet,
    model: Optional[MeshModel] = None,
    material: Optional[FatigueMaterial] = None,
    load_case: Optional[FatigueLoadCase] = None,
    hotspot_info: Optional[Dict[str, Any]] = None,
    benchmark_comp: Optional[Any] = None,
    patient_info: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate a full-featured HTML fatigue analysis report.

    Args:
        rs: ResultSet containing fatigue fields.
        model: Optional MeshModel assembly.
        material: FatigueMaterial with mechanical parameters.
        load_case: FatigueLoadCase with loading parameters.
        hotspot_info: Optional hotspot metadata dictionary.
        benchmark_comp: Optional LiteratureBenchmarkComparison instance.
        patient_info: Optional patient/case metadata (name, id, date, etc.).

    Returns:
        Complete HTML string.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. 핫스팟 및 안전계수 요약
    info = hotspot_info or getattr(rs, "fatigue_hotspot_info", None) or {}
    min_fs = info.get("min_safety_factor")
    if min_fs is None and rs.fatigue_fs is not None and len(rs.fatigue_fs) > 0:
        valid_fs = rs.fatigue_fs[~np.isnan(rs.fatigue_fs)]
        min_fs = float(np.min(valid_fs)) if len(valid_fs) > 0 else 0.0

    fs_val = float(min_fs) if min_fs is not None else 0.0
    is_safe = (fs_val >= 1.0)
    fs_status_str = "안전 (Safe)" if is_safe else "피로 파손 위험 (Fatigue Failure Risk)"
    fs_color = "#4cd964" if is_safe else "#ff3b30"

    mat_name = material.name if material else info.get("material_name", "Ti-6Al-4V ELI")
    mat_std = material.standard if material else info.get("material_standard", "ASTM F136")
    mat_se = material.fatigue_limit_Se if material else info.get("fatigue_limit_Se", 510.0)
    mat_sut = material.tensile_strength_Sut if material else 860.0
    mat_sy = material.yield_strength_Sy if material else 795.0
    mat_sources = ", ".join(material.sources) if material and material.sources else "materials_fatigue.yaml / ASTM F136"

    r_ratio = load_case.stress_ratio_R if load_case else info.get("stress_ratio_R", 0.1)
    target_cycles = load_case.target_cycles if load_case else info.get("applied_cycles", 5.0e6)
    method_name = load_case.mean_stress_method.value if load_case and hasattr(load_case.mean_stress_method, "value") else info.get("mean_stress_method", "goodman")
    measure_name = load_case.stress_measure.value if load_case and hasattr(load_case.stress_measure, "value") else info.get("stress_measure", "signed_von_mises")

    hotspot_part = info.get("part_name", "fixture")
    hotspot_node = info.get("node_id", -1)
    hotspot_coord = info.get("coordinates", [0.0, 0.0, 0.0])
    hotspot_region = info.get("hotspot_region_name", "고정체 목부 / 지대주 나사산")
    hotspot_stress = info.get("max_stress_MPa", 0.0)

    # 문헌 비교
    lit_html = ""
    if benchmark_comp is not None:
        lit_status = getattr(benchmark_comp, "status_display_text", "교차 검증 미실시")
        lit_ref = getattr(benchmark_comp, "literature_reference", "해당 없음")
        lit_val = getattr(benchmark_comp, "literature_fatigue_limit_N", None)
        pred_val = getattr(benchmark_comp, "predicted_fatigue_limit_N", 0.0)
        err_pct = getattr(benchmark_comp, "relative_error_pct", None)
        err_str = f"{err_pct:.1f}%" if err_pct is not None else "--"
        lit_val_str = f"{lit_val:.1f} N" if lit_val is not None else "데이터 부재"

        lit_html = f"""
        <div class="card">
            <h3>4. ISO 14801 문헌 벤치마크 교차 검증</h3>
            <table class="report-table">
                <tr><th>문헌 출처</th><td>{lit_ref}</td></tr>
                <tr><th>문헌 시험 피로한계 하중</th><td>{lit_val_str}</td></tr>
                <tr><th>FEA 예측 피로한계 하중 (FS=1.0)</th><td>{pred_val:.1f} N</td></tr>
                <tr><th>상대 오차</th><td>{err_str}</td></tr>
                <tr><th>검증 상태</th><td><strong>{lit_status}</strong></td></tr>
            </table>
        </div>
        """
    else:
        lit_html = """
        <div class="card">
            <h3>4. ISO 14801 문헌 벤치마크 교차 검증</h3>
            <p><strong>상태: 교차 검증 미실시 (Unverified)</strong></p>
            <p class="muted">동일 형상 및 경계조건에 대한 공인 문헌 시험 데이터가 연동되지 않아 교차 검증을 건너뛰었습니다.</p>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Dental FEA - 피로 해석 종합 평가 보고서</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #24292e;
        background-color: #f6f8fa;
        margin: 0;
        padding: 24px;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        padding: 32px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }}
    .header {{
        border-bottom: 2px solid #0366d6;
        padding-bottom: 16px;
        margin-bottom: 24px;
    }}
    .header h1 {{
        margin: 0 0 8px 0;
        font-size: 24px;
        color: #1f2328;
    }}
    .header .meta {{
        font-size: 13px;
        color: #57606a;
    }}
    .disclaimer-banner {{
        background-color: #fff8c5;
        border: 2px solid #d4a72c;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 24px;
        font-size: 14px;
        font-weight: bold;
        color: #735c0f;
    }}
    .card {{
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 18px;
        margin-bottom: 20px;
    }}
    .card h3 {{
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 16px;
        color: #0969da;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 6px;
    }}
    .report-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    .report-table th, .report-table td {{
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #e1e4e8;
    }}
    .report-table th {{
        background-color: #f6f8fa;
        width: 35%;
        color: #424a53;
    }}
    .fs-highlight {{
        font-size: 20px;
        font-weight: bold;
        color: {fs_color};
    }}
    .muted {{
        color: #6e7781;
        font-size: 12px;
    }}
    .footer {{
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid #d0d7de;
        font-size: 12px;
        color: #57606a;
        text-align: center;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>치과 임플란트 고주기 피로 해석 평가 보고서</h1>
        <div class="meta">발행 일시: {now_str} | Dental FEA Biomechanics System</div>
    </div>

    <div class="disclaimer-banner">
        {DISCLAIMER_TEXT}
    </div>

    <div class="card">
        <h3>1. 피로 평가 핵심 결과 (Summary)</h3>
        <table class="report-table">
            <tr>
                <th>최소 피로 안전계수 (Min FS)</th>
                <td><span class="fs-highlight">{fs_val:.4f}</span> ({fs_status_str})</td>
            </tr>
            <tr>
                <th>피로 파손 취약 부위 (Hotspot)</th>
                <td><strong>{hotspot_region}</strong> ({hotspot_part}, 절점 #{hotspot_node})</td>
            </tr>
            <tr>
                <th>핫스팟 3D 위치 (LPS mm)</th>
                <td>X: {hotspot_coord[0]:.2f}, Y: {hotspot_coord[1]:.2f}, Z: {hotspot_coord[2]:.2f} mm</td>
            </tr>
            <tr>
                <th>핫스팟 최대 정적 응력</th>
                <td>{hotspot_stress:.2f} MPa</td>
            </tr>
            <tr>
                <th>기준 피로 내구한도 ($S_e$)</th>
                <td>{mat_se:.1f} MPa</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h3>2. 해석 조건 및 적용 가정 (Assumptions)</h3>
        <table class="report-table">
            <tr><th>재료 명칭 및 규격</th><td>{mat_name} ({mat_std})</td></tr>
            <tr><th>재료 물성 출처</th><td>{mat_sources}</td></tr>
            <tr><th>정적 강도 물성</th><td>항복강도 $S_y = {mat_sy:.1f}$ MPa, 인장강도 $S_{{ut}} = {mat_sut:.1f}$ MPa</td></tr>
            <tr><th>주기 하중비 (Stress Ratio R)</th><td>R = {r_ratio:.2f}</td></tr>
            <tr><th>목표 설계 수명 (Target Cycles)</th><td>{target_cycles:.1e} cycles</td></tr>
            <tr><th>평균응력 보정 모델</th><td>{method_name.upper()} 보정식</td></tr>
            <tr><th>등가 응력 척도</th><td>{measure_name.upper()}</td></tr>
            <tr><th>구조적 역학 가정</th><td>미소 변형 선형 탄성, 완전 골유착(TIE) 계면 가정</td></tr>
        </table>
    </div>

    {lit_html}

    <div class="card">
        <h3>5. 임상 가이드라인 및 주의사항</h3>
        <ul style="font-size: 13px; color: #424a53; padding-left: 20px;">
            <li>피로 안전계수가 1.0 미만인 경우 저작 사이클 누적에 따른 임플란트 파절 또는 지대주 나사 풀림/파단 위험이 있습니다.</li>
            <li>직경이 굵은 임플란트 선택, 식립 각도 조정, 교합 하중 완화(Occlusal adjustment) 등의 설계 변경을 고려하십시오.</li>
            <li>본 보고서는 유한요소해석 기반 예측치이므로 실제 구강 내 생화학적 부식 피로(Corrosion fatigue) 환경에 따라 거동 차이가 발생할 수 있습니다.</li>
        </ul>
    </div>

    <div class="footer">
        Dental FEA System &copy; 2026. All rights reserved. 본 문서는 컴퓨터 시뮬레이션 기반 보고서입니다.
    </div>
</div>
</body>
</html>
"""
    return html


def generate_fatigue_report_text(
    rs: ResultSet,
    model: Optional[MeshModel] = None,
    material: Optional[FatigueMaterial] = None,
    load_case: Optional[FatigueLoadCase] = None,
    hotspot_info: Optional[Dict[str, Any]] = None,
    benchmark_comp: Optional[Any] = None,
) -> str:
    """Generate plain text summary of the fatigue report."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = hotspot_info or getattr(rs, "fatigue_hotspot_info", None) or {}
    min_fs = info.get("min_safety_factor", 0.0)

    lines = [
        "=" * 70,
        " DENTAL FEA - HIGH-CYCLE FATIGUE EVALUATION REPORT",
        "=" * 70,
        f" Date: {now_str}",
        "-" * 70,
        f" {DISCLAIMER_TEXT}",
        "-" * 70,
        " [1. SUMMARY]",
        f" - Min Fatigue Safety Factor (FS): {min_fs:.4f} ({'SAFE' if min_fs >= 1.0 else 'FAILURE RISK'})",
        f" - Hotspot Region: {info.get('hotspot_region_name', 'Fixture Neck')}",
        f" - Hotspot Part & Node: {info.get('part_name', 'fixture')} (#{info.get('node_id', -1)})",
        f" - Hotspot Coordinates: {info.get('coordinates', [0, 0, 0])} mm",
        f" - Max Stress at Hotspot: {info.get('max_stress_MPa', 0.0):.2f} MPa",
        "-" * 70,
        " [2. ASSUMPTIONS & CONDITIONS]",
        f" - Material: {material.name if material else info.get('material_name', '')} ({material.standard if material else ''})",
        f" - Material Sources: {', '.join(material.sources) if material and material.sources else 'materials_fatigue.yaml'}",
        f" - Stress Ratio R: {load_case.stress_ratio_R if load_case else info.get('stress_ratio_R', 0.1)}",
        f" - Target Cycles: {load_case.target_cycles if load_case else info.get('applied_cycles', 5e6):.1e}",
        f" - Mean Stress Method: {load_case.mean_stress_method if load_case else info.get('mean_stress_method', 'goodman')}",
        f" - Stress Measure: {load_case.stress_measure if load_case else info.get('stress_measure', 'signed_von_mises')}",
        "=" * 70,
    ]
    return "\n".join(lines)
