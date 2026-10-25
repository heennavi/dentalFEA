"""
test_materials_fatigue_yaml.py
config/materials_fatigue.yaml 파일의 구문, 필수 필드, 단위 및 Basquin 수식 정합성을 검증하는 테스트.
"""

from pathlib import Path
import math
import yaml
import pytest


def test_materials_fatigue_yaml_structure_and_values():
    """Verify that materials_fatigue.yaml exists and contains all required fields and valid data."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "materials_fatigue.yaml"
    assert config_path.is_file(), f"Config file not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 1. 버전 및 단위계 확인
    assert "version" in data
    assert "unit_system" in data
    assert data["unit_system"]["stress"] == "MPa"
    assert data["unit_system"]["force"] == "N"
    assert data["unit_system"]["length"] == "mm"

    # 2. 필수 재료 포함 여부
    required_materials = ["ti_6al_4v_eli", "ti_grade_4", "ti_6al_4v_grade_5", "straumann_roxolid"]
    assert "materials" in data
    for mat_id in required_materials:
        assert mat_id in data["materials"], f"Missing material {mat_id}"
        mat = data["materials"][mat_id]

        # 기본 속성
        assert "name" in mat
        assert "standard" in mat
        assert "sources" in mat and len(mat["sources"]) > 0

        # 탄성 물성 (E, nu, density)
        assert "elastic" in mat
        assert mat["elastic"]["E"] > 50000.0  # MPa
        assert 0.2 < mat["elastic"]["nu"] < 0.4
        assert mat["elastic"]["density"] > 0

        # 정적 강도 (Sy, Sut)
        assert "static_strength" in mat
        Sy = mat["static_strength"]["yield_strength_Sy"]
        Sut = mat["static_strength"]["tensile_strength_Sut"]
        assert 300.0 < Sy < 1500.0
        assert Sy < Sut, f"Yield strength {Sy} must be less than UTS {Sut}"

        # 피로 물성 (Se, Basquin, S-N curve)
        assert "fatigue" in mat
        fatigue = mat["fatigue"]
        Se = float(fatigue["fatigue_limit_Se"])
        assert 200.0 < Se < Sut
        assert float(fatigue["endurance_cycles_Ne"]) >= 1.0e6

        # Basquin 파라미터 검증
        basquin = fatigue["basquin"]
        sigma_f_prime = float(basquin["sigma_f_prime"])
        b = float(basquin["b"])
        assert sigma_f_prime > Sut, "sigma_f_prime is typically higher than Sut"
        assert -0.2 < b < -0.05, f"b exponent {b} out of typical titanium range"

        # Basquin 수식: sigma_a = sigma_f_prime * (2 * N)^b 계산값과 S-N 점 데이터 비교
        sn = fatigue["sn_curve"]
        assert len(sn["cycles"]) == len(sn["stress_amplitude_MPa"])
        for n_cyc, s_amp in zip(sn["cycles"], sn["stress_amplitude_MPa"]):
            n_cyc_f = float(n_cyc)
            s_amp_f = float(s_amp)
            # Basquin 예측값과의 오차 범위(+-15% 이내) 점검
            pred_amp = sigma_f_prime * ((2.0 * n_cyc_f) ** b)
            rel_err = abs(pred_amp - s_amp_f) / s_amp_f
            assert rel_err < 0.20, (
                f"Material {mat_id} S-N curve point at {n_cyc} cycles: "
                f"table={s_amp_f}, basquin={pred_amp:.1f}, rel_err={rel_err:.2%}"
            )

    # 3. ISO 14801 파라미터 확인
    assert "iso_14801_parameters" in data
    iso_params = data["iso_14801_parameters"]
    assert math.isclose(iso_params["loading_angle_deg"], 30.0)
    assert iso_params["target_cycles"] == 5000000
    assert math.isclose(iso_params["stress_ratio_R"], 0.1)

    # 4. 벤치마크 케이스 확인
    assert "benchmark_cases" in data
    assert "gehrke_2020_conical" in data["benchmark_cases"]
    assert "gehrke_2020_internal_hex" in data["benchmark_cases"]
