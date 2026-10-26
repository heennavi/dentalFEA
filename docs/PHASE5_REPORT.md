# Phase 5 피로 해석 정량 검증 및 종합 결과 보고서 (회귀 테스트 및 동결)

## 1. 개요
본 보고서는 치과 임플란트 유한요소해석(FEA) 시스템의 **Phase 5 고사이클 피로 해석(High-Cycle Fatigue) 및 ISO 14801 표준 동적 피로 해석 파이프라인의 수치적 정확성 정량 검증, 외팔보/문헌 벤치마크 기준값 동결, 알려진 해석 한계 및 임상 사용자를 위한 가이드라인**을 정리한 종합 문서입니다.

Step 5.8 단계에 따라 Step 5.3(외팔보 굽힘 피로 벤치마크) 및 Step 5.6(ISO 14801 임플란트 피로 벤치마크)의 정량 결과를 기준값(Baseline Golden Values)으로 저장 및 동결하였으며, Phase 1~4 및 Phase 5 전 스위트에 걸친 회귀 검증을 완료하였습니다.

---

## 2. Step 5.3 외팔보 벤치마크 정량 검증 결과 및 기준값 동결

탄성보 굽힘 이론($\sigma = M c / I$)과 수계산 평균응력 보정 안전계수를 기준으로, CalculiX C3D10 2차 사면체 메쉬 기반 2중 하중($F_{\max}, F_{\min}$) FEA 시뮬레이션을 수행하여 정량 비교 검증하였습니다.

### [기준 사양 및 물성]
- **보 규격**: 길이 $L = 100\text{ mm}$, 폭 $b = 10\text{ mm}$, 높이 $h = 10\text{ mm}$
- **단면 특성**: $I = 833.333\text{ mm}^4$, $Z = 166.667\text{ mm}^3$
- **재료 물성 (Ti-6Al-4V)**: $E = 110,000\text{ MPa}$, $\nu = 0.34$, $S_y = 795\text{ MPa}$, $S_{ut} = 860\text{ MPa}$, $S_e = 510\text{ MPa}$ ($R=-1$ 기준)
- **주기 하중**: $F_{\max} = 1000\text{ N}$, $F_{\min} = 100\text{ N}$ (응력비 $R = 0.1$)
- **구속 조건**: 푸아송 구속 특이점을 배제한 등정적(Isostatic 3-2-1) 운동학적 고정 조건

### [정량 검증 결과 및 동결 기준값]

| 물리량 및 평가 지표 | 이론해 (Analytical) | CalculiX 실측치 (Mean) | 상대 오차 (%) | 판정 기준 | 최종 결과 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **최대 인장 굽힘응력 ($\sigma_{\max}$)** | $600.00\text{ MPa}$ | $600.96\text{ MPa}$ | **$0.45\%$** (최대 $1.19\%$) | $\le 3.0\%$ | **PASS** |
| **최소 인장 굽힘응력 ($\sigma_{\min}$)** | $60.00\text{ MPa}$ | $60.10\text{ MPa}$ | **$0.45\%$** (최대 $1.19\%$) | $\le 3.0\%$ | **PASS** |
| **응력 진폭 ($\sigma_a$)** | $270.00\text{ MPa}$ | $270.43\text{ MPa}$ | **$0.45\%$** (최대 $1.19\%$) | $\le 3.0\%$ | **PASS** |
| **평균 응력 ($\sigma_m$)** | $330.00\text{ MPa}$ | $330.53\text{ MPa}$ | **$0.45\%$** (최대 $1.19\%$) | $\le 3.0\%$ | **PASS** |
| **Goodman 피로 안전계수 ($FS$)** | $1.0951$ | $1.0934$ | **$0.45\%$** (최대 $1.17\%$) | $\le 3.0\%$ | **PASS** |
| **Soderberg 피로 안전계수 ($FS$)** | $1.0588$ | $1.0571$ | **$0.45\%$** (최대 $1.17\%$) | $\le 3.0\%$ | **PASS** |
| **Gerber 피로 안전계수 ($FS$)** | $1.3682$ | $1.3661$ | **$0.45\%$** (최대 $1.17\%$) | $\le 3.0\%$ | **PASS** |

- **저장 파일**: `tests/fixtures/fatigue_benchmark_5_3_baseline.json`
- **동결 판정**: 이론해 대비 전 응력 및 안전계수 평균 오차 $< 0.5\%$, 최대 오차 $\le 1.19\%$로 허용치($\pm 3.0\%$) 이내 정합성 동결 완료.

---

## 3. Step 5.6 ISO 14801 벤치마크 정량 검증 및 기준값 동결

치과 임플란트의 국제 표준 피로 시험 규격인 **ISO 14801:2016**의 기하학적 셋업, 임상 호발 파절 부위(핫스팟) 식별, 예측 피로 한도 하중 및 공인 문헌 비교 체계를 정량 검증하였습니다.

### [기하학적 셋업 사양 기준값]
1. **하중 경사각 (Loading Angle)**: 공칭 $30.0^\circ$ (허용 오차 $\pm 2.0^\circ$, 범위: $28.0^\circ \sim 32.0^\circ$)
2. **모멘트 레버 암 (Lever Arm)**: 고정 평면으로부터 하중 중심까지 $11.0\text{ mm}$ (허용 오차 $\pm 0.5\text{ mm}$)
3. **골 흡수 노출 깊이 (Resorption Depth)**: 고정 평면에서 공칭 골 연(Bone crest)까지 $3.0\text{ mm}$ (허용 오차 $\pm 0.2\text{ mm}$)
4. **응력비 (Stress Ratio $R$)**: $R = 0.10$ (허용 오차 $\pm 0.02$)

### [임상 핫스팟 및 문헌 교차 검증 판정 체계]
- **핫스팟 유효 판정**:
  - 체결 나사(Abutment Screw) 또는 골 흡수 경계 직상부 고정체 목부(Fixture Neck / Collar, $z \le 3.0\text{ mm}$)에서 최소 안전계수가 검출될 시 `is_clinically_valid = True`로 자동 판정.
- **문헌 벤치마크 기준치**:
  - `gehrke_2020_conical` (Ti-6Al-4V Conical Connection, $5\times 10^6$ cycles): 문헌 피로 한도 = $316.0\text{ N}$
- **판정 허용 등급**:
  - 오차 $\le 30\%$: **양호(GOOD)** ("수십 % 이내 판정 기준 만족")
  - 오차 $> 30\%$: **재점검 권장(RECHECK_NEEDED)** ("접촉면, 선하중, 물성 재확인 안내")
  - 공개 문헌 부재 시: **정직한 한계 고지(UNVERIFIED)** ("교차 검증 미실시 (공개 자료 부재)")
- **저장 파일**: `tests/fixtures/fatigue_iso14801_5_6_baseline.json`

---

## 4. Phase 5 구현 아키텍처 및 핵심 모듈 요약

| 모듈 경로 | 주요 역할 및 기능 |
| :--- | :--- |
| `fatigue/models.py` | 피로 물성(`FatigueMaterial`), 하중 케이스(`FatigueLoadCase`), 절점 결과(`FatigueResult`) 정의 |
| `fatigue/stress_measures.py` | 부호 있는 폰 미세스(Signed von Mises), 최대 주응력, Sines 다축 응력 척도 계산 |
| `fatigue/mean_stress.py` | Goodman, Soderberg, Gerber, Morrow, Smith-Watson-Topper(SWT) 평균응력 보정 |
| `fatigue/sn_curve.py` | 바스퀸(Basquin) 멱법칙, 다구간 선형 보간 기반 피로 수명($N_f$) 및 누적 손상($D$) 계산 |
| `fatigue/evaluate.py` | 0-based 전역 절점 피로 안전계수, 수명, 손상률 고속 넘파이 벡터 연산 파이프라인 |
| `fatigue/io_frd.py` | CalculiX 2중 하중 `.frd` 파일로부터 주기 하중 응력 텐서 추출 |
| `fatigue/cantilever_benchmark.py` | 외팔보 해석해 및 CalculiX 자동 유한요소 피로 벤치마크 파이프라인 |
| `fatigue/iso14801.py` | ISO 14801 기하 검증, 표준 3파트 어셈블리 생성, 핫스팟 식별, 피로 한도 역산 및 문헌 비교 |
| `dental_fea_app/ui/fatigue_panel.py` | 탭 7 피로 전용 패널: S-N 곡선/Haigh 선도, 핫스팟 요약, 고지 문구 배너, 리포트 내보내기 |
| `dental_fea_app/ui/fatigue_diagrams.py` | CPU 오프스크린(`FigureCanvasAgg`) 렌더링 기반 충돌 없는 안전한 다이어그램 생성 |
| `ui/result_view_controller.py` | 3D VTK 뷰어 내 피로 5종 모수(FS, $N_f$, $D$, $\sigma_a$, $\sigma_m$) 동적 스칼라 필드 전환 및 핫스팟 마커 |

---

## 5. 알려진 해석 한계 및 임상 고지 사항 (Clinical Disclaimer)

1. **임상 판단 보조용 고지**:
   - 본 소프트웨어의 피로 안전계수 및 수명 예측 결과는 임상적 의사결정을 보조하기 위한 참고 지표이며, 최종적인 제품 인증 및 진단은 반드시 공인 시험기관의 물리적 시험(ISO 14801 시험 등) 및 전문의의 판단에 따라야 합니다.
2. **절점 응력 외삽 및 다축 피로 가정**:
   - .frd 응력은 적분점에서 외삽된 값이며, 복잡한 다축 응력 상태(Multiaxial stress state)를 1축 등가 파라미터(Signed von Mises 등)로 변환하여 평가하므로 실제 3차원 비비례 하중 경로(Non-proportional loading)에서는 오차가 발생할 수 있습니다.
3. **환경적 요인 미반영**:
   - 체액 환경(Saline/PBS)에 의한 부식 피로(Corrosion fatigue) 및 생물학적 골 개조(Bone remodeling) 반응은 고려되지 않으며, 상온 공기 중 피로 물성을 기반으로 평가됩니다.

---

## 6. 회귀 테스트 및 동결 결과 (Regression Verification)

- **테스트 일시**: 2026-10-26
- **실행 환경**: Linux x86_64, Python 3.12, PySide6, VTK 9.x, CalculiX ccx 2.21
- **회귀 테스트 스위트**:
  - Phase 1 (DICOM I/O, 볼륨 렌더링, MPR 뷰어): **PASS**
  - Phase 2 (분할, ROI, 표면 메쉬 생성): **PASS**
  - Phase 3 (Gmsh 체적 격자, 재료 매핑, CalculiX INP 생성 및 솔버 러너): **PASS**
  - Phase 4 (고정폭 .frd 파서, 응력/변형률/반력/주변골 통계, 3D 뷰어, 리포트): **PASS**
  - Phase 5 (피로 물성, 외팔보 벤치마크, ISO 14801 파이프라인, UI 연동, 동결 기준값 검증): **PASS**
- **동결 상태**: `tests/fixtures/fatigue_benchmark_5_3_baseline.json` 및 `tests/fixtures/fatigue_iso14801_5_6_baseline.json` 기반의 회귀 불변성 확인 완료.
