# Phase 4 컨텍스트 및 아키텍처 기준 문서 (docs/PHASE4_CONTEXT.md)

> **목적**: Phase 3 완료 시점의 실제 코드베이스를 분석하여 Phase 4(결과 파일 파싱, von Mises/주응력/변위 계산, VTK 3D 시각화, 슬라이스 매핑, 통계/프로브 및 리포트 내보내기)가 접목될 정확한 인터페이스와 데이터 계약을 고정한다. 이후 Phase 4의 모든 하위 Step은 본 문서를 기준으로 개발한다.

---

## 1. 실제 디렉터리 구조 (초기 계획 대비 현황)

현재 프로젝트는 Phase 1(2D MPR 뷰어, 3D 볼륨 뷰어), Phase 2(골 분할, 임플란트 식립, Boolean 절제), Phase 3(Gmsh 3D 체적 메싱, CT HU 물성 매핑, 하중/경계조건, CalculiX .inp 및 비동기 솔버 연동)이 완전히 구현되어 테스트를 통과한 상태이다.

```text
dental_FEA/
├── AGENTS.md                             # Phase 4 공통 규칙 및 핵심 설계 결정
├── requirements.txt                      # 의존성 목록
├── run.py                                # 데스크톱 GUI 진입 편의 스크립트
├── .venv/                                # 프로젝트 공식 가상환경 (Python 3.12.3)
├── asset/                                # 임플란트 및 지대주 파라메트릭 라이브러리 (VTP, JSON)
├── docs/
│   ├── PHASE3_CONTEXT.md                 # Phase 3 아키텍처 기준 문서
│   ├── PHASE3_PLAN.md                    # Phase 3 프롬프트 계획서
│   ├── PHASE3_REPORT.md                  # Phase 3 정량 검증 리포트
│   ├── PHASE4_CONTEXT.md                 # [본 문서] Phase 4 컨텍스트 기준 문서
│   └── PROJECT_MAP.md                    # 초기 프로젝트 구조도
├── fea/                                  # FEA 데이터 모델 및 코어 로직 (Headless)
│   ├── __init__.py
│   ├── models.py                         # MeshModel, MeshPart, MaterialAssignment, TieDef 등
│   ├── results.py                        # [신규] FrdBlock, FrdData, ResultSet (Step 4-0 고정 계약)
│   ├── boundary_conditions.py            # 경계조건 및 하중 생성/검증 순수 로직
│   └── workers.py                        # QThread 기반 비동기 메싱 워커 (MeshWorker)
├── meshing/                              # Gmsh 기반 체적 사면체 메싱 및 품질 검사 (Headless)
│   ├── __init__.py
│   ├── gmsh_mesher.py                    # build_part_mesh, gmsh_tet10_to_ccx 등
│   ├── quality_checker.py                # 야코비안 및 형상비 검사
│   └── surface_extractor.py              # 접촉면 및 표면 절점 추출
├── material/                             # CT HU 기반 골 물성 매핑 (Headless)
│   ├── __init__.py
│   ├── bone_mapping.py                   # 경험식 기반 비균질 탄성계수 할당
│   └── presets.json                      # HU-밀도-탄성계수 관계식 프리셋
├── solver/                               # CalculiX 입출력 및 프로세스 제어 (Headless)
│   ├── __init__.py
│   ├── calculix_writer.py                # CalculiX .inp 작성기 및 GlobalIndexMapper
│   └── runner.py                         # QProcess 기반 비동기 ccx 실행기 (CalculixRunner)
├── postprocess/                          # [Phase 4 주 작업 영역] 결과 파싱 및 후처리
│   ├── __init__.py
│   └── result_reader.py                  # 기존 Phase 3 기초 파서 (Step 4-1에서 확장/대체 예정)
├── saved/                                # 해석 산출물 기본 저장 폴더
│   ├── dental_fea_job.inp                # 생성된 CalculiX 입력 파일
│   ├── dental_fea_job.frd                # 실제 ccx 해석 결과 파일 (128 MB)
│   ├── dental_fea_job.sta                # 해석 상태 파일
│   ├── dental_fea_job.dat                # 해석 텍스트 출력
│   └── spooles.out                       # SPOOLES 솔버 로그
├── scripts/
│   └── run_pipeline_headless.py          # 전체 파이프라인 무헤드 CLI 배치 실행 스크립트
├── tests/
│   ├── conftest.py                       # pytest 공통 픽스처
│   ├── fixtures/                         # [신규] 테스트용 합성/실제 픽스처
│   │   ├── cube_compression.inp         # 10mm 정육면체 압축 해석 입력
│   │   └── cube_compression.frd         # 실제 ccx가 생성한 소형 .frd 샘플 (5.7 KB)
│   ├── test_models.py                    # FEA 데이터 모델 테스트
│   ├── test_results_models.py            # [신규] fea/results.py 계약 테스트
│   ├── test_calculix_writer.py           # .inp 생성기 및 ccx 실행 검증
│   └── ... (총 189개 테스트 전체 통과)
├── dental_fea_app/                       # 메인 애플리케이션 GUI 패키지 (PySide6 + VTK)
│   ├── config.py                         # 단위계, 색상 프리셋, 전역 설정
│   ├── main.py                           # 앱 엔트리포인트
│   ├── core/
│   │   ├── state_manager.py              # 전역 상태 관리자 (StateManager)
│   │   ├── settings_manager.py           # 설정 영속화
│   │   └── events.py                     # 이벤트 버스
│   ├── imaging/                          # DICOM 로더 및 볼륨 변환
│   ├── segmentation/                     # 골 분할 알고리즘
│   ├── placement/                        # 임플란트 식립 관리 엔진 및 2D/3D 기즈모
│   ├── geometry/                         # Boolean 연산 및 watertight 처리
│   └── ui/
│       ├── main_window.py                # 메인 윈도우
│       ├── control_panel.py              # 좌측 워크플로 탭 컨테이너
│       ├── volume_viewer_window.py       # 독립 3D 뷰어 윈도우 (VTK 렌더러)
│       ├── mpr_container.py              # 2D MPR 슬라이스 컨테이너
│       ├── fea_panel.py                  # "5. 해석" 탭 UI
│       └── fea_controller.py             # UI와 FEA 백엔드 연결 컨트롤러
└── ui/                                   # 루트 re-export 래퍼 (dental_fea_app/ui 바인딩)
```

### 초기 계획 대비 변경/특이 사항
1. **루트 패키지와 내부 패키지 구조**:
   `fea/`, `meshing/`, `material/`, `solver/`, `postprocess/`는 GUI 없이 단독 import 가능한 독립 패키지로 프로젝트 루트에 위치하며, 애플리케이션 프런트엔드(`dental_fea_app/`) 및 테스트(`tests/`)에서 직접 임포트하여 사용한다.
2. **Phase 4 모듈 격리 원칙**:
   `postprocess/` 및 `fea/results.py`는 순수 수치/파싱 모듈로 유지되며, PySide6나 VTK 렌더링 모듈(QVTKRenderWindowInteractor 등)을 임포트하지 않는다. VTK 데이터셋(`vtkUnstructuredGrid`) 변환은 Step 4-3에서 `postprocess/vtk_builder.py` 한 곳에서만 격리 구현한다.

---

## 2. StateManager의 FEA 슬롯 이름, 접근 방법 및 last_frd_path 존재 여부

`StateManager`(`dental_fea_app/core/state_manager.py`)에 Phase 3 FEA 파이프라인 슬롯이 이미 구현되어 있으며, `last_frd_path` 슬롯 또한 완벽하게 존재한다.

| 데이터 종류 | 내부 변수명 | Getter 프로퍼티 | Setter 메서드 | 변경 알림 Signal |
| :--- | :--- | :--- | :--- | :--- |
| **MeshModel** | `_mesh_model` | `state_mgr.mesh_model` | `state_mgr.set_mesh_model(model)` | `mesh_model_changed(object)` |
| **MaterialAssignment** | `_material_assignment` | `state_mgr.material_assignment` | `state_mgr.set_material_assignment(mat)` | `material_assignment_changed(object)` |
| **BoundaryConditionSet** | `_bc_set` | `state_mgr.bc_set` | `state_mgr.set_bc_set(bc)` | `bc_set_changed(object)` |
| **List[TieDef]** | `_ties` | `state_mgr.ties` | `state_mgr.set_ties(ties_list)` | `ties_changed(object)` |
| **최종 .frd 파일 경로** | `_last_frd_path` | `state_mgr.last_frd_path` | `state_mgr.set_last_frd_path(path)` | `last_frd_path_changed(str)` |
| **FEA 데이터 무효화 플래그** | `_fea_is_stale` | `state_mgr.fea_is_stale` | `state_mgr.set_fea_stale(stale)` | `fea_stale_changed(bool)` |

- **접근 방법 예시**:
  ```python
  # Getter
  model: Optional[MeshModel] = state_mgr.mesh_model
  frd_path: Optional[str] = state_mgr.last_frd_path

  # Setter
  state_mgr.set_last_frd_path("/path/to/job.frd")

  # Signal 구독
  state_mgr.last_frd_path_changed.connect(on_frd_ready)
  ```
- **상류 변경 시 동작**: 임플란트 식립 위치, 골 분할 마스크, 불리언 기하 변경 시 `state_mgr.invalidate_fea_pipeline()`이 자동 호출되어 `fea_stale_changed(True)`를 발생시킨다.

---

## 3. MeshPart.elements 의 10절점 요소 저장 순서 (Gmsh vs CalculiX/VTK)

- **결론**: **`MeshPart.elements`는 생성 시점에 이미 CalculiX C3D10 순서(= VTK 셀 타입 24 순서)로 변환되어 저장되어 있다.**
- **판단 근거**:
  1. `meshing/gmsh_mesher.py`의 `build_part_mesh()` 내부 (530~532행):
     ```python
     if order == 2:
         elements = gmsh_tet10_to_ccx(elements)  # 마지막 두 중간 절점(8번과 9번) 교환
         element_type = "C3D10"
     ```
  2. `solver/calculix_writer.py`의 `write_inp()` 시그니처 (153, 224행):
     ```python
     def write_inp(..., apply_ccx_ordering: bool = False):
         ...
         if elem_type == "C3D10" and apply_ccx_ordering:
             elems = gmsh_tet10_to_ccx(elems)
     ```
     `apply_ccx_ordering`의 기본값이 `False`인 이유가 바로 `MeshPart`에 이미 CalculiX 순서로 저장되어 있기 때문임.
- **Phase 4 VTK 생성 시 주의사항 (Step 4-3)**:
  - Gmsh 원본: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
  - CalculiX C3D10 = VTK Cell Type 24 (`VTK_QUADRATIC_TETRA`): `[0, 1, 2, 3, 4, 5, 6, 7, 9, 8]`
  - 따라서 이미 CalculiX 순서로 저장된 `MeshPart.elements`를 `vtkUnstructuredGrid`의 셀로 변환할 때는 **추가 순서 교환 없이 그대로 투입**해야 한다. 만약 다시 교환하면 Gmsh 순서로 돌아가 음의 야코비안 왜곡이 발생한다.

---

## 4. 전역 절점 번호 변환 함수 및 매퍼 위치

- **위치**: `solver/calculix_writer.py` 내의 `GlobalIndexMapper` 클래스 및 `fea/models.py` 내의 `MeshModel` 메서드.
- **클래스 및 메서드 명세**:
  - `GlobalIndexMapper(model: MeshModel)`:
    - `mapper.node_id(part_name: str, local_idx: int) -> int`: 특정 파트의 0-based 로컬 절점 인덱스를 1-based 전역 고유 절점 ID로 변환.
    - `mapper.node_ids(part_name: str, local_indices: Sequence[int]) -> np.ndarray`: 로컬 절점 인덱스 배열 $\to$ 1-based 전역 절점 ID 배열 (`int64`).
    - `mapper._node_offsets[part_name]`: 해당 파트의 전역 0-based 시작 인덱스.
  - `MeshModel.global_node_offset(part_name: str) -> int`:
    - 해당 파트의 전역 0-based 절점 시작 오프셋 반환 (`offset = sum(len(p.nodes) for p in parts[:idx])`).
- **전역/로컬 인덱스 변환 공식**:
  - $\text{전역 0-based 인덱스} = \text{로컬 0-based 인덱스} + \text{offset}$
  - $\text{전역 1-based ID (CalculiX .inp 및 .frd)} = \text{전역 0-based 인덱스} + 1$
  - $\text{로컬 0-based 인덱스} = \text{전역 0-based 인덱스} - \text{offset}$ (단, $\text{offset} \le \text{idx} < \text{offset} + N_{\text{part}}$)

---

## 5. calculix_writer.py 의 *NODE FILE 과 *EL FILE 출력 요청 현황

`solver/calculix_writer.py`의 `write_inp()` 함수 내 출력 카드 요청 목록:

```text
*NODE FILE
U, RF
*EL FILE
S, E
*END STEP
```

- **변경 이력**: Step 4-0 요구사항에 따라 `*NODE FILE`에 반력 `RF`를 추가하여 `"U, RF"`로 갱신함 (기존: `"U"`).
- **.frd 결과 블록 매핑**:
  - `*NODE FILE` $\to$ `U`는 `DISP` 블록 (D1, D2, D3, ALL), `RF`는 `FORC` 블록 (F1, F2, F3, ALL)으로 출력.
  - `*EL FILE` $\to$ `S`는 `STRESS` 블록 (SXX, SYY, SZZ, SXY, SYZ, SZX), `E`는 `TOSTRAIN` 블록 (EXX, EYY, EZZ, EXY, EYZ, EZX)으로 절점 외삽되어 출력.

---

## 6. CalculixRunner 시그널 시그니처 및 완료 시 연결 흐름

- **위치**: `solver/runner.py` (`CalculixRunner`)
- **시그널 목록**:
  - `started = Signal()`
  - `progress = Signal(int, str)`: `(percent: int, message: str)`
  - `log_line = Signal(str)`: 표준 출력/에러 라인 스트림
  - `finished = Signal(bool, str, str)`: `(success: bool, frd_path: str, message: str)`
  - `error = Signal(str)`
- **해석 완료 시 연결 흐름**:
  1. `CalculixRunner._on_process_finished()`에서 해석 성공 및 `.frd` 생성 확인 후 `self.finished.emit(True, self._frd_path, "해석이 성공적으로 완료되었습니다.")` 발생.
  2. `FeaController`(`dental_fea_app/ui/fea_controller.py`)의 `_on_solver_finished`에 연결:
     ```python
     def _on_solver_finished(self, success: bool, frd_path: str, msg: str):
         if success and frd_path:
             self.state_mgr.set_last_frd_path(frd_path)
         self.solver_finished.emit(success, frd_path, msg)
     ```
  3. `StateManager.set_last_frd_path()`가 호출되면서 전역으로 `last_frd_path_changed(str)` 시그널이 브로드캐스트됨.
  4. Phase 4에서 결과 뷰어 및 결과 패널(Step 4-6)은 `state_mgr.last_frd_path_changed`를 구독하여 해석 완료 시 자동으로 `.frd` 파싱 및 시각화를 트리거한다.

---

## 7. 작업 폴더 및 Job 이름 규칙 (.frd 생성 위치)

- **기본 디렉터리**: `<프로젝트 루트>/saved/`
- **Job 파일 명명 규칙**:
  - 기본 입력 파일: `saved/dental_fea_job.inp`
  - 작업명(`job_name`): `dental_fea_job`
- **결과 파일 생성 규칙**:
  - CalculiX는 작업 디렉터리를 `os.path.dirname(abs_inp)`(`saved/`)로 설정하고 `ccx -i dental_fea_job`으로 실행됨.
  - 따라서 모든 솔버 출력 파일은 입력 파일과 동일한 디렉터리에 생성됨:
    - `saved/dental_fea_job.frd` (결과 필드 파일)
    - `saved/dental_fea_job.sta` (스텝 수렴 상태)
    - `saved/dental_fea_job.dat` (요약 텍스트)
    - `saved/dental_fea_job.cvg` (반복 수렴 곡선)
    - `saved/spooles.out` (행렬 솔버 로그)

---

## 8. 3D 뷰어 창(VolumeViewerWindow)의 액터 제어 및 렌더 갱신 API

- **위치**: `dental_fea_app/ui/volume_viewer_window.py` (`VolumeViewerWindow`)
- **렌더러 접근**:
  - `self.renderer`: 메인 `vtk.vtkRenderer` 인스턴스.
- **액터 조작 API**:
  - 액터 추가: `self.renderer.AddActor(actor)`
  - 액터 제거: `self.renderer.RemoveActor(actor)`
  - 볼륨 추가/제거: `self.renderer.AddVolume(volume)`, `self.renderer.RemoveVolume(volume)`
  - 가시성 제어: `actor.SetVisibility(True / False)`
  - 컬러바(ScalarBar) 추가: `self.renderer.AddActor2D(scalar_bar_actor)`
- **렌더 갱신 호출 방법**:
  - `self.request_render()`: 60ms `QTimer` 스로틀링을 거쳐 호출 (연속 슬라이더 조작이나 마우스 드래그 시 추천).
  - `self.render()`: 즉시 VTK 렌더 파이프라인 강제 실행 (`self.vtkWidget.GetRenderWindow().Render()`).
  - 전역 렌더 요청: `state_mgr.request_render_all()` 호출 시 `state_mgr.render_all_requested` 시그널을 통해 `volume_viewer_window.request_render()`가 자동 호출됨.

---

## 9. 좌측 제어 패널(ControlPanel)에 새 탭 추가 방법

- **위치**: `dental_fea_app/ui/control_panel.py` (`ControlPanel`)
- **탭 컨테이너**: `self.workflow_tabs = QTabWidget()` (`_create_workflow_tabs()` 메서드)
- **현재 탭 구성**:
  1. `0: 1. 영상`: DICOM 정보
  2. `1: 2. 분할`: `SegmentationPanel`
  3. `2: 3. 임플란트`: `ImplantPanel`
  4. `3: 4. 기하 처리`: `GeometryPanel`
  5. `4: 5. 해석`: `FeaPanel`
- **새 탭 추가 절차**:
  1. `_create_workflow_tabs()` 내부에서 결과 패널 인스턴스 생성 및 탭 등록:
     ```python
     self.postprocess_panel = PostprocessPanel(self.state_mgr, self)
     tabs.addTab(self.postprocess_panel, "6. 결과")
     ```
  2. 초기 비활성화 설정: `tabs.setTabEnabled(5, False)`
  3. 활성화 트리거: 해석 완료(`last_frd_path_changed`) 또는 결과 파일 로드 시 `tabs.setTabEnabled(5, True)` 및 `tabs.setCurrentIndex(5)` 호출.

---

## 10. 설치된 패키지 버전 현황

프로젝트 가상환경(`./.venv/bin/python`) 검사 결과:

| 패키지 | 설치 버전 | 용도 및 상태 |
| :--- | :--- | :--- |
| `vtk` | **9.7.1** | 3D 시각화, 폴리데이터, 언스트럭처드 그리드 |
| `pyvista` | **미설치** | (VTK 직접 사용으로 불필요) |
| `numpy` | **2.5.4** | 고속 텐서 연산, von Mises 및 주응력 계산 |
| `scipy` | **1.18.1** | 고유치 분해(주응력), k-d tree 탐색 |
| `PySide6` | **6.12.0** | GUI 위젯 및 비동기 스레드/프로세스 |
| `gmsh` | **4.15.2** | 3D 체적 사면체 메싱 |

---

## 11. 실제 ccx 생성 .frd 샘플 확보 현황

1. **소형 테스트 픽스처 (자동 생성 완료)**:
   - 경로: `tests/fixtures/cube_compression.frd` (크기: 5,722 bytes, 8개 절점, 5개 C3D4 요소)
   - 입력 파일: `tests/fixtures/cube_compression.inp`
   - 생성 방식: CalculiX 2.21(`/usr/bin/ccx`)을 직접 실행하여 생성.
   - 포함 블록: `DISP` (변위), `STRESS` (응력 6성분), `TOSTRAIN` (변형률 6성분), `FORC` (반력 RF 3성분).
   - 용도: Step 4-1 파서 및 수치 계산의 초고속 경량 유닛 테스트용.
2. **풀 스케일 임상 시뮬레이션 결과**:
   - 경로: `saved/dental_fea_job.frd` (크기: 128 MB, 355,212개 절점, 치과 하악골 + 티타늄 임플란트 실제 해석 결과)
   - 용도: Step 4-8 대규모 통합 성능 검증 및 메모리 스트레스 테스트용.
