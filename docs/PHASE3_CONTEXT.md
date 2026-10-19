# Phase 3 컨텍스트 및 아키텍처 기준 문서 (docs/PHASE3_CONTEXT.md)

> **목적**: Phase 1, Phase 2의 실제 구현 구조를 파악하고, Phase 3 FEA 파이프라인(메쉬, 물성 매핑, 경계조건, CalculiX .inp 및 솔버 연동)이 접목될 인터페이스와 데이터 흐름을 명확히 고정합니다.

---

## 1. 실제 디렉터리 구조 및 초기 계획 대비 현황

현재 코드베이스는 Phase 1(영상 로더, 2D MPR 3분할, 독립 3D 뷰어)과 Phase 2(골 분할, 다중 임플란트 식립, 기하 처리 및 불리언)가 완전히 구현되어 있습니다.

```text
dental_FEA/
├── .antigravity/rules/common_rules.md    # 프로젝트 공통 규칙
├── AGENTS.md                             # 프로젝트 공통 규칙 및 핵심 설계 결정
├── docs/
│   ├── PHASE3_CONTEXT.md                 # [본 문서] Phase 3 구현을 위한 구조 기준 문서
│   ├── PHASE3_PLAN.md                    # Phase 3 단계별 프롬프트 계획서
│   └── PROJECT_MAP.md                    # Phase 1 종료 시점 프로젝트 맵
├── fea/                                  # [Phase 3] 신규 FEA 코어 모듈
│   ├── __init__.py
│   └── models.py                         # FEA 핵심 데이터클래스 정의 (계약 고정)
├── saved/                                # 불리언 결과 STL, 계측, 드로잉 영속화 폴더
├── samples/
│   └── Class 3 malocclusion/             # 실제 치과 CBCT DICOM 시리즈 샘플
├── tests/
│   ├── create_dummy_data.py              # 테스트용 합성 볼륨/데이터 생성기
│   ├── test_models.py                    # [Phase 3] FEA 데이터 모델 테스트
│   └── (Phase 1, 2 테스트 파일들)
└── dental_fea_app/                       # 메인 애플리케이션 패키지
    ├── config.py                         # 전역 상수, W/L 프리셋, 단위계 (mm, N, MPa, tonne/mm³)
    ├── main.py                           # 앱 진입점
    ├── core/
    │   ├── state_manager.py              # 전역 상태 관리자 (StateManager)
    │   ├── settings_manager.py           # UI 영속화 설정 관리자
    │   └── events.py                     # 이벤트 버스
    ├── imaging/
    │   ├── dicom_reader.py               # DICOM 시리즈 및 의료영상 로더
    │   └── volume_data.py                # SimpleITK <-> vtkImageData 변환
    ├── segmentation/
    │   └── cortical_cancellous.py        # 피질골/수질골 분할 순수 로직 (SegmentationResult)
    ├── placement/
    │   ├── implant_generator.py          # 임플란트/지대주/보철물 파라메트릭 생성기
    │   ├── implant_placement_manager.py  # 다중 임플란트 식립 관리 엔진 (PlacedImplant)
    │   ├── implant_gizmo.py              # 3D 기즈모
    │   └── implant_2d_gizmo.py           # 2D MPR 슬라이스 기즈모
    ├── geometry/
    │   ├── surface_extract.py            # 마스크 -> 표면 메쉬 추출 (FlyingEdges/MarchingCubes)
    │   └── boolean_ops.py                # 식립부 불리언 절제 (FeaGeometry: cortical, cancellous, implant)
    └── ui/
        ├── main_window.py                # 메인 윈도우
        ├── control_panel.py              # 좌측 워크플로 컨트롤 패널 (QTabWidget)
        ├── mpr_container.py              # 2D MPR 뷰 컨테이너
        ├── mpr_widget.py                 # 개별 2D MPR VTK 뷰어
        ├── volume_viewer_window.py       # 독립 3D 뷰어 윈도우
        ├── segmentation/                 # 분할 UI 및 QThread 워커
        ├── placement/                    # 식립 UI
        └── geometry/                     # 기하 처리 UI (GeometryPanel, GeometryWorker, FeaGeometryViewer)
```

### 초기 계획 대비 변경/발전 사항
1. **임플란트 식립 모듈의 확장**:
   단순 STL 배치를 넘어 `ImplantPlacementManager`를 중심으로 지대주(Abutment), 보철 크라운(Prosthetic Crown)의 다중 배치, 2D 단면 슬라이스 컨투어 실시간 생성, 언두/리두 히스토리가 완전히 구축됨.
2. **기하 처리 및 불리언 결과물 분리**:
   `FeaGeometry` 데이터클래스에 `cortical`, `cancellous`, `implant`가 각각 watertight한 `trimesh.Trimesh` 객체로 유지되며, Phase 3에서 바로 3D 체적 메싱의 인풋으로 투입될 준비가 완료됨.
3. **루트의 `geometry/` 디렉터리**:
   루트에 존재하는 `geometry/` 폴더는 이전 코드 호환을 위한 re-export 래퍼이며, 실제 소스코드는 `dental_fea_app/geometry/`에 위치함.

---

## 2. CT 볼륨 보관 및 접근 방식

- **보관 위치**: `StateManager` 인스턴스 (`state_mgr.vtk_image`)
- **자료형**: `Optional[vtk.vtkImageData]`
- **접근 방법**:
  ```python
  # StateManager로부터 vtkImageData 참조
  vtk_image = state_mgr.vtk_image
  
  # 볼륨 메타데이터 및 지오메트리 정보
  dimensions = state_mgr.dimensions  # (nx, ny, nz) 복셀 수
  spacing = state_mgr.spacing        # (sx, sy, sz) mm 단위 복셀 크기
  origin = state_mgr.origin          # (ox, oy, oz) LPS mm 단위 원점
  bounds = state_mgr.bounds          # (xmin, xmax, ymin, ymax, zmin, zmax)
  scalar_range = state_mgr.scalar_range # (hu_min, hu_max)
  ```
- **데이터 로드 흐름**:
  `DicomReader.load_dicom_series(folder_path)` 또는 `DicomReader.load_single_file(file_path)`가 SimpleITK로 읽은 후 `itk_to_vtk_image`를 거쳐 `state_mgr.set_volume_data(vtk_image, meta)`를 호출하여 등록.

---

## 3. 피질골/수질골 분할 결과 (마스크 및 표면)

- **보관 위치**: `state_mgr.segmentation_result`
- **자료형**: `dental_fea_app.segmentation.cortical_cancellous.SegmentationResult`
  ```python
  @dataclass
  class SegmentationResult:
      bone_mask: np.ndarray        # bool, shape (nz, ny, nx)
      cortical_mask: np.ndarray    # bool, shape (nz, ny, nx)
      cancellous_mask: np.ndarray  # bool, shape (nz, ny, nx)
      params: Dict[str, Any]
  ```
- **좌표계 및 불변성**:
  - 원본 CT 볼륨과 동일한 크기와 복셀 축 정합 `(nz, ny, nx)` (LPS World 좌표계).
  - 불변성: `cortical_mask | cancellous_mask == bone_mask`, `cortical_mask & cancellous_mask == False` (상호 배타적).
- **라벨 볼륨**:
  - `state_mgr.combined_label_vtk`: VTK 단일 볼륨 (`0: 배경`, `1: 피질골`, `2: 수질골`, uint8).

---

## 4. 임플란트/지대주/보철 STL 및 변환 행렬 적용 방식

- **관리자**: `state_mgr.implant_placement_mgr` (`ImplantPlacementManager`)
- **보관 위치**: `state_mgr.implant_placement_mgr.implants` (`Dict[str, PlacedImplant]`)
- **자료형**: `PlacedImplant`
  - `spec`: `ImplantSpec` (직경, 길이, 플랫폼 타입, 스레드 사양 등)
  - `position`: `np.ndarray` `(3,)` [x, y, z] (mm, 월드 좌표)
  - `orientation`: `np.ndarray` `(3,)` [yaw, pitch, roll] (deg)
  - `abutment`: `Optional[AbutmentSpec]` (칼라 높이, 각도, 직경 등)
  - `prosthetic`: `Optional[ProstheticSpec]` (치아 번호, 재질, 크기 등)
- **변환 행렬(Transform) 적용 및 월드 메쉬 획득**:
  `ImplantPlacementManager`의 공개 메서드로 월드 좌표계(`LPS, mm`)가 적용된 `vtkPolyData`를 즉시 얻을 수 있음:
  ```python
  # 임플란트 월드 메쉬
  implant_poly = implant_placement_mgr.get_transformed_polydata(imp)
  # 지대주 월드 메쉬
  abutment_poly = implant_placement_mgr.get_transformed_abutment_polydata(imp)
  # 보철물 월드 메쉬
  prosthetic_poly = implant_placement_mgr.get_transformed_prosthetic_polydata(imp)
  ```
  - 변환 순서 (PostMultiply):
    - 로컬 Z축 오프셋 정렬 (`-half_len` or `+half_len`)
    - RotateZ(yaw) → RotateX(pitch) → RotateY(roll)
    - Translate(position.x, position.y, position.z)

---

## 5. Boolean 결과물 보관 및 형태

- **생성 함수**: `dental_fea_app.geometry.boolean_ops.build_fea_geometry(cortical_mesh, cancellous_mesh, implant_mesh, ...)`
- **형태**: `dental_fea_app.geometry.boolean_ops.FeaGeometry`
  ```python
  @dataclass
  class FeaGeometry:
      cortical: trimesh.Trimesh     # 식립와가 감산된 피질골 표면 (LPS mm)
      cancellous: trimesh.Trimesh   # 식립와 및 피질골 중복이 제거된 수질골 표면 (LPS mm)
      implant: trimesh.Trimesh      # 식립 위치의 임플란트 표면 메쉬 (LPS mm)
      report: Dict[str, Any]        # 부피, 접촉면적, 메쉬 품질 리포트
  ```
- **보관 위치**:
  - 런타임 상태: `state_mgr.fea_geometry` (`Optional[FeaGeometry]`)
  - 파일 영속화: `saved/cortical.STL`, `saved/cancellous.STL`, `saved/implant.STL`
  - 모든 메쉬는 `ensure_watertight`를 거쳐 닫힌 매니폴드 상태가 보장됨.

---

## 6. 이미지 인덱스 ↔ 월드 좌표(mm) 변환 방식

- **방향 행렬**:
  SimpleITK 로드 시 `itk_to_vtk_image`에서 `reorient_to_lps=True`를 수행하여 표준 LPS 좌표계로 정규화되므로 방향 행렬은 주축 단위행렬(Identity Matrix) 형태임.
- **변환 수식**:
  - 월드 좌표 $(x, y, z)$ [mm] $\rightarrow$ 복셀 인덱스 $(i, j, k)$ [0-based]:
    $$i = \text{round}\left(\frac{x - \text{origin}_x}{\text{spacing}_x}\right)$$
    $$j = \text{round}\left(\frac{y - \text{origin}_y}{\text{spacing}_y}\right)$$
    $$k = \text{round}\left(\frac{z - \text{origin}_z}{\text{spacing}_z}\right)$$
  - 복셀 인덱스 $(i, j, k)$ $\rightarrow$ 월드 좌표 $(x, y, z)$ [mm]:
    $$x = \text{origin}_x + i \times \text{spacing}_x$$
    $$y = \text{origin}_y + j \times \text{spacing}_y$$
    $$z = \text{origin}_z + k \times \text{spacing}_z$$
  - 넘파이 배열 인덱싱:
    - VTK 차원: `dimensions = (nx, ny, nz)`
    - 넘파이 HU 배열 차원: `shape = (nz, ny, nx)`
    - 따라서 복셀 $(i, j, k)$의 HU 값: `hu_value = hu_array[k, j, i]`
  - 내장 헬퍼: `state_mgr.get_voxel_value_at(x, y, z)`로 특정 월드 좌표의 HU 값을 안전하게 샘플링 가능.

---

## 7. 전역 상태 관리자 (StateManager) API 요약

### 주요 Signals
- `coordinates_changed(float, float, float)`: 커서 위치 (x, y, z) [mm]
- `volume_loaded(object)`: `vtk.vtkImageData` 로드 완료
- `window_level_changed(float, float)`: W/L 변경
- `roi_bounds_changed(object)`: 6-tuple `(xmin, xmax, ymin, ymax, zmin, zmax)` 또는 `None`
- `segmentation_result_changed(object)`: `SegmentationResult` 변경
- `implant_transformed(str)`: 임플란트 ID의 위치/자세 변경
- `implants_updated()`: 임플란트 목록 변경
- `fea_geometry_changed(object)`: `FeaGeometry` 생성 완료
- `render_all_requested()`: 전체 뷰 리렌더 요청

### 주요 Properties & Methods
- `vtk_image`, `origin`, `spacing`, `dimensions`, `bounds`, `scalar_range`, `metadata`
- `segmentation_result`, `combined_label_vtk`, `fea_geometry`
- `implant_placement_mgr`: 임플란트 식립 관리 엔진 인스턴스
- `get_voxel_value_at(x, y, z) -> Optional[float]`: HU 샘플링
- `set_fea_geometry(geom: FeaGeometry)`: FEA 기하 데이터 등록

---

## 8. 좌측 제어 패널(ControlPanel) 구조 및 탭 확장

- **위치**: `dental_fea_app/ui/control_panel.py` (`ControlPanel`)
- **구조**: `QVBoxLayout` $\rightarrow$ `QScrollArea` $\rightarrow$ `workflow_tabs: QTabWidget`
- **현재 탭 구성**:
  1. `1. 영상`: 영상 정보 및 슬라이스 상태
  2. `2. 분할`: `SegmentationPanel`
  3. `3. 임플란트`: `ImplantPanel`
  4. `4. 기하 처리`: `GeometryPanel`
  5. `5. 해석`: `t5` (현재 플레이스홀더 QWidget)
- **Phase 3 새 탭 연동 방법**:
  - `control_panel.py`의 `_create_workflow_tabs()`에서 `t5` 위젯 대신 `FeaPanel(self.state_mgr, self)`를 생성하여 `tabs.addTab(self.fea_panel, "5. 해석")`으로 연결.

---

## 9. 3D 뷰어 창(VolumeViewerWindow)의 액터 제어 방식

- **위치**: `dental_fea_app/ui/volume_viewer_window.py` (`VolumeViewerWindow`)
- **렌더러 접근**: `self.renderer` (`vtk.vtkRenderer`)
- **액터 조작 API**:
  - 추가: `self.renderer.AddActor(actor)`
  - 제거: `self.renderer.RemoveActor(actor)`
  - 가시성: `actor.SetVisibility(True / False)`
  - 뷰 갱신: `self.render()` 또는 `self.request_render()` (60ms QTimer 스로틀링)
- **FEA 메쉬/결과 표시 연동 계획**:
  - 체적 메쉬 표면 또는 변위/응력 컨투어 메쉬를 `vtkPolyData` $\rightarrow$ `vtkPolyDataMapper` $\rightarrow$ `vtkActor`로 변환하여 `volume_viewer_window.renderer.AddActor(fea_actor)` 형태로 렌더링.

---

## 10. 설치 패키지 현황 및 requirements 추가 목록

### 확인 결과 (실행 환경 검사)
| 패키지 | 현재 상태 | 용도 |
| :--- | :--- | :--- |
| `numpy` | **2.5.4** (설치됨) | 수치 배열 연산 |
| `scipy` | **1.18.1** (설치됨) | 형태학 필터, 거리 변환, k-d tree |
| `trimesh` | **5.1.1** (설치됨) | 표면 메쉬 분석 및 불리언 |
| `manifold3d` | **설치됨** | watertight 빠른 불리언 엔진 |
| `SimpleITK` | **2.5.6** (설치됨) | 의료영상 I/O |
| `vtk` | **9.7.1** (설치됨) | 시각화 파이프라인 |
| `gmsh` | **미설치** | 3D 비정렬 4면체 체적 메쉬 생성기 (Phase 3 필수) |
| `meshio` | **미설치** | Gmsh 메쉬 읽기 및 변환 지원 |
| `pyvista` | **미설치** | VTK 래퍼 (선택 사항) |

### requirements.txt 반영 필요 목록
Phase 3 메싱 및 I/O를 위해 아래 패키지 설치가 필요함 (본 3-0 단계에서는 설치하지 않고 목록만 정리):
- `gmsh`
- `meshio`

---

## 11. ccx (CalculiX) 실행 파일 현황

- **검색 결과**: `/usr/bin/ccx` 에 정상 설치되어 있음.
- **버전**: `CalculiX Version 2.21`
- **실행 확인**: 정상 동작 확인 완료 (`CalculiX.exe -i jobname`)
- **설정 정책**: 외부 실행 파일 경로는 하드코딩하지 않고, `config.py` 또는 환경변수 `CCX_PATH`에서 읽도록 설계 (기본값으로 `shutil.which("ccx")`를 통해 `/usr/bin/ccx` 탐지).
