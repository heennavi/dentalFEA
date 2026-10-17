# Dental FEA 프로젝트 아키텍처 맵 (Phase 1 현황 및 Phase 2 로드맵)

> 본 문서는 Phase 1 개발이 완료된 현재 코드베이스를 분석하여, 이후 Phase 2 및 후속 개발 단계에서 정확한 파일 경로와 접근 방식을 즉시 참조할 수 있도록 작성된 기준 문서입니다.

---

## 1. 프로젝트 전체 디렉터리 구조

```text
dental_FEA/
├── .gemini/rules/common_rules.md      # Antigravity Workspace Rules
├── .local_libs/                       # Qt 6.5+ XCB 플러그인 호환 라이브러리 (libxcb-cursor.so.0)
├── .venv/                             # Python 3.12 가상환경
├── docs/
│   └── PROJECT_MAP.md                 # [본 문서] 프로젝트 구조 및 접근 가이드 맵
├── requirements.txt                   # 패키지 의존성 정의
├── run.py                             # 애플리케이션 편리 실행 진입 스크립트
├── user_settings.json                 # UI, 뷰어(2D/3D), 스플리터 크기 등 영속화 설정 파일
├── saved/                             # 주석, 계측, 드로잉 데이터 저장 폴더 (JSON)
├── samples/
│   └── Class 3 malocclusion/          # 실제 치과 CBCT DICOM 시리즈 샘플 데이터 (517 슬라이스)
├── tests/
│   ├── create_dummy_data.py           # 테스트용 합성 치과 CT 볼륨 생성 유틸리티
│   ├── test_phase1.py                 # Phase 1 코어 기능 및 UI 통합 단위 테스트
│   ├── test_advanced_features.py      # 고급 UI 기능(회전, 대칭, 3D 클리핑, 최근파일) 테스트
│   ├── test_measurement_and_roi_fixes.py # 계측 도구 및 3D ROI 바운딩박스/크롭 테스트
│   ├── test_user_info_and_drawing.py  # 주석/계측/도형 저장 및 복원 테스트
│   └── sample_data/                   # 단위 테스트용 가상 MHA 데이터 (.mha)
└── dental_fea_app/                    # 메인 애플리케이션 소스 패키지
    ├── __init__.py
    ├── main.py                        # 앱 메인 진입점 (QApplication, CLI 인자 로드)
    ├── config.py                      # 전역 상수, 축 정의, 컬러, W/L 프리셋, 물리 단위계, 다크 테마 QSS
    ├── core/                          # 전역 상태 및 영속 설정 관리 모듈
    │   ├── __init__.py
    │   ├── state_manager.py           # StateManager (좌표, 볼륨, W/L, ROI, 상호작용 모드 싱글톤)
    │   ├── settings_manager.py        # SettingsManager (user_settings.json 읽기/쓰기 싱글톤)
    │   └── events.py                  # AppEvents (경량 이벤트 버스 정의)
    ├── imaging/                       # 의료영상 I/O 및 데이터 변환 모듈
    │   ├── __init__.py
    │   ├── dicom_reader.py            # DicomReader (DICOM 시리즈 재귀 탐색 및 단일 파일 로더)
    │   └── volume_data.py             # itk_to_vtk_image (SimpleITK <-> VTK ImageData, LPS 변환)
    └── ui/                            # PySide6 + VTK UI 컴포넌트 모듈
        ├── __init__.py
        ├── main_window.py             # MainWindow (메인 창, 메뉴바, 툴바, 스플리터, 상태표시줄)
        ├── control_panel.py           # ControlPanel (좌측 240px 워크플로 탭 및 파라미터 패널)
        ├── mpr_container.py           # MPRContainer & ViewPanel (2x2 스플리터, 3분할 2D 뷰 배치)
        ├── mpr_widget.py              # MPRWidget (Axial/Coronal/Sagittal 개별 VTK 2D MPR 뷰어)
        ├── mpr_ruler.py               # MPRRuler (VTK 2D 네이티브 눈금자 렌더러)
        ├── ruler_overlay.py           # [Legacy] 구버전 눈금자 오버레이
        ├── volume_viewer_window.py    # VolumeViewerWindow (독립 3D GPU 볼륨 렌더링 윈도우)
        ├── icons.py                   # IconFactory (QPainter 기반 고해상도 벡터 아이콘 팩토리)
        ├── image_info_dialog.py       # ImageInfoDialog (현재 로드된 영상 메타데이터 정보 창)
        ├── measurement_dialog.py      # MeasurementDialog (길이/각도 측정 관리 창)
        ├── draw_dialog.py             # DrawDialog (선, 화살표, 사각형, 원, 프리폼 그리기 창)
        └── annotation_dialog.py       # AnnotationDialog (소견 메모 및 DICOM 메타데이터 창)
```

---

## 2. 핵심 항목별 위치 및 접근 방법 표

| 항목 | 위치 (파일/클래스) | 주요 메서드 / 속성 | 접근 및 사용 방법 |
| :--- | :--- | :--- | :--- |
| **1. DICOM 로드 결과 (`vtkImageData`)** | `dental_fea_app/core/state_manager.py`<br>`StateManager` | `state_mgr.vtk_image`<br>`state_mgr.set_volume_data(vtk_img, meta)` | • `state_mgr.vtk_image` 속성으로 접근 (`Optional[vtk.vtkImageData]`)<br>• 로드 시 `state_mgr.set_volume_data(...)`를 호출하면 `volume_loaded` 시그널이 전역으로 브로드캐스트됨.<br>• 로더 호출: `DicomReader.load_path(path)` 또는 `load_dicom_series(dir)`, `load_single_file(file)` |
| **2. origin / spacing / direction 정보** | `dental_fea_app/core/state_manager.py`<br>`StateManager`<br>및 `imaging/volume_data.py` | `state_mgr.origin`<br>`state_mgr.spacing`<br>`state_mgr.dimensions`<br>`state_mgr.bounds`<br>`state_mgr.metadata["direction"]` | • `state_mgr.origin`: `Tuple[float, float, float]` (LPS mm)<br>• `state_mgr.spacing`: `Tuple[float, float, float]` (mm)<br>• `state_mgr.dimensions`: `Tuple[int, int, int]` (복셀 수 nx, ny, nz)<br>• `state_mgr.bounds`: `Tuple[float, ...]` (xmin, xmax, ymin, ymax, zmin, zmax)<br>• ITK 원본 방향 행렬: `state_mgr.metadata.get("direction")` (9-tuple)<br>• 복셀(ijk) ↔ 월드(xyz) 변환: $x = \text{origin} + i \times \text{spacing}$ / $i = \text{round}((x - \text{origin}) / \text{spacing})$ |
| **3. 전역 상태 관리자 및 Signals** | `dental_fea_app/core/state_manager.py`<br>`StateManager(QObject)` | 아래 Signal 상세 표 참조 | • 메인 윈도우 인스턴스: `main_win.state_mgr`<br>• 각 위젯에서 `self.state_mgr`로 참조<br>• 좌표 변경: `state_mgr.set_coordinates(x, y, z)` 또는 `state_mgr.set_axis_coordinate(axis, val)`<br>• W/L 변경: `state_mgr.set_window_level(w, l)`<br>• ROI 설정: `state_mgr.set_roi_bounds(bounds)` |
| **4. 2D MPR 위젯 및 슬라이스 평면** | `dental_fea_app/ui/mpr_container.py`<br>`MPRContainer`, `ViewPanel`<br>`dental_fea_app/ui/mpr_widget.py`<br>`MPRWidget` | `mpr_container.axial_panel.mpr_widget`<br>`mpr_container.coronal_panel.mpr_widget`<br>`mpr_container.sagittal_panel.mpr_widget`<br>`mpr_widget.renderer`<br>`mpr_widget.slice_plane`<br>`mpr_widget.reslice_mapper` | • 각 단면 뷰어 접근:<br>  - Axial (Z축): `mpr_container.axial_panel.mpr_widget` (axis=2)<br>  - Coronal (Y축): `mpr_container.coronal_panel.mpr_widget` (axis=1)<br>  - Sagittal (X축): `mpr_container.sagittal_panel.mpr_widget` (axis=0)<br>• Renderer 접근: `mpr_widget.renderer` (`vtk.vtkRenderer`)<br>• 액터 추가/제거: `mpr_widget.renderer.AddActor(actor)` / `RemoveActor(actor)`<br>• 평면 객체: `mpr_widget.slice_plane` (`vtk.vtkPlane`)<br>• 평면 법선: Axial `(0, 0, 1)`, Coronal `(0, -1, 0)`, Sagittal `(1, 0, 0)` |
| **5. 3D 뷰어 창 및 액터 제어** | `dental_fea_app/ui/volume_viewer_window.py`<br>`VolumeViewerWindow(QMainWindow)` | `main_win.volume_window`<br>`main_win.open_3d_viewer()`<br>`volume_window.renderer`<br>`volume_window.render()` | • 뷰어 창 접근: `main_win.volume_window` (미생성 시 `main_win.open_3d_viewer()` 호출)<br>• Renderer 접근: `volume_window.renderer` (`vtk.vtkRenderer`)<br>• 볼륨 액터: `volume_window.volume` (`vtk.vtkVolume`)<br>• 서피스/임플란트 메쉬 추가: `volume_window.renderer.AddActor(actor)`<br>• 서피스/메쉬 제거: `volume_window.renderer.RemoveActor(actor)`<br>• 화면 갱신: `volume_window.render()` |
| **6. 좌측 제어 패널 및 탭 확장** | `dental_fea_app/ui/control_panel.py`<br>`ControlPanel(QWidget)` | `main_win.control_panel`<br>`control_panel._create_workflow_tabs()` | • 패널 너비: 고정 240px (`setFixedWidth(240)`)<br>• 내부 구조: `QVBoxLayout` → `QScrollArea` → `container` → 워크플로 `QTabWidget`<br>• 현재 탭 구성: `1. 영상`, `2. 분할`, `3. 해석`<br>• 탭 추가/교체 방법: `_create_workflow_tabs()` 내에서 `tabs.addTab(widget, "탭이름")`을 호출하거나, 해당 탭 인스턴스에 Phase 2 전용 위젯 배치 |

---

## 3. StateManager 시그널(Signals) 상세 목록

| Signal 시그너처 | 인자 타입 및 단위 | 발생 시점 및 용도 |
| :--- | :--- | :--- |
| `coordinates_changed` | `(float, float, float)`<br>단위: mm (World LPS) | 3D 커서 좌표(x, y, z) 변경 시 발행. 2D 3분할 뷰의 슬라이스 평면, 십자선 교차점, 3D 뷰어 내 평면 위치가 동기화됨. |
| `window_level_changed` | `(float, float)`<br>단위: HU (명암 폭, 레벨) | 밝기/대비(Window, Level) 변경 시 발행. 2D MPR 뷰어의 `vtkImageProperty`에 즉시 반영됨. |
| `volume_loaded` | `(object)`<br>인스턴스: `vtk.vtkImageData` | 새로운 3D 영상 데이터셋이 로드되었을 때 발행. 각 2D 뷰어 리슬라이서 및 3D 볼륨 매퍼에 입력 데이터 연결. |
| `status_message` | `(str, int)`<br>(메시지, 표시시간 ms) | 메인 윈도우 하단 상태표시줄(QStatusBar)에 알림 메시지를 표시할 때 사용. |
| `roi_bounds_changed` | `(object)`<br>`(xmin, xmax, ymin, ymax, zmin, zmax)` 또는 `None` | 관심영역(ROI) 경계가 지정되거나 변경되었을 때 발행. 3D 크롭 영역 및 3D 바운딩 박스 크기 동기화. |
| `roi_setting_mode_toggled` | `(bool)`<br>True / False | 3D 설정 패널의 "boundary 설정" 토글 시 발행. 2D 뷰어에 사각형 드래그 모드 활성화. |
| `roi_box_visibility_changed` | `(bool)`<br>True / False | 3D 파란색 ROI 바운딩 박스 가시성 토글 시 발행. |
| `interaction_mode_changed` | `(str)`<br>`'normal'`, `'draw'`, `'draw_select'`, `'measure_length'`, `'measure_angle'` | 2D 뷰어 내 마우스 상호작용 모드(일반, 선/도형 그리기, 길이/각도 계측) 전환 알림. |

---

## 4. 초기 설계 대비 구현 변경 사항 분석

1. **독립 창 형태의 3D 볼륨 뷰어 분리 (`VolumeViewerWindow`)**
   - **초기 계획**: 메인 화면 우하단 4번째 영역에 3D 뷰어를 함께 배치하는 2x2 고정 분할 방식.
   - **변경 내용**: 고해상도 치과 영상 및 인터랙션 편의를 위해 3D 볼륨 뷰어를 독립 QMainWindow(`VolumeViewerWindow`)로 완전 분리.
   - **영향**: `MPRContainer`의 우하단 영역은 빈 검은색 사분면 패널(`control_quadrant`)로 유지되고 있으며, 3D 창은 툴바 아이콘 및 메뉴(`Ctrl+3`)로 호출됨. 창 닫기 시 OpenGL 컨텍스트 충돌 및 고스팅(ghosting)을 방지하기 위해 닫힌 인스턴스를 파괴하고 새로 생성하는 생명주기 관리 적용.

2. **VTK 2D 네이티브 눈금자 모듈 도입 (`mpr_ruler.py`)**
   - **초기 계획**: 단순 QPainter 기반 오버레이 레이어(`ruler_overlay.py`).
   - **변경 내용**: 뷰포트 확대/축소 및 이동 시 정밀한 1mm, 5mm, 10mm 눈금 틱과 숫자 라벨을 지원하는 순수 VTK 2D 네이티브 렌더러(`MPRRuler`)로 전면 개편.

3. **다이얼로그 분리 및 공유 시스템**
   - 주석(`annotation_dialog.py`), 계측(`measurement_dialog.py`), 드로잉 도구(`draw_dialog.py`), 영상 메타데이터(`image_info_dialog.py`)가 `MPRContainer` 레벨에서 싱글톤 다이얼로그로 공유 관리되며, `saved/` 폴더 내 JSON 영속화 지원.

4. **설정 영속화 매니저 (`settings_manager.py`)**
   - 윈도우 크기, 위치, 2D 뷰별 십자선/자/회전(90도)/대칭 상태, 3D 뷰어 음영/투명도/클리핑/ROI 상태가 `user_settings.json`에 실시간 자동 저장 및 복원됨.

5. **좌표 변환 함수의 일원화 필요성 (Phase 2 준비 과제)**
   - 현재 복셀 인덱스(ijk) ↔ 월드 좌표(xyz) 변환이 `StateManager.get_voxel_value_at` 및 `mpr_widget.py` 내부에 분산되어 있음.
   - **공통 규칙 준수를 위해 Phase 2 시작 시 단일 변환 유틸 함수(`dental_fea_app/geometry/coords.py`)로 일원화 필수**.

---

## 5. Phase 2 신규 파일 및 디렉터리 배치 제안

> **[구조 원칙 준수]**
> - 순수 로직(numpy/vtk/trimesh)과 UI(PySide6)를 엄격히 분리한다. 로직 모듈은 PySide6를 일절 import하지 않는다.
> - 무거운 연산(분할, 표면 추출, Boolean)은 `QThread` 또는 `QRunnable` 워커를 통해 백그라운드에서 수행한다.

```text
dental_fea_app/
├── geometry/                                  # [NEW] 공통 기하 및 좌표계 변환 로직 (순수 로직)
│   ├── __init__.py
│   ├── coords.py                              # [규칙 필수] ijk <-> world(LPS, mm) 단일 유틸리티 함수
│   ├── surface_extract.py                     # vtkFlyingEdges3D / Marching Cubes 표면 메쉬(STL) 추출 및 스무딩
│   └── boolean_ops.py                         # 하악골-임플란트 간 Boolean 연산 (식립와 절제) (trimesh / manifold3d)
│
├── segmentation/                              # [NEW] 뼈 조직 분할 로직 (순수 로직)
│   ├── __init__.py
│   └── cortical_cancellous.py                 # HU 임계값(Thresholding), 3D 형태학 필터, 피질골/수질골 이진 마스크 분할
│
├── placement/                                 # [NEW] 임플란트 식립 관리 로직 (순수 로직)
│   ├── __init__.py
│   └── implant_manager.py                     # 임플란트 STL 로드, 3D 위치(xyz)/각도/식립축(vector) 변환 행렬 연산
│
└── ui/
    ├── segmentation/                          # [NEW] 분할 관련 UI 및 비동기 워커
    │   ├── __init__.py
    │   ├── segmentation_panel.py              # 좌측 패널 탭 "2. 분할"에 임베딩될 UI 위젯
    │   └── segmentation_worker.py             # QThread 기반 비동기 분할 및 표면 추출 연산 워커
    │
    └── placement/                             # [NEW] 임플란트 식립 관련 UI
        ├── __init__.py
        ├── implant_panel.py                   # 임플란트 사양 선택, 위치/각도 미세조정 UI 위젯
        └── implant_actor.py                   # 2D MPR 및 3D 뷰어에 표시될 임플란트 가이드 액터
```

### 테스트 코드 배치 제안
```text
tests/
├── test_coords.py                             # 좌표 변환 단일 유틸 함수 정밀도 검증 (pytest)
├── test_segmentation.py                       # 합성 데이터 기반 피질골/수질골 분할 단독 테스트 (pytest)
├── test_surface_extract.py                    # 마스크 -> STL 표면 추출 및 유효성(매니폴드) 검증 (pytest)
├── test_implant_manager.py                    # 임플란트 3D 회전/이동 변환 행렬 검증 (pytest)
└── test_boolean_ops.py                        # 단순 STL 불리언 감산 연산 검증 (pytest)
```

---

## 6. 패키지 설치 현황 및 Phase 2 추가 필요 패키지

### 6.1 현재 설치된 주요 패키지 (`.venv`)
| 패키지명 | 설치된 버전 | 용도 |
| :--- | :--- | :--- |
| `vtk` | **9.7.1** | 2D MPR 리슬라이싱, 3D Smart GPU 볼륨 렌더링, 시각화 |
| `PySide6` | **6.12.0** | 데스크톱 GUI 프레임워크 (Qt 6.12) |
| `numpy` | **2.5.4** | 대규모 수치 배열 연산 |
| `SimpleITK` | **2.5.6** | 의료영상(DICOM, MHA) I/O 및 영상 정합 |
| `pydicom` | **3.0.2** | DICOM 태그/헤더 메타데이터 파싱 |
| `matplotlib` | **3.11.2** | 2D 플롯 및 컬러맵 |
| `pillow` | **12.3.0** | 이미지 처리 보조 |

### 6.2 Phase 2 추가 설치 필요 패키지 후보 및 사유
| 후보 패키지 | 추천 사유 및 사용처 | 설치 우선순위 |
| :--- | :--- | :--- |
| `pytest` | 공통 규칙에 따른 단독 단위 테스트 자동화 및 실행 프레임워크 (현재 venv에 미설치됨) | **최우선 (필수)** |
| `scipy` | 3D 이진 형태학 연산(`scipy.ndimage.binary_opening`, `closing`), 연결 성분 라벨링, 거리 변환 (피질골 2mm 균일 두께 계산 등에 필수) | **최우선 (필수)** |
| `trimesh` | 3D 삼각 메쉬 처리, STL 읽기/쓰기, 표면 메쉬 검사 (닫힌 매니폴드 여부, 법선 벡터 정합) | **높음 (필수)** |
| `manifold3d` | 빠르고 기하학적으로 완벽한 watertight 3D Boolean 연산 라이브러리 (trimesh boolean 엔진으로 활용, 불리언 오류 원천 방지) | **높음 (강력 권장)** |
| `scikit-image` | 3D 영상 분할 후처리 및 Marching Cubes 대안 알고리즘 지원 | **보통 (선택)** |

---

## 7. Phase 2 개발 시 준수사항 (공통 규칙 재확인)

1. **좌표계 및 단위**:
   - 길이 `mm`, 힘 `N`, 응력 `MPa`.
   - 모든 기하는 DICOM 원본과 동일한 LPS World 좌표계(`mm`)를 사용한다.
   - 인덱스(ijk)와 월드(xyz) 변환은 반드시 `dental_fea_app/geometry/coords.py` 유틸 함수로 일원화하여 사용한다.
2. **구조 분리**:
   - `segmentation/`, `geometry/`, `placement/` 하위 모듈은 순수 로직으로 작성하며, PySide6를 절대 import하지 않는다.
   - 모든 로직 모듈은 GUI 없이 단독으로 `pytest` 실행이 가능해야 한다.
3. **기존 파일 변경 최소화**:
   - 기존 Phase 1 코드는 요청된 범위를 벗어나 수정하지 않는다.
   - 변경이 불가피한 경우 사전 보고 후 진행한다.
4. **비동기 UI 보호**:
   - 분할, 표면 추출, Boolean 연산 등 연산 시간이 수초 이상 소요되는 무거운 작업은 반드시 `QThread`로 처리하여 UI 프리징을 방지한다.
