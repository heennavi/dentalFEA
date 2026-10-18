# Phase 3 실행 계획: FEA 파이프라인 (메쉬, 물성, 경계조건, 솔버)

대상: Google Antigravity + Gemini Flash
전제: Phase 1(뷰어, MPR, 3D 창), Phase 2(분할, 임플란트 배치, Boolean)는 구현 완료. 폴더 구조와 레이아웃은 초기 계획과 일부 다름.

## 0. 사용 방법

1. 아래 "공통 규칙"을 Antigravity의 워크스페이스 Rules에 등록한다. 또는 프로젝트 루트의 GEMINI.md / AGENTS.md에 붙여 넣는다. (사용 중인 버전에서 규칙 파일 위치를 한 번 확인할 것)
2. Step 3-0부터 번호 순서대로, 새 대화(세션)마다 하나씩 실행한다. 한 세션에 여러 Step을 넣지 않는다.
3. 각 Step은 "완료 조건"의 테스트가 통과해야 다음으로 넘어간다. 통과 후 git commit.
4. 각 프롬프트의 코드 블록 안쪽만 복사해서 에이전트에 넣는다.

Flash 계열 모델은 컨텍스트가 길어지면 기존 인터페이스를 임의로 바꾸는 경향이 있으므로, Step마다 입출력 계약(dataclass)을 먼저 고정하고, 합성 데이터로 CT 없이 단독 테스트가 가능하게 설계했다.

## 1. 전체 구조와 의존 관계

```
3-0 현황 파악 + 데이터 계약 고정
 ├─ 3-1 메쉬 생성 (Gmsh)
 ├─ 3-2 메쉬 품질 검사 + 표면/접촉면 추출
 ├─ 3-3 HU → 물성 매핑
 ├─ 3-4 경계조건/하중 모델
 └─ 3-5 CalculiX .inp 작성기
        └─ 3-6 솔버 실행기
               └─ 3-7 UI 연동 (좌측 패널 탭, 3D 창 표시, 스레드)
                      └─ 3-8 합성 모델 검증 + 실제 CT 통합 테스트
```

3-1 ~ 3-4는 서로 독립이라 순서를 바꿔도 되지만, 3-5는 앞의 네 개가 모두 끝나야 한다.

## 2. 핵심 설계 결정 (에이전트가 임의로 바꾸지 못하게 고정)

| 항목 | 결정 | 이유 |
|---|---|---|
| 단위계 | mm, N, MPa, tonne(밀도 t/mm³) | CalculiX는 단위가 없으므로 일관성이 전부 |
| 요소 | 기본 C3D10 (2차 사면체), 미리보기용 C3D4 옵션 | C3D4는 과강성 경향 |
| 피질/수질 구분 | 메쉬는 뼈 전체를 하나의 볼륨으로 생성하고, 요소 중심점에서 마스크와 HU를 샘플링해 물성 할당 | 피질/수질 경계면을 conformal하게 맞추는 과정이 가장 흔한 실패 지점 |
| 뼈-임플란트 | 별도 메쉬 + CalculiX *TIE (골유착 가정). 마찰 접촉은 3-8 이후 옵션 | 초기에는 선형 정적으로 안정적으로 돌리는 것이 우선 |
| 임플란트-지대주-보철 | 모두 *TIE | 동일 |
| Gmsh 2차 요소 번호 | CalculiX 변환 시 마지막 두 절점 교환 | 아래 3-5 참고 |
| 해석 | 선형 탄성 정적 (*STATIC) | 비선형은 이후 단계 |
| 결과 출력 | .frd (변위 U, 응력 S, 변형률 E) | Phase 4에서 파싱 |

Gmsh의 10절점 사면체 절점 순서(0-based)는 [0,1,2,3,4,5,6,7,8,9]이며 CalculiX C3D10 순서는 이 중 마지막 두 개가 바뀐 [0,1,2,3,4,5,6,7,9,8]에 해당한다. 이 변환을 빼먹으면 음의 야코비안 오류가 난다.

## 3. 공통 규칙 (Rules / GEMINI.md에 등록)

```
# 프로젝트 공통 규칙 (치과 임플란트 FEA 앱)

- 언어: 대화와 주석 설명은 한국어, 식별자와 docstring 첫 줄은 영어.
- 단위계는 항상 mm, N, MPa, tonne/mm^3. 단위 변환은 config.py 한 곳에서만.
- Phase 1, 2에서 이미 작성된 파일의 공개 함수 시그니처와 동작을 바꾸지 않는다.
  수정이 꼭 필요하면 먼저 이유를 설명하고, 변경 범위를 최소화한다.
- 새 기능은 새 파일/새 모듈에 작성한다. 기존 파일은 import 연결에 필요한 최소 수정만 한다.
- GUI 없이 import 가능해야 하는 모듈(meshing, material, solver)은 PySide6/VTK 렌더링 모듈을
  import하지 않는다. (headless 테스트 가능해야 함)
- 오래 걸리는 작업(메쉬, 솔브)은 UI 스레드에서 실행하지 않는다. QThread 또는 QProcess 사용.
- 모든 새 모듈에는 pytest 테스트를 tests/ 아래에 함께 작성하고, 작업 종료 전에 실제로 실행해
  결과를 보고한다. 실행하지 않은 테스트를 통과했다고 말하지 않는다.
- 외부 실행 파일(ccx 등) 경로는 하드코딩 금지. config 또는 환경변수에서 읽는다.
- 한 번에 한 Step만 작업한다. 요청 범위 밖의 리팩터링을 하지 않는다.
- 작업이 끝나면 (1) 변경/추가한 파일 목록 (2) 테스트 결과 (3) 남은 문제를 짧게 보고한다.
```

---

## Step 3-0. 현황 파악과 데이터 계약 고정

목적: Phase 1, 2의 실제 구조를 읽고 Phase 3 코드가 붙을 지점을 문서화한다. 이후 모든 Step이 이 문서를 기준으로 삼는다.

```
[Step 3-0] 현황 파악과 데이터 계약 고정

이 프로젝트는 치과 임플란트 유한요소해석(FEA) 데스크톱 앱이다(PySide6 + VTK).
Phase 1(DICOM 로드, 2D MPR 3분할, 별도 3D 뷰어 창)과 Phase 2(피질/수질골 분할,
임플란트 STL 배치, 골-임플란트 Boolean)는 이미 구현되어 있다. 이제 Phase 3(메쉬, 물성,
경계조건, CalculiX 솔버)를 시작한다. 이 단계에서는 코드를 거의 수정하지 않는다.

할 일:
1. 프로젝트 전체를 읽고 docs/PHASE3_CONTEXT.md를 작성한다. 다음을 반드시 포함:
   - 실제 디렉터리 구조 (초기 계획과 다른 부분 표시)
   - CT 볼륨(vtkImageData 또는 SimpleITK 이미지)이 어디에 보관되고 어떻게 접근하는지
   - 피질골/수질골 분할 결과(마스크 또는 표면)의 저장 위치, 자료형, 좌표계
   - 임플란트/지대주/보철 STL(또는 mesh)의 보관 위치와 변환 행렬 적용 방식
   - Boolean 결과물이 어떤 형태(vtkPolyData, trimesh 등)로 어디에 남는지
   - 이미지 인덱스 <-> 월드 좌표(mm) 변환 방법 (origin, spacing, direction matrix)
   - 전역 상태 관리자의 현재 API
   - 좌측 제어 패널이 탭/스택 중 어떤 구조인지, 새 탭을 추가하는 방법
   - 3D 뷰어 창이 액터를 추가/제거하는 방법
   - 설치된 패키지 확인 결과: gmsh, meshio, trimesh, pyvista, scipy, numpy, SimpleITK
     (없으면 requirements에 추가할 목록만 정리. 설치는 하지 않아도 됨)
   - ccx(CalculiX) 실행 파일이 시스템에서 찾아지는지 여부
2. 새 패키지 fea/ 를 만들고 fea/models.py에 아래 dataclass를 정의한다.
   (필드명과 타입은 아래 그대로. 필요하면 필드 추가는 가능하나 삭제/이름변경은 금지)

   @dataclass MeshPart:
       name: str                      # "bone", "implant", "abutment", "prosthesis"
       nodes: np.ndarray              # (N,3) float64, mm, 월드좌표
       elements: np.ndarray           # (M,4 또는 10) int64, 0-based, 전역 절점 인덱스
       element_type: str              # "C3D4" 또는 "C3D10"
       surface_tags: dict[str, np.ndarray]   # 이름 -> (K,) 요소 인덱스 또는 절점 인덱스 (3-2에서 채움)

   @dataclass MeshModel:
       parts: list[MeshPart]
       # 전역 절점 번호는 parts를 순서대로 이어 붙인 오프셋으로 관리하는 헬퍼 메서드 제공:
       # global_node_offset(part_name) -> int, total_nodes() -> int, total_elements() -> int

   @dataclass MaterialDef:
       name: str
       E: float                       # MPa
       nu: float
       density: float | None = None   # tonne/mm^3, 정적해석에서는 생략 가능

   @dataclass MaterialAssignment:
       element_material: dict[str, np.ndarray]   # part 이름 -> (M,) int, materials 리스트의 인덱스
       materials: list[MaterialDef]

   @dataclass BoundaryConditionSet:
       fixed_node_sets: list[dict]    # {"name": str, "part": str, "node_ids": np.ndarray, "dofs": (1,3)}
       loads: list[dict]              # {"name": str, "part": str, "node_ids": np.ndarray,
                                      #  "force_N": (fx,fy,fz) 합력}  -> 절점에 균등 분배
   @dataclass TieDef:
       name: str
       master_part: str
       slave_part: str
       master_faces: np.ndarray       # (K,2) [element_idx, face_id(1~4)]
       slave_nodes: np.ndarray        # (L,) 절점 인덱스(해당 part 로컬)
       position_tolerance_mm: float

   @dataclass AnalysisConfig:
       mesh_size_far_mm: float
       mesh_size_near_mm: float
       element_type: str
       ccx_path: str | None
       n_threads: int

3. fea/models.py에 JSON 직렬화/역직렬화가 필요한 설정 부분(AnalysisConfig)은
   to_dict/from_dict를 제공한다. numpy 배열은 직렬화하지 않아도 된다.
4. tests/test_models.py에 MeshModel.global_node_offset 등의 동작을 검증하는 작은 테스트를 작성하고 실행한다.

금지: Phase 1, 2의 기존 파일 수정, 새 의존성 설치, 메쉬/솔버 구현.
완료 조건: docs/PHASE3_CONTEXT.md 존재, fea/models.py 존재, pytest 통과.
```

---

## Step 3-1. Gmsh 메쉬 생성

```
[Step 3-1] Gmsh 기반 3D 사면체 메쉬 생성

먼저 docs/PHASE3_CONTEXT.md와 fea/models.py를 읽어라. 거기 정의된 계약을 바꾸지 마라.

새 파일: meshing/gmsh_mesher.py, tests/test_gmsh_mesher.py
(meshing 폴더가 이미 다른 구조로 있으면 PHASE3_CONTEXT.md에 적힌 실제 위치를 사용)

구현할 함수:

def surface_to_stl(polydata_or_trimesh, path) -> None
    vtkPolyData 또는 trimesh를 임시 STL로 저장. 삼각형화, 중복점 병합, 법선 정리 포함.

def mesh_closed_surface(stl_path: str, size_far: float, size_near: float,
                        refine_surface_stl: str | None, element_order: int = 2
                       ) -> tuple[np.ndarray, np.ndarray]
    닫힌 표면 STL로 내부 사면체 볼륨 메쉬를 만든다. (nodes (N,3), tets (M,4 또는 10))
    순서:
      gmsh.initialize(); gmsh.merge(stl); gmsh.model.mesh.classifySurfaces(angle, True, True, curveAngle)
      gmsh.model.mesh.createGeometry(); 모든 surface로 surface loop와 volume 생성
      크기장: refine_surface_stl이 있으면 해당 표면 근처에 Distance + Threshold 필드로
              size_near -> size_far 로 점진 증가 (DistMin/DistMax는 size_near의 1~5배 범위로 파라미터화)
      Mesh.Algorithm3D = HXT 또는 Delaunay (HXT 실패 시 Delaunay로 재시도)
      generate(3); setOrder(element_order)
      결과를 numpy로 추출. 반드시 gmsh.finalize()를 finally에서 호출.
    메쉬 요소 수가 지나치게 크면(기본 상한 1,500,000) 생성 전에 예상치를 로그로 남기고,
    상한 초과 시 ValueError로 중단하는 안전장치를 둔다.

def build_part_mesh(name: str, surface, size_far, size_near, refine_surface=None,
                    order=2) -> MeshPart
    위 함수를 감싸 MeshPart를 반환. element_type은 order에 따라 C3D4 / C3D10.

def gmsh_tet10_to_ccx(elements: np.ndarray) -> np.ndarray
    10절점 요소의 마지막 두 열을 교환 (Gmsh [..,8,9] -> CalculiX [..,9,8]).
    4절점이면 그대로 반환. 별도 함수로 분리해 단위 테스트한다.

주의:
- Gmsh 절점 태그는 1부터 시작하고 연속적이지 않을 수 있다. 태그 -> 0-based 연속 인덱스로 재매핑하라.
- 음의 부피 사면체가 있으면 절점 순서를 뒤집어 양의 부피로 정규화하는 함수 fix_orientation을 만들어라.
- 표면이 닫혀 있지 않으면(trimesh is_watertight False) 시도하지 말고 명확한 예외를 던져라.
  (복구는 이 Step의 범위가 아니다)

테스트 (CT 없이, 합성 형상으로):
- 단위 정육면체 또는 반지름 5mm 구를 trimesh로 만들어 STL로 저장하고 mesh_closed_surface 실행
- 총 부피(요소 부피 합)가 해석 부피와 2% 이내 일치
- 모든 요소 부피 > 0
- order=2일 때 gmsh_tet10_to_ccx 적용 후에도 요소 수가 같고 마지막 두 열만 교환됨
- refine_surface 사용 시: 표면 근처 평균 요소 크기 < 먼 곳 평균 요소 크기

완료 조건: 위 테스트 모두 통과. 실행 결과 요약을 보고.
```

---

## Step 3-2. 메쉬 품질 검사와 표면/접촉면 추출

```
[Step 3-2] 메쉬 품질 검사, 표면 및 접촉면 추출

docs/PHASE3_CONTEXT.md와 fea/models.py를 읽어라. fea/models.py의 계약을 바꾸지 마라.

새 파일: meshing/quality_checker.py, meshing/surface_extractor.py,
        tests/test_quality_checker.py, tests/test_surface_extractor.py

[quality_checker.py]
def tet_quality(nodes, elements) -> dict
    4절점 기준(2차 요소는 앞 4개 꼭짓점 사용)으로 numpy 벡터화 계산:
    - volume (부호 포함), 음수 개수
    - aspect ratio (최장 모서리 / 최단 모서리 또는 외접구 기반 중 하나를 선택하고 docstring에 명시)
    - scaled Jacobian 또는 min dihedral angle
    반환: {"n_elements", "n_negative", "min_volume", "aspect_max", "aspect_mean",
           "bad_element_ids": np.ndarray}  (임계값 초과 요소)
def quality_report_text(report) -> str      # UI 표시용 한국어 요약 문자열
def assert_mesh_ok(report, max_negative=0) -> None   # 음의 부피가 있으면 예외

[surface_extractor.py]
def boundary_faces(elements) -> np.ndarray
    사면체 요소에서 한 요소에만 속하는 면을 추출. 반환 (K, 3) 꼭짓점 인덱스, 및
    (K,2) [element_idx, face_id]. CalculiX 사면체 면 번호 규칙:
      S1: 절점 1-2-3,  S2: 1-4-2,  S3: 2-4-3,  S4: 3-4-1  (1-based 꼭짓점 기준)
    이 규칙을 코드 상수로 두고, 법선이 요소 바깥을 향하는지 테스트로 확인한다.

def faces_in_region(nodes, faces, predicate) -> np.ndarray
    predicate(centroid_xyz)가 참인 면만 선택. (평면 아래, 구 내부, 원통 내부 등
    predicate 팩토리 함수 plane_below(point, normal), sphere(center, r)도 제공)

def contact_pair_from_parts(part_a: MeshPart, part_b: MeshPart,
                            tol_mm: float, master="a"|"b") -> TieDef
    두 part의 경계면 중 서로 tol_mm 이내인 영역을 찾는다.
    scipy.spatial.cKDTree로 slave 절점 -> master 경계면 중심점 거리 질의.
    master part의 경계 면(요소 단위)과 slave part의 경계 절점 집합을 TieDef로 반환.
    정책: 일반적으로 더 거친 메쉬 쪽을 master, 더 조밀한 쪽을 slave로 하라는 옵션 master="auto" 지원.

테스트:
- 단위 정육면체 메쉬: 경계면 수가 이론값과 일치, 모든 면 법선이 외향
- 인위적으로 뒤집은 요소 1개를 넣으면 n_negative == 1
- 맞닿은 두 정육면체 part: contact_pair가 접촉면 절점을 찾고, 떨어진 두 개는 빈 결과(또는 예외)

완료 조건: 위 테스트 통과, 결과 보고.
```

---

## Step 3-3. HU → 물성 매핑

```
[Step 3-3] CT HU 기반 이질성 골 물성 매핑

docs/PHASE3_CONTEXT.md(이미지 인덱스<->월드 좌표 변환 부분)와 fea/models.py를 읽어라.

새 파일: material/bone_mapping.py, material/presets.json, tests/test_bone_mapping.py

중요: 문헌 수식의 계수는 연구마다 다르므로 코드에 상수로 박지 말고 presets.json에 두고
사용자가 편집할 수 있게 한다. 기본 프리셋은 "예시값(검증 필요)"이라고 파일 안 description에 명시한다.

presets.json 구조 예:
{
  "density_from_hu": {"type": "linear", "a": 0.0, "b": 0.001, "unit": "g/cm3", "description": "예시. 팬텀 보정값으로 교체할 것"},
  "modulus_from_density": {"type": "power", "c": 6850.0, "d": 1.49, "unit_out": "MPa", "description": "예시. 사용할 논문 값으로 교체할 것"},
  "poisson_bone": 0.3,
  "cortical_fixed": {"E": 13700.0, "nu": 0.3},
  "E_min": 10.0, "E_max": 20000.0,
  "n_bins": 30,
  "titanium": {"E": 110000.0, "nu": 0.34}
}
수식 타입은 linear, power 두 가지를 지원하고 새 타입을 추가하기 쉬운 구조로 한다.

구현할 함수:

def world_to_index(points_xyz, origin, spacing, direction) -> np.ndarray
    월드(mm) 좌표 -> 연속 이미지 인덱스. direction 행렬(3x3)을 반드시 반영.
    (Phase 1에서 이미 같은 기능이 있으면 그것을 import해서 재사용하고 중복 구현하지 말 것)

def sample_hu(volume_array, points_xyz, origin, spacing, direction) -> np.ndarray
    scipy.ndimage.map_coordinates(order=1) 로 요소 중심점의 HU를 샘플링.
    배열 축 순서(z,y,x vs x,y,z)에 주의. 볼륨 밖은 HU=-1000으로 처리하고 경고 로그.

def element_centroids(nodes, elements) -> np.ndarray   # 앞 4개 꼭짓점 평균이면 충분

def assign_bone_materials(part: MeshPart, volume_array, geometry_info,
                          cortical_mask_array | None, preset: dict
                         ) -> MaterialAssignment
    1) 요소 중심점 HU 샘플링
    2) cortical_mask가 있으면 해당 위치가 피질골인 요소는 고정 물성(cortical_fixed)을 사용,
       나머지는 HU -> 밀도 -> E 로 계산.
       cortical_mask가 없으면 모든 요소를 HU 기반으로 계산.
    3) E를 [E_min, E_max]로 clip
    4) n_bins개 구간으로 양자화(E를 로그 또는 선형 등간격 bin 중심값으로)하여 재료 개수를 줄인다.
       (요소마다 별도 재료를 만들면 .inp가 거대해지고 느려진다)
    5) MaterialAssignment 반환 (materials 이름 규칙: "BONE_000", "BONE_001", ..., "CORTICAL")

def titanium_assignment(part: MeshPart, preset) -> MaterialAssignment
    임플란트/지대주 등 균질 재료용.

테스트 (CT 없이):
- 합성 볼륨: x방향으로 HU가 0에서 1000까지 선형 증가하는 100x20x20 배열, spacing (0.5,0.5,0.5), origin 비영(非零)
- 요소 중심점 샘플 HU가 해석값과 1 HU 이내 일치 (origin과 direction이 identity가 아닌 경우도 포함)
- 양자화 후 재료 개수 <= n_bins, 요소별 E 오차(양자화 오차)가 구간폭 이내
- cortical_mask 적용 시 해당 요소만 고정 물성
- 볼륨 밖 점이 있어도 예외 없이 동작

완료 조건: 테스트 통과, 결과 보고. 프리셋 계수는 예시값이므로 사용자가 교체해야 한다는 점을 보고에 포함.
```

---

## Step 3-4. 경계조건과 하중 모델

```
[Step 3-4] 경계조건/하중 데이터 모델과 선택 로직 (UI 없음)

docs/PHASE3_CONTEXT.md, fea/models.py, meshing/surface_extractor.py를 읽어라.

새 파일: fea/boundary_conditions.py, tests/test_boundary_conditions.py

구현:

def fix_by_plane(part: MeshPart, plane_point, plane_normal, side="below",
                 tol_mm=0.01, dofs=(1,3), name="FIX_BASE") -> dict
    평면의 한쪽(또는 평면 위) 절점 집합을 고정. 반환값은 BoundaryConditionSet.fixed_node_sets의 한 항목.
    치과 모델에서는 뼈 블록의 하단/측면 절단면을 고정하는 용도.

def fix_by_region(part, predicate, dofs, name) -> dict
    임의 predicate로 절점 선택.

def load_on_region(part, predicate_or_node_ids, force_vector_N, name) -> dict
    선택된 절점들에 합력 force_vector_N을 절점 수로 균등 분배한 하중 항목.
    (분배 방식은 docstring에 명시. 집중하중 대신 면적 가중 분배를 쓰려면 후속 확장 포인트로 주석)

def load_from_angle(magnitude_N, axis_direction, tilt_deg, tilt_azimuth_deg) -> np.ndarray
    axis_direction(치아 장축, 아래로 향하는 압축방향)을 기준으로 tilt_deg만큼 기울어진 합력 벡터.
    예: 수직 100N, 30도 경사 등. 단위벡터 정규화 포함.

def default_load_presets() -> dict
    {"axial_100N": {...}, "oblique_30deg_100N": {...}}  편집 가능한 예시 프리셋.
    (실제 하중 크기는 연구 목적에 따라 사용자가 정한다는 주석 포함)

def validate_bc(model: MeshModel, bc: BoundaryConditionSet) -> list[str]
    - 고정 절점이 0개이면 경고 (강체 운동 -> 솔버 실패)
    - 고정 DOF가 부족해 강체 운동 가능성 있으면 경고 (최소한 3개의 비공선 절점을 3방향 고정 또는 평면 고정)
    - 하중 절점이 0개이면 에러
    - 하중 합력 크기가 0이면 에러
    문자열 리스트(한국어)로 반환. 빈 리스트면 통과.

테스트:
- 단위 정육면체 메쉬(합성): 하단 평면 고정 절점 수가 기대값과 일치, 상단 하중 분배 후 절점별 힘의 합 == 합력
- load_from_angle: 크기 보존, tilt 0이면 axis와 평행
- validate_bc: 고정 없는 경우 경고 발생, 정상 경우 빈 리스트

완료 조건: 테스트 통과, 결과 보고.
```

---

## Step 3-5. CalculiX .inp 작성기

```
[Step 3-5] CalculiX 입력 파일 작성기

docs/PHASE3_CONTEXT.md, fea/models.py, meshing/gmsh_mesher.py의 gmsh_tet10_to_ccx,
meshing/surface_extractor.py의 TieDef 관련 함수를 읽어라.

새 파일: solver/calculix_writer.py, tests/test_calculix_writer.py

함수:
def write_inp(path: str, model: MeshModel, materials: dict[str, MaterialAssignment],
              bc: BoundaryConditionSet, ties: list[TieDef],
              title: str = "dental_fea") -> None

.inp 구성 순서와 규칙:
1) *HEADING
2) *NODE: 모든 part의 절점을 전역 번호(1-based)로 출력. 형식 "id, x, y, z" (소수 9자리)
3) *ELEMENT, TYPE=C3D10(또는 C3D4), ELSET=EALL_<PART>: part별로 요소 번호가 겹치지 않게 전역 오프셋 적용.
   C3D10은 gmsh_tet10_to_ccx를 적용한 순서로 출력. 한 줄에 최대 16개 데이터 규칙을 지키되,
   C3D10 요소(11개 값)는 CalculiX의 연속 줄 규칙(줄 끝 쉼표)을 따른다. 어떤 방식을 쓰는지 코드 주석에 명시.
4) 재료별 요소 세트: *ELSET, ELSET=<재료이름>  (요소 번호 나열, 한 줄 16개)
   *MATERIAL, NAME=...  / *ELASTIC (E, nu)  / 필요 시 *DENSITY
   *SOLID SECTION, ELSET=..., MATERIAL=...
5) 절점 세트: *NSET, NSET=FIX_xxx, *NSET, NSET=LOAD_xxx
6) *SURFACE 정의와 *TIE:
   master: *SURFACE, NAME=<MASTER>, TYPE=ELEMENT  후 "요소번호, S<face>" 줄들
   slave:  *SURFACE, NAME=<SLAVE>, TYPE=NODE        후 절점 번호 줄들
   *TIE, NAME=<name>, POSITION TOLERANCE=<tol>
   <SLAVE>, <MASTER>
7) *STEP / *STATIC / *BOUNDARY (예: FIX_BASE, 1, 3) / *CLOAD (절점별 분배된 힘) /
   *NODE FILE (U) / *EL FILE (S, E) / *END STEP

핵심 주의:
- 모든 번호는 전역 1-based. part 로컬 인덱스 -> 전역 번호 변환 실수가 가장 흔한 버그이므로
  변환 함수를 하나만 두고 모든 곳에서 재사용한다.
- 키워드는 대문자, 파라미터 사이 쉼표 규칙을 지킨다.
- 파일 인코딩은 ASCII, 줄바꿈은 \n.
- 대형 메쉬를 위해 문자열을 리스트로 모은 뒤 한 번에 쓰거나 스트리밍 쓰기를 사용한다 (+= 반복 금지).

테스트 (CalculiX 실행 없이 텍스트 검증):
- 합성 2요소/작은 메쉬로 write_inp 실행 후 파일을 다시 파싱:
  *NODE 개수, *ELEMENT 개수, 재료 개수, *BOUNDARY/*CLOAD 존재, *TIE 존재 확인
- C3D10 요소의 절점 순서가 변환 규칙(마지막 두 개 교환)대로 출력되는지 확인
- CLOAD의 합이 입력 합력과 일치
- ccx가 설치되어 있으면 (shutil.which("ccx")): 단위 정육면체 압축 문제를 실제로 실행해
  .frd 생성까지 확인 (없으면 pytest.skip)

완료 조건: 텍스트 검증 테스트 통과. ccx 실행 테스트는 환경에 따라 skip 가능하며 skip 여부를 보고.
```

---

## Step 3-6. 솔버 실행기

```
[Step 3-6] CalculiX 비동기 실행기

docs/PHASE3_CONTEXT.md와 fea/models.py를 읽어라.

새 파일: solver/runner.py, tests/test_runner.py

구현 (PySide6 QProcess 기반, UI 위젯은 import하지 않고 QObject만 사용):

class CalculixRunner(QObject):
    signals: started(), progress(int percent, str message), log_line(str),
             finished(bool success, str frd_path, str message), error(str)

    def __init__(self, ccx_path: str | None, n_threads: int = 4)
    def run(self, inp_path: str) -> None
        - ccx 경로 결정: 인자 > config > shutil.which("ccx")/"ccx_static"/"ccx_2.xx".
          못 찾으면 error 시그널과 한국어 안내 메시지(설치/경로 지정 방법)
        - 작업 디렉터리는 inp가 있는 폴더, 인자는 확장자를 뺀 job 이름
        - 환경변수 OMP_NUM_THREADS, CCX_NPROC_EQUATION_SOLVER 설정
        - stdout/stderr를 줄 단위로 log_line에 전달
        - 진행률: stdout과 <job>.sta 파일의 increment/iteration 줄을 폴링(QTimer 1초)해서
          추정. 선형 정적은 단계가 거의 없으므로 "해 구하는 중" 같은 불확정 진행(0, 0)도 허용
        - 종료 시: exitCode 0이고 <job>.frd 존재 여부 확인 -> finished(True/False, frd_path, msg)
        - 로그에서 "*ERROR" 문자열을 감지하면 해당 줄과 이전 3줄을 오류 메시지로 요약
    def cancel(self) -> None  : 프로세스 terminate 후 필요 시 kill, finished(False, ..., "사용자 취소")

테스트:
- ccx 경로를 지정하지 않고 PATH에도 없으면 error 시그널이 나오는지 (QCoreApplication 이벤트 루프로 확인)
- ccx 대신 간단한 가짜 실행 스크립트(파이썬)를 지정해 stdout 전달, 정상 종료, 비정상 종료,
  cancel 동작을 검증 (가짜 스크립트가 .frd를 생성하도록 해서 finished(True) 확인)
- 실제 ccx가 있으면 Step 3-5의 정육면체 압축 입력을 실행해 성공 확인 (없으면 skip)

완료 조건: 테스트 통과, 결과 보고.
```

---

## Step 3-7. UI 연동

```
[Step 3-7] Phase 3 기능을 기존 UI에 연결

docs/PHASE3_CONTEXT.md의 "좌측 패널 구조"와 "3D 뷰어 창 액터 관리" 부분을 먼저 읽어라.
Phase 1, 2의 UI 구조와 시그널 체계를 존중하고 새 탭 하나만 추가하는 방식으로 진행한다.

새 파일: ui/fea_panel.py (좌측 패널에 추가될 탭 위젯), ui/fea_controller.py (UI와 로직 사이 중계),
        fea/workers.py (QThread 작업자)
기존 파일 수정은 새 탭 등록과 컨트롤러 생성에 필요한 최소한으로 한다.

패널 구성 (좌측 패널의 "FEA" 탭, 4개 섹션을 QToolBox 또는 접이식 그룹으로):
1. 메쉬: 요소 종류(C3D4/C3D10), 먼 영역 크기, 임플란트 근처 크기, [메쉬 생성] 버튼,
   결과 요약(절점/요소 수, 품질 보고 문자열)
2. 물성: 프리셋 선택(presets.json 불러오기/다른 파일 선택), bin 개수, [물성 할당] 버튼,
   E 분포 히스토그램 또는 min/max/평균 라벨
3. 경계조건: 하단 고정면(평면 위치 슬라이더 또는 현재 슬라이스 위치 사용),
   하중 프리셋 선택(axial/oblique), 하중 크기/기울기 입력, 하중 적용 면 선택 방식(우선은 "보철 상단 N mm 영역"
   같은 단순 predicate), [검증] 버튼 -> validate_bc 결과 표시
4. 해석: ccx 경로(찾기 버튼, config 저장), 스레드 수, [.inp 저장], [해석 실행], [취소], 진행바, 로그창

백그라운드 실행:
- 메쉬 생성은 QThread worker에서 실행하고 완료 시 MeshModel을 컨트롤러에 전달.
  UI 스레드에서 gmsh를 호출하지 않는다. (gmsh Python API는 메인 스레드 제약이 있을 수 있으므로
  문제가 보이면 multiprocessing/subprocess로 격리하는 방식으로 바꾸고 이유를 보고할 것)
- 해석 실행은 CalculixRunner(QProcess)를 그대로 사용

3D 창 표시 (별도 3D 뷰어 창의 기존 API로 액터 추가):
- 메쉬 표면 표시 (part별 색), 단면 클리핑 평면 슬라이더(선택 기능)
- 물성 할당 후: 요소별 E를 cell data로 하는 컬러맵 표시 전환 토글
- 경계조건: 고정 절점은 파란 점, 하중은 화살표 글리프(합력 방향)
- 결과(Phase 4)는 이번 Step 범위 밖. 해석 완료 시 .frd 경로만 상태 관리자에 저장

상태 관리:
- 전역 상태 관리자에 mesh_model, material_assignment, bc_set, ties, last_frd_path 슬롯을 추가
  (기존 API 스타일을 따를 것)
- 상류 데이터가 바뀌면(임플란트 이동, 분할 재실행) 메쉬/물성/BC를 무효화하고 패널에 "다시 생성 필요" 표시

테스트:
- pytest-qt 또는 QT_QPA_PLATFORM=offscreen 으로 패널이 생성되고 버튼 활성/비활성 상태가
  전제조건(메쉬 없음 -> 물성 버튼 비활성 등)에 맞게 바뀌는지 확인
- 컨트롤러 단위 테스트: 합성 입력으로 메쉬 -> 물성 -> BC -> .inp 저장까지 headless로 호출 성공

완료 조건: 앱이 실행되고 FEA 탭이 보이며, 합성 입력 기준 위 테스트 통과. 실제 CT로 도는지는 Step 3-8에서 확인한다.
```

---

## Step 3-8. 합성 모델 정량 검증과 실제 CT 통합 테스트

```
[Step 3-8] 파이프라인 검증

목적: 코드가 돌아가는지가 아니라 결과가 맞는지를 확인한다.

새 파일: tests/test_pipeline_validation.py, scripts/run_pipeline_headless.py

[A. 해석해와 비교 (ccx 필요, 없으면 skip)]
1) 균질 정육면체 압축
   - 10 x 10 x 10 mm, E = 1000 MPa, nu = 0.3, 하단 z방향 고정(나머지 강체 모션 방지용 최소 구속),
     상단에 총 100 N 압축
   - 이론: 응력 = F/A = 1 MPa, 상단 변위 = 응력 * L / E = 0.01 mm (프와송 구속 영향을 피하도록 구속 설계에 주의)
   - 판정: 상단 평균 변위와 평균 수직응력이 이론값과 5% 이내
2) 외팔보
   - 폭 2, 높이 2, 길이 20 mm, 끝단 하중 P, E = 1000 MPa 로 처짐 PL^3/(3EI) 와 비교, 10% 이내
     (사면체 C3D10 기준. C3D4면 오차가 더 크니 결과를 별도 기록)
3) TIE 검증
   - 서로 맞닿은 두 정육면체 part를 TIE로 연결해 하나의 블록과 같은 변위가 나오는지 (3% 이내)
4) 결과 파일에서 변위를 읽는 최소 .frd 파서가 필요하면 postprocess/result_reader.py에
   변위 읽기 함수만 간단히 구현 (Phase 4에서 확장 예정이므로 인터페이스는 단순하게)

[B. 헤드리스 파이프라인 스크립트]
scripts/run_pipeline_headless.py:
  인자: DICOM 폴더, 임플란트 STL, 출력 폴더, 프리셋 JSON
  흐름: 로드 -> (기존 Phase 2 함수로) 분할 + 배치 + Boolean -> 메쉬 -> 물성 -> BC(기본 프리셋) -> .inp 저장 -> (옵션) ccx 실행
  각 단계 소요 시간과 요소 수를 로그로 출력

[C. 점검 리포트]
docs/PHASE3_REPORT.md에 다음을 정리:
- A의 모든 검증 결과(이론값, 계산값, 오차)
- 알려진 한계 (TIE 가정, 선형해석, 하중 분배 방식, 메쉬 크기 의존성)
- 사용자가 직접 확인/교체해야 할 값 목록 (HU-밀도 계수, 밀도-탄성계수 계수, 하중 크기, 고정 조건)

완료 조건: A 검증 통과(또는 ccx 미설치로 skip 사유 명시), 리포트 작성.
```

---

## 4. 막혔을 때 점검표

| 증상 | 가장 흔한 원인 | 확인 위치 |
|---|---|---|
| ccx에서 negative jacobian | C3D10 마지막 두 절점 순서, 또는 요소 방향 | gmsh_tet10_to_ccx, fix_orientation |
| 해석 결과가 비정상적으로 작은 변위 | 단위 불일치 (m 단위 STL, GPa 입력 등) | 단위계, STL 스케일 |
| TIE 후 변위가 서로 따로 놂 | position tolerance가 너무 작거나 master/slave 면 정의 오류 | TieDef 생성, face 번호 |
| 강체 운동 에러 | 구속 부족 | validate_bc |
| 물성이 전부 같은 값 | HU 샘플링 좌표계(축 순서, direction) 오류 | sample_hu 합성 볼륨 테스트 |
| 메쉬 생성 실패/매우 느림 | STL이 닫혀 있지 않음, 자기교차, 크기장 너무 조밀 | watertight 검사, 요소 수 상한 |
| gmsh가 UI와 함께 크래시 | 스레드 제약 | subprocess 격리로 전환 |

## 5. 진행 팁

- Step마다 새 세션을 열고 "docs/PHASE3_CONTEXT.md와 fea/models.py를 먼저 읽어라"를 유지한다. 맥락이 길어질수록 Flash가 계약을 어기기 쉽다.
- 에이전트가 테스트를 "통과할 것이다"라고만 말하고 실행하지 않는 경우가 있으므로, 보고에 pytest 출력이 있는지 확인한다.
- 3-1에서 STL 기반 메쉬가 계속 실패하면 대안으로 pyvista의 tetgen 연동 또는 Boolean 이전의 voxel 기반 표면 재생성을 검토하고, 이 대안은 3-1 안에서만 교체 가능하도록 meshing 모듈 인터페이스(표면 입력, MeshPart 출력)를 유지한다.
