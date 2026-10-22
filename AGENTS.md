# 프로젝트 공통 규칙 (치과 임플란트 FEA 앱)

- 언어: 대화와 주석 설명은 한국어, 식별자와 docstring 첫 줄은 영어.
- 단위계는 항상 mm, N, MPa, tonne/mm^3. 단위 변환은 config.py 한 곳에서만.
- 모든 기하 데이터는 DICOM과 동일한 world 좌표계(LPS, mm)를 쓴다.
- 볼륨(vtkImageData)의 origin / spacing / direction을 항상 보존한다. 마스크는 원본 볼륨과 동일한 geometry를 가진다.
- Phase 1, 2, 3에서 이미 작성된 파일의 공개 함수 시그니처와 동작을 바꾸지 않는다.
  수정이 꼭 필요하면 먼저 이유를 설명하고, 변경 범위를 최소화한다.
- 새 기능은 새 파일/새 모듈에 작성한다. 기존 파일은 import 연결에 필요한 최소 수정만 한다.
- GUI 없이 import 가능해야 하는 모듈(meshing, material, solver, postprocess의 파서/계산부)은
  PySide6와 VTK 렌더링 관련 모듈(vtkRenderWindow, QVTKRenderWindowInteractor 등)을 import하지 않는다.
  데이터 구조용 vtk 클래스(vtkUnstructuredGrid 등)는 postprocess/vtk_builder.py에서만 사용한다.
- 오래 걸리는 작업(파일 파싱, 통계 계산)은 UI 스레드에서 실행하지 않는다. QThread 사용.
- fea/models.py, fea/results.py의 dataclass 필드는 삭제나 이름 변경 금지. 추가만 허용.
- 모든 새 모듈에는 pytest 테스트를 tests/ 아래에 함께 작성하고, 작업 종료 전에 실제로 실행해
  결과를 보고한다. 실행하지 않은 테스트를 통과했다고 말하지 않는다.
- 외부 실행 파일(ccx 등) 경로는 하드코딩 금지.
- 한 번에 한 Step만 작업한다. 요청 범위 밖의 리팩터링을 하지 않는다.
- 작업이 끝나면 (1) 변경/추가한 파일 목록 (2) 테스트 결과 (3) 남은 문제를 짧게 보고한다.

## Phase 4 핵심 설계 결정
- 결과 파일: ASCII .frd (Phase 3 기본 출력)
- 파싱 방식: 고정폭 슬라이싱 (split 금지, 값 사이에 공백이 없는 케이스 대응)
- 절점 번호: .frd의 절점 번호 = .inp의 전역 1-based 번호. 파서는 이를 MeshModel의 전역 0-based 인덱스로 변환 (Phase 3 전역 번호 변환 함수 재사용)
- 응력 계산: von Mises, 최대/최소 주응력은 6성분(SXX SYY SZZ SXY SYZ SZX)에서 numpy로 직접 계산
- 응력값 성격: .frd의 응력은 적분점에서 절점으로 외삽된 값. 파트 간 절점이 분리되어 있어 파트 경계에서 평균화가 섞이지 않음 (해석 한계로 리포트에 명시)
- 컬러 범위: 파트별 범위 모드를 기본 제공 (뼈 / 임플란트 / 전체). 티타늄 응력이 뼈보다 훨씬 커서 전체 범위로 보면 뼈가 단색으로 보임.
- 렌더링: 큰 메쉬는 선형 사면체(앞 4절점)로 단순화해 표시, 값은 10절점 그대로 보존 (단면 clip 성능 때문)
- VTK 2차 사면체: 셀 타입 24, 절점 순서는 CalculiX C3D10과 동일. Gmsh 10절점 사면체([0..9])를 VTK로 변환 시 CalculiX 순서([0,1,2,3,4,5,6,7,9,8])로 마지막 두 절점 교환 필수.
- 단위: mm, N, MPa. 변위 mm, 응력 MPa, 변형률 무차원 (표시 시 microstrain 변환은 UI에서만).
