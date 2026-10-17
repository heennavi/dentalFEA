이 프로젝트는 PySide6 + VTK 기반 치과 임플란트 유한요소해석(FEA) 데스크톱 앱이다.

[좌표계와 단위]
- 길이 mm, 힘 N, 응력 MPa. 모든 기하 데이터는 DICOM과 동일한 world 좌표계(LPS, mm)를 쓴다.
- 볼륨(vtkImageData)의 origin / spacing / direction을 항상 보존한다. 마스크는 원본 볼륨과 동일한 geometry를 가진다.
- 인덱스(ijk)와 world(xyz) 변환은 한 곳의 유틸 함수로만 한다. 곳곳에서 직접 계산하지 않는다.

[구조 원칙]
- 순수 로직(numpy/vtk/trimesh)과 UI(PySide6)를 분리한다. 로직 모듈은 PySide6를 import하지 않는다.
- 로직 모듈은 단독으로 pytest 가능해야 한다. 합성 데이터(가짜 볼륨, 단순 STL)로 테스트를 만든다.
- 기존 Phase 1 파일은 요청된 범위 외에는 수정하지 않는다. 수정이 필요하면 이유와 diff 범위를 먼저 알린다.
- 뷰 간 상태 전달은 기존 state manager의 signal/slot 방식을 따른다. 새로운 전역 변수를 만들지 않는다.
- 무거운 연산(분할, 표면 추출, Boolean)은 QThread/QRunnable로 돌려 UI를 멈추지 않게 하고, 진행 상태를 signal로 알린다.

[작업 방식]
- 한 번에 한 Step만 구현한다. 요청받지 않은 기능을 추가하지 않는다.
- 새 파일에는 타입 힌트와 docstring(입출력 단위 명시)을 쓴다.
- 구현 후 pytest를 실행해 통과를 확인하고, 실행한 명령과 결과를 요약해서 보고한다.
- 외부 라이브러리를 추가할 때는 requirements.txt에 반영한다.
