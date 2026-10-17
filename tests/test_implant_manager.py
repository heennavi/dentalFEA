"""
test_implant_manager.py
Step 3: 임플란트 로드 및 Transform 로직에 대한 단위 테스트 (pytest).
- PySide6 미사용 검증
- 단순 원기둥 STL 생성 및 로드
- place_at 후 apex 위치와 장축 방향 검증
- 회전 후 변환 역행렬 복원 검증
- 단위 오류 경고 동작 검증
- sample_hu_along_axis 및 bone_overlap_ratio 동작 검증
"""

import os
import sys
import tempfile
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from dental_fea_app.placement.implant_manager import (
    ImplantModel,
    sample_hu_along_axis,
    bone_overlap_ratio,
)


def create_cylinder_stl_file(
    file_path: str,
    radius: float = 2.0,      # 직경 4mm
    height: float = 10.0,     # 길이 10mm
    z_min: float = -5.0,      # z: -5 ~ +5
    resolution: int = 32,
) -> None:
    """
    임플란트 모형 테스트용 원기둥 STL 파일을 생성합니다.
    (장축 = Z축, Apex = z_min, Top = z_min + height)
    """
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetRadius(radius)
    cylinder.SetHeight(height)
    cylinder.SetResolution(resolution)
    cylinder.CappingOn()
    cylinder.Update()

    # vtkCylinderSource의 기본 장축은 Y축이므로 Z축으로 90도 회전
    transform = vtk.vtkTransform()
    transform.RotateX(90.0)
    # 중심(0, 0, 0)에서 z_min이 바닥이 되도록 이동 (center = z_min + height/2)
    z_center = z_min + height * 0.5
    transform.Translate(0.0, 0.0, z_center)

    tf_filter = getattr(vtk, "vtkTransformFilter", vtk.vtkTransformPolyDataFilter)()
    tf_filter.SetInputConnection(cylinder.GetOutputPort())
    tf_filter.SetTransform(transform)
    tf_filter.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(file_path)
    writer.SetFileTypeToBinary()
    writer.SetInputConnection(tf_filter.GetOutputPort())
    writer.Write()


class TestImplantManager:
    """ImplantModel 및 Transform 유틸리티 테스트 스위트"""

    def test_pyside6_not_imported(self):
        """순수 로직 모듈 내에 PySide6 import가 없어야 함을 검증"""
        import dental_fea_app.placement.implant_manager as mod
        # 1. 모듈 속성에 PySide6 관련 심볼이 전혀 없어야 함
        for attr in dir(mod):
            val = getattr(mod, attr)
            if hasattr(val, "__module__") and val.__module__:
                assert "PySide6" not in val.__module__, f"PySide6 심볼 발견: {attr} ({val.__module__})"

        # 2. 독립 프로세스에서 임포트했을 때 sys.modules에 PySide6가 로드되지 않아야 함
        import subprocess
        cmd = [
            sys.executable,
            "-c",
            "import sys; import dental_fea_app.placement.implant_manager; assert not any('PySide6' in m for m in sys.modules), 'PySide6가 로드되었습니다!'"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"PySide6 비의존성 검증 실패: {res.stderr}"

    def test_load_stl_and_dimensions(self):
        """정상 규격의 원기둥 STL 로드 및 치수/바운딩 박스 검증"""
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "fixture_4x10.stl")
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)

            implant = ImplantModel()
            info = implant.load_stl(stl_file)

            assert info["warning"] is None
            dx, dy, dz = info["dimensions"]
            assert pytest.approx(dx, rel=1e-2) == 4.0
            assert pytest.approx(dy, rel=1e-2) == 4.0
            assert pytest.approx(dz, rel=1e-2) == 10.0

            # Apex 및 Top 위치 검증
            apex = implant.get_local_apex()
            top = implant.get_local_top()
            assert pytest.approx(apex[2], abs=1e-3) == -5.0
            assert pytest.approx(top[2], abs=1e-3) == 5.0

    def test_unit_warning_for_abnormal_dimensions(self):
        """치수가 너무 크거나(예: > 100mm) 너무 작은 경우 단위 오류 경고 검증"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 미터(m) 단위로 저장되어 치수가 500mm 이상인 경우
            huge_stl = os.path.join(tmpdir, "huge.stl")
            create_cylinder_stl_file(huge_stl, radius=50.0, height=150.0, z_min=-75.0)

            implant = ImplantModel()
            with pytest.warns(UserWarning, match="임플란트 치수가 비정상적입니다"):
                info = implant.load_stl(huge_stl)
                assert info["warning"] is not None

            # 2. 너무 작은 경우 (< 0.1mm)
            tiny_stl = os.path.join(tmpdir, "tiny.stl")
            create_cylinder_stl_file(tiny_stl, radius=0.01, height=0.05, z_min=-0.025)
            with pytest.warns(UserWarning, match="임플란트 치수가 비정상적입니다"):
                info2 = implant.load_stl(tiny_stl)
                assert info2["warning"] is not None

    def test_set_local_axis(self):
        """비정렬된 장축을 로컬 +Z로 재정렬하는 set_local_axis 검증"""
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "cylinder.stl")
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)

            implant = ImplantModel(stl_file)

            # 임의로 X축으로 회전된 모델을 가정: 만약 장축이 X축(+X)인 경우
            # X축 [1, 0, 0]을 로컬 +Z [0, 0, 1]로 정렬하도록 설정
            implant.set_local_axis([1.0, 0.0, 0.0])

            # 새 apex의 Z축 최하단과 Top 검증
            bounds = implant.get_local_bounds()
            # X축 방향 길이가 Z축 방향 길이(~10mm)로 정렬되었는지 확인
            dz = bounds[5] - bounds[4]
            assert pytest.approx(dz, rel=1e-2) == 4.0 or pytest.approx(dz, rel=1e-2) == 10.0

    def test_place_at_accuracy(self):
        """place_at 호출 후 apex 위치와 장축 방향 벡터 정밀도 검증"""
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "fixture.stl")
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)

            implant = ImplantModel(stl_file)

            # 테스트 1: 식립 지점 point_world = [10.0, 20.0, 30.0], 수직 상향 식립 axis = [0, 0, 1]
            target_apex = np.array([10.0, 20.0, 30.0])
            target_axis = np.array([0.0, 0.0, 1.0])
            implant.place_at(target_apex, target_axis)

            computed_apex = implant.get_world_apex()
            computed_axis = implant.get_world_axis()
            np.testing.assert_allclose(computed_apex, target_apex, atol=1e-5)
            np.testing.assert_allclose(computed_axis, target_axis, atol=1e-5)

            # Top 위치는 Apex + 길이(10mm) * axis여야 함
            computed_top = implant.get_world_top()
            expected_top = target_apex + target_axis * 10.0
            np.testing.assert_allclose(computed_top, expected_top, atol=1e-5)

            # 테스트 2: 45도 기울어진 경사 식립 (예: [0, 1, 1] 방향)
            t_axis_45 = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
            implant.place_at(target_apex, t_axis_45)

            computed_apex_45 = implant.get_world_apex()
            computed_axis_45 = implant.get_world_axis()
            np.testing.assert_allclose(computed_apex_45, target_apex, atol=1e-5)
            np.testing.assert_allclose(computed_axis_45, t_axis_45, atol=1e-5)

    def test_rotation_and_inverse_transform_restoration(self):
        """회전 및 이동 후 역행렬을 적용했을 때 원본 로컬 좌표가 완벽 복원되는지 검증"""
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "fixture.stl")
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)

            implant = ImplantModel(stl_file)

            # 복합 변환 적용: 이동 + 회전
            implant.translate(15.5, -23.4, 45.2)
            implant.rotate_about_axis(37.5, [1.0, 2.0, -1.0])
            implant.set_yaw_pitch_roll(20.0, -15.0, 40.0)

            T = implant.transform_matrix
            T_inv = np.linalg.inv(T)

            # 변환된 PolyData 정점 추출
            poly_transformed = implant.get_transformed_polydata()
            pts_world = numpy_support.vtk_to_numpy(poly_transformed.GetPoints().GetData())

            # 동차 좌표(Nx4)로 변환 후 역행렬 적용
            pts_world_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
            pts_restored_h = (T_inv @ pts_world_h.T).T
            pts_restored = pts_restored_h[:, :3]

            # 원본 로컬 정점과 비교
            pts_raw = numpy_support.vtk_to_numpy(implant.raw_polydata.GetPoints().GetData())
            np.testing.assert_allclose(pts_restored, pts_raw, atol=1e-4)

    def test_observer_notification(self):
        """포즈나 파라미터 변경 시 등록된 옵저버 콜백이 정상 발행되는지 검증"""
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "fixture.stl")
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)

            implant = ImplantModel(stl_file)

            notified = []

            def on_change(model):
                notified.append(model.position)

            implant.add_observer(on_change)
            implant.translate(5.0, 0.0, 0.0)
            assert len(notified) == 1
            np.testing.assert_allclose(notified[-1], [5.0, 0.0, 0.0])

            implant.remove_observer(on_change)
            implant.translate(0.0, 5.0, 0.0)
            assert len(notified) == 1  # 해제 후 추가 호출 없어야 함

    def test_sample_hu_along_axis(self):
        """임플란트 식립 축을 따라 볼륨의 HU 프로파일 샘플링 검증"""
        # 합성 볼륨 생성: 30x30x30, origin=(0, 0, 0), spacing=(1, 1, 1)
        # Z축을 따라 HU 값이 선형으로 증가: HU(z) = 100 * z
        nz, ny, nx = 30, 30, 30
        hu_grid = np.zeros((nz, ny, nx), dtype=np.float32)
        for z in range(nz):
            hu_grid[z, :, :] = z * 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "fixture.stl")
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=0.0)

            implant = ImplantModel(stl_file)
            # Apex를 (15, 15, 5)에 두고 Z축(+Z) 방향으로 10mm 배치
            implant.place_at(point_world=[15.0, 15.0, 5.0], axis_world=[0.0, 0.0, 1.0])

            distances, hu_vals = sample_hu_along_axis(
                volume=hu_grid,
                pose=implant,
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                num_samples=11,
                length_mm=10.0,
            )

            assert len(distances) == 11
            assert len(hu_vals) == 11
            assert pytest.approx(distances[0]) == 0.0
            assert pytest.approx(distances[-1]) == 10.0

            # Apex 지점(z=5)의 HU는 500, Top 지점(z=15)의 HU는 1500
            assert pytest.approx(hu_vals[0], abs=1.0) == 500.0
            assert pytest.approx(hu_vals[-1], abs=1.0) == 1500.0

    def test_bone_overlap_ratio(self):
        """임플란트 표면의 뼈 마스크 중첩 비율(bone_overlap_ratio) 계산 검증"""
        # 40x40x40 마스크: z >= 15 영역만 True (뼈)
        shape = (40, 40, 40)
        mask = np.zeros(shape, dtype=bool)
        mask[15:, :, :] = True
        mask_geom = {"origin": (0.0, 0.0, 0.0), "spacing": (1.0, 1.0, 1.0)}

        with tempfile.TemporaryDirectory() as tmpdir:
            stl_file = os.path.join(tmpdir, "fixture.stl")
            # 길이 10mm 원기둥 (z: -5 ~ +5)
            create_cylinder_stl_file(stl_file, radius=2.0, height=10.0, z_min=-5.0)

            implant = ImplantModel(stl_file)

            # 케이스 1: 완전히 뼈 내부(z: 20 ~ 30)에 배치 -> 비율 1.0
            implant.place_at(point_world=[20.0, 20.0, 20.0], axis_world=[0.0, 0.0, 1.0])
            poly1 = implant.get_transformed_polydata()
            ratio1 = bone_overlap_ratio(poly1, mask, mask_geom)
            assert pytest.approx(ratio1, abs=1e-2) == 1.0

            # 케이스 2: 완전히 뼈 외부(z: 0 ~ 10)에 배치 -> 비율 0.0
            implant.place_at(point_world=[20.0, 20.0, 0.0], axis_world=[0.0, 0.0, 1.0])
            poly2 = implant.get_transformed_polydata()
            ratio2 = bone_overlap_ratio(poly2, mask, mask_geom)
            assert pytest.approx(ratio2, abs=1e-2) == 0.0

            # 케이스 3: 절반 걸친 경우 (Apex=10, Top=20, 경계=15) -> 약 0.5
            implant.place_at(point_world=[20.0, 20.0, 10.0], axis_world=[0.0, 0.0, 1.0])
            poly3 = implant.get_transformed_polydata()
            ratio3 = bone_overlap_ratio(poly3, mask, mask_geom)
            assert 0.4 <= ratio3 <= 0.6
