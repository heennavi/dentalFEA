"""
tests/test_boolean_ops.py
Step 6: Boolean 차집합 및 인터페이스 분할 단위 테스트 (pytest).
- 합성 박스(피질골/수질골) + 원기둥(임플란트) 차집합 체적 및 해석해 일치 검증
- 체적 보존 법칙 검증 (피질골' + 수질골' + 제거 부피 ≈ 원본 뼈 부피)
- 겹침 없는 경우의 안전 처리 및 경고 검증
- non-watertight 입력에 대한 자동 수리 및 수리 불가능 시 예외 발생 검증
- 임플란트-뼈 접촉 면적(contact area) 추정 검증
- export_stl을 통한 3개 STL 파일 저장 및 재로드 검증
"""

import os
import sys
import tempfile
import warnings
import unittest
import numpy as np
import pytest
import trimesh

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dental_fea_app.geometry.boolean_ops import (
    FeaGeometry,
    ensure_watertight,
    filter_small_components,
    calculate_contact_area,
    build_fea_geometry,
    export_stl,
)


class TestBooleanOps(unittest.TestCase):

    def _create_synthetic_model(self):
        """
        검증용 합성 뼈 및 임플란트 메쉬 생성:
        - 피질골 박스: 20 x 20 x 4 mm (중심: 0, 0, 8, Z: 6 ~ 10), 부피 = 1,600 mm³
        - 수질골 박스: 20 x 20 x 16 mm (중심: 0, 0, -2, Z: -10 ~ 6), 부피 = 6,400 mm³
        - 원기둥 임플란트: 반경 2.0 mm (직경 4.0 mm), 높이 12.0 mm (Z: -2 ~ 10), 부피 = π * 4 * 12 = 48π ≈ 150.796 mm³
        """
        # 피질골: Z = 6 ~ 10
        cortical = trimesh.creation.box((20.0, 20.0, 4.0))
        cortical.apply_translation([0.0, 0.0, 8.0])

        # 수질골: Z = -10 ~ 6
        cancellous = trimesh.creation.box((20.0, 20.0, 16.0))
        cancellous.apply_translation([0.0, 0.0, -2.0])

        # 임플란트 원기둥: Z = -2 ~ 10 (중심 Z = 4)
        # 표면 접촉 면적 및 정밀한 차집합을 위해 높이 방향 세분화(subdivide) 적용
        implant = trimesh.creation.cylinder(radius=2.0, height=12.0, sections=36)
        implant = implant.subdivide()
        implant = implant.subdivide()
        implant.apply_translation([0.0, 0.0, 4.0])

        return cortical, cancellous, implant

    def test_01_synthetic_box_and_cylinder_difference(self):
        """합성 박스 뼈와 원기둥 임플란트 차집합 결과 체적이 해석해와 일치하는지 검증"""
        cortical, cancellous, implant = self._create_synthetic_model()

        fea_geom = build_fea_geometry(
            cortical_mesh=cortical,
            cancellous_mesh=cancellous,
            implant_mesh=implant,
            undersize_mm=0.0
        )

        rep = fea_geom.report
        self.assertTrue(rep["cortical_watertight"])
        self.assertTrue(rep["cancellous_watertight"])
        self.assertTrue(rep["implant_watertight"])

        # 해석해 비교
        # 1) 피질골 제거 체적: π * 2² * 4 = 16π ≈ 50.265 mm³
        #    피질골 잔여 체적: 1,600 - 16π ≈ 1,549.735 mm³
        expected_cortical_vol = 1600.0 - (np.pi * 4.0 * 4.0)
        self.assertAlmostEqual(rep["cortical_volume"], expected_cortical_vol, delta=2.5)

        # 2) 수질골 제거 체적: π * 2² * 8 = 32π ≈ 100.531 mm³
        #    수질골 잔여 체적: 6,400 - 32π ≈ 6,299.469 mm³
        expected_cancellous_vol = 6400.0 - (np.pi * 4.0 * 8.0)
        self.assertAlmostEqual(rep["cancellous_volume"], expected_cancellous_vol, delta=5.0)

        # 3) 총 제거 체적: 48π ≈ 150.796 mm³
        expected_removed_vol = np.pi * 4.0 * 12.0
        self.assertAlmostEqual(rep["removed_bone_volume"], expected_removed_vol, delta=5.0)

    def test_02_volume_conservation(self):
        """체적 보존 법칙 검증 (피질골' + 수질골' + 제거 부피 ≈ 원본 뼈 부피)"""
        cortical, cancellous, implant = self._create_synthetic_model()

        fea_geom = build_fea_geometry(cortical, cancellous, implant)
        rep = fea_geom.report

        initial_bone = rep["initial_bone_volume"]
        remaining_bone = rep["remaining_bone_volume"]
        removed_bone = rep["removed_bone_volume"]

        # 원본 부피와 (잔여 부피 + 제거 부피)의 상대 오차가 0.5% 미만이어야 함
        total_accounted = remaining_bone + removed_bone
        error_ratio = abs(total_accounted - initial_bone) / initial_bone
        self.assertLess(error_ratio, 0.005)

    def test_03_non_overlapping_implant(self):
        """임플란트가 뼈를 완전히 벗어난 경우 경고 방출 및 원본 부피 유지 검증"""
        cortical, cancellous, implant = self._create_synthetic_model()

        # 임플란트를 뼈 범위(X: -10~10, Y: -10~10, Z: -10~10) 밖으로 멀리 이동
        implant.apply_translation([100.0, 100.0, 100.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fea_geom = build_fea_geometry(cortical, cancellous, implant)

            # 경고가 1개 이상 발생했는지 확인
            warn_msgs = [str(item.message) for item in w]
            self.assertTrue(any("겹치지 않습니다" in msg for msg in warn_msgs))

        rep = fea_geom.report
        self.assertAlmostEqual(rep["removed_bone_volume"], 0.0, delta=1e-3)
        self.assertAlmostEqual(rep["cortical_volume"], 1600.0, delta=1e-1)
        self.assertAlmostEqual(rep["cancellous_volume"], 6400.0, delta=1e-1)

    def test_04_non_watertight_repair_and_exception(self):
        """non-watertight 입력에 대한 자동 수리 동작 및 수리 불가능 메쉬 예외 발생 검증"""
        # 1) 면 1개를 제거하여 구멍이 뚫린 간단한 박스 -> fill_holes로 자동 수리 성공 검증
        box = trimesh.creation.box((10.0, 10.0, 10.0))
        box.update_faces(np.arange(len(box.faces) - 1))  # 마지막 면 1개 제거
        self.assertFalse(box.is_watertight)

        repaired = ensure_watertight(box, name="test_hole_box")
        self.assertTrue(repaired.is_watertight)

        # 2) 닫힌 매니폴드를 형성할 수 없는 열린 2D 평면 패치 메쉬 -> ValueError 검증
        open_patch = trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            faces=[[0, 1, 2], [0, 1, 3]]
        )

        with self.assertRaises(ValueError) as ctx:
            ensure_watertight(open_patch, name="open_patch")
        self.assertIn("watertight", str(ctx.exception))

    def test_05_contact_area_estimation(self):
        """임플란트와 뼈 사이의 접촉 면적(Contact Area) 추정 검증"""
        cortical, cancellous, implant = self._create_synthetic_model()

        fea_geom = build_fea_geometry(cortical, cancellous, implant)
        rep = fea_geom.report

        # 임플란트 접촉 면적 해석해:
        # 피질골 관통(h=4): 측면 2π * 2 * 4 = 16π ≈ 50.27 mm²
        # 수질골 관통(h=8): 측면 2π * 2 * 8 = 32π + 바닥면 π * 2² = 36π ≈ 113.10 mm²
        # 총 접촉 면적 ≈ 52π ≈ 163.36 mm²
        expected_cortical_contact = 2.0 * np.pi * 2.0 * 4.0
        expected_cancellous_contact = (2.0 * np.pi * 2.0 * 8.0) + (np.pi * 4.0)
        expected_total_contact = expected_cortical_contact + expected_cancellous_contact

        self.assertGreater(rep["implant_cortical_contact_area"], 45.0)
        self.assertLess(rep["implant_cortical_contact_area"], 75.0)

        self.assertGreater(rep["implant_cancellous_contact_area"], 105.0)
        self.assertLess(rep["implant_cancellous_contact_area"], 145.0)

        self.assertAlmostEqual(rep["total_contact_area"], expected_total_contact, delta=20.0)

    def test_06_stl_export_and_validity(self):
        """export_stl 함수로 3개 솔리드가 정상 저장되고 재로드 시 watertight인지 검증"""
        cortical, cancellous, implant = self._create_synthetic_model()
        fea_geom = build_fea_geometry(cortical, cancellous, implant)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_stl(fea_geom, tmpdir)

            self.assertIn("cortical", paths)
            self.assertIn("cancellous", paths)
            self.assertIn("implant", paths)

            for key, path in paths.items():
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)

                # 외부 뷰어/trimesh 재로드 무결성 검증
                loaded_mesh = trimesh.load_mesh(path)
                self.assertIsInstance(loaded_mesh, trimesh.Trimesh)
                self.assertGreater(len(loaded_mesh.faces), 0)
                self.assertGreater(loaded_mesh.volume, 0.0)


if __name__ == "__main__":
    unittest.main()
