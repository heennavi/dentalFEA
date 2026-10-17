"""
test_cortical_cancellous.py
Step 1: 피질골/수질골 분할 로직에 대한 단위 테스트 (pytest).
"""

import sys
import numpy as np
import pytest
import vtk

from dental_fea_app.segmentation.cortical_cancellous import (
    SegmentationResult,
    extract_bone,
    split_cortical_cancellous,
    run_segmentation,
    mask_to_vtk_image,
    vtk_image_to_mask,
)


def create_synthetic_bone_ellipsoid(
    shape=(60, 60, 60),
    spacing=(0.5, 0.5, 0.5),
    radii_mm=(12.0, 10.0, 8.0),
    cortical_thickness_mm=1.5,
    center_mm=(15.0, 15.0, 15.0),
    bg_hu=-1000.0,
    cancellous_hu=400.0,
    cortical_hu=1200.0,
) -> np.ndarray:
    """
    내부 수질골(cancellous_hu)과 외곽 껍질 피질골(cortical_hu)을 가진
    합성 3D 타원체 CT 볼륨을 생성합니다.
    """
    nz, ny, nx = shape
    sx, sy, sz = spacing
    cx, cy, cz = center_mm
    rx, ry, rz = radii_mm

    z_coords = np.arange(nz) * sz
    y_coords = np.arange(ny) * sy
    x_coords = np.arange(nx) * sx

    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    # 외부 타원체 방정식: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 + ((z-cz)/rz)^2 <= 1.0
    dist_outer_sq = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2

    # 내부 타원체 (피질골 두께만큼 수축)
    rx_in = max(1.0, rx - cortical_thickness_mm)
    ry_in = max(1.0, ry - cortical_thickness_mm)
    rz_in = max(1.0, rz - cortical_thickness_mm)
    dist_inner_sq = ((xx - cx) / rx_in) ** 2 + ((yy - cy) / ry_in) ** 2 + ((zz - cz) / rz_in) ** 2

    hu = np.full(shape, bg_hu, dtype=np.float32)

    # 뼈 외곽 (피질골)
    outer_mask = dist_outer_sq <= 1.0
    hu[outer_mask] = cortical_hu

    # 뼈 내부 (수질골)
    inner_mask = dist_inner_sq <= 1.0
    hu[inner_mask] = cancellous_hu

    return hu


class TestCorticalCancellousSegmentation:

    def test_pyside6_not_imported(self):
        """순수 로직 모듈 내에 PySide6 import가 없어야 함을 검증"""
        mod_name = "dental_fea_app.segmentation.cortical_cancellous"
        assert mod_name in sys.modules
        mod = sys.modules[mod_name]
        # 모듈 속성에 PySide6 관련 심볼이 전혀 없어야 함
        for attr in dir(mod):
            val = getattr(mod, attr)
            if hasattr(val, "__module__") and val.__module__:
                assert "PySide6" not in val.__module__, f"PySide6 심볼 발견: {attr} ({val.__module__})"

    def test_extract_bone_threshold_and_keep_largest(self):
        """임계값 이진화 및 keep_largest 옵션(떨어진 작은 덩어리 제거) 검증"""
        shape = (40, 40, 40)
        hu = np.full(shape, -1000.0, dtype=np.float32)

        # 1. 큰 메인 뼈 블록 (중심부 20x20x20)
        hu[10:30, 10:30, 10:30] = 800.0

        # 2. 작은 고립 노이즈 덩어리 (구석 3x3x3)
        hu[2:5, 2:5, 2:5] = 900.0

        spacing = (1.0, 1.0, 1.0)

        # keep_largest=True
        mask_clean = extract_bone(hu, spacing, bone_threshold=300.0, keep_largest=True, fill_holes=False)
        assert mask_clean[20, 20, 20] is True or mask_clean[20, 20, 20] == 1
        # 구석의 작은 노이즈가 제거되어야 함
        assert not np.any(mask_clean[2:5, 2:5, 2:5])

        # keep_largest=False
        mask_all = extract_bone(hu, spacing, bone_threshold=300.0, keep_largest=False, fill_holes=False)
        assert np.any(mask_all[2:5, 2:5, 2:5])

    def test_extract_bone_fill_holes(self):
        """슬라이스별 구멍 메우기(fill_holes) 동작 검증"""
        shape = (30, 30, 30)
        hu = np.full(shape, -1000.0, dtype=np.float32)

        # 속이 빈 링 형태 (외부 500, 내부 구멍 -500)
        hu[10:20, 5:25, 5:25] = 800.0
        hu[10:20, 12:18, 12:18] = -500.0  # 내부 공동

        spacing = (1.0, 1.0, 1.0)
        mask_filled = extract_bone(hu, spacing, bone_threshold=200.0, fill_holes=True)
        # 구멍이 메워졌으므로 중심부도 True여야 함
        assert np.all(mask_filled[10:20, 12:18, 12:18])

    def test_split_distance_method_invariants(self):
        """distance 방식 분할 시 겹치지 않고 합집합이 bone_mask와 일치하는 불변성 검증"""
        hu = create_synthetic_bone_ellipsoid(shape=(50, 50, 50), spacing=(0.5, 0.5, 0.5))
        spacing = (0.5, 0.5, 0.5)
        bone_mask = extract_bone(hu, spacing, bone_threshold=200.0)

        cortical, cancellous = split_cortical_cancellous(
            bone_mask, hu, spacing, method="distance", cortical_thickness_mm=1.5
        )

        # 1. 겹치지 않는다 (상호 배타적)
        overlap = cortical & cancellous
        assert not np.any(overlap), "피질골과 수질골 마스크가 겹치지 않아야 합니다."

        # 2. 합집합이 bone_mask와 같다
        union = cortical | cancellous
        assert np.array_equal(union, bone_mask), "피질골과 수질골의 합집합은 bone_mask와 완전히 일치해야 합니다."

        # 3. 둘 다 비어있지 않음
        assert np.sum(cortical) > 0
        assert np.sum(cancellous) > 0

    def test_split_threshold_method_invariants(self):
        """threshold 방식 분할 시 겹치지 않고 합집합이 일치하는 불변성 검증"""
        hu = create_synthetic_bone_ellipsoid(
            shape=(50, 50, 50),
            spacing=(0.5, 0.5, 0.5),
            cancellous_hu=350.0,
            cortical_hu=1200.0,
        )
        spacing = (0.5, 0.5, 0.5)
        bone_mask = extract_bone(hu, spacing, bone_threshold=200.0)

        cortical, cancellous = split_cortical_cancellous(
            bone_mask, hu, spacing, method="threshold", cortical_hu=700.0
        )

        assert not np.any(cortical & cancellous)
        assert np.array_equal(cortical | cancellous, bone_mask)
        assert np.sum(cortical) > 0
        assert np.sum(cancellous) > 0

    def test_distance_method_thickness_consistency_across_spacings(self):
        """
        요구사항 검증:
        distance 방식에서 복셀 해상도(spacing)를 (0.5, 0.5, 0.5)와 (0.25, 0.25, 0.25)로 바꿔도
        피질골의 실제 물리적 두께/부피(mm^3)가 유사하게 산출되는지 검증.
        """
        cortical_thickness = 1.5

        # Case 1: Spacing 0.5mm, 40x40x40 볼륨 (물리 크기 20x20x20mm)
        spc1 = (0.5, 0.5, 0.5)
        hu1 = create_synthetic_bone_ellipsoid(
            shape=(40, 40, 40),
            spacing=spc1,
            radii_mm=(8.0, 8.0, 8.0),
            center_mm=(10.0, 10.0, 10.0),
            cortical_thickness_mm=cortical_thickness,
        )
        bone1 = extract_bone(hu1, spc1, bone_threshold=200.0)
        cort1, _ = split_cortical_cancellous(bone1, hu1, spc1, method="distance", cortical_thickness_mm=cortical_thickness)
        vol_voxel1 = np.prod(spc1)
        physical_volume1 = np.sum(cort1) * vol_voxel1  # mm^3

        # Case 2: Spacing 0.25mm, 80x80x80 볼륨 (동일 물리 크기 20x20x20mm)
        spc2 = (0.25, 0.25, 0.25)
        hu2 = create_synthetic_bone_ellipsoid(
            shape=(80, 80, 80),
            spacing=spc2,
            radii_mm=(8.0, 8.0, 8.0),
            center_mm=(10.0, 10.0, 10.0),
            cortical_thickness_mm=cortical_thickness,
        )
        bone2 = extract_bone(hu2, spc2, bone_threshold=200.0)
        cort2, _ = split_cortical_cancellous(bone2, hu2, spc2, method="distance", cortical_thickness_mm=cortical_thickness)
        vol_voxel2 = np.prod(spc2)
        physical_volume2 = np.sum(cort2) * vol_voxel2  # mm^3

        # 두 해상도 간 물리적 부피 오차가 10% 이내로 일관성 있게 유지되는지 검증
        relative_diff = abs(physical_volume1 - physical_volume2) / max(physical_volume1, physical_volume2)
        assert relative_diff < 0.10, (
            f"해상도 간 피질골 물리 부피 오차가 10% 이내여야 합니다: "
            f"Vol1={physical_volume1:.1f} mm^3, Vol2={physical_volume2:.1f} mm^3 (오차: {relative_diff*100:.1f}%)"
        )

    def test_run_segmentation_with_vtk_image(self):
        """vtkImageData 입력을 지원하는 run_segmentation 및 SegmentationResult 데이터클래스 검증"""
        hu = create_synthetic_bone_ellipsoid(shape=(30, 30, 30), spacing=(1.0, 1.0, 1.0))
        ref_vtk = mask_to_vtk_image(
            mask=(hu > 0),
            spacing=(1.0, 1.0, 1.0),
            origin=(10.0, 20.0, 30.0),
        )

        res = run_segmentation(
            volume_data=ref_vtk,
            params={
                "bone_threshold": 200.0,
                "method": "distance",
                "cortical_thickness_mm": 1.5,
            },
        )

        assert isinstance(res, SegmentationResult)
        assert res.bone_mask.shape == (30, 30, 30)
        assert res.cortical_mask.shape == (30, 30, 30)
        assert res.cancellous_mask.shape == (30, 30, 30)
        assert res.params["method"] == "distance"

        # 불변성 확인
        assert not np.any(res.cortical_mask & res.cancellous_mask)
        assert np.array_equal(res.cortical_mask | res.cancellous_mask, res.bone_mask)

    def test_run_segmentation_with_roi_ijk(self):
        """ROI Bounding Box 지정 시 해당 영역만 분할되고 외부가 보존되는지 검증"""
        hu = create_synthetic_bone_ellipsoid(shape=(40, 40, 40), spacing=(1.0, 1.0, 1.0), center_mm=(20, 20, 20))
        # ROI: x(15..25), y(15..25), z(15..25)
        roi = (15, 25, 15, 25, 15, 25)

        res = run_segmentation(hu, spacing=(1.0, 1.0, 1.0), roi_ijk=roi)

        # ROI 외부 영역은 마스크가 모두 False여야 함
        assert not np.any(res.bone_mask[0:15, :, :])
        assert not np.any(res.bone_mask[26:40, :, :])
        assert np.any(res.bone_mask[15:26, 15:26, 15:26])

    def test_mask_to_vtk_image_and_back(self):
        """마스크와 vtkImageData 간의 상호 변환 및 geometry 보존 검증"""
        mask_orig = np.zeros((20, 30, 40), dtype=bool)  # (nz, ny, nx)
        mask_orig[5:15, 10:20, 15:30] = True

        spacing = (0.8, 0.9, 1.2)
        origin = (10.0, -20.0, 50.0)

        # 1. mask -> vtkImageData
        vtk_img = mask_to_vtk_image(mask_orig, spacing=spacing, origin=origin)
        assert vtk_img.GetDimensions() == (40, 30, 20)  # (nx, ny, nz)
        assert np.allclose(vtk_img.GetSpacing(), spacing)
        assert np.allclose(vtk_img.GetOrigin(), origin)

        # 2. vtkImageData -> mask
        mask_restored = vtk_image_to_mask(vtk_img, threshold=0.5)
        assert mask_restored.shape == (20, 30, 40)
        assert np.array_equal(mask_orig, mask_restored)
