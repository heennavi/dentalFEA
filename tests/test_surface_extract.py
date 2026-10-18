"""
test_surface_extract.py
Step 5: 마스크를 표면 메쉬로 변환하는 surface_extract 모듈 단위 테스트 (pytest).
"""

import os
import math
import tempfile
import warnings
import numpy as np
import pytest
import vtk
import trimesh

from dental_fea_app.geometry.surface_extract import (
    mask_to_surface,
    mesh_report,
)


def create_synthetic_sphere_mask(
    radius_mm: float = 10.0,
    spacing: tuple = (0.5, 0.5, 0.5),
    margin_voxels: int = 5,
) -> tuple[np.ndarray, tuple, tuple, float]:
    """
    지정된 반지름의 합성 구(Sphere) 3D 이진 마스크와 이론적 부피를 생성합니다.
    """
    sx, sy, sz = spacing
    r_vox_x = int(math.ceil(radius_mm / sx))
    r_vox_y = int(math.ceil(radius_mm / sy))
    r_vox_z = int(math.ceil(radius_mm / sz))

    nx = 2 * (r_vox_x + margin_voxels) + 1
    ny = 2 * (r_vox_y + margin_voxels) + 1
    nz = 2 * (r_vox_z + margin_voxels) + 1

    cx = (nx - 1) / 2.0 * sx
    cy = (ny - 1) / 2.0 * sy
    cz = (nz - 1) / 2.0 * sz

    x = np.arange(nx) * sx - cx
    y = np.arange(ny) * sy - cy
    z = np.arange(nz) * sz - cz

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(X**2 + Y**2 + Z**2)
    mask = (dist <= radius_mm).astype(bool)

    theoretical_vol = (4.0 / 3.0) * math.pi * (radius_mm**3)
    origin = (0.0, 0.0, 0.0)

    return mask, spacing, origin, theoretical_vol


def create_synthetic_cylinder_mask(
    radius_mm: float = 8.0,
    height_mm: float = 20.0,
    spacing: tuple = (0.5, 0.5, 0.5),
    margin_voxels: int = 5,
) -> tuple[np.ndarray, tuple, tuple, float]:
    """
    지정된 반지름과 높이를 갖는 z축 방향 합성 원기둥 3D 이진 마스크를 생성합니다.
    """
    sx, sy, sz = spacing
    r_vox_x = int(math.ceil(radius_mm / sx))
    r_vox_y = int(math.ceil(radius_mm / sy))
    h_vox_z = int(math.ceil(height_mm / sz))

    nx = 2 * (r_vox_x + margin_voxels) + 1
    ny = 2 * (r_vox_y + margin_voxels) + 1
    nz = h_vox_z + 2 * margin_voxels + 1

    cx = (nx - 1) / 2.0 * sx
    cy = (ny - 1) / 2.0 * sy
    cz = (nz - 1) / 2.0 * sz

    x = np.arange(nx) * sx - cx
    y = np.arange(ny) * sy - cy
    z = np.arange(nz) * sz - cz

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    radial_dist = np.sqrt(X**2 + Y**2)
    mask = (radial_dist <= radius_mm) & (np.abs(Z) <= height_mm / 2.0)

    theoretical_vol = math.pi * (radius_mm**2) * height_mm
    origin = (0.0, 0.0, 0.0)

    return mask.astype(bool), spacing, origin, theoretical_vol


class TestSurfaceExtract:
    """마스크 -> 표면 메쉬 추출 파이프라인 검증 테스트 슈트"""

    def test_01_sphere_surface_and_volume_accuracy(self):
        """1. 합성 구 마스크: 표면 변환 후 부피가 해석해(4/3 * pi * r^3)와 5% 이내로 일치하는지 검증"""
        radius = 10.0
        mask, spc, org, v_analytic = create_synthetic_sphere_mask(radius_mm=radius, spacing=(0.5, 0.5, 0.5))

        mesh = mask_to_surface(
            mask,
            spacing=spc,
            origin=org,
            smooth_iterations=15,
            smooth_passband=0.1,
            keep_largest_component=True,
        )

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.is_watertight, "추출된 구 메쉬는 닫힌 매니폴드(watertight)여야 합니다."

        extracted_vol = float(abs(mesh.volume))
        vol_error_ratio = abs(extracted_vol - v_analytic) / v_analytic

        print(f"\n[구 부피 비교] 해석해: {v_analytic:.2f} mm^3, 추출값: {extracted_vol:.2f} mm^3, 오차율: {vol_error_ratio*100:.2f}%")
        assert vol_error_ratio < 0.05, f"부피 오차({vol_error_ratio*100:.2f}%)가 5% 기준을 초과했습니다."

    def test_02_cylinder_surface_and_volume_accuracy(self):
        """2. 합성 원기둥 마스크: 표면 변환 후 부피가 해석해(pi * r^2 * h)와 5% 이내로 일치하는지 검증"""
        radius = 8.0
        height = 20.0
        mask, spc, org, v_analytic = create_synthetic_cylinder_mask(
            radius_mm=radius, height_mm=height, spacing=(0.5, 0.5, 0.5)
        )

        mesh = mask_to_surface(
            mask,
            spacing=spc,
            origin=org,
            smooth_iterations=15,
            smooth_passband=0.1,
            keep_largest_component=True,
        )

        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert mesh.is_watertight, "추출된 원기둥 메쉬는 닫힌 매니폴드(watertight)여야 합니다."

        extracted_vol = float(abs(mesh.volume))
        vol_error_ratio = abs(extracted_vol - v_analytic) / v_analytic

        print(f"\n[원기둥 부피 비교] 해석해: {v_analytic:.2f} mm^3, 추출값: {extracted_vol:.2f} mm^3, 오차율: {vol_error_ratio*100:.2f}%")
        assert vol_error_ratio < 0.05, f"부피 오차({vol_error_ratio*100:.2f}%)가 5% 기준을 초과했습니다."

    def test_03_volume_preservation_and_smoothing_warning(self):
        """3. 스무딩 전/후 부피 변화율 검증 및 3% 초과 시 경고 동작 확인"""
        # 1) 일반 구 마스크: 기본 스무딩(iterations=15, passband=0.1)에서는 Sinc 필터 특성상 부피가 3% 이내로 잘 보존됨
        mask, spc, org, _ = create_synthetic_sphere_mask(radius_mm=10.0, spacing=(0.5, 0.5, 0.5))
        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always")
            mesh = mask_to_surface(
                mask,
                spacing=spc,
                origin=org,
                smooth_iterations=15,
                smooth_passband=0.1,
            )
            vol_warns = [w for w in warn_list if "스무딩 후 부피" in str(w.message)]
            assert len(vol_warns) == 0, "기본 스무딩 시 부피 변화율은 3% 이내여야 합니다."

        # 2) 작은 구(반지름 2mm)에서 극단적 스무딩 시 부피가 3% 초과하여 수축되므로 경고 발생 확인
        small_mask, s_spc, s_org, _ = create_synthetic_sphere_mask(radius_mm=2.0, spacing=(1.0, 1.0, 1.0))
        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always")
            mesh_shrunk = mask_to_surface(
                small_mask,
                spacing=s_spc,
                origin=s_org,
                smooth_iterations=50,
                smooth_passband=0.01,
            )
            vol_warns = [w for w in warn_list if "스무딩 후 부피" in str(w.message)]
            assert len(vol_warns) > 0, "부피 3% 초과 변화 시 경고가 정상적으로 발생해야 합니다."

    def test_04_target_edge_mm_decimation(self):
        """4. target_edge_mm 지정 시 메쉬 삼각형 크기 조절(Decimation) 동작 검증"""
        mask, spc, org, _ = create_synthetic_sphere_mask(radius_mm=12.0, spacing=(0.4, 0.4, 0.4))

        # 1) 원본 세밀 메쉬
        mesh_fine = mask_to_surface(mask, spacing=spc, origin=org, smooth_iterations=5)
        rep_fine = mesh_report(mesh_fine)

        # 2) target_edge_mm = 2.0 mm 로 감축 요청
        mesh_coarse = mask_to_surface(
            mask,
            spacing=spc,
            origin=org,
            smooth_iterations=5,
            target_edge_mm=2.0,
        )
        rep_coarse = mesh_report(mesh_coarse)

        print(f"\n[감축 전] 면 수: {rep_fine['n_faces']}, 평균 변 길이: {rep_fine['mean_edge_length']:.2f} mm")
        print(f"[감축 후] 면 수: {rep_coarse['n_faces']}, 평균 변 길이: {rep_coarse['mean_edge_length']:.2f} mm")

        assert rep_coarse["n_faces"] < rep_fine["n_faces"]
        assert rep_coarse["mean_edge_length"] > rep_fine["mean_edge_length"]
        assert rep_coarse["is_watertight"]

    def test_05_roi_ijk_subvolume_extraction(self):
        """5. ROI(roi_ijk)를 통한 임플란트 주변 Bounding Box 고속 변환 검증"""
        mask, spc, org, _ = create_synthetic_sphere_mask(radius_mm=10.0, spacing=(0.5, 0.5, 0.5))
        nz, ny, nx = mask.shape

        # 구의 중심부 주변 서브 영역만 ROI로 전달
        roi = (nx // 4, 3 * nx // 4, ny // 4, 3 * ny // 4, nz // 4, 3 * nz // 4)

        mesh_roi = mask_to_surface(
            mask,
            spacing=spc,
            origin=org,
            roi_ijk=roi,
            keep_largest_component=False,
        )

        assert isinstance(mesh_roi, trimesh.Trimesh)
        assert len(mesh_roi.vertices) > 0

    def test_06_mesh_report_dictionary(self):
        """6. mesh_report 딕셔너리 필수 키 및 물리 속성 검증"""
        mask, spc, org, _ = create_synthetic_sphere_mask(radius_mm=8.0, spacing=(0.5, 0.5, 0.5))
        mesh = mask_to_surface(mask, spacing=spc, origin=org)

        rep = mesh_report(mesh)

        required_keys = [
            "n_vertices", "n_faces", "volume", "surface_area",
            "is_watertight", "min_edge_length", "mean_edge_length",
            "max_edge_length", "is_self_intersecting"
        ]
        for k in required_keys:
            assert k in rep, f"보고서에 '{k}' 키가 누락되었습니다."

        assert rep["n_vertices"] > 0
        assert rep["n_faces"] > 0
        assert rep["volume"] > 0
        assert rep["surface_area"] > 0
        assert rep["is_watertight"] is True
        assert rep["min_edge_length"] <= rep["mean_edge_length"] <= rep["max_edge_length"]

    def test_07_stl_export_and_reimport(self):
        """7. STL 파일 내보내기 및 외부 뷰어 호환성(재로드) 검증"""
        mask, spc, org, _ = create_synthetic_sphere_mask(radius_mm=7.0, spacing=(0.5, 0.5, 0.5))
        mesh = mask_to_surface(mask, spacing=spc, origin=org)

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # STL 파일로 내보내기
            mesh.export(tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0

            # trimesh로 다시 로드하여 형상 무결성 검증
            reloaded_mesh = trimesh.load_mesh(tmp_path)
            assert len(reloaded_mesh.vertices) == len(mesh.vertices)
            assert len(reloaded_mesh.faces) == len(mesh.faces)
            assert reloaded_mesh.is_watertight
            assert abs(reloaded_mesh.volume - mesh.volume) / mesh.volume < 1e-3
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_08_segmentation_integration_and_stl_export(self):
        """8. 실제 뼈/피질골/수질골 분할 마스크 연동 및 STL 저장 무결성 검증"""
        from dental_fea_app.segmentation.cortical_cancellous import (
            extract_bone,
            split_cortical_cancellous,
        )

        # 합성 타원체 뼈 데이터 생성
        nz, ny, nx = 40, 40, 40
        spc = (0.5, 0.5, 0.5)
        Z, Y, X = np.ogrid[:nz, :ny, :nx]
        cz, cy, cx = nz / 2.0, ny / 2.0, nx / 2.0
        dist = np.sqrt(((X - cx) * spc[0])**2 + ((Y - cy) * spc[1])**2 + ((Z - cz) * spc[2])**2)
        
        hu = np.full((nz, ny, nx), -1000.0)
        hu[dist <= 8.0] = 400.0  # 수질골
        hu[(dist <= 8.0) & (dist >= 6.5)] = 1200.0  # 피질골 외피

        # 분할 파이프라인 수행
        bone_mask = extract_bone(hu, spacing=spc, bone_threshold=200.0)
        cortical_mask, cancellous_mask = split_cortical_cancellous(
            bone_mask, hu, spacing=spc, method="distance", cortical_thickness_mm=1.5
        )

        assert np.any(cortical_mask)
        assert np.any(cancellous_mask)

        # 피질골 표면 추출
        cortical_mesh = mask_to_surface(cortical_mask, spacing=spc, target_edge_mm=1.0)
        assert len(cortical_mesh.faces) > 0
        rep_cort = mesh_report(cortical_mesh)
        assert rep_cort["volume"] > 0

        # STL 저장 검증
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            stl_path = tmp.name

        try:
            cortical_mesh.export(stl_path)
            assert os.path.getsize(stl_path) > 0
            # 재로드
            loaded = trimesh.load_mesh(stl_path)
            assert len(loaded.faces) == len(cortical_mesh.faces)
        finally:
            if os.path.exists(stl_path):
                os.remove(stl_path)
