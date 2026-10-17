"""
create_dummy_data.py
치과 하악골(피질골, 해면골, 치아 에나멜/상아질) 구조를 모사한 합성 3D CT 볼륨 생성기.
"""

import os
import numpy as np
import SimpleITK as sitk


def generate_synthetic_dental_ct(output_path: str, size: tuple = (120, 120, 80), spacing: tuple = (0.5, 0.5, 0.5)):
    """
    치과 하악골 및 치아를 모사한 합성 CT 볼륨(HU 스케일)을 생성하여 파일로 저장합니다.
    """
    nx, ny, nz = size
    # 초기 공기 (-1000 HU)
    vol = np.full((nz, ny, nx), -1000.0, dtype=np.float32)

    # 중심 좌표
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0

    z_grid, y_grid, x_grid = np.ogrid[:nz, :ny, :nx]

    # 1. 연조직 원기둥 (Soft Tissue: ~40 HU) - 볼륨의 약 80% 반경
    soft_radius = min(cx, cy) * 0.8
    dist_xy_sq = (x_grid - cx)**2 + (y_grid - cy)**2
    soft_mask = (dist_xy_sq <= soft_radius**2) & (z_grid >= 0)
    vol[soft_mask] = 40.0

    # 2. 하악골 외곽 (Cortical Bone: ~1200 HU) - 볼륨의 약 55% 반경
    mandible_outer_r = min(cx, cy) * 0.55
    bone_mask = (dist_xy_sq <= mandible_outer_r**2) & (z_grid <= (cz * 1.4))
    vol[bone_mask] = 1200.0

    # 3. 하악골 내부 해면골 (Cancellous / Trabecular Bone: ~350 HU) - 약 40% 반경
    mandible_inner_r = min(cx, cy) * 0.40
    cancellous_mask = (dist_xy_sq <= mandible_inner_r**2) & (z_grid <= (cz * 1.3)) & (z_grid >= 5)
    vol[cancellous_mask] = 350.0

    # 4. 상단 치아 구조 (상아질 Dentin ~1000 HU, 법랑질 Enamel ~2200 HU)
    tooth_cx, tooth_cy = cx, cy
    tooth_dist_xy_sq = (x_grid - tooth_cx)**2 + (y_grid - tooth_cy)**2
    tooth_z_start = int(cz * 1.0)
    tooth_z_end = int(nz - 4)

    # 치아 상아질 (Dentin)
    dentin_r = min(cx, cy) * 0.20
    dentin_mask = (tooth_dist_xy_sq <= dentin_r**2) & (z_grid >= tooth_z_start) & (z_grid <= tooth_z_end)
    vol[dentin_mask] = 1000.0

    # 치아 외곽 에나멜 (Enamel)
    enamel_r = min(cx, cy) * 0.25
    enamel_mask = (tooth_dist_xy_sq <= enamel_r**2) & (z_grid >= (tooth_z_start + int(nz * 0.1))) & (z_grid <= tooth_z_end)
    vol[enamel_mask & ~dentin_mask] = 2200.0

    # 노이즈 추가 (가우시안 노이즈)
    noise = np.random.normal(0, 15.0, vol.shape).astype(np.float32)
    vol += noise

    # int16 변환 (CT 표준 16비트 정수)
    vol_int16 = np.clip(vol, -1024, 3071).astype(np.int16)

    # SimpleITK 이미지 생성
    itk_img = sitk.GetImageFromArray(vol_int16)
    itk_img.SetSpacing(spacing)
    itk_img.SetOrigin((-(nx * spacing[0]) / 2.0, -(ny * spacing[1]) / 2.0, 0.0))
    itk_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sitk.WriteImage(itk_img, output_path)
    print(f"[create_dummy_data] 합성 치과 CT 생성 완료: {output_path} ({size}, Spacing: {spacing})")
    return output_path


if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    sample_file = os.path.join(sample_dir, "synthetic_mandible_ct.mha")
    generate_synthetic_dental_ct(sample_file)
