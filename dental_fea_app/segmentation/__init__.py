"""
dental_fea_app.segmentation

치과 CT/CBCT 영상의 뼈(하악골/상악골), 피질골(Cortical bone),
수질골(Cancellous/Trabecular bone) 분할(Segmentation) 모듈.
"""

from dental_fea_app.segmentation.cortical_cancellous import (
    SegmentationResult,
    extract_bone,
    split_cortical_cancellous,
    run_segmentation,
    mask_to_vtk_image,
    vtk_image_to_mask,
)

__all__ = [
    "SegmentationResult",
    "extract_bone",
    "split_cortical_cancellous",
    "run_segmentation",
    "mask_to_vtk_image",
    "vtk_image_to_mask",
]
