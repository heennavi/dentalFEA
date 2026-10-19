"""
workers.py
Background QThread workers for long-running FEA operations (meshing, preprocessing).
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
import trimesh
from PySide6.QtCore import QThread, Signal

from fea.models import MeshPart, MeshModel, TieDef, AnalysisConfig
from meshing.gmsh_mesher import build_part_mesh
from meshing.quality_checker import quality_report_text
from meshing.surface_extractor import contact_pair_from_parts

logger = logging.getLogger(__name__)


class MeshWorker(QThread):
    """
    Background worker thread for generating 3D tetrahedral meshes and contact ties without blocking the UI.

    Signals:
        progress(int, str): Emitted with progress percentage (0-100) and step description.
        finished(object, object, str): Emitted on completion with (MeshModel, List[TieDef], quality_report_str).
        error(str): Emitted when meshing or contact pair generation encounters an exception.
    """

    progress = Signal(int, str)
    finished = Signal(object, object, str)
    error = Signal(str)

    def __init__(
        self,
        fea_geometry: Any,
        config: AnalysisConfig,
        contact_tolerance_mm: float = 0.3,
        parent: Optional[QThread] = None,
    ):
        super().__init__(parent)
        self.fea_geometry = fea_geometry
        self.config = config
        self.contact_tolerance_mm = float(contact_tolerance_mm)
        self._is_cancelled = False

    def cancel(self):
        """Request worker thread cancellation."""
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit(5, "표면 지오메트리 정돈 및 준비 중...")
            if self._is_cancelled:
                return

            # 1. 기하 객체 추출 및 분리
            bone_mesh = None
            implant_mesh = None

            if hasattr(self.fea_geometry, "cortical") and hasattr(self.fea_geometry, "cancellous"):
                # FeaGeometry 인스턴스
                c_mesh = self.fea_geometry.cortical
                cn_mesh = self.fea_geometry.cancellous
                i_mesh = getattr(self.fea_geometry, "implant", None)

                # 골 조직: 피질골과 수질골을 결합한 단일 솔리드 구성
                if c_mesh is not None and cn_mesh is not None:
                    try:
                        bone_mesh = trimesh.boolean.union([c_mesh, cn_mesh], engine="manifold")
                    except Exception:
                        bone_mesh = trimesh.util.concatenate([c_mesh, cn_mesh])
                elif c_mesh is not None:
                    bone_mesh = c_mesh
                elif cn_mesh is not None:
                    bone_mesh = cn_mesh

                implant_mesh = i_mesh
            elif isinstance(self.fea_geometry, dict):
                bone_mesh = self.fea_geometry.get("bone")
                implant_mesh = self.fea_geometry.get("implant")
            else:
                raise ValueError("유효하지 않은 기하 데이터 형식입니다. (FeaGeometry 또는 dict 필요)")

            if bone_mesh is None:
                raise ValueError("골 조직(Bone) 표면 메쉬를 찾을 수 없습니다.")

            # 메쉬 유효성/watertight 보정
            from dental_fea_app.geometry.boolean_ops import ensure_watertight
            bone_mesh = ensure_watertight(bone_mesh, name="bone")
            if implant_mesh is not None:
                implant_mesh = ensure_watertight(implant_mesh, name="implant")

            order = 2 if self.config.element_type.upper() == "C3D10" else 1

            # 2. 임플란트 체적 메쉬 생성
            parts: List[MeshPart] = []
            implant_part: Optional[MeshPart] = None

            if implant_mesh is not None and len(implant_mesh.faces) > 0:
                self.progress.emit(15, "임플란트 3D 체적 메쉬 생성 중 (Gmsh)...")
                if self._is_cancelled:
                    return

                implant_part = build_part_mesh(
                    name="implant",
                    surface=implant_mesh,
                    size_far=self.config.mesh_size_near_mm,
                    size_near=self.config.mesh_size_near_mm,
                    order=order,
                )
                parts.append(implant_part)

            # 3. 골 조직 체적 메쉬 생성 (임플란트 주변 크기장 정밀화 반영)
            self.progress.emit(45, "골 조직 3D 체적 메쉬 생성 중 (Gmsh)...")
            if self._is_cancelled:
                return

            bone_part = build_part_mesh(
                name="bone",
                surface=bone_mesh,
                size_far=self.config.mesh_size_far_mm,
                size_near=self.config.mesh_size_near_mm,
                refine_surface=implant_mesh,
                order=order,
            )
            parts.insert(0, bone_part)

            if self._is_cancelled:
                return

            # 4. 접촉면(TIE) 추출 (골-임플란트 경계)
            ties: List[TieDef] = []
            if implant_part is not None:
                self.progress.emit(75, "골-임플란트 결합면(*TIE) 추출 중...")
                tie = contact_pair_from_parts(
                    part_a=implant_part,
                    part_b=bone_part,
                    tol_mm=self.contact_tolerance_mm,
                    master="a",
                    name="TIE_BONE_IMPLANT",
                )
                ties.append(tie)

            # 5. 메쉬 모델 조립 및 품질 보고서 생성
            self.progress.emit(90, "메쉬 무결성 및 품질 검사 중...")
            mesh_model = MeshModel(parts=parts)
            q_report = quality_report_text(mesh_model)

            self.progress.emit(100, "체적 메쉬 생성 완료")
            self.finished.emit(mesh_model, ties, q_report)

        except Exception as e:
            logger.exception("체적 메쉬 생성 중 오류 발생")
            self.error.emit(f"메쉬 생성 실패: {str(e)}")
