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

            if hasattr(self.fea_geometry, "bone") and self.fea_geometry.bone is not None and len(getattr(self.fea_geometry.bone, "faces", [])) > 0:
                bone_mesh = self.fea_geometry.bone
                implant_mesh = getattr(self.fea_geometry, "implant", None)
            elif hasattr(self.fea_geometry, "cortical") and hasattr(self.fea_geometry, "cancellous"):
                # FeaGeometry 인스턴스 (하위 호환)
                c_mesh = self.fea_geometry.cortical
                cn_mesh = self.fea_geometry.cancellous
                i_mesh = getattr(self.fea_geometry, "implant", None)

                # 골 조직: 피질골과 수질골을 결합한 단일 솔리드 구성
                has_cn = cn_mesh is not None and len(getattr(cn_mesh, "faces", [])) > 0
                if c_mesh is not None and has_cn:
                    try:
                        import manifold3d as m3d
                        m1 = m3d.Manifold(c_mesh.vertices, c_mesh.faces)
                        m2 = m3d.Manifold(cn_mesh.vertices, cn_mesh.faces)
                        m_u = m1 + m2
                        mesh_raw = m_u.to_mesh()
                        bone_mesh = trimesh.Trimesh(mesh_raw.vert_properties[:, :3], mesh_raw.tri_verts, process=True)
                    except Exception:
                        try:
                            bone_mesh = trimesh.boolean.union([c_mesh, cn_mesh], engine="manifold")
                        except Exception:
                            # 내부 경계면이 남는 concatenate 대신 외곽 피질골을 기본 뼈 체적으로 사용
                            bone_mesh = c_mesh.copy()
                elif c_mesh is not None:
                    bone_mesh = c_mesh
                elif has_cn:
                    bone_mesh = cn_mesh

                implant_mesh = i_mesh
            elif isinstance(self.fea_geometry, dict):
                bone_mesh = self.fea_geometry.get("bone")
                implant_mesh = self.fea_geometry.get("implant")
            else:
                raise ValueError("유효하지 않은 기하 데이터 형식입니다. (FeaGeometry 또는 dict 필요)")

            if bone_mesh is None:
                raise ValueError("골 조직(Bone) 표면 메쉬를 찾을 수 없습니다.")

            abutment_mesh = getattr(self.fea_geometry, "abutment", None) if not isinstance(self.fea_geometry, dict) else self.fea_geometry.get("abutment")
            crown_mesh = getattr(self.fea_geometry, "crown", None) if not isinstance(self.fea_geometry, dict) else self.fea_geometry.get("crown")
            screw_mesh = getattr(self.fea_geometry, "screw", None) if not isinstance(self.fea_geometry, dict) else self.fea_geometry.get("screw")

            # 메쉬 유효성/watertight 보정
            from dental_fea_app.geometry.boolean_ops import ensure_watertight
            bone_mesh = ensure_watertight(bone_mesh, name="bone")
            if implant_mesh is not None and len(getattr(implant_mesh, "faces", [])) > 0:
                implant_mesh = ensure_watertight(implant_mesh, name="implant")
            else:
                implant_mesh = None

            if abutment_mesh is not None and len(getattr(abutment_mesh, "faces", [])) > 0:
                abutment_mesh = ensure_watertight(abutment_mesh, name="abutment")
            else:
                abutment_mesh = None

            if screw_mesh is not None and len(getattr(screw_mesh, "faces", [])) > 0:
                screw_mesh = ensure_watertight(screw_mesh, name="screw")
            else:
                screw_mesh = None

            if crown_mesh is not None and len(getattr(crown_mesh, "faces", [])) > 0:
                crown_mesh = ensure_watertight(crown_mesh, name="crown")
            else:
                crown_mesh = None

            order = 2 if self.config.element_type.upper() == "C3D10" else 1
            algo_3d = getattr(self.config, "mesh_algorithm_3d", "auto")
            opt_netgen = getattr(self.config, "optimize_netgen", True)

            # 2. 임플란트 체적 메쉬 생성
            parts: List[MeshPart] = []
            implant_part: Optional[MeshPart] = None
            abutment_part: Optional[MeshPart] = None
            screw_part: Optional[MeshPart] = None
            crown_part: Optional[MeshPart] = None

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
                    mesh_algorithm_3d=algo_3d,
                    optimize_netgen=opt_netgen,
                )

            # 2-1. 지대주(Abutment) 체적 메쉬 생성
            if abutment_mesh is not None and len(abutment_mesh.faces) > 0:
                self.progress.emit(25, "지대주(Abutment) 3D 체적 메쉬 생성 중 (Gmsh)...")
                if self._is_cancelled:
                    return

                size_abut = getattr(self.config, "mesh_size_abutment_mm", None) or self.config.mesh_size_near_mm
                abutment_part = build_part_mesh(
                    name="abutment",
                    surface=abutment_mesh,
                    size_far=size_abut,
                    size_near=size_abut,
                    order=order,
                    mesh_algorithm_3d=algo_3d,
                    optimize_netgen=opt_netgen,
                )

            # 2-2. 스크류(Screw) 체적 메쉬 생성
            if screw_mesh is not None and len(screw_mesh.faces) > 0:
                self.progress.emit(30, "어버트먼트 스크류(Screw) 3D 체적 메쉬 생성 중 (Gmsh)...")
                if self._is_cancelled:
                    return

                size_sc = getattr(self.config, "mesh_size_screw_mm", None) or min(self.config.mesh_size_near_mm, 0.35)
                screw_part = build_part_mesh(
                    name="screw",
                    surface=screw_mesh,
                    size_far=size_sc,
                    size_near=size_sc,
                    order=order,
                    mesh_algorithm_3d=algo_3d,
                    optimize_netgen=opt_netgen,
                )

            # 2-3. 보철물(Crown) 체적 메쉬 생성
            if crown_mesh is not None and len(crown_mesh.faces) > 0:
                self.progress.emit(38, "보철물(Crown) 3D 체적 메쉬 생성 중 (Gmsh)...")
                if self._is_cancelled:
                    return

                size_cr = getattr(self.config, "mesh_size_crown_mm", None) or self.config.mesh_size_near_mm
                crown_part = build_part_mesh(
                    name="crown",
                    surface=crown_mesh,
                    size_far=size_cr,
                    size_near=size_cr,
                    order=order,
                    mesh_algorithm_3d=algo_3d,
                    optimize_netgen=opt_netgen,
                )

            # 3. 골 조직 체적 메쉬 생성 (임플란트 주변 크기장 정밀화 반영)
            self.progress.emit(50, "골 조직 3D 체적 메쉬 생성 중 (Gmsh)...")
            if self._is_cancelled:
                return

            bone_part = build_part_mesh(
                name="bone",
                surface=bone_mesh,
                size_far=self.config.mesh_size_far_mm,
                size_near=self.config.mesh_size_near_mm,
                refine_surface=implant_mesh,
                order=order,
                mesh_algorithm_3d=algo_3d,
                optimize_netgen=opt_netgen,
            )

            parts.append(bone_part)
            if implant_part is not None:
                parts.append(implant_part)
            if abutment_part is not None:
                parts.append(abutment_part)
            if screw_part is not None:
                parts.append(screw_part)
            if crown_part is not None:
                parts.append(crown_part)

            if self._is_cancelled:
                return

            # 4. 결합면(*TIE) 추출 (골-임플란트, 임플란트-지대주, 지대주-크라운, 스크류-지대주, 임플란트-스크류)
            ties: List[TieDef] = []
            if implant_part is not None:
                self.progress.emit(65, "골-임플란트 결합면(*TIE) 추출 중...")
                tie_bone = contact_pair_from_parts(
                    part_a=implant_part,
                    part_b=bone_part,
                    tol_mm=self.contact_tolerance_mm,
                    master="a",
                    name="TIE_BONE_IMPLANT",
                )
                ties.append(tie_bone)

            if implant_part is not None and abutment_part is not None:
                self.progress.emit(70, "임플란트-지대주 결합면(*TIE) 추출 중...")
                tie_imp_abut = contact_pair_from_parts(
                    part_a=implant_part,
                    part_b=abutment_part,
                    tol_mm=self.contact_tolerance_mm,
                    master="auto",
                    name="TIE_IMPLANT_ABUTMENT",
                )
                ties.append(tie_imp_abut)

            if abutment_part is not None and screw_part is not None:
                self.progress.emit(75, "스크류-지대주 안착턱 결합면(*TIE) 추출 중...")
                tie_sc_abut = contact_pair_from_parts(
                    part_a=abutment_part,
                    part_b=screw_part,
                    tol_mm=self.contact_tolerance_mm,
                    master="auto",
                    name="TIE_SCREW_ABUTMENT",
                )
                ties.append(tie_sc_abut)

            if implant_part is not None and screw_part is not None:
                self.progress.emit(80, "임플란트-스크류 나사산 결합면(*TIE, 일반 공차) 추출 중...")
                # 나사산 플랭크 간 일반 공차 적용 (기본 0.20 mm)
                thread_tol = float(getattr(self.config, "screw_contact_tol_mm", 0.20))
                tie_imp_sc = contact_pair_from_parts(
                    part_a=implant_part,
                    part_b=screw_part,
                    tol_mm=thread_tol,
                    master="a",
                    name="TIE_IMPLANT_SCREW",
                )
                ties.append(tie_imp_sc)

            if abutment_part is not None and crown_part is not None:
                self.progress.emit(85, "지대주-보철물(Crown) 결합면(*TIE) 추출 중...")
                tie_abut_crown = contact_pair_from_parts(
                    part_a=abutment_part,
                    part_b=crown_part,
                    tol_mm=self.contact_tolerance_mm,
                    master="auto",
                    name="TIE_ABUTMENT_CROWN",
                )
                ties.append(tie_abut_crown)

            # 5. 메쉬 모델 조립 및 품질 보고서 생성
            self.progress.emit(90, "메쉬 무결성 및 품질 검사 중...")
            from meshing.quality_checker import clean_mesh_model
            raw_model = MeshModel(parts=parts)
            mesh_model = clean_mesh_model(raw_model)
            q_report = quality_report_text(mesh_model)

            self.progress.emit(100, "체적 메쉬 생성 완료")
            self.finished.emit(mesh_model, ties, q_report)

        except Exception as e:
            logger.exception("체적 메쉬 생성 중 오류 발생")
            self.error.emit(f"메쉬 생성 실패: {str(e)}")
