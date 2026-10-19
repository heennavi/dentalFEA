"""
calculix_writer.py
CalculiX input file (.inp) generator for dental FEA simulations.

[CalculiX .inp Standards & Conventions]
- Units: Length (mm), Force (N), Stress/Modulus (MPa), Density (tonne/mm^3)
- Coordinate System: LPS World coordinates (mm)
- Element Types: C3D10 (quadratic tetrahedron) and C3D4 (linear tetrahedron)
- Node/Element Numbering: 1-based unique global identifiers
- Keywords: Uppercase with comma delimiters
- File Encoding: ASCII with '\\n' line terminators
"""

import os
import io
import logging
from typing import List, Dict, Any, Sequence, Optional
import numpy as np

from fea.models import MeshModel, MeshPart, MaterialAssignment, MaterialDef, BoundaryConditionSet, TieDef
from meshing.gmsh_mesher import gmsh_tet10_to_ccx

logger = logging.getLogger(__name__)


class GlobalIndexMapper:
    """
    Manages unified mapping from part-local 0-based indices to 1-based global IDs.

    Guarantees non-overlapping, contiguous 1-based node and element IDs across all parts.
    """

    def __init__(self, model: MeshModel) -> None:
        """
        Initialize global offsets for nodes and elements from MeshModel.

        Args:
            model: MeshModel containing all solid parts.
        """
        self._node_offsets: Dict[str, int] = {}
        self._elem_offsets: Dict[str, int] = {}

        curr_node = 0
        curr_elem = 0
        for part in model.parts:
            self._node_offsets[part.name] = curr_node
            self._elem_offsets[part.name] = curr_elem
            curr_node += len(part.nodes)
            curr_elem += len(part.elements)

        self._total_nodes = curr_node
        self._total_elements = curr_elem

    @property
    def total_nodes(self) -> int:
        """Return total number of nodes in assembly."""
        return self._total_nodes

    @property
    def total_elements(self) -> int:
        """Return total number of elements in assembly."""
        return self._total_elements

    def node_id(self, part_name: str, local_idx: int) -> int:
        """
        Convert local 0-based node index to 1-based global node ID.

        Args:
            part_name: Part identifier name.
            local_idx: Local 0-based node index.

        Returns:
            int: Global 1-based node ID.
        """
        if part_name not in self._node_offsets:
            raise KeyError(f"Part '{part_name}' not found in GlobalIndexMapper.")
        return self._node_offsets[part_name] + int(local_idx) + 1

    def node_ids(self, part_name: str, local_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """
        Convert array of local 0-based node indices to 1-based global node IDs.

        Args:
            part_name: Part identifier name.
            local_indices: Array or sequence of local 0-based node indices.

        Returns:
            np.ndarray: Array of global 1-based node IDs (int64).
        """
        if part_name not in self._node_offsets:
            raise KeyError(f"Part '{part_name}' not found in GlobalIndexMapper.")
        arr = np.asarray(local_indices, dtype=np.int64)
        return self._node_offsets[part_name] + arr + 1

    def elem_id(self, part_name: str, local_idx: int) -> int:
        """
        Convert local 0-based element index to 1-based global element ID.

        Args:
            part_name: Part identifier name.
            local_idx: Local 0-based element index.

        Returns:
            int: Global 1-based element ID.
        """
        if part_name not in self._elem_offsets:
            raise KeyError(f"Part '{part_name}' not found in GlobalIndexMapper.")
        return self._elem_offsets[part_name] + int(local_idx) + 1

    def elem_ids(self, part_name: str, local_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """
        Convert array of local 0-based element indices to 1-based global element IDs.

        Args:
            part_name: Part identifier name.
            local_indices: Array or sequence of local 0-based element indices.

        Returns:
            np.ndarray: Array of global 1-based element IDs (int64).
        """
        if part_name not in self._elem_offsets:
            raise KeyError(f"Part '{part_name}' not found in GlobalIndexMapper.")
        arr = np.asarray(local_indices, dtype=np.int64)
        return self._elem_offsets[part_name] + arr + 1


def format_card_items(items: Sequence[Any], max_per_line: int = 16) -> List[str]:
    """
    Format sequence of data items into comma-separated lines with at most max_per_line items.

    Args:
        items: List or array of items (e.g. element or node IDs).
        max_per_line: Maximum entries per line (default: 16).

    Returns:
        List[str]: Formatted string lines without trailing comma.
    """
    lines: List[str] = []
    n = len(items)
    for i in range(0, n, max_per_line):
        chunk = items[i:i + max_per_line]
        lines.append(", ".join(str(x) for x in chunk))
    return lines


def write_inp(
    path: str,
    model: MeshModel,
    materials: Dict[str, MaterialAssignment],
    bc: BoundaryConditionSet,
    ties: List[TieDef],
    title: str = "dental_fea",
    apply_ccx_ordering: bool = True
) -> None:
    """
    Write a complete, valid CalculiX .inp deck for linear elastic static FEA.

    The deck is generated according to standard CalculiX/Abaqus conventions:
    1) *HEADING: Model title and simulation notes.
    2) *NODE: 1-based global coordinates (id, x, y, z) with 9 decimal places.
    3) *ELEMENT: Global element connectivity with C3D10 continuation line rule.
    4) Material definitions (*ELSET, *MATERIAL, *ELASTIC, *DENSITY, *SOLID SECTION).
    5) Node sets (*NSET for boundary constraints and applied loads).
    6) Contact constraints (*SURFACE and *TIE definitions).
    7) Analysis step (*STEP, *STATIC, *BOUNDARY, *CLOAD, *NODE FILE, *EL FILE, *END STEP).

    [C3D10 Continuation Line Rule]
    CalculiX and Abaqus require lines not to exceed 16 items. For C3D10 quadratic tetrahedra,
    each element card consists of 11 values (element_id and 10 corner/midside nodes).
    We follow the standard CalculiX continuation convention:
      - Line 1: elem_id, n1, n2, n3, n4, n5, n6, n7, n8, (terminated by a trailing comma ',')
      - Line 2: n9, n10
    This ensures strict compatibility with the CCX input parser across all platforms.

    Args:
        path: Target file path to write (.inp).
        model: MeshModel containing all meshed solid parts.
        materials: Dictionary mapping part name to MaterialAssignment.
        bc: BoundaryConditionSet containing fixed constraints and applied loads.
        ties: List of TieDef contact tie constraints.
        title: Simulation title text.
        apply_ccx_ordering: If True, applies gmsh_tet10_to_ccx to 10-node tetrahedra
                            to ensure correct CalculiX Jacobian orientation.
    """
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    mapper = GlobalIndexMapper(model)

    # Use streaming write with buffered lines
    buffer: List[str] = []
    chunk_size = 5000

    def flush_buffer(out_file: io.TextIOBase) -> None:
        if buffer:
            out_file.write("\n".join(buffer) + "\n")
            buffer.clear()

    with open(path, "w", encoding="ascii", newline="\n") as f:
        # 1) *HEADING
        buffer.append("*HEADING")
        clean_title = title.replace("\n", " ").strip() or "dental_fea"
        buffer.append(f"{clean_title}")

        # 2) *NODE
        buffer.append("*NODE")
        for part in model.parts:
            n_nodes = len(part.nodes)
            for local_nid in range(n_nodes):
                gid = mapper.node_id(part.name, local_nid)
                x, y, z = part.nodes[local_nid]
                buffer.append(f"{gid}, {x:.9f}, {y:.9f}, {z:.9f}")
                if len(buffer) >= chunk_size:
                    flush_buffer(f)

        # 3) *ELEMENT (Part-wise with global offsets)
        for part in model.parts:
            elset_name = f"EALL_{part.name.upper()}"
            elem_type = part.element_type.upper()
            buffer.append(f"*ELEMENT, TYPE={elem_type}, ELSET={elset_name}")

            elems = part.elements
            if elem_type == "C3D10" and apply_ccx_ordering:
                # Apply gmsh_tet10_to_ccx ordering (exchanging last two nodes)
                elems = gmsh_tet10_to_ccx(elems)

            n_elems = len(elems)
            for local_eid in range(n_elems):
                gid = mapper.elem_id(part.name, local_eid)
                conn = elems[local_eid]

                # Convert local node indices to global 1-based node IDs
                gnodes = [mapper.node_id(part.name, n) for n in conn]

                if elem_type == "C3D10":
                    # CalculiX C3D10 요소(11개 값) 연속 줄 규칙:
                    # 첫 줄에 요소 ID와 1~8번 절점을 출력하고 줄 끝 쉼표(,)를 붙여 다음 줄로 연결한 뒤,
                    # 두 번째 줄에 9~10번 절점을 출력한다.
                    line1 = f"{gid}, {gnodes[0]}, {gnodes[1]}, {gnodes[2]}, {gnodes[3]}, {gnodes[4]}, {gnodes[5]}, {gnodes[6]}, {gnodes[7]},"
                    line2 = f"{gnodes[8]}, {gnodes[9]}"
                    buffer.append(line1)
                    buffer.append(line2)
                else:
                    # C3D4 (5개 값: id, n1, n2, n3, n4)
                    line = f"{gid}, {gnodes[0]}, {gnodes[1]}, {gnodes[2]}, {gnodes[3]}"
                    buffer.append(line)

                if len(buffer) >= chunk_size:
                    flush_buffer(f)

        # 4) 재료별 요소 세트 (*ELSET), *MATERIAL, *SOLID SECTION
        # Collect global elements grouped by material name across parts
        mat_elem_map: Dict[str, List[int]] = {}
        mat_defs: Dict[str, MaterialDef] = {}

        for part_name, mat_assign in materials.items():
            part = model.get_part(part_name)
            if part is None:
                continue
            elem_mat_indices = mat_assign.element_material.get(part_name)
            if elem_mat_indices is None:
                continue

            for local_eid, m_idx in enumerate(elem_mat_indices):
                if m_idx < 0 or m_idx >= len(mat_assign.materials):
                    continue
                m_def = mat_assign.materials[m_idx]
                if m_def.name not in mat_defs:
                    mat_defs[m_def.name] = m_def
                if m_def.name not in mat_elem_map:
                    mat_elem_map[m_def.name] = []

                global_eid = mapper.elem_id(part_name, local_eid)
                mat_elem_map[m_def.name].append(global_eid)

        for mat_name, elem_list in mat_elem_map.items():
            clean_mat_name = mat_name.upper().replace(" ", "_")
            elset_mat_name = f"ELSET_{clean_mat_name}"
            m_def = mat_defs[mat_name]

            # *ELSET
            buffer.append(f"*ELSET, ELSET={elset_mat_name}")
            for line in format_card_items(elem_list, max_per_line=16):
                buffer.append(line)

            # *MATERIAL
            buffer.append(f"*MATERIAL, NAME={clean_mat_name}")
            buffer.append("*ELASTIC")
            buffer.append(f"{float(m_def.E):.6f}, {float(m_def.nu):.6f}")
            if m_def.density is not None and m_def.density > 0.0:
                buffer.append("*DENSITY")
                buffer.append(f"{float(m_def.density):.9e}")

            # *SOLID SECTION
            buffer.append(f"*SOLID SECTION, ELSET={elset_mat_name}, MATERIAL={clean_mat_name}")

            if len(buffer) >= chunk_size:
                flush_buffer(f)

        # 5) 절점 세트 (*NSET: Fixed sets & Load sets)
        for fix_item in bc.fixed_node_sets:
            name = fix_item.get("name", "FIX_SET").upper().replace(" ", "_")
            part_name = fix_item.get("part", "")
            node_ids = fix_item.get("node_ids", np.array([], dtype=np.int64))
            if len(node_ids) == 0:
                continue
            gnids = mapper.node_ids(part_name, node_ids)
            buffer.append(f"*NSET, NSET={name}")
            for line in format_card_items(gnids, max_per_line=16):
                buffer.append(line)

        for load_item in bc.loads:
            name = load_item.get("name", "LOAD_SET").upper().replace(" ", "_")
            part_name = load_item.get("part", "")
            node_ids = load_item.get("node_ids", np.array([], dtype=np.int64))
            if len(node_ids) == 0:
                continue
            gnids = mapper.node_ids(part_name, node_ids)
            buffer.append(f"*NSET, NSET={name}")
            for line in format_card_items(gnids, max_per_line=16):
                buffer.append(line)

        # 6) *SURFACE 정의 및 *TIE
        for tie in ties:
            if len(tie.master_faces) == 0 or len(tie.slave_nodes) == 0:
                continue

            tie_name = tie.name.upper().replace(" ", "_")
            surf_master = f"SURF_M_{tie_name}"
            surf_slave = f"SURF_S_{tie_name}"

            # Master Surface: TYPE=ELEMENT, elements + S<face>
            buffer.append(f"*SURFACE, NAME={surf_master}, TYPE=ELEMENT")
            for local_eid, face_id in tie.master_faces:
                geid = mapper.elem_id(tie.master_part, local_eid)
                buffer.append(f"{geid}, S{face_id}")
                if len(buffer) >= chunk_size:
                    flush_buffer(f)

            # Slave Surface: TYPE=NODE, node list
            buffer.append(f"*SURFACE, NAME={surf_slave}, TYPE=NODE")
            slave_gnids = mapper.node_ids(tie.slave_part, tie.slave_nodes)
            for line in format_card_items(slave_gnids, max_per_line=16):
                buffer.append(line)
                if len(buffer) >= chunk_size:
                    flush_buffer(f)

            # *TIE
            tol = float(tie.position_tolerance_mm)
            buffer.append(f"*TIE, NAME={tie_name}, POSITION TOLERANCE={tol:.6f}")
            buffer.append(f"{surf_slave}, {surf_master}")

        # 7) *STEP / *STATIC / *BOUNDARY / *CLOAD / Output / *END STEP
        buffer.append("*STEP")
        buffer.append("*STATIC")

        # Boundary constraints
        if bc.fixed_node_sets:
            buffer.append("*BOUNDARY")
            for fix_item in bc.fixed_node_sets:
                name = fix_item.get("name", "FIX_SET").upper().replace(" ", "_")
                dofs = fix_item.get("dofs", (1, 3))
                buffer.append(f"{name}, {dofs[0]}, {dofs[1]}")

        # Applied loads (*CLOAD)
        # Resultant force uniformly distributed across all target nodes
        cload_lines: List[str] = []
        for load_item in bc.loads:
            part_name = load_item.get("part", "")
            node_ids = load_item.get("node_ids", np.array([], dtype=np.int64))
            force_N = load_item.get("force_N", (0.0, 0.0, 0.0))
            n_target = len(node_ids)
            if n_target == 0:
                continue

            fx = float(force_N[0]) / n_target
            fy = float(force_N[1]) / n_target
            fz = float(force_N[2]) / n_target

            gnids = mapper.node_ids(part_name, node_ids)
            for gid in gnids:
                if abs(fx) > 1e-15:
                    cload_lines.append(f"{gid}, 1, {fx:.9e}")
                if abs(fy) > 1e-15:
                    cload_lines.append(f"{gid}, 2, {fy:.9e}")
                if abs(fz) > 1e-15:
                    cload_lines.append(f"{gid}, 3, {fz:.9e}")

        if cload_lines:
            buffer.append("*CLOAD")
            buffer.extend(cload_lines)

        # Output requests
        buffer.append("*NODE FILE")
        buffer.append("U")
        buffer.append("*EL FILE")
        buffer.append("S, E")
        buffer.append("*END STEP")

        flush_buffer(f)

    logger.info(
        f"CalculiX .inp file written to: {path} "
        f"(Nodes: {mapper.total_nodes:,}, Elements: {mapper.total_elements:,})"
    )
