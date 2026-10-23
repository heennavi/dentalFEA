"""
vtk_builder.py
Conversion of MeshModel and ResultSet into VTK UnstructuredGrid and PolyData.

Provides headless VTK data conversions for FEA visualization without rendering dependencies.

[Units]
- Length / Displacement: mm
- Stress: MPa
- Young's Modulus: MPa
"""

from typing import Optional
import numpy as np
import vtk
from vtk.util import numpy_support

from fea.models import MeshPart, MeshModel, MaterialAssignment
from fea.results import ResultSet
from meshing.gmsh_mesher import gmsh_tet10_to_ccx


def build_part_grid(
    part: MeshPart,
    node_offset: int = 0,
    linear_only: bool = False,
    is_gmsh_order: bool = False,
) -> vtk.vtkUnstructuredGrid:
    """
    Build a vtkUnstructuredGrid from a single MeshPart.

    Args:
        part: MeshPart containing nodes and tetrahedral element connectivity.
        node_offset: Global node index offset for this part in the MeshModel assembly.
        linear_only: If True, uses only the 4 corner vertices (VTK_TETRA) for element topology,
            while preserving all 10 nodal coordinates in the point array for scalar indexing.
        is_gmsh_order: If True, elements are stored in Gmsh 10-node order and need conversion
            via gmsh_tet10_to_ccx. Defaults to False (already in CalculiX/VTK order).

    Returns:
        vtk.vtkUnstructuredGrid: VTK grid with local indexing for the part.
    """
    grid = vtk.vtkUnstructuredGrid()

    # 1. 절점 좌표 설정 (파트 로컬 좌표 및 인덱스 유지)
    nodes = np.ascontiguousarray(part.nodes, dtype=np.float64)
    vtk_pts = vtk.vtkPoints()
    if len(nodes) > 0:
        vtk_pts_arr = numpy_support.numpy_to_vtk(nodes, deep=True)
        vtk_pts.SetData(vtk_pts_arr)
    grid.SetPoints(vtk_pts)

    if len(part.elements) == 0:
        return grid

    # 2. 요소 연결성 처리
    conn = np.array(part.elements, copy=True, dtype=np.int64)

    # 전역 인덱스로 저장되어 있는 경우 로컬 인덱스로 정규화
    if node_offset > 0 and conn.size > 0:
        if np.min(conn) >= node_offset and np.max(conn) >= len(nodes):
            conn = conn - node_offset

    # Gmsh 10절점 순서인 경우 CalculiX/VTK 순서([0,1,2,3,4,5,6,7,9,8])로 변환
    if is_gmsh_order and conn.shape[1] == 10:
        conn = gmsh_tet10_to_ccx(conn)

    is_quadratic = (part.element_type.upper() == "C3D10" or conn.shape[1] == 10)

    # 3. 셀 토폴로지 구성
    if linear_only or not is_quadratic:
        # 4절점 1차 사면체 (VTK_TETRA, 셀 타입 10)
        cell_conn = conn[:, :4]
        cell_type = vtk.VTK_TETRA
        n_pts_per_cell = 4
    else:
        # 10절점 2차 사면체 (VTK_QUADRATIC_TETRA, 셀 타입 24)
        cell_conn = conn
        cell_type = vtk.VTK_QUADRATIC_TETRA
        n_pts_per_cell = 10

    n_cells = len(cell_conn)
    offsets = np.arange(0, (n_cells + 1) * n_pts_per_cell, n_pts_per_cell, dtype=np.int64)
    flat_conn = np.ascontiguousarray(cell_conn, dtype=np.int64).ravel()

    cell_array = vtk.vtkCellArray()
    cell_array.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_support.numpy_to_vtkIdTypeArray(flat_conn, deep=True),
    )

    types = np.full(n_cells, cell_type, dtype=np.uint8)
    vtk_types = numpy_support.numpy_to_vtk(types, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    grid.SetCells(vtk_types, cell_array)

    return grid


def attach_results(
    grid: vtk.vtkUnstructuredGrid,
    rs: ResultSet,
    model: MeshModel,
    part_name: str,
    deform_scale: float = 0.0,
    material_assignment: Optional[MaterialAssignment] = None,
) -> None:
    """
    Attach simulation results from ResultSet as point and cell data to the part grid.

    Attached point data fields (if not None):
        - 'u_mag': Displacement magnitude (mm)
        - 'u': Displacement vector (3 components, mm)
        - 'von_mises': von Mises equivalent stress (MPa)
        - 's_max_principal': Maximum principal stress (MPa)
        - 's_min_principal': Minimum principal stress (MPa)

    Attached cell data fields (optional):
        - 'E_MPa': Young's modulus per element (MPa) if material_assignment is provided.

    Args:
        grid: Target vtkUnstructuredGrid for the part.
        rs: Unified ResultSet containing simulation field arrays.
        model: MeshModel assembly containing part indexing metadata.
        part_name: Name of the part corresponding to grid.
        deform_scale: Scaling factor for mesh deformation (0.0 = undeformed).
        material_assignment: Optional MaterialAssignment to attach element Young's modulus.
    """
    part_slice = rs.slice_for_part(model, part_name)
    n_nodes = grid.GetNumberOfPoints()

    # 1. 절점 데이터(Point Data) 부착
    point_data = grid.GetPointData()

    if rs.u is not None:
        u_part = np.ascontiguousarray(rs.u[part_slice], dtype=np.float64)
        if len(u_part) == n_nodes:
            vtk_u = numpy_support.numpy_to_vtk(u_part, deep=True)
            vtk_u.SetName("u")
            point_data.AddArray(vtk_u)
            point_data.SetActiveVectors("u")

    if rs.u_mag is not None:
        u_mag_part = np.ascontiguousarray(rs.u_mag[part_slice], dtype=np.float64)
        if len(u_mag_part) == n_nodes:
            vtk_umag = numpy_support.numpy_to_vtk(u_mag_part, deep=True)
            vtk_umag.SetName("u_mag")
            point_data.AddArray(vtk_umag)

    if rs.von_mises is not None:
        vm_part = np.ascontiguousarray(rs.von_mises[part_slice], dtype=np.float64)
        if len(vm_part) == n_nodes:
            vtk_vm = numpy_support.numpy_to_vtk(vm_part, deep=True)
            vtk_vm.SetName("von_mises")
            point_data.AddArray(vtk_vm)
            point_data.SetActiveScalars("von_mises")

    if rs.s_max_principal is not None:
        smax_part = np.ascontiguousarray(rs.s_max_principal[part_slice], dtype=np.float64)
        if len(smax_part) == n_nodes:
            vtk_smax = numpy_support.numpy_to_vtk(smax_part, deep=True)
            vtk_smax.SetName("s_max_principal")
            point_data.AddArray(vtk_smax)

    if rs.s_min_principal is not None:
        smin_part = np.ascontiguousarray(rs.s_min_principal[part_slice], dtype=np.float64)
        if len(smin_part) == n_nodes:
            vtk_smin = numpy_support.numpy_to_vtk(smin_part, deep=True)
            vtk_smin.SetName("s_min_principal")
            point_data.AddArray(vtk_smin)

    # 2. 요소 데이터(Cell Data) 부착 - E_MPa
    if material_assignment is not None and part_name in material_assignment.element_material:
        elem_mat_indices = material_assignment.element_material[part_name]
        materials = material_assignment.materials
        e_lookup = np.array([m.E for m in materials], dtype=np.float64)
        e_arr = e_lookup[elem_mat_indices]
        if len(e_arr) == grid.GetNumberOfCells():
            vtk_e = numpy_support.numpy_to_vtk(
                np.ascontiguousarray(e_arr, dtype=np.float64),
                deep=True,
            )
            vtk_e.SetName("E_MPa")
            grid.GetCellData().AddArray(vtk_e)

    # 3. 비변형 원본 좌표 보존 (FieldData에 "orig_points" 저장)
    field_data = grid.GetFieldData()
    if field_data.GetArray("orig_points") is None and grid.GetPoints() is not None:
        orig_pts = numpy_support.vtk_to_numpy(grid.GetPoints().GetData()).copy()
        vtk_orig = numpy_support.numpy_to_vtk(orig_pts, deep=True)
        vtk_orig.SetName("orig_points")
        field_data.AddArray(vtk_orig)

    # 4. 변형 스케일 적용
    set_deform_scale(grid, deform_scale)


def set_deform_scale(grid: vtk.vtkUnstructuredGrid, scale: float) -> None:
    """
    Update point coordinates based on preserved original coordinates and displacement vector.

    Formula: coords = orig_points + scale * u (with NaNs treated as 0.0).

    Args:
        grid: vtkUnstructuredGrid containing points and result arrays.
        scale: Deformation amplification factor.
    """
    if grid.GetPoints() is None:
        return

    field_data = grid.GetFieldData()
    orig_arr = field_data.GetArray("orig_points")

    # 원본 좌표가 저장되어 있지 않으면 현재 좌표를 원본으로 보존
    if orig_arr is None:
        orig_pts = numpy_support.vtk_to_numpy(grid.GetPoints().GetData()).copy()
        vtk_orig = numpy_support.numpy_to_vtk(orig_pts, deep=True)
        vtk_orig.SetName("orig_points")
        field_data.AddArray(vtk_orig)
        orig_coords = orig_pts
    else:
        orig_coords = numpy_support.vtk_to_numpy(orig_arr)

    u_arr = grid.GetPointData().GetArray("u")

    if u_arr is None or scale == 0.0:
        new_coords = orig_coords.copy()
    else:
        u = numpy_support.vtk_to_numpy(u_arr)
        u_clean = np.nan_to_num(u, nan=0.0)
        new_coords = orig_coords + scale * u_clean

    new_pts_arr = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(new_coords, dtype=np.float64),
        deep=True,
    )
    grid.GetPoints().SetData(new_pts_arr)
    grid.GetPoints().Modified()
    grid.Modified()


def surface_of(grid: vtk.vtkUnstructuredGrid) -> vtk.vtkPolyData:
    """
    Extract the outer surface boundary as vtkPolyData while preserving point data.

    Quadratic elements are subdivided into linear triangles on the exterior surface.

    Args:
        grid: Input vtkUnstructuredGrid.

    Returns:
        vtk.vtkPolyData: Extracted boundary surface mesh.
    """
    geo_filter = vtk.vtkGeometryFilter()
    geo_filter.SetInputData(grid)
    geo_filter.Update()
    return geo_filter.GetOutput()
