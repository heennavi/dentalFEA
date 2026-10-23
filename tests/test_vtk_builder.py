"""
test_vtk_builder.py
Unit tests for postprocess/vtk_builder.py.

Verifies:
1. Synthetic C3D4 (2 cells) and C3D10 (1 cell) mesh topology and point count.
2. C3D10 cell volume computation using vtkMeshQuality (positive volume matching analytical volume).
3. Result field attachment (point/cell data) and reversible deformation scaling (0 -> 2.0 -> 0).
4. Preservation of point data (von_mises) after surface extraction via surface_of().
5. Handling of global node offset and Gmsh vs CalculiX node ordering.
"""

import os
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from fea.models import MeshPart, MeshModel, MaterialDef, MaterialAssignment
from fea.results import ResultSet
from postprocess.frd_parser import parse_frd
from postprocess.result_fields import build_result_set
from postprocess.vtk_builder import (
    build_part_grid,
    attach_results,
    set_deform_scale,
    surface_of,
)


@pytest.fixture
def c3d4_two_tetra_part() -> MeshPart:
    """Fixture providing a synthetic two-element C3D4 mesh sharing an interface."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)

    elements = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=np.int64)

    return MeshPart(
        name="bone",
        nodes=nodes,
        elements=elements,
        element_type="C3D4",
    )


@pytest.fixture
def c3d10_single_tetra_part() -> MeshPart:
    """
    Fixture providing a synthetic single-element C3D10 quadratic tetrahedron.

    CalculiX / VTK C3D10 convention:
    Corner vertices: 0, 1, 2, 3
    Midside nodes:
      4: 0-1, 5: 1-2, 6: 2-0, 7: 0-3, 8: 1-3, 9: 2-3
    """
    nodes = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.5, 0.0, 0.0],  # 4 (0-1)
        [0.5, 0.5, 0.0],  # 5 (1-2)
        [0.0, 0.5, 0.0],  # 6 (2-0)
        [0.0, 0.0, 0.5],  # 7 (0-3)
        [0.5, 0.0, 0.5],  # 8 (1-3)
        [0.0, 0.5, 0.5],  # 9 (2-3)
    ], dtype=np.float64)

    elements = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ], dtype=np.int64)

    return MeshPart(
        name="implant",
        nodes=nodes,
        elements=elements,
        element_type="C3D10",
    )


def test_build_part_grid_c3d4(c3d4_two_tetra_part):
    """Test building vtkUnstructuredGrid from 2-element C3D4 part."""
    grid = build_part_grid(c3d4_two_tetra_part, node_offset=0)

    assert isinstance(grid, vtk.vtkUnstructuredGrid)
    assert grid.GetNumberOfPoints() == 5
    assert grid.GetNumberOfCells() == 2
    assert grid.GetCellType(0) == vtk.VTK_TETRA
    assert grid.GetCellType(1) == vtk.VTK_TETRA


def test_build_part_grid_c3d10_quadratic_and_linear_only(c3d10_single_tetra_part):
    """Test building quadratic C3D10 grid and linear_only simplified visualization grid."""
    # 1. Quadratic (VTK_QUADRATIC_TETRA, 24)
    grid_quad = build_part_grid(c3d10_single_tetra_part, node_offset=0, linear_only=False)
    assert grid_quad.GetNumberOfPoints() == 10
    assert grid_quad.GetNumberOfCells() == 1
    assert grid_quad.GetCellType(0) == vtk.VTK_QUADRATIC_TETRA

    # 2. linear_only (VTK_TETRA, 10) but keeping all 10 nodes for indexing
    grid_lin = build_part_grid(c3d10_single_tetra_part, node_offset=0, linear_only=True)
    assert grid_lin.GetNumberOfPoints() == 10
    assert grid_lin.GetNumberOfCells() == 1
    assert grid_lin.GetCellType(0) == vtk.VTK_TETRA

    # Ensure cell connectivity uses only the first 4 corner vertices
    cell_pt_ids = vtk.vtkIdList()
    grid_lin.GetCellPoints(0, cell_pt_ids)
    assert cell_pt_ids.GetNumberOfIds() == 4
    for i in range(4):
        assert cell_pt_ids.GetId(i) == i


def test_c3d10_volume_mesh_quality(c3d10_single_tetra_part):
    """
    Verify that C3D10 cell volume computed by VTK matches analytical 4-vertex tetrahedron volume.

    CalculiX/VTK node ordering ensures positive volume close to 1/6 (0.16667).
    Gmsh node ordering causes severe shape distortion (~0.04167) unless converted.
    """
    grid = build_part_grid(c3d10_single_tetra_part, node_offset=0)

    mq = vtk.vtkMeshQuality()
    mq.SetInputData(grid)
    mq.SetTetQualityMeasureToVolume()
    mq.Update()

    out = mq.GetOutput()
    vol_arr = out.GetCellData().GetArray("Quality")
    assert vol_arr is not None
    vol = vol_arr.GetValue(0)

    expected_vol = 1.0 / 6.0
    assert vol > 0.0
    assert np.isclose(vol, expected_vol, rtol=1e-4)

    # Test Gmsh ordering conversion
    # In Gmsh, 8 is between 2-3 and 9 is between 1-3 (swapped compared to CalculiX/VTK)
    gmsh_elements = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
    ], dtype=np.int64)
    part_gmsh = MeshPart(
        name="implant",
        nodes=c3d10_single_tetra_part.nodes,
        elements=gmsh_elements,
        element_type="C3D10",
    )

    # With is_gmsh_order=True, it converts back to CalculiX/VTK order
    grid_converted = build_part_grid(part_gmsh, node_offset=0, is_gmsh_order=True)
    mq_conv = vtk.vtkMeshQuality()
    mq_conv.SetInputData(grid_converted)
    mq_conv.SetTetQualityMeasureToVolume()
    mq_conv.Update()
    vol_converted = mq_conv.GetOutput().GetCellData().GetArray("Quality").GetValue(0)
    assert np.isclose(vol_converted, expected_vol, rtol=1e-4)


def test_node_offset_handling():
    """Verify that global node indices in elements are correctly normalized using node_offset."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    offset = 100
    elements_global = np.array([[100, 101, 102, 103]], dtype=np.int64)

    part = MeshPart(
        name="implant",
        nodes=nodes,
        elements=elements_global,
        element_type="C3D4",
    )

    grid = build_part_grid(part, node_offset=offset)
    cell_pt_ids = vtk.vtkIdList()
    grid.GetCellPoints(0, cell_pt_ids)
    assert [cell_pt_ids.GetId(i) for i in range(4)] == [0, 1, 2, 3]


def test_attach_results_and_deform_scale():
    """
    Verify attaching ResultSet fields, element Young's modulus, and reversible deformation scaling.
    """
    # 2 parts: bone (5 nodes, 2 C3D4 cells) and implant (10 nodes, 1 C3D10 cell)
    nodes_bone = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    elems_bone = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    part_bone = MeshPart(name="bone", nodes=nodes_bone, elements=elems_bone, element_type="C3D4")

    nodes_implant = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    elems_implant = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    part_implant = MeshPart(name="implant", nodes=nodes_implant, elements=elems_implant, element_type="C3D10")

    model = MeshModel(parts=[part_bone, part_implant])
    tot_nodes = model.total_nodes()  # 5 + 10 = 15

    # Create synthetic ResultSet
    u = np.zeros((tot_nodes, 3), dtype=np.float64)
    u[:, 0] = 0.1  # ux = 0.1
    u[:, 1] = 0.2  # uy = 0.2
    u[:, 2] = 0.3  # uz = 0.3
    # Introduce NaN on one bone node to verify NaN handling
    u[0, :] = np.nan

    u_mag = np.linalg.norm(np.nan_to_num(u), axis=1)
    vm = np.linspace(10.0, 150.0, tot_nodes)
    s_max = vm * 1.2
    s_min = -vm * 0.5

    rs = ResultSet(
        time=1.0,
        u=u,
        u_mag=u_mag,
        von_mises=vm,
        s_max_principal=s_max,
        s_min_principal=s_min,
    )

    mat_assign = MaterialAssignment(
        element_material={
            "bone": np.array([0, 1], dtype=np.int64),
            "implant": np.array([2], dtype=np.int64),
        },
        materials=[
            MaterialDef("cortical", E=13700.0, nu=0.3),
            MaterialDef("cancellous", E=1500.0, nu=0.3),
            MaterialDef("titanium", E=110000.0, nu=0.35),
        ],
    )

    # 1. Build and attach to bone
    grid_bone = build_part_grid(part_bone, node_offset=0)
    attach_results(
        grid=grid_bone,
        rs=rs,
        model=model,
        part_name="bone",
        deform_scale=0.0,
        material_assignment=mat_assign,
    )

    # Check PointData arrays
    pd_bone = grid_bone.GetPointData()
    assert pd_bone.GetArray("u") is not None
    assert pd_bone.GetArray("u").GetNumberOfTuples() == 5
    assert pd_bone.GetArray("u").GetNumberOfComponents() == 3

    assert pd_bone.GetArray("u_mag") is not None
    assert pd_bone.GetArray("u_mag").GetNumberOfTuples() == 5

    assert pd_bone.GetArray("von_mises") is not None
    assert pd_bone.GetArray("von_mises").GetNumberOfTuples() == 5

    assert pd_bone.GetArray("s_max_principal") is not None
    assert pd_bone.GetArray("s_max_principal").GetNumberOfTuples() == 5

    assert pd_bone.GetArray("s_min_principal") is not None
    assert pd_bone.GetArray("s_min_principal").GetNumberOfTuples() == 5

    # Check CellData E_MPa
    cd_bone = grid_bone.GetCellData()
    assert cd_bone.GetArray("E_MPa") is not None
    e_bone = numpy_support.vtk_to_numpy(cd_bone.GetArray("E_MPa"))
    assert np.allclose(e_bone, [13700.0, 1500.0])

    # 2. Check reversible deformation scaling (0.0 -> 2.0 -> 0.0)
    orig_coords = numpy_support.vtk_to_numpy(grid_bone.GetPoints().GetData()).copy()
    assert np.allclose(orig_coords, nodes_bone)

    # Apply scale 2.0
    set_deform_scale(grid_bone, scale=2.0)
    deformed_coords = numpy_support.vtk_to_numpy(grid_bone.GetPoints().GetData())

    # For node 0 (NaN in u), displacement is 0.0 -> coordinates unchanged
    assert np.allclose(deformed_coords[0], nodes_bone[0])
    # For nodes 1..4: coords = nodes_bone + 2.0 * [0.1, 0.2, 0.3]
    expected_deformed = nodes_bone.copy()
    expected_deformed[1:] += 2.0 * np.array([0.1, 0.2, 0.3])
    assert np.allclose(deformed_coords, expected_deformed)

    # Restore to scale 0.0
    set_deform_scale(grid_bone, scale=0.0)
    restored_coords = numpy_support.vtk_to_numpy(grid_bone.GetPoints().GetData())
    assert np.allclose(restored_coords, orig_coords)


def test_surface_of_preserves_von_mises(c3d10_single_tetra_part):
    """Verify that surface extraction via surface_of retains point data (von_mises)."""
    grid = build_part_grid(c3d10_single_tetra_part, node_offset=0)

    # Synthetic ResultSet for 10 nodes
    vm = np.linspace(25.0, 80.0, 10)
    rs = ResultSet(
        time=1.0,
        u_mag=np.zeros(10),
        von_mises=vm,
    )
    model = MeshModel(parts=[c3d10_single_tetra_part])

    attach_results(grid, rs, model, part_name="implant")

    # Extract outer surface
    poly = surface_of(grid)

    assert isinstance(poly, vtk.vtkPolyData)
    assert poly.GetNumberOfPoints() > 0
    assert poly.GetNumberOfPolys() > 0

    vm_poly = poly.GetPointData().GetArray("von_mises")
    assert vm_poly is not None
    assert vm_poly.GetNumberOfTuples() == poly.GetNumberOfPoints()
    r = vm_poly.GetRange()
    assert np.isclose(r[0], 25.0)
    assert np.isclose(r[1], 80.0)


def test_cube_compression_integration():
    """Verify end-to-end integration with the real cube_compression.frd fixture."""
    fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    frd_path = os.path.join(fixture_dir, "cube_compression.frd")

    if not os.path.exists(frd_path):
        pytest.skip("cube_compression.frd fixture not found.")

    frd_data = parse_frd(frd_path)

    # Synthetic 1-part model matching cube nodes (8 nodes, 5 C3D4 cells)
    nodes = frd_data.node_coords
    elements = np.array([
        [0, 1, 2, 4],
        [1, 2, 3, 7],
        [1, 4, 5, 7],
        [2, 4, 6, 7],
        [1, 2, 4, 7],
    ], dtype=np.int64)
    part = MeshPart(name="cube", nodes=nodes, elements=elements, element_type="C3D4")
    model = MeshModel(parts=[part])

    rs = build_result_set(frd_data, model)

    grid = build_part_grid(part, node_offset=0)
    attach_results(grid, rs, model, part_name="cube", deform_scale=1.0)

    assert grid.GetNumberOfPoints() == 8
    assert grid.GetNumberOfCells() == 5
    assert grid.GetPointData().GetArray("von_mises") is not None
    assert grid.GetPointData().GetArray("u") is not None

    poly = surface_of(grid)
    assert poly.GetNumberOfPolys() > 0
    assert poly.GetPointData().GetArray("von_mises") is not None
