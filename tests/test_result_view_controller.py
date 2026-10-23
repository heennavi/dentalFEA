"""
test_result_view_controller.py
Unit tests for ui/result_view_controller.py.

Verifies:
1. Full lifecycle (set_results -> set_field -> range_modes -> colormaps -> clipping -> deform -> clear)
   with actor count restoration.
2. Original scene actor visibility backup and restoration.
3. Deformation scale reversal (scale=0.0 restores exact original nodal coordinates).
4. Section clipping pipeline on linear grids.
5. Large mesh (>300k elements) info signal and linear fallback.
6. Off-screen VTK rendering producing non-empty frame buffer and screenshot saved to docs/.
"""

import os
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support

from fea.models import MeshPart, MeshModel, MaterialDef, MaterialAssignment
from fea.results import ResultSet
from ui.result_view_controller import ResultViewController, FIELD_DISPLAY_CONFIG


@pytest.fixture
def offscreen_renderer():
    """Create a headless VTK renderer and render window."""
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.12, 0.12, 0.14)
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(640, 480)
    ren._rw = rw  # keep python reference alive
    return ren


@pytest.fixture
def synthetic_assembly() -> tuple[MeshModel, ResultSet, MaterialAssignment]:
    """Create a two-part assembly (bone + implant) with realistic result fields."""
    # Part 1: Bone (C3D4, 5 nodes, 2 elements)
    bone_nodes = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
    ], dtype=np.float64)
    bone_elements = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=np.int64)
    bone_part = MeshPart(name="bone", nodes=bone_nodes, elements=bone_elements, element_type="C3D4")

    # Part 2: Implant (C3D10, 10 nodes, 1 element)
    implant_nodes = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.5, 0.0, 0.0],  # 4
        [0.5, 0.5, 0.0],  # 5
        [0.0, 0.5, 0.0],  # 6
        [0.0, 0.0, 0.5],  # 7
        [0.5, 0.0, 0.5],  # 8
        [0.0, 0.5, 0.5],  # 9
    ], dtype=np.float64) + np.array([0.5, 0.5, 1.0])  # slightly shifted

    implant_elements = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ], dtype=np.int64)
    implant_part = MeshPart(name="implant", nodes=implant_nodes, elements=implant_elements, element_type="C3D10")

    model = MeshModel(parts=[bone_part, implant_part])
    total_nodes = len(bone_nodes) + len(implant_nodes)  # 15 nodes

    # Synthetic result fields
    u = np.zeros((total_nodes, 3), dtype=np.float64)
    u[:, 0] = np.linspace(0.01, 0.05, total_nodes)
    u[:, 2] = np.linspace(-0.02, -0.10, total_nodes)
    u_mag = np.linalg.norm(u, axis=1)

    von_mises = np.linspace(5.0, 150.0, total_nodes)
    s_max = np.linspace(10.0, 180.0, total_nodes)
    s_min = np.linspace(-50.0, 5.0, total_nodes)

    rs = ResultSet(
        time=0.0,
        u=u,
        u_mag=u_mag,
        von_mises=von_mises,
        s_max_principal=s_max,
        s_min_principal=s_min,
    )

    # Material assignment
    mat_cortical = MaterialDef(name="cortical", E=13700.0, nu=0.3)
    mat_ti = MaterialDef(name="titanium", E=110000.0, nu=0.35)
    mat_assign = MaterialAssignment(
        materials=[mat_cortical, mat_ti],
        element_material={
            "bone": np.array([0, 0], dtype=np.int32),
            "implant": np.array([1], dtype=np.int32),
        },
    )

    return model, rs, mat_assign


def test_result_view_controller_lifecycle(offscreen_renderer, synthetic_assembly):
    """Verify set_results -> set_field -> range_mode -> colormap -> clip -> deform -> clear lifecycle."""
    ren = offscreen_renderer
    model, rs, mat_assign = synthetic_assembly

    initial_props_count = ren.GetViewProps().GetNumberOfItems()

    controller = ResultViewController(ren)

    # 1. set_results
    controller.set_results(model, rs, mat_assign)
    # 2 mesh actors (bone + implant) + 1 scalar bar = 3 props added
    assert ren.GetViewProps().GetNumberOfItems() == initial_props_count + 3

    # 2. set_field switching
    for field_name in ["von_mises", "u_mag", "s_max_principal", "s_min_principal", "E_MPa"]:
        controller.set_field(field_name)
        assert controller._current_field == field_name

    # 3. set_range_mode
    controller.set_range_mode("global")
    assert controller._range_mode == "global"

    controller.set_range_mode("per_part")
    assert controller._range_mode == "per_part"
    controller.set_active_scalar_part("bone")
    assert controller._active_scalar_part == "bone"
    controller.set_active_scalar_part("implant")
    assert controller._active_scalar_part == "implant"

    controller.set_range_mode("manual", manual=(10.0, 80.0))
    assert controller._range_mode == "manual"
    assert controller._manual_range == (10.0, 80.0)

    # 4. set_colormap
    for cmap in ["rainbow", "viridis", "coolwarm"]:
        controller.set_colormap(cmap)
        assert controller._colormap_name == cmap

    # 5. visibility and opacity
    controller.set_part_visible("implant", False)
    assert controller._actors["implant"].GetVisibility() == 0
    controller.set_part_visible("implant", True)
    assert controller._actors["implant"].GetVisibility() == 1

    controller.set_opacity("bone", 0.5)
    assert controller._actors["bone"].GetProperty().GetOpacity() == pytest.approx(0.5)

    controller.set_edges_visible(True)
    assert controller._actors["bone"].GetProperty().GetEdgeVisibility() == 1
    controller.set_edges_visible(False)
    assert controller._actors["bone"].GetProperty().GetEdgeVisibility() == 0

    # 6. clipping plane
    controller.set_clip_plane(True, origin=(1.0, 1.0, 1.0), normal=(0.0, 0.0, 1.0))
    assert controller._clip_enabled is True
    controller.set_clip_plane(False)
    assert controller._clip_enabled is False

    # 7. deformation scaling and coordinate reversibility
    bone_grid = controller._grids["bone"]
    orig_pts = numpy_support.vtk_to_numpy(bone_grid.GetFieldData().GetArray("orig_points")).copy()

    controller.set_deform_scale(10.0)
    deformed_pts = numpy_support.vtk_to_numpy(bone_grid.GetPoints().GetData())
    assert not np.allclose(orig_pts, deformed_pts)

    controller.set_deform_scale(0.0)
    restored_pts = numpy_support.vtk_to_numpy(bone_grid.GetPoints().GetData())
    np.testing.assert_allclose(orig_pts, restored_pts, atol=1e-12)

    # 8. clear and verify actor count restoration
    controller.clear()
    assert ren.GetViewProps().GetNumberOfItems() == initial_props_count


def test_original_scene_preservation(offscreen_renderer, synthetic_assembly):
    """Verify original scene actors are hidden during set_results and restored on clear()."""
    ren = offscreen_renderer
    model, rs, mat_assign = synthetic_assembly

    # Add 2 dummy actors (one visible, one hidden)
    sphere = vtk.vtkSphereSource()
    sphere.Update()
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputConnection(sphere.GetOutputPort())
    actor_vis = vtk.vtkActor()
    actor_vis.SetMapper(mapper1)
    actor_vis.SetVisibility(True)
    ren.AddActor(actor_vis)

    cone = vtk.vtkConeSource()
    cone.Update()
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputConnection(cone.GetOutputPort())
    actor_hid = vtk.vtkActor()
    actor_hid.SetMapper(mapper2)
    actor_hid.SetVisibility(False)
    ren.AddActor(actor_hid)

    assert actor_vis.GetVisibility() == 1
    assert actor_hid.GetVisibility() == 0

    controller = ResultViewController(ren)
    controller.set_results(model, rs, mat_assign)

    # During results display, original actors must be hidden
    assert actor_vis.GetVisibility() == 0
    assert actor_hid.GetVisibility() == 0

    # After clear(), original visibility states must be restored
    controller.clear()
    assert actor_vis.GetVisibility() == 1
    assert actor_hid.GetVisibility() == 0

    # Cleanup
    ren.RemoveActor(actor_vis)
    ren.RemoveActor(actor_hid)


def test_large_mesh_linear_fallback(offscreen_renderer):
    """Verify mesh with > 300,000 elements emits info signal and enables linear fallback."""
    ren = offscreen_renderer

    # Create dummy part with 300,001 linear elements
    n_elems = 300_001
    nodes = np.zeros((4, 3), dtype=np.float64)
    elements = np.zeros((n_elems, 4), dtype=np.int64)
    large_part = MeshPart(name="large_bone", nodes=nodes, elements=elements, element_type="C3D4")
    model = MeshModel(parts=[large_part])
    rs = ResultSet(time=0.0, u=np.zeros((4, 3)))

    controller = ResultViewController(ren)
    received_msgs = []
    controller.info.connect(received_msgs.append)

    controller.set_results(model, rs)

    assert controller._use_linear_default is True
    assert len(received_msgs) == 1
    assert "300,001" in received_msgs[0]
    controller.clear()


def test_offscreen_render_and_screenshot(offscreen_renderer, synthetic_assembly):
    """Verify off-screen rendering generates valid pixels (variance > 0) and save screenshot."""
    ren = offscreen_renderer
    rw = ren.GetRenderWindow()
    model, rs, mat_assign = synthetic_assembly

    controller = ResultViewController(ren)
    controller.set_results(model, rs, mat_assign)
    controller.set_field("von_mises")
    controller.set_colormap("rainbow")
    controller.set_edges_visible(True)

    # Reset camera to view all actors properly
    ren.ResetCamera()
    cam = ren.GetActiveCamera()
    cam.Azimuth(30)
    cam.Elevation(20)
    ren.ResetCameraClippingRange()
    rw.Render()

    # Capture frame buffer
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()

    img = w2i.GetOutput()
    dims = img.GetDimensions()
    assert dims[0] == 640 and dims[1] == 480

    scalars = img.GetPointData().GetScalars()
    arr = numpy_support.vtk_to_numpy(scalars)
    assert arr.size > 0

    # Ensure rendered image is not completely blank (variance > 0)
    var = float(np.var(arr))
    assert var > 10.0, f"Rendered image has negligible variance: {var}"

    # Save screenshot to docs/step4_4_result_view_screenshot.png
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
    os.makedirs(docs_dir, exist_ok=True)
    screenshot_path = os.path.join(docs_dir, "step4_4_result_view_screenshot.png")

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(screenshot_path)
    writer.SetInputData(img)
    writer.Write()

    assert os.path.exists(screenshot_path)
    assert os.path.getsize(screenshot_path) > 1000

    controller.clear()
