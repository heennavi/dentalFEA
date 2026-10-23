"""
test_result_probe.py
Unit tests for ui/result_probe.py.

Verifies:
1. Pure coordinate-based node query via find_nearest_node (headless/off-screen without VTK window picker).
2. Accuracy of retrieved physical quantities (von Mises, principal stresses, displacement, Young's modulus).
3. Part-filtered nearest node queries.
4. Maximum stress marker creation, caption text verification, camera focus, and max_marked signal emission.
5. Marker lifecycle and complete clean-up via clear_markers() and detach().
"""

import numpy as np
import pytest
import vtk

from fea.models import MeshPart, MeshModel, MaterialDef, MaterialAssignment
from fea.results import ResultSet
from ui.result_view_controller import ResultViewController
from ui.result_probe import ResultProbe


@pytest.fixture
def offscreen_controller():
    """Create an off-screen ResultViewController with synthetic bone and implant parts."""
    ren = vtk.vtkRenderer()
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(400, 300)
    ren._rw = rw  # keep reference

    # Part 1: Bone (C3D4, 4 nodes)
    bone_nodes = np.array([
        [0.0, 0.0, 0.0],  # 0
        [2.0, 0.0, 0.0],  # 1
        [0.0, 2.0, 0.0],  # 2
        [0.0, 0.0, 2.0],  # 3
    ], dtype=np.float64)
    bone_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    bone_part = MeshPart(name="bone", nodes=bone_nodes, elements=bone_elements, element_type="C3D4")

    # Part 2: Implant (C3D4, 4 nodes, translated to X=10)
    implant_nodes = np.array([
        [10.0, 0.0, 0.0],  # 0 (global 4)
        [12.0, 0.0, 0.0],  # 1 (global 5)
        [10.0, 2.0, 0.0],  # 2 (global 6)
        [10.0, 0.0, 2.0],  # 3 (global 7)
    ], dtype=np.float64)
    implant_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    implant_part = MeshPart(name="implant", nodes=implant_nodes, elements=implant_elements, element_type="C3D4")

    model = MeshModel(parts=[bone_part, implant_part])
    total_nodes = 8

    # Results
    u = np.zeros((total_nodes, 3), dtype=np.float64)
    u[1] = [0.01, 0.02, 0.03]  # bone node 1
    u[5] = [0.05, 0.0, -0.05]  # implant node 1
    u_mag = np.linalg.norm(u, axis=1)

    von_mises = np.array([10.0, 25.0, 15.0, 20.0, 50.0, 180.0, 80.0, 95.0], dtype=np.float64)
    s_max = np.array([15.0, 30.0, 20.0, 25.0, 60.0, 210.0, 90.0, 110.0], dtype=np.float64)
    s_min = np.array([-5.0, -10.0, -8.0, -12.0, -30.0, -90.0, -40.0, -50.0], dtype=np.float64)

    rs = ResultSet(
        time=0.0,
        u=u,
        u_mag=u_mag,
        von_mises=von_mises,
        s_max_principal=s_max,
        s_min_principal=s_min,
    )

    mat_bone = MaterialDef(name="CORTICAL", E=14000.0, nu=0.3)
    mat_implant = MaterialDef(name="TITANIUM", E=110000.0, nu=0.35)
    materials = MaterialAssignment(
        element_material={"bone": np.array([0]), "implant": np.array([1])},
        materials=[mat_bone, mat_implant],
    )

    ctrl = ResultViewController(ren)
    ctrl.set_results(model, rs, materials)
    return ctrl


def test_find_nearest_node_coordinate_query(offscreen_controller):
    """Test pure coordinate query without VTK window picker."""
    ctrl = offscreen_controller
    probe = ResultProbe()
    probe.attach(ctrl)

    # Query coordinate very close to bone node 1: (2.0, 0.0, 0.0)
    query_pos = (1.98, 0.01, -0.02)
    res = probe.find_nearest_node(query_pos)

    assert res is not None
    assert res["part"] == "bone"
    assert res["node"] == 1
    assert res["xyz"] == pytest.approx([2.0, 0.0, 0.0])
    assert res["von_mises"] == pytest.approx(25.0)
    assert res["s_max_principal"] == pytest.approx(30.0)
    assert res["s_min_principal"] == pytest.approx(-10.0)
    assert res["u"] == pytest.approx([0.01, 0.02, 0.03])
    assert res["u_mag"] == pytest.approx(np.linalg.norm([0.01, 0.02, 0.03]))
    assert res["E_MPa"] == pytest.approx(14000.0)

    # Query coordinate very close to implant node 1 (global 5): (12.0, 0.0, 0.0)
    query_implant = (11.9, 0.05, 0.0)
    res_imp = probe.find_nearest_node(query_implant)
    assert res_imp is not None
    assert res_imp["part"] == "implant"
    assert res_imp["node"] == 5
    assert res_imp["von_mises"] == pytest.approx(180.0)
    assert res_imp["E_MPa"] == pytest.approx(110000.0)


def test_find_nearest_node_with_part_filter(offscreen_controller):
    """Test filtering by part name even when the query coordinate is closer to another part."""
    ctrl = offscreen_controller
    probe = ResultProbe()
    probe.attach(ctrl)

    # Coordinate is at (9.0, 0.0, 0.0), closer to implant node 0 at (10,0,0) than bone node 1 at (2,0,0)
    query_pos = (9.0, 0.0, 0.0)

    # Without filter -> should pick implant
    res_any = probe.find_nearest_node(query_pos)
    assert res_any["part"] == "implant"
    assert res_any["node"] == 4  # implant node 0 (global 4)

    # With part_name="bone" -> should pick bone node 1
    res_bone = probe.find_nearest_node(query_pos, part_name="bone")
    assert res_bone["part"] == "bone"
    assert res_bone["node"] == 1


def test_mark_maximum_and_signals(offscreen_controller):
    """Test marking maximum stress and signal emission."""
    ctrl = offscreen_controller
    probe = ResultProbe()
    probe.attach(ctrl)

    emitted_payloads = []
    probe.max_marked.connect(lambda data: emitted_payloads.append(data))

    # Mark global maximum von Mises -> node 5 (implant node 1) with 180.0 MPa
    max_res = probe.mark_maximum(field="von_mises", focus_camera=True)

    assert max_res is not None
    assert max_res["node_idx"] == 5
    assert max_res["part_name"] == "implant"
    assert max_res["value"] == pytest.approx(180.0)
    assert len(emitted_payloads) == 1
    assert emitted_payloads[0]["node_idx"] == 5

    # Check that VTK marker actors were added to the renderer
    assert probe._max_sphere_actor is not None
    assert probe._max_caption_actor is not None
    caption_text = probe._max_caption_actor.GetCaption()
    assert "최대" in caption_text
    assert "von Mises" in caption_text
    assert "180.0" in caption_text

    # Camera focal point should match peak coordinates (12, 0, 0)
    cam = ctrl.renderer.GetActiveCamera()
    focal_pt = cam.GetFocalPoint()
    assert focal_pt == pytest.approx((12.0, 0.0, 0.0))

    # Mark maximum in bone only -> bone node 1 with 25.0 MPa
    bone_max = probe.mark_maximum(field="von_mises", part_name="bone")
    assert bone_max["node_idx"] == 1
    assert bone_max["part_name"] == "bone"
    assert bone_max["value"] == pytest.approx(25.0)

    # Clear markers
    probe.clear_markers()
    assert probe._max_sphere_actor is None
    assert probe._max_caption_actor is None


def test_probe_marker_update_and_clear(offscreen_controller):
    """Test probe marker creation via internal update and clean up."""
    ctrl = offscreen_controller
    probe = ResultProbe()
    probe.attach(ctrl)

    probe_data = probe.find_nearest_node((0.0, 0.0, 0.0))
    probe._update_probe_marker(probe_data)

    assert probe._probe_sphere_actor is not None
    assert probe._probe_caption_actor is not None
    assert "Node #0" in probe._probe_caption_actor.GetCaption()

    probe.clear_markers()
    assert probe._probe_sphere_actor is None
    assert probe._probe_caption_actor is None


def test_mouse_drag_threshold(offscreen_controller):
    """Test that mouse drag (>3px) ignores probe while click (<=3px) triggers pick."""
    ctrl = offscreen_controller
    probe = ResultProbe()
    probe.attach(ctrl)
    probe.enable_click_probe(True)

    picked = []
    probe._perform_pick = lambda x, y: picked.append((x, y))

    class MockCaller:
        def __init__(self, pos, ctrl=False):
            self.pos = pos
            self.ctrl = ctrl
        def GetEventPosition(self):
            return self.pos
        def GetControlKey(self):
            return self.ctrl

    # 1. Drag: press at (100, 100), release at (110, 100) -> distance 10px > 3px
    probe._on_left_button_press(MockCaller((100, 100)), "LeftButtonPressEvent")
    probe._on_left_button_release(MockCaller((110, 100)), "LeftButtonReleaseEvent")
    assert len(picked) == 0

    # 2. Click: press at (100, 100), release at (101, 101) -> distance sqrt(2) <= 3px
    probe._on_left_button_press(MockCaller((100, 100)), "LeftButtonPressEvent")
    probe._on_left_button_release(MockCaller((101, 101)), "LeftButtonReleaseEvent")
    assert len(picked) == 1
    assert picked[0] == (101, 101)

