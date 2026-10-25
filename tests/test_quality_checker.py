"""
test_quality_checker.py
Unit tests for tetrahedral mesh quality metrics and validation.
"""

import pytest
import numpy as np

from meshing.quality_checker import (
    tet_quality,
    quality_report_text,
    assert_mesh_ok,
)


class TestQualityChecker:
    """Test suite for mesh quality checker."""

    @pytest.fixture
    def regular_tet(self):
        """Standard regular tetrahedron."""
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2.0, 0.0],
            [0.5, np.sqrt(3) / 6.0, np.sqrt(6) / 3.0],
        ], dtype=np.float64)
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return nodes, elements

    def test_ideal_regular_tetrahedron(self, regular_tet):
        """Verify quality of ideal equilateral tetrahedron."""
        nodes, elements = regular_tet
        rep = tet_quality(nodes, elements)

        assert rep["n_elements"] == 1
        assert rep["n_negative"] == 0
        assert rep["min_volume"] > 0.0
        # Aspect ratio for regular tet is exactly 1.0
        assert np.isclose(rep["aspect_max"], 1.0, atol=1e-3)
        assert np.isclose(rep["aspect_mean"], 1.0, atol=1e-3)
        # Scaled Jacobian for regular tet is 1.0
        assert np.isclose(rep["min_scaled_jacobian"], 1.0, atol=1e-2)
        assert len(rep["bad_element_ids"]) == 0

        # assert_mesh_ok should pass
        assert_mesh_ok(rep)

    def test_inverted_element_detected(self, regular_tet):
        """Verify that an artificially inverted element is detected (n_negative == 1)."""
        nodes, elements = regular_tet
        # Create a second element with swapped vertices 1 and 2 (negative volume)
        inverted_elem = np.array([[0, 2, 1, 3]], dtype=np.int64)
        two_elems = np.vstack([elements, inverted_elem])

        rep = tet_quality(nodes, two_elems)

        assert rep["n_elements"] == 2
        assert rep["n_negative"] == 1
        assert 1 in rep["bad_element_ids"]
        assert rep["min_volume"] < 0.0

        # assert_mesh_ok should raise ValueError
        with pytest.raises(ValueError, match="음의 부피"):
            assert_mesh_ok(rep, max_negative=0)

    def test_quality_report_text(self, regular_tet):
        """Verify Korean summary report text formatting."""
        nodes, elements = regular_tet
        rep = tet_quality(nodes, elements)
        text = quality_report_text(rep)

        assert "3D 사면체 메쉬 품질 검사 보고서" in text
        assert "총 요소 수: 1 개" in text
        assert "정상" in text
        assert "형상비(Aspect Ratio)" in text

    def test_quality_report_text_with_mesh_model(self, regular_tet):
        """Verify quality_report_text and assert_mesh_ok accept MeshModel instance."""
        from fea.models import MeshPart, MeshModel

        nodes, elements = regular_tet
        part1 = MeshPart(name="bone", nodes=nodes, elements=elements, element_type="C3D4")
        part2 = MeshPart(name="implant", nodes=nodes, elements=elements, element_type="C3D4")
        model = MeshModel(parts=[part1, part2])

        text = quality_report_text(model)
        assert "3D 사면체 어셈블리 품질 검사 보고서" in text
        assert "[bone (C3D4)]" in text
        assert "[implant (C3D4)]" in text
        assert "어셈블리 총 요소 수: 2 개" in text
        assert "양호" in text

        # assert_mesh_ok should pass with healthy MeshModel
        assert_mesh_ok(model, max_negative=0)

    def test_filter_degenerate_elements_and_clean_model(self, regular_tet):
        """Verify degenerate and zero-volume elements are eliminated and nodes are reindexed."""
        from meshing.gmsh_mesher import filter_degenerate_elements
        from meshing.quality_checker import clean_mesh_model
        from fea.models import MeshPart, MeshModel

        nodes, elements = regular_tet
        # Add extra unused orphan node
        nodes_with_orphan = np.vstack([nodes, [[10.0, 10.0, 10.0]]])
        # Element 0: valid regular tet
        # Element 1: degenerate with duplicate corner node [0, 0, 1, 2]
        # Element 2: degenerate zero volume coplanar tet [0, 1, 2, 1]
        elem_dup = np.array([0, 0, 1, 2], dtype=np.int64)
        coplanar_elem = np.array([0, 1, 2, 0], dtype=np.int64)
        bad_elements = np.vstack([elements, elem_dup, coplanar_elem])

        cleaned_nodes, cleaned_elems = filter_degenerate_elements(nodes_with_orphan, bad_elements)
        assert len(cleaned_elems) == 1
        assert len(cleaned_nodes) == 4
        # Orphan node at index 4 should be stripped away
        assert np.all(cleaned_elems[0] == [0, 1, 2, 3])

        # Test clean_mesh_model
        part = MeshPart(name="screw", nodes=nodes_with_orphan, elements=bad_elements, element_type="C3D4")
        model = MeshModel(parts=[part])
        cleaned_model = clean_mesh_model(model)
        assert len(cleaned_model.get_part("screw").elements) == 1
        assert len(cleaned_model.get_part("screw").nodes) == 4
