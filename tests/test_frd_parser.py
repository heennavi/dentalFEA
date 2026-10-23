"""
test_frd_parser.py
Unit tests and performance benchmarks for CalculiX .frd result parser.
"""

import os
import time
import logging
import pytest
import numpy as np

from postprocess.frd_parser import parse_frd, parse_frd_text
from fea.results import FrdData, FrdBlock


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SYNTHETIC_TET4_PATH = os.path.join(FIXTURES_DIR, "synthetic_tet4.frd")
CUBE_COMPRESSION_PATH = os.path.join(FIXTURES_DIR, "cube_compression.frd")


def test_parse_synthetic_tet4():
    """Verify parsing accuracy on synthetic 4-node tetrahedron fixture."""
    assert os.path.isfile(SYNTHETIC_TET4_PATH), f"Fixture not found: {SYNTHETIC_TET4_PATH}"
    data = parse_frd(SYNTHETIC_TET4_PATH)

    # 1. Verify Nodes
    assert len(data.node_ids) == 4
    np.testing.assert_array_equal(data.node_ids, [1, 2, 3, 4])

    expected_coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    np.testing.assert_allclose(data.node_coords, expected_coords, atol=1e-5)

    # 2. Verify Result Blocks count
    assert len(data.blocks) == 2

    # 3. Verify DISP block (with negative values adhered without whitespace)
    disp = data.get_block("DISP", "last")
    assert disp is not None
    assert disp.name == "DISP"
    assert disp.time == 0.5
    assert disp.step == 0
    # Component count resolution: ALL should be omitted, yielding D1, D2, D3
    assert disp.component_names == ["D1", "D2", "D3"]
    assert disp.values.shape == (4, 3)

    expected_disp = np.array([
        [0.00000, 0.00000, 0.00000],
        [-0.012345, 0.023456, -0.034567],
        [1.00000, -2.00000, 3.00000],
        [-0.55555, -0.66666, -0.77777],
    ])
    np.testing.assert_allclose(disp.values, expected_disp, atol=1e-5)

    # 4. Verify STRESS block (6 components)
    stress = data.get_block("STRESS", "last")
    assert stress is not None
    assert stress.name == "STRESS"
    assert stress.time == 0.5
    assert stress.step == 1
    assert stress.component_names == ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"]
    assert stress.values.shape == (4, 6)

    expected_stress = np.array([
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0],
        [123.456, -234.567, 345.678, -456.789, 567.89, -678.901],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(stress.values, expected_stress, atol=1e-3)


def test_multi_increment_last_block():
    """Verify that get_block('DISP', 'last') returns the final increment block."""
    text = """    1C
    2C                             2                                     1
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 1.00000E+00 0.00000E+00 0.00000E+00
 -3
    1PSTEP                         1           1           1
  100CL  101 0.500000000           2                     0    1           1
 -4  DISP        3    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -1         1 1.00000E+00 1.00000E+00 1.00000E+00
 -1         2 1.00000E+00 1.00000E+00 1.00000E+00
 -3
    1PSTEP                         2           1           1
  100CL  101 1.000000000           2                     0    1           1
 -4  DISP        3    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -1         1 2.00000E+00 2.00000E+00 2.00000E+00
 -1         2 2.00000E+00 2.00000E+00 2.00000E+00
 -3
"""
    data = parse_frd_text(text)
    assert len(data.blocks) == 2

    # First increment
    b_first = data.get_block("DISP", "first")
    assert b_first is not None
    assert b_first.time == 0.5
    assert b_first.step == 0
    np.testing.assert_allclose(b_first.values, 1.0)

    # Last increment
    b_last = data.get_block("DISP", "last")
    assert b_last is not None
    assert b_last.time == 1.0
    assert b_last.step == 1
    np.testing.assert_allclose(b_last.values, 2.0)


def test_parse_actual_ccx_sample():
    """Verify parser on actual CalculiX-generated cube_compression.frd."""
    assert os.path.isfile(CUBE_COMPRESSION_PATH), f"Fixture not found: {CUBE_COMPRESSION_PATH}"
    data = parse_frd(CUBE_COMPRESSION_PATH)

    # 8 nodes in cube
    assert len(data.node_ids) == 8
    np.testing.assert_array_equal(data.node_ids, np.arange(1, 9))
    assert data.node_coords.shape == (8, 3)

    # Result blocks: DISP, STRESS, TOSTRAIN, FORC, ERROR
    block_names = [b.name for b in data.blocks]
    assert "DISP" in block_names
    assert "STRESS" in block_names
    assert "TOSTRAIN" in block_names
    assert "FORC" in block_names

    # Check DISP (D1, D2, D3)
    disp = data.get_block("DISP", "last")
    assert disp is not None
    assert disp.component_names == ["D1", "D2", "D3"]
    assert disp.values.shape == (8, 3)

    # Check STRESS (SXX..SZX, 6 components)
    stress = data.get_block("STRESS", "last")
    assert stress is not None
    assert stress.component_names == ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"]
    assert stress.values.shape == (8, 6)

    # Check FORC (F1, F2, F3)
    forc = data.get_block("FORC", "last")
    assert forc is not None
    assert forc.component_names == ["F1", "F2", "F3"]
    assert forc.values.shape == (8, 3)


def test_error_handling(tmp_path, caplog):
    """Verify FileNotFoundError, ValueError, and empty-block warning handling."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError, match="찾을 수 없습니다"):
        parse_frd(str(tmp_path / "non_existent.frd"))

    # 2. Empty file
    empty_file = tmp_path / "empty.frd"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="비어 있습니다"):
        parse_frd(str(empty_file))

    # 3. Empty text string
    with pytest.raises(ValueError, match="비어 있습니다"):
        parse_frd_text("   \n\t  ")

    # 4. Solved mesh without result blocks (e.g. solver failed before writing results)
    no_results_text = """    1C
    2C                             2                                     1
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 1.00000E+00 0.00000E+00 0.00000E+00
 -3
"""
    with caplog.at_level(logging.WARNING):
        res = parse_frd_text(no_results_text)
    assert len(res.node_ids) == 2
    assert len(res.blocks) == 0
    assert "결과 블록이 존재하지 않습니다" in caplog.text


def test_nan_inf_and_irregular_floats():
    """Verify that inf, nan, and 3-digit exponent values are parsed safely."""
    text = """    1C
    2C                             4                                     1
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 1.00000E+00 0.00000E+00 0.00000E+00
 -1         3 2.00000E+00 0.00000E+00 0.00000E+00
 -1         4 3.00000E+00 0.00000E+00 0.00000E+00
 -3
    1PSTEP                         1           1           1
  100CL  101 1.000000000           4                     0    1           1
 -4  DISP        3    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -1         11.00000E+100         nan         inf
 -1         2        -inf 1.23456-100 2.34567D+01
 -1         3 0.00000E+00 0.00000E+00 0.00000E+00
 -1         4 1.00000E+00 2.00000E+00 3.00000E+00
 -3
"""
    data = parse_frd_text(text)
    disp = data.get_block("DISP", "last")
    assert disp is not None

    # Node 1: E+100 (overflow), nan and inf all become NaN
    assert np.isnan(disp.values[0, 0])
    assert np.isnan(disp.values[0, 1])
    assert np.isnan(disp.values[0, 2])

    # Node 2: -inf becomes NaN, 1.23456-100 parsed, 2.34567D+01 parsed
    assert np.isnan(disp.values[1, 0])
    assert np.isclose(disp.values[1, 1], 1.23456e-100)
    assert np.isclose(disp.values[1, 2], 23.4567)


def test_benchmark_1m_nodes(tmp_path):
    """
    Benchmark parser speed on 1,000,000 synthetic result records.

    Target: < 30 seconds (Reporting actual measurement).
    """
    num_nodes = 1_000_000
    benchmark_file = tmp_path / "synthetic_1m.frd"

    # Write synthetic 1M dataset with realistic mixed positive/negative values
    lines = [
        "    1C\n",
        f"    2C                    {num_nodes:10d}                                     1\n",
    ]
    # Simple node block (small for generation speed)
    lines.extend([
        " -1         1 0.00000E+00 0.00000E+00 0.00000E+00\n",
        " -3\n",
        "    1PSTEP                         1           1           1\n",
        f"  100CL  101 1.000000000  {num_nodes:10d}                     0    1           1\n",
        " -4  DISP        3    1\n",
        " -5  D1          1    2    1    0\n",
        " -5  D2          1    2    2    0\n",
        " -5  D3          1    2    3    0\n",
    ])

    template = " -1{:10d} 1.23456E-02-2.34567E-02 3.45678E-02\n"
    data_lines = [template.format(i) for i in range(1, num_nodes + 1)]
    lines.extend(data_lines)
    lines.append(" -3\n")

    with open(benchmark_file, "w", encoding="ascii") as f:
        f.writelines(lines)

    # Measure parse time
    t0 = time.perf_counter()
    frd_data = parse_frd(str(benchmark_file))
    elapsed = time.perf_counter() - t0

    disp = frd_data.get_block("DISP", "last")
    assert disp is not None
    assert len(disp.node_ids) == num_nodes
    assert disp.values.shape == (num_nodes, 3)

    print(f"\n==================================================")
    print(f"[BENCHMARK] 1,000,000 nodes parsed in {elapsed:.3f} s")
    print(f"==================================================")

    # Strict performance threshold: must be well below 30 seconds
    assert elapsed < 30.0, f"Parse time too slow: {elapsed:.2f} s"


def test_parse_large_clinical_sample():
    """Verify parsing and measure performance on real 128MB clinical simulation result if available."""
    clinical_frd_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved", "dental_fea_job.frd")
    if not os.path.isfile(clinical_frd_path):
        pytest.skip(f"Clinical sample not present at {clinical_frd_path}")

    t0 = time.perf_counter()
    frd_data = parse_frd(clinical_frd_path)
    elapsed = time.perf_counter() - t0

    assert len(frd_data.node_ids) > 0
    assert len(frd_data.blocks) > 0

    disp = frd_data.get_block("DISP", "last")
    assert disp is not None
    assert disp.component_names == ["D1", "D2", "D3"]

    stress = frd_data.get_block("STRESS", "last")
    assert stress is not None
    assert stress.component_names == ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"]

    print(f"\n==================================================")
    print(f"[CLINICAL SAMPLE] {len(frd_data.node_ids):,} nodes, {len(frd_data.blocks)} blocks parsed in {elapsed:.3f} s")
    print(f"==================================================")

