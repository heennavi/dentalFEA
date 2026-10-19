"""
test_runner.py
Unit and integration tests for CalculixRunner (asynchronous ccx solver runner).
"""

import os
import shutil
import sys
import tempfile
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from solver.runner import CalculixRunner
from solver.calculix_writer import write_inp
from fea.models import MeshPart, MeshModel, MaterialDef, MaterialAssignment, BoundaryConditionSet
from tests.test_calculix_writer import create_cube_mesh_c3d4


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QCoreApplication instance exists for Qt signals and event loops."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def wait_for_signal(signal, timeout_ms=5000):
    """
    Helper to synchronously wait for a Qt signal to emit within timeout_ms.

    Returns:
        tuple: The arguments emitted with the signal.

    Raises:
        TimeoutError: If the signal was not emitted within timeout_ms.
    """
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)

    received_args = []

    def on_signal(*args):
        received_args.append(args)
        loop.quit()

    signal.connect(on_signal)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    timer.stop()

    if not received_args:
        raise TimeoutError(f"Signal timed out after {timeout_ms} ms")
    return received_args[0]


class TestCalculixRunnerUnit:
    def test_ccx_not_found_emits_error(self, qapp, monkeypatch):
        """
        Verify that if ccx path is not provided and ccx is not on PATH,
        an error signal is emitted with Korean guidance text.
        """
        monkeypatch.setenv("PATH", "")
        monkeypatch.delenv("CCX_PATH", raising=False)

        runner = CalculixRunner(ccx_path=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            inp_path = os.path.join(tmpdir, "dummy.inp")
            with open(inp_path, "w") as f:
                f.write("*HEADING\nDummy\n")

            err_args = []
            runner.error.connect(lambda msg: err_args.append(msg))

            runner.run(inp_path)

            assert len(err_args) == 1
            err_msg = err_args[0]
            assert "CalculiX 실행 파일(ccx)을 찾을 수 없습니다" in err_msg
            assert "CCX_PATH" in err_msg

    def test_input_file_not_found_emits_error(self, qapp):
        """Verify that running with a non-existent .inp file emits error signal."""
        runner = CalculixRunner(ccx_path=sys.executable)
        err_args = []
        runner.error.connect(lambda msg: err_args.append(msg))

        runner.run("/path/does/not/exist/model.inp")

        assert len(err_args) == 1
        assert "입력 파일(.inp)이 존재하지 않습니다" in err_args[0]

    def test_fake_ccx_stdout_and_success(self, qapp):
        """
        Verify that fake ccx python script runs, emits logs, and generates .frd,
        resulting in finished(True, frd_path, ...).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_script = os.path.join(tmpdir, "fake_ccx.py")
            with open(fake_script, "w", encoding="utf-8") as f:
                f.write(
                    "import sys, os\n"
                    "job = sys.argv[1]\n"
                    "print('CalculiX Mock Version 2.21')\n"
                    "print('STEP 1 - ITERATION 1')\n"
                    "with open(f'{job}.frd', 'w') as f:\n"
                    "    f.write('Mock FRD Content')\n"
                    "sys.exit(0)\n"
                )

            inp_path = os.path.join(tmpdir, "job_test.inp")
            with open(inp_path, "w") as f:
                f.write("*HEADING\nJob Test\n")

            runner = CalculixRunner(ccx_path=fake_script)

            logs = []
            progress_reports = []
            runner.log_line.connect(lambda line: logs.append(line))
            runner.progress.connect(lambda p, m: progress_reports.append((p, m)))

            loop = QEventLoop()
            finished_args = []

            def on_finished(success, frd_path, msg):
                finished_args.append((success, frd_path, msg))
                loop.quit()

            runner.finished.connect(on_finished)

            runner.run(inp_path)
            loop.exec()

            assert len(finished_args) == 1
            success, frd_path, msg = finished_args[0]
            assert success is True
            assert frd_path == os.path.join(tmpdir, "job_test.frd")
            assert os.path.exists(frd_path)
            assert "성공적으로 완료" in msg

            # Check logs were collected
            assert any("CalculiX Mock Version 2.21" in l for l in logs)
            assert any("STEP 1 - ITERATION 1" in l for l in logs)

            # Check progress reached 100
            assert any(p == 100 for p, _ in progress_reports)

    def test_fake_ccx_error_with_star_error_summary(self, qapp):
        """
        Verify that when solver fails with *ERROR in stdout,
        finished(False, '', msg) is emitted and msg includes the *ERROR line and preceding lines.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_script = os.path.join(tmpdir, "fake_ccx_err.py")
            with open(fake_script, "w", encoding="utf-8") as f:
                f.write(
                    "import sys\n"
                    "print('Reading deck...')\n"
                    "print('Checking nodes...')\n"
                    "print('Checking elements...')\n"
                    "print('*ERROR in calinput: element 1 has zero volume')\n"
                    "print('Aborting run.')\n"
                    "sys.exit(1)\n"
                )

            inp_path = os.path.join(tmpdir, "err_job.inp")
            with open(inp_path, "w") as f:
                f.write("*HEADING\nError Job\n")

            runner = CalculixRunner(ccx_path=fake_script)

            loop = QEventLoop()
            finished_args = []

            def on_finished(success, frd_path, msg):
                finished_args.append((success, frd_path, msg))
                loop.quit()

            runner.finished.connect(on_finished)

            runner.run(inp_path)
            loop.exec()

            assert len(finished_args) == 1
            success, frd_path, msg = finished_args[0]
            assert success is False
            assert frd_path == ""
            # The summary must contain *ERROR line and its preceding 3 lines
            assert "*ERROR in calinput: element 1 has zero volume" in msg
            assert "Checking elements..." in msg
            assert "Checking nodes..." in msg
            assert "Reading deck..." in msg

    def test_fake_ccx_cancel(self, qapp):
        """
        Verify that cancel() terminates the running solver and emits finished(False, '', '사용자 취소').
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_script = os.path.join(tmpdir, "fake_ccx_sleep.py")
            with open(fake_script, "w", encoding="utf-8") as f:
                f.write(
                    "import sys, time\n"
                    "print('Sleep started', flush=True)\n"
                    "while True:\n"
                    "    time.sleep(0.1)\n"
                )

            inp_path = os.path.join(tmpdir, "sleep_job.inp")
            with open(inp_path, "w") as f:
                f.write("*HEADING\nSleep Job\n")

            runner = CalculixRunner(ccx_path=fake_script)

            finished_args = []
            loop = QEventLoop()

            def on_finished(success, frd_path, msg):
                finished_args.append((success, frd_path, msg))
                loop.quit()

            runner.finished.connect(on_finished)

            # Once started, trigger cancel
            runner.started.connect(lambda: QTimer.singleShot(100, runner.cancel))

            runner.run(inp_path)
            loop.exec()

            assert len(finished_args) == 1
            success, frd_path, msg = finished_args[0]
            assert success is False
            assert frd_path == ""
            assert msg == "사용자 취소"


class TestCalculixRunnerIntegration:
    def test_real_ccx_cube_compression(self, qapp):
        """
        Integration test: Solve a 10x10x10 mm cube under compression using actual CalculiX runner.
        Skipped if ccx is not installed on the system.
        """
        ccx_path = shutil.which("ccx")
        if not ccx_path:
            pytest.skip("CalculiX 'ccx' binary is not installed on PATH; skipping real runner test.")

        nodes, tets = create_cube_mesh_c3d4()
        part = MeshPart(name="cube", nodes=nodes, elements=tets, element_type="C3D4")
        model = MeshModel(parts=[part])

        bot_mask = np.isclose(nodes[:, 2], 0.0)
        bot_nodes = np.where(bot_mask)[0].astype(np.int64)

        top_mask = np.isclose(nodes[:, 2], 10.0)
        top_nodes = np.where(top_mask)[0].astype(np.int64)

        mat = MaterialDef(name="CUBE_MAT", E=1000.0, nu=0.3)
        materials = {
            "cube": MaterialAssignment(
                element_material={"cube": np.zeros(len(tets), dtype=np.int32)},
                materials=[mat]
            )
        }

        bc = BoundaryConditionSet(
            fixed_node_sets=[{
                "name": "FIX_BOTTOM",
                "part": "cube",
                "node_ids": bot_nodes,
                "dofs": (1, 3)
            }],
            loads=[{
                "name": "COMPRESS_TOP",
                "part": "cube",
                "node_ids": top_nodes,
                "force_N": (0.0, 0.0, -100.0)
            }]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            job_name = "cube_compression_runner"
            inp_path = os.path.join(tmpdir, f"{job_name}.inp")
            write_inp(
                path=inp_path,
                model=model,
                materials=materials,
                bc=bc,
                ties=[],
                title="Cube Compression Runner Test"
            )

            runner = CalculixRunner(ccx_path=ccx_path, n_threads=2)

            started_called = []
            progress_events = []
            log_lines = []
            finished_args = []

            loop = QEventLoop()

            runner.started.connect(lambda: started_called.append(True))
            runner.progress.connect(lambda p, m: progress_events.append((p, m)))
            runner.log_line.connect(lambda line: log_lines.append(line))

            def on_finished(success, frd_path, msg):
                finished_args.append((success, frd_path, msg))
                loop.quit()

            runner.finished.connect(on_finished)

            runner.run(inp_path)
            loop.exec()

            assert len(started_called) == 1
            assert len(finished_args) == 1

            success, frd_path, msg = finished_args[0]
            assert success is True, f"CalculixRunner failed with message:\n{msg}\nLogs:\n" + "\n".join(log_lines)
            assert os.path.isfile(frd_path)
            assert os.path.getsize(frd_path) > 0
            assert "성공적으로 완료" in msg
            assert len(log_lines) > 0
