"""
runner.py
Asynchronous execution runner for CalculiX (ccx) solver using PySide6 QProcess.
"""

import os
import shutil
import sys
from typing import Optional, List
from PySide6.QtCore import QObject, Signal, QProcess, QTimer, QProcessEnvironment


class CalculixRunner(QObject):
    """
    Asynchronous runner for CalculiX solver using QProcess.

    Signals:
        started(): Emitted when the solver process starts.
        progress(percent, message): Emitted with progress percentage (0-100) and status message.
        log_line(line): Emitted line-by-line for stdout/stderr logs.
        finished(success, frd_path, message): Emitted when the job finishes.
        error(message): Emitted when initialization or solver execution encounters an error.
    """

    started = Signal()
    progress = Signal(int, str)
    log_line = Signal(str)
    finished = Signal(bool, str, str)
    error = Signal(str)

    def __init__(self, ccx_path: Optional[str] = None, n_threads: int = 4, parent: Optional[QObject] = None):
        """
        Initialize CalculixRunner.

        Args:
            ccx_path: Optional explicit path to ccx binary.
            n_threads: Number of threads for multi-threading (OMP / CCX equation solver).
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.ccx_path: Optional[str] = ccx_path
        self.n_threads: int = max(1, int(n_threads))

        self._process: Optional[QProcess] = None
        self._timer: Optional[QTimer] = None

        self._job_name: str = ""
        self._job_dir: str = ""
        self._frd_path: str = ""
        self._sta_path: str = ""

        self._stdout_buffer: str = ""
        self._stderr_buffer: str = ""
        self._log_history: List[str] = []

        self._is_cancelled: bool = False
        self._is_finished_emitted: bool = False

    def resolve_ccx_path(self) -> Optional[str]:
        """
        Resolve ccx executable path following priority:
        1. Explicit self.ccx_path
        2. config.CCX_PATH or env var CCX_PATH
        3. shutil.which("ccx")
        4. shutil.which("ccx_static")
        5. ccx_2.* variants on PATH

        Returns:
            Resolved executable path or None if not found.
        """
        # 1. Explicit argument
        if self.ccx_path:
            expanded = os.path.expanduser(self.ccx_path)
            if os.path.isfile(expanded) and (os.access(expanded, os.X_OK) or expanded.endswith(".py")):
                return expanded
            found = shutil.which(self.ccx_path)
            if found:
                return found

        # 2. Config / Environment variable
        env_ccx = os.environ.get("CCX_PATH")
        if env_ccx:
            expanded = os.path.expanduser(env_ccx)
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return expanded
            found = shutil.which(env_ccx)
            if found:
                return found

        try:
            from dental_fea_app import config
            cfg_ccx = getattr(config, "CCX_PATH", None)
            if cfg_ccx:
                expanded = os.path.expanduser(cfg_ccx)
                if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                    return expanded
                found = shutil.which(cfg_ccx)
                if found:
                    return found
        except Exception:
            pass

        # 3. Standard binaries in PATH
        for candidate in ["ccx", "ccx_static"]:
            found = shutil.which(candidate)
            if found:
                return found

        # 4. Search for versioned binaries ccx_2.* in PATH
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for p in path_dirs:
            if not os.path.isdir(p):
                continue
            try:
                for entry in sorted(os.listdir(p), reverse=True):
                    if entry.startswith("ccx_2.") or entry.startswith("ccx-2."):
                        full_p = os.path.join(p, entry)
                        if os.path.isfile(full_p) and os.access(full_p, os.X_OK):
                            return full_p
            except (OSError, PermissionError):
                continue

        return None

    def run(self, inp_path: str) -> None:
        """
        Run CalculiX solver asynchronously on the specified .inp deck.

        Args:
            inp_path: Absolute or relative path to the CalculiX input file (.inp).
        """
        # Check if already running
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self.error.emit("이미 솔버 프로세스가 실행 중입니다.")
            return

        resolved_ccx = self.resolve_ccx_path()
        if not resolved_ccx:
            msg = (
                "CalculiX 실행 파일(ccx)을 찾을 수 없습니다.\n"
                "- CalculiX를 시스템에 설치해 주십시오 (예: sudo apt-get install calculix-ccx)\n"
                "- 또는 환경변수 CCX_PATH나 설정 파일(config.py)에 ccx 실행 경로를 지정해 주십시오."
            )
            self.error.emit(msg)
            return

        if not os.path.isfile(inp_path):
            self.error.emit(f"입력 파일(.inp)이 존재하지 않습니다: {inp_path}")
            return

        abs_inp = os.path.abspath(inp_path)
        self._job_dir = os.path.dirname(abs_inp)
        self._job_name = os.path.splitext(os.path.basename(abs_inp))[0]
        self._frd_path = os.path.join(self._job_dir, f"{self._job_name}.frd")
        self._sta_path = os.path.join(self._job_dir, f"{self._job_name}.sta")

        # Clean previous frd file if it exists to ensure fresh verification
        if os.path.exists(self._frd_path):
            try:
                os.remove(self._frd_path)
            except OSError:
                pass

        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._log_history = []
        self._is_cancelled = False
        self._is_finished_emitted = False

        # Configure QProcess
        self._process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("OMP_NUM_THREADS", str(self.n_threads))
        env.insert("CCX_NPROC_EQUATION_SOLVER", str(self.n_threads))
        self._process.setProcessEnvironment(env)
        self._process.setWorkingDirectory(self._job_dir)

        self._process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._process.readyReadStandardError.connect(self._on_stderr_ready)
        self._process.started.connect(self._on_process_started)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)

        # Setup polling timer
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll_progress)

        # Determine launch command & args
        if resolved_ccx.endswith(".py"):
            program = sys.executable
            args = [resolved_ccx, self._job_name]
        else:
            program = resolved_ccx
            args = [self._job_name]

        self._process.start(program, args)

    def cancel(self) -> None:
        """Cancel the running solver process."""
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            return

        self._is_cancelled = True
        if self._timer and self._timer.isActive():
            self._timer.stop()

        self._process.terminate()
        if not self._process.waitForFinished(1000):
            self._process.kill()
            self._process.waitForFinished(500)

        self._emit_finished_once(False, "", "사용자 취소")

    def _on_process_started(self) -> None:
        """Slot invoked when QProcess actually starts."""
        self.started.emit()
        self.progress.emit(0, "솔버 시작 및 모델 초기화 중...")
        if self._timer:
            self._timer.start()

    def _on_stdout_ready(self) -> None:
        """Slot invoked when new stdout data is available."""
        if not self._process:
            return
        raw = self._process.readAllStandardOutput().data()
        text = raw.decode("utf-8", errors="replace")
        self._stdout_buffer += text

        lines = self._stdout_buffer.split("\n")
        self._stdout_buffer = lines[-1]
        for line in lines[:-1]:
            line_str = line.rstrip("\r")
            self._log_history.append(line_str)
            self.log_line.emit(line_str)

    def _on_stderr_ready(self) -> None:
        """Slot invoked when new stderr data is available."""
        if not self._process:
            return
        raw = self._process.readAllStandardError().data()
        text = raw.decode("utf-8", errors="replace")
        self._stderr_buffer += text

        lines = self._stderr_buffer.split("\n")
        self._stderr_buffer = lines[-1]
        for line in lines[:-1]:
            line_str = line.rstrip("\r")
            self._log_history.append(line_str)
            self.log_line.emit(line_str)

    def _flush_buffers(self) -> None:
        """Flush remaining buffered text."""
        if self._stdout_buffer:
            line_str = self._stdout_buffer.rstrip("\r")
            self._log_history.append(line_str)
            self.log_line.emit(line_str)
            self._stdout_buffer = ""
        if self._stderr_buffer:
            line_str = self._stderr_buffer.rstrip("\r")
            self._log_history.append(line_str)
            self.log_line.emit(line_str)
            self._stderr_buffer = ""

    def _poll_progress(self) -> None:
        """Periodic polling of .sta file and logs to estimate progress."""
        if self._is_cancelled:
            return

        sta_exists = os.path.exists(self._sta_path)
        if sta_exists:
            try:
                with open(self._sta_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                if content:
                    lines = content.splitlines()
                    self.progress.emit(50, f"방정식 해 구하는 중 (진행 스텝: {len(lines)}행)...")
                    return
            except Exception:
                pass

        # Default indeterminate progress while running
        self.progress.emit(0, "해 구하는 중...")

    def _summarize_error(self) -> str:
        """
        Detect '*ERROR' in logs and extract the error line with previous up to 3 lines.
        If no '*ERROR' found, return the last few lines of log history.
        """
        err_idx = -1
        for i, line in enumerate(self._log_history):
            if "*ERROR" in line.upper():
                err_idx = i

        if err_idx != -1:
            start_idx = max(0, err_idx - 3)
            summary_lines = self._log_history[start_idx : err_idx + 1]
            return "\n".join(summary_lines)

        if self._log_history:
            return "\n".join(self._log_history[-4:])

        return "솔버 실행 중 오류가 발생했으나 로그가 비어 있습니다."

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        """Slot invoked when QProcess terminates."""
        if self._timer and self._timer.isActive():
            self._timer.stop()

        self._flush_buffers()

        if self._is_cancelled:
            self._emit_finished_once(False, "", "사용자 취소")
            return

        frd_exists = os.path.isfile(self._frd_path) and (os.path.getsize(self._frd_path) > 0)

        if exit_code == 0 and frd_exists:
            self.progress.emit(100, "해석 완료")
            self._emit_finished_once(True, self._frd_path, "해석이 성공적으로 완료되었습니다.")
        elif exit_code == 0 and not frd_exists:
            summary = "솔버가 종료 코드 0으로 끝났으나 결과 파일(.frd)이 생성되지 않았습니다."
            self._emit_finished_once(False, "", summary)
        else:
            err_summary = self._summarize_error()
            msg = f"CalculiX 실행이 실패했습니다 (종료 코드 {exit_code}):\n{err_summary}"
            self._emit_finished_once(False, "", msg)

    def _on_process_error(self, proc_error: QProcess.ProcessError) -> None:
        """Slot invoked when QProcess fails to start or crashes."""
        if self._is_cancelled:
            return
        if proc_error == QProcess.ProcessError.FailedToStart:
            msg = "프로세스를 시작할 수 없습니다. 실행 권한 또는 바이너리 경로를 확인하십시오."
            self.error.emit(msg)
            self._emit_finished_once(False, "", msg)

    def _emit_finished_once(self, success: bool, frd_path: str, message: str) -> None:
        """Safely emit finished signal at most once per run."""
        if not self._is_finished_emitted:
            self._is_finished_emitted = True
            self.finished.emit(success, frd_path, message)
