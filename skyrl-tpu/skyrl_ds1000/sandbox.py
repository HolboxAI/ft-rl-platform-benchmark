"""Run one DS-1000 test program in a separate, limited process.

The program is built exactly as DS-1000's ``test_ds1000.py`` builds it::

    code_context + "code = <repr(solution)>" + "test_execution(code)" [+ "test_string(code)"]

and it passes when it runs to the end without an exception. Differences from DS-1000's ``execution.py``,
which runs the program in a forked child of the evaluator:

* the program runs in another interpreter (``DS1000_PYTHON``, the venv from ``setup_exec_venv.sh``), so the
  library versions are DS-1000's and not the trainer's, and torch_tpu / libtpu are not importable at all;
* the TPU, GPU and JAX devices are hidden through the environment as a second guard;
* a pass also needs a random marker that the program prints after the tests. Without it, a solution that
  calls ``os._exit(0)`` before the asserts would pass, and RL finds such shortcuts;
* the process gets its own session (killed as a group on timeout), a temporary working directory, CPU-time
  and file-size limits, and, on Linux where ``unshare -rn`` works, no network;
* 15 problems load sample data (seaborn ``load_dataset``: penguins, tips, exercise, planets; sklearn
  ``fetch_california_housing``). Without a network these come from a read-only cache, ``DS1000_DATA_HOME``,
  that ``setup_exec_venv.sh`` fills.
"""
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional

DEFAULT_TIMEOUT_S = 120.0  # as DS-1000's test_ds1000.py
OUTPUT_LIMIT_BYTES = 16 << 20

_NET_ISOLATION: Optional[bool] = None


def exec_python() -> str:
    return os.environ.get("DS1000_PYTHON", sys.executable)


def data_home() -> str:
    return os.environ.get("DS1000_DATA_HOME", os.path.expanduser("~/.cache/ds1000-data"))


def build_program(code_context: str, solution: str) -> str:
    program = code_context + "\n" + f"code = {solution!r}\n" + "test_execution(code)\n"
    if "test_string(" in code_context:
        program += "test_string(code)\n"
    return program


def _net_isolation_available() -> bool:
    global _NET_ISOLATION
    if _NET_ISOLATION is None:
        _NET_ISOLATION = False
        if sys.platform.startswith("linux") and shutil.which("unshare"):
            try:
                r = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10)
                _NET_ISOLATION = r.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                pass
    return _NET_ISOLATION


def _child_env(workdir: str) -> Dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
        "MPLBACKEND": "Agg",
        "SEABORN_DATA": os.path.join(data_home(), "seaborn"),
        "SCIKIT_LEARN_DATA": os.path.join(data_home(), "sklearn"),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "CUDA_VISIBLE_DEVICES": "-1",
        "TPU_VISIBLE_DEVICES": "",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "JAX_PLATFORMS": "cpu",
        "PJRT_DEVICE": "CPU",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def _limit(cpu_s: int):
    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT_BYTES, OUTPUT_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _tail(path: str, n: int = 2000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def run_test(code_context: str, solution: str, timeout: float = DEFAULT_TIMEOUT_S) -> Dict[str, object]:
    """Return ``{"passed": bool, "status": "passed"|"failed"|"timeout"|"no_marker", "error": str}``."""
    marker = "DS1000_DONE_" + secrets.token_hex(16)
    program = build_program(code_context, solution) + f"print({marker!r}, flush=True)\n"
    workdir = tempfile.mkdtemp(prefix="ds1000_")
    try:
        prog = os.path.join(workdir, "prog.py")
        with open(prog, "w") as f:
            f.write(program)
        cmd = [exec_python(), "-I", prog]
        if _net_isolation_available():
            cmd = ["unshare", "-rn"] + cmd
        out_path, err_path = os.path.join(workdir, "stdout"), os.path.join(workdir, "stderr")
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                env=_child_env(workdir),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
                preexec_fn=_limit(int(timeout) + 5),
            )
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
                proc.wait()
                return {"passed": False, "status": "timeout", "error": f"timeout after {timeout:.0f}s"}
        if rc != 0:
            return {"passed": False, "status": "failed", "error": _tail(err_path)}
        if marker not in _tail(out_path, 4096):
            return {"passed": False, "status": "no_marker", "error": "exited 0 before the tests finished"}
        return {"passed": True, "status": "passed", "error": ""}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def net_isolated() -> bool:
    return _net_isolation_available()
