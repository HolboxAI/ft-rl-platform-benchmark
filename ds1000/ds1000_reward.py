"""DS-1000 RLVR reward for verl.

Scores a generated program pass/fail by running it in a fresh subprocess and
checking the exit code (the DS-1000 evaluator treats "no exception" as passed).
Simplified proxy for the full hidden-test harness: reward 1.0 on clean exit,
0.0 on exception, -0.5 on empty output.

WARNING: this is a timeout + process guard, NOT a security sandbox. Run on an
isolated instance, never a shared machine.
"""

import re
import subprocess
import tempfile


def _extract_code(solution_str: str) -> str:
    """Pull the code block out of a completion, falling back to the raw text."""
    m = re.search(r"```(?:python)?\s*(.*?)```", solution_str, re.DOTALL)
    if m:
        return m.group(1).strip()
    # strip leading markdown fence if the model only opened one
    s = solution_str.strip()
    s = re.sub(r"^```(?:python)?\s*", "", s)
    return s


def check_correctness(program: str, timeout: int = 30) -> dict:
    """Run `program` in a fresh subprocess; return {'passed': bool, 'result': str}."""
    if not program.strip():
        return {"passed": False, "result": "empty"}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/program.py"
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                ["python", path], capture_output=True, timeout=timeout, text=True
            )
            return {"passed": proc.returncode == 0, "result": proc.stderr[:2000]}
        except subprocess.TimeoutExpired:
            return {"passed": False, "result": "timeout"}


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl reward entrypoint. Returns a single float score."""
    code = _extract_code(solution_str)
    if not code:
        return -0.5
    res = check_correctness(code)
    return 1.0 if res["passed"] else 0.0
