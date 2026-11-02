"""DS-1000 pass/fail reward for TitanRL.

Scores a completion by extracting the code block and running it in a fresh
subprocess: 1.0 on clean exit, 0.0 on error, -0.5 on empty. This is a simplified
proxy (it rewards "the generated program runs"), not the full hidden-test
DS-1000 harness.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass

from torchtitan.rl.examples.ds1000.data import Ds1000Sample
from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rollout.rubric import RewardFn


def _extract_code(solution_str: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", solution_str, re.DOTALL)
    if m:
        return m.group(1).strip()
    s = solution_str.strip()
    s = re.sub(r"^```(?:python)?\s*", "", s)
    return s


def _check_correctness(program: str, timeout: int = 30) -> bool:
    if not program.strip():
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/program.py"
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                ["python", path], capture_output=True, timeout=timeout
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False


class RewardDs1000(RewardFn):
    """1.0 if the completion runs cleanly, 0.0 on error, -0.5 if empty."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        timeout: int = 30

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.timeout = config.timeout

    async def __call__(self, rollout: Rollout, env_input: Ds1000Sample) -> float:
        del env_input
        if not rollout.turns:
            return -0.5
        last_turn = rollout.turns[-1]
        message = last_turn.completion_message
        response_text = (message.get("content") or "") if message else ""
        code = _extract_code(response_text)
        if not code:
            return -0.5
        return 1.0 if _check_correctness(code, self.timeout) else 0.0
