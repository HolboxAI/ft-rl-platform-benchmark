"""Single-turn DS-1000 environment for SkyRL.

The model gets one DS-1000 problem (Stack Overflow question + setup code, ending at ``BEGIN SOLUTION``) and
replies with the missing code. ``step()`` takes the code out of the reply, runs DS-1000's own tests on it in a
separate process (``sandbox.py``) and returns:

    reward = 1.0 if the tests pass else DS1000_FORMAT_REWARD (default 0.0) if the reply holds a code block else 0.0

With the default, the reward is DS-1000's pass/fail score. ``metadata["passed"]`` always holds the pass/fail
result, so the metric stays comparable when a format reward is set.
"""
import json
import os
import re
from typing import Any, Dict, Optional

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

try:
    from sandbox import DEFAULT_TIMEOUT_S, run_test
except ImportError:  # imported as a package module (tests)
    from .sandbox import DEFAULT_TIMEOUT_S, run_test

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)
_CODE_TAG_RE = re.compile(r"<code>\s*\n?(.*?)(?:</code>|\Z)", re.DOTALL)
# A line that starts with END SOLUTION ends the answer, as in DS-1000's postprocess ("\nEND SOLUTION"). The
# "### END SOLUTION" comments inside function-style answers do not.
_END_RE = re.compile(r"^[ \t]*END SOLUTION", re.MULTILINE)


def extract_code(text: Optional[str]) -> Optional[str]:
    """Return the solution code in a reply, or None when the reply has no code.

    Order: the first fenced block, else the first ``<code>`` block, else the reply itself when it ends at
    ``</code>`` or ``END SOLUTION`` (the completion style of DS-1000's prompts). DS-1000's own ``postprocess``
    keeps any prose before a fenced block, which then fails as a syntax error; chat models write such prose.
    """
    if text is None:
        return None
    t = _SPECIAL_RE.sub("", _THINK_RE.sub("", text))
    m = _FENCE_RE.search(t)
    if m:
        code = m.group(1)
    else:
        m = _CODE_TAG_RE.search(t)
        if m:
            code = m.group(1)
        elif "</code>" in t or _END_RE.search(t):
            code = t.split("</code>")[0]
        else:
            return None
    m = _END_RE.search(code)
    if m:
        code = code[: m.start()]
    return code if code.strip() else None


def _format_reward() -> float:
    try:
        return float(os.environ.get("DS1000_FORMAT_REWARD", "0"))
    except ValueError:
        return 0.0


class DS1000Env(BaseTextEnv):
    """Single-turn DS-1000: problem in, solution code out, DS-1000's tests as the reward."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 1
        reward_spec = extras.get("reward_spec", {}) or {}
        gt = reward_spec.get("ground_truth", {})
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except (json.JSONDecodeError, TypeError):
                gt = {}
        self.ground_truth = gt or {}
        self.timeout = float(os.environ.get("DS1000_TIMEOUT_S", DEFAULT_TIMEOUT_S))

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        code = extract_code(action)
        meta = {
            "problem_id": self.ground_truth.get("problem_id"),
            "library": self.ground_truth.get("library"),
            "has_code": code is not None,
        }
        if code is None or not self.ground_truth.get("code_context"):
            result = {"passed": False, "status": "no_code" if code is None else "no_ground_truth", "error": ""}
        else:
            result = run_test(self.ground_truth["code_context"], code, timeout=self.timeout)
        meta.update(passed=result["passed"], status=result["status"])
        if result["passed"]:
            reward = 1.0
        elif code is not None:
            reward = _format_reward()
        else:
            reward = 0.0
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=meta)
