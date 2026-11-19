"""DS-1000 execution reward for TRL GRPO.

Each completion is run against the problem's own test (`code_context`), exactly like the
official DS-1000 evaluator, in a separate Python process with a timeout. Reward is 1.0 when
the test passes and 0.0 otherwise.

WARNING: this executes model-generated code. Run it only on a throwaway benchmark instance.

Per-completion results (pass/fail, error type, execution time, completion length) are
handed to `on_result` so they can be stored in rewards.csv.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", re.S)


def extract_solution(text: str) -> str:
    """Pull the code out of a completion (fenced block first, else cut at DS-1000 markers)."""
    m = _FENCE.search(text)
    code = m.group(1) if m else text
    for marker in ("</code>", "END SOLUTION"):
        code = code.split(marker)[0]
    return code.replace("<code>", "").strip("\n")


def build_program(code_context: str, solution: str) -> str:
    prog = code_context + "\n" + f"code = {solution!r}\n" + "test_execution(code)\n"
    if "test_string(" in code_context:
        prog += "test_string(code)\n"
    return prog


def run_one(code_context, solution, timeout_s, python_exe):
    prog = build_program(code_context, solution)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(prog)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("NEURON", "XLA", "PJRT", "CUDA_VISIBLE"))}
        env["CUDA_VISIBLE_DEVICES"] = ""          # keep test code off the accelerator
        env["MPLBACKEND"] = "Agg"
        t0 = time.perf_counter()
        try:
            r = subprocess.run([python_exe, path], cwd=d, env=env, capture_output=True,
                               text=True, timeout=timeout_s)
            dt = time.perf_counter() - t0
            if r.returncode == 0:
                return 1.0, "pass", dt
            last = (r.stderr.strip().splitlines() or ["unknown"])[-1]
            err = last.split(":")[0][:60] if ":" in last else last[:60]
            return 0.0, err, dt
        except subprocess.TimeoutExpired:
            return 0.0, "Timeout", time.perf_counter() - t0


def _text(completion):
    if isinstance(completion, list):          # conversational format
        return completion[-1]["content"] if completion else ""
    return completion


class DS1000Reward:
    """Callable reward function with the signature TRL expects."""

    def __init__(self, timeout_s=30, workers=8, python_exe=None, on_result=None, tokenizer=None,
                 timer=None):
        self.__name__ = "ds1000_exec"            # TRL uses this as the metric name
        self.timeout_s = timeout_s
        self.workers = workers
        self.python_exe = python_exe or os.environ.get("REWARD_PYTHON") or sys.executable
        self.on_result = on_result               # callback(dict) per completion
        self.tokenizer = tokenizer
        self.step = 0                            # set by the trainer callback
        self.timer = timer                       # bench.phase_timer.PhaseTimer -> 'reward' phase

    def __call__(self, prompts, completions, code_context, problem_id, library, **kwargs):
        with self.timer.phase("reward") if self.timer else nullcontext():
            return self._score(completions, code_context, problem_id, library)

    def _score(self, completions, code_context, problem_id, library):
        texts = [_text(c) for c in completions]
        sols = [extract_solution(t) for t in texts]
        with ThreadPoolExecutor(self.workers) as ex:
            results = list(ex.map(lambda a: run_one(a[0], a[1], self.timeout_s, self.python_exe),
                                  zip(code_context, sols)))
        if self.on_result:
            for i, (t, (rew, status, dt)) in enumerate(zip(texts, results)):
                n_tok = len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) if self.tokenizer else ""
                self.on_result({
                    "step": self.step,
                    "sample_idx": i,
                    "problem_id": problem_id[i],
                    "library": library[i],
                    "completion_chars": len(t),
                    "completion_tokens": n_tok,
                    "reward": rew,
                    "passed": int(rew > 0),
                    "status": status,
                    "exec_time_s": round(dt, 4),
                })
        return [r[0] for r in results]
