"""CSV storage for every number a benchmark run produces.

One writer per file. Rows are flushed immediately, so a crashed run still leaves usable data.
All files live in one run directory: results/<platform>_<UTC timestamp>/

  run_info.csv       key,value     platform, instance, versions, full config
  phase_events.csv   one row per timed call of a phase (finest grain, for timelines)
  step_metrics.csv   one row per optimizer step: phase totals, throughput, reward, loss,
                     memory, compiler counters
  rewards.csv        one row per completion: pass/fail, error type, exec time, length
  trl_logs.csv       everything TRL logs (loss, kl, reward std, clip ratio, ...) per step
  device_util.csv    accelerator utilisation / memory / power samples (metrics/device_monitor.py)
  kernel_microbench.csv  op-level timings (kernels/kernel_microbench.py)

Derived files (metrics/summarize_run.py, metrics/compare_runs.py) are built from these.
"""

import csv
import json
import os


class CsvWriter:
    """Append rows to a CSV. Columns are fixed by the first row unless `fields` is given;
    unknown keys in later rows are dropped, missing keys are left empty."""

    def __init__(self, path, fields=None):
        self.path = path
        self.fields = list(fields) if fields else None
        self._f = None
        self._w = None

    def write(self, row):
        if self._w is None:
            if self.fields is None:
                self.fields = list(row.keys())
            new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
            self._f = open(self.path, "a", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                self._w.writeheader()
        self._w.writerow({k: _fmt(row.get(k)) for k in self.fields})
        self._f.flush()

    def close(self):
        if self._f:
            self._f.close()


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


# ------------------------------------------------------------------ fixed schemas
PHASES = [
    "step_total",        # whole optimizer step (callback on_step_begin -> on_step_end)
    "rollout_total",     # TRL _prepare_inputs: generate + score + logprobs + advantages
    "generation",        # model.generate (prefill + decode)
    "reward",            # DS-1000 execution reward (CPU, subprocesses)
    "logprob_nograd",    # reference / old-policy log-prob forward passes (no grad)
    "loss_compute",      # TRL compute_loss (includes policy_forward)
    "policy_forward",    # policy log-prob forward with grad
    "backward",          # accelerator.backward
    "optimizer_step",    # optimizer.step (+ XLA graph execution when not phase-synced)
]

PHASE_EVENT_FIELDS = ["step", "phase", "parent", "call_idx", "start_unix", "rel_start_s",
                      "duration_s"]

STEP_FIELDS = (
    ["step", "is_warmup", "step_start_unix", "step_end_unix"]
    + [f"t_{p}_s" for p in PHASES]
    + ["t_other_s",
       "prompts", "completions",
       "prompt_tokens", "generated_tokens", "effective_completion_tokens", "train_tokens",
       "gen_tokens_per_s", "train_tokens_per_s", "samples_per_s", "step_tokens_per_s",
       "reward_mean", "reward_std", "pass_rate", "frac_groups_zero_std",
       "loss", "grad_norm", "kl", "learning_rate",
       "peak_mem_mb",
       "compile_count_delta", "compile_time_s_delta", "execute_count_delta", "execute_time_s_delta",
       "nki_kernel_calls", "torch_fallback_calls"]
)

REWARD_FIELDS = ["step", "sample_idx", "problem_id", "library", "completion_chars",
                 "completion_tokens", "reward", "passed", "status", "exec_time_s"]

DEVICE_FIELDS = ["timestamp_unix", "backend", "device_idx", "util_pct", "mem_used_mb",
                 "mem_total_mb", "power_w", "sm_clock_mhz", "temp_c"]

KERNEL_FIELDS = ["platform", "backend", "op", "impl", "shape", "dtype", "iters",
                 "first_call_ms", "median_ms", "p10_ms", "p90_ms", "mean_ms",
                 "tflops", "gb_per_s", "status"]


class RunStore:
    """All writers for one run directory."""

    def __init__(self, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.phase_events = CsvWriter(os.path.join(run_dir, "phase_events.csv"), PHASE_EVENT_FIELDS)
        self.steps = CsvWriter(os.path.join(run_dir, "step_metrics.csv"), STEP_FIELDS)
        self.rewards = CsvWriter(os.path.join(run_dir, "rewards.csv"), REWARD_FIELDS)
        self.trl_logs = None   # columns depend on TRL version; fixed at first log

    def write_run_info(self, info: dict):
        with open(os.path.join(self.run_dir, "run_info.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "value"])
            for k, v in info.items():
                w.writerow([k, _fmt(v)])

    def write_trl_log(self, step, logs: dict):
        row = {"step": step, **{k: v for k, v in logs.items() if isinstance(v, (int, float, str))}}
        if self.trl_logs is None:
            self.trl_logs = CsvWriter(os.path.join(self.run_dir, "trl_logs.csv"))
        self.trl_logs.write(row)

    def close(self):
        for w in (self.phase_events, self.steps, self.rewards, self.trl_logs):
            if w:
                w.close()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_run_info(run_dir):
    return {r["key"]: r["value"] for r in read_csv(os.path.join(run_dir, "run_info.csv"))}
