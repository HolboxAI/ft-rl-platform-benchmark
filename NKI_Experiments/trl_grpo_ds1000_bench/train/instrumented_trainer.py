"""Stock TRL GRPOTrainer with timing hooks around each GRPO phase.

Nothing in the algorithm changes. Hooks are attached by wrapping methods, so this works across
TRL versions as long as the method names below exist (missing ones are skipped and reported):

  step_total      TrainerCallback on_step_begin -> on_step_end
  rollout_total   GRPOTrainer._prepare_inputs (generation, scoring, log-probs, advantages)
    generation      model.generate          (instance-level wrap; also counts tokens)
    reward          DS1000Reward.__call__   (see bench/ds1000_reward.py)
    logprob_nograd  _get_per_token_logps[_and_entropies] under no_grad (ref / old policy)
  loss_compute    GRPOTrainer.compute_loss
    policy_forward  _get_per_token_logps[_and_entropies] with grad
  backward        accelerator.backward
  optimizer_step  callback on_pre_optimizer_step -> on_optimizer_step
"""

import statistics
import time

import torch
from transformers import TrainerCallback
from trl import GRPOTrainer

from bench import platform_utils as pu


class BenchState:
    """Per-run bookkeeping shared by the trainer, the callback and the reward function."""

    def __init__(self, timer, store, backend, warmup_steps, extra_generate_kwargs,
                 nki_counters=None, reward_fn=None):
        self.timer = timer
        self.store = store
        self.backend = backend
        self.warmup_steps = warmup_steps
        self.extra_generate_kwargs = extra_generate_kwargs
        self.nki_counters = nki_counters if nki_counters is not None else {}
        self.reward_fn = reward_fn
        self.hooks_installed = []
        self.hooks_missing = []
        self._reset_step()
        self.pending_rows = {}

    def _reset_step(self):
        self.tok = {"prompts": 0, "completions": 0, "prompt_tokens": 0, "generated_tokens": 0,
                    "effective_completion_tokens": 0, "train_tokens": 0}
        self.step_rewards = []

    def on_reward_result(self, row):
        self.step_rewards.append(row)
        self.store.rewards.write(row)


class InstrumentedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, bench: BenchState, **kwargs):
        super().__init__(*args, **kwargs)
        self.bench = bench
        t = bench.timer

        # generation
        gen_model = self.accelerator.unwrap_model(self.model)
        gen_model.generate = self._timed_generate(gen_model.generate)
        bench.hooks_installed.append("generation")

        # log-prob forward passes (name depends on TRL version; wrap only one so nothing double-counts)
        for name in ("_get_per_token_logps_and_entropies", "_get_per_token_logps"):
            if hasattr(self, name):
                setattr(self, name, t.wrap(getattr(self, name), name, name_fn=lambda *a, **k:
                        "policy_forward" if torch.is_grad_enabled() else "logprob_nograd"))
                bench.hooks_installed.append(f"logprobs:{name}")
                break
        else:
            bench.hooks_missing.append("logprobs")

        # backward
        self.accelerator.backward = t.wrap(self.accelerator.backward, "backward")
        bench.hooks_installed.append("backward")
        self.add_callback(BenchCallback(bench))

    # ---- phases implemented by overriding
    def _prepare_inputs(self, *args, **kwargs):
        with self.bench.timer.phase("rollout_total"):
            return super()._prepare_inputs(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        with self.bench.timer.phase("loss_compute"):
            return super().compute_loss(*args, **kwargs)

    def _timed_generate(self, generate):
        bench = self.bench

        def wrapped(*args, **kwargs):
            kwargs = {**kwargs, **bench.extra_generate_kwargs}
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            attn = kwargs.get("attention_mask")
            with bench.timer.phase("generation"):
                out = generate(*args, **kwargs)
            seqs = out.sequences if hasattr(out, "sequences") else out
            if input_ids is not None:
                p_len = input_ids.shape[1]
                comp = seqs[:, p_len:].detach().cpu()
                eos = self.processing_class.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                eff = 0
                for row in comp.tolist():
                    n = len(row)
                    for j, tok in enumerate(row):
                        if tok in eos_ids:
                            n = j + 1
                            break
                    eff += n
                bench.tok["prompts"] += input_ids.shape[0]
                bench.tok["completions"] += comp.shape[0]
                bench.tok["prompt_tokens"] += int(attn.sum().item()) if attn is not None else input_ids.numel()
                bench.tok["generated_tokens"] += comp.numel()
                bench.tok["effective_completion_tokens"] += eff
                bench.tok["train_tokens"] += seqs.numel()
            return out

        wrapped.__wrapped__ = generate
        return wrapped


def _first(logs, *keys):
    for k in keys:
        if k in logs:
            return logs[k]
    return None


class BenchCallback(TrainerCallback):
    def __init__(self, bench: BenchState):
        self.b = bench
        self._step_cm = None
        self._opt_cm = None
        self._counters0 = None
        self._nki0 = {}
        self._start_unix = None

    def on_step_begin(self, args, state, control, **kw):
        if self._step_cm is not None:          # already inside a step
            return
        b = self.b
        step = state.global_step + 1
        b.timer.step = step
        if b.reward_fn is not None:
            b.reward_fn.step = step
        b._reset_step()
        pu.synchronize(b.backend)              # always sync at step edges, even in "step" mode
        pu.reset_peak_memory(b.backend)
        self._counters0 = pu.compiler_counters(b.backend)
        self._nki0 = dict(b.nki_counters)
        self._start_unix = time.time()
        self._step_cm = b.timer.phase("step_total")
        self._step_cm.__enter__()

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self._opt_cm = self.b.timer.phase("optimizer_step")
        self._opt_cm.__enter__()

    def on_optimizer_step(self, args, state, control, **kw):
        if self._opt_cm is not None:
            self._opt_cm.__exit__(None, None, None)
            self._opt_cm = None

    def on_step_end(self, args, state, control, **kw):
        b = self.b
        if self._opt_cm is not None:           # older transformers without on_optimizer_step
            self._opt_cm.__exit__(None, None, None)
            self._opt_cm = None
        pu.synchronize(b.backend)
        self._step_cm.__exit__(None, None, None)
        self._step_cm = None
        step = b.timer.step
        tot = b.timer.step_totals(step)
        c1 = pu.compiler_counters(b.backend)

        row = {"step": step, "is_warmup": int(step <= b.warmup_steps),
               "step_start_unix": round(self._start_unix, 3), "step_end_unix": round(time.time(), 3)}
        for ph in ("step_total", "rollout_total", "generation", "reward", "logprob_nograd",
                   "loss_compute", "policy_forward", "backward", "optimizer_step"):
            row[f"t_{ph}_s"] = tot.get(ph, 0.0)
        row["t_other_s"] = row["t_step_total_s"] - sum(
            row[f"t_{p}_s"] for p in ("rollout_total", "loss_compute", "backward", "optimizer_step"))
        row.update(b.tok)

        def rate(n, t):
            return n / t if t else None
        row["gen_tokens_per_s"] = rate(b.tok["generated_tokens"], row["t_generation_s"])
        row["train_tokens_per_s"] = rate(b.tok["train_tokens"],
                                         row["t_loss_compute_s"] + row["t_backward_s"])
        row["samples_per_s"] = rate(b.tok["completions"], row["t_step_total_s"])
        row["step_tokens_per_s"] = rate(b.tok["generated_tokens"], row["t_step_total_s"])

        rw = [r["reward"] for r in b.step_rewards]
        if rw:
            row["reward_mean"] = statistics.fmean(rw)
            row["reward_std"] = statistics.pstdev(rw)
            row["pass_rate"] = sum(r["passed"] for r in b.step_rewards) / len(rw)
            groups = {}
            for r in b.step_rewards:
                groups.setdefault(r["problem_id"], []).append(r["reward"])
            row["frac_groups_zero_std"] = sum(1 for g in groups.values() if len(set(g)) == 1) / len(groups)

        row["peak_mem_mb"] = pu.peak_memory_mb(b.backend)
        for k in ("compile_count", "compile_time_s", "execute_count", "execute_time_s"):
            if c1[k] is not None and self._counters0[k] is not None:
                row[f"{k}_delta"] = c1[k] - self._counters0[k]
        row["nki_kernel_calls"] = b.nki_counters.get("nki", 0) - self._nki0.get("nki", 0)
        row["torch_fallback_calls"] = b.nki_counters.get("torch", 0) - self._nki0.get("torch", 0)
        # loss / grad_norm / kl arrive in on_log, which runs after on_step_end
        b.pending_rows[step] = row

    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        step = state.global_step
        self.b.store.write_trl_log(step, logs)
        row = self.b.pending_rows.pop(step, None)
        if row is None:
            return
        row["loss"] = _first(logs, "loss", "train_loss")
        row["grad_norm"] = _first(logs, "grad_norm")
        row["kl"] = _first(logs, "kl", "objective/kl")
        row["learning_rate"] = _first(logs, "learning_rate")
        self.b.store.steps.write(row)

    def on_train_end(self, args, state, control, **kw):
        for step in sorted(self.b.pending_rows):
            self.b.store.steps.write(self.b.pending_rows[step])
        self.b.pending_rows.clear()
