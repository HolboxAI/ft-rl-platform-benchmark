# verl-on-Trainium: rollout resolution plan

Status: the platform layer works (gate passes). The rollout is the blocker.
This is the plan to make verl generate with plain PyTorch/transformers on
`neuron`.

## Problem (one line)

verl 0.9.0 only knows three rollout engines — vLLM, SGLang, TensorRT-LLM — all
CUDA. There is no transformers-native entry, so on Trainium there is no engine
to generate with.

## Approach

Write a small transformers-native rollout and register it. Reuse verl's own
`NaiveRollout` / `HFRollout` (still shipped, just unwired); do not write an
inference engine. The `.generate()` step itself already works on `neuron`
(the TRL leg runs it).

## Where verl picks the rollout (two registries, two paths)

1. **Training-loop (colocated) path**
   - `verl/workers/rollout/base.py` → `_ROLLOUT_REGISTRY` maps
     `("vllm","async")`, `("sglang","async")`, `("trtllm","async")` to
     ServerAdapter classes.
   - `get_rollout_class(name, mode)` does the lookup.
   - Called at `verl/workers/engine_workers.py:661`.
   - Exact error: `AssertionError: Rollout naive with mode async not found`.

2. **Standalone-server path**
   - `verl/workers/rollout/replica.py` → `RolloutReplicaRegistry`
     (`register`/`get`); only vllm/sglang/trtllm registered.
   - Used by `verl/trainer/main_generation_server.py`.
   - Exact error: `ValueError: Unknown rollout mode: neuron. Available:
     ['vllm','sglang','trtllm']`.

## What we write

- A `BaseRollout` subclass (working name `NeuronRollout`) whose
  `generate_sequences` calls `self.module.generate(...)` on `neuron`, copied
  from `NaiveRollout` (`verl/workers/rollout/naive/naive_rollout.py`).
- Register it under a new name (`neuron`) in `_ROLLOUT_REGISTRY`, and in
  `RolloutReplicaRegistry` if the standalone path is the one the run uses.

## Open questions (resolve on-instance)

- Does the training loop call the rollout synchronously or asynchronously?
  verl 0.9.0 expects "async server" style; our in-process generator may need to
  present the same interface (or we point verl at its in-process path).
- Weight sync: in colocated mode the rollout shares the actor's `module`, so
  sync may be free. Confirm which mode `main_ppo` runs by default.

## Steps

1. Read `verl/workers/engine_workers.py` around line 661 to see how
   `rollout_cls` is instantiated and called.
2. Write `NeuronRollout` (copy `NaiveRollout`).
3. Register it; add a `rollout.name=neuron` config.
4. Run a minimal `main_ppo` (dummy model/data) to capture the next error.
5. Iterate; log every error in `VERL_TRAINIUM_ISSUES_LOG.md`.

## Why not downgrade to an older verl (checked 2026-10-26)

Downloaded and inspected 0.6.1 / 0.7.0 / 0.8.0 wheels. Downgrading does not
help:

- **0.6.1** has `mode: str = "sync"`, but its registry maps sync to
  `vllm_rollout.vLLMRollout` (vLLM SPMD) — still NVIDIA/vLLM, not transformers.
- **0.7.0** already removed sync (`mode: str = "async"`), async-only.
- **0.8.0** async-only, same as 0.9.0.
- `NaiveRollout` / `HFRollout` is dead code (test utilities) in **all** of them;
  it is never an entry in `_ROLLOUT_REGISTRY` for any version 0.6–0.9.

### The `rollout.name=hf` branch is a dead leftover (checked 0.6.1/0.7.0)

0.6.1 and 0.7.0 still ship `verl/workers/fsdp_workers.py`, which has an
`if rollout_name == "hf":` branch (0.6.1 line 600, 0.7.0 line 628). It only
sets a collect-info flag. The next lines build the rollout with
`get_rollout_class(rollout_config.name, rollout_config.mode)`, and `("hf", ...)`
is **not** in `_ROLLOUT_REGISTRY` in any 0.6–0.9 release. So `rollout.name=hf`
crashes with `AssertionError: Rollout hf with mode ... not found` in every
version — the branch is a leftover from a removed sync path, not a reachable
config.

`HFRollout.__init__(self, module, config)` also does not match the modern
registry call contract `(config, model_config, device_mesh)`, so it would not
even instantiate if it were registered. `fsdp_workers.py` is gone in 0.8.0
(replaced by `engine_workers.py`).

- Only 0.9.0 has the platform plugin mechanism (`plugin/platform`), which is what
  made the `neuron` device work as a clean plugin. 0.6–0.8 would need direct
  edits to `verl/utils/device.py` plus the ~11 files that touch `torch.cuda`.

Net: older versions do not unblock the rollout, and they lose the working device
layer. Staying on 0.9.0 + this glue is the smallest total job.

## Related

- Issues log: `VERL_TRAINIUM_ISSUES_LOG.md` (issue #6 is this blocker).
- Platform plugin (working): `plugin/`.
