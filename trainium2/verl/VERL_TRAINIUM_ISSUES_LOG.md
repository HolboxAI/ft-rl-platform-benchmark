# verl on Trainium2 (TorchNeuron) — issues log

Same purpose as `trainium2/TRAINIUM_ISSUES_LOG.md` (TRL leg): record every error
hit while bringing **verl (GRPO/RLVR)** to a run on **Trainium2 via
TorchNeuron**, with exact error text and a fix class (dev-side vs package-side).

---

## Milestones

- **2026-10-26** — instance launched (trn2.48xlarge spot, ap-south-2b,
  `i-009227ce775189eea`). verl 0.9.0 installed on Python 3.14.4 (after two pin
  overrides). `PlatformNeuron` plugin registers and **gate_test.py PASSES**
  (forward+backward on `neuron`). Rollout selection captured as the blocker
  (async-server-only). Instance terminated after capture.

---

## What works (verified on-instance)

1. **verl platform layer is pluggable.** Our `PlatformNeuron` registers via the
   `verl.plugins` setuptools entry point (no verl source edit). `get_platform()`
   returns it under `VERL_PLATFORM=neuron`; `is_available()` True;
   `device_count()` 1; `empty_cache()`/`synchronize()` wired.
2. **A model runs on `neuron`.** `gate_test.py` does a `Linear→GELU→Linear`
   forward + backward on `neuron` (loss 0.042, 4.2 MB alloc).
3. **`import verl` succeeds on Python 3.14** after the two pin overrides below.
   `torch_neuronx` device module resolves to `torch_neuronx.neuron`.

---

## Issues encountered (in order)

### #1 — verl 0.9.1 requires Python `<3.13`, DLAMI ships 3.14.4

- **Status**: RESOLVED (use verl 0.9.0).
- **Class**: package-side (verl).
- **Error**: `pip install verl==0.9.1` →
  `Ignored ... 0.9.1 Requires-Python >=3.10,<3.13`.
- **Fix**: verl **0.9.0** already ships `verl/plugin/platform/` (the platform
  abstraction) and pins `Requires-Python >=3.10` (no upper bound), so it
  installs on 3.14. Use 0.9.0, not 0.9.1.
- **Library fix (verl)**: release the platform refactor on a Python
  3.13/3.14-compatible line (the `<3.13` cap in 0.9.1 is the whole conflict).

### #2 — verl's `tensordict!=0.9.0,<=0.10.0,>=0.8.0` pin has no 3.14 wheels

- **Status**: RESOLVED (override to 0.15.3).
- **Class**: package-side (verl pin vs torch/tensordict 3.14 support).
- **Error**: `Could not find a version that satisfies the requirement
  tensordict!=0.9.0,<=0.10.0,>=0.8.0` — the 0.8.0–0.10.0 range has no
  Python 3.14 distribution.
- **Fix**: `pip install tensordict==0.15.3` (the 3.14 line). verl imports and
  runs; the pin mismatch is only a resolver warning. Unverified inside the full
  RL loop (verl may use tensordict APIs that changed 0.10→0.15).
- **Library fix (verl)**: relax the tensordict upper bound to the 3.14 line.

### #3 — verl's `transformers!=5.6.0,<5.11,>=5.5.3` conflicts with qwen3_5's 5.18.0

- **Status**: WORKAROUND (keep 5.18.0); unverified in the RL loop.
- **Class**: package-side (verl pin vs model arch requirement).
- **Error**: resolver warning `transformers!=5.6.0,<5.11,>=5.5.3 ... you have
  transformers 5.18.0 which is incompatible`.
- **Fix**: keep transformers 5.18.0 (required for `qwen3_5`; the TRL-leg pin).
  `import verl` works, but verl's own transformers-facing code has not been
  exercised against 5.18.0 yet.
- **Library fix (verl)**: loosen the transformers cap to `>=5.11`.

### #4 — verl imports `orjson` but does not declare it

- **Status**: RESOLVED (manual install).
- **Class**: package-side (verl packaging gap).
- **Error**: `ModuleNotFoundError: No module named 'orjson'` at
  `verl/utils/tracking.py:28`, reached via `verl.workers.config.actor`.
- **Fix**: `pip install orjson` (plus `msgpack`, `aiohttp`).
- **Library fix (verl)**: add `orjson` to `Requires-Dist`.

### #5 — device module resolves to `torch_neuronx.neuron`, not `torch.neuron`

- **Status**: RESOLVED (in `PlatformNeuron`).
- **Class**: run-side (our plugin).
- **Observed**: `DEVICE_MODULE <module 'torch_neuronx.neuron' ...>`. TorchNeuron
  exposes the standard device API at `torch_neuronx.neuron`, so the fallback in
  `_resolve_device_module()` is what actually supplies `is_available` /
  `device_count`. The `torch.neuron` path is absent.
- **Fix**: keep the `torch.neuron` → `torch_neuronx.neuron` fallback (done).

### #6 — verl 0.9.0 rollout is async-server-only (the real blocker)

- **Status**: BLOCKED — needs a custom async rollout server.
- **Class**: package-side (verl) + missing back-end (no Neuron rollout).
- **Error** (exact, on-instance):
  - `get_rollout_class("naive"/"hf"/"neuron", "async")` →
    `AssertionError: Rollout <name> with mode async not found`.
  - `get_rollout_replica_class("naive"/"hf"/"neuron")` →
    `ValueError: Unknown rollout mode: <name>. Available: ['vllm','sglang','trtllm']`.
- **Root cause**: the training loop selects the rollout with
  `get_rollout_class(rollout_config.name, rollout_config.mode)`
  (`verl/workers/engine_workers.py:661`) against `_ROLLOUT_REGISTRY`
  (`verl/workers/rollout/base.py`), which only maps `("vllm","async")`,
  `("sglang","async")`, `("trtllm","async")` — all CUDA. The transformers-native
  `HFRollout` / `NaiveRollout` still exist but are dead code: the sync mode that
  selected them was removed (`RolloutConfig.__post_init__`: "Rollout mode 'sync'
  has been removed").
- **Not a version problem (checked 0.6.1/0.7.0/0.8.0).** `HFRollout` /
  `NaiveRollout` are never in `_ROLLOUT_REGISTRY` in any 0.6–0.9 release; the
  registry only ever held vllm/sglang/(trtllm). 0.6.1/0.7.0 keep an
  `if rollout_name == "hf":` branch in `fsdp_workers.py`, but it only flips a
  collect-info flag and then calls `get_rollout_class("hf", ...)`, which asserts.
  So "use an older verl" and "verl natively supports HF rollout" (a Google-AI
  claim) are both wrong.
- **What "resolve" requires**: a `NeuronReplica(RolloutReplica)` implementing
  `launch_servers()` — an in-process (or Ray-actor) transformers-native
  inference server on `neuron` — registered in `_ROLLOUT_REGISTRY` and
  `RolloutReplicaRegistry`. This is a real implementation (weight-sync,
  generate, actor lifecycle), not a config flag.
- **Library fix (verl)**: keep `HFRollout`/`NaiveRollout` selectable as a
  training rollout (device-agnostic escape hatch), or document a
  `RolloutReplica` authoring contract + registry entry point for non-CUDA
  backends. Without this, every non-NVIDIA/vLLM deployment must write its own
  server.

---

## Model-level walls (expected, carried from the TRL leg)

These will hit once a rollout exists and the loop actually trains on `neuron`:

- Static shapes / on-core TDRAM exhaustion at ~step 40 from variable RL batch
  lengths (`TRAINIUM_ISSUES_LOG.md` #23).
- Qwen3.5 Gated DeltaNet: 24/32 layers fall back to reference PyTorch without
  Neuron `causal_conv1d` / `flash-linear-attention` kernels (#11).
- Dynamic `torch.repeat_interleave` in the vision tower — already patched in
  `PlatformNeuron.apply_model_patches` (#1).

---

## How to avoid these next time

- **Run-side**: install verl `--no-deps` and drive the import chain yourself;
  the two pin overrides (tensordict, transformers) are known and documented in
  `scripts/setup.sh`.
- **Package-side (verl)**: Python 3.13/3.14 support, tensordict upper bound,
  transformers cap, `orjson` declaration, and a device-agnostic rollout path
  (issues #1–#6 above).
- **Package-side (torch-neuronx)**: see the TRL leg log — static-shape/TDRAM,
  GDN kernels, `repeat_interleave` lowering.

---

## Package-side changes we're waiting on (verl leg)

Grouped by package, mirroring the TRL leg's list. Issue numbers refer to this
file. These are the changes the relevant package needs for a native
verl-on-Trainium2 run.

### verl (framework) — 5

- **#1** — verl 0.9.1 caps `Requires-Python <3.13`; the DLAMI ships 3.14.
  Fix: release the platform refactor on a 3.13/3.14-compatible line.
- **#2** — `tensordict!=0.9.0,<=0.10.0,>=0.8.0` has no 3.14 wheels.
  Fix: relax the upper bound to the 3.14 line (0.15.x).
- **#3** — `transformers!=5.6.0,<5.11,>=5.5.3` conflicts with `qwen3_5`'s 5.18.0.
  Fix: loosen the cap to `>=5.11`.
- **#4** — imports `orjson` (plus `msgpack`, `aiohttp`) without declaring them.
  Fix: add `orjson` to `Requires-Dist`.
- **#6** — rollout is async-server-only (vllm/sglang/trtllm); `HFRollout` /
  `NaiveRollout` are dead code never in `_ROLLOUT_REGISTRY`. Fix: keep a
  device-agnostic transformers-native rollout selectable, or document a
  `RolloutReplica` authoring contract + registry entry point for non-CUDA
  backends.

### Carried over from the TRL leg (same runtime/compiler stack) — 7

Not hit yet here (the rollout blocks before training), but unchanged for verl;
see `../TRAINIUM_ISSUES_LOG.md`:

- **torch-neuronx** — #8 unclear OOM, #9 OOM recovery loop, #10 trace-cache pin,
  #11 GDN kernel fallback, #23 TDRAM exhaustion.
- **neuronx-cc** — #1 dynamic `repeat_interleave`, #20 `masked_scatter`
  instruction limit.
