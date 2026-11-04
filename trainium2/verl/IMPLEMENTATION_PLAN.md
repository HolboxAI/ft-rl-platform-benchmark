# Implementation Plan — a Trainium2-native RL library based on verl

## 1. Objective

Build our own RL library on top of verl. Make it run the CholecT50 GRPO/RLVR
loop natively on AWS Trainium2. The library has two layers:

1. **A verl fork** — verl changes that make the trainer, the rollout, and the
   weight-sync work on the `neuron` device. This is the library itself.
2. **vLLM-neuron fixes** — an upstream contribution that makes vLLM-neuron serve
   Qwen3.5 and receive weight updates on neuron. The verl fork uses vLLM-neuron
   as its rollout engine.

Both layers are required. The vLLM-neuron fixes are the foundation. The verl
fork is the library on top.

## 2. Architecture

```
                    ┌────────────────────────────────────┐
                    │  verl fork (our library)            │
                    │   ┌──────────┐     ┌─────────────┐  │
   train data ─────▶│   │ trainer  │     │ rollout     │  │
                    │   │ (neuron) │     │ (vLLM-neuron)│  │
                    │   │ FSDP2    │     │             │  │
                    │   └────┬─────┘     └──────▲──────┘  │
                    │        │ weight-sync      │         │
                    │        └──────────────────┘         │
                    └──────────────────┬─────────────────┘
                                       │ weight-transfer API
                    ┌──────────────────▼─────────────────┐
                    │  vLLM-neuron (fixed)                │
                    │  Qwen3.5 + GatedDeltaNet + weight   │
                    │  sync + sleep/wake_up               │
                    └─────────────────────────────────────┘
```

One node (`trn2.48xlarge`, 48 NeuronCores) runs trainer and rollout colocated.
The trainer trains on the `neuron` device with FSDP2. The rollout serves Qwen3.5
on vLLM-neuron. Weight-sync pushes the new policy weights from the trainer into
the resident rollout every step.

## 3. The two workstreams

### Workstream A — vLLM-neuron fixes (the foundation)

This is a bounded upstream contribution. Upstream vLLM already has Qwen3.5-VL,
the full GatedDeltaNet implementation, the RL weight-sync API, and a pluggable
weight-transfer framework. vLLM-neuron is missing exactly three pieces.

**A1. Qwen3.5 hybrid-GatedDeltaNet support.**

The model is 8 full-attention layers + 24 GatedDeltaNet layers + a vision tower.
vLLM-neuron already covers the full-attention layers (dense Qwen3/Qwen3-VL) and
the vision tower. Only the 24 GDN layers are new.

- Port the 830-line CPU reference `ops/cpu/gdn_attention.py` as the correctness
  baseline.
- First landing: run the GDN math eagerly on `neuron` (pure PyTorch, no NKI).
  Correct but slow. It unblocks the whole benchmark.
- Second landing: an NKI kernel in the empty `vllm_neuron/nki/` for speed. This
  is the one hard piece. It is a port, not a from-scratch kernel.
- Register `Qwen3_5ForCausalLM` and `Qwen3_5ForConditionalGeneration` in
  `vllm_neuron/model/registry.py`. Reuse the Qwen3-VL vision tower.
- Add an `is_neuron()` branch to the GDN dispatch
  (`qwen_gdn_linear_attn.py:715-730`) so neuron does not fall into `forward_cuda`.

**A2. Neuron weight-transfer backend.**

- Implement `WeightTransferEngine` with `backend = "neuron"` and register it in
  `WeightTransferTrainerFactory` (`vllm/distributed/weight_transfer/factory.py`).
- Transport: a neuron collective (neuronx-distributed / gloo mesh) + pinned CPU
  staging buffers + a `copy_=False` load into the resident rollout model.
- LoRA/delta adapter sync rides the same API with a tiny payload.

**A3. Neuron `sleep`/`wake_up` (model weight offload).**

- vLLM-neuron does not implement weight offload. Add it so the trainer can use
  the HBM the rollout frees during the train step.

**Acceptance:** `cholec-sft-qwen3.5-4b` serves on a `trn2.48xlarge`;
`update_weights` / `update_weight_version` round-trip in-place;
`engine.sleep()` / `engine.wake_up()` round-trip.

### Workstream B — verl fork changes (our library)

The verl fork makes verl Trainium2-native and wires it to vLLM-neuron.

**B1. `PlatformNeuron` (done).**

Already in `verl/plugin/verl_neuron_plugin/platform_neuron.py`, auto-loaded via
the setuptools `verl.plugins` entry point. The gate test passes (forward +
backward on `neuron`). No verl source edit needed.

**B2. Neuron rollout (`NeuronReplica`).**

verl 0.9.0 is async-server-only: `get_rollout_class("hf"/"naive")` asserts, and
`_ROLLOUT_REGISTRY` maps only `vllm`/`sglang`/`trtllm` (all CUDA).
`HFRollout`/`NaiveRollout` are dead code. This is verl's real blocker.

- Implement `NeuronReplica(RolloutReplica)` that drives a vLLM-neuron engine
  (serving Qwen3.5 after Workstream A).
- Register it in `_ROLLOUT_REGISTRY` and `RolloutReplicaRegistry`.
- This is the largest verl code change.

**B3. Neuron weight-sync.**

verl's collective backends are `nccl`/`hccl`/`gloo`. Wire verl's weight-update
path to vLLM's weight-transfer API (the neuron backend from A2). The trainer
pushes the new policy into the resident rollout in-place. LoRA/delta sync rides
the same path.

**B4. Trainer on neuron (FSDP2).**

Run verl's trainer with FSDP2 over neuronx-distributed. Shard the model and the
optimizer states across the 48 cores. Replace the CUDA-only collective calls
with neuron collectives.

**B5. Colocation + offload.**

verl colocates trainer and rollout. Use vLLM-neuron `sleep`/`wake_up` (A3) so
the trainer uses the HBM the rollout frees during each train step.

**Acceptance:** the verl fork runs the CholecT50 GRPO loop on Trainium2 and
matches the verl reference reward (0.715).

## 4. Shared model walls (still apply)

These are torch-neuronx / neuronx-cc issues, not verl or vLLM. They hit any
framework on Trainium2. See `../TRAINIUM2_PLATFORM_BLOCKERS.md`.

- **C1 GDN fallback** — fixed by A1.
- **C2 TDRAM exhaustion** — variable RL batch shapes exhaust the on-core DMA pool.
- **C3 dynamic `repeat_interleave`** — vision tower; needs a neuronx-cc fix or a
  CPU fallback (already patched in the TRL/SkyRL legs).
- **C6 `masked_scatter` instruction limit** — the response-only loss mask.

## 5. Phased rollout

| Phase | Work | Outcome |
|---|---|---|
| 0 | serve Qwen3.5 on vLLM-neuron; capture the exact failure | go/no-go + upstream issue |
| 1 | A1 eager GDN + register Qwen3.5 + vision reuse | Qwen3.5 serves (slow, correct) |
| 2 | A1 NKI GDN kernel | Qwen3.5 serves fast |
| 3 | A2 weight-transfer backend + A3 sleep/wake_up | in-place weight-sync + offload |
| 4 | B2 `NeuronReplica` rollout wired to vLLM-neuron | verl rollout runs on neuron |
| 5 | B3 weight-sync + B4 trainer (FSDP2) + B5 colocation | verl fork trains on neuron |
| 6 | CholecT50 GRPO end-to-end | benchmark datapoint, match 0.715 |

Phases 0–3 are Workstream A (vLLM-neuron). Phases 4–6 are Workstream B (verl
fork). Phase 4 depends on Phase 3.

## 6. Test plan

- **Unit:** GDN CPU reference vs CUDA reference — bit-parity on the CholecT50
  checkpoint.
- **On-instance (trn2):** Phase 0 probe; Phase 1 gate (Qwen3.5 serves);
  Phase 3 gate (`update_weights` + `sleep`/`wake_up` round-trips); Phase 4 gate
  (`NeuronReplica` generates); Phase 6 gate (full GRPO loop).
- **Integration:** compare reward to the verl reference (0.715) and the Miles
  reference (0.65125).

## 7. Risks

- The NKI GDN kernel (Phase 2) is the one hard piece. Mitigation: the eager path
  (Phase 1) proves correctness and the benchmark end-to-end first. Co-own the NKI
  work with AWS.
- The shared model walls (C2, C3, C6) still apply to any large run. They are
  independent of this plan.
- verl's rollout is a hard architectural blocker (async-server-only). The
  `NeuronReplica` work is a real code change, not a config change.

## 8. Effort (3-person intermediate DS team)

Phase 0: 1 day. Phase 1: 1–2 weeks. Phase 3: 1–2 weeks. Phase 4: 2–3 weeks.
Phase 5: 2–3 weeks. Phase 6: 1 week. Phase 2 (NKI GDN): bounded if AWS co-owns
it, open-ended alone.

## 9. Dependencies

- AWS NKI team help for the GDN kernel (Phase 2).
- A `trn2.48xlarge` spot instance (access verified).
- NVIDIA GPU access for the GDN parity test.

## 10. Open questions

- Does vLLM-neuron accept an eager (non-NKI) custom op for GDN, or must it be NKI?
- Does the upstream `WeightTransferTrainerFactory` load a neuron backend from the
  neuron package, or must it live in upstream vLLM?
- Does the verl fork rebase onto verl upstream, or diverge into a maintained fork?
- Do we ship the library as a verl fork, or as a verl plugin + patched verl?

## Sources

- vLLM upstream (`5225a1f`): `models/qwen3_5.py`, `layers/mamba/gdn/`,
  `ops/cpu/gdn_attention.py`, `entrypoints/rl/`, `distributed/weight_transfer/`.
- vLLM-neuron (`a080ebf`): `model/registry.py`, `nki/`, `_kernels/`,
  `functional/attention/`.
- verl leg: `verl/plugin/verl_neuron_plugin/platform_neuron.py`,
  `verl/VERL_TRAINIUM_ISSUES_LOG.md`, `../TRAINIUM2_PLATFORM_BLOCKERS.md`.

Related: `../VLLM_NEURON_ASSESSMENT.md`, `../TRAINIUM2_PLATFORM_BLOCKERS.md`.
