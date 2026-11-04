# Implementation Plan — Qwen3.5 + RL weight-sync on vLLM-neuron (Trainium2)

## 1. Objective

Add two things to the AWS `vllm-neuron` plugin so the CholecT50 GRPO/RLVR loop
runs on Trainium2 through vLLM's own RL entrypoint:

1. **Qwen3.5 hybrid-GatedDeltaNet model support.**
2. **A neuron weight-transfer backend** (delta/fast weight-sync) **and neuron
   model `sleep`/`wake_up`** (weight offload).

This is a contribution to vLLM-neuron, not a new library. Upstream vLLM already
has everything else.

## 2. Background — the finding

Source analysis of two repos (cloned 2026-11-04):

| Repo | Commit | Notes |
|---|---|---|
| `vllm-project/vllm` | `5225a1f` | upstream vLLM |
| `vllm-project/vllm-neuron` | `a080ebf` | AWS Neuron plugin, Beta, v0.29.0.1.2.0, Neuron SDK 2.33 |

Upstream vLLM already has the full RL backbone:

- Qwen3.5 model, dense and multimodal — `vllm/model_executor/models/qwen3_5.py`
  (`Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`).
- GatedDeltaNet layer — `vllm/model_executor/layers/mamba/gdn/`.
- GDN kernels: CUDA Triton (`ops/causal_conv1d.py`), CUTLASS
  (`ops/gdn_chunk_cutedsl/`), AMD HIP (`ops/hip_gdn_chunk.py`), and an 830-line
  CPU reference (`ops/cpu/gdn_attention.py`).
- RL entrypoint — `vllm/entrypoints/rl/offline.py` (`RLOfflineMixin`).
- Weight-sync API — `vllm/engine/protocol.py` (`sleep`, `wake_up`,
  `update_weights`, `update_weight_version`).
- Weight-transfer framework — `vllm/distributed/weight_transfer/`
  (`WeightTransferEngine` ABC + `factory.py`). Backends: nccl, sparse_nccl, ipc,
  sharded_rdt, m2n, modelexpress. **All CUDA/RDMA. No neuron.**

The neuron plugin is missing only these parts:

- No Qwen3.5 in the model registry. `vllm_neuron/model/registry.py` lists only
  `Qwen3ForCausalLM`, `Qwen3VLForConditionalGeneration`,
  `Qwen3ASRForConditionalGeneration`, `DFlashDraftModel`.
- No GDN kernel. `vllm_neuron/nki/` is empty. `vllm_neuron/_kernels/` has no GDN.
  `vllm_neuron/functional/attention/` is full-attention only.
- No weight-transfer backend. No `sleep`/`wake_up` (weight offload).

The GDN dispatch in `qwen_gdn_linear_attn.py:715-730` shows the exact problem:

```python
if   current_platform.is_xpu():  self._forward_method = self.forward_xpu
elif current_platform.is_cpu():  self._forward_method = self.forward_cpu   # CPU reference
elif current_platform.is_rocm(): self._forward_method = self.forward_hip
else:                            self._forward_method = self.forward_cuda  # neuron lands here, then fails
```

On neuron the dispatch falls to `forward_cuda` and dies on `causal_conv1d_fn`.

## 3. Scope

**In scope:** Qwen3.5 hybrid-GDN support, neuron weight-transfer backend, neuron
`sleep`/`wake_up`. **Out of scope:** Miles (SGLang, not vLLM), the TRL eager
single-core path, the shared model/toolchain walls that any framework hits
(TDRAM exhaustion, `repeat_interleave`, `masked_scatter` — see
`../TRAINIUM2_PLATFORM_BLOCKERS.md`).

## 4. Proposed changes

### Change 1 — Qwen3.5 hybrid-GDN support

The model is 8 full-attention layers + 24 GatedDeltaNet layers + a vision tower.
The dense Qwen3/Qwen3-VL already in vLLM-neuron covers the full-attention layers
and the vision tower. Only the 24 GDN layers are new.

**1a. Neuron GDN kernel (correctness first, then speed).**

- Port the 830-line CPU reference `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`
  as the correctness baseline. It is a full, self-contained spec of the GDN math
  (`causal_conv1d` + chunked gated-delta-rule).
- First landing: run the GDN math eagerly on the `neuron` device (pure PyTorch,
  no NKI). Correct but slow. This unblocks the whole benchmark immediately.
- Second landing: an NKI kernel in the currently-empty `vllm_neuron/nki/` for the
  perf path. This is the one hard piece. It is a port, not a from-scratch kernel,
  and may need AWS's NKI team.

**1b. Register Qwen3.5.**

Add `Qwen3_5ForCausalLM` and `Qwen3_5ForConditionalGeneration` to
`vllm_neuron/model/registry.py`. Reuse the Qwen3-VL vision tower already wired in
upstream `qwen3_5.py`.

**1c. Dispatch.**

Add an `is_neuron()` branch to `qwen_gdn_linear_attn.py` that selects the neuron
GDN path, so neuron does not fall into `forward_cuda`.

**Acceptance:** the `cholec-sft-qwen3.5-4b` checkpoint serves on a
`trn2.48xlarge`; generation output matches the CUDA reference on a sample of
CholecT50 prompts.

### Change 2 — neuron weight-transfer backend

Upstream vLLM has the full weight-sync framework. What is missing is a neuron
backend.

- Implement `WeightTransferEngine` with `backend = "neuron"` and register it in
  `WeightTransferTrainerFactory` (`vllm/distributed/weight_transfer/factory.py`).
- Transport: a neuron collective (neuronx-distributed / gloo mesh — already
  prototyped in the SkyRL leg) + pinned CPU staging buffers + a
  `copy_=False` load into the resident rollout model. No disk, no re-allocation,
  no full reload.
- LoRA/delta adapter sync rides the same API with a tiny payload.

**Acceptance:** `update_weights` / `update_weight_version` round-trips a policy
update into the rollout in-place, without re-creating the engine.

### Change 3 — neuron `sleep`/`wake_up` (model weight offload)

vLLM-neuron does not implement model-weight offload (only KV offload, and that is
marked unsupported). verl/Miles/SkyRL colocate trainer + rollout and offload one
side to CPU. On neuron this needs a weight-offload path: params to CPU during
training, reload during rollout.

**Acceptance:** `engine.sleep()` frees the on-core HBM for the trainer, and
`engine.wake_up()` restores the weights for the next rollout step.

## 5. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| GDN first landing | eager PyTorch | correct now, unblocks end-to-end; NKI comes after |
| GDN port target | `ops/cpu/gdn_attention.py` | full reference, no guessing at the math |
| Weight-sync transport | neuron collective + pinned CPU staging | no RDMA; matches the SkyRL gloo→neuron prototype |
| Model registration | reuse Qwen3-VL vision tower | no new vision work |

## 6. Test plan

- **Unit:** GDN CPU reference vs CUDA reference — bit-parity on the CholecT50
  checkpoint (on an NVIDIA box for the CUDA side).
- **On-instance (trn2):** Phase 0 probe — serve Qwen3.5 on vLLM-neuron, capture
  the exact failure. Phase 1 gate — Qwen3.5 serves (eager GDN). Phase 3 gate —
  `update_weights` round-trip + `sleep`/`wake_up` round-trip.
- **Integration:** run the CholecT50 GRPO loop via `entrypoints/rl/` and compare
  reward to the verl reference (0.715).

## 7. Phased rollout

| Phase | Work | Outcome |
|---|---|---|
| 0 | serve Qwen3.5 on vLLM-neuron; capture the exact failure | go/no-go + upstream issue text |
| 1 | eager neuron GDN + register Qwen3.5 + vision reuse | Qwen3.5 serves (slow, correct) |
| 2 | NKI GDN kernel | Qwen3.5 serves fast |
| 3 | neuron `WeightTransferEngine` backend + `sleep`/`wake_up` | delta/fast weight-sync |
| 4 | CholecT50 GRPO via `entrypoints/rl/` | benchmark datapoint |

## 8. Risks

- The NKI GDN kernel (Phase 2) is the one hard piece. Mitigation: the eager path
  (Phase 1) proves correctness and the benchmark end-to-end first; the NKI work is
  co-owned with AWS.
- The shared model/toolchain walls (TDRAM, `repeat_interleave`, `masked_scatter`)
  still apply to any large run. They are independent of this plan.

## 9. Effort (3-person intermediate DS team)

Phase 0: 1 day. Phase 1: 1–2 weeks. Phase 3: 1–2 weeks. Phase 4: 1 week.
Phase 2 (NKI GDN): bounded if AWS co-owns it, open-ended alone.

## 10. Dependencies

- AWS NKI team help for the GDN kernel (Phase 2).
- A `trn2.48xlarge` spot instance (access already verified).
- NVIDIA GPU access for the GDN parity test (Phase 1/unit test).

## 11. Open questions

- Does vLLM-neuron accept an eager (non-NKI) custom op for GDN, or must it be NKI?
- Does the upstream `WeightTransferTrainerFactory` load plugins from the neuron
  package, or must the neuron backend live in upstream vLLM?

## Sources

- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`
- `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`
- `vllm/entrypoints/rl/offline.py`
- `vllm/engine/protocol.py`
- `vllm/distributed/weight_transfer/`
- `vllm_neuron/model/registry.py`
- `vllm_neuron/nki/`, `vllm_neuron/_kernels/`, `vllm_neuron/functional/attention/`

Related: `../VLLM_NEURON_ASSESSMENT.md`, `../TRAINIUM2_PLATFORM_BLOCKERS.md`.
