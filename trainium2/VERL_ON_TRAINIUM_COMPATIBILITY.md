# verl on Trainium2 (TorchNeuron) — compatibility assessment

Goal: run the same CholecT50 GRPO/RLVR experiment with **verl + TorchNeuron**
instead of the TRL leg, and list what library changes an effortless run needs.

> **Update (2026-10-26):** the verdict below was revised after reading verl
> 0.9.1 source. verl's device / collective / FSDP layers are **device-agnostic
> and pluggable** (`PlatformBase` + `PlatformRegistry` + the `verl.plugins`
> entry-point group), not hardcoded CUDA. The real port is a bounded patch —
> implemented in `trainium2/verl/plugin/` — and the genuinely hard part is the
> rollout (async-only in 0.9.1). See `trainium2/verl/README.md` and
> `trainium2/verl/VERL_TRAINIUM_ISSUES_LOG.md`.

## Verdict

**verl is closer to native than first assessed, but does not run on Trainium2
out of the box.** verl has no Neuron device backend today, but its platform /
device / collective / FSDP layers are pluggable via `PlatformBase`, so adding a
`neuron` backend is a bounded patch (not a multi-week port). The remaining hard
blocker is the rollout: verl 0.9.1 is async-server-only (vllm/sglang/trtllm)
and removed the transformers-native sync rollout. There is no turnkey
verl + Trainium path as of 2026-10-26; the close-to-native path is the plugin in
`trainium2/verl/` plus a decision on the rollout.

## What verl supports (source-verified)

`verl/utils/device.py` (main branch, `verl-project/verl`) names:

- Device names: `cuda`, `npu`, `cpu`. No `neuron`, `trainium`, or `xla`.
- Collective backends: `nccl`, `hccl`, `gloo`. No Neuron collective.
- Frameworks checked: `torch.cuda` and `torch.npu`.
- Rollout engines: vLLM and SGLang, both CUDA/ROCm.

verl's README lists NVIDIA, AMD (ROCm), and Ascend. Trainium is absent.

## What TorchNeuron already provides (so the gap is precise)

TorchNeuron (Neuron SDK 2.33, public beta 2026-10-20) is a native PyTorch
backend for Trainium2/Trn3. It gives:

- A `neuron` device via PyTorch PrivateUse1 (`.to("neuron")`).
- `torch.distributed` with DDP, FSDP2, tensor/sequence parallelism via DTensor,
  pipeline + context parallelism. Sharding is per NeuronCore.
- `torch.compile(backend="neuron")`, static shapes only.

So the training primitive verl needs (FSDP2 + DTensor) exists on Trainium. The
missing pieces are verl's device, collective, and rollout wiring for Neuron.

## The reference verl run (what this leg compares against)

Full-model FSDP (no LoRA) on 4x A10G, `param_offload` + `optimizer_offload`,
vLLM rollout, val reward 0.715. Nothing in it is Neuron-ready:

- `actor_rollout_ref.rollout.name=vllm` (CUDA).
- `actor_rollout_ref.actor.strategy=fsdp`, `fsdp_size=4` (torch.distributed
  FSDP on CUDA, NCCL).
- Three Qwen3.5 mRoPE patches are already required on CUDA. Neuron needs those
  plus the two Neuron-specific patches below.

## First failure (what an attempted run would show)

verl's platform detection finds no CUDA and no NPU on a Trainium2 box, so it
resolves to CPU and the launch aborts when it initializes the vLLM/FSDP
workers. The run never reaches model or data load. This is not an informative
run; the incompatibility is fully determined by the source above.

## Library fixes for an effortless verl-on-Trainium run

### verl (backend port)

1. Add a `neuron` device type to `verl.plugin.platform` / `verl/utils/device.py`
   (register the PrivateUse1 backend).
2. Add a Neuron collective backend (TorchNeuron distributed) beside
   `nccl`/`hccl`/`gloo`.
3. Register NeuronCore as a Ray resource and honor `NEURON_RT_VISIBLE_CORES`;
   replace the `GPU`/`NPU` resource math.
4. Add a Neuron rollout engine: integrate `vllm-neuron`, or add a
   transformers-native TorchNeuron rollout like the TRL leg's `model.generate`.
5. Port the FSDP2/DTensor worker to per-NeuronCore sharding, with a Neuron
   `param_offload` + `optimizer_offload` equivalent.
6. Remove CUDA-only calls in the actor/rollout workers: `torch.cuda.*` memory
   stats and streams, NCCL, CUDA graph capture, flash-attn.

### torch-neuronx / TorchNeuron (RL-specific fixes)

7. Give variable-length RL batches a stable shape. The backend is static-shape
   only. The TRL leg already hit on-core TDRAM exhaustion at step 40 from
   variable SDPA shapes (`TRAINIUM_ISSUES_LOG.md` #23); verl's variable
   prompt/response lengths will hit the same wall.
8. Lower dynamic `torch.repeat_interleave` in the Qwen3.5 vision tower (or fall
   back to CPU). Same neuronx-cc crash as the TRL leg (#1).
9. Ship `causal_conv1d` + `flash-linear-attention` (or a Neuron Gated DeltaNet
   kernel). Otherwise 24 of 32 layers fall back to reference PyTorch (#11).
10. Memory ergonomics: raise a clear OOM instead of an infinite recovery loop,
    make the trace cache reclaimable, and support a CPU-offload optimizer
    (verl relies on `param_offload` + `optimizer_offload`).

### vLLM-Neuron (rollout)

11. vLLM-Neuron (Beta) must serve the Qwen3.5 VLM (vision + Gated DeltaNet) and
    expose the exact APIs verl calls: `n`-way sampled generation at
    temperature 0.7, weight sync, `free_cache_engine`, and greedy eval.

## Recommendation

- Start with the gate (`trainium2/verl/scripts/gate_test.py`) to prove the
  platform plugin + device + runtime, then resolve the rollout (issue #1 in
  `trainium2/verl/VERL_TRAINIUM_ISSUES_LOG.md`) before attempting the full loop.
- Track `verl-project/verl` issues and the AWS Neuron docs for a formal
  Trainium release.
- For the benchmark now: TRL-on-TorchNeuron (works, this folder) is the
  Trainium datapoint; verl (NVIDIA) and Miles (NVIDIA) are the GPU datapoints.

## Sources

- verl device registry: <https://github.com/verl-project/verl/blob/main/verl/utils/device.py>
- verl README (hardware support): <https://github.com/verl-project/verl>
- TorchNeuron docs: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html>
- HF TorchNeuron blog: <https://huggingface.co/blog/hf-torchneuron>
- torch-neuronx (repo): <https://github.com/aws-neuron/torch-neuronx>
- vLLM-Neuron plugin: <https://github.com/vllm-project/vllm-neuron>
- vLLM Neuron release notes: <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/vllm-neuron.html>
- AWS verl-on-Trainium hiring signal: <https://www.amazon.jobs/en/jobs/10585967/forward-deployed-engineer-model-customization-and-optimization-aws-forward-deployed-engineering>
