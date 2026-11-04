# TitanRL (torchtitan/rl) on Trainium2 — compatibility assessment

Goal: run the same CholecT50 GRPO/RLVR experiment with **TitanRL** (TorchTitan's
RL stack) instead of TRL/verl/Miles/SkyRL, and list exactly what works and what
does not on AWS Trainium2 (TorchNeuron / torch-neuronx).

> **Source-verified against `pytorch/torchtitan` @ `69eac07e` (2026-11-02),
> then empirically probed on a `trn2.48xlarge` spot instance (2026-11-03).**
> The on-instance probe confirmed every CUDA wall below with exact error text.
> See `TITANRL_TRAINIUM_ISSUES_LOG.md` for the numbered issues and the exact
> failure messages. Every claim here cites a file:line in the checkout.

## TL;DR

TitanRL is split into three layers, and they are **not equally CUDA-bound**:

| Layer | On Trainium2 | Why |
|---|---|---|
| Trainer (`TrainingEngine`) | **Mostly works** | `torch.accelerator` + neuron-aware `tools/utils.py`; AWS already runs this for pre-training |
| Weight sync (TorchStore) | **Works (slow)** | default path is CPU-staging + `direct_rdma=False`, transport falls back to SHM/Gloo/RPC |
| Rollout (`VLLMEngine`) | **Does not work** | vLLM CUDA nightlies, CUDA-graph config, `has_cuda_capability(9,0)` gate, FA3/FA4/cuDNN/FlashInfer |

The rollout engine is the wall — same architectural blocker as verl/SkyRL/Miles
(vLLM/SGLang generation), which the SkyRL leg already proved must be bypassed
with a transformers-native `model.generate()` on the `neuron` device.

On top of that, the **CholecT50 experiment cannot even be expressed** in
TitanRL today: Qwen3.5 is listed "text only" for RL, and the rollout interface
takes `prompt_token_ids` (tokens, no image modality).

## What TitanRL is

Three-layer pipeline (`torchtitan/rl/README.md`):

1. **Experiment logic** — `Rollouter`, `MessageEnv`, rubric/rewards
   (`torchtitan/rl/rollout/`).
2. **Controller + dataflow** — `Monarch` actor framework, independent
   trainer/generator loops, a bounded off-policy work buffer
   (`torchtitan/rl/controller/`, `torchtitan/rl/actors/monarch/`).
3. **Distributed execution** — a TorchTitan `TrainingEngine` trainer mesh and a
   **vLLM generator** mesh, plus **TorchStore** for weight transfer
   (`torchtitan/rl/trainer/`, `torchtitan/rl/generator/`).

## What WILL work (device-agnostic or already neuron-aware)

### 1. The trainer is genuinely hardware-agnostic

`torchtitan/tools/utils.py` selects `device_type`/`device_module` at import and
already special-cases neuron:

- `tools/utils.py:74-80` — `get_device_info()` → `device_type, device_module`.
- `tools/utils.py:201-218` — `get_peak_flops("neuron")` returns `79e12 * 2` for
  trn2/trn3, i.e. Trainium BF16 TFLOPS is already in the MFU table.

The RL `Trainer` wraps the shared `TrainingEngine`
(`torchtitan/rl/trainer/trainer.py:124-129`) and uses `torch.accelerator` /
`engine.device`, not `torch.cuda`:

- `trainer.py:132` — `maybe_apply_numa_binding(engine.device.index, engine.device.type)`
  (early-returns for non-cuda, `distributed/offloading.py:18`).
- `trainer.py:201` — `torch.accelerator.current_stream(self.engine.device)`.

CUDA graphs go through `torch.accelerator.Graph()`
(`distributed/accelerator_graph.py:228`, `:309`), with a non-cuda fallback
(`:322-323`). This is the same surface AWS's "TorchTitan Neuron" beta validates
for pre-training on trn2/trn3.

### 2. Weight sync is transport-agnostic — and non-RDMA by default

This is the important difference from verl/OpenRLHF/SkyRL/Miles, which hard-code
NCCL / CUDA IPC / GPUDirect for weight sync.

TitanRL's trainer and generator both push/pull through TorchStore with
**`direct_rdma=False`** (the CPU-staging path), not the RDMA fast path:

- `trainer.py:426-453` — `push_model_state_dict` stages the model into reusable
  **pinned CPU** buffers, then `ts.put_state_dict(..., direct_rdma=False)`
  (`trainer.py:481-486`).
- `generator.py:442-452` — `prefetch_model_state_dict` does
  `ts.get_state_dict(..., direct_rdma=False)`.
- `actors/monarch/system.py:192-196` — StorageVolumes are spawned on the trainer
  mesh "for faster data access in the **non-RDMA path**".

TorchStore's transport ladder is SHM → RDMA → Gloo → RPC, with a pure-CPU base
install. So the weight-sync layer would **run** on neuron (Gloo/RPC fallback),
just at the full-model CPU-round-trip cost every step — the "slow" path. The
RDMA fast path is a CUDA/GPUDirect optimization that is **not** required and not
even the default here.

> **Empirically (2026-11-03):** this "portable" layer is still blocked by its
> *install path* — the Meta TorchStore/Monarch packages are git-only, and the
> `torchstore`/`torchmonarch` names on PyPI are unrelated 0.1.x packages, so a
> plain `pip install` raises `ResolutionImpossible` / "No matching distribution".
> See `TITANRL_TRAINIUM_ISSUES_LOG.md` #1.

## What will NOT work (CUDA/NVIDIA hard-coded)

### 1. Rollout: `VLLMEngine` is the only shipped engine, and it is CUDA-bound

`torchtitan/rl/generator/engine.py` defines a clean `GenerationEngine` ABC
(an extension point), but the only implementation is vLLM:

- `generator.py:41-45` — `_default_engine_config()` returns `VLLMEngine.Config()`,
  imported "only when it is the configured engine".
- `generator/vllm/engine.py:50` — `from torchtitan.tools.utils import has_cuda_capability`.
- `generator/vllm/engine.py:283` — `if not has_cuda_capability(9, 0): ...` (Hopper+ gate).
- `generator/vllm/engine.py:98` — `VLLMCudaGraphConfig`; `:250`
  `enforce_eager=self.config.cuda_graph.mode == "NONE"`.

The vLLM subpackage (`model_wrapper.py`, `attention.py`, `gdn.py`) does paged
attention, CUDA-graph capture, and DTensor resharding for CUDA graphs, with
direct `torch.cuda` calls.

**Install is CUDA-bound too** (`torchtitan/rl/README.md:214-223`):
`scripts/install_rl_env.sh` installs "matching PyTorch, torchvision and vLLM
CUDA nightlies", "FA4 on Blackwell / FA3 on Hopper", `nvcc` for FlashInfer JIT,
and `attn-gym[linear,cudnn]` for the GDN cuDNN chunk kernels. Pre-built vLLM
wheels are CUDA 13.2.

### 2. Gated DeltaNet is a CUDA custom op — the Qwen3.5 wall

`torchtitan/models/common/attention/gdn.py:87`:

```
@torch.library.custom_op("torchtitan::recurrent_gdn_fwd", mutates_args=(), device_types="cuda")
```

Qwen3.5 is 24 Gated-DeltaNet layers (8 full-attn + 24 GDN + vision tower for the
4B). On neuron this CUDA kernel has no counterpart, so GDN must fall back to a
reference/eager implementation — **the exact same wall the TRL and SkyRL legs
hit** (`causal_conv1d` / `flash-linear-attention` are CUDA-only; see
`TRAINIUM_ISSUES_LOG.md`). On Blackwell the GDN runs cuDNN chunk kernels
(`rl/README.md:223`), also NVIDIA-only.

### 3. CholecT50's VLM modality is not representable in the rollout

Two independent facts:

- `torchtitan/rl/README.md:36` — "Qwen 3.5/3.6/3.8 | Supported (**text only**)".
- The rollout/generator interface is token-only: `generator.py:341-399`
  `generate(prompt_token_ids: list[int], ...)`.
- `examples/verifiers/generation_server.py:201` — `raise ValueError("multimodal
  features are not supported")`.

The CholecT50 experiment is surgical-VQA over image frames (base64-image JSONL).
Even on NVIDIA, TitanRL's RL loop would need a vision branch in the rollout +
generator + tokenizer path before it could ingest a frame. That is a model/data
pipeline gap **on top of** the device gap.

### 4. Smaller CUDA assumptions in the RL path

- `torchtitan/rl/_env.py:16-23` — `RL_ENV_DEFAULTS` hard-sets
  `PYTORCH_CUDA_ALLOC_CONF` and `NCCL_DEBUG=WARN`. Harmless no-ops on neuron, but
  a CUDA/NCCL assumption baked into the RL defaults.
- `trainer.py:516-527` — staging buffers use `pin_memory=True` (CUDA page-locked
  host memory). Unsupported/no-op on neuron; needs a device guard.
- `trainer.py:157-161, 470-473` — weight staging uses `torch.Stream(device=...)`
  + `torch.Event().record(stream).synchronize()`. torch-neuronx stream/event
  support is limited (the TRL/SkyRL legs already fought this).
- `distributed/deepep/hybridep.py:119,231,263,296` — DeepEP/HybridEP custom ops
  are `device_types="cuda"`. **Not relevant to CholecT50** (Qwen3.5-4B is dense;
  DeepEP is MoE-only).

## What a native CholecT50 run needs

In order of size, mirroring the SkyRL `NeuronInferenceEngine` port
(`susmit-holbox/SkyRL#2592`):

1. **A neuron `GenerationEngine`.** Implement the `GenerationEngine` ABC
   (`generator/engine.py`) with transformers `model.generate()` on the `neuron`
   device, and select it via `Generator.Config.engine` instead of
   `VLLMEngine.Config`. This is the direct analogue of SkyRL's
   `NeuronInferenceEngine` — the generator already separates engine selection by
   config type, so no core rewrite.
2. **A vision branch in the rollout.** Extend `Rollouter`/`MessageEnv`/the
   `prompt_token_ids` interface to carry image frames (pixel values + `grid_thw`)
   for the qwen3_5 VLM. This is a feature TitanRL lacks even on NVIDIA.
3. **A neuron GDN fallback.** Ship an eager/reference path for the
   `torchtitan::recurrent_gdn_fwd` CUDA op (or a Neuron GDN kernel), else the 24
   GDN layers fall back to slow single-core PyTorch.
4. **Guard the CUDA staging bits.** Gate `pin_memory`, `torch.Stream`/`Event`
   staging, and the `_env.py` CUDA/NCCL defaults on `device_type == "cuda"`.

## Ranking the Trainium legs (closeness to native, updated)

1. **TRL** — works natively (`model.generate()` on neuron); already run.
2. **Miles** — device abstraction + TPU precedent + custom-generate hatch.
3. **SkyRL** — ported (native neuron engine, PR #2592); LoRA weight-sync green.
4. **verl** — device plugin works, rollout async-server-only.
5. **TitanRL** — trainer + weight-sync are the most portable of the CUDA group,
   but the **rollout is vLLM-only** *and* **VLM is text-only**, so two blockers
   stack where the others have one.

## Sources

- `torchtitan/torchtitan/rl/README.md` (model matrix, install, weight sync)
- `torchtitan/torchtitan/rl/generator/{engine,generator}.py`
- `torchtitan/torchtitan/rl/generator/vllm/engine.py`
- `torchtitan/torchtitan/rl/trainer/trainer.py`
- `torchtitan/torchtitan/rl/actors/monarch/system.py`
- `torchtitan/torchtitan/tools/utils.py` (neuron device/flops)
- `torchtitan/torchtitan/distributed/accelerator_graph.py`
- `torchtitan/torchtitan/models/common/attention/gdn.py`
- `torchtitan/torchtitan/rl/_env.py`
- `torchtitan/.ci/docker/requirements-rl.txt`
