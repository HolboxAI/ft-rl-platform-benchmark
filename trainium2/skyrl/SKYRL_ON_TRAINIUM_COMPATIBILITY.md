# SkyRL on Trainium2 (TorchNeuron) — compatibility assessment

Goal: run the same CholecT50 GRPO/RLVR experiment with **SkyRL (NovaSky-AI)**
instead of TRL/verl, and list what library changes a native run needs.

> **Source-verified 2026-10-26 against `NovaSky-AI/SkyRL` @ `69410e7` (main).**
> No Trainium instance was launched: the incompatibility is fully determined by
> source + dependency pins, so an instance run would only re-confirm the first
> `torch.cuda` failure.

## What SkyRL is (and is not)

SkyRL is a modular RL library from NovaSky (UC Berkeley Sky Computing Lab). The
benchmark's `skyrl/` README describes it as "built on verl" — that overstates
it. SkyRL **learns from** verl (and slime/OpenRLHF) and supports verl as one of
several backends via `skyrl-agent`, but `skyrl-train` (the framework the
CholecT50 leg uses) is a separate codebase with its own device, worker, and
inference layers.

Three training backends exist: `fsdp`, `megatron`, and `jax` (`skyrl-tx`). The
CholecT50 leg runs `fsdp` + vLLM.

## Verdict

**SkyRL does not run on Trainium2 out of the box, and is harder to port than
verl.** It has three stacked blockers — a CUDA dependency pin, a CUDA-hardcoded
device layer with no abstraction, and a vLLM-only rollout — whereas verl 0.9.0
had a pluggable `PlatformBase` device layer (so only the rollout was the hard
part).

| Dimension | TRL | verl 0.9.0 | SkyRL (main) |
|---|---|---|---|
| Device layer | native (torch-neuronx) | pluggable `PlatformBase` | none — `torch.cuda` hardcoded in 34 files |
| Rollout | `model.generate()` on neuron | vllm/sglang/trtllm (async), dead `HFRollout` | **vLLM only** |
| Weight sync | n/a (no server) | vLLM's own | `nccl`/`ipc` (CUDA) + `delta` (CUDA-free) |
| Non-CUDA training | — | — | `jax` (`skyrl-tx`), no Qwen3.5 VLM |

## The three blockers

### 1. Dependency wall (CUDA torch + vLLM pinned)

`pyproject.toml` hard-pins the training extras to CUDA builds:

- `fsdp`: `torch==2.13.0`, `vllm==0.31.0`, `vllm-router`, `nixl-cu13`.
- `megatron`: same + `transformer-engine`, `flash-mla +cu13`, `nvidia-*`.
- `jax`: `jax[cuda13]`, `jax[tpu]`.

On Trainium the only working torch is **torch-neuronx** (2.14.x). `pip install
skyrl[fsdp]` would install CUDA torch 2.13.0 + vLLM, clobbering the neuron
runtime. There is no `neuron` extra.

### 2. Device layer is CUDA-hardcoded, no abstraction

34 files touch `torch.cuda` (backends 30, benchmarks 2, train 2). There is no
`PlatformBase`-style hook. Representative call sites:

- `distributed/fsdp_strategy.py` → `torch.cuda.set_device(local_rank)`,
  `torch.cuda.current_device()`.
- `distributed/fsdp_utils.py` → `torch.cuda.current_device()` throughout.
- `distributed/megatron/fused_linear_logprob_triton.py` →
  `torch.cuda.get_device_capability()` (Triton kernel selection).
- `train/utils/utils.py` → `torch.cuda.is_available()` P2P gate.

verl 0.9.0 let us register a `neuron` platform as a clean plugin; SkyRL has no
equivalent, so the port is a ~34-file device refactor (plus real replacements
for CUDA-IPC weight sync and Triton capability dispatch).

### 3. Rollout is vLLM-only

`InferenceEngineConfig.backend: str = "vllm"` (docstring is literally
`"vllm"`). SGLang was dropped in v0.1.0. The dispatch raises for anything else:

```python
def get_sampling_params_for_backend(backend, sampling_params):
    if backend == "vllm":
        return get_vllm_sampling_params(sampling_params)
    raise ValueError(f"Unsupported generation backend: {backend}")
```

vLLM is CUDA-only. There is no transformers-native engine, and verl's dead
`HFRollout`/`NaiveRollout` has no SkyRL counterpart. A native path needs a new
inference engine (`model.generate()` on `neuron`) behind
`InferenceEngineInterface` + `backend="neuron"`.

## What is already device-agnostic (partial bright spots)

- **Weight sync `delta`** — checkpoint deltas via disk/`gs://`/`s3://`, no CUDA.
  This is a real CUDA-free transport, but it only matters once a rollout exists.
- **`jax` backend (`skyrl-tx`)** — JAX can target Trainium via JAX-XLA, the
  "native" non-CUDA path in principle. But it is a different runtime (not
  TorchNeuron native PyTorch), only ships `llama3` + RoPE in `skyrl/tx/models/`
  (no Qwen3.5 VLM), and pins `jax[cuda13]`/`jax[tpu]` rather than a neuron JAX.

## What a native run needs (package-side)

See `SKYRL_TRAINIUM_ISSUES_LOG.md` for the numbered list. In short:

1. Unpin CUDA torch/vLLM; add a `neuron` extra.
2. Introduce a device abstraction (or patch the 34 `torch.cuda` sites).
3. Add a transformers-native inference engine on `neuron`.
4. Make Triton capability dispatch + P2P + weight sync device-aware.
5. (If JAX is the target) add a Qwen3.5 VLM to `skyrl/tx/models/` + a neuron JAX extra.

Plus the same torch-neuronx/neuronx-cc runtime walls the TRL/verl legs hit
(TDRAM, GDN kernels, `repeat_interleave`, OOM ergonomics).

## Recommendation

For the benchmark now: SkyRL stays a GPU datapoint (it already ran on 4×A10G and
collapsed). On Trainium, TRL is the working datapoint and verl is the
close-to-native port; SkyRL is a larger port and should be tracked upstream
rather than attempted locally without a device-abstraction refactor.

## Sources

- <https://github.com/NovaSky-AI/SkyRL> (inspected @ `69410e7`)
- <https://pypi.org/project/skyrl-train/0.2.0/>
- <https://github.com/NovaSky-AI/SkyRL/releases> (v0.1.0 drops SGLang, v0.2.0 HTTP refactor)
- <https://arxiv.org/pdf/2511.16108v1.pdf> (SkyRL-Agent)
