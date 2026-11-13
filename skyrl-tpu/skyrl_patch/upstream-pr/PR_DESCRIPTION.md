# What does this PR do?

Lets the FSDP2 trainer (`skyrl-train`) run on Cloud TPU through TorchTPU (`torch_tpu`), with rollouts from external servers. Proposed in #2553.

TLDR: an accelerator module selected by `SKYRL_DEVICE` (`cuda` by default, unchanged behavior) replaces the direct `torch.cuda` calls in the FSDP workers; a `reload` weight-sync backend exports the checkpoint and has an agent restart each external server on it; render `kwargs_data` can be decoded without vLLM; flash-attn's padding helpers are vendored so `model_wrapper` imports without flash-attn.

## Why it is needed

On a TPU host the FSDP path fails at the first device call:

- the workers call `torch.cuda.set_device`, `torch.cuda.synchronize`, `torch.cuda.empty_cache`, `torch.cuda.mem_get_info`, CUDA RNG functions, and `torch.autocast(device_type="cuda")` directly;
- process groups are created with NCCL;
- every weight-sync backend needs vLLM's transfer API on the receiving side, and the `nccl` / `ipc` transports need NCCL or CUDA IPC;
- `model_wrapper` imports `flash_attn.bert_padding` at module scope, and flash-attn does not install off CUDA;
- decoding render `kwargs_data` requires vLLM's serde in the trainer.

TorchTPU 0.2 additionally rejects some operations the FSDP2 / Qwen3.5 path uses:

- `fully_shard` moves CPU parameters with `tensor.data = tensor.to(device)`, which fails with `Attempted to call variable.set_data(tensor), but variable and tensor have incompatible tensor type`;
- Qwen3.5 interleaved M-RoPE writes in place into a view of the same tensor, which aborts at `inversion.cc:112`;
- `Qwen3_5VisionRotaryEmbedding.inv_freq` can stay on CPU under FSDP, so the vision forward mixes `cpu` and `tpu:0` tensors;
- with backend autoload on, every process that imports torch (DataLoader workers included) opens a TPU chip.

## Usage

```bash
export SKYRL_DEVICE=tpu TORCH_DEVICE_BACKEND_AUTOLOAD=0
# per actor group, the TorchTPU topology and the per-rank chip:
export SKYRL_TPU_GROUP_ENV_POLICY='{"env": {"TORCH_TPU_TOPOLOGY": "2,2,1", "TORCH_TPU_SLICEBUILDER_ADDRESSES": "..."}, "per_rank": [{"TPU_VISIBLE_DEVICES": "0"}, ...]}'

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  trainer.strategy=fsdp \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.external_server_urls='["http://rollout:8000"]' \
  generator.inference_engine.weight_sync_backend=reload ...
```

`reload` configuration is read from the environment on the policy workers: `SKYRL_RELOAD_AGENT_URLS` (default: each server URL with port + 1), `SKYRL_RELOAD_TRANSPORT` (`http` uploads the files; `path` sends a directory the agent can read), `SKYRL_RELOAD_EXPORT_DIR`, `SKYRL_RELOAD_TIMEOUT_S`, `SKYRL_RELOAD_SKIP_INITIAL`.

Agent protocol: `PUT /v1/files/<version>/<name>` per file, then `POST /v1/reload {"version": v, "files": [...]}`; or, for `path`, `POST /v1/reload {"version": v, "model_dir": d}`. The agent answers once its server serves the new weights.

## How it works

- `skyrl/backends/skyrl_train/utils/device.py`: `device_type()` reads `SKYRL_DEVICE` (`cuda` | `tpu`). Each function (`synchronize`, `empty_cache`, `current_device`, `mem_get_info`, `manual_seed_all`, RNG state, `autocast`, `dist_backend`) calls the CUDA function when the device is `cuda`, and `torch.accelerator` / TorchTPU otherwise. `dist_backend()` returns `cpu:gloo,cuda:nccl` or `cpu:gloo,tpu:tpu_dist`. `configure_process_env()` applies the per-group TorchTPU environment before the first TPU op and installs the TorchTPU workarounds. All of them are no-ops when the device is `cuda`.
- TorchTPU workarounds (TPU only): incompatible tensors in FSDP2's `_move_states_to_device` go through `torch.utils.swap_tensors`, the path FSDP2 already uses for wrapper subclasses. The Qwen3.5 interleave is rebuilt with `torch.where` (identical values; covers both `apply_interleaved_mrope` and the newer `recomposition_frequencies`). The vision RoPE `forward` is wrapped so `inv_freq` follows the input's device.
- `weight_sync/reload.py`: every policy rank joins the full-state-dict gather; rank 0 writes HF safetensors in the inference dtype and publishes to all agents concurrently. Validation requires `trainer.strategy=fsdp` and `run_engines_locally=false`; LoRA is rejected.
- `renderer.decode_mm_kwargs`: items prefixed with `st:` are base64 safetensors holding `pixel_values` and `image_grid_thw`, decoded without importing vLLM.
- `utils/bert_padding.py`: `pad_input` / `unpad_input` vendored from flash-attn (BSD-3-Clause, attribution in the file header).

Alternatives considered:

- `delta` weight sync also avoids NCCL, but its receive side is a vLLM weight-transfer engine, so it does not help with engines that are not vLLM.
- vLLM-TPU (`tpu-inference`) as the rollout engine: its Qwen3.5 vision path was too slow for RL rollouts, so the external server used HF `generate` on TorchTPU instead.
- Running the reference model on 2 chips: TorchTPU process groups of 2 chips fail with `START_SESSION failed` on v6e-8; 1 and 4 chips work.

## Constraints

- No TPU CI. The TPU path is covered by CPU tests (selection, no-op off TPU, equivalence of the replaced Qwen3.5 functions, `reload` protocol), not by hardware tests.
- The TorchTPU workarounds patch FSDP2 internals and transformers classes at runtime and should be removed once fixed upstream.
- Port 8470 on Cloud TPU VMs is held by the TPU VM agent; TorchTPU slice-builder addresses must use other ports.
- `reload` needs an agent next to each external server; no agent is included here.

## Test plan

- `tests/backends/skyrl_train/utils/test_device.py`: `cuda` default and backend string, `tpu` selection, `SKYRL_DIST_BACKEND` override, invalid value, all TPU patches are no-ops off TPU, the replacement interleave equals the installed transformers', and the per-rank TorchTPU environment is applied.
- `tests/backends/skyrl_train/utils/test_bert_padding.py`: unpad → pad round trip and `cu_seqlens`.
- `tests/backends/skyrl_train/weight_sync/test_reload.py`: agent URL default, `from_env` validation, `http` upload + reload call and `path` reload call against a local HTTP agent, the skipped initial sync.
- `tests/backends/skyrl_train/test_renderer.py`: `st:` items decode with vLLM's serde module blocked.
- `tests/train/test_config.py`: `reload` rejects `run_engines_locally=true` and non-FSDP strategies.

## Validation

Every step of `.github/workflows/cpu_skyrl.yaml`, with its exact commands, on a clean Linux VM (Ubuntu 22.04, e2-standard-4), with this change applied to `main` at `dd8769f8`. The branch is rebased onto `e4fdd8c1` (no conflicts):

| CI step | Command | Result |
|---|---|---|
| `check_code_quality` | `pre-commit run --all-files --config .pre-commit-config.yaml` | ruff, black, gitleaks: passed |
| cpu tests (without vllm or megatron) | `uv run --isolated --extra skyrl-train --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu -m "not (vllm or megatron)"` | 2514 passed, 41 skipped |
| cpu tests (with vllm) | `uv run --isolated --extra fsdp --extra dev pytest ... -m "vllm and not megatron"` | 433 passed, 13 skipped |
| cpu tests (with megatron) | `uv run --isolated --extra megatron --extra dev pytest ... -m "megatron"` | 113 passed |
| tinker tests | `uv run --isolated --extra fsdp --extra tinker --extra dev pytest tests/tinker/skyrl_train/` | 93 passed, 22 skipped |
| `skyrl_tinker_tests` | `uv run --isolated --extra tinker --extra jax --extra dev pytest --forked -s tests/tinker tests/utils --ignore=tests/tinker/skyrl_train` | 144 passed |
| `skyrl_gym_tests` | `uv sync --frozen --extra dev && uv run --frozen pytest tests/` (in `skyrl-gym/`) | 119 passed |

No test fails. The new tests (`test_device.py`, `test_bert_padding.py`, `test_reload.py`, the renderer and config tests) are in the first step.

<!-- CURSOR_SUMMARY -->
---

> [!NOTE]
> **Medium Risk**
> Touches distributed FSDP training, weight sync, and runtime monkey-patches on TPU; CUDA defaults are preserved but reload and export/restart are new failure modes for production rollouts.
> 
> **Overview**
> Adds **Cloud TPU / TorchTPU** support for the FSDP2 trainer by routing device, distributed, and autocast calls through a new **`SKYRL_DEVICE`-controlled accelerator module** (`cuda` default, unchanged GPU behavior). Workers configure per-group TPU topology before the backend loads, use **`cpu:gloo,tpu:tpu_dist`**, and apply **TPU-only runtime fixes** for FSDP2 parameter moves and Qwen3.5 M-RoPE / vision RoPE incompatibilities.
> 
> Introduces a **`reload` weight-sync backend** for external rollout servers that cannot use vLLM’s in-place transfer API: each step **gathers an HF safetensors export**, rank 0 **uploads or points agents at it**, and servers **restart on the new weights** (FSDP + external servers only; LoRA rejected). Dispatch skips pause/resume for this path; policy workers use a **long-timeout gloo barrier** while rank 0 publishes.
> 
> **VLM / non-CUDA trainer conveniences:** render multimodal kwargs can use an **`st:` safetensors prefix** decoded without vLLM; **`pad_input` / `unpad_input`** are vendored so `model_wrapper` no longer hard-requires flash-attn. Ray runtime env forwards TPU-related variables and sets **`TORCH_DEVICE_BACKEND_AUTOLOAD=0`** to avoid DataLoader workers grabbing chips.
> 
> <sup>Reviewed by [Cursor Bugbot](https://cursor.com/bugbot) for commit 923ca875d8df0f457543313c78968f0748229f6d. Bugbot is set up for automated code reviews on this repo. Configure [here](https://www.cursor.com/dashboard/bugbot).</sup>
<!-- /CURSOR_SUMMARY -->
