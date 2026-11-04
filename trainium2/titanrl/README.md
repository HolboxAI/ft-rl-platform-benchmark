# TitanRL (torchtitan/rl) on Trainium2 — CholecT50 GRPO/RLVR leg

Same CholecT50 surgical action-triplet experiment as the TRL, verl, SkyRL, and
Miles legs, but with **TitanRL** (TorchTitan's RL stack). Goal: determine
compatibility and list exactly what blocks a run.

## What is here

- `TITANRL_ON_TRAINIUM_COMPATIBILITY.md` — the compatibility assessment
  (trainer, weight-sync, rollout, VLM modality), source-verified and empirically
  probed.
- `TITANRL_TRAINIUM_ISSUES_LOG.md` — the numbered issues with exact error text.
- `scripts/setup_probe.sh` — runs **on the instance**. Clones torchtitan, probes
  the CUDA dependency installs and the vLLM device/import paths, writes
  `~/titanrl_probe.log`.
- `scripts/deploy_probe.sh` — runs **on the workstation**. Waits for SSH, scps
  `setup_probe.sh`, launches it detached.
- `scripts/poll_probe.sh` — runs **on the workstation**. Polls the probe to a
  terminal state, pulls the log home, terminates the spot instance.
- `logs/titanrl_probe.log` — the captured probe output (cleaned; the raw log had
  a duplicated tail from the double redirect in the harness).

## Status (2026-11-03, probed on a `trn2.48xlarge`)

**Not runnable.** TitanRL splits into three layers; two are portable and one is
not:

| Layer | On Trainium2 |
|---|---|
| Trainer (`TrainingEngine`) | mostly works (neuron-aware, `torch.accelerator`) |
| Weight sync (TorchStore) | works in code (non-RDMA CPU path), but its deps are not installable |
| Rollout (`VLLMEngine`) | does not work (CUDA-bound, `has_cuda_capability(9,0)` gate) |

The probe confirmed six CUDA/install walls. The decisive ones:

1. `torchstore` / `torchmonarch` are git-only Meta packages; the PyPI names are
   unrelated, so `pip install` raises `ResolutionImpossible`.
2. vLLM cannot infer a device on neuron (`DeviceConfig("auto")` →
   `RuntimeError: Failed to infer device type`; `current_platform =
   UnspecifiedPlatform`).
3. `pip install vllm` pulls the full CUDA toolchain (cuda-toolkit 13.0.3,
   nvidia-nccl-cu13, flashinfer).
4. `flash-attn` source build fails (`ModuleNotFoundError: No module named
   'torch'`).

On top of the device walls, the CholecT50 experiment cannot be expressed in
TitanRL today: Qwen3.5 is listed "text only" for RL, and the rollout interface
takes `prompt_token_ids` (no image modality).

## Reproduction

The probe scripts assume the workstation-side files in a single directory with a
`instance_titanrl.txt` holding `instance_id=` and `public_ip=` lines. The
workstation paths are hard-coded for the original author's machine; adjust
`BASE` and `KEY` before reuse.

1. Launch a `trn2.48xlarge` spot in `ap-south-2` (keypair
   `trainium-ap-south-2`). Write the instance id and public IP to
   `instance_titanrl.txt`.
2. `scripts/deploy_probe.sh` — wait for SSH, scp `setup_probe.sh`, launch it.
3. `scripts/poll_probe.sh` — poll to `TITANRL_PROBE_COMPLETE`, pull the log, stop
   the instance.

The probe runs in an isolated venv and does not touch the DLAMI torch-neuronx
environment.

## Order of operations (what a native run would need)

1. A neuron `GenerationEngine` (the `GenerationEngine` ABC is the extension
   point; implement it with transformers `model.generate()` on `neuron`).
2. A vision branch in the rollout (TitanRL is token-only even on NVIDIA).
3. A neuron GDN fallback for the 24 GatedDeltaNet layers.
4. Guard the CUDA staging bits (`pin_memory`, `torch.Stream`/`Event`, `_env.py`
   CUDA/NCCL defaults) on `device_type == "cuda"`.

See `TITANRL_ON_TRAINIUM_COMPATIBILITY.md` for the full detail.
