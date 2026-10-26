# verl-tpu — status

CholecT50 LoRA GRPO on verl v0.9.1 + Cloud TPU v6e-8 (TorchTPU).

## Working (verified on the live v6e-8)

- `PlatformTPU` plugin — `tpu` device, mRoPE/vision-RoPE patches, FSDP2 `_move_states_to_device` patch.
- FSDP2 engine re-registered for `tpu` (in the plugin entry point).
- `trainer.use_v1=False` — V0 trainer (V1 hard-requires `transfer_queue`, Ascend-only).
- `attn_implementation=sdpa` — patched into the checkpoint `config.json` (flash-attn is CUDA-only).
- `torch.accelerator.mem_get_info` shim (TorchTPU exposes `get_memory_info`).
- Data load → model load (10 GB Qwen3.5-4B + LoRA r=16, FSDP2).

## Last blocker — async HF rollout

verl v0.9.1 is async-only (`rollout.mode=async`; "sync mode is deprecated" in
`ray_trainer.py`). The HF rollout is sync-only and not in `_ROLLOUT_REGISTRY`,
so `rollout.name=hf` fails with `Rollout hf with mode async not found`.

To finish: write a custom async `ServerAdapter` for the HF engine —
`resume()`/`release()`, `update_weights()` via the `/dev/shm` reload transport,
`generate_sequences()` via HF `model.generate` — reusing the `rollout/` server
code already in this repo (ported from the SkyRL TPU run).

## Layout

- `verl/plugin/verl_tpu_plugin/` — PlatformTPU plugin + engine re-registration.
- `scripts/train.sh` — launch config (HF rollout, LoRA r=16, fsdp2, sdpa).
- `scripts/launch_smoke.sh` — single-chip smoke (`n_gpus_per_node=1`, 1 step).
- `probes/` — device (`check_device.py`) + FSDP2 (`p0_fsdp2.py`) checks.
- `rollout/` — SkyRL TPU rollout server + supervisor (base for the verl adapter).
- `patches/` — the three Qwen3.5 mRoPE/tensordict patches.
- `setup/setup_trainer_vm.sh` — TorchTPU + verl install.
