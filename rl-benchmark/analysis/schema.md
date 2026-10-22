# Metrics schema

Each CholecT50 run writes `final_results.json` in the schema of
`ft-rl-platform-benchmark-main 2/analysis/schema.md`, so the two repos join on
`experiment` + the shared field names. This repo adds one block, `accelerator`,
because here the hardware is what is being compared.

```json
{
  "platform": "trainium | torchtpu-torchtpu | torchtpu-torchtpu-compile | torchtpu-torchxla | skyrl-gpu-fsdp-<preset> | skyrl-tpu-jax",
  "generated_utc": "2026-10-22T10:01:03Z",
  "experiments": {
    "cholect50_rl": {
      "experiment": "cholect50_rl",
      "method": "grpo",
      "job_id": "cholect50-rl-20261022-100103",
      "state": "COMPLETED",
      "base_model": "/home/ubuntu/models/cholec-sft-qwen3.5-4b",
      "hf_model_id": null,
      "lora_rank": 16,
      "learning_rate": 1e-5,
      "kl_beta": 0.04,
      "temperature": 0.7,
      "n_samples_per_prompt": 8,
      "rollout_batch_size": 16,
      "max_output_tokens": 256,
      "max_prompt_tokens": 2048,
      "steps": 16,
      "start_score": 0.33, "final_score": 0.52,
      "score_curve": [0.33, "..."],
      "start_triplet_f1": 0.13, "final_triplet_f1": 0.40, "triplet_f1_curve": ["..."],
      "start_val_reward": 0.32, "final_val_reward": 0.60,
      "eval_curve": [{"step": 0, "reward": 0.32, "triplet_f1": 0.13, "parse_ok": 0.97}, "..."],
      "wall_time_s": 3600.0,
      "estimated_cost_usd": 1.34,
      "accelerator": {
        "backend": "neuron | tpu | tpu-xla | cuda | cpu",
        "device": "xla:0",
        "torch": "2.x", "torch_xla": "2.x", "torch_neuronx": "2.x", "torch_tpu": "x.y",
        "instance_type": "trn1.2xlarge | v6e-1",
        "hourly_usd": 1.34,
        "gen_impl": "static | hf",
        "dtype": "bf16", "autocast_bf16": false,
        "model_load_s": 60.0,
        "first_step_s": 900.0,
        "steady_step_s": 120.0,
        "gen_tok_per_s_median": 300.0,
        "peak_mem_gb": null
      }
    }
  },
  "findings": ["<free-form strings: failures, workarounds, version pins>"]
}
```

## Field notes

- **Two reward measurements.** `score_curve` is the per-step rollout reward
  (16 frames × 8 samples, sampled). `start_val_reward` / `final_val_reward` are
  greedy decoding on the 32 val frames before and after training: the number
  the verl / Miles / NeMo-RL runs report. Compare val to val.
- **`first_step_s` vs `steady_step_s`** is the compile overhead. `steady_step_s`
  is the median of steps 3+. Note whether the compile cache was cold or warm in
  the platform's config file.
- **`estimated_cost_usd`** = `hourly_usd` × `wall_time_s` / 3600. It excludes
  model download and setup time. Use on-demand list prices and say which.
- **`peak_mem_gb`** is `null` on XLA backends (no portable allocator query).
  Read device memory from `neuron-top` (Trainium) or the TPU runtime metrics and
  record it in `findings`.
- Per-step detail (timings, response length, truncation, KL, grad norm) is in
  the matching `*_steps.jsonl` next to each results file.
