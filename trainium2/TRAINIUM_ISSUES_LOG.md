# Trainium2 — CholecT50 GRPO (Experiment A) issues log

Purpose: a running, structured record of every error hit while bringing TRL GRPO
(RLVR) on the CholecT50 surgical-triplet task to a completed run on Trainium2
(`trn2.48xlarge`, TorchNeuron). Each entry is written so a library maintainer
can reproduce and fix it natively.

Stack (verified working pins): torch 2.14.0+cu130, torch-neuronx 2.14.3.0.282,
transformers 5.18.0, trl 1.16.0, accelerate 1.15.0, torchvision 0.29.1, python 3.14.

Status legend: `FIXED` (code workaround in-repo), `OPEN` (no workaround yet),
`NATIVE-FIX` (needs upstream change).

---

## 1. `torch.repeat_interleave` with dynamic `repeats` crashes neuronx-cc
- **Status:** FIXED (monkeypatch in `grpo_train.py:_patch_vision_interp`).
- **Where:** `transformers.vision_utils.get_vision_interpolation_indices_and_weights`
  (called by the Qwen3.5 vision tower) computes `torch.repeat_interleave(grid_thw[:, 1], counts)`
  with a **dynamic** `counts` tensor.
- **Error (compile path):** `BIRCodeGenLoop error: tuple index out of range` inside
  neuronx-cc when lowering the dynamic-repeat op to BIR.
- **Error (eager path):** the op silently falls back to CPU in eager mode (a sync,
  not a failure).
- **Root cause:** neuronx-cc (build 282) cannot lower `torch.repeat_interleave`
  with a tensor-valued `repeats`. The earlier AMI build (2.14.3.0.1) crashed the
  StableHLO `concatenate` lowering entirely.
- **Workaround:** compute the small `(total_thw, n_taps)` index/weight tensors on
  CPU and move them back to the device. Imported at module level in
  `modeling_qwen3_5`, so the patch must target `modeling_qwen3_5`, not `vision_utils`.
- **Native fix:** neuronx-cc should either lower dynamic `repeat_interleave` or emit a
  CPU fallback under `torch.compile` instead of a BIR codegen crash.

## 2. `torch_compile_backend` forces compile even when `torch_compile=False`
- **Status:** FIXED (gate the backend arg).
- **Where:** `transformers.TrainingArguments` / Trainer init.
- **Error:** no visible error; model compiled anyway (`torch._dynamo.is_compiling()` True)
  and hit issue #1 + an OOM.
- **Root cause:** transformers always builds a `TorchDynamoPlugin` when
  `torch_compile_backend` is non-None (accelerate >= 1.2.0). A non-None backend
  therefore compiles the model even with `torch_compile=False`.
- **Workaround:** set `torch_compile_backend = "neuron" if TORCH_COMPILE else None`.
- **Native fix:** warn loudly (or refuse) when `torch_compile=False` but a backend is set.

## 3. `max_prompt_length` removed in trl 1.16.0 `GRPOConfig`
- **Status:** FIXED (dropped the key).
- **Error:** `AttributeError` / `TypeError` at config construction.
- **Root cause:** trl 1.16.0 removed `max_prompt_length` (only `max_completion_length`
  remains; prompt length is derived from the tokenizer/processor).
- **Native fix:** document the removal in the trl changelog; the 1.16 upgrade guide
  is silent on it.

## 4. `generation_batch_size` must be divisible by `num_generations`
- **Status:** FIXED (set both to 2 in smoke mode, 8 otherwise).
- **Error:** `ValueError: generation_batch_size (1) must be divisible by num_generations (2)`.
- **Root cause:** trl 1.16 validates this invariant at config build time.

## 5. `torch_dtype` deprecated in `from_pretrained`
- **Status:** FIXED (`dtype=torch.bfloat16`).
- **Root cause:** transformers 5.x renamed the kwarg.

## 6. `AutoModelForConditionalGeneration` gone in transformers 5.x
- **Status:** FIXED (`AutoModelForImageTextToText`).
- **Root cause:** qwen3_5 is a VLM; transformers 5.x removed the legacy auto class.
- **Note:** the checkpoint `config.json` says `transformers_version: 4.57.0.dev0`,
  which is stale and misleads dependency resolution.

## 7. TRL passes conversational completions as message lists, not strings
- **Status:** FIXED (`reward_tr.py:_completion_text`).
- **Where:** `GRPOTrainer._calculate_rewards` → user reward func.
- **Error:** `TypeError: expected string or bytes-like object, got 'list'` in the
  reward's `re.sub`/`.strip`.
- **Root cause:** with a conversational prompt, trl 1.16 passes each completion as
  `[{"role": "assistant", "content": text}]`, not a bare string. The reward must
  unwrap `msg["content"]`.
- **Native fix:** document this in trl's reward-function contract; it is a silent
  breaking change vs. the string-completion contract older trl documented.

## 8. Single-core HBM OOM during the GRPO training step (blocking)
- **Status:** OPEN — this is the current blocker.
- **Where:** `_get_per_token_logps_and_entropies` → `model(..., fused_lm_head=True)`,
  i.e. the old-policy logprob forward, the first heavy forward after generation.
- **Error:** `NRT Execution error … neuron::alloc::lazy … Failed to allocate resource`,
  then `Failed to allocate DEVICE memory (67108864 bytes)`.
- **Numbers:** `total_hbm = 25.77 GB`, `allocated_bytes.peak ≈ 22.6 GB`,
  `largest_cached_free = 35 MB` (so a 64 MB request fails despite ~12 GB "free").
- **Root cause (two compounding facts):**
  1. trl GRPO with `beta != 0` loads a **second full copy** of the model as the
     reference model (`grpo_trainer.py:1003`). Model = 9.9 GB bf16 (~4.95 B params:
     hidden 2560 × 32 layers, vocab 248320). Policy + ref ≈ 20 GB before activations.
  2. Full fine-tune AdamW keeps fp32 `m`/`v` states ≈ 2 × 4.95 B × 4 B ≈ 40 GB, which
     cannot fit even with the ref model removed.
- **Reference comparison:** the verl reference runs FSDP(size 4) with
  `param_offload=True` + `optimizer_offload=True` (params + optimizer on CPU). TRL on
  Neuron has no equivalent offload today.
- **Planned workaround:** (a) `beta=0.0` to skip the ref model; (b) LoRA (PEFT) so the
  optimizer states are tiny. Full fine-tune will need either tensor parallelism
  (NeuronX Distributed) or a CPU-offload optimizer — both open.
- **Native fixes:** torch_neuronx should (a) surface a clearer OOM with the stage,
  and (b) reduce the NKI trace-cache / reserved-memory footprint (see #9, #10).

## 9. OOM recovery loops forever instead of raising
- **Status:** NATIVE-FIX (torch_neuronx).
- **Error:** `OOM recovery failed after shelves cleanup + cache drain` repeats forever;
  no Python exception is raised, so the trainer neither crashes nor progresses.
- **Impact:** a hung process has to be killed manually (`pkill`).
- **Native fix:** after N failed OOM-recovery attempts, raise `torch.OutOfMemoryError`
  so the trainer can unwind instead of spinning.

## 10. NKI trace cache holds ~10 GB of HBM and is not reclaimed on OOM
- **Status:** OPEN (mitigation: `NKI_ENABLE_TRACE_CACHE=0`).
- **Evidence:** `cached_free ≈ 10.5 GB` at OOM; `largest_cached_free = 35 MB`.
- **Native fix:** make the trace cache reclaimable under memory pressure, and document
  the HBM cost of `NKI_ENABLE_TRACE_CACHE=1` (the default).

## 11. Gated DeltaNet / linear-attention kernels fall back to reference PyTorch
- **Status:** OPEN (performance only, not correctness).
- **Warnings:** `causal_conv1d_fn`, `chunk_gated_delta_rule`, `causal_conv1d_update`,
  `fused_recurrent_gated_delta_rule` all "falling back to reference PyTorch because
  `causal_conv1d` / `flash-linear-attention` is not installed".
- **Impact:** the Gated DeltaNet LLM (24 of 32 layers) runs much slower than native.
- **Native fix:** ship/allow the `causal_conv1d` and `flash-linear-attention` kernels
  on the Neuron AMI (or a Neuron-equivalent fused GDN kernel).

## 12. `model_init_kwargs` ignored when the model is already instantiated
- **Status:** FIXED (benign; the model is created explicitly with `dtype=bfloat16`).
- **Error:** `[RANK 0] You passed model_init_kwargs to GRPOConfig, but your model is
  already instantiated … ignored`.

## 13. Flaky SSH to the spot instance (banner-exchange timeouts)
- **Status:** WORKAROUND (retry loops; `ServerAliveInterval`).
- **Error:** `Connection timed out during banner exchange` on ~every other connect.
- **Root cause:** unknown; correlates with the stuck OOM-retry loop consuming the box.
- **Impact:** scripts must retry `scp`/`ssh`.

## 14. Eval phase OOM after a successful training step
- **Status:** FIXED (verified) — `per_device_eval_batch_size=1`.
- **Where:** `trainer.evaluate()` (the final greedy val eval), after `trainer.train()`
  completed the step.
- **Error:** `Failed to allocate DEVICE memory (285212672 bytes)` → infinite
  `OOM recovery failed` retry loop (issue #9 again).
- **Root cause:** eval generation batches all val samples (default
  `per_device_eval_batch_size=8`) on top of the resident model + the training
  step's still-allocated memory.
- **Workaround:** `per_device_eval_batch_size=1`.

## 15. Eval crash: `rewards.view(-1, num_generations)` shape error
- **Status:** FIXED (verified) — `num_generations_eval=1`.
- **Where:** `trl/trainer/grpo_trainer.py:2765`, `_generate_and_score_completions`,
  reached from `prediction_step` during `trainer.evaluate()`.
- **Error:** `RuntimeError: shape '[-1, 2]' is invalid for input of size 1`.
- **Root cause:** trl repeats the eval dataset `num_generations_eval` times
  (`repeat_iterable_dataset`, grpo_trainer.py:1363). With `per_device_eval_batch_size=1`
  each repeat lands in its own batch, so each `_generate_and_score_completions` call
  sees 1 completion while the reshape still assumes `num_generations_eval` (defaults to
  `num_generations` = 2 in smoke mode). `num_generations_eval` must be 1 (or
  `per_device_eval_batch_size` must be a multiple of it).
- **Note (comparability):** the reference eval is **greedy** mean@1 (`n-samples-per-eval-prompt 1`),
  whereas trl eval keeps `do_sample=True`/`temperature` from the training generation config.
  This greedy-vs-sampled eval divergence is still open for the final comparable run.

## 16. Eval OOM recurs after a multi-step training run (allocator cache)
- **Status:** FIXED (verified) — `torch_neuronx.empty_cache()` before eval.
- **Where:** `trainer.evaluate()`, ~2/8 eval samples in.
- **Error:** `Failed to allocate DEVICE memory (301989888 bytes)` → OOM retry loop. The
  288 MB allocation is the fused-LM-head logits for ~290 tokens (one eval batch).
- **Root cause:** the training step's logprob/generation buffers stay **cached** in the
  Neuron device allocator; they are not released to the OS, so the eval's fused-LM-head
  logits have no contiguous headroom. `torch_neuronx.empty_cache()` (the Neuron
  equivalent of `torch.cuda.empty_cache()`) releases them.
- **Note:** the 1-step smoke (fewer generations) left enough headroom to eval without
  the flush; the 4-step / n=8 run did not.

---

## Not yet reached (expected next)
- `fused_lm_head` interaction with PEFT/LoRA — **resolved: training step passed.**
- Multi-core tensor parallelism (NeuronX Distributed) vs. single-core memory ceiling.
- `beta=0.04` (reference model) — retry once single-core memory headroom is confirmed.

## Milestone (2026-10-23)

The full GRPO training step **and** the final eval both completed on Trainium2
(single core, eager, LoRA r=16, `beta=0.0`, `NKI_ENABLE_TRACE_CACHE=0`):

- generation → reward (0.4 blended F1, parsing works) → logprob → loss → backward →
  optimizer → step, then
- `torch_neuronx.empty_cache()` (`POST_TRAIN_MEM_ALLOCATED` 10.84 GB) → greedy eval.

Final 4-step smoke: `train_loss` 0.0002976, `train_runtime` 141.1 s,
`eval_rewards/reward_func/mean` 0.2958, `eval_runtime` 194.7 s, exit code 0.
The reward std is 3.186e-08 (the SFT checkpoint emits near-identical completions),
so no learning happened — an RL exploration issue, not a platform issue.

