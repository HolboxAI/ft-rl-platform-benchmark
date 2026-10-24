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

## 17. `generation_batch_size` must also divide by the global batch size
- **Status:** FIXED (`GEN_BATCH=16` for the full run).
- **Where:** `trl/trainer/grpo_config.py:1103` `__post_init__`.
- **Error:** `ValueError: generation_batch_size (8) must be divisible by the global batch size (16)`.
- **Root cause:** trl 1.16 requires `generation_batch_size % (per_device_train_batch_size * num_processes) == 0`
  (the "global batch size"), **in addition to** the older `generation_batch_size % num_generations == 0`
  (issue #4). So `generation_batch_size` must be a multiple of
  `LCM(num_generations, per_device_train_batch_size * num_processes)`.
- **Numbers:** full run `BATCH=16, NUM_GEN=8` → `LCM(8, 16) = 16`. The smoke `BATCH=1, NUM_GEN=8` →
  `LCM(8, 1) = 8`, which is why `GEN_BATCH=8` worked there.
- **Native fix:** document this second divisibility rule next to the `num_generations` one; the two
  constraints are easy to trip on together.

## 18. `max_steps=None` crashes transformers `_validate_args`
- **Status:** FIXED (`max_steps=-1` instead of `None`).
- **Where:** `transformers/trainer.py:714` `_validate_args`.
- **Error:** `TypeError: '>' not supported between instances of 'NoneType' and 'int'` on
  `if args.max_steps > 0 and args.num_train_epochs > 0`.
- **Root cause:** `GRPOConfig(max_steps=None)` (to mean "no step cap") is invalid; transformers expects an
  `int`, with `-1` as the sentinel for "use `num_train_epochs`".
- **Native fix:** transformers could accept `max_steps=None` as equivalent to `-1`, or trl's GRPOConfig
  should not forward `None` through to `TrainingArguments`.

## 19. `Qwen3VLVideoProcessor` requires torchvision (AutoProcessor load)
- **Status:** FIXED (`torchvision==0.29.1 --no-deps` in `setup2.sh`).
- **Where:** `AutoProcessor.from_pretrained` → `video_processing_auto.py:396` →
  `Qwen3VLVideoProcessor.from_pretrained` → `requires_backends(cls, ("torchvision",))`.
- **Error:** `ImportError: Qwen3VLVideoProcessor requires the Torchvision library but it was not found`.
- **Root cause:** the qwen3_5 checkpoint registers a video processor (`video_preprocessor_config.json`),
  and transformers loads it even though the task only feeds still frames. torchvision is not pulled in
  by trl/transformers by default and was missing from the setup.
- **Workaround:** `pip install torchvision==0.29.1 --no-deps` (ABI-matched to torch 2.14.0+cu130;
  `--no-deps` stops pip from replacing the Neuron torch).
- **Native fix:** transformers could fail soft on the video-processor backend (skip video preprocessing
  when the input is a single image) instead of hard-importing torchvision at processor load.

## 20. neuronx-cc instruction limit (NCC_EXTP003) on the response-masking `masked_scatter`
- **Status:** OPEN (mitigation: smaller `per_device_train_batch_size`; TP is the real fix).
- **Where:** `_get_per_token_logps_and_entropies` → response-only loss mask → `masked_scatter`
  (`[16, 640, 2560]` bf16 dest, `[16, 640, 1]` bool mask, `[6480, 2560]` source) during the
  **full run** (`BATCH=16, NUM_GEN=8`). The 4-step smoke (`BATCH=1`) never hit this.
- **Error:** `Compilation error … error= Command failed (neuronx-cc compilation) with exit code 70`
  → `[ERROR] [NCC_EXTP003] Instructions generated by compiler 131072000 exceeds the typical limit of
  300000. Input computation graph is too big due to large operators - Consider using smaller batches or
  sequence length, or applying tensor parallelism.`
- **Root cause:** neuronx-cc unrolls the scatter over every `(batch, seq, hidden)` element; at
  batch 16 × seq 640 × hidden 2560 that is 131 M instructions, ~437× the compiler's 300 k "typical
  limit". The failed compile then falls back and OOMs (`Failed to allocate DEVICE memory (536870912
  bytes)`, the 512 MB `UNKNOWN` tensor, issue #8/#9 recovery loop again).
- **Impact:** the single-core batch ceiling for this task is well below the verl reference's 16
  (which runs FSDP×4 + offload). A comparable full-epoch run on one core must use a small
  `per_device_train_batch_size` and more steps.
- **Native fixes:** (a) neuronx-cc should lower a large `masked_scatter` as a gather/reindex instead of
  fully unrolling it; (b) trl/transformers should expose a fused "masked logprob" kernel that does not
  materialize a `[batch, seq, hidden]` scatter; (c) raise the instruction limit or auto-split the op
  across cores.

## 21. `num_train_epochs=1` inflates the step count by `num_generations`
- **Status:** FIXED (set `max_steps` explicitly; `MAX_STEPS=64` for 256 prompts @ batch 4).
- **Where:** `trl/trainer/grpo_trainer.py` `get_train_dataloader` / `RepeatSampler`
  (`mini_repeat_count=num_generations`) + `generation_batch_size = per_device_train_batch_size × num_processes × steps_per_generation`.
- **Symptom:** 256 prompts, `per_device_train_batch_size=4`, `num_generations=8`, `generation_batch_size=8`,
  `num_train_epochs=1` → TRL runs **512** steps (`0/512`, epoch = step/512), i.e. 8× the expected 64
  (= 256 / 4), ETA ~12 h.
- **Root cause:** `generation_batch_size` must be divisible by both `num_generations` (8) and
  `per_device_train_batch_size × num_processes` (4), forcing `generation_batch_size=8` and
  `steps_per_generation=2`. The `RepeatSampler` then repeats each prompt `num_generations` times, so
  `len(train_dataloader) = num_prompts × num_generations / per_device_batch = 512`, which transformers uses
  as `max_steps`. So `num_train_epochs=1` means "one pass over completions", not "one pass over prompts".
- **Workaround:** set `max_steps = num_prompts / per_device_train_batch_size` explicitly (here 64).
- **Native fix:** trl should either document that `num_train_epochs` counts completion-groups and derive
  `max_steps` from prompt count, or warn when `num_generations > per_device_train_batch_size` forces
  `steps_per_generation > 1`.

## 22. Training-time OOM ~step 40: allocator cache accumulation ("Failed to load model")
- **Status:** SUPERSEDED by #23 — the `empty_cache()` workaround did **not** fix it: the run still
  failed at exactly step 40/64. The HBM-allocator hypothesis was wrong; real root cause + fix in #23.
- **Where:** `_scaled_dot_product_fused_attention_overrideable` (SDPA) inside `model(...)` forward, at
  step 40/64 (~52 min in) of the `BATCH=4` full-epoch run.
- **Error:** `RuntimeError: NRT Execution error occurred on Neuron for operation=_scaled_dot_product_fused_attention_overrideable; error message="NRT EXECUTION FAILED: Failed to load model, Failed to allocate resource"`.
- **Root cause:** over a multi-step run the Neuron device allocator keeps freed buffers cached; they
  accumulate until the attention kernel's workspace (model/NEFF load) has no contiguous HBM headroom. Same
  class as #16 (the *eval* OOM after training), but here it hits *mid-training* once enough distinct shapes
  have been cached.
- **Workaround:** a `TrainerCallback.on_step_end` calls `torch_neuronx.synchronize()` +
  `torch_neuronx.empty_cache()` every 4 steps, releasing cached-but-unreferenced buffers to the driver.
- **Native fix:** torch_neuronx should reclaim its allocator cache under pressure automatically (see #10)
  instead of requiring a manual `empty_cache()` between steps.

## 23. Step-40 "Failed to load model / Failed to allocate resource" = on-core TDRAM exhaustion (not HBM)
- **Status:** FIXED + VALIDATED (constant forward shape: `IMAGE_SIZE` normalization + `pad_to_multiple_of=256`).
  The full-epoch run passed the old step-40 failure point and completed all 64 steps + the 32-frame eval
  (`eval_reward` 0.3573) with memory flat — confirming the TDRAM diagnosis was correct and the fix holds.
- **Where:** a NEFF *load* at step 40/64 of the `BATCH=4` full-epoch run (~52 min in). Runtime log repeats
  `NMGR:kmgr_load_nn_post_metrics → Failed to load NN: module.neff, err: 4` then `NRT:nrt_load →
  Failed to load nrt model`, surfacing in Python as the SDPA error below.
- **Error:** `RuntimeError: NRT Execution error … operation=_scaled_dot_product_fused_attention_overrideable;
  error message="NRT EXECUTION FAILED: Failed to load model, Failed to allocate resource"`.
- **Root cause (corrected from #22):** `err: 4` is `NERR_RESOURCE`, and the resource is **TDRAM** — the
  small on-core DMA scratch pool — *not* HBM. `torch_neuronx.memory_allocated()` stayed ~10.8 GB of
  25.8 GB through the run, so `empty_cache()` (HBM-only) could never help. The trigger is a **new NEFF
  being loaded at step 40**: in eager mode each unique attention/SDPA shape compiles a new NEFF that pins
  on-core TDRAM, and by ~40 distinct shapes TDRAM is exhausted. Two independent sources of shape variance
  created those ~40 shapes: (a) CholecT50 frames are 854×480 *and* 1920×1080 (21/256 rows) → two prompt
  lengths; (b) TRL right-pads `completion_ids` to the **batch max** (variable 1..256) → the scoring
  forward's sequence length changes every batch.
- **Workaround:** make the forward shape constant. `_decode_image` normalizes every frame to
  `IMAGE_SIZE=(854, 480)` (the majority resolution), and `GRPOConfig(pad_to_multiple_of=256)` forces every
  completion to exactly `max_completion_length`. Together they collapse the scoring forward to a single
  sequence length → one SDPA NEFF → no per-batch NEFF growth.
- **Native fix:** torch_neuronx eager should evict/reuse NEFFs (and their TDRAM staging) when the on-core
  DMA pool fills, or fail with an actionable "TDRAM exhausted (N resident NEFFs)" message instead of the
  generic "Failed to load model". trl could also expose a fixed-length completion padding option so RLVR
  batches have a stable shape.
- **Validation:** the fixed run also showed **real RLVR signal** the 4-step smoke never did — `reward_std`
  0.04–0.28 (vs. the smoke's 3e-8), `rewards/reward_func/mean` varying 0.28–0.85, and variable
  `completions/mean_length` (8–19). The smoke's flat advantages were a short-run exploration artifact,
  not a platform issue.

---

## Infra (not TorchNeuron): S3 presigned-URL region mismatch
- **Status:** FIXED (regenerate URLs against `us-east-1`).
- **Error:** `curl: (22) 400` → S3 body `IllegalLocationConstraintException: The unspecified location
  constraint is incompatible for the region specific endpoint this request was sent to.`
- **Root cause:** bucket `stanford-train-data` is in `us-east-1`; the presigned URLs were generated with
  `aws s3 presign --region ap-south-2`, so the endpoint + signature targeted the wrong region.
- **Fix:** `aws s3 presign --profile riley --region us-east-1 --expires-in 604800 s3://stanford-train-data/...`.

---

## Infra (not TorchNeuron): launch script log-clobbering + `pgrep` self-match
- **Status:** FIXED (unique timestamped log + anchored `pgrep -f "^python grpo_train"`).
- **Symptom 1 (log clobber):** `launch_grpo.sh` ran `rm -f ~/run_grpo.log` inside the ssh retry loop. A
  banner-timeout "failure" on attempt 1 still started the remote process; the retry's `rm -f` then unlinked
  the live log (the run kept writing to a deleted inode) and spawned a duplicate that died with
  "Failed to initialize Neuron Runtime: status code 1" (device held). The final `EVAL_RESULT` was lost.
- **Symptom 2 (guard self-match):** the idempotency guard `pgrep -f "python grpo_train.py"` matched its own
  `bash -lc` wrapper (whose command line contains the launch string), so every launch was skipped with
  "ALREADY RUNNING".
- **Fix:** (1) each launch writes to `run_grpo_$(date +%s).log` — never `rm -f` a shared log; (2) the guard is
  `pgrep -f "^python grpo_train"` (anchored), matching only the real `python grpo_train.py` process, not the
  `bash -lc` wrapper.

---

## Not yet reached (expected next)
- Multi-core tensor parallelism (NeuronX Distributed) vs. single-core memory ceiling.
- `beta=0.04` (reference model) — needs multi-core; the ~10 GB ref-model copy does not fit one core.
- Checkpoint saving: the full-epoch run used `save_strategy="no"`, so no trained LoRA adapter was
  written. Re-run with `save_strategy="steps"`/`"epoch"` to keep the trained weights.

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

## Milestone (2026-10-24) — full 1-epoch

The full 1-epoch run (all 256 train rows, 64 steps, `BATCH=4`, `NUM_GEN=8`) completed end to end on a
single NeuronCore, then ran the 32-frame eval:

- 64/64 train steps (final step: loss -0.7667, `rewards/reward_func/mean` 0.3083, std 0.0988),
- `EVAL_RESULT` `eval_reward` **0.3573** (`eval_runtime` 511 s) — vs SFT baseline 0.31875 = **+0.0385 (+12%)**.

Real RLVR learning (train `reward_std` 0.05–0.28 across steps), unlike the smoke's flat 3e-08. Still not
comparable to verl 0.715 / Miles 0.65125 (full-fine-tune); this is LoRA r=16 + `beta=0.0` on one core.
The step-40 TDRAM fix (#23) held for the entire epoch.

## How to avoid these next time

### Dev-side (in-repo, already codified)

1. **Pin the stack** — torch-neuronx **2.14.3.0.282** (not the AMI-bundled 2.14.3.0.1), transformers
   5.18.0, trl 1.16.0, accelerate 1.15.0 (`config/TRAINIUM_CONFIG.md`).
2. **Keep the forward shape constant** (#22/#23) — normalize every image to one resolution and set
   `pad_to_multiple_of=max_completion_length`. Variable shapes compile one NEFF each and exhaust on-core
   TDRAM ~step 40.
3. **Single core ⇒ LoRA + `beta=0.0`** — full-fine-tune fp32 states (~40 GB) and the KL ref model (~10 GB)
   do not fit 25.8 GB HBM. Set `NKI_ENABLE_TRACE_CACHE=0` to drop the ~10 GB trace-cache pin.
4. **Geometry** — `generation_batch_size` must divide by both `num_generations` and the global batch size;
   use `max_steps=-1` (never `None`) with `num_train_epochs`.
5. **Launch hygiene** — unique timestamped logs (never `rm -f` a shared log) and an *anchored*
   `pgrep -f "^python grpo_train"` guard (an unanchored pattern self-matches the `bash -lc` wrapper).
6. **Run `scripts/healthcheck.sh` in cron** (every 5 min) — it detects crash/stall/death/success, pulls the
   log home, and terminates the spot instance so a dead job stops billing.
7. **Save the checkpoint** — the full-epoch run used `save_strategy="no"`, so the trained LoRA adapter was
   discarded with the instance. Use `save_strategy="steps"` (or `"epoch"`) to keep the trained weights.

### Package-side (native fixes to ask upstream)

1. **torch_neuronx** — evict/reuse NEFFs (and their TDRAM staging) when the on-core DMA pool fills, or fail
   with an actionable "TDRAM exhausted (N resident NEFFs)" instead of "Failed to load model".
2. **torch_neuronx** — raise `torch.OutOfMemoryError` after N failed OOM-recovery attempts instead of
   looping forever; reclaim the NKI trace-cache HBM on OOM (~10 GB pinned by default).
3. **neuronx-cc** — lower `torch.repeat_interleave` with a dynamic `repeats` tensor (or fall back to CPU)
   instead of a `BIRCodeGenLoop` crash; raise the 300k-instruction limit that hit the response-masking
   `masked_scatter`.
4. **torch_neuronx** — ship `causal_conv1d` + `flash-linear-attention` (or a Neuron Gated DeltaNet kernel)
   so the 24 GDN layers stop falling back to reference PyTorch.
5. **trl** — expose a fixed-length completion-padding option so RLVR batches have a stable forward shape.

