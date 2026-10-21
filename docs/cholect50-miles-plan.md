# CholecT50 GRPO on Miles — strict reproduction plan

## 1. Goal

Reproduce the verl CholecT50 GRPO run on Miles, exactly.

One number to reproduce:

| Metric | verl value |
|---|---|
| `cholec_t50/reward/mean@1` on val, after 14 steps | **0.715** |
| SFT baseline, same metric, before training | **0.000** |

The comparison is only valid if the model, data, reward, hyperparameters, and eval
protocol are identical. Only the library mechanism may differ. That mechanism is
what this study measures.

The first draft of this plan was wrong on three facts. The README of the reference
repo corrects them:

1. The effective data is **235 train / 28 val** rows, not 256 / 32. verl drops
   overlong prompts with `filter_overlong_prompts` at 2048 tokens.
2. The run has **14 steps**, not 16. 235 effective samples at batch 16 leaves 14
   full steps and an 11-sample remainder that is dropped.
3. The val metric is **greedy mean@1**, not mean-of-8.

## 2. The reference, exactly

Source: `HolboxAI/stanford-challenge-Team1/tree/main/cholec-t50-grpo`.

### 2.1 Model

- `Qwen/Qwen3.5-4B` (vision-language), SFT checkpoint at
  `s3://stanford-train-data/models/cholec-sft-qwen3.5-4b`.
- bfloat16, thinking disabled (`enable_thinking=false` baked into the chat
  template and `generation_config.json`).
- Current S3 layout: one `model.safetensors` (10.6 GB) plus
  `model.safetensors.index.json`. The README text that says two shards is stale.
- Architecture: `Qwen3_5ForConditionalGeneration`, 32 layers, Gated DeltaNet
  linear-attention hybrid (`linear_attention` x3 then `full_attention`, interval 4),
  3D mRoPE (`mrope_interleaved: true`, `mrope_section: [11, 11, 10]`), GQA 16/4,
  head dim 256, `image_token_id` 248056.

### 2.2 Data

- `s3://stanford-train-data/cholec/parquet/train.parquet` (256 rows) and
  `val.parquet` (32 rows). verl-format.
- Prompt (exact): `<image>\nYou are a surgical activity recognition model. List
  every surgical action triplet (instrument-verb-target) visible in the endoscopic
  frame. Reply only as comma-separated triplets, each formatted
  instrument-verb-target. If no action is visible, reply 'none'.`
- `ground_truth` is a JSON string: `{"triplets": [...], "instruments": [...],
  "verbs": [...], "targets": [...]}`.
- After the 2048-token filter: 235 train, 28 val.

### 2.3 Reward

`cholec_reward.py`, entrypoint `compute_score(data_source, solution_str,
ground_truth, extra_info) -> float`.

```
reward = 0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)
```

The parser is exact. Key rules that a naive port gets wrong:

- Strip `<think>...</think>` with `re.DOTALL`.
- Split on commas after replacing newlines with commas.
- Each part must be exactly `instrument-verb-target` with all three in the
  vocabularies. Any malformed part returns `None` and the whole reward is 0.
- `"none"`, `"no action"`, `"nothing"`, `"no triplets"`, empty -> empty set.
- `_f1` returns 0.0 when true positives are 0, even when both sets are empty.
  There is no special case for the empty-empty pair.

Vocabularies (canonical, case-insensitive):

- instruments (6): `grasper bipolar hook scissors clipper irrigator`
- verbs (10): `grasp retract dissect coagulate clip cut aspirate irrigate pack null_verb`
- targets (15): `gallbladder cystic_plate cystic_duct cystic_artery cystic_pedicle
  blood_vessel fluid abdominal_wall_cavity liver adhesion omentum peritoneum gut
  specimen_bag null_target`

### 2.4 Hyperparameters (full, from train.sh)

Data: `train_batch_size 16`, `max_prompt_length 2048`, `max_response_length 256`,
`image_patch_size 16`, `filter_overlong_prompts True`, `truncation error`,
`enable_thinking False`, `image_key images`.

Algorithm: `adv_estimator grpo`, `use_kl_in_reward False`.

Model: `use_remove_padding False`, `enable_gradient_checkpointing True`.

Actor (FSDP): `strategy fsdp`, `lr 1e-5`, `ppo_mini_batch_size 16`,
`ppo_micro_batch_size_per_gpu 1`, `use_dynamic_bsz False`, `use_kl_loss True`,
`kl_loss_coef 0.04`, `kl_loss_type low_var_kl`, `entropy_coeff 0`,
`use_torch_compile False`, `fsdp_size 4`, `model_dtype bfloat16`,
`reshard_after_forward True`, `offload_policy False`, `param_offload True`,
`optimizer_offload True`.

Reference model: `strategy fsdp2`, `log_prob_micro_batch_size_per_gpu 1`,
`model_dtype bfloat16`, `param_offload False`, `reshard_after_forward True`,
`use_torch_compile False`.

Rollout (vLLM): `tensor_model_parallel_size 1`, `gpu_memory_utilization 0.6`,
`n 8`, `temperature 0.7`, `top_p 1.0`, `prompt_length 2048`, `response_length 256`,
`max_model_len 4096`, `max_num_batched_tokens 4096`, `max_num_seqs 128`,
`ignore_eos False`, `log_prob_micro_batch_size_per_gpu 1`,
`enable_chunked_prefill False`, `enforce_eager False`, `enable_prefix_caching
False`, `free_cache_engine True`, `update_weights_bucket_megabytes 256`.

Trainer: `critic_warmup 0`, `logger ["console"]`, `n_gpus_per_node 4`, `nnodes 1`,
`balance_batch True`, `val_before_train True`, `save_freq 16`, `test_freq 16`,
`total_epochs 1`, `resume_mode disable`.

### 2.5 Compute and results

- `ml.g5.12xlarge` (4 x A10G 24 GB). Step time ~388 s. Full run ~90 min.
  Peak GPU memory 19.7 GB.
- SFT baseline val reward 0.000. GRPO val reward 0.715 after 14 steps.
- Training reward (`critic/rewards/mean`) peaks at 0.601 on step 13.

## 3. What must match, no exceptions

1. The model checkpoint. Same S3 path, same bfloat16, same thinking-disabled state.
2. The data partition. Same 256 / 32 rows, same prompt, same ground truth, same
   overlong filter so we land on the same 235 / 28.
3. The reward. Byte-identical float output on identical inputs.
4. Every hyperparameter in section 2.4 that has a Miles equivalent.
5. The eval protocol. Greedy mean@1 on val, before training and at step 14.

Anything in this list that does not match makes the result non-comparable. We do
not trade any of it away for convenience.

## 4. Port mapping — verl to Miles

Miles uses SGLang for rollout and its own FSDP for training. The verl actor uses
FSDP (`fsdp_size 4`) and the reference model uses FSDP2. Use the Miles FSDP
backend, not Megatron.

The values are fixed by section 2.4. The flag spelling is what gets confirmed at
the smoke test.

### 4.1 Data and algorithm

| verl | Miles |
|---|---|
| adv_estimator grpo | `--advantage-estimator grpo` |
| use_kl_in_reward False | `--kl-coef 0.00` |
| use_kl_loss True, kl_loss_coef 0.04 | `--kl-loss-coef 0.04` |
| kl_loss_type low_var_kl | `--kl-loss-type low_var_kl` |
| entropy_coeff 0 | `--entropy-coef 0.0` |
| train_batch_size 16 | `--rollout-batch-size 16` |
| n 8 | `--n-samples-per-prompt 8` |
| max_prompt_length 2048 | prompt cap 2048 (verify flag) |
| max_response_length 256 | `--rollout-max-response-len 256` |
| filter_overlong_prompts True | overlong filter on (verify flag) |
| enable_thinking False | `--apply-chat-template-kwargs '{"enable_thinking": false}'` |
| image_key images | `--multimodal-keys '{"image": "images"}'` |

`global-batch-size` must be 128 (16 prompts x 8 samples), matching the verl
mini-batch of 16 over the full 128-sample rollout.

### 4.2 Optimizer

| verl | Miles |
|---|---|
| lr 1e-5 | `--lr 1e-5` |
| — | `--optimizer adam` (verl default is AdamW) |
| — | `--lr-decay-style constant` (verl uses constant LR here) |

Confirm Adam betas and weight decay match verl's defaults. Log the resolved values.

### 4.3 Model and training backend (FSDP)

| verl | Miles |
|---|---|
| strategy fsdp, fsdp_size 4 | `--train-backend fsdp` |
| model_dtype bfloat16 | bf16 (default) |
| enable_gradient_checkpointing True | `--gradient-checkpointing` |
| use_torch_compile False | no torch.compile |
| use_remove_padding False | verify padding is not removed |
| param_offload True, optimizer_offload True | verify FSDP CPU offload flag |
| reshard_after_forward True | FSDP reshard default |
| ppo_micro_batch_size_per_gpu 1 | micro batch 1 per GPU |

The offload flags matter. verl offloads params and optimizer to CPU so a 4 B model
with Adam fits in 24 GB. Confirm Miles FSDP has the same offload. If it does not,
the run cannot fit on 4 x A10G and this is a finding, not a reason to switch to
Megatron.

### 4.4 Rollout (SGLang replaces vLLM)

| verl (vLLM) | Miles (SGLang) |
|---|---|
| tensor_model_parallel_size 1 | `--rollout-num-gpus-per-engine 1` |
| gpu_memory_utilization 0.6 | `--sglang-mem-fraction-static 0.6` |
| temperature 0.7 | `--rollout-temperature 0.7` |
| top_p 1.0 | `--rollout-top-p 1.0` |
| max_model_len 4096 | model max len 4096 (verify flag) |
| ignore_eos False | do not ignore EOS (verify flag) |
| enforce_eager False | use CUDA graph, not eager (verify flag) |
| enable_chunked_prefill False | no chunked prefill (verify flag) |
| enable_prefix_caching False | no prefix caching (verify flag) |

vLLM and SGLang do not share every knob. Where a flag has no exact SGLang
equivalent, match the intent and log the effective value. The rollout engine is an
inherent difference between the two libraries.

### 4.5 Trainer

| verl | Miles |
|---|---|
| total_epochs 1 | `--num-epoch 1` |
| n_gpus_per_node 4, nnodes 1 | `--actor-num-gpus-per-node 4 --actor-num-nodes 1` |
| val_before_train True | eval at step 0 (see note) |
| test_freq 16 | `--eval-interval 16` |
| save_freq 16 | `--save-interval 16` |
| balance_batch True | balance batch (verify flag) |
| logger ["console"] | console log, no wandb |

Note on `val_before_train`. Miles has `--eval-interval` and `--eval-prompt-data`.
If it has no "before" flag, run one eval-only step first and read the step-0 metric.
The baseline must read 0.000 or the thinking-disable flag is wrong.

## 5. What is inherently different (logged, not changed)

These differ by design between the libraries. We do not force them to match. We
measure them.

1. **Rollout engine.** verl uses vLLM. Miles uses SGLang. Different scheduler,
   different memory layout, different token batching.
2. **Training backend numerics.** Both are FSDP, but Miles' implementation differs
   from verl's (which also splits actor FSDP from reference FSDP2). Adam updates,
   gradient accumulation order, and resharding can differ in low-order bits.
3. **Position-id handling.** Qwen3.5 emits 3D mRoPE position ids. verl needed three
   patches to survive them. Miles either handles them natively or needs its own
   fix. This is the single most important thing to log.
4. **KL estimator numerics.** `low_var_kl` is specified the same, but the
   implementation can differ.
5. **Reward dtype.** Miles' default reward path can add fp32 noise for fractional
   rewards (section 8.6). verl does not have this exact footgun.

## 6. Code to write

All in `miles/scripts/`, mirroring the existing ORena files.

### 6.1 `reward_cholect50.py`

A verbatim port of `cholec_reward.py` with the Miles entrypoint swapped in. The
parsing and F1 logic is copied line for line. The Miles contract is:

```
async def reward_fn(args, sample, **kwargs) -> dict
```

Read `sample.response` as `solution_str` and `sample.label` as the `ground_truth`
JSON string. Return:

```
{"reward_value": blended,
 "triplet_f1": ..., "instrument_f1": ..., "verb_f1": ..., "target_f1": ...}
```

`--reward-key reward_value` pulls the float. The other four keys feed the log hook.

Keep the exact `_f1` behavior: return 0.0 when true positives are 0, with no
empty-empty special case. Keep the exact malformed-part rule: any bad part returns
`None` and the whole reward is 0.

### 6.2 `generate_cholect50.py`

Start from `generate_orena.py`. It wraps the base SGLang generate and drops
non-tensor `multimodal_train_inputs` fields. Keep the wrapper. The smoke test
checks whether Qwen3.5's processor returns a list-valued field that breaks the
Megatron-style tensor path. If the base generate works, use it directly.

### 6.3 `log_rollout_cholect50.py`

A copy of `log_rollout_orena.py` that logs the four F1 keys and the blended reward
every step. See section 8.

### 6.4 `prepare_cholect50_data.py`

Convert the verl parquet to Miles format. Do not mutate the S3 source.

Miles row:

```
{"messages": [{"role": "user", "content": "<instruction without the <image> prefix>"}],
 "images": [<image>],
 "label": "<ground-truth JSON string>"}
```

Steps:

1. Read `train.parquet` and `val.parquet` from S3.
2. Take the user content and strip the leading `<image>` token. Miles inserts the
   image from the `images` column.
3. Keep the `images` column. Confirm the object format Miles and SGLang expect
   (PIL, path, or base64) at the smoke test.
4. `label` = `reward_model.ground_truth`, the JSON string, kept verbatim.

Write `train.parquet` (256 rows) and `val.parquet` (32 rows) in Miles format.

### 6.5 `run_cholect50_rl.sh`

A copy of `run_orena_rl.sh` changed to the FSDP backend and the values in section 4:

- Pull the checkpoint from S3, not `hf download`.
- `--train-backend fsdp --gradient-checkpointing`.
- `--input-key messages --label-key label --multimodal-keys '{"image": "images"}'`.
- `--custom-rm-path reward_cholect50.reward_fn --reward-key reward_value`.
- `--custom-rollout-log-function-path log_rollout_cholect50.log_rollout_data`.
- GRPO, KL, optimizer, SGLang args from section 4.
- `--num-epoch 1 --save-interval 16 --eval-interval 16`.
- Rotary base must match the checkpoint `rope_theta` 10,000,000, not the Qwen3-VL
  value 5,000,000. Confirm the `qwen3.5-4B` model args script already sets this.

## 7. Compute and launch

- Instance: `ml.g5.12xlarge` (4 x A10G 24 GB). Same as verl.
- Spot, on the `riley` profile, `--region us-east-1` on every AWS call.
- Container: `radixark/miles:latest`.
- Uninstall `flash-attn-4` before training. It crashes on Qwen3.5's 16/4 GQA
  layout. This is the same fix as the ORena run.
- `--num-rollout` counts steps, not rollouts. Use `--num-epoch 1`. With 16 prompts
  per step and 235 effective samples, one epoch is 14 steps.
- Save metrics and checkpoint to S3 before the instance terminates.

## 8. Logging plan

The logs must prove the identical parts are identical and show where the libraries
differ.

### 8.1 The identical parts — parity logs

- Data parity: log train and val row counts after the overlong filter. They must be
  235 and 28. If they differ, the tokenizer or template differs and the result is
  not comparable.
- Reward parity: before training, run the ported reward and the verl reward on the
  same canned responses. Assert every float is identical. This includes the
  empty-empty case and the malformed-part case.
- Baseline parity: the step-0 greedy val reward must read 0.000. If not, the
  thinking-disable flag is wrong.

### 8.2 The per-step metric set

The log hook writes every step:

- `rollout/reward_value` (blended), `rollout/triplet_f1`, `rollout/instrument_f1`,
  `rollout/verb_f1`, `rollout/target_f1`.
- `rollout/response_len_mean`.
- `rollout/kl` (the low-var KL estimate).

These reproduce the verl training curve (`critic/rewards/mean`, peak 0.601 step 13)
and the val curve (`cholec_t50/reward/mean@1`, 0.0 to 0.715).

### 8.3 Position-id handling (the headline difference)

Log once at startup and once per eval:

- Whether SGLang emitted 3D mRoPE position ids for Qwen3.5. Print the position-id
  shape from a probe request.
- Whether the FSDP forward consumed them without a reshape or unbind crash.

verl needed three patches here. A clean native path in Miles is the headline
finding. A crash is equally a finding, and it says Miles needs the same class of
patch.

### 8.4 Engine and backend

- SGLang version and resolved attention backend.
- The FSDP shard count and whether CPU offload is active.
- Per-GPU memory at steady state.
- Whether `flash-attn-4` is present or uninstalled.

### 8.5 Rollout-engine difference (SGLang vs vLLM)

Per step: tokens per second, wall time per rollout step, rollout batch size, n
samples per prompt. vLLM and SGLang schedule a 16-prompt n=8 batch differently.

### 8.6 Training-backend difference (Miles FSDP vs verl FSDP)

Per step: gradient accumulation count, effective micro batch, optimizer step time.
verl splits actor FSDP from reference FSDP2; Miles uses one FSDP path. The time and
memory split is the visible difference.

### 8.7 KL difference

verl puts KL only in the loss (`use_kl_in_reward False`). Miles does the same with
`--kl-coef 0` and `--kl-loss-coef 0.04`. Log the per-step loss-side KL and confirm
the reward-side KL is zero.

### 8.8 Non-binary reward precision

The triplet reward is fractional in [0, 1]. The geo3k README warns that under fp32
a group whose members all share one fractional reward still yields a nonzero
`reward - mean`, which adds spurious gradient. This is our case. Cast the reward
tensor to fp16 in `miles/ray/rollout/train_data_conversion.py`
(`_post_process_rewards`) and log the max `|reward - group_mean|` before and after.
This is a Miles-specific behavior verl and Tinker do not have.

### 8.9 Thinking-mode check

Log a sample of raw completions from step 0. They must contain no `<think>` block.

### 8.10 Data-pipeline check

Log the tokenized prompt length and image-token count for a sample. Log the dropped
row count. This proves the pipeline matches verl's.

## 9. Execution order

Phase 0 — code. Write the five files. Unit-test the reward offline against
`cholec_reward.py` on the canned cases (exact, one-of-two, thinking-leak, none,
malformed). The floats must match bit for bit.

Phase 1 — smoke test. Launch a spot `g5.12xlarge`. Load the checkpoint. Run one
SGLang generation and print the position-id shape and a sample completion. Run one
FSDP forward and confirm the GDN kernel and offload load. Confirm thinking is off.
This is the go/no-go gate.

Phase 2 — the run. One epoch, 14 steps, val before train and at step 14. Collect
every metric in section 8.

Phase 3 — the result. Read the greedy val reward on the 28 frames. Report the
blended reward and the four F1s. Compare to 0.715.

Phase 4 — record. Save metrics and checkpoint to S3. Add the results and charts to
the observation document.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Miles FSDP lacks CPU offload, cannot fit 4 B + Adam in 24 GB | Smoke test; if true, record it as a Miles limitation |
| Qwen3.5 3D mRoPE path not wired in SGLang or FSDP | Smoke test probes the shape before any long run |
| GDN linear-attention kernel mismatch in FSDP | Smoke test the forward; this is the unverified path |
| FA4 crashes on GQA | Uninstall flash-attn-4 |
| Non-binary reward fp32 noise | fp16 reward dtype fix, logged before and after |
| `--num-rollout` counts steps | Use `--num-epoch 1` |
| Thinking mode on by default | `enable_thinking false` in the template kwargs |
| Overlong filter lands on a different row count | Parity log; if not 235/28, stop and fix tokenization |
| Spot reclaim mid-run | `--save-interval 16`; sync to S3 each save |
| Wrong rotary base | Confirm rope_theta 10,000,000 at smoke test |

## 11. Metrics for the comparison

One table per metric, three rows (verl, Tinker, Miles), one scale. The common scale:

- Same 28 val frames as verl, greedy decoding, same blended reward.
- Blended reward, triplet F1, instrument F1, verb F1, target F1.
- Steps to peak and final, wall time, cost.
- Engine (vLLM vs SGLang), backend (verl-FSDP vs Miles-FSDP), KL path, reward
  dtype, position-id handling.

The success bar is 0.715 on the 28 val frames, within a small tolerance. A miss is
acceptable only if the logs explain which mechanism caused it.
