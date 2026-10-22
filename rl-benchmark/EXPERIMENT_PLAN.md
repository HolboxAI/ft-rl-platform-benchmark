# Experiment plan: CholecT50 GRPO on non-NVIDIA accelerators

## Question

Can the CholecT50 GRPO run from `ft-rl-platform-benchmark-main 2` run on
**AWS Trainium** (torch-neuronx) and **Google Cloud TPU** (TorchTPU)? And at
what cost in porting effort, wall time, dollars and val reward, compared with
the GPU runs already measured (Tinker, verl, Miles, SkyRL, NeMo-RL)?

## Why a new trainer as well as Miles / SkyRL

The GPU frameworks generate rollouts with CUDA inference engines (SGLang or
vLLM). None runs on Trainium or TorchTPU as shipped (see `miles/README.md` and
`skyrl/README.md` for the source-level evidence). So the main accelerator
legs use one small GRPO loop in plain PyTorch + transformers
(`common/grpo/`, ~900 lines). It runs unchanged on every backend; the only
per-chip code is `common/grpo/device.py` (~180 lines).

The trade-off: a pure-PyTorch rollout without a KV cache is much slower than
vLLM / SGLang. Throughput here measures "CholecT50 GRPO in portable PyTorch on
chip X", not "the best stack on chip X". For a fair baseline, run the same
trainer on a GPU too (`--backend cuda` on a g5 instance).

## Phases (per leg; each gates the next)

| Phase | What | Pass criteria | Compares to |
|---|---|---|---|
| **0: smoke** | SFT checkpoint + one val frame at the run's shapes: forward, LoRA train step, 16-token sampling, reward | all stages pass; compile times recorded | — |
| **1: CholecT50 GRPO, LoRA 16** | 16 steps, greedy val eval before / after | completes; val reward rises above the start; no collapse | Tinker 0.364 → 0.419 (LoRA); SkyRL GPU control `tinker` preset |
| full-model GRPO | needs multi-device sharding | out of scope until Phase 1 passes | verl 0.715, Miles 0.651 |

If a phase fails, the failing stage and its error are the result. Record them
in the leg's findings log and in `final_results.json`.

## Controlled variables (same on every backend)

- **Inputs:** the parquet frames (hash-checked) and the merged SFT checkpoint.
- **Prompt and response shape:** chat template with thinking disabled;
  prompts over 2,048 tokens dropped; 256-token responses.
- **Sampling:** 16 frames × 8 samples per step, temperature 0.7, no top-k or
  top-p, shuffle seed 1234.
- **Optimisation:** LoRA rank 16; lr 1e-5; AdamW (0.9, 0.98, wd 0.1); KL 0.04
  `low_var_kl`; clip 0.2 / 0.28; token-mean loss; one optimizer step per
  rollout batch.
- **Precision:** bf16 base weights, fp32 adapter.
- **Code:** the static-shape sampler and the reward.

## What gets measured

| Metric | Where |
|---|---|
| Val reward before / after (greedy, 32 frames) | `start_val_reward`, `final_val_reward`, `eval_curve` |
| Rollout reward and triplet F1 per step | `score_curve`, `triplet_f1_curve` |
| Compile overhead | `first_step_s` vs `steady_step_s` |
| Throughput | `gen_tok_per_s_median`, `rollout_s` / `train_s` per step |
| Cost | `estimated_cost_usd` (on-demand $/hr × wall time) |
| Response length, truncation, parse rate | per-step `steps.jsonl` |
| Porting effort | findings log: every workaround, patch, version pin, failed attempt |

## Known risks

1. **Qwen3.5 on non-CUDA.** Its Gated-DeltaNet layers use `fla` /
   `causal_conv1d` CUDA kernels on GPU. On Trainium and TPU they need the
   pure-PyTorch fallback, which may be slow or may not compile.
2. **Rollout speed.** The static sampler has no KV cache and re-runs the vision
   tower each token: O(T·(P+T)) per step with P≈2,048 and T=256. Try
   `--gen-impl hf` (transformers static cache) once the static path works.
3. **Compile time** on Trainium and PyTorch/XLA: every new shape compiles. The
   trainer keeps shapes fixed, but the first step still compiles several
   graphs (sampler forward, train forward/backward, reference forward, optimizer).
4. **TorchTPU maturity.** A new API. Check the install line, supported TPU
   generations, the synchronize call and `torch.compile` coverage against the
   docs for the installed version.
5. **Memory on one NeuronCore (16 GB).** Qwen3.5-4B bf16 is ~8.5 GB before
   activations. Fall back to `trn2` if needed.

## Validated so far

On a laptop CPU (2026-10-22), with a 2-layer random Qwen3 text model and the
real Qwen3 tokenizer. The trainer's math is task-independent:

- `common/tests/test_rollout.py`: the static greedy sampler matches
  `transformers.generate` token for token; `response_logprobs` matches a direct
  log-softmax; early stop only fires on sampled EOS.
- `common/tests/test_grpo_direction.py`: one GRPO update raises the log-prob of
  every positive-advantage sample and lowers it for every negative one.

**Not yet validated:** the CholecT50 / Qwen3.5 VLM path of `common/grpo`
(`tasks.CholecT50Task`, the smoke test) on any device, and any run on
Trainium or TPU.
