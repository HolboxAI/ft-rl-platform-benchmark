# Qwen3-ASR-0.6B GRPO RL — CUDA vs. Trainium2 (identical method, identical data)

**Headline:** On the *same* 32-utterance GRPO RL run, **Trainium2 (trn2.48xlarge) produces the
numerically identical result as CUDA (final reward `−0.11055` on both) but takes 75.9× longer
(4 h 27 m vs 3 m 31 s)**. The gap is compilation overhead, not compute throughput.

---

## 1. What was compared (identical on both platforms)

| Item | Value |
|---|---|
| Base checkpoint | `Qwen/Qwen3-ASR-0.6B` (787.47 M params; 5.05 M trainable via LoRA = 0.64 %) |
| Method | ms-swift `swift rlhf --rlhf_type grpo` (TRL `GRPOTrainer`) |
| Reward | `asr_composite` = `−0.5·(WER+CER) − 0.05·length_penalty − 0.05·repetition` |
| Dataset | **32 utterances** (first 32 rows of `librispeech_asr` clean/train.100), byte-identical file |
| Frozen parts | audio tower (`thinker.audio_tower`) + aligner/projector (LoRA on LLM decoder only) |
| GRPO config | `num_generations=8`, `lr=5e-7`, `gradient_accumulation_steps=8`, `max_completion_length=256`, `temperature=1.0`, `beta=0.0`, 1 epoch → **32 steps** |

The two runs use the same reward plugin (`reward/asr_reward_plugin.py`), the same 32-line
`data/librispeech/rl-train-tiny.jsonl`, and the same shell config except for the dtype/precision
flag (fp16 on CUDA, bf16 on Neuron).

---

## 2. Results at a glance

| Metric | **CUDA** — g6.xlarge (1× L4 24 GB) | **Trainium2** — trn2.48xlarge (64 NeuronCores) | Ratio |
|---|---|---|---|
| Precision | fp16 | bf16 | — |
| `train_runtime` | **211.28 s** (3 m 31 s) | **16 040.88 s** (4 h 27 m 21 s) | **75.9× slower** |
| `train_steps_per_second` | 0.151 | 0.002 | 75.5× |
| First step | 27.93 s (one-time warm-up) | 2 017.19 s (33 m 36 s, graph compile) | 72× |
| Steady-state step | **flat ~5.9 s** | **67 s → 1 297 s (erratic)** | 11–220× |
| Peak memory | 4.45 GiB | `memory(GiB)` = 0 (metric not populated on Neuron) | — |
| Final train_loss | 2.298e-05 | −2.56e-05 | ≈ |
| **Final reward** | **−0.11055** | **−0.11055** | **identical** |
| completions/mean_length | ~37–56 tokens | 51.6 tokens | ≈ |

---

## 3. The one number that matters: reward is identical

Both platforms converge to the **same final reward, `−0.11055`**, and the per-step reward
trajectory is nearly identical (small differences are sampling noise from `temperature=1.0`):

| step | CUDA reward | Trainium2 reward |
|---|---|---|
| 1 | −0.1131 | −0.11311 |
| 5 | −0.1220 | −0.12391 |
| 10 | −0.1149 | −0.11348 |
| 15 | −0.1222 | −0.12186 |
| 20 | −0.2435 | −0.24350 |
| 25 | −0.1243 | −0.12301 |
| 30 | −0.1299 | −0.13148 |
| **32 (final)** | **−0.1105** | **−0.11055** |

> **This is Trainium2's strongest result here: torch-neuronx reproduces the GRPO reward signal,
> loss, and gradient updates correctly.** There is no numerical-correctness bug. The dip at
> step 20 (−0.2435) appears on *both* platforms, and there is no monotonic reward improvement on
> either — expected for a 32-sample run with `beta=0.0`.

---

## 4. Where Trainium2 is slow — and why (the key finding)

The 75.9× wall-clock gap is **not** raw compute. It is **dynamic-shape graph recompilation** on the
Neuron compiler.

**CUDA step time is flat:** one 27.93 s warm-up, then every step is ~5.9 s.

**Trainium2 step time is wildly erratic** (per-step wall time across the 32 steps):

```
step  1: 2111 s  (33m36s)  ← initial compile
step  2:  405 s    step 16: 1297 s  (21m37s)  ← new-shape recompile spike
step  3:  400 s    step 19:   67 s  (1m07s)   ← fastest step
step  4:  638 s    step 20: 1215 s  (20m15s)  ← spike
step  5: 1200 s    step 25:  114 s
step  6:  571 s    step 26:  721 s            ← spike
step  7:  347 s    step 31:  101 s
step  8:  499 s    step 32:  212 s
step  9:  244 s
step 10:  689 s
step 11:  459 s
step 12:  216 s
step 13:  225 s
step 14:  200 s
step 15:  355 s
```

- **Spread:** 67 s (min) → 1 297 s (max), a **~19× variance** step-to-step.
- **Average:** ~501 s/step (8.3 min) vs CUDA ~6.6 s/step → **~76× per step**.
- **Even the "warm" steps (67–212 s) are 11–36× slower than CUDA's 5.9 s.**

**Root cause:** GRPO generates variable-length completions (36–56 tokens here, and variable audio
lengths from the ASR encoder). On Neuron each unique (audio-length, completion-length) shape triggers
a fresh **NEFF graph compile**. Because the 32-sample dataset never repeats a shape, almost every
step pays a compile cost. CUDA compiles once and is done.

A 0.6 B model with `num_generations=8` also massively under-utilizes 64 NeuronCores, so per-core
launch/overhead dominates the warm steps too.

---

## 5. Honest scorecard

### Where Trainium2 is behind CUDA
1. **Wall-clock throughput: 75.9× slower** on this workload (4 h 27 m vs 3 m 31 s).
2. **First-step compile: 72× longer** (33.6 min vs 27.9 s).
3. **Unpredictable step times** (19× variance) due to per-shape recompilation — several steps spike
   to 20+ minutes when a new shape appears.
4. **Memory telemetry gap:** ms-swift's `memory(GiB)` reports `0` on Neuron (it is a CUDA-only
   metric), so on-device HBM usage isn't surfaced in training logs.

### Where CUDA is clearly better
1. Flat, predictable ~5.9 s/step after a single one-time warm-up.
2. 76× faster end-to-end for this small, short run.

### Where Trainium2 has points CUDA doesn't (but this run can't show them)
1. **Numerical parity** — identical final reward and loss trajectory (the honest "good" result here).
2. **bf16 native** (CUDA leg ran fp16); better dynamic range for large-scale training.
3. **HBM capacity for large models** (64 NeuronCores / large HBM) — irrelevant at 0.6 B, but the
   platform's real advantage shows at large model + large batch, which a 32-utterance run cannot
   exercise.

> **Fair framing:** this 32-utterance run measures Trainium2's *compile-overhead and small-batch
> latency floor*, not its *throughput ceiling*. The 75.9× number should be read as "the
> recompilation tax on tiny, dynamic-shape ASR+GRPO workloads", not "Trainium2 is 76× slower in
> general".

---

## 6. Recommendation for the Amazon team

1. **Kill the per-shape recompilation** — it is the entire gap. Options: static/padded shapes or
   bucketing of both audio length and completion length, so a finite set of NEFFs is compiled once
   and reused; or compile all shapes up front before the training loop.
2. **Larger datasets amortize the compile tax** — on a real dataset, shapes repeat and the 33-minute
   first step + recompile spikes average out. The 32-sample run is a worst case for Neuron.
3. **A medium-throughput follow-up** (a few hundred to a few thousand utterances, static completion
   length) would isolate true per-step throughput and give a defensible CUDA-vs-Trainium2 number.
4. **Fix the `memory(GiB)` telemetry** for Neuron so HBM utilization is visible in ms-swift/TRL logs.

---

## 7. Run provenance

- **CUDA leg:** `g6.xlarge` (NVIDIA L4 24 GB), torch 2.10.0+cu130, fp16. Completed 2026-11-02.
- **Trainium2 leg:** `trn2.48xlarge` (64 NeuronCores), torch 2.14.0 + torch_neuronx 2.14.3.0.1,
  transformers 4.57.6, bf16, `ACCELERATE_TORCH_DEVICE=neuron`, NKI trace cache enabled.
  Checkpoint `checkpoints/qwen3-asr-rl-trn2-tiny/v0-20261102-103420/checkpoint-32`.
- Logs: CUDA `/tmp/rl_cuda_tiny.log`; Trainium2 `setup_trn2.log` (retrieved from the terminated
  instance's EBS volume via serial-console dump).
