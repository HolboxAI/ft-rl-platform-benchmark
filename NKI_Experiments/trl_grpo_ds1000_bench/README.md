# TRL GRPO on DS-1000: Trainium (NKI) vs g6e GPU

This benchmark runs the same small RL job, stock TRL `GRPOTrainer` on 32 DS-1000 problems for 10 steps, on a g6e GPU and on Trainium (trn1, trn2 or trn3). It times every RL phase so you can see which parts of a step are faster on Trainium than on the GPU, and whether NKI kernels help. Every number is written to CSV for plotting later.

## Layout

```
configs/grpo_ds1000.json        shared settings (model, batch, lengths, steps); identical on every platform
data/prepare_ds1000_subset.py   picks the fixed 32-problem subset -> data/ds1000_subset32.jsonl
bench/
  platform_utils.py             backend detection, device sync, memory, XLA compile counters, versions
  phase_timer.py                device-synchronised nested phase timer
  ds1000_reward.py              DS-1000 execution reward (runs each problem's own test in a subprocess)
  static_shapes.py              fixed prompt/completion shapes so Neuron compiles each graph once
  nki_ops.py                    NKI RMSNorm kernel, NKI flash-attention loader, rollout patch
train/
  instrumented_trainer.py       GRPOTrainer subclass + callback that time each phase
  grpo_ds1000_bench.py          entry point (one run)
kernels/kernel_microbench.py    op-level timings: torch on GPU vs torch on Neuron vs NKI kernel
metrics/                        everything that stores or derives numbers (no training code)
  csv_store.py                  CSV writers and the schema of every raw file
  device_monitor.py             nvidia-smi / neuron-monitor sampler -> device_util.csv
  summarize_run.py              per-run summaries (phase_summary, run_summary, ...)
  compare_runs.py               cross-platform comparison tables (results/comparison/)
scripts/                        setup_*.sh, run_g6e.sh, run_trainium.sh
results/                        created by runs (one folder per run)
```

## What a step is

A step is 8 prompts × 4 generations = 32 completions. The batch is 8 completions with 4 accumulation steps, so there is one optimizer step per training step. With 10 steps the model sees the 32 problems 2.5 times. The default model is `Qwen/Qwen2.5-Coder-0.5B-Instruct` in bf16, and β = 0.04, so a reference model is used.

| Phase (CSV name) | What it covers |
|---|---|
| `step_total` | the whole optimizer step |
| `rollout_total` | TRL `_prepare_inputs`: everything below plus advantages |
| `generation` | `model.generate` (prefill and decode, static KV cache) |
| `reward` | running the DS-1000 tests (CPU subprocesses) |
| `logprob_nograd` | reference-model log-prob forward pass |
| `loss_compute` | TRL `compute_loss`, which includes `policy_forward` |
| `policy_forward` | policy log-prob forward pass with grad |
| `backward` | `accelerator.backward` |
| `optimizer_step` | `optimizer.step()` |
| `other` | `step_total` minus the four top-level phases (dataloader, clipping, logging) |

## Fairness rules built in

- **Identical work.** Prompts are padded to 512 tokens and every completion is exactly 256 tokens (`min_new_tokens`), on both platforms. This is like `--ignore-eos` in a serving benchmark, and it is also what lets Neuron compile each graph once.
- **Compile time is reported separately.** Steps 1–2 are marked `is_warmup` and left out of steady-state statistics. Their time and the XLA compile counters are reported in their own columns.
- **One accelerator.** Each run uses one L40S or one NeuronCore (`NEURON_RT_VISIBLE_CORES=0`). On trn2/trn3 that core is one logical core under the default LNC=2.
- **Two sync modes.** `--sync-mode phase` waits for the device at every phase edge, which gives accurate per-phase times. On Neuron this also splits the lazy XLA graph per phase. `--sync-mode step` waits only at step edges, which gives the true step time. Report step time from `step` runs and the phase breakdown from `phase` runs.
- **NKI on/off.** `--nki-ops rmsnorm` uses an NKI RMSNorm kernel in rollout forward passes, which run without grad. The kernel has no backward, so training keeps the PyTorch op. Compare the same Trainium platform with and without it. The `nki_kernel_calls` column shows the kernel was actually used.
- **Same reward environment.** Tests run in a separate env pinned to the DS-1000 versions (`scripts/setup_reward_env.sh`), so library versions don't change the rewards.

## Running

Copy the whole `NKI_Experiments/` folder to each instance. The subset script reads `../data/ds1000/test.jsonl`. That file is not in git; download it first (from `NKI_Experiments/`):

```bash
curl -sSfL -o data/ds1000/test.jsonl https://huggingface.co/datasets/xlangai/DS-1000/resolve/main/test.jsonl
```

```bash
# g6e (Deep Learning AMI with PyTorch)
bash scripts/setup_g6e.sh
INSTANCE_PRICE_USD_HR=<price> REPEATS=3 bash scripts/run_g6e.sh

# trn1 / trn2 / trn3 (Deep Learning AMI Neuron)
bash scripts/setup_trainium.sh
PLATFORM=trn2.3xlarge INSTANCE_PRICE_USD_HR=<price> REPEATS=3 bash scripts/run_trainium.sh
```

Optional: set `S3_RESULTS=s3://holbox-rl-benchmark-trn2/nki-experiments/` to sync results to S3. Run `data/prepare_ds1000_subset.py` once and copy `data/ds1000_subset32.jsonl` to the other machines, so every platform trains on the same file.

To compare, collect all run folders into one `results/` and run `python -m metrics.compare_runs --baseline g6e`.

## Output files

For each run, in `results/<platform>_<backend>_nki-<x>_sync-<y>_<time>/`:

| File | Grain | Key columns |
|---|---|---|
| `run_info.csv` | key/value | instance, region, price, all package versions, Neuron env, config, model load time |
| `phase_events.csv` | every timed call | step, phase, parent, start time, duration |
| `step_metrics.csv` | per step | `t_<phase>_s` for every phase; tokens; gen / train tokens/s; reward, pass rate, zero-std groups; loss, grad norm, KL; peak memory; XLA compile/execute deltas; NKI call counts |
| `rewards.csv` | per completion | problem, library, pass/fail, error type, execution time, length |
| `trl_logs.csv` | per step | everything TRL logs |
| `device_util.csv` | ~0.5–1 s samples | utilisation %, memory, power (GPU), per NeuronCore |
| `phase_summary.csv` | per phase | mean/median/p90 per step, first call, share of step % |
| `step_device_util.csv` | per step | mean/max utilisation, max memory, mean power |
| `reward_by_library.csv` | per library | pass rate, execution time, timeouts |
| `run_summary.csv` | one row | headline numbers incl. cost per step and per 1M generated tokens |

Additional outputs:

- `results/kernels_<platform>_<time>/kernel_microbench.csv` holds per-op medians, TFLOP/s and GB/s for GEMMs, linear forward+backward, LM-head log-prob gather, prefill/decode attention, RMSNorm and SiLU·mul, each as `torch` and (on Trainium) `nki`.
- `results/comparison/` holds `phase_comparison.csv` (speedup per phase vs g6e), `metric_comparison.csv` (headline metrics, better/worse vs g6e), `variant_summary.csv`, `all_runs.csv` and `kernel_comparison.csv`.

## Caveats

- **Not run yet.** The code has not been run on a GPU or Trainium yet. The Trainium path depends on stock TRL working on torch-neuronx (XLA). Expect to fix version-specific issues on the first run. `run_info.csv` lists any timing hooks or GRPOConfig options the installed TRL didn't support (`hooks_missing`, `grpo_config_dropped_keys`).
- **NKI API differences.** NKI import paths and the flash-attention signature differ between Neuron SDK releases. If a kernel can't load, the microbenchmark records an `error:` status instead of crashing.
- **Reward runs on the CPU.** The reward phase depends on the host's CPU count, which differs between instance types. Compare it as a host-side cost, not as an accelerator result.
- **Untrusted code.** The reward executes model-generated code. Use throwaway instances.
- **Sample size.** 10 steps is a small sample. Use `REPEATS=3` and the medians in `variant_summary.csv`.
