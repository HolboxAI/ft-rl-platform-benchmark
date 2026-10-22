# Accelerator RL Benchmark: CholecT50 GRPO on Trainium and TPU

This is a companion to `ft-rl-platform-benchmark-main 2`. That repo ran the
CholecT50 surgical action-triplet task through RL platforms on NVIDIA GPUs:
Tinker, verl, Miles, SkyRL and NeMo-RL. This repo asks whether **the same
CholecT50 GRPO** can run on non-NVIDIA accelerators, and how it compares:

1. **AWS Trainium**: via **torch-neuronx** (PyTorch/XLA + Neuron compiler).
2. **Google Cloud TPU**: via **TorchTPU** (`torch_tpu`), with PyTorch/XLA as a baseline.

The task is **CholecT50 only**: the same frames, SFT checkpoint and reward as
the GPU runs ([`data/README.md`](data/README.md)). Plan, risks and validation
status are in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

## The task

| | |
|---|---|
| Data | CholecT50 endoscopic frames: 256 train / 32 val (verl parquet) |
| Task | list every `instrument-verb-target` action triplet in a frame |
| Model | `Qwen3.5-4B` VLM, merged SFT checkpoint (`cholec-sft-qwen3.5-4b`) |
| Reward | 0.4 × triplet F1 + 0.6 × mean(instrument, verb, target F1) |
| RL | GRPO, 16 frames × 8 samples per step, 16 steps (1 epoch), 256-token responses |
| Headline metric | greedy val reward before → after (32 frames) |

GPU reference results (val reward): Tinker 0.364 → 0.419 (LoRA), verl 0.000 → 0.715,
Miles 0.319 → 0.651, NeMo-RL 0.000 → 0.471, SkyRL collapsed (0.329 → 0.0).

## Library × chip matrix

| Library | Trainium | TPU | GPU reference / control |
|---|---|---|---|
| **Plain PyTorch** (`common/grpo`) | [`trainium/`](trainium/): torch-neuronx, LoRA 16 | [`torchtpu/`](torchtpu/): TorchTPU (+ PyTorch/XLA), LoRA 16 | `--backend cuda` |
| **Miles** | [`miles/trainium/`](miles/): probe; no Neuron platform | [`miles/tpu/`](miles/): probe; experimental `torch_tpu` training hook, no TPU rollout | Miles GPU run: 0.319 → 0.651 |
| **SkyRL** | [`skyrl/trainium/`](skyrl/): JAX backend on jax-neuronx (beta) | [`skyrl/tpu/`](skyrl/): JAX backend via Tinker API | [`skyrl/gpu_control/`](skyrl/): same client, FSDP backend |
| **HF TRL** (`GRPOTrainer`) | [`hf-trl/`](hf-trl/): stock TRL on native TorchNeuron (accelerate picks `"neuron"`), LoRA 16 | — | — |

The source already shows two blockers before any hardware run:
- **Miles:** no non-CUDA rollout engine.
- **SkyRL:** its JAX backend has no image input, and CholecT50 needs images.

So the Miles and SkyRL accelerator legs start as staged probes that record
where each stack stops. Their CholecT50 runs are scripted for when the blocking
stage passes.

## Status

_As of 2026-10-22, 15:20 UTC._

| Leg | Probe / smoke | CholecT50 run |
|---|---|---|
| Trainium: our GRPO loop (`trainium/`) | 🟡 run `-mini` failed at stage `device`: on-device RNG recurses in torch-neuronx 2.14.3; fixed (host-side RNG), rerunning as `-mini2` | ⏳ mini GRPO step (2 frames × 4 samples) |
| Trainium: HF TRL (`hf-trl/`) | 🟡 TorchNeuron, accelerate, TRL stack and Qwen3.5 vision tower work; blocked in `generate` prefill (int64 `input_ids` corrupted) | — not reached |
| TPU: TorchTPU / PyTorch/XLA | ⏳ | ⏳ |
| Miles | ❌ not compatible with Trainium: audit (552 NVIDIA + 67 SGLang call sites, 0 JAX) + probe; see `miles/TRAINIUM_INCOMPATIBILITY.md` | 🚫 no Neuron platform, no non-CUDA rollout |
| SkyRL: GPU control (g5.24xlarge) | 🟡 attempt 2: connect / train / RL-loss OK, vLLM engine failed to start for sampling; log capture added | ⏳ `tinker` + `skyrl-grpo` presets |
| SkyRL: TPU / Trainium | ⏳ (TPU waits on gcloud setup) | 🚫 no image input in JAX backend |

TorchNeuron issues found so far (torch-neuronx 2.14.3, trn1.2xlarge): data-dependent
`repeat_interleave` not lowered (worked around: `common/neuron_compat.py`); `int64.sum()` /
`int64.cumsum()` wrong for large values; int64 `input_ids` corrupted inside transformers'
`generate`; `torch.rand*` on device recurses (worked around: host-side RNG).

## Repository layout

```
.
├── README.md, EXPERIMENT_PLAN.md
├── requirements-common.txt    # transformers/peft etc. — installed on top of each chip's torch
├── common/
│   ├── grpo/                  # hardware-agnostic GRPO trainer (CholecT50)
│   │   ├── device.py          # the ONLY per-chip code: cpu | cuda | neuron | tpu | tpu-xla
│   │   ├── rollout.py         # static-shape sampler (compile-friendly) + log-probs
│   │   ├── tasks.py           # CholecT50 data / prompt / reward wiring
│   │   └── train_grpo.py      # trainer → steps.jsonl + final_results.json
│   ├── rewards/cholect50.py   # triplet-F1 reward (verbatim from the verl leg)
│   ├── smoke_test.py          # Phase 0 on the real checkpoint + frame
│   ├── fetch_cholect50.sh     # SFT checkpoint + parquet from S3, hash-checked
│   ├── porting_audit.py       # static NVIDIA-dependency audit of a framework
│   └── tests/                 # CPU correctness tests (rollout, GRPO update direction)
├── data/cholect50/            # parquet lands here (not committed)
├── trainium/                  # torch-neuronx leg
├── torchtpu/                  # TorchTPU (+ PyTorch/XLA) leg
├── miles/                     # Miles probes + porting audit
├── skyrl/                     # SkyRL: Tinker-API client, probes, GPU control + TPU/Trainium servers
└── analysis/                  # schema.md + compare.py (joins runs with the GPU references)
```

## Quick start

```bash
# Local correctness checks (CPU; any small Qwen3 text checkpoint)
pip install torch -r requirements-common.txt
python -m common.tests.test_rollout --model Qwen/Qwen3-0.6B
python -m common.tests.test_grpo_direction Qwen/Qwen3-0.6B

# On the accelerator: see trainium/README.md or torchtpu/README.md
bash common/fetch_cholect50.sh
bash trainium/scripts/run_smoke.sh        # or torchtpu/scripts/run_smoke.sh

# After runs land in */metrics/
python3 analysis/compare.py --plot
```

## How to add a result

1. Run the leg's script. It copies `final_results.json` and `steps.jsonl` into
   `<leg>/metrics/`.
2. Fill in the ⏳ cells in `<leg>/config/*_CONFIG.md`: instance, SDK versions,
   and cold or warm compile cache.
3. Add every workaround or failure to that file's findings log, and to
   `findings` in the results JSON.
4. Update the status table above.
