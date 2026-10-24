# Trainium2 — CholecT50 GRPO (Experiment A)

This directory is the **Trainium2** leg of the `ft-rl-platform-benchmark`. It runs
the same CholecT50 surgical action-triplet GRPO (RLVR) experiment as the Miles,
verl, and SkyRL legs, on AWS Trainium2 hardware through the TorchNeuron
(`torch-neuronx`) native PyTorch backend.

## Status

**The platform go/no-go passed — including a full 1-epoch run.** A full GRPO step
and the final eval both complete on a single Trainium2 NeuronCore. The full-epoch
run (64 steps, all 256 train rows) also completes and shows a real RLVR gain over
the SFT baseline.

| Metric | Value |
|---|---|
| State | succeeded (full 1-epoch, 64 steps) |
| Val reward (eval) | **0.3573** |
| Train reward (final step 64) | 0.3083 (std 0.0988) |
| Train loss (final step) | -0.7667 |
| SFT baseline (Miles/verl reference) | 0.31875 |
| Miles final (full run) | 0.65125 |
| verl final (full run) | 0.715 |

The full-epoch run improves on the SFT baseline (+0.0385, ~12%) and shows real
reward variance (train `reward_std` 0.05–0.28, vs the smoke's 3e-08), so GRPO is
genuinely learning — unlike the 4-step smoke, whose reward was flat. It is still
not comparable to verl/Miles: those are full-fine-tune runs, while this is
LoRA r=16 + beta=0.0 on one core. See `metrics/final_results.json` for the full
record.

## What works

- Qwen3.5-4B VLM (8 full-attention + 24 Gated DeltaNet layers) runs on the
  `neuron` device in eager mode.
- The full TRL GRPO loop runs: generation → reward → logprob → loss → backward →
  optimizer → step.
- The final eval runs after a `torch_neuronx.empty_cache()`.

## Directory layout

```
trainium2/
├── README.md                this file
├── TRAINIUM_ISSUES_LOG.md   the bugfix / native-fix doc (23 issues, with asks for torch_neuronx)
├── config/
│   └── TRAINIUM_CONFIG.md   exact stack pins and hyperparameters
├── metrics/
│   ├── final_results.json   the run's results and findings
│   └── run_manifest.json    platform, compute, and experiment manifest
├── scripts/
│   ├── grpo_train.py        main TRL GRPO training script
│   ├── reward_tr.py         CholecT50 blended-triplet reward
│   ├── gate_test.py         go/no-go: VLM forward + backward on neuron
│   ├── llm_gate_test.py     go/no-go: text decoder (Gated DeltaNet) on neuron
│   ├── inspect_trl.py       dump trl 1.16.0 GRPOConfig fields
│   ├── setup.sh             setup for a fresh instance (legacy)
│   ├── setup2.sh            setup for the torch-neuronx 2.14.0 DLAMI
│   ├── download_model.sh    fetch the SFT checkpoint from S3
│   ├── download_data.sh     fetch the CholecT50 JSONL from S3
│   ├── launch_grpo.sh       push + launch the GRPO run detached
│   ├── poll_grpo.sh         poll the remote log to a terminal state
│   └── push_run.sh          push any script and run it detached
└── logs/
    ├── gate.log             gate test output (incl. neuron memory dump)
    ├── llm_gate.log         LLM gate test output
    ├── run_grpo.log         the successful 4-step GRPO run + eval
    ├── run_inspect_trl.py.log
    ├── dl_model.log / dl_data.log / setup2.log
    └── setup2.log
```

## The bugfix / native-fix doc

`TRAINIUM_ISSUES_LOG.md` is the deliverable for library maintainers. It records
every error hit while bringing TRL GRPO to a completed run, with exact error
text and a suggested native fix for `torch_neuronx`, `neuronx-cc`, and TRL. The
highest-value asks are:

1. Lower (or reclaim) the NKI trace-cache HBM footprint; it pins ~10 GB by
   default.
2. Raise `torch.OutOfMemoryError` after N failed OOM-recovery attempts instead of
   looping forever.
3. Lower `torch.repeat_interleave` with a dynamic `repeats` tensor (or fall back
   to CPU) instead of a `BIRCodeGenLoop` crash.
4. Ship/allow `causal_conv1d` + `flash-linear-attention` (or a Neuron Gated
   DeltaNet kernel) so the 24 GDN layers stop falling back to reference PyTorch.

## Why the smoke is not the full run

One NeuronCore has 25.8 GB of HBM. Two things do not fit:

- GRPO with `beta != 0` loads a **second full model copy** as the reference
  model (~10 GB).
- Full fine-tune AdamW keeps **fp32 optimizer states** (~40 GB).

So the smoke uses **LoRA** (32.5M trainable) and **beta = 0.0**. A comparable
run needs multi-core tensor parallelism (NeuronX Distributed) or a CPU-offload
optimizer. Both are open items in the issues log.

## Reproduce (full 1-epoch run)

1. Launch a `trn2.48xlarge` **spot** instance in `ap-south-2b` (keypair
   `trainium-ap-south-2`, DLAMI `aws_neuronx_venv_torch_neuronx_2_14_0`).
2. `scripts/setup2.sh` — install the pinned transformers/trl/accelerate/peft on top of the DLAMI venv.
3. `scripts/download_model.sh` + `scripts/download_data.sh` — pull the SFT checkpoint and
   `cholect50_{train,val}.jsonl` from the presigned S3 URLs in `/home/ubuntu/urls.sh`.
4. Gate checks: `push_run.sh gate_test.py` and `push_run.sh llm_gate_test.py` (both exit 0).
5. Full epoch: `scripts/launch_grpo.sh` — pushes `grpo_train.py` + `reward_tr.py` and runs detached with
   `SMOKE=0 MAX_STEPS=64 BATCH=4 NUM_GEN=8 GEN_BATCH=8` (64 steps = one pass over 256 rows).
6. Healthcheck: install `scripts/healthcheck.sh` in cron (`*/5 * * * *`) so a crash/stall/success pulls the
   log home and terminates the spot instance (stops billing).
7. Collect: the run prints `EVAL_RESULT` as the last log line; record it in `metrics/final_results.json`
   and this README's status table.

The 4-step smoke is the same but with `SMOKE=1 MAX_STEPS=4 BATCH=1` (see the script's own comment).

See `config/TRAINIUM_CONFIG.md` for the exact environment and hyperparameters, and
`TRAINIUM_ISSUES_LOG.md` for the full error + native-fix record.
