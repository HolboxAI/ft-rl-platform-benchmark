# Trainium2 — CholecT50 GRPO (Experiment A)

This directory is the **Trainium2** leg of the `ft-rl-platform-benchmark`. It runs
the same CholecT50 surgical action-triplet GRPO (RLVR) experiment as the Miles,
verl, and SkyRL legs, on AWS Trainium2 hardware through the TorchNeuron
(`torch-neuronx`) native PyTorch backend.

## Status

**The platform go/no-go passed.** A full GRPO step and the final eval both
complete on a single Trainium2 NeuronCore. This is a smoke run, not a comparable
RL result.

| Metric | Value |
|---|---|
| State | succeeded (4-step smoke) |
| Val reward (eval) | **0.2958** |
| Train reward (step 1) | 0.4 (std 3.186e-08) |
| Train loss | 0.0002976 |
| SFT baseline (Miles/verl reference) | 0.31875 |
| Miles final (full run) | 0.65125 |
| verl final (full run) | 0.715 |

The reward is flat (std ~3e-08): the SFT checkpoint emits near-identical
completions at temperature 0.7, so GRPO advantages are ~0 and the model does not
move. That is an RL exploration issue, not a Trainium2 issue. See
`metrics/final_results.json` for the full record.

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
├── TRAINIUM_ISSUES_LOG.md   the bugfix / native-fix doc (16 issues, with asks for torch_neuronx)
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

## Reproduce

1. Launch a `trn2.48xlarge` spot instance (ap-south-2b works).
2. Run `scripts/setup2.sh` to install the pinned transformers/trl/accelerate on
   top of the DLAMI venv.
3. Run `scripts/download_model.sh` and `scripts/download_data.sh` (needs the
   presigned S3 URLs in `/home/ubuntu/urls.sh`).
4. Gate checks: `push_run.sh gate_test.py` and `push_run.sh llm_gate_test.py`.
5. Full loop: `scripts/launch_grpo.sh`, then `scripts/poll_grpo.sh`.

See `config/TRAINIUM_CONFIG.md` for the exact environment and hyperparameters.
