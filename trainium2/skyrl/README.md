# SkyRL on Trainium2 (TorchNeuron) — compatibility assessment

Same CholecT50 GRPO/RLVR experiment as the TRL and verl legs, but with **SkyRL
(NovaSky-AI)** instead. Goal: determine what library changes a native
SkyRL-on-Trainium2 run needs.

> This leg is **source-verified**, not instance-verified. No Trainium instance
> was launched: SkyRL's incompatibility is fully determined by its source and
> dependency pins (unlike verl, where the device layer was a genuine unknown
> that needed a live run).

## What is here

- `SKYRL_ON_TRAINIUM_COMPATIBILITY.md` — the assessment (verdict + blockers).
- `SKYRL_TRAINIUM_ISSUES_LOG.md` — the issues log + the package-changes list.

## Verdict (one line)

SkyRL is the hardest of the three Trainium legs. It hardcodes CUDA in 34 files
and supports only vLLM for rollout, with no device abstraction to hook `neuron`
into. TRL works natively; verl needed a device plugin + a rollout; SkyRL needs a
device refactor + a new inference engine.

See `SKYRL_ON_TRAINIUM_COMPATIBILITY.md` for the full breakdown.
