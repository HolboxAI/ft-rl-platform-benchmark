# Miles on Trainium2 (TorchNeuron) — compatibility assessment

Same CholecT50 GRPO/RLVR experiment as the TRL, verl, and SkyRL legs, but with
**Miles** (`radixark/miles`, a fork of `slime`) instead. Goal: determine what
library changes a native Miles-on-Trainium2 run needs.

> This leg is **source-verified**, not instance-verified. No Trainium instance
> was launched: the assessment below is from reading Miles' source
> (`radixark/miles` @ `ba60431`).

## What is here

- `MILES_ON_TRAINIUM_COMPATIBILITY.md` — the assessment (verdict + blockers).
- `MILES_TRAINIUM_ISSUES_LOG.md` — the issues log + the package-changes list.

## Verdict (one line)

Miles is the closest to native of the three RL frameworks. It has a real device
abstraction (`Platform` + `MILES_ACCELERATOR=cuda|tpu`, with `torch.accelerator`
and a `TpuPlatform` precedent) and a documented custom-generate-function hook
for rollout. The port is bounded, not a rewrite.

See `MILES_ON_TRAINIUM_COMPATIBILITY.md` for the full breakdown.
