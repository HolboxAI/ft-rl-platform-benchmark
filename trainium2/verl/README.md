# verl on Trainium2 (TorchNeuron) — CholecT50 GRPO/RLVR leg

Same CholecT50 surgical action-triplet experiment as the TRL leg, but with
**verl + TorchNeuron** instead of TRL. Goal: determine compatibility and list
library fixes for an effortless run.

## What is here

- `plugin/` — a verl `PlatformBase` backend for the `neuron` device
  (`PlatformNeuron`), auto-loaded via the setuptools `verl.plugins` entry point.
  No verl source edit needed.
- `scripts/setup.sh` — verified recipe: verl 0.9.0 + torch-neuronx pin + the two
  dependency overrides (tensordict, transformers), then the plugin.
- `scripts/gate_test.py` — go/no-go gate: plugin loads, model lands on `neuron`,
  forward+backward pass. **Passes.**
- `scripts/launch.sh` — first-attempt verl RL launch (fails at rollout
  selection, issue #6).
- `VERL_TRAINIUM_ISSUES_LOG.md` — the issues log (mirrors the TRL leg).

## Status (2026-10-26, verified on-instance)

- **Platform layer: WORKS.** `PlatformNeuron` registers, `get_platform()`
  returns `neuron`, and a model runs forward+backward on-core (gate passes).
- **verl on Python 3.14: WORKS** with two pin overrides — verl 0.9.0 (not 0.9.1,
  which caps Python `<3.13`), tensordict 0.15.3, transformers 5.18.0.
- **Rollout: BLOCKED.** verl 0.9.0 is async-server-only (vllm/sglang/trtllm).
  The transformers-native rollout (`HFRollout`/`NaiveRollout`) is dead code and
  not selectable. The next step is a `NeuronReplica(RolloutReplica)` — a
  transformers-native inference server on `neuron` — registered in
  `_ROLLOUT_REGISTRY` and `RolloutReplicaRegistry` (issue #6 in the log).

## Order of operations

1. Launch a `trn2.48xlarge` spot in `ap-south-2b` (default profile, keypair
   `trainium-ap-south-2`, AMI `ami-0f7a2cecca50fc858`).
2. `scripts/setup.sh`, then `scripts/gate_test.py` (passes).
3. Write the `NeuronReplica` rollout (issue #6), then attempt `scripts/launch.sh`.
