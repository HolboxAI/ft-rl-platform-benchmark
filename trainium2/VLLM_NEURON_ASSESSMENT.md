# vllm-neuron — does it unblock the three RL libs on Trainium2?

Goal: state whether the `vllm-neuron` plugin lets verl, SkyRL, or Miles run the
CholecT50 GRPO/RLVR loop natively on Trainium2.

> **Source-verified 2026-10-26** against the vllm-neuron repo, AWS Neuron
> release notes, and the three frameworks' source. No instance was launched.

## What vllm-neuron is

- A beta community hardware plugin for vLLM on AWS Neuron (Trn2/Trn3).
- It **serves** a fixed model. It does not train.
- Versioning looks like `0.21.0.1.0.0` (vLLM 0.21.0 + plugin 1.0.0). It is a
  separate distribution from the vLLM that verl and SkyRL pin.

## Supported models

- Llama 3 1B / 8B / 70B.
- GPT-OSS 20B / 120B.
- Qwen3-VL 32B (Dense).
- Qwen3-Embedding 8B.

## Not supported

- LoRA adapters.
- Weight reload / sleep mode.
- Chunked prefill.

## Why it does not help THIS benchmark

1. **Model mismatch.** Our model is `cholec-sft-qwen3.5-4b` (Qwen3.5-4B VLM,
   Gated DeltaNet linear attention). vllm-neuron supports Qwen3-VL, a different
   model with different attention. Gated DeltaNet is not on the list.
2. **No weight reload.** RLVR updates the rollout's weights every step.
   vllm-neuron serves a fixed model and cannot reload weights.
3. **Version pinning.** verl and SkyRL pin their own vLLM versions. vllm-neuron
   is a different distribution with its own versioning.
4. **Miles uses SGLang, not vLLM.** vllm-neuron is irrelevant to Miles.

## Would a supported model change the answer?

No. Even with a supported model, weight sync is the crux.

The RLVR loop is: train, push new policy weights into the rollout, generate,
train. The rollout engine's weights are **not** static. vllm-neuron is built to
serve a **fixed** model. Two mechanisms would be needed, and neither works:

- **In-place weight sync** (NCCL/IPC). This is CUDA-only.
- **Checkpoint reload** ("weight reload"). This is on the unsupported list.

So the rollout can generate, but it can never receive the trained policy.

## Would weight sync + more models fix everything?

No. That closes one blocker, for one lib. It does not touch the others.

### verl — the only lib this helps

Adding weight sync + our model would remove verl's rollout blocker, but:

1. "Add weight sync" is not a small patch. It re-implements vLLM's dynamic
   `update_weights` on a static server, matched to verl's exact contract (shard
   layout, offload, quant).
2. verl's package pins still stand: Python<3.13, tensordict, transformers cap,
   undeclared orjson. These are independent of weight sync.
3. "Add our model" re-opens the model walls: Gated DeltaNet kernel fallback,
   TDRAM exhaustion, dynamic `repeat_interleave`.

### SkyRL — vllm-neuron does nothing here

SkyRL's hard blocker is its **training** device layer (34 files hardcode
`torch.cuda`). vllm-neuron touches only rollout. SkyRL still cannot train on
neuron until that refactor happens. Its CUDA torch/vLLM dependency pin also
stays.

### Miles — untouched

Miles rolls out on SGLang, not vLLM. vllm-neuron weight sync is irrelevant.
Miles still needs a `NeuronPlatform`, the ~10 FSDP `torch.cuda` migrations, and
a transformers-native generate function.

## Bottom line

RLVR has two halves: train and rollout. vllm-neuron only ever addresses the
rollout half, and only for vLLM-based frameworks. There are three independent
problem classes:

1. **Train device layer** — blocks SkyRL (34 files) and Miles (partial).
2. **Weight sync** — blocks verl and SkyRL.
3. **Model support** — blocks everyone, and re-opens the model walls.

vllm-neuron fixes none of them by itself. Adding weight sync + models closes
class 2 for verl only, and even then leaves verl's pins and the model walls. It
moves SkyRL and Miles not at all.

For the benchmark now: TRL-on-TorchNeuron (transformers-native `model.generate`)
remains the working path. vllm-neuron is an inference path, not a training-loop
path.

## Sources

- <https://github.com/vllm-project/vllm-neuron>
- <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/vllm-neuron.html>
