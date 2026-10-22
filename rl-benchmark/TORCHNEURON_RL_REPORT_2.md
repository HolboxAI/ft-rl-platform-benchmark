# RL on Trainium with TorchNeuron: status report


## 1. Summary

This report gives the results of reinforcement learning (RL) tests on AWS Trainium.
The tests use torch-neuronx 2.14.3 in the native TorchNeuron mode on a `trn1.2xlarge` instance.

The training part of GRPO operates correctly. The model forward pass, the LoRA backward pass and the optimizer step operate on Trainium.

The rollout part of GRPO does not operate. Token generation fails in Hugging Face TRL and in our GRPO loop. The probable cause is the same in the two loops: the incorrect processing of int64 values.

No GRPO step is complete on Trainium. Miles does not operate on Trainium.

## 2. Test configuration

| Item | Value |
|---|---|
| Hardware | `trn1.2xlarge`: 1 Trainium1 chip, 2 NeuronCores, 32 GB HBM. The tests use 1 NeuronCore. Region: us-east-1. |
| Software | Neuron DLAMI `torch-neuronx-2.14.0`. torch 2.14.0. torch-neuronx 2.14.3.0.1. neuronx-cc 2.28.7631. transformers 5.19.0. peft 0.21.2. |
| torch-neuronx mode | Native TorchNeuron. The device name is `"neuron"`. The mode is eager. The software does not include `torch_xla`. |
| Task | CholecT50 surgical action triplets. |
| Model | Merged SFT checkpoint `cholec-sft-qwen3.5-4b` (Qwen3.5-4B vision-language model), with LoRA rank 16. |
| Reward | Triplet F1 reward: 0.4 × triplet F1 + 0.6 × mean(instrument F1, verb F1, target F1). |

The tests do not include these items:
- Trn2 and Trn3 hardware. TorchNeuron supports these two families. The account quota does not permit them.
- The XLA-based torch-neuronx (PyTorch 2.9 and earlier). This software has official support on Trn1.

## 3. Results

### 3.1 Results for each RL method

| Method | Last step that operates correctly | Cause of the stop |
|---|---|---|
| Miles | `import miles` operates. 480 of 645 modules import. | Miles has no Neuron platform. `MILES_ACCELERATOR` accepts only `cuda` and `tpu`. The SGLang rollout engine does not install. Refer to `miles/TRAINIUM_INCOMPATIBILITY.md`. |
| Hugging Face TRL `GRPOTrainer` | accelerate selects the `"neuron"` device. TRL, transformers and peft import. The Qwen3.5 vision tower compiles and operates. | The `generate` function gives incorrect `input_ids` to the model. The image token count is 203, then 0. The correct count is 405. |
| Our PyTorch GRPO loop | Model load, forward pass and LoRA training step operate. | Sampling fails with this error: `NRT EXECUTION FAILED … aten::embedding … out of bound`. |

### 3.2 Measured times (our GRPO loop, run `grpo-trn-20261022-1454-mini2`)

| Step | First call (includes compilation) | Second call |
|---|---|---|
| Load the model to the device (4B model and LoRA) | 1,913 s (32 min) | Not applicable |
| Forward pass, 1 frame (470-token prompt and 1 image) | 442 s | 18 s |
| LoRA training step (forward pass, backward pass, AdamW) | 604 s | 62 s |
| Sampling, 16 tokens | Failed | Not applicable |

## 4. Defects in torch-neuronx 2.14.3 (native TorchNeuron)

| No. | Defect | Cause | Status |
|---|---|---|---|
| 1 | `repeat_interleave` with tensor repeat counts does not compile. The Qwen3.5 vision index functions use this operation. | The output shape changes with the data. The Torch-MLIR to StableHLO conversion uses a fixed shape. | Temporary solution: `common/neuron_compat.py` calculates these indexes on the CPU. The vision tower then operates. |
| 2 | The `generate` function in transformers changes the `input_ids` values. | The cause is not known. A possible cause is int64 index calculation in the prefill step. | Open. This defect stops TRL. |
| 3 | `int64.sum()` and `int64.cumsum()` give incorrect results for large values. | The sum uses float32 accumulation. The result is correct only below 2²⁴. The cumsum conversion is incorrect. | Open. RL values at usual sizes are below the limit. 18 of 20 mask and count operations give correct results. |
| 4 | `torch.rand` and `torch.randn` on the device cause a `RecursionError`. | The TorchNeuron function `rand_decomp` calls `torch.empty` on the device. This call starts `rand_decomp` again. | Temporary solution: the code makes the random numbers on the host. |
| 5 | An embedding lookup reads outside the table during sampling. | The cause is not known. A probable cause: the device writes incorrect int64 token IDs (`argmax`, then `index_copy_`). This defect is possibly related to defect 2. | Open. This defect stops the rollouts in our loop. |

## 5. Analysis

Rollout generation is the primary problem. The two generation methods fail on int64 token values (defects 2 and 5). The training path operates with the temporary solutions for defects 1 and 4.

Compilation and data transfer take a long time. The first call of each graph takes 7 to 10 minutes. The transfer of a 4B model to the device takes 32 minutes. The second calls take a usual time.

Native TorchNeuron does not have official support on Trn1. Some defects can occur only on Trn1.

