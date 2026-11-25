# attention_cte: Softmax-Scale Patch A/B

HolboxAI Model Performance Engineering

November 25, 2026

## 1 Overview

This report isolates the effect of one patch to the nkilib **prefill attention kernel** (`attention_cte`). For a non-power-of-two softmax scale (1/√128), the patch keeps scores in fp32 and applies the scale inside the softmax `exp`, instead of multiplying it into the bf16 Q tile on `range_select` paths. It also enables sequence packing at this scale.

### 1.1 Setup

- **Hardware:** trn2.48xlarge (16× Trainium2, 64 NeuronCores), spot, ap-south-2.
- **Software:** SDK 2.33.1, torch-neuronx 2.14, NKI 0.7.0, neuronx-cc 2.28.
- **A/B:** two library copies differing only in `attention_cte.py` (base `ad2f1b9`; patch: 2 files, +100 / −22), run order alternated.
- **Model test:** Qwen3-0.6B, BF16, 28 layers, head size 128, 2 logical NeuronCores; a custom forward pass calls the kernel directly. Concurrency 1, random token IDs; 3 repeats, each the median of 20 warm requests.
- **Kernel test:** Qwen3-235B per-device shape (batch 16, head size 128, causal), device time, 5 repeats.

### 1.2 Position in the inference pipeline

```text
Audio input -> Audio encoder -> Decoder prefill -> First token -> Decode loop
                     ^                  ^
          Audio-MLP NKI report   This patch: attention_cte
           (encoder MLP)         (prefill self-attention)
|<----------------------- TTFT window ------------------>|
```

**Figure 1:** Position of each change, drawn on a speech-to-text pipeline. In a text-only model the pipeline starts at decoder prefill. This patch was measured on the text-only Qwen3-0.6B.

Both changes act before the first token, so both belong to the TTFT window. They target different blocks: the reference report changes the **audio encoder's MLP**; this patch changes **self-attention in decoder prefill**.

## 2 Results

### 2.1 Model TTFT and prefill throughput

| Prompt tokens | TTFT orig (ms) | TTFT patched (ms) | Change | TPS orig | TPS patched | Change |
|---|---|---|---|---|---|---|
| 512 | 13.38 | 13.36 | -0.15% | 38,258 | 38,315 | +0.15% |
| 2,048 | 36.87 | 37.21 | +0.91% | 55,539 | 55,036 | -0.91% |
| 4,096 | 94.66 | 94.04 | -0.65% | 43,272 | 43,554 | +0.65% |

**Table 1:** Qwen3-0.6B. TPS = prefill input tokens/s (prompt length ÷ TTFT).

### 2.2 Prefill kernel latency

| Prompt tokens | Original (ms) | Patched (ms) | Change in time |
|---|---|---|---|
| 512 | 0.107 | 0.101 | -4.99% (within spread) |
| 1,024 | 0.216 | 0.219 | +1.30% (within spread) |
| 2,048 | 0.505 | 0.535 | +5.85% |
| 4,096 | 1.354 | 1.330 | -1.75% |
| 8,192 | 4.427 | 4.397 | -0.67% |

**Table 2:** One `attention_cte` call. Within spread: paired runs moved in both directions.

## 3 Conclusion

| Metric | Original | Patched | Change | Faster / better |
|---|---|---|---|---|
| TTFT, 512 tokens (ms) | 13.38 | 13.36 | -0.15% | Even (<1%) |
| TTFT, 2,048 tokens (ms) | 36.87 | 37.21 | +0.91% | Even (<1%) |
| TTFT, 4,096 tokens (ms) | 94.66 | 94.04 | -0.65% | Even (<1%) |
| Kernel latency, 2,048 tokens (ms) | 0.505 | 0.535 | +5.85% | Original |
| Kernel latency, 4,096 tokens (ms) | 1.354 | 1.330 | -1.75% | Patched |
| Kernel latency, 8,192 tokens (ms) | 4.427 | 4.397 | -0.67% | Even (<1%) |
| Sequence packing, head size 128 | Fails | Passes | Fixed | Patched |

**Table 3:** Summary comparison.

### 3.1 Interpretation

The patch is a **functional fix**: sequence packing now runs at head size 128. **TTFT and prefill TPS stay within 1%** at all three prompt lengths. At kernel level the effect depends on length: slower at 2,048 tokens, slightly faster at 4,096 and 8,192. In the 9 model outputs checked against plain PyTorch attention (of 18 timed runs), the first token and top 10 candidates matched.
