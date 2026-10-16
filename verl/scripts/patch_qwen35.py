#!/usr/bin/env python3
"""Patch verl's Qwen3.5 decoder layer so the 3D mRoPE position_ids is not
forwarded to the full-attention module.

Root cause of the flash-attention varlen reshape crash (shape '[467, 4, 4, 256]'
invalid for 1,929,216 elements) in the FSDP ref/actor forward:

  * Qwen3.5 uses 3D mRoPE position_ids (temporal / height / width). verl pads and
    transposes them to (4, batch, seq) and passes them down to the decoder layers.
  * qwen3_5_decoder_layer_forward forwards position_ids to Qwen3_5Attention,
    which hands it to transformers' flash-attention wrapper. There
    `_is_packed_sequence` treats the 3D tensor as a "packed sequence" (its
    arange-diff check assumes a 2D (batch, seq) tensor), which routes the call
    into `flash_attn_varlen_func` with a cu_seqlens derived from the 3D tensor —
    a length that does not match the flattened query (off by merge_size^2 = 4).
  * position_ids is only used by flash-attention for that varlen heuristic; RoPE
    is already applied upstream via `position_embeddings`. So the fix is simply
    to stop forwarding position_ids to the attention layer.
"""
import sys

PATH = "/workspace/verl/verl/models/transformers/qwen3_5.py"

OLD = """            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,"""

NEW = """            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,"""

with open(PATH, encoding="utf-8") as f:
    src = f.read()

if OLD not in src:
    print("ERROR: target block not found in qwen3_5.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("patched qwen3_5.py to stop forwarding position_ids to full attention")
