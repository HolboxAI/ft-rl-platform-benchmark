"""TTFT and decode TPS for Qwen3-0.6B (head_dim 128) on Trainium, one request at a time.

Prefill attention in all layers calls whichever nkilib `attention_cte` is first on PYTHONPATH
(original or patched), through nki_attn.py. Decode attention is plain torch ops with a fixed-size
KV cache and is the same code for both versions: the patched kernel is NOT used in decode.
KERNEL=torch replaces the prefill kernel call with plain torch attention (correctness reference).

  python e2e_ttft_tps.py <prompt_len> <out.json>
"""

import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

S, OUT = int(sys.argv[1]), sys.argv[2]
NEW = 64                      # tokens decoded after the first one
MAXLEN = S + NEW
DECODE = os.environ.get("DECODE", "0") == "1"   # off by default: decode does not use the patched kernel
USE_NKI = os.environ.get("KERNEL", "nki") == "nki"
if USE_NKI:
    import nki_attn
    import nkilib.core.attention.attention_cte as km

def torch_attention(q, k, v, mask, scale):
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    return torch.matmul(s.softmax(-1), v.float()).to(q.dtype)


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model = model.eval().to("neuron")
cfg = model.config
SCALE = 1.0 / math.sqrt(cfg.head_dim)
REP = cfg.num_attention_heads // cfg.num_key_value_heads


def forward(ids, position_ids, attend):
    """Qwen3 forward as a pure function over the HF modules; attend(i, q, k, v) -> (out, cache_i)."""
    body = model.model
    x = body.embed_tokens(ids)
    cos, sin = body.rotary_emb(x, position_ids)
    caches = []
    for i, layer in enumerate(body.layers):
        a = layer.self_attn
        h = layer.input_layernorm(x)
        B, L = h.shape[0], h.shape[1]
        q = a.q_norm(a.q_proj(h).view(B, L, -1, cfg.head_dim)).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).view(B, L, -1, cfg.head_dim)).transpose(1, 2)
        v = a.v_proj(h).view(B, L, -1, cfg.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out, cache = attend(i, q.contiguous(), k.contiguous(), v.contiguous())
        caches.append(cache)
        x = x + a.o_proj(out.transpose(1, 2).reshape(B, L, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    logits = model.lm_head(body.norm(x)[:, -1]).float()
    return logits, caches


def prefill_fn(ids, position_ids, pad):
    def attend(i, q, k, v):
        if USE_NKI:
            out = nki_attn.attn(q, k, v, SCALE)
        else:
            L = q.shape[2]
            mask = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
            out = torch_attention(q, k.repeat_interleave(REP, 1), v.repeat_interleave(REP, 1), mask, SCALE)
        return out, (torch.cat([k, pad], 2), torch.cat([v, pad], 2))
    logits, caches = forward(ids, position_ids, attend)
    return logits, logits.argmax(-1, keepdim=True), caches


def decode_fn(tok, pos, idx, past):
    """One decode step: write the new key/value at `pos`, attend over positions <= pos."""
    def attend(i, q, key, value):
        sel = (idx == pos).view(1, 1, -1, 1)
        k = torch.where(sel, key, past[i][0])
        v = torch.where(sel, value, past[i][1])
        out = torch_attention(q, k.repeat_interleave(REP, 1), v.repeat_interleave(REP, 1),
                              (idx <= pos).view(1, 1, 1, -1), SCALE)
        return out, (k, v)
    logits, caches = forward(tok, pos.view(1, 1), attend)
    return logits, logits.argmax(-1, keepdim=True), caches


prefill = torch.compile(prefill_fn, backend="neuron", fullgraph=True)
decode = torch.compile(decode_fn, backend="neuron", fullgraph=True)


ids_cpu = torch.randint(1000, 20000, (1, S), generator=torch.Generator().manual_seed(0))
pad = torch.zeros(1, cfg.num_key_value_heads, NEW, cfg.head_dim, dtype=torch.bfloat16).to("neuron")
position_ids = torch.arange(S).unsqueeze(0).to("neuron")
idx = torch.arange(MAXLEN).to("neuron")


def run_prefill():
    """TTFT: token ids on the host -> first generated token id on the host."""
    t = time.perf_counter()
    logits, tok, caches = prefill(ids_cpu.to("neuron"), position_ids, pad)
    first = int(tok.cpu())
    return time.perf_counter() - t, logits, tok, caches, first


def run_decode(tok, caches, n):
    toks, first_logits, times = [], None, []
    for step in range(n):
        pos = torch.tensor(S + step).to("neuron")
        t = time.perf_counter()
        logits, tok, caches = decode(tok, pos, idx, caches)
        toks.append(int(tok.cpu()))
        times.append(time.perf_counter() - t)
        if step == 0:
            first_logits = logits.cpu()
    return toks, first_logits, times


with torch.no_grad():
    t_compile = time.time()
    _, logits, tok, caches, first = run_prefill()      # compiles prefill
    if DECODE:
        run_decode(tok, caches, 2)                     # compiles decode
    compile_s = time.time() - t_compile
    for _ in range(3):                                 # warm-up
        run_prefill()
    ttft = [run_prefill()[0] for _ in range(20)]
    tps, step_ms, total = [], [], []
    for _ in range(5 if DECODE else 0):
        t0 = time.perf_counter()
        dt, _, tok, caches, first = run_prefill()
        toks, dec_logits, times = run_decode(tok, caches, NEW)
        total.append(time.perf_counter() - t0)
        tps.append(NEW / sum(times))
        step_ms += [x * 1e3 for x in times]

torch.save({"prefill_logits": logits.cpu(), "decode_logits": dec_logits if DECODE else None,
            "tokens": [first] + (toks if DECODE else [])}, OUT + ".pt")
res = {"prompt_len": S, "new_tokens": NEW + 1, "kernel": os.environ.get("KERNEL", "nki"),
       "kernel_file": os.path.realpath(km.__file__) if USE_NKI else None,
       "logical_neuron_cores": nki_attn.LNC if USE_NKI else None,
       "compile_s": compile_s, "ttft_ms": [x * 1e3 for x in ttft], "decode_tps": tps, "decode_step_ms": step_ms,
       "e2e_tps": [(NEW + 1) / x for x in total]}
json.dump(res, open(OUT, "w"))
med = lambda a: sorted(a)[len(a) // 2] if a else float("nan")
print("RESULT", S, res["kernel"], "compile_s %.0f ttft_ms %.2f decode_tps %.1f e2e_tps %.1f" %
      (compile_s, med(res["ttft_ms"]), med(tps), med(res["e2e_tps"])))
