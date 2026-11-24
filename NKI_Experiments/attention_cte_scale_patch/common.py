"""Shared helpers: bf16 emulation of attention_cte arithmetic, and the kernel's scale decision
logic before and after attention_cte_fp32_scale_in_exp.patch.

Decision logic is copied from attention_cte.py (L844-883 and L1655-1722) and the patch.
Numerics emulate one head: bf16 Q/K/V, fp32 QK^T (PSUM), fp32 softmax, exp output cast to bf16
for the PV matmul, fp32 normalisation. Reference is float64 attention on the same bf16 inputs.
"""
import math
from dataclasses import dataclass

import ml_dtypes
import numpy as np

BF16 = ml_dtypes.bfloat16
TOL = 2e-2  # repo comparator: rtol, "max" mode (test/utils/comparators.py)
Q_TILE, K_TILE = 128, 512


def bf16(x):
    """Round to bf16, return float32 holding the rounded value."""
    return np.asarray(x, dtype=np.float32).astype(BF16).astype(np.float32)


# ----------------------------------------------------------------------------- decision logic
@dataclass
class Cfg:
    d: int = 128
    scale: float = 1 / math.sqrt(128)
    causal: bool = True
    prefix_caching: bool = False
    sliding_window: int = 0
    cp: bool = False
    kv_used_len: bool = False
    seq_packed: bool = False
    position_bias: bool = False


def _pow2(scale):
    return scale > 0.0 and math.frexp(scale)[0] == 0.5


def decide(c: Cfg, patched: bool):
    """Return where the scale is applied, whether the fast range_select diagonal is on,
    and whether the call runs at all."""
    uses_rs = c.prefix_caching or c.sliding_window > 0 or c.cp or c.kv_used_len
    if patched:
        uses_rs = uses_rs or c.seq_packed
    plain = (c.causal and c.sliding_window == 0 and not c.cp and not c.kv_used_len
             and not c.seq_packed and not c.prefix_caching)
    scale, where, runs, note = c.scale, "MM1 evict, fp32", True, ""
    if patched:
        cond = scale != 1.0 and (uses_rs or (plain and scale > 0.0 and not c.position_bias))
    else:
        cond = scale != 1.0 and (uses_rs or (plain and _pow2(scale) and not c.position_bias))
    if cond:
        if patched and scale > 0.0 and (not _pow2(scale) or c.position_bias):
            where, scale = "exp, fp32", 1.0
        else:
            if c.position_bias:
                runs, note = False, "assert: position_bias needs scale=1 / >0"
            where = ("Q, bf16, exact" if _pow2(c.scale) else "Q, bf16, rounded")
            scale = 1.0
    # sequence packing: dynamic range_select mask path asserts scale == 1.0 (L4290)
    if c.seq_packed and scale != 1.0 and runs:
        runs, note = False, "assert ac.scale == 1.0 (L4290)"
    dynamic = c.cp or c.seq_packed
    is_causal = c.causal and not (c.cp)  # CP w/o strided/striped turns causal off
    fast_diag = is_causal and not dynamic and c.sliding_window == 0 and scale == 1.0
    return {"where": where, "fast_diag": bool(fast_diag), "runs": runs, "note": note}


# ----------------------------------------------------------------------------- numerics
def make_mask(sq, skv, prior=0, window=0, causal=True):
    """True where attention is allowed. Q rows sit after `prior` cached keys."""
    qi = np.arange(sq)[:, None] + prior
    kj = np.arange(skv)[None, :]
    m = np.ones((sq, skv), bool)
    if causal:
        m &= kj <= qi
    if window > 0:
        m &= kj > qi - window
    return m


def make_inputs(sq, skv, d, family, seed, gain=8.0, n_out=4):
    rng = np.random.default_rng(seed)
    if family == "repo_uniform":
        q, k, v = (rng.random((n, d)) for n in (sq, skv, skv))
    else:
        q, k = rng.standard_normal((sq, d)), rng.standard_normal((skv, d))
        v = rng.standard_normal((skv, d))
        if family == "outlier":
            ch = rng.choice(d, n_out, replace=False)
            q[:, ch] *= gain
            k[:, ch] *= gain
    return bf16(q), bf16(k), bf16(v)


def reference(q, k, v, mask, scale):
    s = (q.astype(np.float64) @ k.astype(np.float64).T) * scale
    s = np.where(mask, s, -np.inf)
    p = np.exp(s - s.max(1, keepdims=True))
    return (p / p.sum(1, keepdims=True)) @ v.astype(np.float64)


def kernel(q, k, v, mask, scale, mode):
    """mode: 'evict' (scale on fp32 scores), 'fold' (scale into bf16 Q), 'patch' (scale in exp)."""
    f32 = np.float32
    sc = f32(scale)
    qq = bf16(q * sc) if mode == "fold" else q
    s = (qq @ k.T).astype(f32)                       # fp32 PSUM accumulation
    if mode == "evict":
        s = (s * sc).astype(f32)
    s = np.where(mask, s, f32(-3.0e38))
    neg_max = -s.max(1, keepdims=True)
    if mode == "patch":
        bias = (sc * neg_max).astype(f32)             # exp_scale * neg_max
        e = np.exp((s * sc + bias).astype(f32))
    else:
        e = np.exp((s + neg_max).astype(f32))
    e = np.where(mask, e, f32(0))
    ssum = e.sum(1, keepdims=True, dtype=f32)
    out = (bf16(e) @ v).astype(f32) / ssum
    return out


def rel_err(out, ref):
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
