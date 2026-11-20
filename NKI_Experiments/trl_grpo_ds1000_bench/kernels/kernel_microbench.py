"""Op-level micro-benchmark of the operations that dominate a GRPO step.

Shapes come from the benchmark model (Qwen2.5-Coder-0.5B: hidden 896, MLP 4864, 14 query /
2 KV heads, head_dim 64, vocab 151936) and the RL batch (8 sequences x 768 tokens per training
micro-batch, 32 sequences per decode step). Each op runs as:

  impl=torch  stock PyTorch op   (CUDA kernels on g6e; neuronx-cc compiled graph on Trainium)
  impl=nki    hand-written NKI kernel (Trainium only; see bench/nki_ops.py)

so one CSV answers both "is Trainium faster than the GPU for this op?" and "does the NKI
kernel beat the compiler on Trainium?". Results -> results/kernels_<platform>_<time>/kernel_microbench.csv

  python -m kernels.kernel_microbench --platform g6e.xlarge
  python -m kernels.kernel_microbench --platform trn2.3xlarge
"""

import argparse
import os
import statistics
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench import platform_utils as pu  # noqa: E402
from metrics.csv_store import CsvWriter, KERNEL_FIELDS  # noqa: E402

H, I, NH, NKV, HD, V = 896, 4864, 14, 2, 64, 151936
TRAIN_B, TRAIN_S, GEN_B, KV_LEN = 8, 768, 32, 768
T = TRAIN_B * TRAIN_S


def timed(fn, backend, warmup, iters):
    # Keep each result referenced until after the sync: on XLA, a lazy tensor that is dropped
    # before mark_step is never computed, which would time an empty graph.
    pu.synchronize(backend)
    t0 = time.perf_counter()
    out = fn()
    pu.synchronize(backend)
    first = (time.perf_counter() - t0) * 1e3          # includes compile on Neuron
    del out
    for _ in range(warmup):
        out = fn()
        pu.synchronize(backend)
        del out
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        pu.synchronize(backend)
        times.append((time.perf_counter() - t0) * 1e3)
        del out
    times.sort()
    return {"first_call_ms": first, "median_ms": statistics.median(times),
            "p10_ms": times[int(0.1 * (len(times) - 1))], "p90_ms": times[int(0.9 * (len(times) - 1))],
            "mean_ms": statistics.fmean(times)}


def build_ops(torch, dev, dtype, nki_ok, flash_fwd):
    F = torch.nn.functional

    def rnd(*shape, grad=False):
        return torch.randn(*shape, dtype=dtype).to(dev).requires_grad_(grad)

    ops = []   # (op, impl, shape_str, fn, flops, bytes)

    # GEMMs: compute-bound training shape and memory-bound decode shape
    x_t, w_up = rnd(T, H), rnd(H, I)
    ops.append(("matmul_mlp_up_train", "torch", f"{T}x{H}x{I}", lambda: x_t @ w_up,
                2 * T * H * I, 2 * (T * H + H * I + T * I)))
    x_d = rnd(GEN_B, H)
    ops.append(("matmul_mlp_up_decode", "torch", f"{GEN_B}x{H}x{I}", lambda: x_d @ w_up,
                2 * GEN_B * H * I, 2 * (GEN_B * H + H * I + GEN_B * I)))

    # Linear forward + backward (what the policy update does in every layer)
    x_g, w_g = rnd(T, H, grad=True), rnd(I, H, grad=True)

    def lin_fb():
        y = F.linear(x_g, w_g)
        y.float().sum().backward()
        g = (x_g.grad, w_g.grad)      # returned so the backward graph is actually executed
        x_g.grad = None
        w_g.grad = None
        return g
    ops.append(("linear_fwd_bwd_train", "torch", f"{T}x{H}x{I}", lin_fb, 6 * T * H * I, None))

    # GRPO's signature op: logits -> log_softmax -> gather selected-token log-probs
    M = 1024
    h_lm, w_lm = rnd(M, H), rnd(V, H)
    ids = torch.randint(0, V, (M, 1)).to(dev)

    def logprob():
        logits = F.linear(h_lm, w_lm).float()
        return torch.gather(logits.log_softmax(-1), 1, ids)
    ops.append(("lm_head_logprob_gather", "torch", f"{M}x{H}x{V}", logprob,
                2 * M * H * V, 2 * V * H + 4 * M * V * 2))

    # attention (GQA heads expanded to 14), prefill (causal) and single-token decode.
    # Causal FLOPs = half of the full S x S score/value matmuls. s2048 exists because the NKI
    # flash-attention kernel needs seqlen divisible by 2048.
    for S in (TRAIN_S, 1024, 2048):
        q, k, v = rnd(TRAIN_B, NH, S, HD), rnd(TRAIN_B, NH, S, HD), rnd(TRAIN_B, NH, S, HD)
        ops.append((f"attention_prefill_s{S}", "torch", f"{TRAIN_B}x{NH}x{S}x{HD}",
                    lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                    2 * TRAIN_B * NH * S * S * HD, 2 * 4 * TRAIN_B * NH * S * HD))
        if flash_fwd is not None and S % 2048 == 0:
            # NKI flash attention layout: q,k [b, h, d, s]; v [b, h, s, d]
            qn, kn = q.transpose(2, 3).contiguous(), k.transpose(2, 3).contiguous()
            ops.append((f"attention_prefill_s{S}", "nki", f"{TRAIN_B}x{NH}x{S}x{HD}",
                        lambda qn=qn, kn=kn, v=v: flash_fwd[TRAIN_B, NH](qn, kn, v, None, use_causal_mask=True,
                                                                        mixed_precision=True),
                        2 * TRAIN_B * NH * S * S * HD, 2 * 4 * TRAIN_B * NH * S * HD))
        # (2*B*H*S*S*D = causal half of QK^T + PV, i.e. 4*B*H*S*S*D / 2)
    qd, kd, vd = rnd(GEN_B, NH, 1, HD), rnd(GEN_B, NH, KV_LEN, HD), rnd(GEN_B, NH, KV_LEN, HD)
    ops.append(("attention_decode", "torch", f"{GEN_B}x{NH}x1x{KV_LEN}x{HD}",
                lambda: F.scaled_dot_product_attention(qd, kd, vd),
                4 * GEN_B * NH * KV_LEN * HD, 2 * 2 * GEN_B * NH * KV_LEN * HD))

    # RMSNorm and SwiGLU activation (elementwise / reduction, memory-bound)
    from bench.nki_ops import torch_rmsnorm
    xr, gr = rnd(T, H), rnd(H)
    ops.append(("rmsnorm", "torch", f"{T}x{H}", lambda: torch_rmsnorm(xr, gr), 4 * T * H, 2 * 2 * T * H))
    if nki_ok:
        from bench.nki_ops import nki_rmsnorm
        ops.append(("rmsnorm", "nki", f"{T}x{H}", lambda: nki_rmsnorm(xr, gr), 4 * T * H, 2 * 2 * T * H))
    xg, xu = rnd(T, I), rnd(T, I)
    ops.append(("silu_mul", "torch", f"{T}x{I}", lambda: F.silu(xg) * xu, 5 * T * I, 2 * 3 * T * I))
    return ops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--platform", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--output-root", default=os.path.join(ROOT, "results"))
    a = p.parse_args()

    import torch
    backend = pu.detect_backend()
    dev = pu.get_device(backend)
    dtype = torch.bfloat16
    nki_ok, nki_reason, flash_fwd = False, "not a neuron backend", None
    if backend == "neuron":
        from bench.nki_ops import available, get_flash_fwd
        nki_ok, nki_reason = available()
        flash_fwd, flash_reason = get_flash_fwd()
        if flash_fwd is None:
            print(f"[kernels] NKI flash attention unavailable: {flash_reason}")
    if not nki_ok:
        print(f"[kernels] NKI kernels skipped: {nki_reason}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(a.output_root, f"kernels_{a.platform}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    w = CsvWriter(os.path.join(out_dir, "kernel_microbench.csv"), KERNEL_FIELDS)

    for op, impl, shape, fn, flops, nbytes in build_ops(torch, dev, dtype, nki_ok, flash_fwd):
        row = {"platform": a.platform, "backend": backend, "op": op, "impl": impl, "shape": shape,
               "dtype": "bf16", "iters": a.iters}
        try:
            r = timed(fn, backend, a.warmup, a.iters)
            row.update(r)
            s = r["median_ms"] / 1e3
            row["tflops"] = flops / s / 1e12 if flops else None
            row["gb_per_s"] = nbytes / s / 1e9 if nbytes else None
            row["status"] = "ok"
        except Exception as e:
            row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
        w.write(row)
        print(f"[kernels] {op:28s} {impl:5s} {row.get('median_ms', '-')!s:>10} ms  {row['status'][:60]}", flush=True)
    w.close()
    print(f"[kernels] -> {out_dir}/kernel_microbench.csv")


if __name__ == "__main__":
    main()
