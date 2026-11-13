#!/usr/bin/env python3
"""P2: does training on TPU compute the same gradients and updates as CPU? Runs on a v6e-1.

Why: the CholecT50 TPU run collapsed after one step (KL 1.3 in step 1), and in steps 2-6 the
KL-only gradient moved the policy *away* from the reference. This probe checks the device, not
the recipe. It uses the real Qwen3.5-4B SFT checkpoint and one real val frame with its
ground-truth answer, and applies the same Qwen3.5 patches the run used (M-RoPE without in-place
writes, vision RoPE device fix, FSDP2 move-to-TPU) to every pass, so only the device differs.

  A  plain model, bf16 on TPU      vs  fp32 on CPU   (TorchTPU forward/backward)
  A0 plain model, bf16 on CPU      vs  fp32 on CPU   (how much difference bf16 alone makes)
  B  FSDP2 + CPU offload + bf16 mixed precision on TPU (the training path)  vs  fp32 on CPU
  C  one optimizer step after B: SkyRL's fsdp2_clip_grad_norm_ (max 1.0) + AdamW(1e-5, 0.9/0.999,
     wd 0.01), vs the same first AdamW step computed on CPU from the CPU gradients:
     delta = -lr * g_c / (|g_c| + eps) - lr * wd * p   (exact for the first step from zero state)

Pass criteria: global gradient cosine for A and B close to A0's; C update cosine high (>0.9) and the
pre-clip grad norms close. Writes logs/p2_grad_parity.json.

    python probes/p2_grad_parity.py                 # on the TPU VM
    python probes/p2_grad_parity.py --device cpu    # local dry run: the "TPU" passes run on CPU
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "rollout"))


def log(msg):
    print(f"[p2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/cholec-sft-qwen3.5-4b"))
    ap.add_argument("--parquet", default=os.path.expanduser("~/cholec/skyrl_parquet/val.parquet"))
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--device", default="tpu", choices=["tpu", "cpu"])
    ap.add_argument("--skip", default="", help="comma list of passes to skip: A0,A,B")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "p2_grad_parity.json"))
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))

    os.environ.setdefault("SKYRL_DEVICE", "tpu" if args.device == "tpu" else "cuda")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    import torch
    import torch.distributed as dist
    import pyarrow.parquet as pq
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import mm
    from skyrl.backends.skyrl_train.utils import device as accel

    if args.device == "tpu":
        # tpu_dist needs the slice topology even for one process (run_cholect50 gets it from tpu_env.sh).
        # Port 9470, not 8470: the TPU VM agent always holds 8470.
        os.environ.setdefault("TORCH_TPU_TOPOLOGY", "1,1,1")
        os.environ.setdefault("TORCH_TPU_SLICEBUILDER_ADDRESSES", "localhost:9470")
        os.environ.setdefault("TPU_VISIBLE_DEVICES", "0")
        accel._ensure_backend_loaded()
        accel.patch_fsdp_move_states_for_tpu()
        dev = torch.device("tpu")
        backend = accel.dist_backend()
    else:
        dev = torch.device("cpu")
        backend = "gloo"
        accel.current_device = lambda: torch.device("cpu")  # dry run: SkyRL's clip helper targets the CPU
    # Same model-code patches as the run, applied for every pass (CPU and TPU alike).
    import tpu_compat

    tpu_compat.patch_qwen35_mrope()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm

    def _vision_rope_forward(self, position_ids):
        return (position_ids.unsqueeze(-1) * self.inv_freq.to(position_ids.device)).flatten(1)

    qm.Qwen3_5VisionRotaryEmbedding.forward = _vision_rope_forward
    dist.init_process_group(backend=backend)
    report = {"device": args.device, "model": args.model, "torch": torch.__version__}

    # ---------------------------------------------------------------- inputs
    proc = AutoProcessor.from_pretrained(args.model)
    row = pq.read_table(args.parquet).slice(args.row, 1).to_pylist()[0]
    gt = row["reward_spec"]["ground_truth"]
    gt = json.loads(gt) if isinstance(gt, str) else gt
    answer = ", ".join(gt["triplets"]) if gt.get("triplets") else "none"
    r = mm.render(proc, row["prompt"])
    prompt_ids = r["token_ids"]
    resp_ids = proc.tokenizer(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids_cpu = torch.tensor([prompt_ids + resp_ids])
    pv_cpu, thw_cpu = mm.features_to_tensors(r["features"])
    itid = mm.image_token_id_of(proc)
    n_prompt, n_resp = len(prompt_ids), len(resp_ids)
    report["inputs"] = {"prompt_tokens": n_prompt, "response_tokens": n_resp, "answer": answer,
                        "pixel_values": list(pv_cpu.shape)}
    log(f"inputs: {n_prompt} prompt + {n_resp} response tokens; answer={answer!r}")

    def batch(device, dtype):
        ids = ids_cpu.to(device)
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv_cpu.to(device, dtype),
                    image_grid_thw=thw_cpu.to(device), mm_token_type_ids=(ids == itid).long())

    def response_nll(logits, device):
        # logits[t] predicts token t+1; response tokens sit at positions n_prompt .. end
        lg = logits[0, n_prompt - 1 : -1].float()
        tgt = ids_cpu[0, n_prompt:].to(device)
        return torch.nn.functional.cross_entropy(lg, tgt), lg

    def run_plain(device, dtype, tag):
        t0 = time.time()
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(device)
        # As SkyRL trains (gradient_checkpointing=True, non-reentrant). Without it the full backward of
        # 2.1k tokens + 8k image patches does not fit one 32 GB v6e chip (P2 attempt 2, 2026-10-25).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        out = model(**batch(device, dtype))
        loss, lg = response_nll(out.logits, device)
        loss.backward()
        # .cpu() before .float(): an fp32 copy on the device does not fit next to the bf16 grads (v6e-1 OOM).
        grads = {n: p.grad.detach().cpu().float() for n, p in model.named_parameters() if p.grad is not None}
        res = {"loss": float(loss.item()), "resp_logits": lg.detach().cpu().float(), "seconds": round(time.time() - t0, 1)}
        del model, out, loss, lg
        gc.collect()
        log(f"{tag}: loss={res['loss']:.5f} ({res['seconds']}s, {len(grads)} grads)")
        return res, grads

    def compare(grads, ref_grads, ref, res, tag):
        dot = na = nb = 0.0
        worst = []
        for n, g in grads.items():
            gr = ref_grads.get(n)
            if gr is None:
                continue
            g, gr = g.double().flatten(), gr.double().flatten()
            d, a, b = float(g @ gr), float(g @ g), float(gr @ gr)
            dot, na, nb = dot + d, na + a, nb + b
            if b > 0 and a > 0:
                worst.append((d / math.sqrt(a * b), n, math.sqrt(a / b)))
        worst.sort()
        lr, lg = ref["resp_logits"], res["resp_logits"]
        out = {
            "loss": res["loss"], "loss_ref": ref["loss"],
            "grad_cosine_global": dot / math.sqrt(na * nb) if na and nb else None,
            "grad_norm": math.sqrt(na), "grad_norm_ref": math.sqrt(nb),
            "resp_logits_max_abs_diff": float((lg - lr).abs().max()),
            "resp_argmax_agree": float((lg.argmax(-1) == lr.argmax(-1)).float().mean()),
            "worst_params": [{"param": n, "cosine": round(c, 4), "norm_ratio": round(q, 4)} for c, n, q in worst[:8]],
            "seconds": res.get("seconds"),
        }
        log(f"{tag}: grad cos {out['grad_cosine_global']:.5f}, |g| {out['grad_norm']:.4f} vs {out['grad_norm_ref']:.4f}, "
            f"loss {out['loss']:.5f} vs {out['loss_ref']:.5f}, worst param cos {worst[0][0]:.4f} ({worst[0][1]})")
        return out

    # ------------------------------------------------------- CPU fp32 reference
    ref, ref_grads = run_plain(torch.device("cpu"), torch.float32, "CPU fp32 reference")

    if "A0" not in skip:
        res, grads = run_plain(torch.device("cpu"), torch.bfloat16, "A0 CPU bf16")
        report["A0_cpu_bf16_vs_cpu_fp32"] = compare(grads, ref_grads, ref, res, "A0")
        del grads; gc.collect()

    if "A" not in skip:
        res, grads = run_plain(dev, torch.bfloat16, f"A {args.device} bf16")
        report["A_device_bf16_vs_cpu_fp32"] = compare(grads, ref_grads, ref, res, "A")
        del grads; gc.collect()

    # ------------------------------- B + C: the training path (FSDP2 + CPU offload)
    if "B" not in skip:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

        from skyrl.backends.skyrl_train.distributed.fsdp_utils import fsdp2_clip_grad_norm_

        t0 = time.time()
        mesh = init_device_mesh(dev.type, mesh_shape=(1,), mesh_dim_names=["fsdp"])
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # before fully_shard, as SkyRL does
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        kw = dict(mesh=mesh, mp_policy=mp, offload_policy=CPUOffloadPolicy(pin_memory=False))
        for mod in model.modules():
            if type(mod).__name__.endswith(("DecoderLayer", "VisionBlock")):
                fully_shard(mod, **kw)
        fully_shard(model, **kw)
        model.train()
        out = model(**batch(dev, torch.bfloat16))
        loss, lg = response_nll(out.logits, dev)
        loss.backward()
        local = lambda t: t.to_local() if hasattr(t, "to_local") else t
        grads = {n: local(p.grad).detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
        res = {"loss": float(loss.item()), "resp_logits": lg.detach().float().cpu(), "seconds": round(time.time() - t0, 1)}
        report["B_fsdp_offload_bf16_vs_cpu_fp32"] = compare(grads, ref_grads, ref, res, "B")
        del out, loss, lg

        # C: one optimizer step, SkyRL's settings and clip function
        lr, betas, wd, eps, max_norm = 1e-5, (0.9, 0.999), 1e-2, 1e-8, 1.0
        before = {n: local(p).detach().float().cpu().clone() for n, p in model.named_parameters()}
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd, eps=eps)
        pre_clip = float(fsdp2_clip_grad_norm_(model.parameters(), max_norm=max_norm).float().cpu())
        opt.step()
        after = {n: local(p).detach().float().cpu() for n, p in model.named_parameters()}

        ref_norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in ref_grads.values()))
        coef = min(1.0, max_norm / (ref_norm + 1e-6))
        dot = na = nb = 0.0
        sign_agree = sign_total = 0
        worst = []
        for n, p0 in before.items():
            g = ref_grads.get(n)
            if g is None:
                continue
            gc_ = g * coef
            d_ref = (-lr * gc_ / (gc_.abs() + eps) - lr * wd * p0).double().flatten()
            d_dev = (after[n] - p0).double().flatten()
            a, b, d = float(d_dev @ d_dev), float(d_ref @ d_ref), float(d_dev @ d_ref)
            dot, na, nb = dot + d, na + a, nb + b
            big = gc_.abs().flatten() > gc_.abs().flatten().median()   # elements whose sign is well defined
            sign_agree += int((torch.sign(d_dev.float()[big]) == torch.sign(d_ref.float()[big])).sum())
            sign_total += int(big.sum())
            if a > 0 and b > 0:
                worst.append((d / math.sqrt(a * b), n))
        worst.sort()
        report["C_optimizer_step"] = {
            "pre_clip_grad_norm_device": pre_clip, "grad_norm_cpu_ref": ref_norm, "clip_coef_ref": coef,
            "update_cosine_global": dot / math.sqrt(na * nb) if na and nb else None,
            "update_norm_ratio": math.sqrt(na / nb) if nb else None,
            "sign_agreement_large_grads": sign_agree / max(sign_total, 1),
            "worst_params": [{"param": n, "cosine": round(c, 4)} for c, n in worst[:8]],
            "seconds": round(time.time() - t0, 1),
        }
        c = report["C_optimizer_step"]
        log(f"C: update cos {c['update_cosine_global']:.5f}, norm ratio {c['update_norm_ratio']:.4f}, sign agree "
            f"{c['sign_agreement_large_grads']:.4f}, pre-clip |g| {pre_clip:.4f} vs CPU {ref_norm:.4f}")

    a0 = report.get("A0_cpu_bf16_vs_cpu_fp32", {}).get("grad_cosine_global")
    verdict = {}
    for k in ("A_device_bf16_vs_cpu_fp32", "B_fsdp_offload_bf16_vs_cpu_fp32"):
        if k in report:
            cs = report[k]["grad_cosine_global"]
            verdict[k] = "ok" if a0 is None or cs >= a0 - 0.05 else "DIFFERENT from bf16 noise"
    if "C_optimizer_step" in report:
        verdict["C_optimizer_step"] = "ok" if report["C_optimizer_step"]["update_cosine_global"] > 0.9 else "DIFFERENT"
    report["verdict"] = verdict
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log(f"verdict: {verdict}")
    log(f"wrote {args.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
