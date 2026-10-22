#!/usr/bin/env python3
"""Stage check for TRL on Trainium (native TorchNeuron). Stops at the first failure.

  1. neuron      torch_neuronx imports, torch.neuron reports devices, a tensor op runs on "neuron"
  2. accelerate  PartialState() picks the neuron device (accelerate >= 1.15: is_neuron_available)
  3. stack       transformers / trl / peft import; versions recorded
  4. model       CholecT50 SFT checkpoint (Qwen3.5-4B VLM) loads on the device; one val frame,
                 greedy 16 tokens, scored with the triplet reward

    python hf-trl/scripts/check_env.py --out hf-trl/logs/check_env.json [--skip-model]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def op_correctness() -> dict:
    """Integer / boolean ops that TRL and transformers use for masks and counts, on neuron vs CPU.

    Motivated by run trl-trn-20261022-1141-r2: `(input_ids == image_token_id).sum()` returned 203 on
    neuron for a frame with 405 image tokens. Each case runs on CPU and on "neuron" and is compared
    exactly (ints/bools) or with a tolerance (floats). Mismatches are recorded, never raised.
    """
    import torch

    g = torch.Generator().manual_seed(0)
    tok = 248056  # Qwen3.5 image_token_id
    ids = torch.randint(0, 250000, (1, 2304), generator=g, dtype=torch.long)
    ids[0, 100:505] = tok                                   # 405 image tokens, like a CholecT50 frame
    ids32 = ids.to(torch.int32)
    mask = ids == tok
    resp = torch.randint(0, 5, (16, 256), generator=g)       # completion ids with some "EOS" = 0
    x = torch.randn(16, 256, generator=g)
    idx = torch.randint(0, 256, (16, 256, 1), generator=g)

    cases = {
        "eq_int64_sum": lambda d: (ids.to(d) == tok).sum(),
        "eq_int32_sum": lambda d: (ids32.to(d) == tok).sum(),
        "bool_sum": lambda d: mask.to(d).sum(),
        "bool_to_long_sum": lambda d: mask.to(d).long().sum(),
        "bool_to_int_sum": lambda d: mask.to(d).int().sum(),
        "bool_sum_dim": lambda d: (resp.to(d) == 0).sum(dim=1),
        "int64_sum": lambda d: ids.to(d).sum(),
        "int64_cumsum": lambda d: ids.to(d).cumsum(-1)[0, -5:],
        "bool_cumsum": lambda d: mask.to(d).long().cumsum(-1)[0, -1],
        "eq_argmax_int": lambda d: (resp.to(d) == 0).int().argmax(dim=1),
        "nonzero_count": lambda d: torch.tensor(mask.to(d).nonzero().shape[0]),
        "masked_select_count": lambda d: torch.tensor(ids.to(d).masked_select(mask.to(d)).numel()),
        "any_all": lambda d: torch.stack([mask.to(d).any(), mask.to(d).all()]),
        "gather_logprob": lambda d: x.to(d).unsqueeze(-1).expand(-1, -1, 1).gather(-1, idx.to(d) % 1).squeeze(-1).sum(),
        "masked_mean": lambda d: (x.to(d) * (resp.to(d) != 0)).sum() / (resp.to(d) != 0).sum(),
        "float_sum": lambda d: x.to(d).sum(),
        # Suspects for the in-model 405 -> 203 miscount (run -r3: standalone counts were right):
        # 1) attention-mask-style cumsum (small values) as transformers computes position_ids
        "attn_mask_cumsum": lambda d: (torch.ones(1, 2304, dtype=torch.long).to(d).cumsum(-1) - 1)[0, -3:],
        # 2) the kernel cache reusing a compiled `== const` for a different constant (log warns
        #    NKI_ENABLE_TRACE_CACHE "may not capture ... non-argument values"); Qwen3.5 compares
        #    image_token_id then video_token_id back to back on the same tensor.
        "eq_two_consts": lambda d: torch.stack([(ids.to(d) == tok).sum(), (ids.to(d) == tok + 1).sum(),
                                                (ids.to(d) == tok).sum()]),
        "eq_then_unsqueeze_sum": lambda d: (ids.to(d) == tok).unsqueeze(-1).sum(),
        "int64_sum_small": lambda d: (ids.to(d) % 7).sum(),
    }
    results, n_wrong = {}, 0
    for name, fn in cases.items():
        try:
            ref = fn("cpu")
            got = fn("neuron").cpu()
            if ref.dtype.is_floating_point:
                ok = bool(torch.allclose(got.float(), ref.float(), rtol=1e-3, atol=1e-3))
            else:
                ok = bool(torch.equal(got.to(ref.dtype), ref))
            results[name] = {"match": ok, "cpu": ref.flatten()[:6].tolist(), "neuron": got.flatten()[:6].tolist()}
        except Exception as e:  # an op that does not run is also a finding
            ok = False
            results[name] = {"match": False, "error": f"{type(e).__name__}: {e}"[:300]}
        n_wrong += not ok
    return {"ok": n_wrong == 0, "n": len(cases), "n_wrong": n_wrong, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/cholec-sft-qwen3.5-4b"))
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data", "cholect50", "parquet"))
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--no-compat", action="store_true", help="skip neuron_compat (reproduces the repeat_interleave failure)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {"stages": {}, "ok": False}
    stage = "neuron"

    def done(name, t0, **extra):
        res["stages"][name] = {"ok": True, "s": round(time.perf_counter() - t0, 2), **extra}
        print(f"[check] {name}: ok {extra}", flush=True)

    try:
        import torch

        t = time.perf_counter()
        import torch_neuronx  # noqa: F401

        info = {"torch": torch.__version__,
                "has_torch_neuron": hasattr(torch, "neuron"),
                "device_count": torch.neuron.device_count() if hasattr(torch, "neuron") else None,
                "is_available": torch.neuron.is_available() if hasattr(torch, "neuron") else None}
        x = (torch.arange(8, dtype=torch.float32, device="neuron") * 2).sum().item()
        info["check"] = x
        done("neuron", t, **info)

        stage = "ops"
        t = time.perf_counter()
        res["stages"]["ops"] = op_correctness()
        res["stages"]["ops"]["s"] = round(time.perf_counter() - t, 2)
        print(f"[check] ops: {res['stages']['ops']['n_wrong']} / {res['stages']['ops']['n']} wrong "
              f"{[k for k, v in res['stages']['ops']['results'].items() if not v['match']]}", flush=True)

        stage = "accelerate"
        t = time.perf_counter()
        from accelerate import PartialState

        st = PartialState()
        if st.device.type != "neuron":
            raise RuntimeError(f"accelerate picked {st.device}, not neuron")
        done("accelerate", t, device=str(st.device), distributed_type=str(st.distributed_type))

        # Qwen3.5's data-dependent index helpers -> CPU (see neuron_compat.py). --no-compat reproduces the failure.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        if not args.no_compat:
            from common import neuron_compat
            res["compat_patched"] = neuron_compat.install()

        stage = "stack"
        t = time.perf_counter()
        from importlib.metadata import version

        import peft  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
        from trl import GRPOConfig, GRPOTrainer  # noqa: F401

        done("stack", t, **{p: version(p) for p in ("transformers", "trl", "peft", "accelerate", "torch-neuronx")})

        if not args.skip_model:
            stage = "model"
            t = time.perf_counter()
            from transformers import AutoModelForImageTextToText, AutoProcessor

            from common.rewards.cholect50 import parse_triplets, reward_single
            from train_grpo_trl import CHAT_KW, PROMPT, load_split

            proc = AutoProcessor.from_pretrained(args.model)
            model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(st.device)
            row = load_split(args.data_dir, "val", proc, 2048)[0]
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_KW)
            enc = proc(text=[text], images=[row["image"]], return_tensors="pt").to(st.device)
            t_gen = time.perf_counter()
            out = model.generate(**enc, max_new_tokens=16, do_sample=False)
            gen_s = time.perf_counter() - t_gen
            resp = proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            done("model", t, prompt_tokens=int(enc["input_ids"].shape[1]), generate_16tok_s=round(gen_s, 1),
                 sample=resp, reward=reward_single(parse_triplets(resp), json.loads(row["gt"])))
        res["ok"] = True
        stage = "done"
    except Exception as e:
        res["failed_stage"] = stage
        res["error"] = f"{type(e).__name__}: {e}"[:2000]
        res["traceback"] = traceback.format_exc()[-4000:]
        print(f"[check] FAILED at {stage}: {res['error']}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
