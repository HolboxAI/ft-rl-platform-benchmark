#!/usr/bin/env python3
"""Stage-by-stage probe of a SkyRL Tinker-API server (any backend, any chip).

Runs the smallest possible version of each operation the CholecT50 RL client
needs, in order, and stops at the first failure. The failing stage and its error
message are the result.

  1. connect         ServiceClient + create a LoRA training client
  2. train_text      forward_backward(cross_entropy) + optim_step on one text example
  3. rl_loss         forward_backward(importance_sampling), the loss the RL client uses
  4. sample_text     save weights -> sampling client -> sample 2 completions
  5. image_input     the same sample call with an image chunk in the prompt
                     (CholecT50 needs this; the SkyRL JAX backend has no vision path)

    TINKER_API_KEY=tml-dummy python skyrl/scripts/probe_tinker_server.py \
        --base-url http://localhost:8000 --model Qwen/Qwen3-0.6B --out skyrl/logs/probe.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="must equal the server's --base-model")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-image", action="store_true")
    args = ap.parse_args()

    import tinker
    from tinker import types

    res = {"base_url": args.base_url, "model": args.model, "stages": {}, "ok": False}
    stage = "connect"

    def done(name, t0, **extra):
        res["stages"][name] = {"ok": True, "s": round(time.perf_counter() - t0, 2), **extra}
        print(f"[probe] {name}: ok ({res['stages'][name]['s']}s) {extra or ''}", flush=True)

    try:
        t = time.perf_counter()
        sc = tinker.ServiceClient(base_url=args.base_url, api_key=os.environ.get("TINKER_API_KEY", "tml-dummy"))
        tc = sc.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
        tok = tc.get_tokenizer()
        done("connect", t)

        stage = "train_text"
        t = time.perf_counter()
        ids = tok.encode("The surgeon uses a grasper to retract the gallbladder.")
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0] * (len(ids) - 1)},
        )
        fb = tc.forward_backward([datum], "cross_entropy").result()
        tc.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        done("train_text", t, metrics=getattr(fb, "metrics", None))

        stage = "rl_loss"
        t = time.perf_counter()
        n = len(ids) - 1
        rl_datum = types.Datum(
            model_input=types.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={"target_tokens": ids[1:], "logprobs": [-1.0] * n, "advantages": [0.5] * n},
        )
        tc.forward_backward([rl_datum], "importance_sampling").result()
        tc.optim_step(types.AdamParams(learning_rate=1e-5)).result()
        done("rl_loss", t)

        stage = "sample_text"
        t = time.perf_counter()
        sampler = tc.save_weights_and_get_sampling_client(name="probe")
        prompt = types.ModelInput.from_ints(tok.encode("List one surgical instrument:"))
        out = sampler.sample(prompt=prompt, num_samples=2,
                             sampling_params=types.SamplingParams(max_tokens=16, temperature=0.7)).result()
        done("sample_text", t, sample=tok.decode(out.sequences[0].tokens))

        if not args.skip_image:
            stage = "image_input"
            t = time.perf_counter()
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="PNG")
            img_prompt = types.ModelInput(chunks=[
                types.ImageChunk(data=buf.getvalue(), format="png"),
                types.EncodedTextChunk(tokens=tok.encode("What colour is this image?")),
            ])
            out = sampler.sample(prompt=img_prompt, num_samples=1,
                                 sampling_params=types.SamplingParams(max_tokens=8, temperature=0.0)).result()
            done("image_input", t, sample=tok.decode(out.sequences[0].tokens))
        res["ok"] = True
        stage = "done"
    except Exception as e:
        res["failed_stage"] = stage
        res["error"] = f"{type(e).__name__}: {e}"[:2000]
        res["traceback"] = traceback.format_exc()[-4000:]
        print(f"[probe] FAILED at {stage}: {res['error']}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
