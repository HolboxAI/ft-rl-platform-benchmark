#!/usr/bin/env python3
"""P1: does the TPU rollout server do what SkyRL needs, on CholecT50? Run from the trainer VM.

Talks to the server exactly as SkyRL @2c54cefc does (same routes, payloads and
sampling params), then scores real val frames with the GPU run's reward:

  1. health / version / get_world_size
  2. render       one val frame (SkyRL parquet prompt, image as data URL) -> token ids + features;
                  placeholders match the image-pad runs; features decode to pixel_values with
                  SkyRL's own ``decode_mm_kwargs`` (skyrl-tpu patch) when SkyRL is importable
  3. generate     SkyRL's default sampling params (temperature 0.7, 256 tokens, logprobs 1);
                  logprobs count == token count; EOS kept
  4. pause/resume and reset_prefix_cache
  5. eval         greedy, first ``--n`` val frames, reward = GPU run's ``reward_single``;
                  the GPU run's eval_before_train was 0.329 on all 32
  6. reload       (``--reload``) re-publish the base checkpoint through the agent with
                  SkyRL's ``ReloadWeightSync`` transport and time the restart

    python probes/p1_rollout.py --url http://<rollout-ip>:8000 --parquet ~/cholec/skyrl_parquet/val.parquet --n 32
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "skyrl_cholect50"))
sys.path.insert(0, os.path.join(ROOT, "rollout"))


def http(method, url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def skyrl_sampling_params(temperature, max_tokens):
    """skyrl.backends.skyrl_train.inference_servers.engine_utils.get_vllm_sampling_params, GPU-run values."""
    return {"min_tokens": 1, "skip_special_tokens": True, "include_stop_str_in_output": True,
            "max_tokens": max_tokens, "temperature": temperature, "top_p": 1.0, "top_k": -1,
            "min_p": 0.0, "logprobs": 1, "stop": None}


def load_rows(parquet, n):
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet).slice(0, n).to_pylist()
    out = []
    for r in rows:
        prompt = r["prompt"]
        gt = r["reward_spec"]["ground_truth"]
        out.append((prompt, json.loads(gt) if isinstance(gt, str) else gt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--agent-url", default=None, help="default: --url with port + 1")
    ap.add_argument("--parquet", default=os.path.expanduser("~/cholec/skyrl_parquet/val.parquet"))
    ap.add_argument("--model", default=os.path.expanduser("~/models/cholec-sft-qwen3.5-4b"),
                    help="checkpoint dir, used for the tokenizer and for --reload")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "p1_rollout.json"))
    args = ap.parse_args()
    url = args.url.rstrip("/")
    report = {"url": url, "stages": {}}

    def stage(name, fn):
        t0 = time.perf_counter()
        try:
            d = fn()
            report["stages"][name] = {"ok": True, "s": round(time.perf_counter() - t0, 2), **(d or {})}
            print(f"[ok]   {name}: {json.dumps(d)[:300] if d else ''}")
            return d
        except Exception as e:
            report["stages"][name] = {"ok": False, "s": round(time.perf_counter() - t0, 2), "error": f"{type(e).__name__}: {e}"}
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            return None

    from cholec_env import parse_triplets, reward_single

    rows = load_rows(args.parquet, args.n)
    ctx = {}

    stage("health", lambda: {"version": http("GET", f"{url}/version"),
                             "world_size": http("GET", f"{url}/get_world_size")["world_size"]})

    def s_render():
        prompt, _ = rows[0]
        r = http("POST", f"{url}/v1/chat/completions/render", {"model": args.model, "messages": prompt})
        ids, feats = r["token_ids"], r["features"]
        ph = feats["mm_placeholders"]["image"]
        assert len(ph) == 1, ph
        try:
            from skyrl.backends.renderer import decode_mm_kwargs

            mm = decode_mm_kwargs(feats["kwargs_data"])
            pv, thw, via = mm["pixel_values"], mm["image_grid_thw"], "skyrl.backends.renderer"
        except ImportError:
            import mm as rmm

            pv, thw = rmm.features_to_tensors(feats)
            via = "rollout/mm.py (SkyRL not importable here)"
        merge = 4  # Qwen VL: 2x2 spatial merge -> placeholder count = patches / 4
        assert ph[0]["length"] * merge == pv.shape[0], (ph, tuple(pv.shape))
        ctx["render"] = r
        return {"prompt_tokens": len(ids), "placeholder": ph[0], "pixel_values": list(pv.shape),
                "image_grid_thw": thw.tolist(), "decoded_via": via}

    stage("render", s_render)

    def s_generate():
        r = ctx["render"]
        out = http("POST", f"{url}/inference/v1/generate", {"model": args.model, "token_ids": r["token_ids"],
                   "features": r["features"], "sampling_params": skyrl_sampling_params(0.7, 256)})
        c = out["choices"][0]
        lps = [x["logprob"] for x in (c.get("logprobs") or {}).get("content", [])]
        assert len(lps) == len(c["token_ids"]), (len(lps), len(c["token_ids"]))
        return {"tokens": len(c["token_ids"]), "finish_reason": c["finish_reason"], "mean_logprob": statistics.mean(lps) if lps else None}

    if "render" in ctx:
        stage("generate", s_generate)

    stage("pause_resume", lambda: {"pause": http("POST", f"{url}/pause?mode=keep"), "resume": http("POST", f"{url}/resume"),
                                   "reset": http("POST", f"{url}/reset_prefix_cache")})

    def s_eval():
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        rewards, malformed, samples, t0 = [], 0, [], time.perf_counter()
        for prompt, gt in rows:
            r = http("POST", f"{url}/v1/chat/completions/render", {"model": args.model, "messages": prompt})
            out = http("POST", f"{url}/inference/v1/generate", {"model": args.model, "token_ids": r["token_ids"],
                       "features": r["features"], "sampling_params": skyrl_sampling_params(0.0, 256)})
            text = tok.decode(out["choices"][0]["token_ids"], skip_special_tokens=True)
            pred = parse_triplets(text)
            malformed += pred is None
            rewards.append(reward_single(pred, gt))
            if len(samples) < 3:
                samples.append({"reply": text[:200], "reward": rewards[-1]})
        return {"n": len(rows), "mean_reward": round(statistics.mean(rewards), 4), "malformed": malformed,
                "gpu_run_eval_before_train_32": 0.3286, "seconds": round(time.perf_counter() - t0, 1), "samples": samples}

    stage("eval_greedy", s_eval)

    if args.reload:
        def s_reload():
            from skyrl.backends.skyrl_train.weight_sync.reload import ReloadWeightSync

            agent = (args.agent_url or "").rstrip("/") or None
            rs = ReloadWeightSync.from_env(server_urls=[url], model_dtype="bfloat16") if not agent else \
                ReloadWeightSync(agent_urls=[agent], export_dir="/tmp")
            rs.version = int(time.time()) % 100000
            import shutil
            import tempfile

            d = tempfile.mkdtemp(prefix="p1_reload_")
            for n in os.listdir(args.model):
                if n.endswith((".safetensors", ".safetensors.index.json")):
                    os.symlink(os.path.join(args.model, n), os.path.join(d, n))
            t0 = time.perf_counter()
            res = rs._publish_one(rs.agent_urls[0], d)
            shutil.rmtree(d)
            after = http("GET", f"{url}/version")
            return {"agent": rs.agent_urls[0], "publish_and_restart_s": round(time.perf_counter() - t0, 1),
                    "agent_reply": res, "server_weight_version": after.get("weight_version")}

        stage("reload", s_reload)

    report["passed"] = all(s["ok"] for s in report["stages"].values())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"P1 {'PASSED' if report['passed'] else 'FAILED'} -> {args.out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
