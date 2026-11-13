#!/usr/bin/env python3
"""Drive a running rollout server with SkyRL's own HTTP client, as the trainer does.

Uses ``RemoteInferenceClient`` from the skyrl-tpu checkout (not hand-written
requests), the GPU run's sampling settings through SkyRL's own
``get_vllm_sampling_params``, and SkyRL's ``decode_mm_kwargs`` on the features,
i.e. the exact calls ``SkyRLVLMGymGenerator.agent_loop`` and the weight-sync
bracket make. Needs SkyRL importable (PYTHONPATH=<SkyRL-tpu>:<SkyRL-tpu>/skyrl-gym).

    python tests/test_skyrl_client.py --url http://127.0.0.1:8000 --model <ckpt> --parquet <skyrl val.parquet>
"""
import argparse
import asyncio
import json
import sys

import pyarrow.parquet as pq


async def run(args):
    from transformers import AutoTokenizer

    from skyrl.backends.renderer import decode_mm_kwargs
    from skyrl.backends.skyrl_train.inference_servers.engine_utils import get_vllm_sampling_params
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
    from skyrl.train.config.config import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    client = RemoteInferenceClient(proxy_url=args.url, server_urls=[args.url], data_parallel_size=1,
                                   model_name=args.model, tokenizer=tok)
    # GPU run: temperature 0.7, top_p 1.0, max_generate_length 256 (SKYRL_CONFIG.md).
    sp = get_vllm_sampling_params(SamplingParams(temperature=0.7, top_p=1.0, max_generate_length=256))
    out = {"sampling_params": sp}

    out["world_size"] = await client.get_world_size()
    rows = pq.read_table(args.parquet).slice(0, args.n).to_pylist()
    lengths = []
    for i, row in enumerate(rows):
        conv = row["prompt"]
        rendered = await client.render_chat_completion({"json": {"model": client.model_name, "messages": conv}})
        ids, feats = rendered["token_ids"], rendered.get("features")
        lengths.append(len(ids))
        res = await client.generate({"prompt_token_ids": [ids], "session_ids": [f"s{i}"], "sampling_params": sp,
                                     "mm_features": [feats] if feats is not None else None}, model=client.model_name)
        assert len(res["response_ids"][0]) > 0
        assert res["response_logprobs"] is not None and len(res["response_logprobs"][0]) == len(res["response_ids"][0])
        mm = decode_mm_kwargs((feats or {}).get("kwargs_data"))
        if i == 0:
            out["first"] = {"prompt_tokens": len(ids), "response_tokens": len(res["response_ids"][0]),
                            "stop_reason": res["stop_reasons"][0], "response": res["responses"][0][:120],
                            "pixel_values": list(mm["pixel_values"].shape), "image_grid_thw": mm["image_grid_thw"].tolist()}
    out["prompt_tokens"] = {"min": min(lengths), "max": max(lengths), "n": len(lengths), "limit_max_prompt_length": 4096}
    assert max(lengths) <= 4096, "a prompt exceeds trainer.max_prompt_length=4096"

    # The trainer's weight-sync bracket (non-colocated, non-delta): pause -> sync -> resume, then cache reset.
    await client.pause_generation()
    await client.resume_generation()
    await client.reset_prefix_cache()
    out["control_plane"] = "pause/resume/reset_prefix_cache ok"
    await client.aclose()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    out = asyncio.run(run(args))
    print(json.dumps(out, indent=1))
    print("SKYRL CLIENT TEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
