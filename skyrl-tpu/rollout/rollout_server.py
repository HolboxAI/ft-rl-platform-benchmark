#!/usr/bin/env python3
"""SkyRL-compatible rollout server for Cloud TPU.

SkyRL's external-server mode (``generator.inference_engine.run_engines_locally=false``)
drives an HTTP server it did not start. This server implements the routes SkyRL
@3a6e3b4e calls during a synchronous, non-colocated GRPO run, so the rollout
engine can be anything that runs on TPU, independent of vLLM's version:

==================================  =================================================
route                               SkyRL caller
==================================  =================================================
POST /v1/chat/completions/render    VLM generator, once per prompt (images in)
POST /inference/v1/generate         generator, once per sample (tokens in, tokens out)
GET  /get_world_size                weight-sync init (cached)
POST /pause, /resume                around each weight sync
POST /reset_prefix_cache            after a weight sync (no-op: no prefix cache)
POST /sleep, /wake_up, GET /is_sleeping   colocated mode only; accepted as no-ops
POST /tokenize, /detokenize         only if the client has no tokenizer
POST /v1/chat/completions           not used by SkyRL; for the standalone eval script
GET  /health, /v1/models, /version  readiness
==================================  =================================================

The same server also speaks SGLang's API for Miles (``sglang_routes.py``).

Weights are changed by restarting this process on a new checkpoint (SkyRL's
``reload`` weight-sync backend, driven by ``rollout_supervisor.py``), so there are
no weight-transfer routes.

    python rollout_server.py --model /data/models/cholec-sft-qwen3.5-4b --engine vllm --port 8000
    python rollout_server.py --model ... --engine hf --device tpu --max-batch 16 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import mm  # noqa: E402
from engines import build_engine  # noqa: E402
from sglang_routes import add_sglang_routes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rollout.server")


class State:
    def __init__(self, args):
        from transformers import AutoProcessor, AutoTokenizer

        self.args = args
        self.model_path = args.model
        self.served_name = args.served_model_name or args.model
        self.weight_version = args.weight_version
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        from transformers import AutoConfig

        is_vlm = getattr(AutoConfig.from_pretrained(args.model, trust_remote_code=True), "vision_config", None) is not None
        try:
            self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        except Exception as e:
            if is_vlm:  # every CholecT50 prompt has an image: fail at startup, not per request
                raise RuntimeError(f"cannot load the processor for vision model {args.model}: {e}") from e
            logger.warning(f"no processor for text-only {args.model}: {e}")
            self.processor = None
        self.engine = build_engine(args.engine, args.model, args)
        self.load_s = time.perf_counter() - t0
        self.unpaused = asyncio.Event()
        self.unpaused.set()
        self.in_flight = 0
        self.stats = {"render": 0, "generate": 0, "gen_tokens": 0, "gen_seconds": 0.0}


def create_app(args) -> FastAPI:
    app = FastAPI()
    st: Dict[str, State] = {}

    @app.on_event("startup")
    async def _startup():
        st["s"] = State(args)
        logger.info(f"ready: {json.dumps(describe())}")

    def s() -> State:
        return st["s"]

    def describe() -> dict:
        return {
            "model": s().model_path,
            "served_model_name": s().served_name,
            "weight_version": s().weight_version,
            "load_s": round(s().load_s, 1),
            **s().engine.describe(),
        }

    # ---------------------------------------------------------------- health
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/version")
    async def version():
        return {"version": "skyrl-tpu-rollout-0.1", **describe()}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": s().served_name, "object": "model", "owned_by": "skyrl-tpu"}]}

    @app.get("/stats")
    async def stats():
        return {**s().stats, "in_flight": s().in_flight, "paused": not s().unpaused.is_set(), **describe()}

    # ---------------------------------------------------------- control plane
    @app.get("/get_world_size")
    async def get_world_size():
        return {"world_size": int(args.tensor_parallel_size)}

    @app.post("/pause")
    async def pause(request: Request):
        # SkyRL's synchronous trainer pauses only after generation finished, so a gate
        # on new requests is enough; in-flight ones (none, in practice) run to completion.
        s().unpaused.clear()
        return {"paused": True, "mode": request.query_params.get("mode")}

    @app.post("/resume")
    async def resume():
        s().unpaused.set()
        return {"paused": False}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache():
        return {"reset": True}

    @app.post("/sleep")
    async def sleep():
        return {"sleeping": False, "note": "no-op: this server does not support sleep mode"}

    @app.post("/wake_up")
    async def wake_up():
        return {"sleeping": False}

    @app.get("/is_sleeping")
    async def is_sleeping():
        return {"is_sleeping": False}

    # ------------------------------------------------------------ data plane
    @app.post("/v1/chat/completions/render")
    async def render(request: Request):
        body = await request.json()
        if s().processor is None:
            raise HTTPException(400, "this checkpoint has no processor; cannot render")
        s().stats["render"] += 1
        loop = asyncio.get_running_loop()
        try:
            out = await loop.run_in_executor(
                None, mm.render, s().processor, body["messages"], body.get("chat_template_kwargs")
            )
        except Exception as e:
            logger.exception("render failed")
            raise HTTPException(400, f"render failed: {e}")
        return out

    async def _generate(token_ids, sp, features):
        await s().unpaused.wait()
        s().in_flight += 1
        t0 = time.perf_counter()
        try:
            res = await s().engine.generate(token_ids, sp, features)
        finally:
            s().in_flight -= 1
        s().stats["generate"] += 1
        s().stats["gen_tokens"] += len(res.token_ids)
        s().stats["gen_seconds"] += time.perf_counter() - t0
        return res

    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        body = await request.json()
        sp = dict(body.get("sampling_params") or {})
        if int(sp.get("n", 1)) != 1:
            raise HTTPException(400, "n > 1 is not supported")
        res = await _generate(list(body["token_ids"]), sp, body.get("features"))
        choice: Dict[str, Any] = {"index": 0, "token_ids": res.token_ids, "finish_reason": res.finish_reason}
        if res.logprobs is not None:
            choice["logprobs"] = {"content": [{"token_id": t, "logprob": lp} for t, lp in zip(res.token_ids, res.logprobs)]}
        else:
            choice["logprobs"] = None
        return {"id": f"gen-{time.time_ns()}", "choices": [choice], "weight_version": s().weight_version}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        rendered = await asyncio.get_running_loop().run_in_executor(
            None, mm.render, s().processor, body["messages"], body.get("chat_template_kwargs")
        )
        sp = {
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "top_k": body.get("top_k", -1),
            "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 256,
            "min_tokens": body.get("min_tokens", 0),
            "stop": body.get("stop"),
            "skip_special_tokens": True,
        }
        res = await _generate(rendered["token_ids"], sp, rendered["features"])
        text = s().tokenizer.decode(res.token_ids, skip_special_tokens=True)
        return {
            "id": f"chat-{time.time_ns()}",
            "object": "chat.completion",
            "model": s().served_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": res.finish_reason}],
            "usage": {"prompt_tokens": len(rendered["token_ids"]), "completion_tokens": len(res.token_ids)},
        }

    @app.post("/tokenize")
    async def tokenize(request: Request):
        body = await request.json()
        ids = s().tokenizer(body["prompt"], add_special_tokens=body.get("add_special_tokens", True))["input_ids"]
        return {"tokens": ids, "count": len(ids)}

    @app.post("/detokenize")
    async def detokenize(request: Request):
        body = await request.json()
        return {"prompt": s().tokenizer.decode(body["tokens"])}

    # Miles (external SGLang engines). Mounted always: no route overlaps SkyRL's.
    add_sglang_routes(app, s, args)

    @app.exception_handler(HTTPException)
    async def _http_exc(_, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail}})

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF checkpoint dir")
    p.add_argument("--served-model-name", default=None, help="name reported by /v1/models (any name is accepted)")
    p.add_argument("--weight-version", type=int, default=0)
    p.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    # vllm
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # hf
    p.add_argument("--device", default=None, help="hf engine: tpu | cpu (default: tpu if torch_tpu is installed)")
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--batch-wait-ms", type=int, default=200)
    p.add_argument("--len-bucket", type=int, default=128)
    p.add_argument("--cache-implementation", default="", help="hf engine: e.g. 'static' (default: model default)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
