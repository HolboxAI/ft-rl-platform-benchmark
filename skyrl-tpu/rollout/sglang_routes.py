"""SGLang-compatible routes on the TPU rollout server, for Miles' external-engine mode.

Miles @8e99947 drives "external" rollout engines (``--rollout-external-engine-addrs``) over SGLang's HTTP API
(``miles/backends/sglang_utils/sglang_api_client.py``, ``miles/rollout/sglang_rollout.py``). This adds the routes a
synchronous, non-colocated FSDP GRPO run with disk-delta weight sync calls, on top of the same HF engine the SkyRL run
used, so Miles needs no SGLang engine:

=================================================  =====================================================
route                                              Miles caller
=================================================  =====================================================
POST /generate                                     rollout + eval, once per sample (token ids + images)
GET  /server_info, /get_server_info                engine discovery (``internal_states[0].world_size``)
GET  /health_generate, /health, /flush_cache       readiness, before and after weight syncs
GET  /model_info, /get_weight_version              weight-version check after each sync
POST /update_weight_version                        end of each sync
POST /pause_generation, /continue_generation       around each sync
POST /begin_weight_update, /end_weight_update      sync session bracket (no-ops: weights change by restart)
POST /abort_request                                fault handling
POST /release_memory_occupation, /resume_memory_occupation   colocate/offload only; accepted as no-ops
=================================================  =====================================================

``/pull_weights`` and ``/update_weights_from_disk`` restart this process, so ``rollout_supervisor.py`` answers them
(it fronts this server in Miles mode and proxies everything else here).

Request/response shapes follow SGLang 0.5.22 (``/generate`` with ``input_ids`` + ``image_data`` +
``sampling_params`` + ``return_logprob``; ``meta_info.output_token_logprobs`` as ``[logprob, token_id, None]``).
Log-probs are of the temperature-scaled distribution, as SGLang reports them and as Miles' trainer recomputes them.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Dict

from fastapi import FastAPI, HTTPException, Request

import mm


def to_engine_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
    """SGLang sampling params -> the vLLM-style dict the engines take."""
    temperature = float(sp.get("temperature", 1.0))
    out: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": float(sp.get("top_p", 1.0) if sp.get("top_p") is not None else 1.0),
        "top_k": int(sp.get("top_k", -1) if sp.get("top_k") is not None else -1),
        "max_tokens": int(sp.get("max_new_tokens", 128)),
        "min_tokens": int(sp.get("min_new_tokens", 0) or 0),
        "stop": sp.get("stop") or None,
        "logprobs": 0,
        "logprob_temperature": temperature if temperature > 0 else 1.0,
    }
    if sp.get("min_p"):
        out["min_p"] = float(sp["min_p"])
    if sp.get("sampling_seed") is not None:
        out["seed"] = int(sp["sampling_seed"])
    unsupported = [k for k in ("stop_token_ids", "regex", "json_schema", "logit_bias") if sp.get(k)]
    if unsupported:
        raise HTTPException(400, f"sampling params not supported by this server: {unsupported}")
    return out


def _weight_checksums(model) -> Dict[str, str]:
    import hashlib

    import torch

    out = {}
    with torch.no_grad():
        for name, t in model.state_dict().items():
            buf = t.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy()
            out[name] = hashlib.blake2b(buf.tobytes(), digest_size=16).hexdigest()
    return out


def _image_tensors(processor, image_data):
    if not image_data:
        return None, None
    images = [mm.load_image(u if isinstance(u, str) else u.get("url")) for u in image_data]
    out = processor.image_processor(images=images, return_tensors="pt")
    return out["pixel_values"], out["image_grid_thw"]


def add_sglang_routes(app: FastAPI, state: Callable[[], Any], args) -> None:
    def server_info() -> dict:
        s = state()
        return {
            "model_path": s.model_path,
            "served_model_name": s.served_name,
            "tp_size": int(args.tensor_parallel_size),
            "dp_size": 1,
            "disaggregation_mode": "null",
            "weight_version": str(s.weight_version),
            "internal_states": [{"world_size": int(args.tensor_parallel_size), "weight_version": str(s.weight_version)}],
            "version": "tpu-rollout-sglang-api-0.1",
            "engine": s.engine.describe(),
        }

    @app.get("/server_info")
    async def get_server_info():
        return server_info()

    app.add_api_route("/get_server_info", get_server_info, methods=["GET"])

    @app.get("/health_generate")
    async def health_generate():
        return {"status": "ok"}

    @app.api_route("/flush_cache", methods=["GET", "POST"])
    async def flush_cache():
        return {"success": True, "message": "no prefix cache"}

    @app.get("/model_info")
    async def model_info():
        s = state()
        return {"model_path": s.model_path, "is_generation": True, "weight_version": str(s.weight_version)}

    app.add_api_route("/get_model_info", model_info, methods=["GET"])
    app.add_api_route("/get_weight_version", model_info, methods=["GET"])

    @app.post("/update_weight_version")
    async def update_weight_version(request: Request):
        body = await request.json()
        state().weight_version = body["new_version"]
        return {"success": True, "message": f"weight version {body['new_version']}"}

    @app.post("/pause_generation")
    async def pause_generation():
        # Miles' synchronous loop pauses after generation finished: gating new requests is enough.
        state().unpaused.clear()
        return {"success": True}

    @app.post("/continue_generation")
    async def continue_generation():
        state().unpaused.set()
        return {"success": True}

    @app.post("/weights_checker")
    async def weights_checker(request: Request):
        """Miles' audit after each weight sync (``action="checksum"``): one checksum per loaded tensor, in the
        ``{"success", "ranks": [{"parallelism_info", "checksums"}]}`` shape ``merge_inference_engine_ranks`` reads.
        Without it Miles retries for 300 s per sync before logging the audit as failed (2026-10-26)."""
        body = await request.json()
        if body.get("action") != "checksum":
            return {"success": True, "message": f"action {body.get('action')!r}: no-op on this server", "ranks": []}
        sums = await asyncio.get_running_loop().run_in_executor(None, _weight_checksums, state().engine.model)
        return {"success": True, "ranks": [{"parallelism_info": [{"role": "tp", "rank": 0}], "checksums": sums}]}

    def _noop_for(route: str):
        async def _noop():
            return {"success": True, "message": f"{route}: no-op on this server"}

        return _noop

    for route in ("/begin_weight_update", "/end_weight_update", "/release_memory_occupation",
                  "/resume_memory_occupation", "/abort_request"):
        app.add_api_route(route, _noop_for(route), methods=["POST"])

    @app.post("/generate")
    async def generate(request: Request):
        s = state()
        body = await request.json()
        if "input_ids" not in body:
            raise HTTPException(400, "send input_ids (Miles tokenizes on its side)")
        if isinstance(body["input_ids"][0], list):
            raise HTTPException(400, "batched /generate is not supported; send one request per sample")
        sp_in = dict(body.get("sampling_params") or {})
        sp = to_engine_sampling_params(sp_in)
        loop = asyncio.get_running_loop()
        pv, grid = await loop.run_in_executor(None, _image_tensors, s.processor, body.get("image_data"))
        await s.unpaused.wait()
        s.in_flight += 1
        t0 = time.perf_counter()
        try:
            res = await s.engine.generate_mm(list(body["input_ids"]), sp, pv, grid)
        finally:
            s.in_flight -= 1
        s.stats["generate"] += 1
        s.stats["gen_tokens"] += len(res.token_ids)
        s.stats["gen_seconds"] += time.perf_counter() - t0
        text = s.tokenizer.decode(res.token_ids, skip_special_tokens=bool(sp_in.get("skip_special_tokens", True)),
                                  spaces_between_special_tokens=bool(sp_in.get("spaces_between_special_tokens", True)))
        if res.finish_reason == "stop":
            last = res.token_ids[-1] if res.token_ids else None
            finish = {"type": "stop", "matched": last}
        else:
            finish = {"type": "length", "length": len(res.token_ids)}
        meta: Dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "finish_reason": finish,
            "prompt_tokens": len(body["input_ids"]),
            "completion_tokens": len(res.token_ids),
            "cached_tokens": 0,
            "weight_version": str(s.weight_version),
            "e2e_latency": round(time.perf_counter() - t0, 4),
        }
        if body.get("return_logprob"):
            lps = res.logprobs or [None] * len(res.token_ids)
            meta["output_token_logprobs"] = [[lp, t, None] for lp, t in zip(lps, res.token_ids)]
        return {"text": text, "output_ids": res.token_ids, "meta_info": meta}
