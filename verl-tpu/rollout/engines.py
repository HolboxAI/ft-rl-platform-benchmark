"""Generation engines behind the rollout server.

Both take SkyRL's request (prompt token ids with image placeholders already
expanded, vLLM-style sampling params, the render ``features``) and return what
``/inference/v1/generate`` returns: generated token ids (EOS included when the
model stopped on it, as vLLM does), a finish reason, and per-token logprobs.

``vllm``  vLLM on TPU (``vllm-tpu`` / ``tpu-inference``), run in-process with
          ``AsyncLLM``. Gets the raw image from ``features`` and lets vLLM run its
          own processor; the prompt ids already contain the expanded placeholders,
          which vLLM detects and keeps.
``hf``    ``transformers`` ``generate`` on TorchTPU (device ``"tpu"``), or CPU for
          local tests. Uses ``pixel_values`` from ``features`` directly. Requests
          are micro-batched; batch size and prompt length are padded to buckets to
          bound the number of distinct shapes the TPU compiles.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from mm import features_raw_images, features_to_tensors

logger = logging.getLogger("rollout.engines")


@dataclass
class GenResult:
    token_ids: List[int]
    finish_reason: str
    logprobs: Optional[List[float]]


def _top_k(sp: Dict[str, Any]) -> Optional[int]:
    k = sp.get("top_k")
    return int(k) if k is not None and int(k) > 0 else None


def _wants_logprobs(sp: Dict[str, Any]) -> bool:
    return sp.get("logprobs") is not None


# ====================================================================== vLLM
class VLLMEngine:
    name = "vllm"

    def __init__(self, model: str, *, max_model_len: int, max_num_seqs: int, tensor_parallel_size: int,
                 gpu_memory_utilization: float, dtype: str = "bfloat16", extra: Optional[dict] = None):
        from vllm import AsyncEngineArgs

        try:
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:  # older vLLM
            from vllm import AsyncLLMEngine as AsyncLLM

        args = AsyncEngineArgs(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            **(extra or {}),
        )
        self.engine = AsyncLLM.from_engine_args(args)
        import vllm

        self.version = vllm.__version__

    def _sampling_params(self, sp: Dict[str, Any]):
        from vllm import SamplingParams

        kw = dict(
            n=1,
            temperature=float(sp.get("temperature", 1.0)),
            top_p=float(sp.get("top_p", 1.0)),
            min_p=float(sp.get("min_p", 0.0) or 0.0),
            max_tokens=int(sp.get("max_tokens", 256)),
            min_tokens=int(sp.get("min_tokens", 0) or 0),
            stop=sp.get("stop"),
            include_stop_str_in_output=bool(sp.get("include_stop_str_in_output", False)),
            skip_special_tokens=bool(sp.get("skip_special_tokens", True)),
            logprobs=0 if _wants_logprobs(sp) else None,
        )
        if _top_k(sp) is not None:
            kw["top_k"] = _top_k(sp)
        if sp.get("seed") is not None:
            kw["seed"] = int(sp["seed"])
        return SamplingParams(**kw)

    async def generate(self, token_ids: List[int], sp: Dict[str, Any], features: Optional[dict]) -> GenResult:
        prompt: Dict[str, Any] = {"prompt_token_ids": token_ids}
        images = features_raw_images(features)
        if images:
            prompt["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}
        final = None
        async for out in self.engine.generate(prompt, self._sampling_params(sp), request_id=uuid.uuid4().hex):
            final = out
        c = final.outputs[0]
        lps = None
        if _wants_logprobs(sp) and c.logprobs is not None:
            lps = [float(step[tid].logprob) for tid, step in zip(c.token_ids, c.logprobs)]
        return GenResult(list(c.token_ids), c.finish_reason or "stop", lps)

    def describe(self) -> dict:
        return {"engine": "vllm", "vllm": self.version}


# ====================================================================== HF
@dataclass
class _Req:
    token_ids: List[int]
    sp: Dict[str, Any]
    pixel_values: Optional[torch.Tensor]
    grid: Optional[torch.Tensor]
    future: asyncio.Future


class HFEngine:
    """transformers ``generate`` with request micro-batching."""

    name = "hf"

    def __init__(self, model: str, *, device: str, max_batch: int, batch_wait_ms: int,
                 len_bucket: int, cache_implementation: Optional[str], dtype: str = "bfloat16"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.device = torch.device(device)
        if self.device.type == "tpu":
            from tpu_compat import load_torch_tpu, patch_qwen35_mrope

            load_torch_tpu()
            logger.info(f"TorchTPU: patched apply_interleaved_mrope on {patch_qwen35_mrope()} Qwen3.5 class(es)")
        torch_dtype = getattr(torch, dtype)
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        is_vlm = getattr(cfg, "vision_config", None) is not None
        auto = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
        t0 = time.perf_counter()
        self.model = auto.from_pretrained(model, dtype=torch_dtype, trust_remote_code=True).to(self.device).eval()
        logger.info(f"loaded {model} on {self.device} in {time.perf_counter() - t0:.1f}s")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # transformers 5 needs mm_token_type_ids for Qwen3.5 M-RoPE; built the way SkyRL's model_wrapper does.
        self.image_token_id = getattr(cfg, "image_token_id", None)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        gen_cfg = getattr(self.model, "generation_config", None)
        eos = getattr(gen_cfg, "eos_token_id", None) or self.tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.max_batch = max_batch
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.len_bucket = max(1, len_bucket)
        self.cache_implementation = cache_implementation
        self.queue: "asyncio.Queue[_Req]" = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def describe(self) -> dict:
        import transformers

        return {"engine": "hf", "device": str(self.device), "transformers": transformers.__version__,
                "max_batch": self.max_batch, "len_bucket": self.len_bucket,
                "cache_implementation": self.cache_implementation}

    async def generate(self, token_ids: List[int], sp: Dict[str, Any], features: Optional[dict]) -> GenResult:
        if self._worker is None:
            self._worker = asyncio.get_running_loop().create_task(self._loop())
        pv, grid = features_to_tensors(features)
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put(_Req(token_ids, sp, pv, grid, fut))
        return await fut

    @staticmethod
    def _key(sp: Dict[str, Any]) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in sp.items() if k not in ("seed",)))

    async def _loop(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = loop.time() + self.batch_wait_s
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            # One generate call per sampling config; requests in a step share one.
            groups: Dict[tuple, List[_Req]] = {}
            for r in batch:
                groups.setdefault(self._key(r.sp), []).append(r)
            for reqs in groups.values():
                try:
                    results = await loop.run_in_executor(None, self._run, reqs)
                    for r, res in zip(reqs, results):
                        r.future.set_result(res)
                except Exception as e:  # surface to every waiting request
                    logger.exception("generate failed")
                    for r in reqs:
                        if not r.future.done():
                            r.future.set_exception(e)

    @torch.no_grad()
    def _run(self, reqs: List[_Req]) -> List[GenResult]:
        sp = reqs[0].sp
        n_real = len(reqs)
        # Pad the batch with copies of the first request so the batch dim is fixed.
        rows = reqs + [reqs[0]] * (self.max_batch - n_real) if self.max_batch > 1 else reqs
        max_len = max(len(r.token_ids) for r in rows)
        seq_len = int(math.ceil(max_len / self.len_bucket) * self.len_bucket)

        ids = torch.full((len(rows), seq_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), seq_len), dtype=torch.long)
        for i, r in enumerate(rows):  # left padding
            n = len(r.token_ids)
            ids[i, seq_len - n :] = torch.tensor(r.token_ids, dtype=torch.long)
            mask[i, seq_len - n :] = 1
        inputs: Dict[str, Any] = {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}
        if any(r.pixel_values is not None for r in rows):
            if not all(r.pixel_values is not None for r in rows):
                raise ValueError("mixing text-only and image requests in one batch is not supported")
            inputs["pixel_values"] = torch.cat([r.pixel_values for r in rows]).to(self.device, self.model.dtype)
            inputs["image_grid_thw"] = torch.cat([r.grid for r in rows]).to(self.device)
            if self.image_token_id is not None:
                inputs["mm_token_type_ids"] = (ids == self.image_token_id).long().to(self.device)

        temperature = float(sp.get("temperature", 1.0))
        do_sample = temperature > 0
        gen_kw: Dict[str, Any] = dict(
            max_new_tokens=int(sp.get("max_tokens", 256)),
            min_new_tokens=int(sp.get("min_tokens", 0) or 0),
            do_sample=do_sample,
            pad_token_id=self.pad_id,
            eos_token_id=sorted(self.eos_ids),
            return_dict_in_generate=True,
            output_logits=_wants_logprobs(sp),
        )
        if do_sample:
            gen_kw.update(temperature=temperature, top_p=float(sp.get("top_p", 1.0)), top_k=_top_k(sp) or 0)
            mp = float(sp.get("min_p", 0.0) or 0.0)
            if mp > 0:
                gen_kw["min_p"] = mp
        if self.cache_implementation:
            gen_kw["cache_implementation"] = self.cache_implementation
        if sp.get("seed") is not None:
            torch.manual_seed(int(sp["seed"]))

        out = self.model.generate(**inputs, **gen_kw)
        new = out.sequences[:, seq_len:].cpu()
        logits = torch.stack(out.logits, dim=1).float().cpu() if _wants_logprobs(sp) else None  # [B, T, V]
        stops = sp.get("stop") or []

        results: List[GenResult] = []
        for i in range(n_real):
            toks = new[i].tolist()
            finish = "length"
            cut = len(toks)
            for j, t in enumerate(toks):
                if t in self.eos_ids:
                    cut, finish = j + 1, "stop"  # keep EOS, as vLLM does
                    break
            toks = toks[:cut]
            if stops:
                text = self.tokenizer.decode(toks, skip_special_tokens=True)
                for s in stops:
                    k = text.find(s)
                    if k >= 0:
                        # Truncate at the first token whose decoded prefix contains the stop string.
                        for j in range(1, len(toks) + 1):
                            if s in self.tokenizer.decode(toks[:j], skip_special_tokens=True):
                                toks = toks[:j] if sp.get("include_stop_str_in_output") else toks[: j - 1]
                                break
                        finish = "stop"
                        break
            lps = None
            if logits is not None:
                lp = torch.log_softmax(logits[i, : len(toks)], dim=-1)
                lps = lp.gather(-1, torch.tensor(toks, dtype=torch.long).unsqueeze(-1)).squeeze(-1).tolist()
            results.append(GenResult(toks, finish, lps))
        return results


def build_engine(kind: str, model: str, args) -> Any:
    if kind == "vllm":
        return VLLMEngine(
            model,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    if kind == "hf":
        device = args.device or ("tpu" if os.environ.get("PJRT_DEVICE") == "TPU" or _has_torch_tpu() else "cpu")
        return HFEngine(
            model,
            device=device,
            max_batch=args.max_batch,
            batch_wait_ms=args.batch_wait_ms,
            len_bucket=args.len_bucket,
            cache_implementation=args.cache_implementation or None,
        )
    raise ValueError(f"unknown engine {kind!r}")


def _has_torch_tpu() -> bool:
    try:
        import torch_tpu  # noqa: F401

        return True
    except ImportError:
        return False
