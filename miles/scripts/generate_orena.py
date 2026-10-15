"""ORena VLM rollout generate function for Miles.

Mirrors the canonical `examples/geo3k_vlm/rollout.py` single-turn VLM generate
function. The base `miles.rollout.sglang_rollout.generate` does the actual SGLang
request (it calls the Qwen3-VL processor on `sample.prompt` + the multimodal
inputs and sends `image_data` to the SGLang server). This wrapper exists for one
reason: Qwen3-VL's processor returns `mm_token_type_ids` as a Python list, and
the Megatron data path assumes every `multimodal_train_inputs` value is a torch
tensor, so the list-valued field raises "AttributeError: 'list' object has no
attribute 'to'". We drop the non-tensor fields, matching the geo3k example.

The legacy `(args, sample, sampling_params)` signature is detected by
`load_generate_function` and wrapped in `LegacyGenerateFnAdapter`, which calls it
as `fn(args, sample, sampling_params)` and wraps the returned `Sample` in
`GenerateFnOutput`.
"""
from __future__ import annotations

import torch

from miles.rollout.sglang_rollout import generate as _generate
from miles.utils.types import Sample


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    sample = await _generate(args, sample, sampling_params)
    mm = sample.multimodal_train_inputs
    if mm:
        sample.multimodal_train_inputs = {
            k: v for k, v in mm.items() if isinstance(v, torch.Tensor)
        } or None
    return sample
