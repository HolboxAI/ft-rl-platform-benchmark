"""CholecT50 VLM rollout generate function for Miles.

Identical to the canonical ``examples/geo3k_vlm/rollout.py`` and to
``generate_orena.py``. The base ``miles.rollout.sglang_rollout.generate`` runs the
Qwen3.5 processor on ``sample.prompt`` + the multimodal inputs and sends image
data to the SGLang server. This wrapper drops non-tensor ``multimodal_train_inputs``
fields (a list-valued processor field breaks the tensor-only training data path).
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
