"""Chat rendering and multimodal features in the wire format SkyRL expects.

SkyRL's VLM generator (``skyrl/train/generators/skyrl_vlm_generator.py``) calls
``POST /v1/chat/completions/render`` and expects vLLM 0.31's response shape::

    {"token_ids": [...],                                   # prompt with image placeholders expanded
     "features": {"mm_hashes": {"image": [h0, ...]},
                  "mm_placeholders": {"image": [{"offset": o, "length": n}, ...]},
                  "kwargs_data": {"image": [item0, ...]}}}

It then sends ``features`` back unchanged with ``/inference/v1/generate`` and,
after the rollout, decodes ``kwargs_data`` into ``pixel_values`` /
``image_grid_thw`` for the training forward pass.

vLLM serialises each ``kwargs_data`` item with its own codec, which the trainer
can only decode with vLLM installed. This server writes
``"st:" + base64(safetensors{pixel_values, image_grid_thw})`` instead; the
``skyrl-tpu`` patch teaches ``skyrl.backends.renderer.decode_mm_kwargs`` to read it
without vLLM. ``features["skyrl_tpu_images"]`` carries the original data URLs so
an engine that does its own image processing (vLLM) can get the raw image back.

Rendering uses the checkpoint's own processor and chat template with
``add_generation_prompt=True`` and no extra template kwargs, which is what
vLLM's render route does, so the token ids match the GPU run's.
"""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Any, Dict, List, Optional, Tuple

import torch

ST_PREFIX = "st:"
RAW_IMAGES_KEY = "skyrl_tpu_images"


# ----------------------------------------------------------------- images
def data_url_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    if url.startswith("file://"):
        with open(url[len("file://") :], "rb") as f:
            return f.read()
    raise ValueError(f"only data: and file:// image URLs are supported, got {url[:40]!r}")


def load_image(url: str):
    from PIL import Image

    img = Image.open(io.BytesIO(data_url_bytes(url)))
    return img.convert("RGB") if img.mode != "RGB" else img


def _image_url_of(part: Dict[str, Any]) -> Optional[str]:
    if part.get("type") == "image_url":
        iu = part.get("image_url")
        return iu.get("url") if isinstance(iu, dict) else iu
    if part.get("type") == "image" and isinstance(part.get("image"), str):
        return part["image"]
    return None


def to_hf_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """OpenAI-style messages -> (HF processor messages, image URLs in order)."""
    hf_msgs: List[Dict[str, Any]] = []
    urls: List[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str) or content is None:
            hf_msgs.append({"role": m["role"], "content": content or ""})
            continue
        parts = []
        for part in content:
            url = _image_url_of(part)
            if url is not None:
                urls.append(url)
                parts.append({"type": "image"})
            elif part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
        hf_msgs.append({"role": m["role"], "content": parts})
    return hf_msgs, urls


# --------------------------------------------------------- serialisation
def encode_item(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> str:
    from safetensors.torch import save

    blob = save({"pixel_values": pixel_values.contiguous(), "image_grid_thw": image_grid_thw.contiguous()})
    return ST_PREFIX + base64.b64encode(blob).decode("ascii")


def decode_item(item: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load

    if not item.startswith(ST_PREFIX):
        raise ValueError("kwargs_data item was not written by this server (missing 'st:' prefix)")
    return load(base64.b64decode(item[len(ST_PREFIX) :]))


def placeholder_runs(token_ids: List[int], image_token_id: int) -> List[Dict[str, int]]:
    """Contiguous runs of the image placeholder token, in order."""
    runs: List[Dict[str, int]] = []
    start = None
    for i, t in enumerate(token_ids):
        if t == image_token_id and start is None:
            start = i
        elif t != image_token_id and start is not None:
            runs.append({"offset": start, "length": i - start})
            start = None
    if start is not None:
        runs.append({"offset": start, "length": len(token_ids) - start})
    return runs


def image_token_id_of(processor) -> int:
    tid = getattr(processor, "image_token_id", None)
    if tid is not None:
        return int(tid)
    tok = getattr(processor, "image_token", None) or "<|image_pad|>"
    return int(processor.tokenizer.convert_tokens_to_ids(tok))


# ----------------------------------------------------------------- render
def render(processor, messages: List[Dict[str, Any]], chat_template_kwargs: Optional[dict] = None) -> Dict[str, Any]:
    """The ``/v1/chat/completions/render`` response for ``messages``."""
    hf_msgs, urls = to_hf_messages(messages)
    kwargs = dict(chat_template_kwargs or {})
    text = processor.apply_chat_template(hf_msgs, add_generation_prompt=True, tokenize=False, **kwargs)
    if not urls:
        ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        return {"token_ids": list(ids), "features": None}

    images = [load_image(u) for u in urls]
    enc = processor(text=[text], images=images, return_tensors="pt")
    token_ids = enc["input_ids"][0].tolist()
    pixel_values = enc["pixel_values"]
    grid = enc["image_grid_thw"]

    runs = placeholder_runs(token_ids, image_token_id_of(processor))
    if len(runs) != len(images):
        raise ValueError(f"found {len(runs)} image placeholder runs for {len(images)} images")

    items, hashes, start = [], [], 0
    for i in range(len(images)):
        n = int(torch.prod(grid[i]).item())
        items.append(encode_item(pixel_values[start : start + n], grid[i]))
        start += n
        hashes.append(hashlib.sha256(data_url_bytes(urls[i])).hexdigest())
    if start != pixel_values.shape[0]:
        raise ValueError(f"pixel_values has {pixel_values.shape[0]} patches, grids account for {start}")

    return {
        "token_ids": token_ids,
        "features": {
            "mm_hashes": {"image": hashes},
            "mm_placeholders": {"image": runs},
            "kwargs_data": {"image": items},
            RAW_IMAGES_KEY: urls,
        },
    }


def features_to_tensors(features: Optional[Dict[str, Any]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``features`` -> (pixel_values, image_grid_thw) concatenated over images."""
    items = ((features or {}).get("kwargs_data") or {}).get("image") or []
    if not items:
        return None, None
    pv, thw = [], []
    for item in items:
        t = decode_item(item)
        pv.append(t["pixel_values"])
        g = t["image_grid_thw"]
        thw.append(g.reshape(1, -1) if g.dim() == 1 else g)
    return torch.cat(pv, dim=0), torch.cat(thw, dim=0)


def features_raw_images(features: Optional[Dict[str, Any]]):
    urls = (features or {}).get(RAW_IMAGES_KEY) or []
    return [load_image(u) for u in urls]
