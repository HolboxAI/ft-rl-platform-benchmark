#!/usr/bin/env python3
"""Convert a multimodal Qwen3.5 HF checkpoint into a text-only checkpoint.

The verl-hardware-plugin TorchTitan TPU engine (``torchtitan.experiments.tpu.qwen3_5``)
drops the vision encoder, so its state-dict adapter expects a text-only HF checkpoint
with ``model.*`` keys (``Qwen3_5[Moe]ForCausalLM``). The stock ``Qwen/Qwen3.5-*`` repos
are multimodal (``Qwen3_5[Moe]ForConditionalGeneration``) with the language model nested
under ``model.language_model.*`` plus a ``model.visual.*`` vision tower and ``mtp.*``
multi-token-prediction heads.

This script produces that text-only checkpoint:

* renames ``model.language_model.*``  -> ``model.*``
* drops ``model.visual.*`` (vision encoder)
* drops ``mtp.*`` (MTP heads; verl disables MTP)
* keeps ``lm_head.weight``
* rewrites ``config.json`` to ``Qwen3_5ForCausalLM`` with ``text_config`` flattened

Usage:
    python convert_qwen35_text.py --src /home/vishnunair/models/Qwen3.5-9B \
                                  --dst /home/vishnunair/models/Qwen3.5-9B-text
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def build_text_config(src_cfg: dict) -> dict:
    text = dict(src_cfg["text_config"])
    # The language model alone, without the vision tower / MTP heads.
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    text["model_type"] = text.get("model_type", "qwen3_5_text")
    text["mtp_num_hidden_layers"] = 0
    # Carry over top-level fields the text model still needs.
    for k in ("tie_word_embeddings", "transformers_version", "bos_token_id", "pad_token_id"):
        if k in src_cfg:
            text[k] = src_cfg[k]
    # eos_token_id already lives in text_config; keep whatever is there.
    return text


def transform_key(k: str) -> str | None:
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    if k.startswith("model.visual.") or k.startswith("mtp."):
        return None
    return k


def convert(args):
    src = args.src
    dst = args.dst
    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(src, "config.json")) as f:
        src_cfg = json.load(f)
    text_cfg = build_text_config(src_cfg)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(text_cfg, f, indent=2)

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    shard_files = sorted(set(weight_map.values()))
    new_weight_map = {}
    for shard in shard_files:
        src_path = os.path.join(src, shard)
        dst_path = os.path.join(dst, shard)
        out = {}
        with safe_open(src_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                nk = transform_key(k)
                if nk is None:
                    continue
                out[nk] = f.get_tensor(k)
        save_file(out, dst_path)
        for nk in out:
            new_weight_map[nk] = shard
        print(f"  {shard}: {len(out)} tensors -> {dst_path}")

    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_weight_map}, f, indent=2)

    # Tokenizer / template files verl needs.
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
               "preprocessor_config.json"):
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, fn))

    print(f"\nDone: {len(new_weight_map)} text-only tensors written to {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    convert(ap.parse_args())


if __name__ == "__main__":
    main()
