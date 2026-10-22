#!/usr/bin/env python3
"""Convert verl-format CholecT50 parquet to Miles JSONL.

verl parquet row:
    data_source, prompt (list of {role, content}), images (HF Image),
    ability, reward_model ({style, ground_truth}), extra_info

Miles JSONL row (the proven ORena VLM format, base64 image in content):
    {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<bytes>"}},
        {"type": "text", "text": "<instruction>"}
    ]}],
     "label": "<ground_truth JSON string>"}

The image bytes come from the parquet ``images[0]["bytes"]`` field (raw encoded
frame). The MIME type is sniffed from the magic bytes so the frame is passed to
SGLang byte-for-byte as verl did.

Usage:
    python3 prepare_cholect50_data.py --train /root/data/train.parquet \
        --val /root/data/val.parquet --out-dir /root/miles_bench
"""
from __future__ import annotations

import argparse
import base64
import json
import os

import pyarrow.parquet as pq

PROMPT = (
    "You are a surgical activity recognition model. List every surgical action "
    "triplet (instrument-verb-target) visible in the endoscopic frame. Reply only "
    "as comma-separated triplets, each formatted instrument-verb-target. If no "
    "action is visible, reply 'none'."
)


def sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def row_to_miles(row) -> dict:
    img = row["images"][0]
    raw = img["bytes"]
    mime = sniff_mime(raw)
    b64 = base64.b64encode(raw).decode("ascii")
    # image_url must be a bare data-URL STRING, not the OpenAI {"url": ...} dict:
    # qwen_vl_utils.fetch_image (the fallback for Qwen3.5, which lacks a processor
    # `extract_media` hook) does `ele["image_url"].startswith(...)`.
    user_content = [
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
        {"type": "text", "text": PROMPT},
    ]
    gt = row["reward_model"]["ground_truth"]
    return {"messages": [{"role": "user", "content": user_content}], "label": gt}


def convert(src: str, dst: str) -> int:
    df = pq.read_table(src).to_pandas()
    n = 0
    with open(dst, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row_to_miles(row)) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", default="/root/miles_bench")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t = convert(args.train, os.path.join(args.out_dir, "cholect50_train.jsonl"))
    v = convert(args.val, os.path.join(args.out_dir, "cholect50_val.jsonl"))
    print(f"wrote {t} train rows and {v} val rows to {args.out_dir}")


if __name__ == "__main__":
    main()
