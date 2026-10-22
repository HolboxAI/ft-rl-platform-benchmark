#!/usr/bin/env python3
"""Build verl-format parquet for CholecT50 surgical triplet recognition (RLVR).

Downloads the manifest-listed frames from the remote CholecT50 zip (HTTP range
requests via remotezip) and writes two parquet files:

    train.parquet  (256 rows)   val.parquet  (32 rows)

Each row follows verl's VLM contract (geo3k style):
    data_source   : "cholec_t50"
    prompt        : [{"role": "user", "content": "<image>\\n" + PROMPT}]
    images        : [frame]          (HF Image feature; one frame per row)
    ability       : "recognition"
    reward_model  : {"style": "rule", "ground_truth": <json str>}
    extra_info    : {"video": ..., "frame": ...}
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset
from PIL import Image
from remotezip import RemoteZip

HOME = os.path.expanduser("~")
MANIFEST_DIR = os.path.join(HOME, "cholec")
CACHE_DIR = os.path.join(HOME, "cholec", "frames")
OUT_DIR = os.path.join(HOME, "cholec", "parquet")

ZIP_URL = "https://s3.unistra.fr/camma_public/datasets/cholect50/CholecT50.zip"

PROMPT = (
    "You are a surgical activity recognition model. List every surgical action "
    "triplet (instrument-verb-target) visible in the endoscopic frame. Reply only "
    "as comma-separated triplets, each formatted instrument-verb-target. If no "
    "action is visible, reply 'none'."
)

TRAIN_N = 256
VAL_N = 32


def load_manifest(split):
    path = os.path.join(MANIFEST_DIR, f"{split}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def download_frames(members):
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {}
    missing = []
    for m in members:
        p = os.path.join(CACHE_DIR, m)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            paths[m] = p
        else:
            missing.append(m)

    if missing:
        with RemoteZip(ZIP_URL) as rz:
            names = set(rz.namelist())
        present = [m for m in missing if m in names]
        for m in missing:
            if m not in names:
                print(f"  WARN: member not in zip: {m}", flush=True)

        _local = threading.local()

        def _rz():
            rz = getattr(_local, "rz", None)
            if rz is None:
                rz = _local.rz = RemoteZip(ZIP_URL)
            return rz

        def _extract(m):
            p = os.path.join(CACHE_DIR, m)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            _rz().extract(m, CACHE_DIR)
            return m, p

        n_workers = min(12, max(1, len(present)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for m, p in ex.map(_extract, present, chunksize=8):
                paths[m] = p
    return paths


def build_split(split, n):
    rows = load_manifest(split)[:n]
    members = [r["member"] for r in rows]
    paths = download_frames(members)
    records = []
    for r in rows:
        p = paths.get(r["member"])
        if not p:
            continue
        img = Image.open(p).convert("RGB")
        gt = {
            "triplets": r["triplets"],
            "instruments": r["instruments"],
            "verbs": r["verbs"],
            "targets": r["targets"],
        }
        records.append({
            "data_source": "cholec_t50",
            "prompt": [{"role": "user", "content": "<image>\n" + PROMPT}],
            "images": [img],
            "ability": "recognition",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"video": r["video"], "frame": r["frame"]},
        })
    ds = Dataset.from_list(records)
    return ds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split, n in (("train", TRAIN_N), ("val", VAL_N)):
        print(f"building {split} ({n} rows) ...", flush=True)
        ds = build_split(split, n)
        out = os.path.join(OUT_DIR, f"{split}.parquet")
        ds.to_parquet(out)
        print(f"  wrote {out} ({len(ds)} rows)", flush=True)
        print("  columns:", list(ds.features.keys()), flush=True)
        print("  images feature:", ds.features.get("images"), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
