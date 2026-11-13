"""Convert verl-format CholecT50 parquet to SkyRL VLM parquet.

The verl parquet (built by ``verl-cholec/builder/prep_data.py``) carries columns:

    data_source, prompt, images (list[PIL.Image]), ability,
    reward_model ({"style", "ground_truth"}), extra_info

SkyRL's VLM dataset contract (see the SkyRL vision-language tutorial) expects:

    prompt       -> list of messages whose content is a list of parts:
                    [{"type":"image_url","image_url":{"url":"data:..."}},
                     {"type":"text","text":"..."}]
    env_class    -> "cholec"
    reward_spec  -> {"method":"rule","ground_truth": <json string>}
    extra_info   -> passthrough

The ground-truth JSON and the exact 256/32 partition are carried over unchanged.
"""
import argparse
import base64
import io
import json
import os

from datasets import Dataset
from PIL import Image

PROMPT = (
    "You are a surgical activity recognition model. List every surgical action "
    "triplet (instrument-verb-target) visible in the endoscopic frame. Reply only "
    "as comma-separated triplets, each formatted instrument-verb-target. If no "
    "action is visible, reply 'none'."
)


def _pil_to_data_uri(img: Image.Image) -> str:
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def convert_row(example: dict) -> dict:
    imgs = example.get("images") or []
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    content = []
    for img in imgs:
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_uri(img)}})
    content.append({"type": "text", "text": PROMPT})

    reward_model = example.get("reward_model") or {}
    gt = reward_model.get("ground_truth", {})
    if isinstance(gt, dict):
        gt = json.dumps(gt)

    return {
        "prompt": [{"role": "user", "content": content}],
        "env_class": "cholec",
        "reward_spec": {"method": "rule", "ground_truth": gt},
        "extra_info": example.get("extra_info", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verl_dir", required=True, help="dir with verl train.parquet/val.parquet")
    ap.add_argument("--out_dir", required=True, help="dir to write SkyRL parquet")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("train", "val"):
        src = os.path.join(args.verl_dir, f"{split}.parquet")
        ds = Dataset.from_parquet(src)
        ds = ds.map(convert_row, remove_columns=ds.column_names)
        out = os.path.join(args.out_dir, f"{split}.parquet")
        ds.to_parquet(out)
        print(f"wrote {out} ({len(ds)} rows)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
