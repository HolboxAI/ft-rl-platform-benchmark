"""Benchmark tasks: data loading, prompt encoding, and reward.

The benchmark task is CholecT50 only. It reuses the verl parquet (256 / 32 frames)
and the triplet-F1 reward of `ft-rl-platform-benchmark-main 2`, so results join
one-to-one with the Tinker / verl / Miles / SkyRL / NeMo-RL legs.

Prompts are always LEFT-padded to a fixed `max_prompt_len`. With left padding,
every prompt in the batch ends at the same position, which lets the rollout use
one static-shape graph for the whole generation (see rollout.py). Prompts longer
than `max_prompt_len` are dropped at load time, like verl's
`filter_overlong_prompts`.
"""
from __future__ import annotations

import io
import json
import os
import random
from dataclasses import dataclass, field

import torch

from common.rewards import cholect50 as cholect50_reward

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Example:
    prompt: object  # chat messages (text) or (messages, PIL image) for VLM
    gold: object
    meta: dict = field(default_factory=dict)


class Task:
    name = "task"
    is_vlm = False
    # Name of the "headline" metric for final_results.json curves.
    curve_keys = ("reward",)

    def __init__(self, tokenizer_or_processor, max_prompt_len: int):
        self.proc = tokenizer_or_processor
        self.tok = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_prompt_len = max_prompt_len

    def load(self, split: str) -> list:
        raise NotImplementedError

    def encode(self, examples: list) -> dict:
        raise NotImplementedError

    def score(self, completion: str, ex: Example) -> dict:
        raise NotImplementedError

    def batches(self, examples: list, batch_size: int, seed: int, epochs: int = 1):
        """Yield shuffled prompt batches; drops the ragged remainder (as verl does)."""
        rng = random.Random(seed)
        for _ in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for i in range(0, len(order) - batch_size + 1, batch_size):
                yield [examples[j] for j in order[i:i + batch_size]]


class CholecT50Task(Task):
    """VLM surgical action-triplet recognition, Qwen3.5-4B from the Tinker SFT
    checkpoint. Reads the same verl-format parquet as the verl / SkyRL / Miles
    legs (256 train / 32 val) and uses the same blended triplet-F1 reward.

    Not yet run on Trainium / TorchTPU — see EXPERIMENT_PLAN.md.
    """

    name = "cholect50_rl"
    is_vlm = True
    curve_keys = ("reward", "triplet_f1")

    def __init__(self, processor, max_prompt_len: int = 2048, data_dir: str = None):
        super().__init__(processor, max_prompt_len)
        self.data_dir = data_dir or os.path.join(REPO_ROOT, "data", "cholect50", "parquet")

    @staticmethod
    def _image(cell):
        import numpy as np
        from PIL import Image

        # verl parquet stores `images` as a one-element list of {"bytes", "path"}.
        img = cell[0] if isinstance(cell, (list, tuple, np.ndarray)) else cell
        if isinstance(img, dict):
            img = Image.open(io.BytesIO(img["bytes"]))
        return img.convert("RGB")

    @staticmethod
    def _messages(prompt_cell):
        text = prompt_cell[0]["content"].replace("<image>\n", "").replace("<image>", "")
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    def _text(self, messages):
        return self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def load(self, split: str = "train") -> list:
        import pandas as pd

        df = pd.read_parquet(os.path.join(self.data_dir, f"{split}.parquet"))
        out, dropped, sizes = [], 0, set()
        for _, row in df.iterrows():
            msgs = self._messages(list(row["prompt"]))
            img = self._image(row["images"])
            sizes.add(img.size)
            enc = self.proc(text=[self._text(msgs)], images=[img], return_tensors="pt")
            n = enc["input_ids"].shape[1]
            if n > self.max_prompt_len:
                dropped += 1
                continue
            gt = json.loads(row["reward_model"]["ground_truth"])
            out.append(Example((msgs, img), gt, {"prompt_tokens": n}))
        if len(sizes) > 1:
            # Different image sizes -> different image-token counts -> different
            # vision-encoder shapes -> recompiles on XLA. Flag it loudly.
            print(f"[cholect50] WARNING: {len(sizes)} distinct image sizes {sorted(sizes)[:4]}...", flush=True)
        print(f"[cholect50] {split}: loaded {len(out)}, dropped {dropped} over {self.max_prompt_len} tokens", flush=True)
        return out

    def encode(self, examples: list) -> dict:
        texts = [self._text(ex.prompt[0]) for ex in examples]
        images = [ex.prompt[1] for ex in examples]
        return dict(self.proc(text=texts, images=images, padding="max_length",
                              max_length=self.max_prompt_len, return_tensors="pt"))

    def score(self, completion: str, ex: Example) -> dict:
        pred = cholect50_reward.parse_triplets(completion)
        reward = cholect50_reward.reward_single(pred, ex.gold)
        triplet_f1 = cholect50_reward._f1(pred, set(ex.gold.get("triplets", []))) if pred is not None else 0.0
        return {"reward": reward, "triplet_f1": triplet_f1, "parse_ok": float(pred is not None)}


def make_task(name: str, tok_or_proc, max_prompt_len: int, data_path: str = None) -> Task:
    if name == "cholect50":
        return CholecT50Task(tok_or_proc, max_prompt_len, data_path)
    raise ValueError(f"unknown task {name!r}")


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
