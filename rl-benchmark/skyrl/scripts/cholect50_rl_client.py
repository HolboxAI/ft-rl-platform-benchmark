#!/usr/bin/env python3
"""CholecT50 GRPO through the Tinker API, served by a self-hosted SkyRL server.

Port of `ft-rl-platform-benchmark-main 2/tinker/scripts/tinker_rl.py`. The
training loop is tinker_cookbook's RL trainer, unchanged; only `base_url` moves
from Thinking Machines' service to a SkyRL server, so the same client runs
against every SkyRL backend:

    GPU control   skyrl.tinker.api --backend fsdp   (vision supported)
    TPU           skyrl.tinker.api --backend jax    (no vision path today)
    Trainium      skyrl.tinker.api --backend jax    on jax-neuronx (beta)

Differences from the Tinker leg (recorded in final_results.json):
  * starts from the MERGED SFT checkpoint served as the base model (same weights
    as the verl / Miles / SkyRL GPU legs) with a fresh LoRA, instead of loading
    Tinker's LoRA SFT weights;
  * reads the shared 256 / 32 verl parquet (same frames as verl / Miles / SkyRL);
  * greedy val eval (n=32) before and after, like the other legs.

Two presets (see PRESETS below): `tinker` (default) reproduces the Tinker leg's
recipe; `skyrl-grpo` gets as close to the existing SkyRL GPU run as the Tinker
API allows (full model, clipped GRPO policy loss, std-normalised advantages,
8 updates per step, temp 0.7; KL is still 0).

    TINKER_API_KEY=tml-dummy python skyrl/scripts/cholect50_rl_client.py \
        --base-url http://localhost:8000 --model ~/models/cholec-sft-qwen3.5-4b \
        --preset skyrl-grpo --out-dir skyrl/logs/run1
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import chz  # noqa: E402
import tinker  # noqa: E402
from PIL import Image  # noqa: E402
from tinker_cookbook import checkpoint_utils, renderers  # noqa: E402
from tinker_cookbook.image_processing_utils import get_image_processor  # noqa: E402
from tinker_cookbook.rl import train  # noqa: E402
from tinker_cookbook.rl.types import (  # noqa: E402
    Action, ActionExtra, Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from common.rewards.cholect50 import _f1, parse_triplets, reward_single  # noqa: E402

PROMPT = (
    "You are a surgical activity recognition model. List every surgical action "
    "triplet (instrument-verb-target) visible in the endoscopic frame. Reply only "
    "as comma-separated triplets, each formatted instrument-verb-target. If no "
    "action is visible, reply 'none'."
)
RENDERER_NAME = "qwen3_5_disable_thinking"
DEFAULT_DATA = os.path.join(REPO, "data", "cholect50", "parquet")


def load_rows(data_dir: str, split: str) -> list[dict]:
    """verl parquet -> [{"image": PIL, "gt": dict}]; identical frames to the other legs."""
    import pandas as pd

    df = pd.read_parquet(os.path.join(data_dir, f"{split}.parquet"))
    rows = []
    for _, r in df.iterrows():
        cell = r["images"][0]
        img = cell if isinstance(cell, Image.Image) else Image.open(io.BytesIO(cell["bytes"]))
        rows.append({"image": img.convert("RGB"), "gt": json.loads(r["reward_model"]["ground_truth"])})
    return rows


def convo(image):
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]


class TripletEnv(Env):
    def __init__(self, image, gt, renderer):
        self.image, self.gt, self.renderer = image, gt, renderer

    async def initial_observation(self):
        return self.renderer.build_generation_prompt(convo(self.image)), self.renderer.get_stop_sequences()

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, _ = self.renderer.parse_response(action)
        pred = parse_triplets(renderers.get_text_content(message))
        reward = reward_single(pred, self.gt)
        return StepResult(
            reward=reward, episode_done=True, next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={"triplet_reward": reward,
                     "triplet_f1": _f1(pred, set(self.gt.get("triplets", []))) if pred is not None else 0.0,
                     "parse_ok": float(pred is not None)},
        )


class TripletGroupBuilder(EnvGroupBuilder):
    def __init__(self, image, gt, renderer, n):
        self.image, self.gt, self.renderer, self.n = image, gt, renderer, n

    async def make_envs(self) -> Sequence[Env]:
        return [TripletEnv(self.image, self.gt, self.renderer) for _ in range(self.n)]

    def logging_tags(self) -> list[str]:
        return ["cholec50", "triplet"]


class TripletRLDataset(RLDataset):
    def __init__(self, rows, batch_size, group_size, renderer):
        self.rows, self.batch_size, self.group_size, self.renderer = rows, batch_size, group_size, renderer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch = self.rows[index * self.batch_size: (index + 1) * self.batch_size]
        return [TripletGroupBuilder(r["image"], r["gt"], self.renderer, self.group_size) for r in batch]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)


def make_renderer(model_name: str):
    tok = get_tokenizer(model_name)
    return renderers.get_renderer(RENDERER_NAME, tok, image_processor=get_image_processor(model_name),
                                  model_name=model_name)


@chz.chz
class TripletRLDatasetBuilder(RLDatasetBuilder):
    data_dir: str = DEFAULT_DATA
    max_samples: int = 256
    batch_size: int = 16
    group_size: int = 8
    model_name: str = ""

    async def __call__(self):
        rows = load_rows(self.data_dir, "train")[: self.max_samples]
        return TripletRLDataset(rows, self.batch_size, self.group_size, make_renderer(self.model_name)), None


def evaluate(sampling_client, rows, renderer, max_tokens) -> dict:
    """Greedy val reward (temp 0), same metric as the verl / Miles / NeMo-RL legs."""
    futures = []
    for r in rows:
        prompt = renderer.build_generation_prompt(convo(r["image"]))
        futures.append(sampling_client.sample(
            prompt=prompt, num_samples=1,
            sampling_params=tinker.types.SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                                        stop=renderer.get_stop_sequences())))
    tot = {"reward": 0.0, "triplet_f1": 0.0, "parse_ok": 0.0}
    for r, fut in zip(rows, futures):
        msg, _ = renderer.parse_response(fut.result().sequences[0].tokens)
        pred = parse_triplets(renderers.get_text_content(msg))
        tot["reward"] += reward_single(pred, r["gt"])
        tot["triplet_f1"] += _f1(pred, set(r["gt"].get("triplets", []))) if pred is not None else 0.0
        tot["parse_ok"] += float(pred is not None)
    return {k: round(v / len(rows), 4) for k, v in tot.items()} | {"n": len(rows)}


def reward_curve(log_path: str) -> list:
    path = os.path.join(log_path, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    curve = []
    for line in open(path, encoding="utf-8"):
        m = json.loads(line)
        key = next((k for k in m if k.endswith("reward/total")), None) or \
            next((k for k in m if "triplet_reward" in k), None)
        if key is not None:
            curve.append(round(float(m[key]), 4))
    return curve


# Two ways to run the same client. Explicit flags override the preset.
#   tinker      : the Tinker leg's recipe (LoRA 16, importance sampling, mean-centred
#                 advantages, 1 update per step, temperature 1.0).
#   skyrl-grpo  : as close to the existing SkyRL GPU run (reference_gpu/run_cholec.sh)
#                 as the Tinker API allows: full model, clipped policy loss (SkyRL
#                 "regular", eps 0.2/0.2), advantages / group std, 8 updates per step
#                 (policy_mini_batch_size=2 of 16 prompts), temperature 0.7, constant-
#                 reward groups kept. KL stays 0: SkyRL's Tinker backend forces it off.
PRESETS = {
    "tinker": dict(loss="importance_sampling", norm_adv_by_std=False, num_substeps=1, lora_rank=16,
                   temperature=1.0, remove_constant_reward_groups=True, clip_low=0.2, clip_high=0.2),
    "skyrl-grpo": dict(loss="ppo", norm_adv_by_std=True, num_substeps=8, lora_rank=0,
                       temperature=0.7, remove_constant_reward_groups=False, clip_low=0.2, clip_high=0.2),
}


def install_std_normalised_advantages(eps: float = 1e-6):
    """Replace the cookbook's mean-centred advantages with SkyRL's GRPO advantages:
    (r - mean) / (std + eps) per group, sample std (torch.std), as in
    skyrl/backends/skyrl_train/utils/ppo_utils.py:compute_grpo_outcome_advantage
    (a singleton group gets mean 0, std 1)."""
    import torch

    def compute_advantages_std(trajectory_groups_P):
        out = []
        for g in trajectory_groups_P:
            r = torch.tensor(g.get_total_rewards(), dtype=torch.float32)
            if r.numel() == 1:
                out.append(r / (1.0 + eps))
            else:
                out.append((r - r.mean()) / (r.std() + eps))
        return out

    train.compute_advantages = compute_advantages_std  # rl/train.py calls it by this module-level name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="the server's --base-model (merged SFT checkpoint path)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = one epoch (16 steps)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tinker")
    # Preset-controlled (None = take the preset's value)
    ap.add_argument("--loss", choices=("importance_sampling", "ppo"), default=None,
                    help="ppo = clipped policy loss; SkyRL's FSDP backend maps it to its 'regular' GRPO loss")
    ap.add_argument("--clip-low", type=float, default=None, help="eps_clip_low (ppo only)")
    ap.add_argument("--clip-high", type=float, default=None, help="eps_clip_high (ppo only)")
    ap.add_argument("--norm-adv-by-std", action=argparse.BooleanOptionalAction, default=None,
                    help="divide group-centred advantages by the group std (SkyRL GRPO default)")
    ap.add_argument("--num-substeps", type=int, default=None, help="optimizer updates per rollout batch")
    ap.add_argument("--lora-rank", type=int, default=None, help="0 = full-model training")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--remove-constant-reward-groups", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--platform", default="skyrl")
    ap.add_argument("--hourly-usd", type=float, default=float(os.environ.get("HOURLY_USD", "0") or 0))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.norm_adv_by_std:
        install_std_normalised_advantages()
    loss_fn_config = ({"clip_low_threshold": 1.0 - args.clip_low, "clip_high_threshold": 1.0 + args.clip_high}
                      if args.loss == "ppo" else None)
    print(f"[config] preset={args.preset} loss={args.loss} cfg={loss_fn_config} norm_by_std={args.norm_adv_by_std} "
          f"substeps={args.num_substeps} lora_rank={args.lora_rank} temp={args.temperature} lr={args.lr}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    model = os.path.expanduser(args.model)
    log_path = os.path.join(args.out_dir, "cookbook")

    renderer = make_renderer(model)
    val = [] if args.skip_eval else load_rows(args.data_dir, "val")
    service = tinker.ServiceClient(base_url=args.base_url, api_key=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    evals = []
    t0 = time.perf_counter()
    if val:
        e = evaluate(service.create_sampling_client(base_model=model), val, renderer, args.max_tokens)
        evals.append({"step": 0, **e})
        print(f"[eval] before: {e}", flush=True)

    config = train.Config(
        base_url=args.base_url,
        learning_rate=args.lr,
        dataset_builder=TripletRLDatasetBuilder(data_dir=args.data_dir, max_samples=args.max_samples,
                                                batch_size=args.batch_size, group_size=args.group_size,
                                                model_name=model),
        model_name=model,
        recipe_name="cholec_rl_skyrl",
        renderer_name=RENDERER_NAME,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_path=log_path,
        loss_fn=args.loss,
        loss_fn_config=loss_fn_config,
        num_substeps=args.num_substeps,
        lora_rank=args.lora_rank,
        eval_every=0,
        save_every=0,
        remove_constant_reward_groups=args.remove_constant_reward_groups,
        num_groups_to_log=2,
        max_steps=(None if args.max_steps == 0 else args.max_steps),
    )
    t_train = time.perf_counter()
    asyncio.run(train.main(config))
    train_s = time.perf_counter() - t_train

    rec = checkpoint_utils.get_last_checkpoint(log_path, required_key="sampler_path")
    if val and rec is not None and rec.sampler_path:
        e = evaluate(service.create_sampling_client(model_path=rec.sampler_path), val, renderer, args.max_tokens)
        evals.append({"step": "final", **e})
        print(f"[eval] after: {e}", flush=True)
    wall = time.perf_counter() - t0

    curve = reward_curve(log_path)
    exp = {
        "experiment": "cholect50_rl",
        "method": "grpo",
        "framework": "SkyRL Tinker API server + tinker_cookbook RL client",
        "state": "COMPLETED" if curve else "UNKNOWN",
        "base_model": model,
        "hf_model_id": "Qwen/Qwen3.5-4B",
        "preset": args.preset,
        "lora_rank": args.lora_rank or None,
        "learning_rate": args.lr,
        "kl_beta": 0.0,
        "loss_fn": args.loss,
        "loss_fn_config": loss_fn_config,
        "advantage": "group (r-mean)/(std+1e-6)" if args.norm_adv_by_std else "group r-mean",
        "num_substeps": args.num_substeps,
        "remove_constant_reward_groups": args.remove_constant_reward_groups,
        "temperature": args.temperature,
        "n_samples_per_prompt": args.group_size,
        "rollout_batch_size": args.batch_size,
        "max_output_tokens": args.max_tokens,
        "dataset_size": args.max_samples,
        "val_dataset_size": len(val),
        "steps": len(curve),
        "score_curve": curve,
        "start_score": curve[0] if curve else None,
        "final_score": curve[-1] if curve else None,
        "eval_curve": evals,
        "start_val_reward": evals[0]["reward"] if evals else None,
        "final_val_reward": evals[-1]["reward"] if len(evals) > 1 else None,
        "wall_time_s": round(wall, 1),
        "train_time_s": round(train_s, 1),
        "estimated_cost_usd": round(args.hourly_usd * wall / 3600, 2) if args.hourly_usd else None,
        "base_url": args.base_url,
    }
    results = {"platform": args.platform,
               "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "experiments": {"cholect50_rl": exp},
               "findings": ["kl_beta=0: the SkyRL Tinker API does not support kl_penalty_coef > 0 (docs: limitations). "
                            "Matches the Tinker leg, differs from verl/Miles (0.04).",
                            f"temperature={args.temperature}: tinker_cookbook 0.5.7 defaults to 1.0 (the Tinker leg ran at 1.0); verl/Miles used 0.7.",
                            "Adam betas (0.9, 0.95), eps 1e-8: hard-coded in tinker_cookbook rl/train.py:train_step; "
                            "the SkyRL entrypoint run used SkyRL's own optimizer defaults."]
                           + (["loss_fn=ppo through the Tinker API is mapped to SkyRL's 'regular' policy loss but is "
                               "documented as not yet validated (docs: tinker/limitations)."] if args.loss == "ppo" else [])}
    with open(os.path.join(args.out_dir, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'final_results.json')}", flush=True)


if __name__ == "__main__":
    main()
