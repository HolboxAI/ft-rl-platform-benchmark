"""GRPO on 32 DS-1000 problems for 10 steps with stock TRL, timing every RL phase.

The same script runs on a g6e GPU (CUDA) and on Trainium (trn1 / trn2 / trn3, via
torch-neuronx / XLA). The backend is detected automatically (override: BENCH_BACKEND).

  python -m train.grpo_ds1000_bench --platform g6e.xlarge
  python -m train.grpo_ds1000_bench --platform trn2.3xlarge --nki-ops rmsnorm
  python -m train.grpo_ds1000_bench --platform trn1.2xlarge --sync-mode step

Outputs go to results/<run_name>/ (see metrics/csv_store.py for every file and column).
After training, metrics/summarize_run.py runs automatically and writes phase_summary.csv
and run_summary.csv into the same folder.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench import platform_utils as pu  # noqa: E402
from bench.phase_timer import PhaseTimer  # noqa: E402
from bench.ds1000_reward import DS1000Reward  # noqa: E402
from bench.static_shapes import force_prompt_padding, generation_kwargs  # noqa: E402
from metrics.csv_store import RunStore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs/grpo_ds1000.json"))
    p.add_argument("--platform", required=True,
                   help="label for this machine, e.g. g6e.xlarge, trn1.2xlarge, trn2.3xlarge, trn3.*")
    p.add_argument("--sync-mode", choices=["phase", "step"], default="phase",
                   help="phase: wait for the device at every phase edge (accurate per-phase times; "
                        "on Neuron this also splits the XLA graph per phase). "
                        "step: wait only at step edges (true end-to-end step time; phase times "
                        "are then host-side only).")
    p.add_argument("--nki-ops", choices=["none", "rmsnorm"], default="none",
                   help="Trainium only: use NKI kernels in rollout forward passes")
    p.add_argument("--output-root", default=os.path.join(ROOT, "results"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--model", default=None, help="override config model_name")
    p.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    return p.parse_args()


def build_grpo_config(cfg, out_dir):
    """Create GRPOConfig, dropping options this TRL version doesn't have (they are logged)."""
    from trl import GRPOConfig
    wanted = dict(
        output_dir=out_dir,
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="constant",
        warmup_steps=0,
        beta=cfg["beta"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_grad_norm=cfg["max_grad_norm"],
        num_iterations=1,
        seed=cfg["seed"],
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=False,            # weights are loaded in bf16 directly; no autocast on either platform
        fp16=False,
        use_vllm=False,        # stock HF generate on both platforms
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        log_completions=False,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in wanted if k not in fields)
    return GRPOConfig(**{k: v for k, v in wanted.items() if k in fields}), dropped


def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    if args.model:
        cfg["model_name"] = args.model
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    gen_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    assert gen_batch == cfg["prompts_per_step"] * cfg["num_generations"], \
        "per_device_train_batch_size * gradient_accumulation_steps must equal prompts_per_step * num_generations"

    backend = pu.detect_backend()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.platform}_{backend}_nki-{args.nki_ops}_sync-{args.sync_mode}_{stamp}"
    run_dir = os.path.join(args.output_root, run_name)
    store = RunStore(run_dir)
    print(f"[bench] backend={backend} run_dir={run_dir}", flush=True)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train.instrumented_trainer import BenchState, InstrumentedGRPOTrainer

    torch.manual_seed(cfg["seed"])
    timer = PhaseTimer(backend, sync=(args.sync_mode == "phase"), on_event=store.phase_events.write)

    # ---- data
    subset = os.path.join(ROOT, cfg["subset_file"])
    if not os.path.exists(subset):
        sys.exit(f"missing {subset}; run: python -m data.prepare_ds1000_subset")
    ds = load_dataset("json", data_files=subset, split="train")
    assert len(ds) == cfg["num_instances"], f"subset has {len(ds)} rows, expected {cfg['num_instances']}"

    # ---- model + tokenizer (timed: load time differs a lot between platforms)
    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if cfg["static_shapes"]:
        force_prompt_padding(tok, cfg["max_prompt_length"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=getattr(torch, cfg["torch_dtype"]),
        attn_implementation=cfg["attn_implementation"])
    load_s = time.perf_counter() - t_load

    # ---- NKI kernels (Trainium only)
    nki_counters, nki_patched = {}, 0
    if args.nki_ops != "none":
        if backend != "neuron":
            sys.exit("--nki-ops needs a Trainium (neuron) backend")
        from bench.nki_ops import patch_rmsnorm_for_rollout
        nki_patched = patch_rmsnorm_for_rollout(model, nki_counters)
        print(f"[bench] NKI RMSNorm patched into {nki_patched} modules", flush=True)

    reward = DS1000Reward(timeout_s=cfg["reward_timeout_s"], workers=cfg["reward_workers"],
                          tokenizer=tok, timer=timer)
    bench = BenchState(timer, store, backend, cfg["warmup_steps_excluded_from_stats"],
                       generation_kwargs(cfg, tok), nki_counters, reward)
    reward.on_result = bench.on_reward_result

    grpo_cfg, dropped = build_grpo_config(cfg, os.path.join(run_dir, "trainer_out"))
    t_init = time.perf_counter()
    trainer = InstrumentedGRPOTrainer(model=model, args=grpo_cfg, train_dataset=ds,
                                      reward_funcs=[reward], processing_class=tok, bench=bench)
    if args.nki_ops != "none" and getattr(trainer, "ref_model", None) is not None:
        nki_patched += patch_rmsnorm_for_rollout(trainer.ref_model, nki_counters)
    init_s = time.perf_counter() - t_init

    info = pu.collect_platform_info(backend)
    info.update({
        "run_name": run_name, "platform_label": args.platform, "sync_mode": args.sync_mode,
        "nki_ops": args.nki_ops, "nki_modules_patched": nki_patched,
        "model_load_s": round(load_s, 3), "trainer_init_s": round(init_s, 3),
        "hooks_installed": ";".join(bench.hooks_installed),
        "hooks_missing": ";".join(bench.hooks_missing),
        "grpo_config_dropped_keys": ";".join(dropped),
        "generate_kwargs": json.dumps({k: v for k, v in bench.extra_generate_kwargs.items()}),
        "cmdline": " ".join(sys.argv),
        **{f"cfg_{k}": v for k, v in cfg.items() if not k.startswith("_")},
    })
    store.write_run_info(info)

    # ---- train
    t_train = time.perf_counter()
    status = "ok"
    try:
        trainer.train()
    except Exception as e:
        status = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        info["train_wall_s"] = round(time.perf_counter() - t_train, 3)
        info["status"] = status[:500]
        store.write_run_info(info)
        store.close()
        from metrics.summarize_run import summarize
        try:
            summarize(run_dir)
        except Exception as e:  # summary must never hide the training result
            print(f"[bench] summarize failed: {e}", flush=True)
    print(f"[bench] done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
