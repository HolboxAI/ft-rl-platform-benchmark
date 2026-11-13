"""SkyRL entrypoint for the CholecT50 GRPO (RLVR) run on Cloud TPU.

Same as the GPU run's ``ft-rl-platform-benchmark-git/skyrl/scripts/entrypoint.py``
(register the CholecT50 env, run SkyRL's standard PPO/GRPO experiment from
Hydra-style overrides), with four changes:

* the env is registered from ``cholec_env`` (a verbatim copy of the GPU run's
  ``env.py``, renamed so the module name cannot shadow anything on ``PYTHONPATH``);
* it refuses to start unless ``SKYRL_DEVICE=tpu`` and the skyrl-tpu patch are in place,
  so a mis-set environment fails here and not inside a Ray worker;
* the Qwen3.5 vision-RoPE device patch is kept but is only needed with
  ``fsdp_config.cpu_offload=true``; the TPU run trains without CPU offload;
* every metrics dict the trainer logs is also appended to ``$SKYRL_METRICS_JSONL``
  (one JSON object per ``tracker.log`` call), so the run can be plotted
  (``analysis/plot_run.py``) while keeping the GPU run's ``trainer.logger=console``.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl_gym.envs import register

mp.set_start_method("spawn", force=True)


def apply_vision_rope_device_patch() -> None:
    """Make ``Qwen3_5VisionRotaryEmbedding.inv_freq`` follow ``position_ids.device``.

    Verbatim from the GPU run. Under FSDP ``cpu_offload=true`` the non-persistent
    ``inv_freq`` buffer stays on CPU while activations are on the accelerator.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen35

    def _forward(self, position_ids):
        inv_freq = self.inv_freq.to(position_ids.device)
        return (position_ids.unsqueeze(-1) * inv_freq).flatten(1)

    cls = getattr(_qwen35, "Qwen3_5VisionRotaryEmbedding", None)
    if cls is None:
        print("[cholec] WARN: Qwen3_5VisionRotaryEmbedding not found; skipping patch", flush=True)
        return
    cls.forward = _forward
    print("[cholec] patched Qwen3_5VisionRotaryEmbedding.forward (device-safe inv_freq)", flush=True)


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    try:
        return float(v)  # numpy / torch scalars
    except (TypeError, ValueError):
        return str(v)


def install_metrics_jsonl() -> None:
    """Tee ``Tracking.log`` into a JSONL file (path from ``SKYRL_METRICS_JSONL``)."""
    path = os.environ.get("SKYRL_METRICS_JSONL")
    if not path:
        return
    from skyrl.train.utils.tracking import Tracking

    original = Tracking.log
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, data, step, commit=False):
        try:
            with open(path, "a") as f:
                f.write(json.dumps({"step": step, "time": time.time(), "data": _jsonable(data)}) + "\n")
        except Exception as e:  # never let metrics capture break training
            print(f"[cholec] WARN: could not write metrics to {path}: {e}", flush=True)
        return original(self, data, step, commit=commit)

    Tracking.log = log
    print(f"[cholec] metrics -> {path}", flush=True)


def check_tpu_setup(cfg: SkyRLTrainConfig) -> None:
    if os.environ.get("SKYRL_DEVICE") != "tpu":
        sys.exit("SKYRL_DEVICE=tpu is not set; source skyrl_cholect50/tpu_env.sh first")
    try:
        from skyrl.backends.skyrl_train.utils import device  # noqa: F401  (added by the skyrl-tpu patch)
    except ImportError:
        sys.exit("this SkyRL checkout lacks the skyrl-tpu patch (skyrl/backends/skyrl_train/utils/device.py)")
    ie = cfg.generator.inference_engine
    problems = []
    if cfg.trainer.placement.colocate_all or cfg.trainer.placement.colocate_policy_ref:
        problems.append("TorchTPU allows one process per chip: set colocate_all=false and colocate_policy_ref=false")
    if ie.run_engines_locally:
        problems.append("rollouts run on the TPU rollout server: set run_engines_locally=false and external_*_url")
    if ie.weight_sync_backend != "reload":
        problems.append("only weight_sync_backend=reload works against the TPU rollout server")
    if cfg.trainer.flash_attn:
        problems.append("flash-attn is CUDA-only: set trainer.flash_attn=false")
    if problems:
        sys.exit("TPU config errors:\n  - " + "\n  - ".join(problems))


@ray.remote(num_cpus=1)
def cholec_entrypoint(cfg: SkyRLTrainConfig):
    apply_vision_rope_device_patch()
    install_metrics_jsonl()
    register(
        id="cholec",
        entry_point="cholec_env:CholecEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    check_tpu_setup(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(cholec_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
