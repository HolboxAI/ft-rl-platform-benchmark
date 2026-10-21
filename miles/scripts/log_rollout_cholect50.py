"""CholecT50 rollout metrics hook for Miles.

Miles logs ``rollout/raw_reward_unfiltered`` as the mean of ``--reward-key``
(here ``reward_value``, the blended triplet reward that is the GRPO training
signal). This hook logs the four component F1 scores so the per-step curves match
the verl reference (``critic/rewards/mean`` plus the component breakdown).

Contract (``--custom-rollout-log-function-path``):
    def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool
"""
from __future__ import annotations


def _reward_field(sample, key: str) -> float:
    r = getattr(sample, "reward", None)
    if isinstance(r, dict):
        v = r.get(key)
        return float(v) if v is not None else 0.0
    return 0.0


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    extra = {
        "rollout/triplet_f1": _mean([_reward_field(s, "triplet_f1") for s in samples]),
        "rollout/instrument_f1": _mean([_reward_field(s, "instrument_f1") for s in samples]),
        "rollout/verb_f1": _mean([_reward_field(s, "verb_f1") for s in samples]),
        "rollout/target_f1": _mean([_reward_field(s, "target_f1") for s in samples]),
    }

    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(extra)
        return False

    from miles.utils.metric_utils import compute_rollout_step
    from miles.utils.tracking_utils import tracking

    extra["rollout/step"] = compute_rollout_step(args, rollout_id)
    tracking.log(args, extra, step_key="rollout/step")
    return False
