"""ORena rollout metrics hook for Miles.

Miles' default rollout logging reports `rollout/raw_reward_unfiltered` as the
mean of `--reward-key` (here `reward_value`, i.e. the dense "correctness" that is
the GRPO *training* signal). Fireworks reports the strict 0/1 `exact_match` as its
headline number (0.1925 -> 0.3200), which is a *separate* key in our reward dict.
This hook logs it (plus the two format-constraint flags) so the curves are directly
comparable. Returning False keeps the default logging intact.

Contract (--custom-rollout-log-function-path):
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
        "rollout/exact_match": _mean([_reward_field(s, "exact_match") for s in samples]),
        "rollout/no_hallucinated_class": _mean([_reward_field(s, "no_hallucinated_class") for s in samples]),
        "rollout/no_excluded_class": _mean([_reward_field(s, "no_excluded_class") for s in samples]),
    }

    if rollout_extra_metrics is not None:
        # The default logger merges this dict into its output after we return False.
        rollout_extra_metrics.update(extra)
        return False

    # rollout_extra_metrics was None (legacy rollout path): log directly.
    from miles.utils.metric_utils import compute_rollout_step
    from miles.utils.tracking_utils import tracking

    extra["rollout/step"] = compute_rollout_step(args, rollout_id)
    tracking.log(args, extra, step_key="rollout/step")
    return False
