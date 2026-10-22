"""Validate the existing XPolicyLab prefix request and normalize physical EEF20."""
import numpy as np


def normalize_action(architecture, actions):
    raw = np.asarray(actions)
    normalizer = architecture.normalizer
    return raw if normalizer is None else normalizer.normalize_action(raw)


def parse_request(architecture, sampling, horizon, width=20):
    allowed = {"mode", "prefix_steps", "action_condition", "condition_weights"}
    if sampling.get("mode") != "train_time_rtc" or set(sampling)-allowed:
        raise ValueError("Expected a train_time_rtc prefix request")
    count = sampling.get("prefix_steps")
    if type(count) is not int or not 0 <= count < horizon:
        raise ValueError("prefix_steps must be an integer in [0, horizon)")
    cfg = architecture.continuation
    if cfg.schedule != "step":
        raise ValueError("The existing runtime supplies a hard prefix only; deploy a step-schedule checkpoint")
    if count > cfg.prefix_max_steps or count + (cfg.ramp_steps if count else 0) >= horizon:
        raise ValueError("Requested prefix exceeds checkpoint continuation range")
    condition, weights = sampling.get("action_condition"), sampling.get("condition_weights")
    if count == 0:
        if condition is not None or weights is not None:
            raise ValueError("An empty prefix must not supply conditions or weights")
        return None, 0
    raw = np.asarray(condition, dtype=np.float32)
    if raw.shape != (horizon, width) or not np.isfinite(raw).all():
        raise ValueError(f"action_condition must be finite with shape {(horizon, width)}")
    if not np.array_equal(np.asarray(weights), np.arange(horizon)<count):
        raise ValueError("condition_weights must be a contiguous binary prefix mask")
    # Physical-to-model action mapping, not the proprio mapping. Includes any
    # unified action padding; the weight channel never enters normalization.
    return normalize_action(architecture, raw), count
