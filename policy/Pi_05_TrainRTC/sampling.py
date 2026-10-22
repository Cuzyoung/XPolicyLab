"""Validate physical joint prefix requests before the native policy normalizes them."""
import numpy as np


def parse_request(sampling, horizon, width, config):
    allowed = {"mode", "prefix_steps", "action_condition", "condition_weights"}
    if sampling.get("mode") != "train_time_rtc" or set(sampling)-allowed:
        raise ValueError("Expected train_time_rtc prefix request")
    count = sampling.get("prefix_steps")
    if type(count) is not int or not 0 <= count <= config.prefix_max_steps or count >= horizon:
        raise ValueError("prefix_steps exceeds the trained range")
    target, weights = sampling.get("action_condition"), sampling.get("condition_weights")
    if count == 0:
        if target is not None or weights is not None:
            raise ValueError("Empty prefix must not supply a target or weights")
        return None, None
    if count < config.prefix_min_steps:
        raise ValueError("Prefix is shorter than the trained range")
    target = np.asarray(target, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if target.shape != (horizon, width) or not np.isfinite(target).all():
        raise ValueError("Expected a finite full physical action chunk")
    if not np.array_equal(weights, np.arange(horizon) < count):
        raise ValueError("Expected a contiguous binary prefix mask")
    return target.copy(), weights.copy()
