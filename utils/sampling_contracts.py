"""CPU request validation shared by model adapters; no Torch/JAX dependency.

Validation precedes model-specific normalization and device transfers. Native
samplers retain their own schedules, masks, action transforms and compilation.
"""

import math
import numpy as np


def validate_samples(sampling):
    if set(sampling) - {"mode", "num_samples"}:
        raise ValueError("AAC only accepts num_samples")
    count = sampling.get("num_samples", 20)
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("AAC num_samples must be an integer greater than one")
    return count


def validate_prefix(sampling, horizon, width):
    if set(sampling) - {"mode", "action_prefix", "delay_steps"}:
        raise ValueError("PAINT only accepts action_prefix and delay_steps")
    delay = sampling.get("delay_steps")
    if isinstance(delay, bool) or not isinstance(delay, int) or not 0 < delay < horizon:
        raise ValueError("PAINT delay_steps must be an integer in (0, horizon)")
    prefix = np.asarray(sampling.get("action_prefix"), dtype=np.float32)
    if prefix.shape != (delay, width) or not np.isfinite(prefix).all():
        raise ValueError(f"PAINT prefix must be finite with shape {(delay, width)}")
    return prefix, delay


def validate_rtc(sampling, horizon, width, *, copy=False):
    """Return physical action condition [H,D], weights [H] and positive beta.

    Pi05 requests writable copies because its native transforms mutate arrays.
    Other adapters retain zero-copy NumPy views where the input permits them.
    """
    required = {"action_condition", "condition_weights", "beta"}
    missing = sorted(required - set(sampling))
    if missing:
        raise ValueError(f"RTC sampling is missing fields: {missing}")
    if set(sampling) - (required | {"mode"}):
        raise ValueError("RTC only accepts action_condition, condition_weights and beta")
    if sampling.get("mode", "rtc") != "rtc":
        raise ValueError("RTC request mode must be rtc")
    condition = np.asarray(sampling["action_condition"], dtype=np.float32)
    weights = np.asarray(sampling["condition_weights"], dtype=np.float32)
    beta = float(sampling["beta"])
    if condition.shape != (horizon, width):
        raise ValueError(f"action_condition must have shape {(horizon, width)}, got {condition.shape}")
    if weights.shape != (horizon,):
        raise ValueError(f"condition_weights must have shape {(horizon,)}, got {weights.shape}")
    if not np.isfinite(condition).all() or not np.isfinite(weights).all():
        raise ValueError("RTC sampling arrays must be finite")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("RTC condition_weights must be in [0, 1]")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("RTC beta must be finite and positive")
    if copy:
        condition, weights = condition.copy(), weights.copy()
    return condition, weights, beta
