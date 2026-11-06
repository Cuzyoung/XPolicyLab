"""Explicit per-model AutoHorizon options without replacing sampler functions.

The selector algorithm remains the pinned implementation. The request context is
local to one synchronous model invocation; it never changes process-wide defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass

_DEFAULTS = {"hold_thr": 0.3, "run_len": 1, "max_entropy_q": 0.9}


@dataclass
class _Selection:
    options: dict
    used: bool = False


_selection: ContextVar[_Selection | None] = ContextVar("autohorizon_selection", default=None)


def validate_autohorizon(options: Mapping) -> dict:
    """Validate server YAML without importing model dependencies."""
    if not isinstance(options, Mapping) or options.keys() - _DEFAULTS.keys():
        raise ValueError("autohorizon accepts only hold_thr, run_len, max_entropy_q")
    values = {**_DEFAULTS, **options}
    for key in ("hold_thr", "max_entropy_q"):
        value = values[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"autohorizon.{key} must be a finite number")
    if values["hold_thr"] < 0:
        raise ValueError("autohorizon.hold_thr must be nonnegative")
    if not 0 <= values["max_entropy_q"] <= 1:
        raise ValueError("autohorizon.max_entropy_q must be in [0, 1]")
    run_len = values["run_len"]
    if isinstance(run_len, bool) or not isinstance(run_len, int) or run_len < 1:
        raise ValueError("autohorizon.run_len must be a positive integer")
    return values


def select_horizon(attention, *, selector=None):
    """Call the native selector with this invocation's validated configuration."""
    if selector is None:
        from XPolicyLab.utils.autohorizon_official import bidir_soft_pointer

        selector = bidir_soft_pointer
    active = _selection.get()
    options = _DEFAULTS if active is None else active.options
    # The pinned convolution squeezes a full-horizon window into a scalar.
    if options["run_len"] >= attention.shape[0]:
        raise ValueError("autohorizon.run_len must be smaller than the action horizon")
    result = selector(attention, **options)
    if active is not None:
        active.used = True
    return result


class ConfiguredAutoHorizon:
    """Delegate the model contract; configure only explicit AutoHorizon calls."""

    def __init__(self, model, options):
        if "autohorizon" not in model.sampling_modes():
            raise ValueError("Model does not support AutoHorizon")
        self._model = model
        self._options = validate_autohorizon(options)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def get_action_autohorizon(self, sampling):
        active = _Selection(dict(self._options))
        token = _selection.set(active)
        try:
            result = self._model.get_action_autohorizon(sampling)
        finally:
            _selection.reset(token)
        if not active.used:
            raise RuntimeError("Model did not use the configurable AutoHorizon selector")
        return {
            **result,
            "autohorizon": {
                **result["autohorizon"],
                "hold_threshold": self._options["hold_thr"],
                "run_length": self._options["run_len"],
                "entropy_quantile": min(self._options["max_entropy_q"], 0.999),
            },
        }
