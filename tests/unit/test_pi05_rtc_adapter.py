from __future__ import annotations

import numpy as np

from XPolicyLab.policy.Pi_05.model import Model


def test_pi05_rtc_copies_read_only_wire_arrays_before_openpi_transform() -> None:
    model = Model.__new__(Model)
    model.action_horizon = 5
    model.robot_action_dim = 14
    model.action_dim = 32
    captured: dict[str, object] = {}

    def get_action(**kwargs):
        captured.update(kwargs)
        return []

    model.get_action = get_action  # type: ignore[method-assign]
    condition = np.zeros((5, 14), dtype=np.float32)
    weights = np.ones(5, dtype=np.float32)
    condition.flags.writeable = False
    weights.flags.writeable = False

    model.get_action_rtc(
        {
            "action_condition": condition,
            "condition_weights": weights,
            "beta": 9.1,
        }
    )

    copied_condition = captured["action_condition"]
    copied_weights = captured["condition_weights"]
    assert isinstance(copied_condition, np.ndarray)
    assert isinstance(copied_weights, np.ndarray)
    assert copied_condition.flags.writeable
    assert copied_weights.flags.writeable
    assert not np.shares_memory(copied_condition, condition)
    assert not np.shares_memory(copied_weights, weights)
