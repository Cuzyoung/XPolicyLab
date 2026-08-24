from __future__ import annotations

from collections import deque

import numpy as np

from XPolicyLab.policy.Pi_05.model import Model


class _FakePolicy:
    def __init__(self, actions: np.ndarray, variances: list[np.ndarray]) -> None:
        self.actions = actions
        self.variances = list(variances)
        self.calls: list[dict[str, object]] = []

    def infer(self, _observation, **kwargs):
        self.calls.append(kwargs)
        variance = self.variances.pop(0)
        return {
            "actions": self.actions,
            "dvac": {
                "variance": variance.tolist(),
                "total_variance": float(variance.sum()),
                "tail_steps": 5,
                "action_dim": 14,
                "variance_space": "normalized_valid_action",
            },
        }


def _model(variances: list[np.ndarray]) -> Model:
    model = Model.__new__(Model)
    model.observation_window = {
        "observation/state": np.zeros((1, 14), dtype=np.float32),
        "observation/image": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "observation/left_wrist": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "observation/right_wrist": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "prompt": ["pick"],
    }
    model._latest_env_idx_list = [0]
    model.action_type = "joint"
    model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    model.robot_action_dim = 14
    model.action_dim = 32
    model.action_horizon = 4
    model.num_steps = 10
    model._dvac_history = deque()
    model._dvac_history_size = None
    model.policy = _FakePolicy(np.zeros((4, 14), dtype=np.float32), variances)
    return model


def test_pi05_dvac_uses_current_variance_only_for_cold_start() -> None:
    model = _model(
        [
            np.array([0.0, 0.0, 0.0, 10.0]),
            np.array([0.0, 10.0, 10.0, 10.0]),
        ]
    )
    sampling = {
        "mode": "dvac",
        "tail_steps": 5,
        "alpha": 0.0,
        "rolling_window_size": 5,
        "min_execution_steps": 1,
        "max_execution_steps": 4,
    }

    first = model.get_action_dvac(sampling)
    second = model.get_action_dvac(sampling)

    assert first["dvac"]["cold_start"] is True
    assert first["dvac"]["threshold"] == 2.5
    assert first["dvac"]["execution_steps"] == 3
    assert second["dvac"]["cold_start"] is False
    assert second["dvac"]["threshold"] == 2.5
    assert second["dvac"]["execution_steps"] == 1
    assert model.policy.calls == [
        {
            "num_steps": 10,
            "dvac": True,
            "dvac_tail_steps": 5,
            "dvac_action_dim": 14,
        },
        {
            "num_steps": 10,
            "dvac": True,
            "dvac_tail_steps": 5,
            "dvac_action_dim": 14,
        },
    ]


def test_pi05_dvac_reset_clears_calibration_history() -> None:
    model = _model([np.zeros(4)])
    model._dvac_history = deque([np.ones(4)], maxlen=5)
    model._dvac_history_size = 5

    model.reset()

    assert not model._dvac_history
    assert model._dvac_history_size is None


def test_pi05_dvac_follows_equation_seven_nmax_no_crossing_branch_only() -> None:
    model = _model([np.array([0.0, 0.0, 0.0, 10.0])])

    result = model.get_action_dvac(
        {
            "mode": "dvac",
            "tail_steps": 5,
            "alpha": 0.0,
            "rolling_window_size": 5,
            "min_execution_steps": 1,
            "max_execution_steps": 2,
        }
    )

    assert result["dvac"]["first_threshold_crossing"] == 3
    assert result["dvac"]["execution_steps"] == 3
