from __future__ import annotations

import numpy as np

from XPolicyLab.policy.Pi_05.model import Model


class _FakePolicy:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.calls: list[dict[str, object]] = []

    def infer(self, _observation, **kwargs):
        self.calls.append(kwargs)
        return {"actions": self.actions}


def _model(actions: np.ndarray) -> Model:
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
    model.action_horizon = actions.shape[0]
    model.num_steps = 10
    model.policy = _FakePolicy(actions)
    return model


def test_pi05_paint_pads_prefix_before_official_input_transform() -> None:
    actions = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
    model = _model(actions)
    prefix = np.arange(2 * 14, dtype=np.float32).reshape(2, 14)

    result = model.get_action_paint(
        {"mode": "paint", "action_prefix": prefix, "delay_steps": 2}
    )

    call = model.policy.calls[0]
    condition = call["paint_action_condition"]
    assert isinstance(condition, np.ndarray)
    assert condition.shape == (5, 14)
    np.testing.assert_array_equal(condition[:2], prefix)
    np.testing.assert_array_equal(condition[2:], 0.0)
    assert call["paint_delay_steps"] == 2
    assert call["num_steps"] == 10
    assert len(result["actions"]) == 5
    assert result["paint"] == {
        "delay_steps": 2,
        "num_steps": 10,
        "model_evaluations": 30,
        "inversion": "backward_euler",
    }


def test_pi05_paint_rejects_prefix_shape_mismatch() -> None:
    model = _model(np.zeros((5, 14), dtype=np.float32))

    try:
        model.get_action_paint(
            {
                "mode": "paint",
                "action_prefix": np.zeros((3, 14), dtype=np.float32),
                "delay_steps": 2,
            }
        )
    except ValueError as exc:
        assert "must have shape (2, 14)" in str(exc)
    else:
        raise AssertionError("wrong PAINT prefix shape was accepted")
