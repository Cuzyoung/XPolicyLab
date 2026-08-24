from __future__ import annotations

import numpy as np

from XPolicyLab.policy.Pi_05.model import Model


class _FakePolicy:
    def __init__(self, actions: np.ndarray, execution_steps: int) -> None:
        self.actions = actions
        self.execution_steps = execution_steps
        self.calls: list[dict[str, object]] = []

    def infer(self, _observation, **kwargs):
        self.calls.append(kwargs)
        return {
            "actions": self.actions,
            "autohorizon": {
                "execution_steps": self.execution_steps,
                "attention_step": 3,
                "method": "bidir_soft_pointer",
            },
        }


def _model(actions: np.ndarray, execution_steps: int) -> Model:
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
    model.policy = _FakePolicy(actions, execution_steps)
    return model


def test_pi05_autohorizon_preserves_full_chunk_and_returns_execution_steps() -> None:
    actions = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
    model = _model(actions, execution_steps=3)

    result = model.get_action_autohorizon({"mode": "autohorizon"})

    assert model.policy.calls == [{"num_steps": 10, "autohorizon": True}]
    assert len(result["actions"]) == 5
    assert result["autohorizon"]["execution_steps"] == 3
    assert result["actions"][0]["left_arm_joint_state"].tolist() == list(range(6))


def test_pi05_autohorizon_rejects_out_of_range_execution_steps() -> None:
    model = _model(np.zeros((5, 14), dtype=np.float32), execution_steps=6)

    try:
        model.get_action_autohorizon({"mode": "autohorizon"})
    except ValueError as exc:
        assert "1 <= e <= 5" in str(exc)
    else:
        raise AssertionError("out-of-range execution horizon was accepted")
