from __future__ import annotations

import numpy as np

from XPolicyLab.policy.Pi_05.model import Model


class _FakePolicy:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.calls: list[dict[str, int]] = []

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
    model.action_horizon = actions.shape[1]
    model.num_steps = 10
    model.policy = _FakePolicy(actions)
    return model


def test_pi05_aac_requests_one_batched_sample_and_decodes_candidates() -> None:
    actions = np.arange(4 * 5 * 14, dtype=np.float32).reshape(4, 5, 14)
    model = _model(actions)

    result = model.get_action_aac({"mode": "aac", "num_samples": 4})

    assert model.policy.calls == [{"num_steps": 10, "num_samples": 4}]
    assert len(result["actions"]) == 4
    assert len(result["actions"][0]) == 5
    first = result["actions"][0][0]
    assert first["left_arm_joint_state"].tolist() == list(range(6))
    assert first["left_ee_joint_state"].tolist() == [6.0]
    assert first["right_arm_joint_state"].tolist() == list(range(7, 13))
    assert first["right_ee_joint_state"].tolist() == [13.0]


def test_pi05_aac_rejects_wrong_candidate_shape() -> None:
    model = _model(np.zeros((3, 5, 14), dtype=np.float32))

    try:
        model.get_action_aac({"mode": "aac", "num_samples": 4})
    except ValueError as exc:
        assert "must have shape (4, 5, 14)" in str(exc)
    else:
        raise AssertionError("wrong candidate count was accepted")
