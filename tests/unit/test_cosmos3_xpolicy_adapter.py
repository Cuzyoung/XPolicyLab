from __future__ import annotations

import numpy as np

from XPolicyLab.policy.Cosmos3 import model

ROBOT_INFO = {"arm_dim": [7], "ee_dim": [1]}


def _observation() -> dict:
    head = np.full((8, 10, 3), 1, dtype=np.uint8)
    wrist = np.full((8, 10, 3), 2, dtype=np.uint8)
    exterior = np.full((8, 10, 3), 3, dtype=np.uint8)
    return {
        "vision": {
            "cam_head": {"color": head},
            "cam_left_wrist": {"color": wrist},
            "cam_right_wrist": {"color": exterior},
        },
        "state": {
            "arm_joint_state": np.arange(7, dtype=np.float32),
            "ee_joint_state": np.array([0.25], dtype=np.float32),
        },
        "instruction": "pick the object",
    }


def test_observation_maps_to_official_droid_keys_without_value_changes() -> None:
    obs = _observation()
    native = model.encode_observation(obs, {}, ROBOT_INFO)
    assert native["prompt"] == "pick the object"
    assert native["observation/wrist_image_left"] is obs["vision"]["cam_left_wrist"]["color"]
    assert native["observation/exterior_image_1_left"] is obs["vision"]["cam_head"]["color"]
    assert native["observation/exterior_image_2_left"] is obs["vision"]["cam_right_wrist"]["color"]
    assert np.shares_memory(
        native["observation/joint_position"], obs["state"]["arm_joint_state"]
    )
    assert np.shares_memory(
        native["observation/gripper_position"], obs["state"]["ee_joint_state"]
    )


def test_official_action_is_split_without_numerical_changes() -> None:
    native_action = np.arange(24, dtype=np.float32).reshape(3, 8)
    decoded = model.decode_action({"action": native_action}, ROBOT_INFO)
    repacked = np.stack(
        [np.concatenate([step["arm_joint_state"], step["ee_joint_state"]]) for step in decoded]
    )
    np.testing.assert_array_equal(repacked, native_action)


def test_model_delegates_inference_to_official_service(monkeypatch) -> None:
    native_action = np.arange(16, dtype=np.float32).reshape(2, 8)

    class FakeOfficialService:
        def __init__(self) -> None:
            self.observation = None

        def infer(self, observation):
            self.observation = observation
            return {"action": native_action}

    service = FakeOfficialService()
    monkeypatch.setattr(model, "get_robot_action_dim_info", lambda _: ROBOT_INFO)
    monkeypatch.setattr(model, "get_batch_size", lambda _: 1)
    monkeypatch.setattr(model, "create_official_service", lambda _: service)
    policy = model.Model({"env_cfg_type": "droid_single", "action_type": "joint"})
    policy.update_obs(_observation())
    decoded = policy.get_action()

    assert service.observation["prompt"] == "pick the object"
    repacked = np.stack(
        [np.concatenate([step["arm_joint_state"], step["ee_joint_state"]]) for step in decoded]
    )
    np.testing.assert_array_equal(repacked, native_action)
