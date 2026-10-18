from __future__ import annotations

import os

import numpy as np

from XPolicyLab.policy.LingBot_VLA2 import model
from XPolicyLab.policy.LingBot_VLA2.process_data import split_yam_vector

ROBOT_INFO = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


def test_temporary_environment_restores_missing_variable() -> None:
    os.environ.pop("LINGBOT_TEST_PATH", None)
    with model._temporary_environment("LINGBOT_TEST_PATH", "/local/processor"):
        assert os.environ["LINGBOT_TEST_PATH"] == "/local/processor"
    assert "LINGBOT_TEST_PATH" not in os.environ


def test_temporary_environment_restores_existing_variable() -> None:
    os.environ["LINGBOT_TEST_PATH"] = "/existing/processor"
    try:
        with model._temporary_environment("LINGBOT_TEST_PATH", "/local/processor"):
            assert os.environ["LINGBOT_TEST_PATH"] == "/local/processor"
        assert os.environ["LINGBOT_TEST_PATH"] == "/existing/processor"
    finally:
        os.environ.pop("LINGBOT_TEST_PATH", None)


def _observation() -> dict:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    return {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
        },
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": np.array([6], dtype=np.float32),
            "right_arm_joint_state": np.arange(10, 16, dtype=np.float32),
            "right_ee_joint_state": np.array([16], dtype=np.float32),
        },
        "instruction": "pick the block",
    }


def test_observation_uses_official_lingbot_feature_names() -> None:
    encoded = model.encode_observation(_observation(), "fallback", ROBOT_INFO)
    assert encoded["observation.state.arm.position"].tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        10,
        11,
        12,
        13,
        14,
        15,
    ]
    assert encoded["observation.state.effector.position"].tolist() == [6, 16]
    assert encoded["task"] == "pick the block"


def test_absolute_joint_chunk_decodes_to_xpolicy_keys() -> None:
    arms = np.arange(36, dtype=np.float32).reshape(3, 12)
    effectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    actions = model.decode_actions(
        {
            "action.arm.position": arms,
            "action.effector.position": effectors,
        },
        ROBOT_INFO,
    )
    assert actions[1]["left_arm_joint_state"].tolist() == arms[1, :6].tolist()
    assert actions[1]["right_arm_joint_state"].tolist() == arms[1, 6:].tolist()
    assert actions[1]["left_ee_joint_state"].tolist() == effectors[1, :1].tolist()
    assert actions[1]["right_ee_joint_state"].tolist() == effectors[1, 1:].tolist()


def test_training_converter_splits_packed_yam_order() -> None:
    packed = np.arange(28, dtype=np.float32).reshape(2, 14)
    arms, effectors = split_yam_vector(packed)
    assert arms[0].tolist() == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    assert effectors[0].tolist() == [6, 13]


def test_hello_reports_loaded_compile_mode_and_rejects_other_mode():
    """Exercise the real HELLO builder and ManiMux identity comparator, without weights."""
    import asyncio
    from types import SimpleNamespace
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from client_server.ws.protocol.messages import MessageType
    from client_server.ws.protocol.schemas import Frame
    from manimux.policies.capabilities import metadata_mismatches

    for enabled in (True, False):
        adapter = object.__new__(model.Model)
        adapter.model_cfg = {"use_compile": not enabled}
        adapter.action_horizon = 50
        adapter.action_semantics = model.RELATIVE_ACTION_SEMANTICS
        adapter.deployment = dict.fromkeys(
            ("checkpoint_path", "training_config_path", "robot_config_path", "norm_stats_path", "qwen3vl_path"), "fixture"
        )
        adapter.deployment["native_hz"] = 30.0
        adapter.model = SimpleNamespace(use_compile=enabled, vla=SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(num_steps=10, n_action_steps=50))))
        server = PolicyServer(adapter, PolicyServerConfig(model_metadata={"use_compile": not enabled}))
        reply = asyncio.run(server.process_frame(Frame(
            message_type=MessageType.HELLO, request_id="hello", evaluation_id="offline", payload={})))
        actual = {"server": reply.payload["server"], "model": reply.payload["model_metadata"]}
        assert actual["model"]["use_compile"] is enabled
        assert not metadata_mismatches({"model": {"use_compile": enabled}}, actual)
        assert metadata_mismatches({"model": {"use_compile": not enabled}}, actual)
        del actual["model"]["use_compile"]
        assert "is missing" in metadata_mismatches({"model": {"use_compile": enabled}}, actual)[0]
