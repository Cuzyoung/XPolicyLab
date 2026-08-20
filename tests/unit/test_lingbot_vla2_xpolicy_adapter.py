from __future__ import annotations

from pathlib import Path

import numpy as np

from XPolicyLab.policy.LingBot_VLA2 import model
from XPolicyLab.policy.LingBot_VLA2.process_data import split_yam_vector

ROBOT_INFO = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


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


def test_explicit_bundle_uses_shared_checkpoint_resolver(tmp_path: Path) -> None:
    manifest = tmp_path / "bundle.yaml"
    manifest.write_text("schema_version: test\n", encoding="utf-8")
    assert model.resolve_bundle_manifest({"bundle_manifest_path": str(manifest)}) == manifest


def test_standard_run_name_resolves_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "RoboDojo-demo-yam_dual-joint-0"
    run_dir.mkdir()
    manifest = run_dir / "bundle.yaml"
    manifest.touch()
    monkeypatch.setattr(model, "CHECKPOINTS_DIR", tmp_path)
    resolved = model.resolve_bundle_manifest(
        {
            "bench_name": "RoboDojo",
            "ckpt_name": "demo",
            "env_cfg_type": "yam_dual",
            "action_type": "joint",
            "seed": 0,
        }
    )
    assert resolved == manifest
