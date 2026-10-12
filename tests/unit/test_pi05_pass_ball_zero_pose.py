"""Numerical contracts and model interfaces; no checkpoint/GPU or hardware needed."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.Pi_05.model import Model
from XPolicyLab.policy.Pi_05.pass_ball_model import (
    PassBallInputs,
    PassBallZeroPoseModel,
    decode_actions,
    encode_observation,
)
from XPolicyLab.policy.Pi_05.pass_ball_pose import OPTICAL_FROM_FLU, relative_action_chunk
from XPolicyLab.policy.Pi_05.pass_ball_state import POSE_INDICES, EncodeState, ZeroNormalizedPose


def observation(index=0):
    obs = {"env_idx": index, "instruction": "pass_ball", "state": {}, "vision": {}}
    for side, position, opening, angle in (
        ("left", [0.2, -0.3, 0.4], 0.2, [0.3, -0.2, 0.1]),
        ("right", [-0.4, 0.5, 0.6], 0.8, [-0.1, 0.4, 0.2]),
    ):
        quat = Rotation.from_rotvec(angle).as_quat()
        obs["state"][f"{side}_ee_pose"] = np.r_[position, quat[[3, 0, 1, 2]]]
        obs["state"][f"{side}_ee_joint_state"] = np.array([opening])
        obs["vision"][f"cam_{side}_wrist"] = {"color": np.full((4, 6, 3), 19, np.uint8)}
    return obs


def neutral_actions(anchor):
    return relative_action_chunk(anchor, np.tile(anchor, (32, 1)))


def test_layout_axis_conversion_and_absolute_round_trip():
    obs = observation()
    anchor = encode_observation(obs)["state"]
    np.testing.assert_allclose(anchor[:3], obs["state"]["right_ee_pose"][:3])
    np.testing.assert_allclose(anchor[10:13], obs["state"]["left_ee_pose"][:3])
    assert anchor[9] == pytest.approx(0.8)
    assert anchor[19] == pytest.approx(0.2)
    actions = neutral_actions(anchor)
    steps = decode_actions(anchor, actions)
    for step in steps:
        for side in ("left", "right"):
            np.testing.assert_allclose(
                step[f"{side}_ee_pose"], obs["state"][f"{side}_ee_pose"], atol=1e-6
            )
            np.testing.assert_allclose(
                step[f"{side}_ee_joint_state"], obs["state"][f"{side}_ee_joint_state"], atol=1e-6
            )
    # Local model +X is a transformed tool direction, not robot-base +X.
    actions[:, :3] = [0.01, 0, 0]
    moved = decode_actions(anchor, actions)
    q = obs["state"]["right_ee_pose"][[4, 5, 6, 3]]
    direction = Rotation.from_quat(q).as_matrix() @ OPTICAL_FROM_FLU.T @ [0.01, 0, 0]
    for step in moved:
        np.testing.assert_allclose(step["right_ee_pose"][:3], anchor[:3] + direction, atol=1e-6)


def test_zero_pose_after_normalization_and_camera_mask():
    from openpi import transforms
    from openpi.shared.normalize import NormStats

    encoded = encode_observation(observation())
    raw = PassBallInputs()(deepcopy(encoded))
    assert not raw["image_mask"]["base_0_rgb"]
    assert raw["image_mask"]["left_wrist_0_rgb"]
    assert raw["image_mask"]["right_wrist_0_rgb"]
    assert not raw["image"]["base_0_rgb"].any()
    np.testing.assert_array_equal(raw["image"]["left_wrist_0_rgb"], encoded["images"]["left_wrist"])
    zero = EncodeState("zero_pose")(raw)
    stats = {
        "state": NormStats(mean=np.zeros(20), std=np.ones(20), q01=np.zeros(20), q99=np.ones(20))
    }
    normalized = transforms.Normalize(stats, use_quantiles=True)(zero)
    assert np.all(normalized["state"][POSE_INDICES] == -1)
    final = ZeroNormalizedPose()(normalized)
    assert np.all(final["state"][POSE_INDICES] == 0)
    np.testing.assert_allclose(final["state"][[9, 19]], normalized["state"][[9, 19]])
    np.testing.assert_allclose(encoded["state"][:3], [-0.4, 0.5, 0.6])


def test_model_batch_anchors_are_independent_and_reset_rejects_actions():
    model = PassBallZeroPoseModel.__new__(PassBallZeroPoseModel)
    model.num_steps = 10
    captured = []

    def infer(obs, **kwargs):
        captured.append(kwargs)
        return {"actions": neutral_actions(obs["state"])}

    model.policy = SimpleNamespace(infer=infer)
    model.reset()
    first, second = observation(7), observation(3)
    second["state"]["right_ee_pose"][0] += 1
    model.update_obs_batch([first, second])
    chunks = model.get_action_batch([3, 7])
    np.testing.assert_allclose(
        chunks[0][0]["right_ee_pose"][:3], second["state"]["right_ee_pose"][:3], atol=1e-6
    )
    np.testing.assert_allclose(
        chunks[1][0]["right_ee_pose"][:3], first["state"]["right_ee_pose"][:3], atol=1e-6
    )
    assert captured == [{"num_steps": 10}, {"num_steps": 10}]
    assert model.sampling_modes() == ["default"]
    assert not hasattr(model, "get_action_rtc")
    model.reset()
    with pytest.raises(ValueError, match="update_obs"):
        model.get_action()


def test_existing_model_allocation_and_explicit_profile_dispatch(monkeypatch):
    assert type(Model.__new__(Model)) is Model
    assert type(Model.__new__(Model, {"observation_profile": "yam_native"})) is Model
    sentinel = object()
    monkeypatch.setattr(
        "XPolicyLab.policy.Pi_05.pass_ball_model.PassBallZeroPoseModel", lambda cfg: sentinel
    )
    assert Model.__new__(Model, {"observation_profile": "tianji_taccap_pi05_zero_pose"}) is sentinel


@pytest.mark.parametrize("mutation", ["missing_pose", "invalid_gripper", "invalid_quaternion"])
def test_invalid_observations_fail(mutation):
    obs = observation()
    if mutation == "missing_pose":
        del obs["state"]["left_ee_pose"]
    elif mutation == "invalid_gripper":
        obs["state"]["left_ee_joint_state"] = [1.1]
    else:
        obs["state"]["left_ee_pose"][3:] = 0
    with pytest.raises((ValueError, KeyError)):
        encode_observation(obs)


def test_artifact_identity_rejects_wrong_step_and_normalization(tmp_path):
    import hashlib
    import json

    from XPolicyLab.policy.Pi_05.pass_ball_contract import validate_artifacts

    export = tmp_path / "test-zero-pose-export"
    checkpoint = export / "checkpoints/20000"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params/_METADATA").write_text("{}")
    stats_dir = export / "normalization"
    stats_dir.mkdir()
    stats = {
        name: {field: [0.0] * 20 for field in ("mean", "std", "q01", "q99")}
        for name in ("state", "actions")
    }
    path = stats_dir / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": stats}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = {
        "policy_name": "Pi_05",
        "protocol": "ws",
        "env_cfg_type": "tianji_dual",
        "action_type": "ee",
        "observation_profile": "tianji_taccap_pi05_zero_pose",
        "train_config_name": "pi05_pass_ball_hifi_umi_lora_zero_pose",
        "model_state_encoding": "zero_pose",
        "action_horizon": 32,
        "output_format": "xpolicylab",
        "action_semantics": "absolute_per_arm_base_xyz_wxyz",
        "model_path": str(checkpoint),
        "norm_stats_path": str(stats_dir),
        "checkpoint_num": 20000,
        "checkpoint_variant": "pi05_pass_ball_zero_pose_step20000",
        "checkpoint_source": export.name,
        "norm_stats_sha256": digest,
        "norm_stats_source": "sha256_" + digest,
        "num_steps": 10,
    }
    assert validate_artifacts(cfg)["contract_status"] == "ready"
    with pytest.raises(ValueError, match="step"):
        validate_artifacts({**cfg, "checkpoint_num": 59999})
    with pytest.raises(ValueError, match="source"):
        validate_artifacts({**cfg, "checkpoint_source": "previous-step-rel-export"})
    with pytest.raises(ValueError, match="checksum"):
        validate_artifacts({**cfg, "norm_stats_sha256": "invalid"})
