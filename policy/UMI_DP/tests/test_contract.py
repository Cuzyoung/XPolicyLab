"""Model transforms and lifecycle checks; real-weight forwards use validate.py."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.UMI_DP.artifact_identity import NAMESPACE, remap_targets
from XPolicyLab.policy.UMI_DP.debug_client import synthetic_observation
from XPolicyLab.policy.UMI_DP.model import Model
from XPolicyLab.policy.UMI_DP.transforms import (
    absolute_actions,
    build_observation,
    from_pose9,
    pose9,
    relative_condition,
)
from XPolicyLab.utils.process_data import (
    decode_obs_images,
    get_robot_action_dim_info,
    images_encoding,
)


def shape_meta(horizon=16):
    obs = {}
    for arm in range(2):
        obs[f"camera{arm}_rgb"] = {"shape": [3, 224, 224]}
        for key, width in (("eef_pos", 3), ("eef_rot_axis_angle", 6), ("gripper_width", 1)):
            obs[f"robot{arm}_{key}"] = {"shape": [width]}
    return {"obs": obs, "action": {"horizon": horizon, "shape": [20]}}


def test_row_rotation_relative_absolute_roundtrip():
    reference = np.tile(np.eye(4), (2, 1, 1))
    reference[0, :3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_matrix()
    reference[0, :3, 3] = [0.2, 0.1, 0.4]
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("xyz", [-0.5, 0.2, 0.1]).as_matrix()
    target[:3, 3] = [0.03, -0.02, 0.01]
    raw = np.tile(np.r_[pose9(target), 0.3, pose9(target), 0.8], (64, 1))
    steps = absolute_actions(raw, reference)
    absolute = np.asarray(
        [
            np.r_[
                step["left_ee_pose"],
                step["left_ee_joint_state"],
                step["right_ee_pose"],
                step["right_ee_joint_state"],
            ]
            for step in steps
        ]
    )
    actual = relative_condition(absolute, reference, np.ones(64))
    np.testing.assert_allclose(actual, raw, atol=2e-7)
    np.testing.assert_allclose(from_pose9(pose9(target)), target, atol=1e-7)


def test_encoded_rgb_shared_decode_matches_raw():
    obs = synthetic_observation()
    # Uniform distinct channels survive JPEG with small quantization error.
    for camera in obs["vision"].values():
        camera["color"][:] = [231, 42, 17]
    encoded = copy.deepcopy(obs)
    for camera in encoded["vision"].values():
        camera["color"] = images_encoding([camera["color"]])[0][0]
    decode_obs_images(encoded)
    raw_tensors, _ = build_observation(obs, shape_meta(), 0.1, 0.04)
    decoded, _ = build_observation(encoded, shape_meta(), 0.1, 0.04)
    for name in ("camera0_rgb", "camera1_rgb"):
        np.testing.assert_allclose(decoded[name], raw_tensors[name], atol=2 / 255)
        assert decoded[name][0, 0].mean() > decoded[name][0, 2].mean()


@pytest.mark.parametrize("interval", [0, 10000000, 250000000])
def test_reject_incorrect_observation_intervals(interval):
    obs = synthetic_observation()
    obs["additional_info"]["umi_dp"]["frame_times_ns"] = [10**9, 10**9 + interval]
    with pytest.raises(ValueError):
        build_observation(obs, shape_meta(), 0.1, 0.04)


def test_batch_ids_reset_and_failed_update_do_not_replace_valid_state():
    model = Model.__new__(Model)
    model.shape_meta = shape_meta()
    model._metadata = {"observation_period_s": 0.1}
    model.tolerance_s = 0.04
    model.reset()
    obs = synthetic_observation()
    obs["env_idx"] = 7
    model.update_obs(obs)
    with pytest.raises(ValueError, match="Duplicate"):
        model.update_obs_batch([obs, obs])
    assert model._order == [7]
    model.reset()
    assert model._observations == {} and model._order == []
    with pytest.raises(ValueError, match="exactly one"):
        model.get_action()


def test_namespace_and_robot_dimension_registration():
    assert remap_targets({"_target_": "diffusion_policy.policy.X", "other": "no-change"}) == {
        "_target_": NAMESPACE + "policy.X",
        "other": "no-change",
    }
    for name in ("tianji_umi", "tianji_dual"):
        assert get_robot_action_dim_info(name) == {"arm_dim": [7, 7], "ee_dim": [1, 1]}
