"""Inter-hand state must not replace per-arm action anchors or RTC targets."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from XPolicyLab.policy.Pi_05.pack_plate_model import (
    PackPlateZeroPoseModel,
    encode_rtc_condition,
    make_pack_plate_train_config,
)
from XPolicyLab.policy.Pi_05.pack_plate_state import InterHandRelativeState
from XPolicyLab.policy.Pi_05.pass_ball_model import PassBallInputs, decode_actions, encode_observation
from XPolicyLab.policy.Pi_05.pass_ball_pose import OPTICAL_FROM_FLU, rotation_from_6d
from XPolicyLab.policy.Pi_05.pass_ball_state import POSE_INDICES, ZeroNormalizedPose

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "manimux/configs/policy/pi05/tianji/pack_plate"


def recipe(step=20000):
    return yaml.safe_load((RECIPES / f"inter-hand-relative-step{step}.yaml").read_text())


def observation():
    obs = {"instruction": "plate", "state": {}, "vision": {}}
    for side, xyz, rotvec, opening in (
        ("right", [0.4, -0.2, 0.3], [0.2, -0.4, 0.6], 0.3),
        ("left", [0.3, 0.1, 0.5], [-0.3, 0.5, 0.1], 0.7),
    ):
        q = Rotation.from_rotvec(rotvec).as_quat()
        obs["state"][f"{side}_ee_pose"] = np.r_[xyz, q[[3, 0, 1, 2]]]
        obs["state"][f"{side}_ee_joint_state"] = np.array([opening])
        obs["vision"][f"cam_{side}_wrist"] = {"color": np.full((8, 8, 3), 60, np.uint8)}
    return obs


def matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def feature_matrix(state, offset):
    result = np.eye(4)
    result[:3, :3] = rotation_from_6d(state[offset+3:offset+9], "first_two_rows")
    result[:3, 3] = state[offset:offset+3]
    return result


def test_inter_hand_matches_independent_homogeneous_transform_and_preserves_anchor():
    obs = observation()
    encoded = encode_observation(obs)
    original = encoded["state"].copy()
    bases = recipe()["state_base_to_common"]
    converted = InterHandRelativeState(bases)(encoded)["state"]
    tool = np.eye(4)
    tool[:3, :3] = OPTICAL_FROM_FLU.T
    common = {side: np.array(bases[side]) @ matrix(obs["state"][f"{side}_ee_pose"]) @ tool
              for side in ("left", "right")}
    right = feature_matrix(converted, 0)
    left = feature_matrix(converted, 10)
    np.testing.assert_allclose(right, np.linalg.inv(common["left"]) @ common["right"], atol=2e-7)
    np.testing.assert_allclose(left, np.linalg.inv(common["right"]) @ common["left"], atol=2e-7)
    np.testing.assert_allclose(left @ right, np.eye(4), atol=2e-7)
    np.testing.assert_array_equal(encoded["state"], original)
    np.testing.assert_array_equal(converted[[9, 19]], original[[9, 19]])


def test_common_world_choice_cancels_out():
    data = encode_observation(observation())
    bases = recipe()["state_base_to_common"]
    world = np.eye(4)
    world[:3, :3] = Rotation.from_rotvec([0.8, -0.1, 0.3]).as_matrix()
    world[:3, 3] = [1.2, -3.4, 2.1]
    moved = {side: world @ value for side, value in bases.items()}
    np.testing.assert_allclose(InterHandRelativeState(bases)(data)["state"],
                               InterHandRelativeState(moved)(data)["state"], atol=2e-7)


@pytest.mark.parametrize("bad", ["missing", "reflection", "translation_row", "nan"])
def test_invalid_extrinsics_fail_at_configuration_boundary(bad):
    bases = recipe()["state_base_to_common"]
    if bad == "missing":
        bases.pop("right")
    elif bad == "reflection":
        bases["left"][0][0] = -1
    elif bad == "translation_row":
        bases["left"][3][0] = 0.1
    else:
        bases["left"][0][3] = float("nan")
    with pytest.raises(ValueError, match="state_base_to_common"):
        InterHandRelativeState(bases)


def test_rtc_targets_and_decoded_actions_do_not_use_inter_hand_anchor():
    obs = observation()
    encoded = encode_observation(obs)
    anchor = encoded["state"].copy()
    row = np.concatenate([np.r_[obs["state"][f"{side}_ee_pose"], obs["state"][f"{side}_ee_joint_state"]]
                          for side in ("left", "right")])
    targets = np.tile(row, (32, 1))
    targets[:, :3] += [0.01, -0.03, 0.02]
    targets[:, 8:11] += [-0.02, 0.02, 0.04]
    weights = np.r_[np.ones(8), np.zeros(24)]
    absolute = encode_rtc_condition(targets, weights, anchor)
    relative = PassBallInputs()({**encoded, "actions": absolute})
    transformed = InterHandRelativeState(recipe()["state_base_to_common"])(relative)
    np.testing.assert_array_equal(relative["actions"], transformed["actions"])
    np.testing.assert_array_equal(encoded["state"], anchor)
    output = decode_actions(anchor, transformed["actions"])
    for i in range(32):
        for side, offset in (("left", 0), ("right", 8)):
            expected = targets[i, offset:offset+7] if i < 8 else row[offset:offset+7]
            np.testing.assert_allclose(matrix(output[i][f"{side}_ee_pose"]), matrix(expected), atol=2e-7)


@pytest.mark.parametrize("step", [20000, 59999])
def test_real_openpi_transform_chain_retains_nonzero_normalized_pose(step):
    pytest.importorskip("openpi")
    from openpi import transforms
    from openpi.shared import normalize

    cfg = recipe(step)
    train = make_pack_plate_train_config(cfg)
    data = train.data.create(train.assets_dirs, train.model)
    assert train.model.paligemma_variant == "gemma_2b_lora"
    assert train.model.action_expert_variant == "gemma_300m_lora"
    assert not any(isinstance(t, ZeroNormalizedPose) for t in data.model_transforms.inputs)
    assert train.policy_metadata["model_state_encoding"] == "inter_hand_relative"
    stats = normalize.load(Path('/home/jw/Desktop/policy') / cfg["norm_stats_path"])
    encoded = encode_observation(observation())
    processed = transforms.compose(data.data_transforms.inputs)(deepcopy(encoded))
    normalized = transforms.Normalize(stats, use_quantiles=data.use_quantile_norm)(processed)
    expected = normalized["state"].copy()
    final = transforms.compose(data.model_transforms.inputs)(normalized)
    np.testing.assert_allclose(final["state"][:20], expected)
    assert np.count_nonzero(final["state"][POSE_INDICES]) > 10
    assert final["state"].shape == (32,)
    assert final["tokenized_prompt"].shape == (200,)
    assert not final["image_mask"]["base_0_rgb"]
    old = make_pack_plate_train_config({"repo_id": "old", "train_config_name": "old"})
    old_data = old.data.create(old.assets_dirs, old.model)
    assert isinstance(old_data.model_transforms.inputs[0], ZeroNormalizedPose)


def test_runtime_metadata_reports_the_exact_extrinsics():
    cfg = recipe()
    model = PackPlateZeroPoseModel.__new__(PackPlateZeroPoseModel)
    model.model_cfg = cfg
    model.model_root = Path(cfg["model_path"])
    model.norm_stats_path = Path(cfg["norm_stats_path"])
    model.num_steps = cfg["num_steps"]
    assert model.runtime_metadata()["state_base_to_common"] == cfg["state_base_to_common"]
