"""Joint/EEF training and deployment share the same YAM action contract."""

import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies import yam_policy
from openpi.training import config


def test_joint_ee_config_is_registered():
    cfg = config.get_config("pi05_yam_joint_ee")
    assert isinstance(cfg.data, config.LeRobotYamJointEeDataConfig)
    assert cfg.model.action_horizon == 50
    assert cfg.model.action_dim == 32
    assert isinstance(config.get_config("pi05_yam").data, config.LeRobotYamDataConfig)


def test_joint_ee_targets_match_stats_and_deploy_as_joint_only():
    state = np.arange(14, dtype=np.float32) / 20
    actions = np.repeat(state[None, :], 50, axis=0)
    actions[:, :6] += .25
    actions[:, 7:13] -= .5
    actions[:, [6, 13]] = [.2, .8]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    pose = np.tile(np.r_[np.zeros(3), rotation.ravel()], 2).astype(np.float32)
    target = np.repeat(pose[None, :], 50, axis=0)
    target[:, 0] = .1
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    data = {"observation/state": state, "observation/image": rgb,
            "observation/left_wrist": rgb, "observation/right_wrist": rgb,
            "observation/ee_pose": pose, "action_ee_pose": target, "actions": actions}
    packed = yam_policy.YamJointEeInputs(model.ModelType.PI05)(data)
    mask = transforms.make_bool_mask(6, -1, 6, -1)
    packed = transforms.DeltaActions(mask)(packed)
    assert packed["actions"].shape == (50, 26)
    np.testing.assert_allclose(packed["actions"][:, :6], .25, atol=1e-7)
    np.testing.assert_allclose(packed["actions"][:, 7:13], -.5, atol=1e-7)
    np.testing.assert_allclose(packed["actions"][:, [6, 13]], np.tile([.2, .8], (50, 1)))
    np.testing.assert_allclose(packed["actions"][:, 14:17], np.tile([0, -.1, 0], (50, 1)))
    np.testing.assert_allclose(packed["actions"][:, 17:], 0, atol=1e-7)
    absolute = transforms.AbsoluteActions(mask)(packed)
    deployed = yam_policy.YamOutputs()(absolute)["actions"]
    assert deployed.shape == (50, 14)
    np.testing.assert_allclose(deployed, actions, atol=1e-7)
    # Runtime inference only needs joint state and RGB, not recorded EE labels.
    observation = {k: v for k, v in data.items()
                   if k not in ("actions", "action_ee_pose", "observation/ee_pose")}
    assert "actions" not in yam_policy.YamJointEeInputs(model.ModelType.PI05)(observation)


def test_joint_ee_rejects_half_present_auxiliary_labels():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    data = {"observation/state": np.zeros(14), "observation/image": rgb,
            "observation/left_wrist": rgb, "observation/right_wrist": rgb,
            "observation/ee_pose": np.zeros(24)}
    with pytest.raises(ValueError, match="both"):
        yam_policy.YamJointEeInputs(model.ModelType.PI05)(data)
