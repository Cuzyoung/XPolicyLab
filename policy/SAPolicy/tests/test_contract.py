from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from XPolicyLab.policy.SAPolicy.model import Model


def observation(value=0.1, index=0):
    return {
        "env_idx": index,
        "vision": {
            name: {
                "color": np.full((8, 12, 3), [10, 30, 200], dtype=np.uint8),
                "intrinsic_matrix": np.eye(3),
                "shape": [8, 12],
            }
            for name in ("top", "left", "right")
        },
        "state": {
            f"{side}_{key}": value_
            for side in ("left", "right")
            for key, value_ in [
                ("ee_pose", [value, 0, 0.5, 1, 0, 0, 0]),
                ("ee_joint_state", [value]),
            ]
        },
    }


def model(**kwargs):
    return Model(
        {"dry_run": True, "action_horizon": 3, "camera_names": ["top", "left", "right"], **kwargs}
    )


def test_standard_and_legacy_actions_preserve_pose_and_aperture():
    standard, legacy = model(), model(output_format="packed_ee_wire")
    obs = observation()
    for policy in (standard, legacy):
        policy.update_obs(obs)
    actions, wire = standard.get_action(), legacy.get_action()
    assert len(actions) == 3
    for side, offset in (("left", 0), ("right", 8)):
        assert set(actions[0]) == {
            f"{s}_{k}" for s in ("left", "right") for k in ("ee_pose", "ee_joint_state")
        }
        np.testing.assert_array_equal(actions[0][f"{side}_ee_pose"], [0.1, 0, 0.5, 1, 0, 0, 0])
        np.testing.assert_array_equal(wire[0, offset : offset + 8], [0.1, 0, 0.5, 0, 0, 0, 1, 0.1])


def test_batch_subset_order_and_reset():
    policy = model()
    policy.update_obs_batch([observation(0.2, 4), observation(0.7, 9)])
    actions = policy.get_action_batch([9, 4])
    assert [a[0]["left_ee_joint_state"][0] for a in actions] == [0.7, 0.2]
    policy.update_obs_batch([observation(0.3, 4)])
    assert policy.get_action_batch([9])[0][0]["left_ee_joint_state"][0] == 0.7
    policy.reset()
    with pytest.raises(RuntimeError):
        policy.get_action_batch([4])
    with pytest.raises(RuntimeError):
        policy.get_action()


def test_histories_survive_rpc_thread_changes_and_remain_isolated():
    policy = model(output_format="packed_ee_wire")

    class Backend:
        def __init__(self):
            self.frames = deque(maxlen=2)
            self.seen = []

        def reset_model(self):
            self.frames.clear()

        def update_obs(self, obs):
            self.frames.append(obs)

        def get_action(self):
            self.seen.append([f["left_gripper"] for f in self.frames])
            return np.tile([0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0, 0, 1, 0, 0, 0, 0.5], (3, 1))

    policy._backend = Backend()
    policy._dry_run = False
    policy._history_length = 2
    with ThreadPoolExecutor(1) as pool:
        pool.submit(policy.update_obs_batch, [observation(0.2, 4), observation(0.8, 9)]).result()
        pool.submit(policy.update_obs_batch, [observation(0.3, 4)]).result()
    policy.get_action_batch([9, 4])
    assert policy._backend.seen == [[0.8, 0.8], [0.2, 0.3]]


def test_rgb_passthrough_and_state_layout():
    policy = model()
    obs = observation()
    spatial = policy._to_spatial_obs(obs)
    np.testing.assert_array_equal(spatial["images"]["top"][0, 0], [10, 30, 200])
    state = policy._pack_state([1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1], 0.2, 0.8)
    np.testing.assert_array_equal(state[[0, 1, 2, 9, 10, 11, 12, 19]], [1, 2, 3, 0.2, 4, 5, 6, 0.8])


@pytest.mark.parametrize(
    "kwargs",
    [{"action_type": "joint"}, {"env_cfg_type": "droid_single"}, {"output_format": "unknown"}],
)
def test_unsupported_contract_fails(kwargs):
    with pytest.raises(ValueError):
        model(**kwargs)
