from __future__ import annotations

import numpy as np

from XPolicyLab.policy.Pi_05.model import encode_obs, slice_stacked_obs, stack_obs

UMI_DIM_INFO = {"arm_dim": [9, 9], "ee_dim": [1, 1]}


def _raw_observation(fill: float) -> dict:
    return {
        "state": {
            "left_ee_pose": np.full(9, fill, dtype=np.float32),
            "left_ee_joint_state": np.full(1, fill, dtype=np.float32),
            "right_ee_pose": np.full(9, fill, dtype=np.float32),
            "right_ee_joint_state": np.full(1, fill, dtype=np.float32),
        },
        "images": {
            "left_camera": np.zeros((4, 6, 3), dtype=np.uint8),
            "right_camera": np.ones((4, 6, 3), dtype=np.uint8),
        },
        "instruction": "exchange ball",
    }


def test_pi05_umi_encode_packs_tcp_state_and_wrist_cameras_only() -> None:
    encoded = encode_obs(
        _raw_observation(0.5), "ee", UMI_DIM_INFO, observation_profile="umi_native"
    )

    assert set(encoded) == {
        "observation/left_wrist",
        "observation/right_wrist",
        "observation/state",
        "prompt",
    }
    assert encoded["observation/state"].shape == (20,)
    # Raw HWC frames: UmiInputs owns the resize and the base-camera padding.
    assert encoded["observation/left_wrist"].shape == (4, 6, 3)
    assert encoded["prompt"] == "exchange ball"


def test_pi05_umi_stacked_observations_round_trip_without_a_base_camera() -> None:
    encoded = [
        encode_obs(
            _raw_observation(fill), "ee", UMI_DIM_INFO, observation_profile="umi_native"
        )
        for fill in (0.25, 0.75)
    ]

    stacked = stack_obs(encoded)
    assert stacked["observation/state"].shape == (2, 20)
    assert stacked["prompt"] == ["exchange ball", "exchange ball"]

    single = slice_stacked_obs(stacked, 1)
    assert single["prompt"] == "exchange ball"
    np.testing.assert_array_equal(
        single["observation/state"], encoded[1]["observation/state"]
    )
