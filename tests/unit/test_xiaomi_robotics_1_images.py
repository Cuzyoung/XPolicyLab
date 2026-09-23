from __future__ import annotations

import numpy as np
import pytest

from XPolicyLab.policy.Xiaomi_Robotics_1.model import (
    Model,
    _add_xr1_to_path,
    _pure_black_ego_image,
)


def test_black_ego_defaults_to_left_wrist_shape() -> None:
    wrist = np.full((240, 320, 3), 127, dtype=np.uint8)

    ego = _pure_black_ego_image(wrist)

    assert ego.shape == wrist.shape
    assert ego.dtype == np.uint8
    assert not np.any(ego)


def test_black_ego_accepts_explicit_shape() -> None:
    wrist = np.ones((12, 16, 3), dtype=np.float32)

    ego = _pure_black_ego_image(wrist, (8, 10))

    assert ego.shape == (8, 10, 3)
    assert ego.dtype == np.uint8
    assert not np.any(ego)


@pytest.mark.parametrize("shape", ((0, 10), (8, 0), (-1, 10)))
def test_black_ego_rejects_non_positive_shape(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="black_ego_shape"):
        _pure_black_ego_image(np.zeros((12, 16, 3), dtype=np.uint8), shape)


def test_black_ego_mode_encodes_without_head_camera() -> None:
    _add_xr1_to_path()
    model = Model.__new__(Model)
    model.ego_view_mode = "black"
    model.black_ego_shape = None
    model.image_factor = 32
    model.image_max_pixels = 160000
    model.output_format = "packed_ee_delta"
    model.default_prompt = "Pass the ball."
    model._compose_state = lambda **_kwargs: np.zeros((1, 60), dtype=np.float32)
    wrist = np.full((240, 320, 3), 127, dtype=np.uint8)
    observation = {
        "vision": {
            "cam_left_wrist": {"color": wrist},
            "cam_right_wrist": {"color": wrist},
        },
        "state": {
            "left_arm_joint_state": np.zeros(7),
            "left_ee_joint_state": np.zeros(1),
            "right_arm_joint_state": np.zeros(7),
            "right_ee_joint_state": np.zeros(1),
        },
    }

    encoded = model._encode_observation(observation)

    ego = np.asarray(encoded["messages"][0]["content"][1]["image"])
    assert ego.shape == (256, 320, 3)
    assert not np.any(ego)
