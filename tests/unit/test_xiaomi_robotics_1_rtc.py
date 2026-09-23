from __future__ import annotations

import numpy as np
import pytest

from XPolicyLab.policy.Xiaomi_Robotics_1.model import (
    Model,
    _axis_angle_to_rotm,
    _ee_pose_sim_to_mibot,
    _rotm_to_axis_angle,
    _rotm_to_quat_wxyz,
)

HORIZON = 30
# A non-trivial EEF reframe checks that both directions share one frame.
REFRAME = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _model(output_format: str = "xpolicylab") -> Model:
    model = Model.__new__(Model)
    model.output_format = output_format
    model.action_shape = (HORIZON, 60)
    model.action_length = HORIZON
    model._eef_reframe_p = REFRAME
    model._eef_reframe_p_inv = REFRAME.T
    return model


def _pose(position, axis_angle) -> np.ndarray:
    quaternion = _rotm_to_quat_wxyz(_axis_angle_to_rotm(np.asarray(axis_angle)))
    return np.r_[position, quaternion]


def _current_state() -> dict:
    left = _ee_pose_sim_to_mibot([0.3, 0.2, 0.4], _pose([0, 0, 0], [0.2, -0.1, 0.4])[3:], REFRAME)
    right = _ee_pose_sim_to_mibot([0.3, -0.2, 0.4], _pose([0, 0, 0], [-0.3, 0.2, 0.1])[3:], REFRAME)
    return {
        "left_ee_pos_mibot": left[0],
        "left_ee_rotm_mibot": left[1],
        "left_gripper": 0.4,
        "right_ee_pos_mibot": right[0],
        "right_ee_rotm_mibot": right[1],
        "right_gripper": 0.6,
    }


def _condition() -> np.ndarray:
    steps = np.linspace(0.0, 1.0, HORIZON)[:, None]
    rows = []
    for step in steps[:, 0]:
        rows.append(
            np.r_[
                _pose([0.3 + 0.02 * step, 0.2, 0.4 - 0.01 * step], [0.2, -0.1 + 0.3 * step, 0.4]),
                0.4 + 0.3 * step,
                _pose([0.3, -0.2 - 0.02 * step, 0.4], [-0.3, 0.2, 0.1 + 2.5 * step]),
                0.6 - 0.3 * step,
            ]
        )
    return np.asarray(rows)


def _rotation(quaternion_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    angle = 2.0 * np.arctan2(np.linalg.norm([x, y, z]), w)
    axis = np.array([x, y, z])
    norm = np.linalg.norm(axis)
    return _axis_angle_to_rotm(axis / norm * angle if norm > 0 else np.zeros(3))


@pytest.mark.parametrize("angle", (0.0, 1e-9, 0.7, np.pi - 1e-6, np.pi))
def test_axis_angle_inverts_rotation_matrix(angle: float) -> None:
    axis_angle = np.array([0.6, -0.8, 0.0]) * angle
    rotation = _axis_angle_to_rotm(axis_angle)
    restored = _axis_angle_to_rotm(_rotm_to_axis_angle(rotation))
    np.testing.assert_allclose(restored, rotation, atol=1e-9)


def test_absolute_condition_round_trips_through_packed_deltas() -> None:
    model = _model()
    current = _current_state()
    condition = _condition()
    weights = np.ones(HORIZON)
    weights[-4:] = 0.0

    native = model._relative_condition(
        {"action_condition": condition, "condition_weights": weights}, current
    )

    assert native.shape == (HORIZON, 60) and native.dtype == np.float32
    np.testing.assert_array_equal(native[-4:], 0.0)
    np.testing.assert_array_equal(native[:, 7], 0.0)
    np.testing.assert_array_equal(native[:, 15:], 0.0)
    restored = model._actions_to_xpl_format(native[:-4].astype(np.float64), current)
    for row, action in zip(condition[:-4], restored, strict=True):
        for side, base in (("left", 0), ("right", 8)):
            pose = action[f"{side}_ee_pose"]
            np.testing.assert_allclose(pose[:3], row[base : base + 3], atol=1e-6)
            np.testing.assert_allclose(
                _rotation(pose[3:]), _rotation(row[base + 3 : base + 7]), atol=1e-5
            )
            np.testing.assert_allclose(
                action[f"{side}_ee_joint_state"], row[base + 7 : base + 8], atol=1e-6
            )


def test_absolute_condition_rejects_packed_width() -> None:
    model = _model()
    with pytest.raises(ValueError, match="absolute dual"):
        model._relative_condition(
            {"action_condition": np.zeros((HORIZON, 60)), "condition_weights": np.ones(HORIZON)},
            _current_state(),
        )


@pytest.mark.parametrize("output_format", ("xpolicylab", "packed_ee_delta"))
def test_rtc_condition_is_encoded_only_for_absolute_output(output_format: str) -> None:
    model = _model(output_format)
    current = _current_state()
    model._encoded_obs_list = [{"current_state": current}]
    received = {}

    def predict(encoded_obs_list, sampling):
        received.update(sampling)
        return ["chunk"]

    model._predict_action_chunks = predict
    condition = _condition() if output_format == "xpolicylab" else np.zeros((HORIZON, 60))
    sampling = {"action_condition": condition, "condition_weights": np.ones(HORIZON), "beta": 5.0}

    assert model.get_action_rtc(sampling) == "chunk"

    assert received["beta"] == 5.0
    if output_format == "packed_ee_delta":
        assert received["action_condition"] is condition
    else:
        assert received["action_condition"].shape == (HORIZON, 60)
        np.testing.assert_allclose(
            received["action_condition"],
            model._relative_condition(sampling, current),
        )
