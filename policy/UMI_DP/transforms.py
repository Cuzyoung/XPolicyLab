"""UMI checkpoint transforms. Input images are already decoded RGB arrays."""

import math

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def pose_matrix(value):
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite [xyz, quaternion wxyz]")
    if not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-4):
        raise ValueError("Quaternion must have unit length")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_pose(matrix):
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], quat[[3, 0, 1, 2]]]


def pose9(matrix):
    value = np.asarray(matrix)
    return np.concatenate(
        (value[..., :3, 3], value[..., :2, :3].reshape(value.shape[:-2] + (6,))), axis=-1
    ).astype(np.float32)


def from_pose9(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 9 or not np.isfinite(value).all():
        raise ValueError("Invalid pose9")
    a, b = value[..., 3:6], value[..., 6:9]
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(norm < 1e-7):
        raise ValueError("Degenerate first rotation row")
    a = a / norm
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    norm = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(norm < 1e-7):
        raise ValueError("Collinear rotation rows")
    b = b / norm
    matrix = np.broadcast_to(np.eye(4), value.shape[:-1] + (4, 4)).copy()
    matrix[..., :3, :3] = np.stack((a, b, np.cross(a, b)), axis=-2)
    matrix[..., :3, 3] = value[..., :3]
    return matrix


def preprocess_rgb(rgb, shape):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected decoded RGB uint8 HWC")
    _, height, width = shape
    ih, iw = rgb.shape[:2]
    scale = max(width / iw, height / ih)
    rw, rh = math.ceil(iw * scale), math.ceil(ih * scale)
    image = Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR)
    left, top = (rw - width) // 2, (rh - height) // 2
    return (
        np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.float32).transpose(
            2, 0, 1
        )
        / 255
    )


def build_observation(obs, shape, period_s, tolerance_s):
    info = obs["additional_info"]["umi_dp"]
    timestamps = np.asarray(info["frame_times_ns"], dtype=np.int64)
    if timestamps.shape != (2,) or timestamps[1] <= timestamps[0]:
        raise ValueError("Two distinct measured snapshots are required")
    if abs((timestamps[1] - timestamps[0]) / 1e9 - period_s) > tolerance_s:
        raise ValueError("Observation interval differs from the checkpoint period")
    tensors, references = {}, []
    for index, side in enumerate(("left", "right")):
        matrices = np.stack(
            [pose_matrix(obs["state"][f"{side}_ee_pose{suffix}"]) for suffix in ("_prev", "")]
        )
        references.append(matrices[-1])
        relative = pose9(np.linalg.inv(matrices[-1]) @ matrices)
        tensors[f"robot{index}_eef_pos"] = relative[:, :3]
        tensors[f"robot{index}_eef_rot_axis_angle"] = relative[:, 3:]
        grip = np.asarray(
            [obs["state"][f"{side}_ee_joint_state{suffix}"] for suffix in ("_prev", "")],
            dtype=np.float32,
        )
        if grip.shape != (2, 1) or not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
            raise ValueError("UMI aperture observations must lie in [0, 1]")
        tensors[f"robot{index}_gripper_width"] = grip
        key = f"camera{index}_rgb"
        tensors[key] = np.stack(
            [
                preprocess_rgb(
                    obs["vision"][f"cam_{side}_wrist{suffix}"]["color"], shape["obs"][key]["shape"]
                )
                for suffix in ("_prev", "")
            ]
        )
    for key, value in tensors.items():
        if value.shape != (2, *shape["obs"][key]["shape"]) or not np.isfinite(value).all():
            raise ValueError(f"Invalid observation {key}: {value.shape}")
    return tensors, np.stack(references)


def absolute_actions(action, reference):
    result = []
    for row in action:
        step = {}
        for index, side in enumerate(("left", "right")):
            base = index * 10
            matrix = reference[index] @ from_pose9(row[base : base + 9])
            step[f"{side}_ee_pose"] = matrix_pose(matrix)
            # Keep raw predictions. The embodiment rejects invalid apertures.
            step[f"{side}_ee_joint_state"] = np.asarray([row[base + 9]], dtype=np.float32)
        result.append(step)
    return result


def relative_condition(condition, reference, weights):
    condition = np.asarray(condition, dtype=np.float64)
    if condition.shape != (len(weights), 16) or not np.isfinite(condition).all():
        raise ValueError("RTC expects absolute dual TCP/aperture rows with width 16")
    result = np.zeros((len(condition), 20), dtype=np.float32)
    for row, weight in enumerate(weights):
        for index in range(2):
            target = (
                reference[index]
                if weight == 0
                else pose_matrix(condition[row, index * 8 : index * 8 + 7])
            )
            result[row, index * 10 : index * 10 + 9] = pose9(
                np.linalg.inv(reference[index]) @ target
            )
            result[row, index * 10 + 9] = condition[row, index * 8 + 7]
    return result
