"""HiFi-UMI numeric layout and explicit rigid-pose coordinate conversions.

The source is left/right; the target is right/left, xyz(m), Rot6D(rows),
gripper(rad). No default guesses are made about source units or rotation axes.
"""

from __future__ import annotations

import numpy as np

# Column-vector coordinates: forward/left/up -> right/down/forward.
OPTICAL_FROM_FLU = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
SOURCE_NAMES = [
    name
    for side in ("left", "right")
    for name in [
        *[f"{side}_tcp.{axis}" for axis in ("x", "y", "z")],
        *[f"{side}_tcp.r{i}" for i in range(1, 7)],
        f"{side}_gripper.pos",
    ]
]
TARGET_NAMES = [
    name
    for side in ("right", "left")
    for name in [
        *[f"{side}_tcp.{axis}" for axis in ("x", "y", "z")],
        *[f"{side}_tcp.rot6d_{i}" for i in range(6)],
        f"{side}_gripper.angle_rad",
    ]
]
REFERENCE_URL = (
    "https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K/blob/"
    "c85a6daae096069ee49cf2181120cd64519c0ec0/chunk-0048/part-0000/meta/info.json"
)


def target_names(config: dict) -> list[str]:
    if config.get("target_gripper_unit", "rad") == "normalized_opening":
        return [name.replace(".angle_rad", ".opening") for name in TARGET_NAMES]
    return list(TARGET_NAMES)


def validate_convention(config: dict) -> None:
    """Validate all semantic choices before any converted dataset is written."""
    choices = {
        "source_rotation_layout": ("first_two_rows", "first_two_columns", "first_two_columns_interleaved"),
        "source_rotation_direction": ("local_to_reference", "reference_to_local"),
        "source_position_unit": ("m", "mm"),
        "source_state_semantics": ("absolute_pose",),
        "axis_change": ("reference", "tool", "both"),
    }
    missing = [key for key, allowed in choices.items() if config.get(key) not in allowed]
    target_unit = config.get("target_gripper_unit", "rad")
    if target_unit not in ("rad", "normalized_opening"):
        missing.append("target_gripper_unit")
    grippers = config.get("source_gripper") or {}
    for side in ("left", "right"):
        calibration = grippers.get(side) or {}
        if calibration.get("unit") == "normalized_opening":
            if target_unit == "normalized_opening":
                continue
            theta = calibration.get("theta_max_rad")
            if not isinstance(theta, (float, int)) or not np.isfinite(theta) or theta <= 0:
                missing.append(f"source_gripper.{side}.theta_max_rad (device EncoderMaxCal required)")
            continue
        if target_unit == "normalized_opening":
            missing.append(f"source_gripper.{side} must be normalized_opening for a normalized export")
            continue
        if calibration.get("unit") == "rad":
            continue
        if calibration.get("unit") != "calibrated":
            missing.append(f"source_gripper.{side} (rad or calibrated lookup table)")
            continue
        source = np.asarray(calibration.get("values", []), dtype=float)
        angles = np.asarray(calibration.get("angles_rad", []), dtype=float)
        if (
            source.ndim != 1
            or len(source) < 2
            or angles.shape != source.shape
            or not np.isfinite(source).all()
            or not np.isfinite(angles).all()
            or not (np.diff(source) > 0).all()
            or not ((np.diff(angles) > 0).all() or (np.diff(angles) < 0).all())
        ):
            missing.append(f"source_gripper.{side} (finite, strictly monotone calibration)")
    if missing:
        raise ValueError("Source conventions need confirmation: " + ", ".join(missing))


def rotation_from_6d(values: np.ndarray, layout: str, *, validate: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected finite Rot6D values")
    if layout in ("first_two_rows", "first_two_columns"):
        vectors = values.reshape(*values.shape[:-1], 2, 3)
    elif layout == "first_two_columns_interleaved":
        vectors = values.reshape(*values.shape[:-1], 3, 2).swapaxes(-1, -2)
    else:
        raise ValueError(f"Unknown Rot6D layout: {layout}")
    a, b = vectors[..., 0, :], vectors[..., 1, :]
    norm_a, norm_b = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    residual = np.maximum(np.maximum(abs(norm_a - 1), abs(norm_b - 1)), abs(np.sum(a * b, axis=-1)))
    if validate and np.any(residual > 5e-4):
        raise ValueError(f"Source Rot6D is not orthonormal for {layout}; maximum residual={residual.max()}")
    if np.any(norm_a < 1e-8):
        raise ValueError("Degenerate first Rot6D vector")
    a = a / norm_a[..., None]
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    length = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(length < 1e-8):
        raise ValueError("Collinear Rot6D vectors")
    b = b / length
    matrix = np.stack([a, b, np.cross(a, b)], axis=-2)
    return matrix if layout == "first_two_rows" else matrix.swapaxes(-1, -2)


def rotation_to_6d(matrix: np.ndarray, layout: str = "first_two_rows") -> np.ndarray:
    if layout == "first_two_rows":
        vectors = matrix[..., :2, :]
    elif layout == "first_two_columns":
        vectors = matrix[..., :, :2].swapaxes(-1, -2)
    elif layout == "first_two_columns_interleaved":
        vectors = matrix[..., :, :2]
    else:
        raise ValueError(f"Unknown Rot6D layout: {layout}")
    return vectors.reshape(*matrix.shape[:-2], 6)


def _gripper(values: np.ndarray, calibration: dict, inverse: bool, target_unit: str) -> np.ndarray:
    if calibration["unit"] == "normalized_opening":
        theta = calibration.get("theta_max_rad") if target_unit == "rad" else 1.0
        normalized = values / theta if inverse else values
        if np.any(normalized < -1e-7) or np.any(normalized > 1 + 1e-7):
            raise ValueError("Normalized gripper opening outside [0, 1]")
        return normalized if inverse else values * theta
    if calibration["unit"] == "rad":
        return values
    x = np.asarray(calibration["values"], dtype=float)
    y = np.asarray(calibration["angles_rad"], dtype=float)
    if inverse:
        x, y = y, x
        order = np.argsort(x)
        x, y = x[order], y[order]
    if np.any(values < x[0] - 1e-7) or np.any(values > x[-1] + 1e-7):
        raise ValueError("Gripper value outside supplied calibration; extrapolation is not defined")
    return np.interp(values, x, y)


def convert_state(values: np.ndarray, config: dict, *, inverse: bool = False) -> np.ndarray:
    """Convert absolute poses; preserve origins, never invent camera extrinsics.

    For R mapping tool to reference, reference re-expression gives C @ R;
    tool re-expression gives R @ C.T; changing both gives C @ R @ C.T.
    """
    validate_convention(config)
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 20 or not np.isfinite(values).all():
        raise ValueError("Expected finite [..., 20] state")
    result = np.empty_like(values)
    change = config["axis_change"]
    reference = OPTICAL_FROM_FLU if change in ("reference", "both") else np.eye(3)
    tool = OPTICAL_FROM_FLU if change in ("tool", "both") else np.eye(3)
    scale = 0.001 if config["source_position_unit"] == "mm" else 1.0
    for side, source_offset, target_offset in (("left", 0, 10), ("right", 10, 0)):
        src, dst = (target_offset, source_offset) if inverse else (source_offset, target_offset)
        block = values[..., src : src + 10]
        if inverse:
            result[..., dst : dst + 3] = (block[..., :3] @ reference) / scale
            rotation = rotation_from_6d(block[..., 3:9], "first_two_rows")
            rotation = reference.T @ rotation @ tool
            if config["source_rotation_direction"] == "reference_to_local":
                rotation = rotation.swapaxes(-1, -2)
            result[..., dst + 3 : dst + 9] = rotation_to_6d(rotation, config["source_rotation_layout"])
        else:
            result[..., dst : dst + 3] = (block[..., :3] * scale) @ reference.T
            rotation = rotation_from_6d(block[..., 3:9], config["source_rotation_layout"])
            if config["source_rotation_direction"] == "reference_to_local":
                rotation = rotation.swapaxes(-1, -2)
            result[..., dst + 3 : dst + 9] = rotation_to_6d(reference @ rotation @ tool.T)
        result[..., dst + 9] = _gripper(
            block[..., 9], config["source_gripper"][side], inverse, config.get("target_gripper_unit", "rad")
        )
    return result.astype(np.float32)


def next_state_targets(states: np.ndarray, episodes: list[dict]) -> np.ndarray:
    """HiFi-UMI stored actions: state[t+1], repeating the final state per episode."""
    targets = np.empty_like(states)
    covered = np.zeros(len(states), dtype=np.int8)
    for ep in episodes:
        lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
        if not 0 <= lo < hi <= len(states):
            raise ValueError("Invalid episode bounds")
        targets[lo : hi - 1] = states[lo + 1 : hi]
        targets[hi - 1] = states[hi - 1]
        covered[lo:hi] += 1
    if not np.all(covered == 1):
        raise ValueError("Episode ranges must cover every row exactly once")
    return targets


def relative_action_chunk(state: np.ndarray, actions: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Change absolute right/left 20-D target poses to/from the chunk anchor.

    Inputs use first-two-rows Rot6D, with local-to-reference rotations. Every
    action in the horizon uses the SAME current observation pose as its anchor.
    Gripper commands stay absolute. Source reference frames may differ per arm.
    The result uses the local axes already defined by the anchor state.
    """
    state, actions = np.asarray(state, dtype=np.float64), np.asarray(actions, dtype=np.float64)
    if (
        state.shape[-1] != 20
        or actions.shape[-1] != 20
        or actions.ndim != state.ndim + 1
        or actions.shape[:-2] != state.shape[:-1]
        or not np.isfinite(state).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Expected finite state [...,20] and actions [...,horizon,20]")
    result = np.array(actions, copy=True)
    for offset in (0, 10):
        anchor_p = state[..., None, offset : offset + 3]
        anchor_r = rotation_from_6d(state[..., offset + 3 : offset + 9], "first_two_rows")[..., None, :, :]
        action_p = actions[..., offset : offset + 3]
        action_r = rotation_from_6d(actions[..., offset + 3 : offset + 9], "first_two_rows", validate=not inverse)
        if inverse:
            # Predicted Rot6D is projected to SO(3) by rotation_from_6d.
            result[..., offset : offset + 3] = (anchor_r @ action_p[..., None])[..., 0] + anchor_p
            result[..., offset + 3 : offset + 9] = rotation_to_6d(anchor_r @ action_r)
        else:
            inverse_r = anchor_r.swapaxes(-1, -2)
            result[..., offset : offset + 3] = (inverse_r @ (action_p - anchor_p)[..., None])[..., 0]
            result[..., offset + 3 : offset + 9] = rotation_to_6d(inverse_r @ action_r)
    return result.astype(np.float32)


def pose_metadata(config: dict) -> dict:
    """Keep the requested optical-axis variant distinct from native HiFi-UMI."""
    validate_convention(config)
    return {
        "format": "hifi_umi_numeric_layout_with_custom_axes",
        "reference": REFERENCE_URL,
        "source_conventions": config,
        "order": ["right", "left"],
        "rotation_6d_layout": "first_two_rows",
        "rotation_direction": "local_to_reference",
        "position_unit": "m",
        "gripper_unit": config.get("target_gripper_unit", "rad"),
        "standard_hifi_umi_gripper_units": config.get("target_gripper_unit", "rad") == "rad",
        "source_axes": {"x": "forward", "y": "left", "z": "up"},
        "requested_axes": {"x": "right", "y": "down", "z": "forward"},
        "axis_change": config["axis_change"],
        "coordinate_matrix": OPTICAL_FROM_FLU.tolist(),
        "origin_policy": "preserve source origins; no translation or camera extrinsic calibration applied",
        "action_encoding": "absolute_next_state_target",
        "action_source": "observation.state[t+1], clamped within each episode",
        "recorded_action_policy": "original recorded action is not used to construct next-state targets",
        "media_policy": "source video keys and timestamps retained; no synthetic HiFi-UMI camera views",
    }

# Training submission source: hifi_umi.py; SHA256 920958dc8dbb9637c067634d5163f9011a94887075893dff9cade64c263413c5
