"""Model-only state ablations; absolute TCP anchors remain outside the model.

All representations retain right(10), left(10), xyz(3), Rot6D rows(6), gripper(1).
Inter-hand transforms require the two source poses to share a reference frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pass_ball_pose import rotation_from_6d, rotation_to_6d

STATE_ENCODINGS = ("absolute", "zero_pose", "inter_hand_relative")
POSE_INDICES = np.array([*range(9), *range(10, 19)])


def encode_state(state: np.ndarray, encoding: str) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] != 20 or not np.isfinite(state).all():
        raise ValueError("Expected finite state [..., 20]")
    if encoding not in STATE_ENCODINGS:
        raise ValueError(f"Unsupported model state encoding: {encoding}")
    result = state.copy()
    if encoding == "zero_pose":
        result[..., POSE_INDICES] = 0
    if encoding == "inter_hand_relative":
        for own, other in ((0, 10), (10, 0)):
            own_r = rotation_from_6d(state[..., own + 3 : own + 9], "first_two_rows")
            other_r = rotation_from_6d(state[..., other + 3 : other + 9], "first_two_rows")
            inverse_other = other_r.swapaxes(-1, -2)
            delta = state[..., own : own + 3] - state[..., other : other + 3]
            result[..., own : own + 3] = (inverse_other @ delta[..., None])[..., 0]
            result[..., own + 3 : own + 9] = rotation_to_6d(inverse_other @ own_r)
    return result.astype(np.float32)


def model_state_names(absolute_names: list[str], encoding: str) -> list[str]:
    if encoding == "absolute":
        return list(absolute_names)
    if encoding == "zero_pose":
        return [name if i in (9, 19) else f"zero.{name}" for i, name in enumerate(absolute_names)]
    if encoding != "inter_hand_relative":
        raise ValueError(encoding)
    return [
        name
        for own, other in (("right", "left"), ("left", "right"))
        for name in (
            *[f"{own}_tcp_wrt_{other}.{axis}" for axis in ("x", "y", "z")],
            *[f"{own}_tcp_wrt_{other}.rot6d_{i}" for i in range(6)],
            f"{own}_gripper.opening",
        )
    ]


@dataclass(frozen=True)
class EncodeState:
    encoding: str

    def __call__(self, data: dict) -> dict:
        # Relative action targets must already have been computed from the original state.
        return {**data, "state": encode_state(data["state"], self.encoding)}


@dataclass(frozen=True)
class ZeroNormalizedPose:
    """Guarantee zero pose slots AFTER normalization, BEFORE tokenization.

    A constant-zero empirical distribution otherwise normalizes to -1 in OpenPI.
    Gripper slots retain ordinary normalization. This is not a token-attention mask.
    """

    def __call__(self, data: dict) -> dict:
        state = np.array(data["state"], dtype=np.float32, copy=True)
        state[..., POSE_INDICES] = 0
        return {**data, "state": state}

# Training submission source: state_encoding.py; SHA256 04b7be435b3064d511e6fe34f8da62b15f9972ef5521a551b474ee67f9bf6013
