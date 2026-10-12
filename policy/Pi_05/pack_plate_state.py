"""Inter-hand model state from Tianji TCP poses expressed in separate arm bases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pass_ball_pose import rotation_from_6d, rotation_to_6d
from .pass_ball_state import encode_state


@dataclass(frozen=True)
class InterHandRelativeState:
    """Apply common<-arm-base extrinsics only to the model state branch.

    Input TCP rotations already use the training tool axes. Relative action
    targets must be computed before this transform, from the per-arm anchors.
    """

    base_to_common: dict[str, np.ndarray]

    def __post_init__(self):
        if set(self.base_to_common) != {"left", "right"}:
            raise ValueError("state_base_to_common requires left and right transforms")
        matrices = {}
        for side, value in self.base_to_common.items():
            matrix = np.array(value, dtype=np.float64, copy=True)
            if (
                matrix.shape != (4, 4)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-7, rtol=0)
            ):
                raise ValueError(
                    f"state_base_to_common.{side} must be a finite rigid 4x4 transform"
                )
            matrix.setflags(write=False)
            matrices[side] = matrix
        object.__setattr__(self, "base_to_common", matrices)

    def __call__(self, data: dict) -> dict:
        state = np.array(data["state"], dtype=np.float64, copy=True)
        for side, offset in (("right", 0), ("left", 10)):
            base = self.base_to_common[side]
            rotation = rotation_from_6d(state[offset + 3 : offset + 9], "first_two_rows")
            state[offset : offset + 3] = base[:3, :3] @ state[offset : offset + 3] + base[:3, 3]
            state[offset + 3 : offset + 9] = rotation_to_6d(base[:3, :3] @ rotation)
        return {**data, "state": encode_state(state, "inter_hand_relative")}
