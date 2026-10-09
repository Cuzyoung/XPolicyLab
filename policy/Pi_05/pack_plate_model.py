"""Pack-plate Pi05 inference using its own checkpoint and normalization.

The exported contract is two wrist images, zero pose / live opening state,
and one current-TCP-relative 32-step action chunk. Numerical pose helpers are
shared with the pass-ball profile; artifact identity and task are separate.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .pack_plate_contract import PROFILE, validate_artifacts
from .pass_ball_model import PassBallZeroPoseModel, decode_actions, make_train_config
from .pass_ball_pose import OPTICAL_FROM_FLU, rotation_to_6d


def make_pack_plate_train_config(model_cfg: dict):
    """Reuse zero-pose transforms with the selected checkpoint's model architecture."""
    base = make_train_config(model_cfg)
    return dataclasses.replace(
        base,
        name=str(model_cfg["train_config_name"]),
        model=dataclasses.replace(
            base.model,
            paligemma_variant=model_cfg.get("paligemma_variant", "gemma_2b_lora"),
            action_expert_variant=model_cfg.get("action_expert_variant", "gemma_300m_lora"),
        ),
    )


def encode_rtc_condition(
    condition: np.ndarray, weights: np.ndarray, anchor: np.ndarray
) -> np.ndarray:
    """Convert left/right absolute TCP rows to native right/left 20-D actions."""
    condition = np.asarray(condition, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if condition.shape != (32, 16) or weights.shape != (32,) or anchor.shape != (20,):
        raise ValueError("pack-plate RTC requires (32,16) conditions and a 20-D anchor")
    if (
        not np.isfinite(condition).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(anchor).all()
        or np.any((weights < 0) | (weights > 1))
    ):
        raise ValueError("pack-plate RTC condition, weights and anchor must be finite")

    # Zero-weight rows still pass through OpenPI's rigid-pose transform; use the
    # observation anchor instead of the wire format's all-zero padding there.
    native = np.repeat(anchor[None, :], 32, axis=0)
    active = np.flatnonzero(weights > 0)
    if not len(active):
        return native.astype(np.float32)
    for source, target in ((0, 10), (8, 0)):
        pose = condition[active, source : source + 7]
        opening = condition[active, source + 7]
        quaternions = pose[:, 3:7]
        if (
            np.any((opening < 0) | (opening > 1))
            or np.any(np.abs(np.linalg.norm(quaternions, axis=1) - 1) > 1e-4)
        ):
            raise ValueError("pack-plate RTC requires unit TCP quaternions and [0,1] openings")
        rotation = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
        native[active, target : target + 3] = pose[:, :3]
        native[active, target + 3 : target + 9] = rotation_to_6d(
            rotation @ OPTICAL_FROM_FLU.T
        )
        native[active, target + 9] = opening
    return native.astype(np.float32)


class PackPlateZeroPoseModel(PassBallZeroPoseModel):
    """Separate deployment profile; existing pass-ball allocation is untouched."""

    def __init__(self, model_cfg: dict):
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared import normalize

        from .model import _resolve_pi05_model_root

        validate_artifacts(model_cfg)
        if model_cfg["observation_profile"] != PROFILE:
            raise ValueError("pack-plate deployment profile mismatch")
        dims = get_robot_action_dim_info(model_cfg)
        if dims["arm_dim"] != [7, 7] or dims["ee_dim"] != [1, 1]:
            raise ValueError("pack-plate requires Tianji dual 7+1 dimensions")
        self.model_cfg = dict(model_cfg)
        self.num_steps = int(model_cfg["num_steps"])
        self.model_root = _resolve_pi05_model_root(model_cfg)
        self.norm_stats_path = Path(model_cfg["norm_stats_path"]).expanduser().resolve()
        norm_stats = normalize.load(self.norm_stats_path)
        train_config = make_pack_plate_train_config(model_cfg)
        self.policy = create_trained_policy(
            train_config, str(self.model_root), norm_stats=norm_stats
        )
        self.reset()

    def sampling_modes(self) -> list[str]:
        return ["default", "rtc"]

    def get_action_rtc(self, sampling: dict) -> list[dict]:
        if len(self.env_indices) != 1:
            raise ValueError("pack-plate RTC requires exactly one current observation")
        required = {"action_condition", "condition_weights", "beta"}
        if required - sampling.keys():
            raise ValueError(f"pack-plate RTC is missing {sorted(required - sampling.keys())}")
        beta = float(sampling["beta"])
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("pack-plate RTC beta must be finite and positive")
        observation = self.observations[self.env_indices[0]]
        anchor = np.asarray(observation["state"], dtype=np.float32)
        weights = np.asarray(sampling["condition_weights"], dtype=np.float32)
        condition = encode_rtc_condition(sampling["action_condition"], weights, anchor)
        predicted = self.policy.infer(
            observation,
            num_steps=self.num_steps,
            action_condition=condition,
            condition_weights=weights,
            rtc_beta=beta,
        )
        return decode_actions(anchor, predicted["actions"])
