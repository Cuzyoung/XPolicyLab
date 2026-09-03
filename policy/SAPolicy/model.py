"""XPolicyLab shim around ``~/sa/SpatialAlignPolicy``.

Inference is ``SAPolicyRoboTwinModel`` from that tree. This file only:
- satisfies the XPolicyLab ``Model`` / dry-run unit tests without importing torch
- remaps ManiMux xyzw grasp-site / ABC TCP observations onto that server
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.SAPolicy import DEFAULT_SAPOLICY_ROOT, ensure_sapolicy_on_path, get_model

WIRE_ACTION_DIM = 16
NATIVE_ACTION_DIM = 20


def _as_endpose(values: object) -> np.ndarray:
    endpose = np.asarray(values, dtype=np.float64).reshape(-1)
    if endpose.shape != (7,) or not np.isfinite(endpose).all():
        raise ValueError(f"SAPolicy endpose must have 7 finite values, got {endpose.shape}")
    return endpose


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    vectors = np.asarray(rot6d, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, 6)
    first = vectors[:, :3]
    second = vectors[:, 3:]
    first = first / np.clip(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8, None)
    second = second - (first * second).sum(-1, keepdims=True) * first
    second = second / np.clip(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8, None)
    rotation = np.stack([first, second, np.cross(first, second)], axis=-1)
    return rotation[0] if np.asarray(rot6d).ndim == 1 else rotation


def _quat_xyzw_to_rot6d(quaternion_xyzw: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64)).as_matrix()
    return rotation[:, :2].reshape(6, order="F")


def relative_actions_to_wire(
    relative: np.ndarray,
    left_endpose: np.ndarray,
    right_endpose: np.ndarray,
    *,
    body_frame: bool = True,
) -> np.ndarray:
    """Convert DiT ``[pose18 | grip2]`` relative actions to absolute EE wire.

    Wire layout is ``pos3 + quat_xyzw + grip`` per arm (16D). Poses stay in the
    YAM grasp-site / ABC TCP frame.
    """
    actions = np.asarray(relative, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != NATIVE_ACTION_DIM:
        raise ValueError(f"relative actions must have shape (H, {NATIVE_ACTION_DIM}), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("relative actions contain non-finite values")

    wire = np.empty((actions.shape[0], WIRE_ACTION_DIM), dtype=np.float64)
    for arm_index, measured in enumerate((_as_endpose(left_endpose), _as_endpose(right_endpose))):
        pose = actions[:, arm_index * 9 : arm_index * 9 + 9]
        current_position = measured[:3]
        current_rotation = Rotation.from_quat(measured[3:7]).as_matrix()
        relative_rotation = _rot6d_to_matrix(pose[:, 3:9])
        if body_frame:
            absolute_position = current_position + (current_rotation @ pose[:, :3].T).T
            absolute_rotation = current_rotation @ relative_rotation
        else:
            absolute_position = current_position + pose[:, :3]
            absolute_rotation = relative_rotation
        offset = arm_index * 8
        wire[:, offset : offset + 3] = absolute_position
        wire[:, offset + 3 : offset + 7] = Rotation.from_matrix(absolute_rotation).as_quat()
        wire[:, offset + 7] = actions[:, 18 + arm_index]
    return wire


def _xyzw_to_wxyz_endpose(endpose: np.ndarray) -> np.ndarray:
    values = _as_endpose(endpose)
    return np.concatenate([values[:3], values[6:7], values[3:6]])


def _wxyz_wire_to_xyzw(wire: np.ndarray) -> np.ndarray:
    actions = np.asarray(wire, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != WIRE_ACTION_DIM:
        raise ValueError(f"wire actions must have shape (H, {WIRE_ACTION_DIM}), got {actions.shape}")
    converted = actions.copy()
    for offset in (0, 8):
        wxyz = actions[:, offset + 3 : offset + 7]
        converted[:, offset + 3 : offset + 7] = np.concatenate([wxyz[:, 1:4], wxyz[:, :1]], axis=-1)
    return converted


def _sapolicy_payload(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    extra = obs.get("additional_info")
    if not isinstance(extra, Mapping):
        raise KeyError("SAPolicy observation is missing additional_info")
    payload = extra.get("sapolicy")
    if not isinstance(payload, Mapping):
        raise KeyError("SAPolicy observation is missing additional_info.sapolicy")
    return payload


def _camera_image(obs: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        image = np.asarray(obs["vision"][name]["color"])
    except KeyError as exc:
        raise KeyError(f"missing RGB camera: vision.{name}.color") from exc
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"vision.{name}.color must have shape [H,W,3], got {image.shape}")
    return image


def _load_spatial_align(model_cfg: Mapping[str, Any]) -> Any:
    root = ensure_sapolicy_on_path(model_cfg.get("sapolicy_root") or DEFAULT_SAPOLICY_ROOT)
    return get_model(
        {
            "sapolicy_cfg": model_cfg["cfg_file"],
            "ckpt_path": model_cfg["model_path"],
            "workspace": model_cfg.get("workspace", str(root.parent)),
            "n_action_steps": int(model_cfg.get("action_horizon", 16)),
            "device": str(model_cfg.get("device", "cuda")),
            "use_ema": bool(model_cfg.get("use_ema", True)),
            "backbone_path": model_cfg.get("backbone_path"),
            "normalizer_path": model_cfg.get("normalizer_path"),
            "sapolicy_root": str(root),
            # ABC / ManiMux wire is TCP. RoboTwin eval still defaults to 0.12.
            "tcp_forward_offset_m": float(model_cfg.get("tcp_forward_offset_m", 0.0)),
        }
    )


class Model(ModelTemplate):
    def __init__(self, model_cfg: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(model_cfg or {})
        self._cfg = cfg
        self._dry_run = bool(cfg.get("dry_run", False))
        self._horizon = int(cfg.get("action_horizon", 16))
        if self._horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self._horizon}")
        cameras = cfg.get("camera_names", ["agentview"])
        if not isinstance(cameras, Sequence) or isinstance(cameras, str | bytes) or not cameras:
            raise ValueError("camera_names must be a non-empty list")
        self._camera_names = [str(name) for name in cameras]
        self._body_frame = bool(cfg.get("body_frame_actions", True))
        self._obs: dict[str, Any] | None = None
        self._backend = None if self._dry_run else _load_spatial_align(cfg)
        self.model = self._backend

    def _pack_state(
        self,
        left_endpose: np.ndarray,
        right_endpose: np.ndarray,
        left_gripper: float,
        right_gripper: float,
    ) -> np.ndarray:
        left = _as_endpose(left_endpose)
        right = _as_endpose(right_endpose)
        return np.concatenate(
            [
                left[:3],
                _quat_xyzw_to_rot6d(left[3:7]),
                right[:3],
                _quat_xyzw_to_rot6d(right[3:7]),
                np.array([left_gripper, right_gripper], dtype=np.float64),
            ]
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "sapolicy",
            "wire_action_dim": WIRE_ACTION_DIM,
            "horizon_steps": self._horizon,
            "dry_run": self._dry_run,
            "camera_names": list(self._camera_names),
        }

    def reset(self) -> None:
        self._obs = None
        if self._backend is not None:
            self._backend.reset_model()

    def update_obs(self, obs: Mapping[str, Any]) -> bool:
        if not isinstance(obs, Mapping):
            raise TypeError("SAPolicy update_obs requires an observation mapping")
        self._obs = dict(obs)
        if self._backend is not None:
            self._backend.update_obs(self._to_spatial_obs(self._obs))
        return True

    def get_action(self) -> np.ndarray:
        if self._obs is None:
            raise RuntimeError("get_action called before any update_obs")
        payload = _sapolicy_payload(self._obs)
        left = _as_endpose(payload["left_endpose"])
        right = _as_endpose(payload["right_endpose"])
        left_grip = float(payload["left_gripper"])
        right_grip = float(payload["right_gripper"])
        if self._dry_run:
            row = np.concatenate(
                [left, np.array([left_grip]), right, np.array([right_grip])]
            )
            return np.broadcast_to(row, (self._horizon, WIRE_ACTION_DIM)).copy()
        assert self._backend is not None
        wire = np.asarray(self._backend.get_action(), dtype=np.float64)
        converted = _wxyz_wire_to_xyzw(wire)
        if converted.shape != (self._horizon, WIRE_ACTION_DIM):
            raise ValueError(
                f"SAPolicy wire actions must have shape ({self._horizon}, {WIRE_ACTION_DIM}), "
                f"got {converted.shape}"
            )
        return converted

    def _to_spatial_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        payload = _sapolicy_payload(obs)
        camera_names = [str(name) for name in payload.get("camera_names", self._camera_names)]
        intrinsics = payload.get("intrinsics") or {}
        images = {name: _camera_image(obs, name) for name in camera_names}
        Ks = {}
        for name in camera_names:
            matrix = np.asarray(intrinsics[name], dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(f"intrinsics for {name!r} must be 3x3, got {matrix.shape}")
            Ks[name] = matrix
        first = camera_names[0]
        return {
            "left_endpose": _xyzw_to_wxyz_endpose(payload["left_endpose"]),
            "right_endpose": _xyzw_to_wxyz_endpose(payload["right_endpose"]),
            "left_gripper": float(payload["left_gripper"]),
            "right_gripper": float(payload["right_gripper"]),
            "camera_names": camera_names,
            "images": images,
            "intrinsics": Ks,
            "image": images[first],
            "intrinsic_cv": Ks[first],
        }
