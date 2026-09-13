"""SAPolicy YAM adapter: standard RGB/EE observations, actions and isolated batches.

The vendored checkpoint sampler owns preprocessing, normalization and diffusion.
The explicit packed_ee_wire mode retains the previous ManiMux action contract.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.SAPolicy import DEFAULT_SAPOLICY_ROOT, ensure_sapolicy_on_path, get_model
from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state

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
        raise ValueError(
            f"relative actions must have shape (H, {NATIVE_ACTION_DIM}), got {actions.shape}"
        )
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
        raise ValueError(
            f"wire actions must have shape (H, {WIRE_ACTION_DIM}), got {actions.shape}"
        )
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
    from .assets import resolve_assets

    model_cfg = resolve_assets(model_cfg)
    root = ensure_sapolicy_on_path(model_cfg.get("sapolicy_root") or DEFAULT_SAPOLICY_ROOT)
    return get_model(
        {
            "sapolicy_cfg": model_cfg["cfg_file"],
            "resolved_cfg": model_cfg["resolved_cfg"],
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
        self._action_type = cfg.get("action_type") or "ee"
        self._env_cfg_type = cfg.get("env_cfg_type") or "yam_dual"
        if self._action_type != "ee" or self._env_cfg_type != "yam_dual":
            raise ValueError("SAPolicy supports env_cfg_type=yam_dual, action_type=ee")
        self._dimensions = get_robot_action_dim_info(self._env_cfg_type)
        if self._dimensions != {"arm_dim": [6, 6], "ee_dim": [1, 1]}:
            raise ValueError("SAPolicy checkpoint requires two YAM arms and scalar grippers")
        # Shared pack/unpack helpers take representation widths. EE poses have
        # pos3 + quat4 regardless of the robot's number of arm joints.
        self._ee_dimensions = {
            "arm_dim": [7 for _ in self._dimensions["arm_dim"]],
            "ee_dim": self._dimensions["ee_dim"],
        }
        self._output_format = cfg.get("output_format", "action_dict")
        if self._output_format not in {"action_dict", "packed_ee_wire"}:
            raise ValueError(f"Unsupported SAPolicy output_format: {self._output_format}")
        self._camera_map = dict(cfg.get("camera_map") or {})
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
        self._history_length = int(getattr(self._backend, "obs_hist", 1))
        self._histories: dict[int, deque] = {}
        self._batch_indices: list[int] = []
        self._lock = RLock()

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
                np.array([left_gripper], dtype=np.float64),
                right[:3],
                _quat_xyzw_to_rot6d(right[3:7]),
                np.array([right_gripper], dtype=np.float64),
            ]
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "sapolicy",
            "wire_action_dim": WIRE_ACTION_DIM,
            "horizon_steps": self._horizon,
            "dry_run": self._dry_run,
            "camera_names": list(self._camera_names),
            "output_format": self._output_format,
            "batch_mode": "sequential_isolated_histories",
            "rtc_condition_format": "absolute_model_frame_ee_wxyz_16",
        }

    def sampling_modes(self) -> list[str]:
        if self._dry_run:
            return ["default"]
        pipeline = getattr(getattr(self._backend, "policy", None), "pipeline", None)
        head = getattr(pipeline, "action_head", None)
        if (
            self._body_frame
            and callable(getattr(head, "rtc_condition", None))
            and getattr(head, "sequence_length", None) == self._horizon
            and getattr(head, "action_dim", None) == NATIVE_ACTION_DIM
        ):
            return ["default", "rtc"]
        return ["default"]

    def reset(self) -> None:
        with self._lock:
            self._obs = None
            self._histories.clear()
            self._batch_indices.clear()
            if self._backend is not None:
                self._backend.reset_model()

    def update_obs(self, obs: Mapping[str, Any]) -> bool:
        if not isinstance(obs, Mapping):
            raise TypeError("SAPolicy update_obs requires an observation mapping")
        with self._lock:
            self._store_observation(-1, obs)
            self._obs = dict(obs)
        return True

    def _store_observation(self, index: int, obs: Mapping[str, Any]) -> None:
        spatial = self._to_spatial_obs(obs)
        history = self._histories.setdefault(index, deque(maxlen=self._history_length))
        if not history:
            history.extend([spatial] * self._history_length)
        else:
            history.append(spatial)

    def update_obs_batch(self, obs_list: Sequence[Mapping[str, Any]]) -> bool:
        indices = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        if len(indices) != len(set(indices)) or any(i < 0 for i in indices):
            raise ValueError("Batch observations require distinct nonnegative env_idx values")
        with self._lock:
            for index, obs in zip(indices, obs_list, strict=True):
                self._store_observation(index, obs)
            self._batch_indices = indices
        return True

    def get_action(self):
        with self._lock:
            if self._obs is None:
                raise RuntimeError("get_action called before any update_obs")
            return self._action_for(-1)

    def get_action_rtc(self, sampling: Mapping[str, Any]):
        """Condition the actual DiT sampler on absolute EE16 (WXYZ) actions."""
        with self._lock:
            if "rtc" not in self.sampling_modes():
                raise NotImplementedError(
                    "RTC requires a real DiT checkpoint with its full horizon"
                )
            if self._obs is None:
                raise RuntimeError("get_action_rtc called before any update_obs")
            condition = np.asarray(sampling["action_condition"], dtype=np.float64)
            weights = np.asarray(sampling["condition_weights"], dtype=np.float64)
            beta = float(sampling.get("beta", 5.0))
            if condition.shape != (self._horizon, WIRE_ACTION_DIM):
                raise ValueError(
                    f"RTC action_condition must be ({self._horizon}, {WIRE_ACTION_DIM})"
                )
            if weights.shape != (self._horizon,):
                raise ValueError(f"RTC condition_weights must be ({self._horizon},)")
            if not np.isfinite(condition).all() or not np.isfinite(weights).all():
                raise ValueError("RTC condition and weights must be finite")
            if np.any((weights < 0) | (weights > 1)) or not np.isfinite(beta) or beta <= 0:
                raise ValueError(
                    "RTC weights must be in [0,1] and beta must be positive and finite"
                )
            for offset in (3, 11):
                if np.any(np.linalg.norm(condition[:, offset : offset + 4], axis=-1) < 1e-8):
                    raise ValueError("RTC conditions require nonzero WXYZ quaternions")
            return self._action_for(
                -1,
                {
                    "action_condition": condition,
                    "condition_weights": weights,
                    "beta": beta,
                },
            )

    def get_action_batch(self, env_idx_list=None):
        with self._lock:
            indices = self._batch_indices if env_idx_list is None else list(env_idx_list)
            if env_idx_list is None and not indices:
                raise RuntimeError("Batch action requested before observation")
            if len(indices) != len(set(indices)):
                raise ValueError("Duplicate batch environment indices")
            if any(i not in self._histories for i in indices):
                raise RuntimeError("Batch action requested before observation")
            return [self._action_for(i) for i in indices]

    def _action_for(self, index: int, rtc_sampling=None):
        history = self._histories[index]
        spatial = history[-1]
        if self._dry_run:
            row = np.concatenate(
                [
                    spatial["left_endpose"],
                    [spatial["left_gripper"]],
                    spatial["right_endpose"],
                    [spatial["right_gripper"]],
                ]
            )
            wire = np.broadcast_to(row, (self._horizon, WIRE_ACTION_DIM)).copy()
        else:
            assert self._backend is not None
            # RPC calls may run on different worker threads. Install the selected
            # environment's complete history immediately before its forward pass.
            self._backend.reset_model()
            for frame in history:
                self._backend.update_obs(frame)
            output = (
                self._backend.get_action()
                if rtc_sampling is None
                else self._backend.get_action(rtc_sampling=rtc_sampling)
            )
            wire = np.asarray(output, dtype=np.float64)
        converted = _wxyz_wire_to_xyzw(wire)
        if converted.shape != (self._horizon, WIRE_ACTION_DIM):
            raise ValueError(
                f"SAPolicy wire actions must have shape ({self._horizon}, {WIRE_ACTION_DIM}), "
                f"got {converted.shape}"
            )
        if not np.isfinite(converted).all():
            raise ValueError("SAPolicy returned non-finite actions")
        if self._output_format == "packed_ee_wire":
            return converted
        return unpack_robot_state(wire, "ee", self._ee_dimensions)

    def _to_spatial_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        extra = obs.get("additional_info") or {}
        if "left_ee_pose" in obs.get("state", {}) or not isinstance(extra.get("sapolicy"), Mapping):
            return self._standard_observation(obs)
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
        spatial = {
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
        native_hw = payload.get("image_native_hw")
        if native_hw is not None:
            spatial["image_native_hw"] = native_hw
        return spatial
    def _standard_observation(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        state = obs["state"]
        metadata = (obs.get("additional_info") or {}).get("sapolicy") or {}
        result: dict[str, Any] = {"camera_names": list(self._camera_names)}
        for side in ("left", "right"):
            # Standard XPolicyLab poses already use wxyz, as does the sampler.
            result[f"{side}_endpose"] = _as_endpose(state[f"{side}_ee_pose"])
            grip = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float64)
            if grip.shape != (1,) or not np.isfinite(grip).all():
                raise ValueError(f"{side}_ee_joint_state must contain one finite aperture")
            result[f"{side}_gripper"] = float(grip[0])
        images, intrinsics, native_hw = {}, {}, {}
        for name in self._camera_names:
            source = self._camera_map.get(name, name)
            view = obs["vision"][source]
            images[name] = _camera_image(obs, source)
            matrix = np.asarray(
                view.get("intrinsic_matrix", (metadata.get("intrinsics") or {}).get(name)),
                dtype=np.float64,
            )
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid intrinsic_matrix for {source}")
            intrinsics[name] = matrix
            native_hw[name] = list(
                (metadata.get("image_native_hw") or {}).get(
                    name, view.get("shape", images[name].shape[:2])
                )[:2]
            )
        result.update(images=images, intrinsics=intrinsics, image_native_hw=native_hw)
        return result
