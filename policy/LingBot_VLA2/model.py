"""Formal XPolicyLab adapter for the official LingBot-VLA2 server.

The upstream source is loaded lazily from ``lingbot_vla2_root``. This module
owns only the XPolicyLab observation/action translation; model construction,
feature normalization and flow sampling remain in the official implementation.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
OFFICIAL_DEPLOY_MODULE = "deploy.lingbot_vla_v2_policy"
OFFICIAL_FEATURE_MODULE = "lingbotvla.data.vla_data.utils"


def _resolve_path(value: object, *, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be configured")
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def expected_training_config(checkpoint_path: Path) -> Path:
    """Return the path hard-coded by the official V2 checkpoint loader."""

    return checkpoint_path.parent.parent.parent / "lingbotvla_cli.yaml"


def validate_bundle(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate deployment metadata without importing torch or loading weights."""

    source_root = _resolve_path(model_cfg.get("lingbot_vla2_root"), name="lingbot_vla2_root")
    checkpoint_path = _resolve_path(model_cfg.get("checkpoint_path"), name="checkpoint_path")
    robot_config_path = _resolve_path(model_cfg.get("robot_config_path"), name="robot_config_path")
    norm_stats_path = _resolve_path(model_cfg.get("norm_stats_path"), name="norm_stats_path")
    training_config_path = expected_training_config(checkpoint_path)

    required = {
        "official_deploy": source_root / "deploy/lingbot_vla_v2_policy.py",
        "official_feature_transform": source_root / "lingbotvla/data/vla_data/utils.py",
        "checkpoint_index": checkpoint_path / "model.safetensors.index.json",
        "training_config": training_config_path,
        "robot_config": robot_config_path,
        "norm_stats": norm_stats_path,
    }
    missing = [name for name, path in required.items() if not path.is_file()]

    errors: list[str] = []
    if missing:
        errors.append("missing files: " + ", ".join(missing))

    checkpoint_index = required["checkpoint_index"]
    if checkpoint_index.is_file():
        try:
            index = json.loads(checkpoint_index.read_text(encoding="utf-8"))
            shard_names = sorted(set(index["weight_map"].values()))
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"invalid checkpoint index: {exc}")
        else:
            missing_shards = [
                name for name in shard_names if not (checkpoint_path / name).is_file()
            ]
            if missing_shards:
                errors.append("checkpoint is missing shards: " + ", ".join(missing_shards))

    robot_config: dict[str, Any] = {}
    if robot_config_path.is_file():
        robot_config = yaml.safe_load(robot_config_path.read_text(encoding="utf-8")) or {}
        action_entries = robot_config.get("actions", [])
        expected_actions = {
            "action.arm.position": False,
            "action.effector.position": False,
        }
        seen: dict[str, bool] = {}
        for entry in action_entries:
            if isinstance(entry, dict) and len(entry) == 1:
                key, options = next(iter(entry.items()))
                if isinstance(options, dict):
                    seen[key] = bool(options.get("subtract_state"))
        if seen != expected_actions:
            errors.append(
                "robot config must expose absolute action.arm.position and action.effector.position"
            )

    training_config: dict[str, Any] = {}
    if training_config_path.is_file():
        training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8")) or {}
        train = training_config.get("train", {})
        data = training_config.get("data", {})
        for key in ("action_dim", "max_action_dim", "max_state_dim"):
            if int(train.get(key, -1)) != 55:
                errors.append(f"training config {key} must be 55")
        cameras = list(data.get("cameras", []))
        expected_cameras = ["camera_top", "camera_wrist_left", "camera_wrist_right"]
        if cameras != expected_cameras:
            errors.append(f"training config cameras must be {expected_cameras}")
        chunk_size = int(train.get("chunk_size", 50))
        action_horizon = int(model_cfg.get("action_horizon", 50))
        if action_horizon > chunk_size:
            errors.append(
                f"action_horizon {action_horizon} exceeds trained chunk_size {chunk_size}"
            )

    if norm_stats_path.is_file():
        try:
            norm_stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid norm stats: {exc}")
        else:
            feature_stats = norm_stats.get("norm_stats", {})
            required_stats = {
                "observation.state.arm.position",
                "observation.state.effector.position",
                "action.arm.position",
                "action.effector.position",
            }
            absent = sorted(required_stats.difference(feature_stats))
            if absent:
                errors.append("norm stats missing features: " + ", ".join(absent))
            expected_dims = {
                "observation.state.arm.position": 12,
                "observation.state.effector.position": 2,
                "action.arm.position": 12,
                "action.effector.position": 2,
            }
            for feature, dim in expected_dims.items():
                stats = feature_stats.get(feature)
                if not isinstance(stats, dict):
                    continue
                for field in ("mean", "std"):
                    values = stats.get(field)
                    if not isinstance(values, list) or len(values) != dim:
                        errors.append(f"norm stats {feature}.{field} must have {dim} values")

    return {
        "status": "ready" if not errors else "blocked",
        "source_root": str(source_root),
        "checkpoint_path": str(checkpoint_path),
        "training_config_path": str(training_config_path),
        "robot_config_path": str(robot_config_path),
        "norm_stats_path": str(norm_stats_path),
        "action_space": "absolute_joint_position",
        "errors": errors,
    }


def _extract_image(observation: Mapping[str, Any], name: str) -> np.ndarray:
    vision = observation.get("vision")
    if not isinstance(vision, Mapping) or name not in vision:
        raise KeyError(f"observation is missing vision.{name}")
    value = vision[name]
    if isinstance(value, Mapping):
        value = value.get("color")
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"vision.{name} must be HxWx3 RGB, got {image.shape}")
    return np.ascontiguousarray(image)


def _state_value(observation: Mapping[str, Any], key: str, dim: int) -> np.ndarray:
    state = observation.get("state")
    if not isinstance(state, Mapping) or key not in state:
        raise KeyError(f"observation is missing state.{key}")
    value = np.asarray(state[key], dtype=np.float32).reshape(-1)
    if value.size != dim:
        raise ValueError(f"state.{key} must have {dim} values, got {value.size}")
    return value


def encode_observation(observation: Mapping[str, Any], default_prompt: str) -> dict[str, Any]:
    """Translate one XPolicyLab YAM observation to official LingBot features."""

    prompt = observation.get("instruction") or observation.get("prompt") or default_prompt
    return {
        "observation.images.camera_top": _extract_image(observation, "cam_head"),
        "observation.images.camera_wrist_left": _extract_image(observation, "cam_left_wrist"),
        "observation.images.camera_wrist_right": _extract_image(observation, "cam_right_wrist"),
        "observation.state.arm.position": np.concatenate(
            [
                _state_value(observation, "left_arm_joint_state", 6),
                _state_value(observation, "right_arm_joint_state", 6),
            ]
        ),
        "observation.state.effector.position": np.concatenate(
            [
                _state_value(observation, "left_ee_joint_state", 1),
                _state_value(observation, "right_ee_joint_state", 1),
            ]
        ),
        "task": str(prompt),
    }


def decode_actions(result: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
    """Translate official absolute joint chunks to XPolicyLab action steps."""

    arms = np.asarray(result["action.arm.position"], dtype=np.float32)
    effectors = np.asarray(result["action.effector.position"], dtype=np.float32)
    if arms.ndim != 2 or arms.shape[1] != 12:
        raise ValueError(f"action.arm.position must be (H, 12), got {arms.shape}")
    if effectors.shape != (arms.shape[0], 2):
        raise ValueError(
            f"action.effector.position must be {(arms.shape[0], 2)}, got {effectors.shape}"
        )
    if not np.isfinite(arms).all() or not np.isfinite(effectors).all():
        raise ValueError("LingBot-VLA2 returned non-finite actions")
    return [
        {
            "left_arm_joint_state": arms[index, :6].copy(),
            "left_ee_joint_state": effectors[index, :1].copy(),
            "right_arm_joint_state": arms[index, 6:].copy(),
            "right_ee_joint_state": effectors[index, 1:].copy(),
        }
        for index in range(arms.shape[0])
    ]


class Model(ModelTemplate):
    """XPolicyLab wrapper around the official ``LingbotVLAv2Server``."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        self.model_cfg = dict(model_cfg)
        if self.model_cfg.get("action_type") != "joint":
            raise ValueError("LingBot_VLA2 YAM adapter only supports action_type: joint")
        robot_info = get_robot_action_dim_info(str(self.model_cfg["env_cfg_type"]))
        if robot_info != {"arm_dim": [6, 6], "ee_dim": [1, 1]}:
            raise ValueError(f"LingBot_VLA2 YAM adapter requires 6+1 dual arms, got {robot_info}")

        report = validate_bundle(self.model_cfg)
        if report["status"] != "ready":
            raise RuntimeError(
                "LingBot-VLA2 deployment bundle is incomplete: " + "; ".join(report["errors"])
            )

        self.default_prompt = str(
            self.model_cfg.get("default_prompt")
            or self.model_cfg.get("task_name")
            or "Perform the instructed bimanual manipulation task."
        )
        self.action_horizon = int(self.model_cfg.get("action_horizon", 50))
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._observations: list[dict[str, Any]] | None = None
        self._latest_env_idx_list = [0]
        self.model = self._load_official_server()

    def _load_official_server(self):
        source_root = _resolve_path(self.model_cfg["lingbot_vla2_root"], name="lingbot_vla2_root")
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        deploy_module = importlib.import_module(OFFICIAL_DEPLOY_MODULE)
        feature_module = importlib.import_module(OFFICIAL_FEATURE_MODULE)

        server = deploy_module.LingbotVLAv2Server(
            path_to_pi_model=str(
                _resolve_path(self.model_cfg["checkpoint_path"], name="checkpoint_path")
            ),
            robot_norm_path=str(
                _resolve_path(self.model_cfg["norm_stats_path"], name="norm_stats_path")
            ),
            use_length=self.action_horizon,
            chunk_ret=True,
            use_bf16=bool(self.model_cfg.get("use_bf16", True)),
            use_fp32=bool(self.model_cfg.get("use_fp32", False)),
            use_compile=bool(self.model_cfg.get("use_compile", False)),
        )
        feature_transform = feature_module.FeatureTransform(
            str(_resolve_path(self.model_cfg["robot_config_path"], name="robot_config_path")),
            server.data_config,
            server.config,
            server.processor,
            chunk_size=server.config.chunk_size,
            norm_stats_path=str(
                _resolve_path(self.model_cfg["norm_stats_path"], name="norm_stats_path")
            ),
        )
        server.vla.feature_transform = feature_transform
        server.action_key = feature_transform.org_features["actions"]
        return server

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: Sequence[Mapping[str, Any]]) -> None:
        self._latest_env_idx_list = [
            int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)
        ]
        self._observations = [encode_observation(obs, self.default_prompt) for obs in obs_list]

    def get_action(self, **_: Any) -> list[dict[str, np.ndarray]]:
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None, **_: Any):
        if self._observations is None:
            raise RuntimeError("update_obs or update_obs_batch must be called first")
        indices = list(env_idx_list or self._latest_env_idx_list)
        if len(indices) != len(self._observations):
            raise ValueError("env_idx_list size does not match the observation batch")
        return [decode_actions(self.model.infer(observation)) for observation in self._observations]

    def reset(self) -> None:
        self._observations = None
        self._latest_env_idx_list = [0]
        self.model.global_step = 0
        self.model.last_action_chunk = None
        self.model.last_normalized_action_chunk = None
