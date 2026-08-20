"""Formal XPolicyLab adapter for the official LingBot-VLA2 server.

The upstream source is loaded lazily from ``lingbot_vla2_root``. This module
owns only the XPolicyLab observation/action translation; model construction,
feature normalization and flow sampling remain in the official implementation.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
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
BUNDLE_SCHEMA_VERSION = "manimux.lingbot_vla2_yam_bundle.v1"
EXPECTED_CAMERAS = ["camera_top", "camera_wrist_left", "camera_wrist_right"]
EXPECTED_JOINTS = [
    "arm.position: 14",
    "end.position: 14",
    "effector.position: 2",
    "waist.position: 4",
    "head.position: 2",
    "base.position: 3",
    "hand.position: 12",
]


def _resolve_path(value: object, *, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be configured")
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def expected_training_config(checkpoint_path: Path) -> Path:
    """Return the path hard-coded by the official V2 checkpoint loader."""

    return checkpoint_path.parent.parent.parent / "lingbotvla_cli.yaml"


def _bundle_artifact(bundle_root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest artifacts.{name} must be a non-empty relative path")
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise ValueError(f"manifest artifacts.{name} must be relative to the bundle root")
    resolved = (bundle_root / raw_path).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError(f"manifest artifacts.{name} escapes the bundle root") from exc
    return resolved


def _source_revision(source_root: Path) -> str | None:
    if not source_root.is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip().lower()
    if result.returncode == 0 and len(revision) == 40:
        return revision
    return None


def _check_keys(
    value: object,
    *,
    label: str,
    required: set[str],
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"manifest {label} must be a mapping")
        return {}
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required))
    if missing:
        errors.append(f"manifest {label} is missing keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"manifest {label} has unknown keys: {', '.join(unknown)}")
    return value


def validate_bundle(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate deployment metadata without importing torch or loading weights."""

    source_root = _resolve_path(model_cfg.get("lingbot_vla2_root"), name="lingbot_vla2_root")
    manifest_path = _resolve_path(
        model_cfg.get("bundle_manifest_path"), name="bundle_manifest_path"
    )
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        errors.append("missing files: bundle_manifest")
    else:
        try:
            loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"invalid bundle manifest: {exc}")
        else:
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                errors.append("bundle manifest must be a mapping")

    if not manifest:
        source_files = [
            source_root / "deploy/lingbot_vla_v2_policy.py",
            source_root / "lingbotvla/data/vla_data/utils.py",
        ]
        if not all(path.is_file() for path in source_files):
            errors.append("missing files: official_deploy, official_feature_transform")
        return {
            "status": "blocked",
            "schema_version": None,
            "manifest_path": str(manifest_path),
            "bundle_root": str(manifest_path.parent),
            "source_root": str(source_root),
            "source_revision": _source_revision(source_root),
            "expected_source_revision": None,
            "checkpoint_path": None,
            "training_config_path": None,
            "robot_config_path": None,
            "norm_stats_path": None,
            "action_space": "absolute_joint_position",
            "action_horizon": 0,
            "native_hz": 0.0,
            "errors": errors,
        }

    _check_keys(
        manifest,
        label="root",
        required={"schema_version", "model", "artifacts", "control", "embodiment"},
        errors=errors,
    )
    model = _check_keys(
        manifest.get("model"),
        label="model",
        required={"family", "official_source_revision"},
        errors=errors,
    ) if manifest else {}
    artifacts = _check_keys(
        manifest.get("artifacts"),
        label="artifacts",
        required={"training_config", "checkpoint", "norm_stats", "robot_config"},
        errors=errors,
    ) if manifest else {}
    control = _check_keys(
        manifest.get("control"),
        label="control",
        required={"native_hz", "action_horizon", "action_space"},
        errors=errors,
    ) if manifest else {}
    embodiment = _check_keys(
        manifest.get("embodiment"),
        label="embodiment",
        required={"name", "arm_dofs", "gripper_dofs", "cameras"},
        errors=errors,
    ) if manifest else {}
    if manifest:
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            errors.append(f"schema_version must be {BUNDLE_SCHEMA_VERSION}")
        if model.get("family") != "lingbot-vla-v2":
            errors.append("manifest model.family must be lingbot-vla-v2")
        revision = model.get("official_source_revision")
        if not isinstance(revision, str) or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision.lower()
        ):
            errors.append("manifest model.official_source_revision must be a 40-char hex commit")
        if control.get("action_space") != "absolute_joint_position":
            errors.append("manifest control.action_space must be absolute_joint_position")
        if embodiment.get("name") != "yam_dual":
            errors.append("manifest embodiment.name must be yam_dual")
        if embodiment.get("arm_dofs") != [6, 6]:
            errors.append("manifest embodiment.arm_dofs must be [6, 6]")
        if embodiment.get("gripper_dofs") != [1, 1]:
            errors.append("manifest embodiment.gripper_dofs must be [1, 1]")
        if embodiment.get("cameras") != EXPECTED_CAMERAS:
            errors.append(f"manifest embodiment.cameras must be {EXPECTED_CAMERAS}")

    raw_action_horizon = control.get("action_horizon")
    action_horizon = (
        raw_action_horizon
        if isinstance(raw_action_horizon, int) and not isinstance(raw_action_horizon, bool)
        else 0
    )
    if action_horizon <= 0:
        errors.append("manifest control.action_horizon must be a positive integer")
    raw_native_hz = control.get("native_hz")
    native_hz = (
        float(raw_native_hz)
        if isinstance(raw_native_hz, int | float) and not isinstance(raw_native_hz, bool)
        else 0.0
    )
    if not np.isfinite(native_hz) or native_hz <= 0:
        errors.append("manifest control.native_hz must be finite and positive")

    bundle_root = manifest_path.parent
    resolved_artifacts: dict[str, Path] = {}
    for name in ("training_config", "checkpoint", "norm_stats", "robot_config"):
        try:
            resolved_artifacts[name] = _bundle_artifact(
                bundle_root, artifacts.get(name), name=name
            )
        except ValueError as exc:
            errors.append(str(exc))

    checkpoint_path = resolved_artifacts.get("checkpoint", bundle_root / "__missing_checkpoint__")
    training_config_path = resolved_artifacts.get(
        "training_config", bundle_root / "__missing_training_config__"
    )
    norm_stats_path = resolved_artifacts.get("norm_stats", bundle_root / "__missing_stats__")
    robot_config_path = resolved_artifacts.get(
        "robot_config", bundle_root / "__missing_robot_config__"
    )

    if training_config_path != expected_training_config(checkpoint_path):
        errors.append(
            "manifest checkpoint layout is incompatible with the official loader: "
            "training_config must equal checkpoint.parent.parent.parent/lingbotvla_cli.yaml"
        )

    required = {
        "bundle_manifest": manifest_path,
        "official_deploy": source_root / "deploy/lingbot_vla_v2_policy.py",
        "official_feature_transform": source_root / "lingbotvla/data/vla_data/utils.py",
        "checkpoint_index": checkpoint_path / "model.safetensors.index.json",
        "training_config": training_config_path,
        "robot_config": robot_config_path,
        "norm_stats": norm_stats_path,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        missing_message = "missing files: " + ", ".join(missing)
        if missing_message not in errors:
            errors.append(missing_message)

    expected_revision = model.get("official_source_revision")
    actual_revision = _source_revision(source_root)
    if isinstance(expected_revision, str) and len(expected_revision) == 40:
        if actual_revision is None:
            errors.append("official source revision cannot be verified from Git metadata")
        elif actual_revision != expected_revision.lower():
            errors.append(
                f"official source revision is {actual_revision}, "
                f"expected {expected_revision.lower()}"
            )

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
        try:
            loaded_robot_config = yaml.safe_load(robot_config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"invalid robot config: {exc}")
            loaded_robot_config = {}
        robot_config = loaded_robot_config if isinstance(loaded_robot_config, dict) else {}
        action_entries = robot_config.get("actions", [])
        expected_actions = {
            "action.arm.position": {
                "origin_keys": "action.arm.position",
                "subtract_state": False,
            },
            "action.effector.position": {
                "origin_keys": "action.effector.position",
                "subtract_state": False,
            },
        }
        seen_actions: dict[str, dict[str, Any]] = {}
        for entry in action_entries:
            if isinstance(entry, dict) and len(entry) == 1:
                key, options = next(iter(entry.items()))
                if isinstance(options, dict):
                    seen_actions[key] = {
                        "origin_keys": options.get("origin_keys"),
                        "subtract_state": options.get("subtract_state"),
                    }
        if seen_actions != expected_actions:
            errors.append(
                "robot config must expose absolute action.arm.position and action.effector.position"
            )
        state_entries = robot_config.get("states")
        seen_states: dict[str, str] = {}
        if isinstance(state_entries, list):
            for entry in state_entries:
                if isinstance(entry, dict) and len(entry) == 1:
                    key, options = next(iter(entry.items()))
                    if isinstance(options, dict):
                        seen_states[key] = options.get("origin_keys")
        expected_states = {
            "observation.state.arm.position": "observation.state.arm.position",
            "observation.state.effector.position": "observation.state.effector.position",
        }
        images = robot_config.get("images")
        if seen_states != expected_states:
            errors.append("robot config must expose arm and effector position states")
        if images != [f"observation.images.{name}" for name in EXPECTED_CAMERAS]:
            errors.append("robot config camera order does not match the bundle manifest")

    training_config: dict[str, Any] = {}
    if training_config_path.is_file():
        try:
            loaded_training_config = yaml.safe_load(
                training_config_path.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"invalid training config: {exc}")
            loaded_training_config = {}
        training_config = (
            loaded_training_config if isinstance(loaded_training_config, dict) else {}
        )
        train = training_config.get("train", {})
        data = training_config.get("data", {})
        training_model = training_config.get("model", {})
        if training_model.get("config_key") != "LingbotVLAV2Config":
            errors.append("training config model.config_key must be LingbotVLAV2Config")
        if training_model.get("post_training") is not True:
            errors.append("training config model.post_training must be true")
        for key in ("action_dim", "max_action_dim", "max_state_dim"):
            if train.get(key) != 55:
                errors.append(f"training config {key} must be 55")
        cameras = list(data.get("cameras", []))
        if cameras != EXPECTED_CAMERAS:
            errors.append(f"training config cameras must be {EXPECTED_CAMERAS}")
        if list(data.get("joints", [])) != EXPECTED_JOINTS:
            errors.append(f"training config joints must be {EXPECTED_JOINTS}")
        chunk_size = train.get("chunk_size")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            errors.append("training config train.chunk_size must be an explicit positive integer")
            chunk_size = 0
        if action_horizon != chunk_size:
            errors.append(
                f"manifest action_horizon {action_horizon} must equal "
                f"trained chunk_size {chunk_size}"
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
        "schema_version": manifest.get("schema_version"),
        "manifest_path": str(manifest_path),
        "bundle_root": str(bundle_root),
        "source_root": str(source_root),
        "source_revision": actual_revision,
        "expected_source_revision": expected_revision,
        "checkpoint_path": str(checkpoint_path),
        "training_config_path": str(training_config_path),
        "robot_config_path": str(robot_config_path),
        "norm_stats_path": str(norm_stats_path),
        "action_space": "absolute_joint_position",
        "action_horizon": action_horizon,
        "native_hz": native_hz,
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
        self.bundle = report

        self.default_prompt = str(
            self.model_cfg.get("default_prompt")
            or self.model_cfg.get("task_name")
            or "Perform the instructed bimanual manipulation task."
        )
        self.action_horizon = int(report["action_horizon"])
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
            path_to_pi_model=self.bundle["checkpoint_path"],
            robot_norm_path=self.bundle["norm_stats_path"],
            use_length=self.action_horizon,
            chunk_ret=True,
            use_bf16=bool(self.model_cfg.get("use_bf16", True)),
            use_fp32=bool(self.model_cfg.get("use_fp32", False)),
            use_compile=bool(self.model_cfg.get("use_compile", False)),
        )
        feature_transform = feature_module.FeatureTransform(
            self.bundle["robot_config_path"],
            server.data_config,
            server.config,
            server.processor,
            chunk_size=server.config.chunk_size,
            norm_stats_path=self.bundle["norm_stats_path"],
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
