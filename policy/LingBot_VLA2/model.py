"""Formal XPolicyLab adapter for the official LingBot-VLA2 server.

The upstream source is loaded lazily from ``lingbot_vla2_root``. This module
owns only the XPolicyLab observation/action translation; model construction,
feature normalization and flow sampling remain in the official implementation.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"
OFFICIAL_DEPLOY_MODULE = "deploy.lingbot_vla_v2_policy"
OFFICIAL_FEATURE_MODULE = "lingbotvla.data.vla_data.utils"
ABSOLUTE_ACTION_SEMANTICS = "absolute_joint_position"
RELATIVE_ACTION_SEMANTICS = "anchor_relative_arm_absolute_gripper"
EXPECTED_CAMERAS = ["camera_top", "camera_wrist_left", "camera_wrist_right"]
EXPECTED_JOINTS = [
    {"arm.position": 14},
    {"end.position": 14},
    {"effector.position": 2},
    {"waist.position": 4},
    {"head.position": 2},
    {"base.position": 3},
    {"hand.position": 12},
]


def _normalize_official_feature_literals(data_config: Any) -> Any:
    """Bridge official YAML mappings to the strings FeatureTransform parses."""

    for field in ("joints", "norm_type"):
        values = getattr(data_config, field, None)
        if values is None:
            continue
        normalized = []
        for value in values:
            if isinstance(value, Mapping):
                normalized.append(repr(dict(value)))
            elif isinstance(value, str):
                normalized.append(value)
            else:
                raise TypeError(f"data.{field} entries must be mappings or strings")
        setattr(data_config, field, normalized)
    return data_config


def _parse_feature_literals(values: object) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        return []
    parsed: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, Mapping):
            parsed.append(dict(value))
            continue
        if not isinstance(value, str):
            return []
        try:
            literal = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        if not isinstance(literal, dict):
            return []
        parsed.append(literal)
    return parsed


def _resolve_path(value: object, *, name: str, base_dir: Path = POLICY_DIR) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be configured")
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@contextlib.contextmanager
def _temporary_environment(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def expected_training_config(checkpoint_path: Path) -> Path:
    """Return the path hard-coded by the official V2 checkpoint loader."""

    return checkpoint_path.parent.parent.parent / "lingbotvla_cli.yaml"


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("global_step_"))
    except ValueError:
        return -1


def _resolve_model_root(model_cfg: Mapping[str, Any]) -> Path:
    checkpoint_root = resolve_checkpoint_root(
        dict(model_cfg),
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("model_root",),
    )
    if (checkpoint_root / "model.safetensors.index.json").is_file():
        return checkpoint_root
    candidates = sorted(
        checkpoint_root.glob("checkpoints/global_step_*/hf_ckpt"),
        key=_checkpoint_step,
    )
    complete = [
        path for path in candidates if (path / "model.safetensors.index.json").is_file()
    ]
    if not complete:
        raise FileNotFoundError(f"no complete HuggingFace checkpoint under {checkpoint_root}")
    return complete[-1]


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


def validate_deployment(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate deployment metadata without importing torch or loading weights."""

    errors: list[str] = []
    resolved_paths: dict[str, Path] = {}
    for key in ("lingbot_vla2_root", "qwen3vl_path"):
        try:
            resolved_paths[key] = _resolve_path(model_cfg.get(key), name=key)
        except ValueError as exc:
            errors.append(str(exc))

    try:
        checkpoint_path = _resolve_model_root(model_cfg)
    except (FileNotFoundError, ValueError) as exc:
        checkpoint_path = POLICY_DIR / "__missing_checkpoint__"
        errors.append(str(exc))
    run_root = checkpoint_path.parent.parent.parent
    for key, default in (
        ("training_config_path", run_root / "lingbotvla_cli.yaml"),
        ("robot_config_path", run_root / "robot_config.yaml"),
        ("norm_stats_path", run_root / "norm_stats.json"),
    ):
        value = model_cfg.get(key)
        if value is None or not str(value).strip():
            resolved_paths[key] = default
            continue
        try:
            resolved_paths[key] = _resolve_path(value, name=key)
        except ValueError as exc:
            errors.append(str(exc))

    source_root = resolved_paths.get(
        "lingbot_vla2_root", POLICY_DIR / "__missing_source__"
    )
    training_config_path = resolved_paths.get(
        "training_config_path", POLICY_DIR / "__missing_training_config__"
    )
    robot_config_path = resolved_paths.get(
        "robot_config_path", POLICY_DIR / "__missing_robot_config__"
    )
    norm_stats_path = resolved_paths.get(
        "norm_stats_path", POLICY_DIR / "__missing_stats__"
    )
    qwen3vl_path = resolved_paths.get(
        "qwen3vl_path", POLICY_DIR / "__missing_qwen3vl_processor__"
    )

    raw_action_horizon = model_cfg.get("action_horizon")
    action_horizon = (
        raw_action_horizon
        if isinstance(raw_action_horizon, int) and not isinstance(raw_action_horizon, bool)
        else 0
    )
    if action_horizon <= 0:
        errors.append("action_horizon must be a positive integer")
    raw_native_hz = model_cfg.get("native_hz")
    native_hz = (
        float(raw_native_hz)
        if isinstance(raw_native_hz, int | float) and not isinstance(raw_native_hz, bool)
        else 0.0
    )
    if not np.isfinite(native_hz) or native_hz <= 0:
        errors.append("native_hz must be finite and positive")

    if training_config_path != expected_training_config(checkpoint_path):
        errors.append(
            "checkpoint layout is incompatible with the official loader: "
            "training_config must equal checkpoint.parent.parent.parent/lingbotvla_cli.yaml"
        )

    required = {
        "official_deploy": source_root / "deploy/lingbot_vla_v2_policy.py",
        "official_feature_transform": source_root / "lingbotvla/data/vla_data/utils.py",
        "checkpoint_index": checkpoint_path / "model.safetensors.index.json",
        "training_config": training_config_path,
        "robot_config": robot_config_path,
        "norm_stats": norm_stats_path,
        "qwen3vl_config": qwen3vl_path / "config.json",
        "rtc_integration": POLICY_DIR / "rtc.py",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        missing_message = "missing files: " + ", ".join(missing)
        if missing_message not in errors:
            errors.append(missing_message)

    expected_revision = model_cfg.get("official_source_revision")
    actual_revision = _source_revision(source_root)
    if expected_revision is not None and (
        not isinstance(expected_revision, str)
        or len(expected_revision) != 40
        or any(char not in "0123456789abcdef" for char in expected_revision.lower())
    ):
        errors.append("official_source_revision must be a 40-char hex commit")
    elif isinstance(expected_revision, str):
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
    action_semantics: str | None = None
    if robot_config_path.is_file():
        try:
            loaded_robot_config = yaml.safe_load(robot_config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"invalid robot config: {exc}")
            loaded_robot_config = {}
        robot_config = loaded_robot_config if isinstance(loaded_robot_config, dict) else {}
        action_entries = robot_config.get("actions", [])
        expected_absolute_actions = {
            "action.arm.position": {
                "origin_keys": "action.arm.position",
                "subtract_state": False,
            },
            "action.effector.position": {
                "origin_keys": "action.effector.position",
                "subtract_state": False,
            },
        }
        expected_relative_actions = {
            "action.arm.position": {
                "origin_keys": [
                    {"action": {"start": 0, "end": 6}},
                    {"action": {"start": 7, "end": 13}},
                ],
                "subtract_state": True,
                "relative_type": "vector",
            },
            "action.effector.position": {
                "origin_keys": [
                    {"action": {"start": 6, "end": 7}},
                    {"action": {"start": 13, "end": 14}},
                ],
                "subtract_state": False,
                "relative_type": None,
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
                        "relative_type": options.get("relative_type"),
                    }
        comparable_absolute_actions = {
            key: {
                "origin_keys": value["origin_keys"],
                "subtract_state": value["subtract_state"],
                "relative_type": None,
            }
            for key, value in expected_absolute_actions.items()
        }
        if seen_actions == comparable_absolute_actions:
            action_semantics = ABSOLUTE_ACTION_SEMANTICS
        elif seen_actions == expected_relative_actions:
            action_semantics = RELATIVE_ACTION_SEMANTICS
        else:
            errors.append(
                "robot config must expose the supported absolute or "
                "anchor-relative YAM action contract"
            )
        state_entries = robot_config.get("states")
        seen_states: dict[str, Any] = {}
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
        expected_relative_states = {
            "observation.state.arm.position": [
                {"observation.state": {"start": 0, "end": 6}},
                {"observation.state": {"start": 7, "end": 13}},
            ],
            "observation.state.effector.position": [
                {"observation.state": {"start": 6, "end": 7}},
                {"observation.state": {"start": 13, "end": 14}},
            ],
        }
        images = robot_config.get("images")
        if action_semantics == ABSOLUTE_ACTION_SEMANTICS and seen_states != expected_states:
            errors.append("robot config must expose arm and effector position states")
        if (
            action_semantics == RELATIVE_ACTION_SEMANTICS
            and seen_states != expected_relative_states
        ):
            errors.append("relative robot config must expose the packed 14D YAM state")
        expected_absolute_images = [
            f"observation.images.{name}" for name in EXPECTED_CAMERAS
        ]
        expected_relative_images = [
            {"observation.images.camera_top": {"origin_keys": "observation.images.top_rgb"}},
            {
                "observation.images.camera_wrist_left": {
                    "origin_keys": "observation.images.left_rgb"
                }
            },
            {
                "observation.images.camera_wrist_right": {
                    "origin_keys": "observation.images.right_rgb"
                }
            },
        ]
        expected_images = (
            expected_relative_images
            if action_semantics == RELATIVE_ACTION_SEMANTICS
            else expected_absolute_images
        )
        if images != expected_images:
            errors.append("robot config camera mapping does not match its action contract")

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
        if _parse_feature_literals(data.get("joints", [])) != EXPECTED_JOINTS:
            errors.append(f"training config joints must be {EXPECTED_JOINTS}")
        chunk_size = train.get("chunk_size")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            errors.append("training config train.chunk_size must be an explicit positive integer")
            chunk_size = 0
        if action_horizon != chunk_size:
            errors.append(
                f"action_horizon {action_horizon} must equal "
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
        "source_root": str(source_root),
        "source_revision": actual_revision,
        "expected_source_revision": expected_revision,
        "checkpoint_path": str(checkpoint_path),
        "training_config_path": str(training_config_path),
        "robot_config_path": str(robot_config_path),
        "norm_stats_path": str(norm_stats_path),
        "qwen3vl_path": str(qwen3vl_path),
        "action_space": ABSOLUTE_ACTION_SEMANTICS,
        "action_semantics": action_semantics,
        "action_horizon": action_horizon,
        "native_hz": native_hz,
        "rtc_capability": (
            "pi_guided_v1_sampler"
            if action_semantics == ABSOLUTE_ACTION_SEMANTICS
            else "blocked_relative_action_contract"
        ),
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


def encode_observation(
    observation: Mapping[str, Any],
    default_prompt: str,
    robot_info: Mapping[str, Sequence[int]],
    action_semantics: str = ABSOLUTE_ACTION_SEMANTICS,
) -> dict[str, Any]:
    """Translate one XPolicyLab YAM observation to official LingBot features."""

    arm_dims = list(robot_info["arm_dim"])
    effector_dims = list(robot_info["ee_dim"])
    if len(arm_dims) != 2 or len(effector_dims) != 2:
        raise ValueError("LingBot_VLA2 YAM adapter requires exactly two arms")
    prompt = observation.get("instruction") or observation.get("prompt") or default_prompt
    left_arm = _state_value(observation, "left_arm_joint_state", arm_dims[0])
    left_gripper = _state_value(observation, "left_ee_joint_state", effector_dims[0])
    right_arm = _state_value(observation, "right_arm_joint_state", arm_dims[1])
    right_gripper = _state_value(observation, "right_ee_joint_state", effector_dims[1])
    if action_semantics == RELATIVE_ACTION_SEMANTICS:
        return {
            "observation.images.top_rgb": _extract_image(observation, "cam_head"),
            "observation.images.left_rgb": _extract_image(observation, "cam_left_wrist"),
            "observation.images.right_rgb": _extract_image(observation, "cam_right_wrist"),
            "observation.state": np.concatenate(
                [left_arm, left_gripper, right_arm, right_gripper]
            ),
            "task": str(prompt),
        }
    if action_semantics != ABSOLUTE_ACTION_SEMANTICS:
        raise ValueError(f"unsupported LingBot-VLA2 action semantics: {action_semantics}")
    return {
        "observation.images.camera_top": _extract_image(observation, "cam_head"),
        "observation.images.camera_wrist_left": _extract_image(observation, "cam_left_wrist"),
        "observation.images.camera_wrist_right": _extract_image(observation, "cam_right_wrist"),
        "observation.state.arm.position": np.concatenate([left_arm, right_arm]),
        "observation.state.effector.position": np.concatenate(
            [left_gripper, right_gripper]
        ),
        "task": str(prompt),
    }


def decode_actions(
    result: Mapping[str, Any],
    robot_info: Mapping[str, Sequence[int]],
) -> list[dict[str, np.ndarray]]:
    """Translate official arm/gripper features to XPolicyLab action steps."""

    arm_dims = list(robot_info["arm_dim"])
    effector_dims = list(robot_info["ee_dim"])
    if len(arm_dims) != 2 or len(effector_dims) != 2:
        raise ValueError("LingBot_VLA2 YAM adapter requires exactly two arms")
    arm_width = sum(arm_dims)
    effector_width = sum(effector_dims)
    arms = np.asarray(result["action.arm.position"], dtype=np.float32)
    effectors = np.asarray(result["action.effector.position"], dtype=np.float32)
    if arms.ndim != 2 or arms.shape[1] != arm_width:
        raise ValueError(f"action.arm.position must be (H, {arm_width}), got {arms.shape}")
    if effectors.shape != (arms.shape[0], effector_width):
        raise ValueError(
            "action.effector.position must be "
            f"{(arms.shape[0], effector_width)}, got {effectors.shape}"
        )
    if not np.isfinite(arms).all() or not np.isfinite(effectors).all():
        raise ValueError("LingBot-VLA2 returned non-finite actions")
    return [
        {
            "left_arm_joint_state": arms[index, : arm_dims[0]].copy(),
            "left_ee_joint_state": effectors[index, : effector_dims[0]].copy(),
            "right_arm_joint_state": arms[index, arm_dims[0] :].copy(),
            "right_ee_joint_state": effectors[index, effector_dims[0] :].copy(),
        }
        for index in range(arms.shape[0])
    ]


class Model(ModelTemplate):
    """XPolicyLab wrapper around the official ``LingbotVLAv2Server``."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        self.model_cfg = dict(model_cfg)
        if self.model_cfg.get("action_type") != "joint":
            raise ValueError("LingBot_VLA2 YAM adapter only supports action_type: joint")
        self.robot_info = get_robot_action_dim_info(str(self.model_cfg["env_cfg_type"]))
        if self.robot_info != {"arm_dim": [6, 6], "ee_dim": [1, 1]}:
            raise ValueError(
                f"LingBot_VLA2 YAM adapter requires 6+1 dual arms, got {self.robot_info}"
            )
        self.action_dim = sum(self.robot_info["arm_dim"]) + sum(self.robot_info["ee_dim"])

        report = validate_deployment(self.model_cfg)
        if report["status"] != "ready":
            raise RuntimeError(
                "LingBot-VLA2 deployment config is incomplete: "
                + "; ".join(report["errors"])
            )
        self.deployment = report
        self.action_semantics = str(report["action_semantics"])

        self.default_prompt = str(
            self.model_cfg.get("default_prompt")
            or self.model_cfg.get("task_name")
            or "Perform the instructed bimanual manipulation task."
        )
        self.action_horizon = int(report["action_horizon"])
        self._observations: list[dict[str, Any]] | None = None
        self._latest_env_idx_list = [0]
        self.model = self._load_official_server()
        self._rtc_bridge = None
        if self.action_semantics == ABSOLUTE_ACTION_SEMANTICS:
            from .rtc import LingBotRtcBridge

            self._rtc_bridge = LingBotRtcBridge(self.model, self.robot_info)

    def _load_official_server(self):
        source_root = _resolve_path(self.model_cfg["lingbot_vla2_root"], name="lingbot_vla2_root")
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        deploy_module = importlib.import_module(OFFICIAL_DEPLOY_MODULE)
        feature_module = importlib.import_module(OFFICIAL_FEATURE_MODULE)

        with _temporary_environment("QWEN3VL_PATH", self.deployment["qwen3vl_path"]):
            server = deploy_module.LingbotVLAv2Server(
                path_to_pi_model=self.deployment["checkpoint_path"],
                robot_norm_path=self.deployment["norm_stats_path"],
                use_length=self.action_horizon,
                chunk_ret=True,
                use_bf16=bool(self.model_cfg.get("use_bf16", True)),
                use_fp32=bool(self.model_cfg.get("use_fp32", False)),
                use_compile=bool(self.model_cfg.get("use_compile", False)),
            )
        data_config = _normalize_official_feature_literals(server.data_config)
        feature_transform = feature_module.FeatureTransform(
            self.deployment["robot_config_path"],
            data_config,
            server.config,
            server.processor,
            chunk_size=server.config.chunk_size,
            norm_stats_path=self.deployment["norm_stats_path"],
        )
        server.data_config = data_config
        server.vla.feature_transform = feature_transform
        server.action_key = feature_transform.org_features["actions"]
        return server

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: Sequence[Mapping[str, Any]]) -> None:
        self._latest_env_idx_list = [
            int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)
        ]
        self._observations = [
            encode_observation(
                obs,
                self.default_prompt,
                self.robot_info,
                self.action_semantics,
            )
            for obs in obs_list
        ]

    def get_action(self, **_: Any) -> object:
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_rtc(self, sampling: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
        if self.action_semantics != ABSOLUTE_ACTION_SEMANTICS or self._rtc_bridge is None:
            raise ValueError(
                "LingBot-VLA2 RTC is unavailable for anchor-relative checkpoints; "
                "their condition must first receive a model-native relative-action RTC contract"
            )
        if self._observations is None or len(self._observations) != 1:
            raise RuntimeError("RTC requires exactly one update_obs observation")
        required = {"action_condition", "condition_weights", "beta"}
        missing = sorted(required.difference(sampling))
        if missing:
            raise ValueError(f"RTC sampling is missing fields: {missing}")
        condition = np.asarray(sampling["action_condition"], dtype=np.float32)
        weights = np.asarray(sampling["condition_weights"], dtype=np.float32)
        beta = float(sampling["beta"])
        if condition.shape != (self.action_horizon, self.action_dim):
            raise ValueError(
                "RTC action_condition must have shape "
                f"{(self.action_horizon, self.action_dim)}, got {condition.shape}"
            )
        if weights.shape != (self.action_horizon,):
            raise ValueError(
                f"RTC condition_weights must have shape {(self.action_horizon,)}, "
                f"got {weights.shape}"
            )
        if not np.isfinite(condition).all() or not np.isfinite(weights).all():
            raise ValueError("RTC sampling arrays must be finite")
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("RTC condition_weights must be in [0, 1]")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"RTC beta must be finite and positive, got {beta}")
        result = self._rtc_bridge.infer(
            self._observations[0], condition, weights, beta
        )
        return decode_actions(result, self.robot_info)

    def _infer_one(self, observation: Mapping[str, Any]) -> object:
        if self.action_semantics == ABSOLUTE_ACTION_SEMANTICS:
            return decode_actions(self.model.infer(observation), self.robot_info)

        result = self.model.infer(observation, return_normalized=True)
        normalized_actions = result.get("_normalized_actions")
        if normalized_actions is None:
            raise RuntimeError("official LingBot-VLA2 inference did not return normalized actions")

        transformed = self.model._prepare_model_input(observation)
        transformed["actions"] = normalized_actions
        if self.model.use_bf16 and "state" in transformed:
            transformed["state"] = transformed["state"].float()
        feature_transform = self.model.vla.feature_transform
        native = feature_transform.reverse_pad_and_concat(transformed)
        native = feature_transform.normalizer.unnormalize(native)
        return {
            "actions": decode_actions(native, self.robot_info),
            "action_semantics": RELATIVE_ACTION_SEMANTICS,
        }

    def get_action_batch(self, env_idx_list=None, **_: Any):
        if self._observations is None:
            raise RuntimeError("update_obs or update_obs_batch must be called first")
        indices = list(env_idx_list or self._latest_env_idx_list)
        if len(indices) != len(self._observations):
            raise ValueError("env_idx_list size does not match the observation batch")
        return [self._infer_one(observation) for observation in self._observations]

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "lingbot_vla2",
            "task_name": self.model_cfg.get("task_name"),
            "checkpoint_variant": self.model_cfg.get("checkpoint_variant"),
            "model_root": self.deployment["checkpoint_path"],
            "training_config_path": self.deployment["training_config_path"],
            "robot_config_path": self.deployment["robot_config_path"],
            "norm_stats_path": self.deployment["norm_stats_path"],
            "qwen3vl_path": self.deployment["qwen3vl_path"],
            "action_horizon": self.action_horizon,
            "native_hz": self.deployment["native_hz"],
            "action_semantics": self.action_semantics,
        }

    def sampling_modes(self) -> list[str]:
        modes = ["default"]
        if self.action_semantics == ABSOLUTE_ACTION_SEMANTICS:
            modes.append("rtc")
        return modes

    def reset(self) -> None:
        self._observations = None
        self._latest_env_idx_list = [0]
        self.model.global_step = 0
        self.model.last_action_chunk = None
        self.model.last_normalized_action_chunk = None
