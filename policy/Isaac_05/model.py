"""XPolicyLab wrapper for the official PerceptronAI Isaac 0.5 LeRobot policy."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate

XPOLICYLAB_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = XPOLICYLAB_ROOT.parent
CHECKPOINTS_DIR = WORKSPACE_ROOT / "checkpoints"
OFFICIAL_LEROBOT_COMMIT = "e12389c1f8f591ad05dced4e284d4e92e48c5df4"
DEFAULT_LEROBOT_ROOT = POLICY_DIR / "lerobot"
DEFAULT_MODEL_PATH = (
    WORKSPACE_ROOT / "checkpoints/pretrained/perceptron-ai/isaac-0.5/lerobot_policy"
)
EXPECTED_ADAPTER_SCHEMA = "perceptron_isaac_deployment_adapter_v1"


def _resolve_source_path(value: object, *, default: Path) -> Path:
    raw = str(value or default)
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _git_revision(path: Path) -> str | None:
    if not path.is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _model_root(policy_path: Path, config: Mapping[str, Any]) -> Path:
    relative = Path(str(config.get("hf_model_path") or "."))
    return (policy_path / relative).resolve()


def _weight_package_status(root: Path) -> tuple[bool, int | None]:
    index_path = root / "model.safetensors.index.json"
    single_weight_path = root / "model.safetensors"
    if single_weight_path.is_file():
        return True, 0
    if not index_path.is_file():
        return False, None
    payload = _load_json(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"invalid safetensors weight map: {index_path}")
    shard_names = {str(value) for value in weight_map.values()}
    missing = sum(not (root / name).is_file() for name in shard_names)
    return missing == 0, missing


def validate_deployment(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate source and checkpoint contracts without importing Torch or allocating a model."""

    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

    lerobot_root = _resolve_source_path(
        model_cfg.get("lerobot_root"), default=DEFAULT_LEROBOT_ROOT
    )
    checkpoint_config = dict(model_cfg)
    checkpoint_config.setdefault("model_path", str(DEFAULT_MODEL_PATH))
    policy_path = resolve_checkpoint_root(
        checkpoint_config,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("model_path",),
        must_exist=False,
    )
    errors: list[str] = []

    revision = _git_revision(lerobot_root)
    if revision is None:
        errors.append(f"official LeRobot source is missing: {lerobot_root}")
    elif revision != OFFICIAL_LEROBOT_COMMIT:
        errors.append(
            "official LeRobot revision mismatch: "
            f"expected {OFFICIAL_LEROBOT_COMMIT}, got {revision}"
        )

    config_path = policy_path / "config.json"
    adapter_path = policy_path / "isaac_deployment_adapter.json"
    preprocessor_path = policy_path / "policy_preprocessor.json"
    postprocessor_path = policy_path / "policy_postprocessor.json"
    required = (config_path, adapter_path, preprocessor_path, postprocessor_path)
    for path in required:
        if not path.is_file():
            errors.append(f"checkpoint package file is missing: {path}")

    config: dict[str, Any] = {}
    adapter: dict[str, Any] = {}
    if config_path.is_file():
        config = _load_json(config_path)
        if config.get("type") != "perceptron_isaac":
            errors.append(f"checkpoint type must be 'perceptron_isaac', got {config.get('type')!r}")
        for key in ("action_dim", "proprio_dim", "chunk_size", "n_action_steps"):
            value = config.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"checkpoint config {key} must be a positive integer, got {value!r}")
        if (
            isinstance(config.get("chunk_size"), int)
            and isinstance(config.get("n_action_steps"), int)
            and config["n_action_steps"] > config["chunk_size"]
        ):
            errors.append("checkpoint n_action_steps cannot exceed chunk_size")
    if adapter_path.is_file():
        adapter = _load_json(adapter_path)
        if adapter.get("schema") != EXPECTED_ADAPTER_SCHEMA:
            errors.append(
                f"deployment adapter schema must be {EXPECTED_ADAPTER_SCHEMA!r}, "
                f"got {adapter.get('schema')!r}"
            )
        matching_fields = {
            "policy_state_dataset": "policy_state_dataset",
            "robot_type": "robot_type",
            "control_mode": "control_mode",
            "image_size": "image_size",
            "camera_order": "camera_order",
            "n_action_steps": "n_action_steps",
            "num_inference_steps": "num_inference_steps",
        }
        for config_key, adapter_key in matching_fields.items():
            config_value = config.get(config_key)
            adapter_value = adapter.get(adapter_key)
            if config_value != adapter_value:
                errors.append(
                    f"checkpoint config {config_key}={config_value!r} does not match "
                    f"deployment adapter {adapter_key}={adapter_value!r}"
                )
        stats_path = policy_path / str(config.get("native_stats_path") or "isaac_stats.json")
        if not stats_path.is_file():
            errors.append(f"checkpoint-native stats are missing: {stats_path}")
        else:
            stats = _load_json(stats_path)
            stats_fields = {
                "action_dim": "action_dim",
                "proprio_dim": "proprio_dim",
                "chunk_size": "action_horizon",
                "target_fps": "target_fps",
            }
            for config_key, stats_key in stats_fields.items():
                config_value = config.get(config_key)
                stats_value = stats.get(stats_key)
                if config_value != stats_value:
                    errors.append(
                        f"checkpoint config {config_key}={config_value!r} does not match "
                        f"native stats {stats_key}={stats_value!r}"
                    )

    root = _model_root(policy_path, config) if config else policy_path
    weights_complete, missing_weight_shards = _weight_package_status(root)
    if not weights_complete:
        detail = (
            f": {missing_weight_shards} indexed shards missing"
            if missing_weight_shards
            else ""
        )
        errors.append(f"model weights are incomplete under {root}{detail}")

    return {
        "status": "ready" if not errors else "blocked",
        "errors": errors,
        "lerobot_root": str(lerobot_root),
        "lerobot_revision": revision,
        "policy_path": str(policy_path),
        "model_root": str(root),
        "weights_complete": weights_complete,
        "missing_weight_shards": missing_weight_shards,
        "policy_state_dataset": config.get("policy_state_dataset"),
        "robot_type": config.get("robot_type"),
        "control_mode": config.get("control_mode"),
        "action_dim": config.get("action_dim"),
        "proprio_dim": config.get("proprio_dim"),
        "model_chunk_size": config.get("chunk_size"),
        "action_horizon": config.get("n_action_steps"),
        "target_fps": config.get("target_fps"),
        "camera_order": config.get("camera_order"),
    }


def _image_from_wire(vision: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in vision:
        raise KeyError(f"Isaac observation is missing camera {name!r}")
    value = vision[name]
    if isinstance(value, Mapping):
        value = value.get("color")
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Isaac camera {name!r} must be HWC RGB, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Isaac camera {name!r} must use uint8 pixels, got {image.dtype}")
    return np.ascontiguousarray(image).copy()


def _state_from_wire(
    state: Mapping[str, Any],
    *,
    state_keys: Sequence[str],
    proprio_dim: int,
) -> np.ndarray:
    direct = state.get("observation.state")
    if direct is not None:
        values = np.asarray(direct, dtype=np.float32).reshape(-1)
    else:
        missing = [key for key in state_keys if key not in state]
        if missing:
            raise KeyError(f"Isaac observation is missing state keys: {missing}")
        values = np.concatenate(
            [np.asarray(state[key], dtype=np.float32).reshape(-1) for key in state_keys]
        )
    if values.shape != (proprio_dim,) or not np.isfinite(values).all():
        raise ValueError(
            f"Isaac observation.state must contain {proprio_dim} finite values, got {values.shape}"
        )
    return np.ascontiguousarray(values).copy()


def encode_observation(
    observation: Mapping[str, Any],
    *,
    camera_map: Mapping[str, str],
    state_keys: Sequence[str],
    proprio_dim: int,
    default_prompt: str,
) -> tuple[dict[str, np.ndarray], str, int | None]:
    """Translate the XPolicy wire dictionary into the official LeRobot observation keys."""

    vision = observation.get("vision")
    state = observation.get("state")
    if not isinstance(vision, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Isaac requires mapping-valued vision and state observations")
    expected_roles = ("image", "wrist_image")
    if tuple(camera_map) != expected_roles:
        raise ValueError(f"Isaac camera_map keys must be {expected_roles}, got {tuple(camera_map)}")
    prepared = {
        f"observation.images.{role}": _image_from_wire(vision, wire_name)
        for role, wire_name in camera_map.items()
    }
    prepared["observation.state"] = _state_from_wire(
        state,
        state_keys=state_keys,
        proprio_dim=proprio_dim,
    )
    instruction = observation.get("instruction") or default_prompt
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Isaac instruction must be a non-empty string")
    additional = observation.get("additional_info")
    timestep = additional.get("timestep") if isinstance(additional, Mapping) else None
    if timestep is not None and (not isinstance(timestep, int) or timestep < 0):
        raise ValueError("Isaac additional_info.timestep must be a non-negative integer")
    return prepared, instruction.strip(), timestep


def decode_actions(actions: Any, *, action_dim: int, horizon: int) -> list[dict[str, np.ndarray]]:
    array = np.asarray(actions, dtype=np.float32)
    if array.shape != (horizon, action_dim) or not np.isfinite(array).all():
        raise ValueError(
            "official Isaac action chunk must be "
            f"{(horizon, action_dim)} and finite, got {array.shape}"
        )
    return [{"action": row.copy()} for row in array]


class Model(ModelTemplate):
    """Thin XPolicy wrapper around the official PerceptronIsaacPolicy pipeline."""

    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        report = validate_deployment(self.model_cfg)
        if report["status"] != "ready":
            raise RuntimeError("Isaac 0.5 deployment is incomplete: " + "; ".join(report["errors"]))
        self.deployment = report
        self.device = str(self.model_cfg.get("device", "cuda"))
        self.default_prompt = str(
            self.model_cfg.get("default_prompt")
            or self.model_cfg.get("task_name")
            or "Perform the instructed manipulation task."
        )
        self.camera_map = dict(
            self.model_cfg.get(
                "camera_map",
                {"image": "primary", "wrist_image": "wrist"},
            )
        )
        self.state_keys = tuple(
            self.model_cfg.get("state_keys", ["arm_joint_state", "ee_joint_state"])
        )
        if not self.state_keys or not all(isinstance(key, str) and key for key in self.state_keys):
            raise ValueError("Isaac state_keys must be a non-empty list of strings")
        self.action_dim = int(report["action_dim"])
        self.action_horizon = int(report["action_horizon"])
        self.target_fps = float(report["target_fps"])
        self._observation: dict[str, np.ndarray] | None = None
        self._instruction = self.default_prompt
        self._next_timestep = 0
        self._load_official_policy()

    def _load_official_policy(self) -> None:
        lerobot_src = Path(str(self.deployment["lerobot_root"])) / "src"
        if str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference
        from lerobot.policies.perceptron_isaac.modeling_perceptron_isaac import (
            PerceptronIsaacPolicy,
        )

        policy_path = str(self.deployment["policy_path"])
        config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
        config.device = self.device
        self.model = PerceptronIsaacPolicy.from_pretrained(
            policy_path,
            config=config,
            local_files_only=True,
        )
        self.model.to(self.device)
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._torch = torch
        self._prepare_observation_for_inference = prepare_observation_for_inference

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        prepared, instruction, timestep = encode_observation(
            obs,
            camera_map=self.camera_map,
            state_keys=self.state_keys,
            proprio_dim=int(self.deployment["proprio_dim"]),
            default_prompt=self.default_prompt,
        )
        self._observation = prepared
        self._instruction = instruction
        if timestep is not None:
            self._next_timestep = timestep

    def update_obs_batch(self, obs_list: Sequence[Mapping[str, Any]]) -> None:
        if len(obs_list) != 1:
            raise ValueError("Isaac 0.5 XPolicy integration currently supports batch size 1")
        self.update_obs(obs_list[0])

    def get_action(self, **_: Any) -> list[dict[str, np.ndarray]]:
        if self._observation is None:
            raise RuntimeError("update_obs must be called before get_action")
        observation = self._prepare_observation_for_inference(
            dict(self._observation),
            self._torch.device(self.device),
            self._instruction,
            str(self.deployment["robot_type"]),
        )
        with self._torch.inference_mode():
            observation = self.preprocessor(observation)
            adapt = getattr(self.model, "adapt_async_observation", None)
            if callable(adapt):
                observation = adapt(
                    observation,
                    timestep=self._next_timestep,
                    fps=self.target_fps,
                )
            normalized = self.model.predict_action_chunk(observation)
            normalized = normalized[:, : self.action_horizon, :]
            actions = self.postprocessor(normalized)
        array = self._torch.as_tensor(actions).squeeze(0).detach().cpu().numpy().copy()
        self._next_timestep += self.action_horizon
        return decode_actions(array, action_dim=self.action_dim, horizon=self.action_horizon)

    def get_action_batch(self, env_idx_list=None, **_: Any):
        indices = list(env_idx_list or [0])
        if indices != [0]:
            raise ValueError("Isaac 0.5 XPolicy integration currently supports env index 0 only")
        return [self.get_action()]

    def sampling_modes(self) -> list[str]:
        return ["default"]

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "isaac_05",
            "checkpoint_variant": self.model_cfg.get("checkpoint_variant", "isaac_05_base"),
            "checkpoint_source": self.model_cfg.get(
                "checkpoint_source", "PerceptronAI/Isaac-0.5"
            ),
            "model_root": self.deployment["model_root"],
            "lerobot_revision": self.deployment["lerobot_revision"],
            "policy_state_dataset": self.deployment["policy_state_dataset"],
            "robot_type": self.deployment["robot_type"],
            "control_mode": self.deployment["control_mode"],
            "action_semantics": "checkpoint_native_absolute_ee",
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "model_chunk_size": self.deployment["model_chunk_size"],
            "target_fps": self.target_fps,
        }

    def reset(self) -> None:
        self._observation = None
        self._instruction = self.default_prompt
        self._next_timestep = 0
        if self.model is not None:
            self.model.reset()
        if hasattr(self, "preprocessor"):
            self.preprocessor.reset()
        if hasattr(self, "postprocessor"):
            self.postprocessor.reset()
