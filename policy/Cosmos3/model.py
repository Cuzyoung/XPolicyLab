"""Thin XPolicyLab contract around NVIDIA's official Cosmos 3 policy service.

The adapter translates observation and action field names only. Model loading,
prompt formatting, image composition, normalization, sampling, denoising and
action post-processing stay inside the pinned official ``cosmos-framework``
implementation.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info

POLICY_DIR = Path(__file__).resolve().parent
FRAMEWORK_ROOT = POLICY_DIR / "cosmos-framework"
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"
OFFICIAL_SERVER_MODULE = "cosmos_framework.scripts.action_policy_server_robolab"
OFFICIAL_FRAMEWORK_REVISION = "c7e8d76b5da8aeae38cdac91c6cfd57185b2f6bc"
EXPECTED_ARM_DIM = 7
EXPECTED_GRIPPER_DIM = 1

_OFFICIAL_ARG_KEYS = {
    "checkpoint_path",
    "hf_revision",
    "allow_dcp_checkpoint",
    "experiment",
    "experiment_overrides",
    "credential_path",
    "domain_name",
    "decode_video",
    "output_dir",
    "sampler",
    "seed",
    "deterministic_seed",
    "guidance",
    "guidance_interval",
    "num_steps",
    "shift",
    "resolution",
    "conditioning_fps",
    "action_chunk_size",
    "action_dim",
    "image_height",
    "image_width",
    "action_space",
    "use_state",
    "history_length",
    "format_prompt_as_json",
}


def resolve_cosmos_checkpoint(model_cfg: Mapping[str, Any]) -> str:
    explicit = model_cfg.get("checkpoint_path")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return str(resolve_checkpoint_root(model_cfg, CHECKPOINTS_DIR))


def _camera_image(obs: Mapping[str, Any], camera_name: str) -> np.ndarray:
    try:
        image = np.asarray(obs["vision"][camera_name]["color"])
    except KeyError as exc:
        raise KeyError(f"missing RGB camera: vision.{camera_name}.color") from exc
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"vision.{camera_name}.color must have shape [H,W,3], got {image.shape}"
        )
    return image


def encode_observation(
    obs: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    robot_action_dim_info: Mapping[str, list[int]],
) -> dict[str, Any]:
    if robot_action_dim_info != {
        "arm_dim": [EXPECTED_ARM_DIM],
        "ee_dim": [EXPECTED_GRIPPER_DIM],
    }:
        raise ValueError(
            "Cosmos3 DROID checkpoints require one 7-DoF arm and one 1-DoF gripper"
        )

    state = obs.get("state")
    if not isinstance(state, Mapping):
        raise KeyError("observation must contain a state mapping")
    joint_position = np.asarray(state.get("arm_joint_state"))
    gripper_position = np.asarray(state.get("ee_joint_state"))
    if joint_position.shape[-1:] != (EXPECTED_ARM_DIM,):
        raise ValueError(
            f"state.arm_joint_state must end in 7 values, got {joint_position.shape}"
        )
    if gripper_position.shape[-1:] != (EXPECTED_GRIPPER_DIM,):
        raise ValueError(
            f"state.ee_joint_state must end in 1 value, got {gripper_position.shape}"
        )

    instruction = obs.get("instruction", obs.get("instructions"))
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = model_cfg.get("default_prompt")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Cosmos3 requires a non-empty instruction string")

    wrist_camera = str(model_cfg.get("wrist_camera", "cam_left_wrist"))
    exterior_camera_1 = str(model_cfg.get("exterior_camera_1", "cam_head"))
    exterior_camera_2 = str(model_cfg.get("exterior_camera_2", "cam_right_wrist"))
    return {
        "prompt": instruction,
        "observation/wrist_image_left": _camera_image(obs, wrist_camera),
        "observation/exterior_image_1_left": _camera_image(obs, exterior_camera_1),
        "observation/exterior_image_2_left": _camera_image(obs, exterior_camera_2),
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
    }


def decode_action(
    output: Mapping[str, Any],
    robot_action_dim_info: Mapping[str, list[int]],
) -> list[dict[str, np.ndarray]]:
    if "action" not in output:
        raise KeyError("official Cosmos3 output is missing 'action'")
    action = np.asarray(output["action"])
    expected_dim = sum(robot_action_dim_info["arm_dim"]) + sum(
        robot_action_dim_info["ee_dim"]
    )
    if action.ndim != 2 or action.shape[1] != expected_dim:
        raise ValueError(
            f"official Cosmos3 action must have shape [H,{expected_dim}], got {action.shape}"
        )
    if not np.isfinite(action).all():
        raise ValueError("official Cosmos3 action contains non-finite values")
    arm_dim = robot_action_dim_info["arm_dim"][0]
    return [
        {
            "arm_joint_state": step[:arm_dim].copy(),
            "ee_joint_state": step[arm_dim:].copy(),
        }
        for step in action
    ]


def create_official_service(model_cfg: Mapping[str, Any]) -> Any:
    if not (FRAMEWORK_ROOT / "cosmos_framework").is_dir():
        raise FileNotFoundError(
            f"official cosmos-framework source is missing: {FRAMEWORK_ROOT}; "
            "run git submodule update --init --recursive"
        )
    framework_path = str(FRAMEWORK_ROOT)
    if framework_path not in sys.path:
        sys.path.insert(0, framework_path)
    official = importlib.import_module(OFFICIAL_SERVER_MODULE)
    official_args = {
        key: value
        for key, value in model_cfg.items()
        if key in _OFFICIAL_ARG_KEYS and value is not None
    }
    official_args["checkpoint_path"] = resolve_cosmos_checkpoint(model_cfg)
    args = official.RobolabServerArgs.model_validate(official_args)
    return official.RobolabPolicyService(args)


class Model(ModelTemplate):
    def __init__(self, model_cfg: Mapping[str, Any]) -> None:
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type")
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.action_type != "joint":
            raise ValueError("Cosmos3 DROID checkpoints support action_type='joint' only")
        if not isinstance(self.env_cfg_type, str) or not self.env_cfg_type:
            raise ValueError("env_cfg_type is required")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if self.robot_action_dim_info != {
            "arm_dim": [EXPECTED_ARM_DIM],
            "ee_dim": [EXPECTED_GRIPPER_DIM],
        }:
            raise ValueError(
                f"env_cfg_type={self.env_cfg_type!r} is incompatible with the official "
                "Cosmos3 DROID 7+1 joint action contract"
            )
        self.batch_size = get_batch_size(self.env_cfg_type)
        self._service = create_official_service(self.model_cfg)
        self._obs: Mapping[str, Any] | None = None
        self._obs_batch: list[Mapping[str, Any]] = []

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self._obs = obs

    def update_obs_batch(self, obs_list: list[Mapping[str, Any]]) -> None:
        self._obs_batch = list(obs_list)

    def _infer(self, obs: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
        native_obs = encode_observation(
            obs,
            self.model_cfg,
            self.robot_action_dim_info,
        )
        return decode_action(
            self._service.infer(native_obs),
            self.robot_action_dim_info,
        )

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._obs is None:
            raise RuntimeError("update_obs must be called before get_action")
        return self._infer(self._obs)

    def get_action_batch(
        self, env_idx_list: list[int] | None = None
    ) -> list[list[dict[str, np.ndarray]]]:
        expected = len(env_idx_list) if env_idx_list is not None else self.batch_size
        if len(self._obs_batch) != expected:
            raise ValueError(
                f"observation batch has {len(self._obs_batch)} items, expected {expected}"
            )
        return [self._infer(obs) for obs in self._obs_batch]

    def reset(self) -> None:
        self._obs = None
        self._obs_batch = []
