"""XPolicyLab adapter for PatchWAM on RoboTwin 2.0."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = POLICY_DIR / "PatchWAM"
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _instruction(obs: dict, fallback: str) -> str:
    value = obs.get("instruction", obs.get("instructions", obs.get("task_instruction")))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    text = str(value or fallback).strip()
    return text or fallback


def _rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.5 else 1.0
        array = array * scale
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type") or "joint"
        if self.action_type != "joint":
            raise ValueError(f"PatchWAM C2R supports action_type='joint', got {self.action_type!r}")
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 16)
        self.default_instruction = str(self.model_cfg.get("default_instruction") or "follow the instruction")
        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        self._obs: dict[int, dict] = {}
        self._instructions: dict[int, str] = {}
        self._order: list[int] = []
        self.policy = None

        if self.allow_dummy_policy:
            print("[PatchWAM] dummy policy enabled for protocol-only debug")
            return

        checkpoint = resolve_checkpoint_root(
            self.model_cfg,
            CHECKPOINTS_DIR,
            policy_dir=POLICY_DIR,
        )
        if checkpoint.is_dir():
            candidates = sorted(checkpoint.glob("step_*.pt"))
            if not candidates:
                candidates = sorted((checkpoint / "checkpoints" / "weights").glob("step_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No step_*.pt found under {checkpoint}")
            checkpoint = candidates[-1]

        required = {
            "flux2_model_path": self.model_cfg.get("flux2_model_path"),
            "ae_model_path": self.model_cfg.get("ae_model_path"),
            "dataset_stats_path": self.model_cfg.get("dataset_stats_path"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing PatchWAM model assets: {', '.join(missing)}")

        for path in (UPSTREAM_DIR, UPSTREAM_DIR / "src"):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
        from runtime_policy import PatchWAMPolicy

        runtime_cfg = dict(self.model_cfg)
        runtime_cfg["checkpoint_path"] = str(checkpoint)
        runtime_cfg.setdefault("runtime_config_path", str(POLICY_DIR / "assets" / "c2r_dr4_resolved.yaml"))
        self.policy = PatchWAMPolicy(runtime_cfg)
        print(f"[PatchWAM] loaded {checkpoint}")

    def _encode_obs(self, obs: dict) -> dict:
        vision = obs["vision"]
        packed = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)
        return {
            "observation": {
                "head_camera": {"rgb": _rgb(vision["cam_head"]["color"])},
                "left_camera": {"rgb": _rgb(vision["cam_left_wrist"]["color"])},
                "right_camera": {"rgb": _rgb(vision["cam_right_wrist"]["color"])},
            },
            "joint_action": {"vector": packed},
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        if not obs_list:
            raise ValueError("update_obs_batch received no observations")
        self._obs = {}
        self._instructions = {}
        self._order = []
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            self._obs[env_idx] = self._encode_obs(obs)
            self._instructions[env_idx] = _instruction(obs, self.default_instruction)
            self._order.append(env_idx)

    def _zero_actions(self) -> list[dict]:
        dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        chunk = np.zeros((self.replan_steps, dim), dtype=np.float32)
        return unpack_robot_state(chunk, self.action_type, self.robot_action_dim_info, source_type="obs")

    def _infer(self, env_idx: int) -> list[dict]:
        if self.allow_dummy_policy:
            return self._zero_actions()
        if env_idx not in self._obs:
            raise ValueError("No observation available; call update_obs first")
        action = self.policy.infer(self._obs[env_idx], self._instructions[env_idx])
        return unpack_robot_state(action, self.action_type, self.robot_action_dim_info, source_type="obs")

    def get_action(self):
        env_idx = self._order[0] if self._order else 0
        return self._infer(env_idx)

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = self._order if env_idx_list is None else [int(i) for i in env_idx_list]
        return [self._infer(env_idx) for env_idx in env_idx_list]

    def reset(self):
        self._obs = {}
        self._instructions = {}
        self._order = []
