from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state


POLICY_DIR = Path(__file__).resolve().parent
UPSTREAM_ROOT = POLICY_DIR / "OpenLoopVLA"
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"

# RoboTwin/XPolicyLab packs dual-arm state as L6,Lgrip,R6,Rgrip. OpenLoopVLA
# was trained with L6,R6,Lgrip,Rgrip. Apply each permutation exactly once.
ENV_TO_TRAIN = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13])
TRAIN_TO_ENV = np.argsort(ENV_TO_TRAIN)
EXPECTED_KEYS = ["left_joints", "right_joints", "left_gripper", "right_gripper"]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _instruction(obs: dict, fallback: str) -> str:
    value = obs.get("instruction")
    if value is None:
        value = obs.get("task_instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _rgb(image: Any, camera: str) -> np.ndarray:
    array = np.asarray(image)
    if array.shape != (240, 320, 3) or array.dtype != np.uint8:
        raise ValueError(
            f"{camera} must be current-frame uint8 RGB with shape (240, 320, 3), "
            f"got shape={array.shape}, dtype={array.dtype}"
        )
    return array


def _resolve_checkpoint(model_cfg: dict) -> Path:
    path = resolve_checkpoint_root(
        model_cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
    )
    if path.is_file():
        checkpoint = path
    else:
        candidates = sorted((path / "checkpoints").glob("steps_*_model.pt"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one checkpoints/steps_*_model.pt under {path}, "
                f"found {len(candidates)}"
            )
        checkpoint = candidates[0]
    if checkpoint.suffix != ".pt":
        raise ValueError(f"OpenLoopVLA checkpoint must be a .pt file, got {checkpoint}")
    for required in (checkpoint.parents[1] / "config.yaml", checkpoint.parents[1] / "dataset_statistics.json"):
        if not required.is_file():
            raise FileNotFoundError(f"Required checkpoint metadata is missing: {required}")
    return checkpoint


class Model(ModelTemplate):
    """XPolicyLab model adapter for native OpenLoopVLA inference."""

    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg["action_type"])
        self.env_cfg_type = str(self.model_cfg["env_cfg_type"])
        if self.action_type != "joint":
            raise ValueError("OpenLoopVLA currently supports action_type=joint only")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if self.robot_action_dim_info != {"arm_dim": [6, 6], "ee_dim": [1, 1]}:
            raise ValueError(
                "OpenLoopVLA requires the RoboTwin dual-arm 6+1 / 6+1 joint profile; "
                f"got {self.robot_action_dim_info} for {self.env_cfg_type}"
            )

        self.execute_steps = int(self.model_cfg.get("execute_steps", 12))
        if not 1 <= self.execute_steps <= 50:
            raise ValueError("execute_steps must be in 1..50")
        self.default_instruction = str(
            self.model_cfg.get("default_instruction") or "follow the instruction"
        )
        self.sampling_seed = int(self.model_cfg.get("sampling_seed", 42))
        self.unnorm_key = self.model_cfg.get("unnorm_key") or None

        checkpoint = _resolve_checkpoint(self.model_cfg)
        v2_package = self.model_cfg.get("v2_package_root") or os.environ.get(
            "OPENLOOPVLA_V2_PACKAGE_ROOT"
        )
        if not v2_package:
            raise FileNotFoundError(
                "Set v2_package_root in deploy.yml or OPENLOOPVLA_V2_PACKAGE_ROOT"
            )
        v2_package = Path(str(v2_package)).expanduser().resolve()
        if not v2_package.is_dir():
            raise FileNotFoundError(f"V2 package directory does not exist: {v2_package}")
        if not UPSTREAM_ROOT.is_dir():
            raise FileNotFoundError(
                f"OpenLoopVLA source submodule is missing: {UPSTREAM_ROOT}; "
                "run git submodule update --init policy/OpenLoopVLA/OpenLoopVLA"
            )

        upstream_text = str(UPSTREAM_ROOT)
        if upstream_text not in sys.path:
            sys.path.insert(0, upstream_text)

        from deployment.model_server.policy_wrapper import PolicyServerWrapper

        self.wrapper = PolicyServerWrapper(
            str(checkpoint),
            device=str(self.model_cfg.get("device") or "cuda"),
            use_bf16=_as_bool(self.model_cfg.get("use_bf16", True)),
            unnorm_key=self.unnorm_key,
            mmap_checkpoint=_as_bool(self.model_cfg.get("mmap_checkpoint", True)),
            config_overrides=[f"framework.v2_package_root={v2_package}"],
        )
        metadata = self.wrapper.metadata
        if metadata.get("framework") != "OpenLoopVLA":
            raise ValueError(f"Loaded checkpoint is not OpenLoopVLA: {metadata.get('framework')}")
        if int(metadata.get("action_chunk_size", 0)) != 50:
            raise ValueError("OpenLoopVLA checkpoint must predict 50 actions")
        if metadata.get("state_normalization") != "server":
            raise ValueError("OpenLoopVLA must normalize raw state through its training transform")
        for modality in ("state", "action"):
            expected = [f"{modality}.{key}" for key in EXPECTED_KEYS]
            if metadata.get(f"{modality}_keys") != expected:
                raise ValueError(
                    f"Unexpected {modality} order: {metadata.get(f'{modality}_keys')}"
                )
        if self.unnorm_key is None:
            self.unnorm_key = metadata.get("default_unnorm_key")

        self.last_example: dict | None = None
        self._batch_examples: dict[int, dict] = {}
        self.reset()

    def _encode_obs(self, obs: dict) -> dict:
        vision = obs["vision"]
        images = [
            _rgb(vision["cam_head"]["color"], "cam_head"),
            _rgb(vision["cam_left_wrist"]["color"], "cam_left_wrist"),
            _rgb(vision["cam_right_wrist"]["color"], "cam_right_wrist"),
        ]
        state_env = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32, copy=False)
        if state_env.shape != (14,) or not np.isfinite(state_env).all():
            raise ValueError(f"Expected finite 14D state, got {state_env.shape}")
        return {
            "image": images,
            "lang": _instruction(obs, self.default_instruction),
            "state": state_env[ENV_TO_TRAIN],
        }

    def update_obs(self, obs):
        self.last_example = self._encode_obs(obs)

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list")
        self._batch_examples = {
            int(obs["env_idx"]): self._encode_obs(obs) for obs in obs_list
        }

    def _unpack_actions(self, actions: np.ndarray) -> list[dict]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite denormalized actions [50,14], got {actions.shape}")
        selected_env_order = actions[: self.execute_steps, TRAIN_TO_ENV]
        return unpack_robot_state(
            selected_env_order,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )

    def _predict(self, examples: list[dict]) -> list[list[dict]]:
        result = self.wrapper.predict_action(examples=examples, unnorm_key=self.unnorm_key)
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected = (len(examples), 50, 14)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite denormalized actions {expected}, got {actions.shape}")
        return [self._unpack_actions(chunk) for chunk in actions]

    def get_action(self):
        if self.last_example is None:
            raise ValueError("Call update_obs() before get_action()")
        return self._predict([self.last_example])[0]

    def get_action_batch(self, env_idx_list=None):
        if not self._batch_examples:
            raise ValueError("Call update_obs_batch() before get_action_batch()")
        indices = (
            sorted(self._batch_examples)
            if env_idx_list is None
            else [int(index) for index in env_idx_list]
        )
        return self._predict([self._batch_examples[index] for index in indices])

    def reset(self):
        self.last_example = None
        self._batch_examples = {}
        random.seed(self.sampling_seed)
        np.random.seed(self.sampling_seed)
        try:
            import torch

            torch.manual_seed(self.sampling_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.sampling_seed)
        except ImportError:
            pass

