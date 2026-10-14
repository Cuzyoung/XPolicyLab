import inspect
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_DIR = Path(__file__).resolve().parent
FASTWAM_ROOT = POLICY_DIR / "FastWAM"
FASTWAM_SRC = FASTWAM_ROOT / "src"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _standardize_rgb(image: np.ndarray, *, resize: bool = True) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if resize and image.shape[:2] != (240, 320):
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if resize and image.shape != (240, 320, 3):
        raise ValueError(f"Expected standardized RGB shape (240, 320, 3), got {image.shape}")
    return image


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else fallback
    if value is None:
        return fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    return text if text else fallback


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg["action_type"]
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.action_horizon = int(self.model_cfg.get("action_horizon") or 32)
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 24)
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "follow the instruction"
        )
        self.robot_action_dim_info = get_robot_action_dim_info(self.model_cfg)
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self.model = None
        self._batch_obs = {}
        self._batch_instruction = {}
        self._sampling_rng = np.random.default_rng(int(self.model_cfg.get("inference_seed") or 0))

        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        dataset_stats_path = self.model_cfg.get("dataset_stats_path")

        if self.allow_dummy_policy:
            print("[FastWAM] allow_dummy_policy=true; real checkpoint loading is skipped for debug flow only.")
            return

        checkpoint_path = resolve_checkpoint_root(
            self.model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR,
            explicit_keys=("checkpoint_path", "ckpt_setting"),
        )
        if _is_none_like(dataset_stats_path):
            raise FileNotFoundError("FastWAM requires dataset_stats_path for real deployment.")

        for path in (str(FASTWAM_ROOT), str(FASTWAM_SRC)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from experiments.robotwin.fastwam_policy.deploy_policy import get_model

        upstream_cfg = dict(self.model_cfg)
        upstream_cfg["ckpt_setting"] = str(Path(checkpoint_path).expanduser().resolve())
        upstream_cfg["dataset_stats_path"] = str(Path(dataset_stats_path).expanduser().resolve())
        upstream_cfg.setdefault("sim_cfg_name", "sim_robotwin.yaml")
        upstream_cfg.setdefault("sim_task", "robotwin_uncond_3cam_384_1e-4")
        self.model = get_model(upstream_cfg)
        self.action_horizon = int(self.model.action_horizon)
        self.replan_steps = int(self.model.replan_steps)
        dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        if self.action_type != "joint" or int(self.model.model.action_expert.action_dim) != dim:
            raise ValueError("FastWAM deployment requires matching native joint-action dimensions")

    def _encode_obs_for_fastwam(self, obs: dict) -> dict:
        vision = obs["vision"]
        resize = _is_none_like(self.model_cfg.get("training_config_path"))
        adapted = {
            "observation": {
                "head_camera": {"rgb": _standardize_rgb(vision["cam_head"]["color"], resize=resize)},
                "left_camera": {"rgb": _standardize_rgb(vision["cam_left_wrist"]["color"], resize=resize)},
                "right_camera": {"rgb": _standardize_rgb(vision["cam_right_wrist"]["color"], resize=resize)},
            },
            "joint_action": {
                "vector": pack_robot_state(
                    obs,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                    state_type="state",
                ).astype(np.float32)
            },
        }
        return adapted

    def update_obs(self, obs):
        self._batch_obs = {}
        self._batch_instruction = {}
        self.last_obs = self._encode_obs_for_fastwam(obs)
        self.last_instruction = _get_instruction(obs, self.default_instruction)

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        self._batch_obs = {}
        self._batch_instruction = {}
        self.last_obs = None
        for obs in obs_list:
            env_idx = int(obs["env_idx"])
            self._batch_obs[env_idx] = self._encode_obs_for_fastwam(obs)
            self._batch_instruction[env_idx] = _get_instruction(obs, self.default_instruction)
        if len(self._batch_obs) == 1:
            env_idx = next(iter(self._batch_obs))
            self.last_obs = self._batch_obs[env_idx]
            self.last_instruction = self._batch_instruction[env_idx]

    def _zero_actions(self):
        dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        zeros = np.zeros((self.action_horizon, dim), dtype=np.float32)
        return unpack_robot_state(zeros, self.action_type, self.robot_action_dim_info, source_type="obs")

    def _infer_actions(self, obs, instruction, **sampling_kwargs):
        if self.allow_dummy_policy:
            return self._zero_actions()
        if obs is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        action_chunk = self.model._infer_action_chunk(obs, instruction, **sampling_kwargs)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        if action_chunk.shape != (self.action_horizon, dim) or not np.isfinite(action_chunk).all():
            raise ValueError(f"FastWAM must return finite full action chunk {(self.action_horizon, dim)}")
        return unpack_robot_state(action_chunk, self.action_type, self.robot_action_dim_info, source_type="obs")

    def get_action(self):
        return self._infer_actions(self.last_obs, self.last_instruction)

    def get_action_batch(self, env_idx_list=None):
        if not self._batch_obs:
            raise ValueError("No batch observation is available. Call update_obs_batch() first.")
        if env_idx_list is None:
            env_idx_list = list(self._batch_obs)
        return [
            self._infer_actions(self._batch_obs[int(env_idx)], self._batch_instruction[int(env_idx)])
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self._batch_obs = {}
        self._batch_instruction = {}
        self._sampling_rng = np.random.default_rng(int(self.model_cfg.get("inference_seed") or 0))
        if self.model is not None:
            self.model.reset()

    def sampling_modes(self):
        if self.allow_dummy_policy or self.model is None or self.action_type != "joint":
            return ["default"]
        native = self.model.model
        if "sampling_seed" not in inspect.signature(self.model._infer_action_chunk).parameters:
            return ["default"]
        modes = ["default", "aac"]
        processor = self.model.processor
        if ("denoise_loop" not in inspect.signature(native.infer_action).parameters
                or float(self.model.text_cfg_scale) != 1.0
                or processor.action_state_transforms
                or processor.delta_action_dim_mask is not None
                or len(processor.shape_meta["action"]) != 1):
            return modes
        modes += ["rtc", "paint"]
        if (getattr(native.mot, "supports_action_attention_capture", False)
                and self.model.num_inference_steps >= 3 and self.action_horizon >= 2):
            modes.append("autohorizon")
        return modes

    def _require_sampling(self, mode):
        if mode not in self.sampling_modes():
            raise ValueError(f"FastWAM {mode} is unavailable for the loaded backend")
        if self.last_obs is None or len(self._batch_obs) > 1:
            raise ValueError("FastWAM specialized sampling requires exactly one current observation")

    def _next_sampling_seed(self):
        return int(self._sampling_rng.integers(0, 2**31 - 1))

    def get_action_aac(self, sampling):
        from XPolicyLab.utils.flow_sampling import validate_samples

        self._require_sampling("aac")
        count = validate_samples(sampling)
        return {"actions": [{"actions": self._infer_actions(
            self.last_obs, self.last_instruction, sampling_seed=self._next_sampling_seed()
        )} for _ in range(count)]}

    def _sample_conditioned(self, sampling, mode):
        from .sampling import ActionFlowSampler, normalize_actions
        from XPolicyLab.utils.flow_sampling import validate_prefix

        self._require_sampling(mode)
        horizon = self.action_horizon
        width = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        target = weights = delay = None
        beta = 5.0
        if mode == "paint":
            prefix, delay = validate_prefix(sampling, horizon, width)
            raw = np.zeros((horizon, width), dtype=np.float32)
            raw[:delay] = prefix
            target = normalize_actions(self.model, raw)
        elif mode == "rtc":
            if set(sampling) != {"mode", "action_condition", "condition_weights", "beta"}:
                raise ValueError("RTC requires action_condition, condition_weights and beta")
            raw = np.asarray(sampling["action_condition"], dtype=np.float32)
            if raw.shape != (horizon, width) or not np.isfinite(raw).all():
                raise ValueError(f"FastWAM RTC condition must be finite raw joint actions [{horizon},{width}]")
            target = normalize_actions(self.model, raw)
            weights, beta = sampling["condition_weights"], float(sampling["beta"])
        elif set(sampling) != {"mode"}:
            raise ValueError("AutoHorizon accepts no overrides")
        sampler = ActionFlowSampler(mode, target=target, weights=weights, beta=beta, delay=delay)
        actions = self._infer_actions(
            self.last_obs, self.last_instruction, denoise_loop=sampler,
            sampling_seed=self._next_sampling_seed(),
        )
        result = {"actions": actions}
        if mode != "rtc":
            result[mode] = sampler.metadata
        return result

    def get_action_rtc(self, sampling):
        return self._sample_conditioned(sampling, "rtc")

    def get_action_paint(self, sampling):
        return self._sample_conditioned(sampling, "paint")

    def get_action_autohorizon(self, sampling):
        return self._sample_conditioned(sampling, "autohorizon")

    def runtime_metadata(self):
        return {
            "action_horizon": self.action_horizon,
            "action_semantics": "absolute_joint_position",
            "conditioned_sampler": "native_action_flow_v1",
            "sampling_seed": int(self.model_cfg.get("inference_seed") or 0),
            "training_config_path": self.model_cfg.get("training_config_path"),
            "dummy_policy": self.allow_dummy_policy,
        }
