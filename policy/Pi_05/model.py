#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next(
        (candidate for candidate in candidates if candidate.exists()),
        candidates[0],
    )
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [
        candidate
        for candidate in candidate_dirs
        if _extract_step_number(candidate.name) is not None
    ]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


def _resolve_train_config(model_cfg: dict[str, Any]):
    config = _config.get_config(model_cfg.get("train_config_name", "pi05_aloha"))
    data_updates: dict[str, Any] = {}
    for key in ("repo_id", "use_delta_joint_actions", "adapt_to_pi"):
        if key in model_cfg:
            data_updates[key] = model_cfg[key]

    norm_stats_path = model_cfg.get("norm_stats_path")
    if norm_stats_path is not None:
        stats_dir = Path(str(norm_stats_path)).expanduser().resolve()
        data_updates["assets"] = _config.AssetsConfig(
            assets_dir=str(stats_dir.parent),
            asset_id=stats_dir.name,
        )

    model_updates: dict[str, Any] = {}
    if "action_horizon" in model_cfg:
        action_horizon = int(model_cfg["action_horizon"])
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        model_updates["action_horizon"] = action_horizon

    replacements: dict[str, Any] = {}
    if data_updates:
        replacements["data"] = dataclasses.replace(config.data, **data_updates)
    if model_updates:
        replacements["model"] = dataclasses.replace(config.model, **model_updates)
    return dataclasses.replace(config, **replacements) if replacements else config


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        env_cfg_type = model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(env_cfg_type) if env_cfg_type is not None else None
        )
        self.robot_action_dim = (
            sum(self.robot_action_dim_info["arm_dim"])
            + sum(self.robot_action_dim_info["ee_dim"])
            if self.robot_action_dim_info is not None
            else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]
        self._dvac_history: deque[np.ndarray] = deque()
        self._dvac_history_size: int | None = None

        self._train_config = _resolve_train_config(model_cfg)
        expected_profile = (
            "yam_native"
            if type(self._train_config.data).__name__ == "LeRobotYamDataConfig"
            else "aloha"
        )
        self.observation_profile = model_cfg.get("observation_profile", expected_profile)
        if self.observation_profile != expected_profile:
            raise ValueError(
                f"train_config_name={self._train_config.name!r} requires "
                f"observation_profile={expected_profile!r}, got {self.observation_profile!r}"
            )
        self.action_horizon = int(self._train_config.model.action_horizon)
        self.action_dim = int(self._train_config.model.action_dim)
        self.num_steps = int(model_cfg.get("num_steps", 10))
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)

        norm_stats = None
        norm_stats_path = model_cfg.get("norm_stats_path")
        if norm_stats_path is not None:
            norm_stats = _normalize.load(Path(str(norm_stats_path)).expanduser().resolve())
        elif repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        return _policy_config.create_trained_policy(
            self._train_config,
            str(model_root),
            norm_stats=norm_stats,
        )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [
            obs.get("env_idx", index) for index, obs in enumerate(obs_list)
        ]
        encoded_obs_list = [
            encode_obs(
                obs,
                self.action_type,
                self.robot_action_dim_info,
                observation_profile=self.observation_profile,
            )
            for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        kwargs.setdefault("num_steps", self.num_steps)
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_rtc(self, sampling: dict[str, Any]):
        required = {"action_condition", "condition_weights", "beta"}
        missing = sorted(required - set(sampling))
        if missing:
            raise ValueError(f"RTC sampling is missing fields: {missing}")

        condition = np.array(
            sampling["action_condition"], dtype=np.float32, copy=True
        )
        weights = np.array(
            sampling["condition_weights"], dtype=np.float32, copy=True
        )
        beta = float(sampling["beta"])
        expected_action_dim = self.robot_action_dim or self.action_dim
        if condition.shape != (self.action_horizon, expected_action_dim):
            raise ValueError(
                "action_condition must have shape "
                f"{(self.action_horizon, expected_action_dim)}, got {condition.shape}"
            )
        if weights.shape != (self.action_horizon,):
            raise ValueError(
                f"condition_weights must have shape {(self.action_horizon,)}, got {weights.shape}"
            )
        if not np.isfinite(condition).all() or not np.isfinite(weights).all():
            raise ValueError("RTC sampling arrays must be finite")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"RTC beta must be finite and positive, got {beta}")

        return self.get_action(
            action_condition=condition,
            condition_weights=weights,
            rtc_beta=beta,
        )

    def get_action_paint(self, sampling: dict[str, Any]):
        required = {"action_prefix", "delay_steps"}
        missing = sorted(required - set(sampling))
        if missing:
            raise ValueError(f"PAINT sampling is missing fields: {missing}")

        delay_steps = int(sampling["delay_steps"])
        expected_action_dim = self.robot_action_dim or self.action_dim
        prefix = np.asarray(sampling["action_prefix"], dtype=np.float32)
        if not 0 < delay_steps < self.action_horizon:
            raise ValueError(
                "PAINT delay_steps must satisfy "
                f"0 < d < {self.action_horizon}, got {delay_steps}"
            )
        if prefix.shape != (delay_steps, expected_action_dim):
            raise ValueError(
                "PAINT action_prefix must have shape "
                f"{(delay_steps, expected_action_dim)}, got {prefix.shape}"
            )
        if not np.isfinite(prefix).all():
            raise ValueError("PAINT action_prefix must be finite")

        condition = np.zeros(
            (self.action_horizon, expected_action_dim),
            dtype=np.float32,
        )
        condition[:delay_steps] = prefix
        actions = self.get_action(
            paint_action_condition=condition,
            paint_delay_steps=delay_steps,
        )
        return {
            "actions": actions,
            "paint": {
                "delay_steps": delay_steps,
                "num_steps": self.num_steps,
                "model_evaluations": 3 * self.num_steps,
                "inversion": "backward_euler",
            },
        }

    def get_action_aac(self, sampling: dict[str, Any]):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        if len(self._latest_env_idx_list) != 1:
            raise ValueError("AAC sampling requires exactly one observation")

        num_samples = int(sampling.get("num_samples", 20))
        if num_samples <= 1:
            raise ValueError("AAC num_samples must be greater than one")

        single_observation = slice_stacked_obs(self.observation_window, 0)
        actions = np.asarray(
            self.policy.infer(
                single_observation,
                num_steps=self.num_steps,
                num_samples=num_samples,
            )["actions"]
        )
        expected_action_dim = self.robot_action_dim or self.action_dim
        expected_shape = (num_samples, self.action_horizon, expected_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Pi05 AAC actions must have shape {expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Pi05 AAC actions must be finite")

        if self.robot_action_dim_info is None:
            candidates = [sample for sample in actions]
        else:
            candidates = [
                unpack_robot_state(
                    sample,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
                for sample in actions
            ]
        return {"actions": candidates}

    def get_action_autohorizon(self, sampling: dict[str, Any]):
        if set(sampling) - {"mode"}:
            raise ValueError("AutoHorizon does not accept sampling overrides")
        if self.num_steps < 3:
            raise ValueError("AutoHorizon requires at least three denoising steps")
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        if len(self._latest_env_idx_list) != 1:
            raise ValueError("AutoHorizon sampling requires exactly one observation")

        single_observation = slice_stacked_obs(self.observation_window, 0)
        result = self.policy.infer(
            single_observation,
            num_steps=self.num_steps,
            autohorizon=True,
        )
        actions = np.asarray(result["actions"])
        expected_action_dim = self.robot_action_dim or self.action_dim
        expected_shape = (self.action_horizon, expected_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Pi05 AutoHorizon actions must have shape {expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Pi05 AutoHorizon actions must be finite")

        metadata = result.get("autohorizon")
        if not isinstance(metadata, dict):
            raise ValueError("Pi05 AutoHorizon inference did not return metadata")
        execution_steps = metadata.get("execution_steps")
        if (
            not isinstance(execution_steps, int)
            or isinstance(execution_steps, bool)
            or not 1 <= execution_steps <= self.action_horizon
        ):
            raise ValueError(
                "Pi05 AutoHorizon execution_steps must satisfy "
                f"1 <= e <= {self.action_horizon}, got {execution_steps!r}"
            )

        if self.robot_action_dim_info is None:
            decoded = actions
        else:
            decoded = unpack_robot_state(
                actions,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
        return {"actions": decoded, "autohorizon": metadata}

    def get_action_dvac(self, sampling: dict[str, Any]):
        allowed = {
            "mode",
            "alpha",
            "min_execution_steps",
            "max_execution_steps",
            "rolling_window_size",
            "tail_steps",
        }
        unexpected = sorted(set(sampling) - allowed)
        if unexpected:
            raise ValueError(f"DVAC received unsupported fields: {unexpected}")
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        if len(self._latest_env_idx_list) != 1:
            raise ValueError("DVAC sampling requires exactly one observation")

        alpha = float(sampling.get("alpha", 2.0))
        integer_settings = {
            "min_execution_steps": sampling.get("min_execution_steps", 1),
            "max_execution_steps": sampling.get(
                "max_execution_steps", self.action_horizon
            ),
            "rolling_window_size": sampling.get("rolling_window_size", 5),
            "tail_steps": sampling.get("tail_steps", 5),
        }
        for name, value in integer_settings.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"DVAC {name} must be an integer")
        minimum = int(integer_settings["min_execution_steps"])
        maximum = int(integer_settings["max_execution_steps"])
        window_size = int(integer_settings["rolling_window_size"])
        tail_steps = int(integer_settings["tail_steps"])
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("DVAC alpha must be finite and non-negative")
        if not 1 <= minimum <= maximum <= self.action_horizon:
            raise ValueError(
                "DVAC execution bounds must satisfy "
                f"1 <= N_min <= N_max <= {self.action_horizon}"
            )
        if window_size <= 0:
            raise ValueError("DVAC rolling_window_size must be positive")
        if not 1 < tail_steps <= self.num_steps:
            raise ValueError(
                f"DVAC tail_steps must satisfy 1 < L <= {self.num_steps}"
            )

        if self._dvac_history_size != window_size:
            if self._dvac_history:
                raise ValueError("DVAC rolling_window_size cannot change within a session")
            self._dvac_history = deque(maxlen=window_size)
            self._dvac_history_size = window_size

        single_observation = slice_stacked_obs(self.observation_window, 0)
        result = self.policy.infer(
            single_observation,
            num_steps=self.num_steps,
            dvac=True,
            dvac_tail_steps=tail_steps,
            dvac_action_dim=self.robot_action_dim or self.action_dim,
        )
        actions = np.asarray(result["actions"])
        expected_action_dim = self.robot_action_dim or self.action_dim
        expected_shape = (self.action_horizon, expected_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Pi05 DVAC actions must have shape {expected_shape}, got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Pi05 DVAC actions must be finite")

        sampler_metadata = result.get("dvac")
        if not isinstance(sampler_metadata, dict):
            raise ValueError("Pi05 DVAC inference did not return metadata")
        variance = np.asarray(sampler_metadata.get("variance"), dtype=np.float64)
        if (
            variance.shape != (self.action_horizon,)
            or not np.isfinite(variance).all()
            or np.any(variance < 0)
        ):
            raise ValueError(
                "Pi05 DVAC variance must contain one finite non-negative value "
                f"per action step, got {variance.shape}"
            )

        cold_start = not self._dvac_history
        calibration = (
            variance
            if cold_start
            else np.concatenate(tuple(self._dvac_history), axis=0)
        )
        rolling_mean = float(np.mean(calibration))
        rolling_std = float(np.std(calibration, ddof=0))
        threshold = rolling_mean + alpha * rolling_std
        high_variance = np.flatnonzero(variance > threshold)
        if high_variance.size == 0:
            execution_steps = maximum
            first_crossing = None
        else:
            first_crossing = int(high_variance[0])
            execution_steps = max(minimum, first_crossing)

        self._dvac_history.append(variance.copy())
        metadata = {
            **sampler_metadata,
            "execution_steps": execution_steps,
            "first_threshold_crossing": first_crossing,
            "threshold": threshold,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "alpha": alpha,
            "min_execution_steps": minimum,
            "max_execution_steps": maximum,
            "rolling_window_size": window_size,
            "rolling_states": len(self._dvac_history),
            "cold_start": cold_start,
            "cold_start_policy": "current_variance_bootstrap",
            "method": "denoising_variance_adaptive_chunking",
            "source": "arxiv:2606.03847v1",
        }
        if self.robot_action_dim_info is None:
            decoded = actions
        else:
            decoded = unpack_robot_state(
                actions,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
        return {"actions": decoded, "dvac": metadata}

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        # actions = self.policy.infer(self.observation_window, **kwargs)["actions"]
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(single_observation, **kwargs)["actions"]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._dvac_history = deque()
        self._dvac_history_size = None

    def reset_obsrvationwindows(self):
        self.reset()


def encode_obs(
    observation,
    action_type,
    robot_action_dim_info,
    *,
    observation_profile="aloha",
):
    if observation_profile == "yam_native":
        return encode_yam_obs(observation, action_type, robot_action_dim_info)
    if observation_profile != "aloha":
        raise ValueError(f"Unsupported Pi05 observation profile: {observation_profile!r}")

    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def encode_yam_obs(observation, action_type, robot_action_dim_info):
    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding YAM observations.")

    if "observation/state" in observation:
        state = np.asarray(observation["observation/state"], dtype=np.float32)
    elif "state" in observation and not isinstance(observation["state"], dict):
        state = np.asarray(observation["state"], dtype=np.float32)
    else:
        state = pack_robot_state(
            observation,
            action_type,
            robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)

    def raw_image(key, candidates):
        if key in observation:
            return np.asarray(observation[key])
        return np.asarray(extract_image(observation, candidates))

    return {
        "observation/image": raw_image(
            "observation/image",
            ["cam_high", "cam_head", "head_camera", "top_camera"],
        ),
        "observation/left_wrist": raw_image(
            "observation/left_wrist",
            ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
        ),
        "observation/right_wrist": raw_image(
            "observation/right_wrist",
            ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
        ),
        "observation/state": state,
        "prompt": observation.get("instruction", observation.get("prompt")),
    }


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    if "observation/state" in obs_list[0]:
        return {
            "observation/state": np.stack(
                [obs["observation/state"] for obs in obs_list], axis=0
            ),
            "observation/image": np.stack(
                [obs["observation/image"] for obs in obs_list], axis=0
            ),
            "observation/left_wrist": np.stack(
                [obs["observation/left_wrist"] for obs in obs_list], axis=0
            ),
            "observation/right_wrist": np.stack(
                [obs["observation/right_wrist"] for obs in obs_list], axis=0
            ),
            "prompt": [obs["prompt"] for obs in obs_list],
        }
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    if "observation/state" in obs:
        return {
            "observation/state": obs["observation/state"][batch_index],
            "observation/image": obs["observation/image"][batch_index],
            "observation/left_wrist": obs["observation/left_wrist"][batch_index],
            "observation/right_wrist": obs["observation/right_wrist"][batch_index],
            "prompt": obs["prompt"][batch_index],
        }
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", observation.get("images", {}))
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
