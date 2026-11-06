"""Pack-plate native joint checkpoints: measured 16-D state, absolute targets.

Training order is left arm (7 radians), left opening, right arm, right opening.
No pose transform, ALOHA sign conversion, delta action or IK is involved.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info, pack_robot_state, unpack_robot_state,
)
from XPolicyLab.utils.sampling_contracts import validate_rtc

from .pack_plate_contract import validate_artifacts


@dataclasses.dataclass(frozen=True)
class JointInputs:
    camera_mode: str

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("pack-plate joint state must be finite (16,)")
        images = data["images"]
        use_main = self.camera_mode == "wrist_main"
        image = {
            "base_0_rgb": images["main"] if use_main else np.zeros_like(images["left_wrist"]),
            "left_wrist_0_rgb": images["left_wrist"],
            "right_wrist_0_rgb": images["right_wrist"],
        }
        for frame in image.values():
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError("pack-plate joint images must be HWC uint8 RGB")
        result = {
            "state": state,
            "image": image,
            "image_mask": {
                "base_0_rgb": np.bool_(use_main),
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": data["prompt"],
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        return result


@dataclasses.dataclass(frozen=True)
class JointOutputs:
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :16]}


def make_joint_train_config(model_cfg: dict):
    """Rebuild the exported absolute-joint training transforms for inference."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as oc

    @dataclasses.dataclass(frozen=True)
    class JointDataConfig(oc.DataConfigFactory):
        def create(self, assets_dirs: Path, model_config):
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=transforms.Group(
                    inputs=(JointInputs(model_cfg["camera_mode"]),),
                    outputs=(JointOutputs(),),
                ),
                model_transforms=oc.ModelTransformFactory()(model_config),
                prompt_from_task=False,
            )

    return oc.TrainConfig(
        name=model_cfg["train_config_name"],
        model=Pi0Config(
            pi05=True, action_dim=32, action_horizon=32,
            paligemma_variant=model_cfg["paligemma_variant"],
            action_expert_variant=model_cfg["action_expert_variant"],
        ),
        data=JointDataConfig(repo_id=model_cfg["repo_id"]),
        policy_metadata={
            "model_state_encoding": "absolute",
            "action_encoding": "recorded",
            "camera_mode": model_cfg["camera_mode"],
        },
    )


class PackPlateJointModel(ModelTemplate):
    def __init__(self, model_cfg: dict):
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared import normalize
        from .model import _resolve_pi05_model_root

        validate_artifacts(model_cfg)
        self.dims = get_robot_action_dim_info(model_cfg)
        if self.dims["arm_dim"] != [7, 7] or self.dims["ee_dim"] != [1, 1]:
            raise ValueError("pack-plate joint requires Tianji dual 7+1 dimensions")
        self.model_cfg = dict(model_cfg)
        self.num_steps = int(model_cfg["num_steps"])
        self.gripper_clip_tolerance = float(model_cfg.get("gripper_clip_tolerance", 0.1))
        if not np.isfinite(self.gripper_clip_tolerance) or self.gripper_clip_tolerance < 0:
            raise ValueError("gripper_clip_tolerance must be finite and nonnegative")
        self.model_root = _resolve_pi05_model_root(model_cfg)
        self.norm_stats_path = Path(model_cfg["norm_stats_path"]).expanduser().resolve()
        self.policy = create_trained_policy(
            make_joint_train_config(model_cfg), str(self.model_root),
            norm_stats=normalize.load(self.norm_stats_path),
            checkpoint_low_memory=model_cfg.get("checkpoint_low_memory", False),
        )
        self.reset()

    def sampling_modes(self):
        return ["default", "rtc"]

    def runtime_metadata(self):
        keys = (
            "policy_name", "task_name", "checkpoint_variant", "checkpoint_source",
            "norm_stats_source", "train_config_name", "observation_profile",
            "model_state_encoding", "action_semantics", "output_format",
            "action_type", "camera_mode", "action_horizon", "num_steps",
        )
        return {
            **{key: self.model_cfg[key] for key in keys},
            "action_dim": 32,
            "native_action_encoding": "absolute_joint_position",
            "native_order": "left_right",
            "base_image_mask": self.model_cfg["camera_mode"] == "wrist_main",
            "gripper_clip_tolerance": self.gripper_clip_tolerance,
            "model_root": str(self.model_root),
            "norm_stats_path": str(self.norm_stats_path),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        names = ["left_wrist", "right_wrist"]
        if self.model_cfg["camera_mode"] == "wrist_main":
            names.append("main")
        encoded = {}
        for index, obs in enumerate(obs_list):
            env_idx = obs.get("env_idx", index)
            if env_idx in encoded:
                raise ValueError("observation batch requires unique environment indices")
            encoded[env_idx] = {
                "state": pack_robot_state(obs, "joint", self.dims, source_type="obs").astype(np.float32),
                "images": {name: np.asarray(obs["vision"][f"cam_{name}"]["color"]) for name in names},
                "prompt": obs["instruction"],
            }
        self.observations = encoded

    def _infer(self, index, **kwargs):
        predicted = self.policy.infer(self.observations[index], num_steps=self.num_steps, **kwargs)
        actions = np.asarray(predicted["actions"]).copy()
        if actions.shape != (32, 16) or not np.isfinite(actions).all():
            raise ValueError("pack-plate joint actions must be finite (32,16)")
        # The learned opening can slightly exceed its physical [0,1] domain.
        # Match the TCP deployment's tolerance; never clip robot joint angles.
        openings = actions[:, [7, 15]]
        tolerance = self.gripper_clip_tolerance
        if np.any((openings < -tolerance) | (openings > 1.0 + tolerance)):
            raise ValueError(f"pack-plate joint opening exceeds [0,1] by more than {tolerance}")
        actions[:, [7, 15]] = np.clip(openings, 0.0, 1.0)
        return unpack_robot_state(actions, "joint", self.dims, source_type="obs")

    def get_action(self):
        return self.get_action_batch()[0]

    def get_action_batch(self, env_idx_list=None):
        if not self.observations:
            raise ValueError("update_obs is required before inference")
        return [self._infer(i) for i in (self.observations if env_idx_list is None else env_idx_list)]

    def get_action_rtc(self, sampling):
        if len(self.observations) != 1:
            raise ValueError("RTC requires exactly one current observation")
        condition, weights, beta = validate_rtc(sampling, 32, 16, copy=True)
        return self._infer(
            next(iter(self.observations)), action_condition=condition,
            condition_weights=weights, rtc_beta=beta,
        )

    def reset(self):
        self.observations = {}
