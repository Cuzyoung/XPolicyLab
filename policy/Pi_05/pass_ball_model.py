"""Pi05 LoRA pass-ball deployment matching the submitted zero-pose training.

Native layout is right then left, xyz metres + Rot6D rows + absolute opening.
The model sees zero pose; real per-request anchors remain outside its inputs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .pass_ball_pose import OPTICAL_FROM_FLU, relative_action_chunk, rotation_from_6d
from .pass_ball_state import EncodeState, ZeroNormalizedPose

PROFILE = "tianji_taccap_pi05_zero_pose"
TRAIN_CONFIG = "pi05_pass_ball_hifi_umi_lora_zero_pose"
ACTION_SEMANTICS = "absolute_per_arm_base_xyz_wxyz"


@dataclasses.dataclass(frozen=True)
class PassBallInputs:
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("pass-ball requires a finite 20D right/left TCP state")
        left, right = (np.asarray(data["images"][key]) for key in ("left_wrist", "right_wrist"))
        for image in (left, right):
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError("pass-ball requires HWC uint8 RGB wrist images")
        result = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(left),
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": str(data["prompt"]),
        }
        if "actions" in data:
            result["actions"] = relative_action_chunk(state, data["actions"])
        return result


@dataclasses.dataclass(frozen=True)
class PassBallOutputs:
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :20]}


def make_train_config(model_cfg: dict):
    """Rebuild deployment transforms without the training machine/dataset paths."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as oc

    @dataclasses.dataclass(frozen=True)
    class PassBallDataConfig(oc.DataConfigFactory):
        def create(self, assets_dirs: Path, model_config):
            model_transforms = oc.ModelTransformFactory()(model_config)
            model_transforms = dataclasses.replace(
                model_transforms, inputs=(ZeroNormalizedPose(), *model_transforms.inputs)
            )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=transforms.Group(
                    inputs=(PassBallInputs(), EncodeState("zero_pose")),
                    outputs=(PassBallOutputs(),),
                ),
                model_transforms=model_transforms,
                prompt_from_task=False,
            )

    return oc.TrainConfig(
        name=TRAIN_CONFIG,
        model=Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=PassBallDataConfig(repo_id=model_cfg["repo_id"]),
        policy_metadata={
            "model_state_encoding": "zero_pose",
            "action_encoding": "tcp_relative_chunk",
        },
    )


def encode_observation(obs: dict) -> dict:
    state = obs["state"]
    blocks = []
    for side in ("right", "left"):
        pose = np.asarray(state[f"{side}_ee_pose"], dtype=np.float64)
        grip = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float64)
        if (
            pose.shape != (7,)
            or grip.shape != (1,)
            or not np.isfinite(pose).all()
            or not np.isfinite(grip).all()
            or not 0 <= grip[0] <= 1
            or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-4)
        ):
            raise ValueError("pass-ball requires finite xyz/wxyz TCPs and [0,1] openings")
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        # Match training axis_change=tool: R_train = R_TCP @ C.T.
        rotation = rotation @ OPTICAL_FROM_FLU.T
        blocks.append(np.r_[pose[:3], rotation[:2].reshape(6), grip])
    prompt = obs.get("instruction", obs.get("instructions"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("pass-ball requires the training task instruction")
    return {
        "state": np.concatenate(blocks).astype(np.float32),
        "images": {
            side: np.asarray(obs["vision"][f"cam_{side}"]["color"])
            for side in ("left_wrist", "right_wrist")
        },
        "prompt": prompt,
    }


def decode_actions(anchor: np.ndarray, actions: np.ndarray) -> list[dict]:
    actions = np.asarray(actions)
    if actions.shape != (32, 20) or not np.isfinite(actions).all():
        raise ValueError("pass-ball must return finite (32,20) native actions")
    absolute = relative_action_chunk(anchor, actions, inverse=True)
    steps = []
    for row in absolute:
        step = {}
        for side, offset in (("right", 0), ("left", 10)):
            rotation = rotation_from_6d(row[offset + 3 : offset + 9], "first_two_rows")
            rotation = rotation @ OPTICAL_FROM_FLU
            quat = Rotation.from_matrix(rotation).as_quat()
            step[f"{side}_ee_pose"] = np.r_[row[offset : offset + 3], quat[[3, 0, 1, 2]]]
            step[f"{side}_ee_joint_state"] = row[offset + 9 : offset + 10].copy()
        steps.append(step)
    return steps


class PassBallZeroPoseModel(ModelTemplate):
    """Deployment-only profile; unsupported sampler methods are deliberately absent."""

    def __init__(self, model_cfg: dict):
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared import normalize

        from .model import _resolve_pi05_model_root
        from .pass_ball_contract import validate_artifacts

        validate_artifacts(model_cfg)
        required = {
            "policy_name": "Pi_05",
            "env_cfg_type": "tianji_dual",
            "action_type": "ee",
            "observation_profile": PROFILE,
            "train_config_name": TRAIN_CONFIG,
            "model_state_encoding": "zero_pose",
            "action_horizon": 32,
            "output_format": "xpolicylab",
            "action_semantics": ACTION_SEMANTICS,
        }
        for key, value in required.items():
            if model_cfg.get(key) != value:
                raise ValueError(f"pass-ball requires {key}={value!r}")
        dims = get_robot_action_dim_info(model_cfg)
        if dims["arm_dim"] != [7, 7] or dims["ee_dim"] != [1, 1]:
            raise ValueError("pass-ball requires Tianji dual 7+1 dimensions")
        self.model_cfg = dict(model_cfg)
        self.num_steps = int(model_cfg.get("num_steps", 10))
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.model_root = _resolve_pi05_model_root(model_cfg)
        self.norm_stats_path = Path(model_cfg["norm_stats_path"]).expanduser().resolve()
        norm_stats = normalize.load(self.norm_stats_path)
        for name in ("state", "actions"):
            for field in ("mean", "std", "q01", "q99"):
                values = np.asarray(getattr(norm_stats[name], field))
                if values.shape != (20,) or not np.isfinite(values).all():
                    raise ValueError(f"pass-ball normalization {name}.{field} must be finite (20,)")
        self.policy = create_trained_policy(
            make_train_config(model_cfg), str(self.model_root), norm_stats=norm_stats
        )
        self.reset()

    def sampling_modes(self):
        return ["default"]

    def runtime_metadata(self):
        return {
            **{
                key: self.model_cfg[key]
                for key in (
                    "policy_name",
                    "task_name",
                    "checkpoint_variant",
                    "checkpoint_source",
                    "norm_stats_source",
                    "train_config_name",
                    "observation_profile",
                    "model_state_encoding",
                    "action_semantics",
                    "output_format",
                )
            },
            "action_type": "ee",
            "action_horizon": 32,
            "action_dim": 32,
            "num_steps": self.num_steps,
            "native_action_encoding": "tcp_relative_chunk",
            "native_order": "right_left",
            "rotation_6d_layout": "first_two_rows",
            "axis_change": "tool",
            "base_image_mask": False,
            "model_root": str(self.model_root.resolve()),
            "norm_stats_path": str(self.norm_stats_path),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        indices = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded = [encode_observation(obs) for obs in obs_list]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("pass-ball observation batch requires unique environment indices")
        self.observations = dict(zip(indices, encoded, strict=True))
        self.env_indices = indices

    def get_action(self):
        if not self.env_indices:
            raise ValueError("update_obs is required before get_action")
        return self.get_action_batch([self.env_indices[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        indices = self.env_indices if env_idx_list is None else env_idx_list
        if not indices:
            raise ValueError("update_obs_batch is required before get_action_batch")
        result = []
        for index in indices:
            obs = self.observations[index]
            anchor = np.array(obs["state"], copy=True)
            predicted = self.policy.infer(obs, num_steps=self.num_steps)
            result.append(decode_actions(anchor, predicted["actions"]))
        return result

    def reset(self):
        self.observations = {}
        self.env_indices = []
