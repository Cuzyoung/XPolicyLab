"""Full Pi05 Flexiv flange policy, matched to the submitted H50 UMI training.

Reuse the training-derived UMI pose/state transforms. The model sees zero pose
and live openings; absolute per-request flange anchors never enter tokenization.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info

from .flexiv_joint import SingleFrameCompression, resize_training_rgb
from .pass_ball_model import decode_actions, encode_observation
from .pass_ball_pose import OPTICAL_FROM_FLU, relative_action_chunk
from .pass_ball_state import POSE_INDICES, EncodeState, ZeroNormalizedPose

PROFILE = "flexiv_ee_umi"
CONTRACT = "stack_cube_flexiv_flange_ee_umi_v1"
SEMANTICS = "absolute_per_arm_base_xyz_wxyz"


def validate_artifacts(config):
    if config.get("image_processing") is not None:
        SingleFrameCompression.parameters(config["image_processing"])
    required = {
        "policy_name": "Pi_05",
        "observation_profile": PROFILE,
        "preprocessing_contract": CONTRACT,
        "action_type": "ee",
        "action_semantics": SEMANTICS,
        "model_state_encoding": "zero_pose",
        "action_horizon": 50,
        "output_format": "xpolicylab",
        "pose_frame": "flange",
        "native_action_encoding": "tcp_relative_chunk",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Flexiv EE requires {key}={expected!r}")
    if get_robot_action_dim_info(config) != {"arm_dim": [7, 7], "ee_dim": [1, 1]}:
        raise ValueError("Flexiv EE requires dual 7+1 physical coordinates")
    if get_batch_size(config) != 1:
        raise ValueError("Flexiv EE requires num_envs=1")
    if type(config.get("num_steps")) is not int or config["num_steps"] <= 0:
        raise ValueError("num_steps must be a positive integer")
    seed = config.get("inference_seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("inference_seed must be an integer in [0, 2**32)")
    checkpoint = Path(config["model_path"]).expanduser().resolve()
    stats_dir = Path(config["norm_stats_path"]).expanduser().resolve()
    if (
        checkpoint.name != str(config["checkpoint_num"])
        or checkpoint.parent.name != config["checkpoint_source"]
    ):
        raise ValueError("Flexiv EE checkpoint source/step mismatch")
    if stats_dir != checkpoint / "assets/flexiv-ee-h50-umi":
        raise ValueError("Flexiv EE normalization must come from the selected checkpoint")
    if not (checkpoint / "params/_METADATA").is_file():
        raise FileNotFoundError(f"Orbax params metadata missing: {checkpoint}")
    stats = stats_dir / "norm_stats.json"
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    if digest != config["norm_stats_sha256"] or config["norm_stats_source"] != "sha256_" + digest:
        raise ValueError("Flexiv EE normalization checksum mismatch")
    norm = json.loads(stats.read_text())["norm_stats"]
    for name in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(norm[name][field], dtype=float)
            if value.shape != (20,) or not np.isfinite(value).all():
                raise ValueError(f"normalization {name}.{field} must be finite (20,)")
        if np.any(np.asarray(norm[name]["std"]) < 0) or np.any(
            np.asarray(norm[name]["q99"]) < norm[name]["q01"]
        ):
            raise ValueError(f"invalid normalization distribution: {name}")
    for field in ("mean", "std", "q01", "q99"):
        if np.any(np.asarray(norm["state"][field])[POSE_INDICES] != 0):
            raise ValueError("Flexiv EE requires zero-pose normalization")
    return {
        "contract_status": "ready",
        "checkpoint": str(checkpoint),
        "norm_stats": str(stats),
        "norm_stats_sha256": digest,
        "native_action_dim": 20,
        "action_horizon": 50,
        "action_semantics": SEMANTICS,
        "sampling_modes": ["default", "rtc"],
        "inference_status": "not_run_by_artifact_check",
    }


@dataclasses.dataclass(frozen=True)
class FlexivEEInputs:
    compression: SingleFrameCompression | None = None

    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("Flexiv EE state must be finite right/left (20,)")
        images = {}
        for slot, camera in (
            ("base_0_rgb", "main"),
            ("left_wrist_0_rgb", "left_wrist"),
            ("right_wrist_0_rgb", "right_wrist"),
        ):
            image = np.asarray(data["images"][camera])
            if (
                image.ndim != 3
                or image.shape[-1] != 3
                or image.dtype != np.uint8
                or min(image.shape[:2]) < 1
            ):
                raise ValueError(f"Flexiv EE {camera} must be HWC uint8 RGB")
            if self.compression is not None:
                image = self.compression.process(image, camera=camera)
            images[slot] = resize_training_rgb(image)
        result = {
            "state": state,
            "image": images,
            "image_mask": {key: np.True_ for key in images},
            "prompt": str(data["prompt"]),
        }
        if "actions" in data:
            result["actions"] = relative_action_chunk(state, data["actions"])
        return result


@dataclasses.dataclass(frozen=True)
class FlexivEEOutputs:
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.shape != (50, 32) or not np.isfinite(actions).all():
            raise ValueError(f"Flexiv EE output must be finite (50,32); got {actions.shape}")
        return {"actions": actions[:, :20]}


def rtc_condition(sampling, anchor):
    """Absolute left/right xyz+wxyz+opening -> training-native right/left poses.

    Rebase to this observation inside FlexivEEInputs, before zeroing its state.
    Unweighted rows use the anchor so zero-padding never creates invalid SO(3).
    """
    from XPolicyLab.utils.sampling_contracts import validate_rtc

    condition, weights, beta = validate_rtc(sampling, 50, 16, copy=True)
    native = np.tile(np.asarray(anchor, dtype=np.float32), (50, 1))
    active = weights > 0
    for source, destination in ((0, 10), (8, 0)):
        rows = condition[active, source:source + 8]
        if not len(rows):
            continue
        norms = np.linalg.norm(rows[:, 3:7], axis=1)
        if not np.all(np.isclose(norms, 1, atol=1e-4)):
            raise ValueError("Flexiv EE RTC active pose quaternions must be unit wxyz")
        if np.any((rows[:, 7] < 0) | (rows[:, 7] > 1)):
            raise ValueError("Flexiv EE RTC active gripper openings must be in [0,1]")
        rotation = Rotation.from_quat(rows[:, [4, 5, 6, 3]]).as_matrix()
        rotation = rotation @ OPTICAL_FROM_FLU.T
        native[active, destination:destination + 3] = rows[:, :3]
        native[active, destination + 3:destination + 9] = rotation[:, :2, :].reshape(-1, 6)
        native[active, destination + 9] = rows[:, 7]
    return native, weights, beta


def make_train_config(config):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as oc

    compression = SingleFrameCompression.from_config(config)

    @dataclasses.dataclass(frozen=True)
    class FlexivEEDataConfig(oc.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            model_transforms = oc.ModelTransformFactory()(model_config)
            model_transforms = dataclasses.replace(
                model_transforms, inputs=(ZeroNormalizedPose(), *model_transforms.inputs)
            )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=transforms.Group(
                    inputs=(FlexivEEInputs(compression), EncodeState("zero_pose")),
                    outputs=(FlexivEEOutputs(),),
                ),
                model_transforms=model_transforms,
                action_sequence_keys=("action",),
                prompt_from_task=False,
            )

    stats = Path(config["norm_stats_path"])
    return oc.TrainConfig(
        name=config["train_config_name"],
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=FlexivEEDataConfig(
            repo_id=config["repo_id"],
            assets=oc.AssetsConfig(assets_dir=str(stats.parent), asset_id=stats.name),
        ),
        policy_metadata={
            "image_processing": None if compression is None else dict(config["image_processing"]),
            "image_processing_effective": None if compression is None else compression.options,
            "model_state_encoding": "zero_pose",
            "action_encoding": "tcp_relative_chunk",
        },
    )


class FlexivEEModel(ModelTemplate):
    def __init__(self, config):
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared import normalize

        from .model import _resolve_pi05_model_root

        validate_artifacts(config)
        self.model_cfg = dict(config)
        self.model_root = _resolve_pi05_model_root(config)
        self.policy = create_trained_policy(
            make_train_config(config),
            str(self.model_root),
            norm_stats=normalize.load(Path(config["norm_stats_path"])),
        )
        self.reset()

    def sampling_modes(self):
        return ["default"] if self.policy._is_pytorch_model else ["default", "rtc"]

    def runtime_metadata(self):
        return {
            "policy_family": "pi05",
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
                    "preprocessing_contract",
                    "pose_frame",
                    "native_action_encoding",
                    "inference_seed",
                    "num_steps",
                )
            },
            "model_root": str(self.model_root),
            "norm_stats_path": str(Path(self.model_cfg["norm_stats_path"]).resolve()),
            "action_type": "ee",
            "action_horizon": 50,
            "action_dim": 32,
            "native_order": "right_left",
            "rotation_6d_layout": "first_two_rows",
            "axis_change": "tool",
            "base_image_mask": True,
            "image_processing": self.policy.metadata.get("image_processing"),
            "image_processing_effective": self.policy.metadata.get("image_processing_effective"),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("Flexiv EE requires one observation")
        obs = obs_list[0]
        encoded = encode_observation(obs)
        encoded["images"]["main"] = np.asarray(obs["vision"]["cam_head"]["color"])
        self.env_indices = [obs.get("env_idx", 0)]
        self.observations = {self.env_indices[0]: encoded}

    def get_action(self):
        return self.get_action_batch()[0]

    def get_action_rtc(self, sampling):
        if "rtc" not in self.sampling_modes():
            raise ValueError("Flexiv EE RTC requires the JAX sampler")
        if not self.env_indices:
            raise ValueError("update_obs is required before get_action_rtc")
        obs = self.observations[self.env_indices[0]]
        anchor = obs["state"].copy()
        condition, weights, beta = rtc_condition(sampling, anchor)
        if not np.any(weights):
            return self.get_action()
        prediction = self.policy.infer(
            obs, num_steps=self.model_cfg["num_steps"], action_condition=condition,
            condition_weights=weights, rtc_beta=beta,
        )
        return decode_actions(anchor, prediction["actions"], horizon=50)

    def get_action_batch(self, env_idx_list=None):
        indices = self.env_indices if env_idx_list is None else env_idx_list
        if not indices:
            raise ValueError("update_obs is required before get_action")
        chunks = []
        for index in indices:
            obs = self.observations[index]
            anchor = obs["state"].copy()
            prediction = self.policy.infer(obs, num_steps=self.model_cfg["num_steps"])
            chunks.append(decode_actions(anchor, prediction["actions"], horizon=50))
        return chunks

    def reset(self):
        self.observations = {}
        self.env_indices = []
        self.policy.reset_rng(self.model_cfg["inference_seed"])
