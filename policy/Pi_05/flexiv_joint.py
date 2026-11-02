"""Deployment transforms for absolute 7+1, left/right Flexiv joint policies.

This is an explicit profile, not an ALOHA sign/gripper conversion. Training
must use these same image slots, quantile normalization and absolute targets.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROFILE = "flexiv_joint"
ACTION_DIM = 16
PREPROCESSING_CONTRACT = "stack_cube_joint_3cam_v1"

# Submitted stack-cube training source, src/pass_ball/dataset.py (2026-11-02):
# sha256 13db9083da25eacef2fc6d9e28bddc3ca09de0b6ea1ee78fb10f73de84ef0e57.
# Resize before OpenPI's model transforms, exactly as its native dataset does.
def resize_training_rgb(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = max(width / 224, height / 224)
    resized = Image.fromarray(image).resize(
        (max(1, int(round(width / ratio))), max(1, int(round(height / ratio)))),
        Image.Resampling.BILINEAR,
    )
    result = Image.new("RGB", (224, 224))
    result.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
    return np.asarray(result)


def validate_artifacts(config: dict) -> dict:
    required = {
        "policy_name": "Pi_05",
        "observation_profile": PROFILE,
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "action_type": "joint",
        "action_semantics": "absolute_joint_position",
        "use_delta_joint_actions": False,
        "robot_action_dim_info": {"arm_dim": [7, 7], "ee_dim": [1, 1]},
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Flexiv joint profile requires {key}={expected!r}")
    for key in ("action_horizon", "num_steps"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    checkpoint = Path(config["model_path"]).expanduser().resolve()
    if not (checkpoint / "params/_METADATA").is_file():
        raise FileNotFoundError(f"Orbax Pi05 params metadata missing: {checkpoint}")
    if checkpoint.name != str(config["checkpoint_num"]):
        raise ValueError("checkpoint directory step does not match checkpoint_num")
    stats = Path(config["norm_stats_path"]).expanduser().resolve() / "norm_stats.json"
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    if digest != config["norm_stats_sha256"] or config["norm_stats_source"] != "sha256_" + digest:
        raise ValueError("normalization checksum does not match the Flexiv recipe")
    norm = json.loads(stats.read_text())["norm_stats"]
    for name in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(norm[name][field], dtype=float)
            if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
                raise ValueError(f"normalization {name}.{field} must be finite (16,)")
        if np.any(np.asarray(norm[name]["std"]) <= 0):
            raise ValueError(f"normalization {name}.std must be positive")
        if np.any(np.asarray(norm[name]["q99"]) <= np.asarray(norm[name]["q01"])):
            raise ValueError(f"normalization {name} quantiles must be ordered")
    return {
        "contract_status": "ready",
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "checkpoint": str(checkpoint),
        "norm_stats": str(stats),
        "norm_stats_sha256": digest,
        "native_action_dim": ACTION_DIM,
        "action_horizon": config["action_horizon"],
        "action_semantics": config["action_semantics"],
        "sampling_modes": ["default"],
        "inference_status": "not_run_by_artifact_check",
    }


@dataclasses.dataclass(frozen=True)
class FlexivInputs:
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError("Flexiv state must be finite (16,) in left/right 7+1 order")
        image = {}
        for target, source in (
            ("base_0_rgb", "observation/image"),
            ("left_wrist_0_rgb", "observation/left_wrist"),
            ("right_wrist_0_rgb", "observation/right_wrist"),
        ):
            rgb = np.asarray(data[source])
            if (
                rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8
                or min(rgb.shape[:2]) < 1
            ):
                raise ValueError(f"{source} must be an HWC uint8 RGB image")
            image[target] = resize_training_rgb(rgb)
        result = {
            "state": state,
            "image": image,
            "image_mask": {name: np.bool_(True) for name in image},
        }
        if "prompt" in data:
            result["prompt"] = str(data["prompt"])
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
                raise ValueError("Flexiv actions must be finite (horizon,16)")
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class FlexivOutputs:
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < ACTION_DIM or not np.isfinite(actions).all():
            raise ValueError("Flexiv model output must be a finite horizon-by-action matrix")
        return {"actions": actions[..., :ACTION_DIM]}


def make_train_config(config: dict):
    """Rebuild the explicit deployment pipeline without training-site paths."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as oc

    @dataclasses.dataclass(frozen=True)
    class FlexivDataConfig(oc.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=transforms.Group(
                    inputs=(FlexivInputs(),), outputs=(FlexivOutputs(),)
                ),
                model_transforms=oc.ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),
                prompt_from_task=False,
            )

    stats = Path(config["norm_stats_path"])
    return oc.TrainConfig(
        name=config["train_config_name"],
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=config["action_horizon"]),
        data=FlexivDataConfig(
            repo_id=config["repo_id"],
            assets=oc.AssetsConfig(assets_dir=str(stats.parent), asset_id=stats.name),
        ),
        policy_metadata={"action_semantics": "absolute_joint_position", "native_action_dim": ACTION_DIM},
    )
