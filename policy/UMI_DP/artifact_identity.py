"""Trusted UMI checkpoint loading and its deployment contract."""

import hashlib
import json
import math
from pathlib import Path

from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

POLICY_DIR = Path(__file__).resolve().parent
NAMESPACE = "XPolicyLab.policy.UMI_DP.upstream.diffusion_policy."
LEGACY_RGB_SHA256 = "1d7d9def9742b8cc549e5e785e84e61120a8beb7648da60662962fd09b5e4767"
SEMANTICS = "absolute_per_arm_base_xyz_wxyz"
PROFILE = "tianji_umi_relative2"


def remap_targets(value):
    """Avoid the unrelated top-level diffusion_policy package in XPolicyLab DP."""
    if isinstance(value, dict):
        return {key: remap_targets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_targets(item) for item in value]
    if isinstance(value, str) and value.startswith("diffusion_policy."):
        return NAMESPACE + value.removeprefix("diffusion_policy.")
    return value


def load_artifact(model_cfg):
    import dill
    import torch
    from omegaconf import OmegaConf

    root = resolve_checkpoint_root(model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
    path = (
        root
        if root.is_file()
        else root / model_cfg.get("checkpoint_file", "checkpoints/latest.ckpt")
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(
        path, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True
    )
    original = OmegaConf.to_container(payload["cfg"], resolve=True)
    cfg = OmegaConf.create(remap_targets(original))
    ds = cfg.task.dataset
    if (
        ds.action_mode,
        ds.observation_mode,
        ds.position_unit,
        ds.pose_frame,
        ds.rotation_layout,
    ) != ("relative_trajectory", "relative", "m", "world", "column"):
        raise ValueError("Expected relative UMI trajectories, metre/world/column training data")
    shape = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    expected_obs = {f"camera{i}_rgb" for i in range(2)} | {
        f"robot{i}_{suffix}"
        for i in range(2)
        for suffix in ("eef_pos", "eef_rot_axis_angle", "gripper_width")
    }
    if set(shape["obs"]) != expected_obs or shape["action"]["shape"] != [20]:
        raise ValueError("Unsupported checkpoint observation/action representation")
    if any(
        int(value["horizon"]) != 2 or value.get("latency_steps", 0) != 0
        for value in shape["obs"].values()
    ):
        raise ValueError("UMI_DP requires two observations with zero training latency")
    obs_steps = {int(value["down_sample_steps"]) for value in shape["obs"].values()}
    if len(obs_steps) != 1:
        raise ValueError("Mixed observation sampling is unsupported")
    observation_steps = obs_steps.pop()
    action_steps = int(shape["action"]["down_sample_steps"])
    fps = float(model_cfg.get("source_fps", original.get("source_fps", 0)))
    if "source_fps" in original and fps != float(original["source_fps"]):
        raise ValueError("source_fps differs from the checkpoint embedded dataset FPS")
    if not math.isfinite(fps) or fps <= 0 or min(action_steps, observation_steps) <= 0:
        raise ValueError("Set source_fps from training metadata; sampling steps must be positive")
    if shape["action"].get("latency_steps", 0) != 0:
        raise ValueError("Nonzero action latency is unsupported")
    key = "ema_model" if cfg.training.use_ema else "model"
    state = payload["state_dicts"][key]
    if not any(name.startswith("normalizer.") for name in state):
        raise ValueError("Checkpoint has no embedded normalizer")
    rgb_normalize = bool(cfg.policy.obs_encoder.imagenet_norm) and sha != LEGACY_RGB_SHA256
    cfg.policy.obs_encoder.imagenet_norm = rgb_normalize
    cfg.policy.obs_encoder.pretrained = False
    if not str(cfg.policy._target_).endswith(".DiffusionUnetTimmPolicy"):
        raise ValueError("Unsupported UMI policy class")
    if not str(cfg.policy.noise_scheduler._target_).endswith(".DDIMScheduler"):
        raise ValueError("UMI_DP requires the checkpoint DDIM sampler")
    if cfg.policy.noise_scheduler.prediction_type != "epsilon":
        raise ValueError("UMI_DP RTC requires epsilon prediction")
    report = {
        "policy_family": "umi_dp",
        "observation_profile": PROFILE,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_file": path.name,
        "checkpoint_sha256": sha,
        "training_config_sha256": hashlib.sha256(
            json.dumps(original, sort_keys=True).encode()
        ).hexdigest(),
        "weight_key": key,
        "action_horizon": int(shape["action"]["horizon"]),
        "action_dt_s": action_steps / fps,
        "first_action_offset_s": 1.0 / fps,
        "source_fps": fps,
        "action_down_sample_steps": action_steps,
        "observation_horizon": 2,
        "observation_period_s": observation_steps / fps,
        "num_inference_steps": int(cfg.policy.num_inference_steps),
        "rgb_normalize": rgb_normalize,
        "inference_augmentation": "checkpoint",
        "action_semantics": SEMANTICS,
        "rtc_guidance": model_cfg.get("rtc_guidance", "pigdm"),
    }
    for name, expected in (model_cfg.get("expected_artifacts") or {}).items():
        if report.get(name) != expected:
            raise ValueError(
                f"Artifact identity mismatch for {name}: {report.get(name)!r} != {expected!r}"
            )
    return cfg, state, shape, report


def validate_deployment(model_cfg):
    _, _, _, report = load_artifact(model_cfg)
    return report
