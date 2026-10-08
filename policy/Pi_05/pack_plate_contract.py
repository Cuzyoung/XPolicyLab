"""Artifact identity for the isolated Tianji pack-plate Pi05 profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROFILE = "tianji_taccap_pi05_pack_plate"
TRAIN_CONFIG = "pi05_pack_plate_wrist_only_zero_pose_deploy"
SOURCE = "pi05-pack-plate-wrist-only-final-59999"
ASSET = "pack-plate-taccap-h32-zero-pose"
STEP = 59999


def validate_artifacts(config: dict) -> dict:
    required = {
        "policy_name": "Pi_05",
        "protocol": "ws",
        "task_name": "plate",
        "env_cfg_type": "tianji_dual",
        "action_type": "ee",
        "observation_profile": PROFILE,
        "train_config_name": TRAIN_CONFIG,
        "model_state_encoding": "zero_pose",
        "action_horizon": 32,
        "output_format": "xpolicylab",
        "action_semantics": "absolute_per_arm_base_xyz_wxyz",
        "checkpoint_source": SOURCE,
        "checkpoint_num": STEP,
        "checkpoint_variant": "pi05_pack_plate_wrist_only_step59999",
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"pack-plate requires {key}={value!r}")
    checkpoint = Path(config["model_path"]).expanduser().resolve()
    stats_dir = Path(config["norm_stats_path"]).expanduser().resolve()
    stats = stats_dir / "norm_stats.json"
    if checkpoint.name != f"checkpoint-{STEP}" or checkpoint.parent.name != SOURCE:
        raise ValueError("pack-plate checkpoint step or export directory does not match recipe")
    if stats_dir != checkpoint / "assets" / ASSET:
        raise ValueError("pack-plate normalization must come from this checkpoint's assets")
    if not (checkpoint / "params/_METADATA").is_file():
        raise FileNotFoundError(f"Orbax Pi05 params metadata missing: {checkpoint}")
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    if (
        digest != config.get("norm_stats_sha256")
        or config.get("norm_stats_source") != "sha256_" + digest
    ):
        raise ValueError("pack-plate normalization checksum does not match recipe")
    values = json.loads(stats.read_text())["norm_stats"]
    for name in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(values[name][field], dtype=float)
            if value.shape != (20,) or not np.isfinite(value).all():
                raise ValueError(f"pack-plate normalization {name}.{field} must be finite (20,)")
    if int(config["num_steps"]) <= 0:
        raise ValueError("num_steps must be positive")
    return {
        "contract_status": "ready",
        "inference_status": "not_run",
        "checkpoint": str(checkpoint),
        "norm_stats": str(stats),
        "norm_stats_sha256": digest,
        "action_horizon": 32,
        "action_dt_s": 1 / 30,
        "sampling_modes": ["default", "rtc"],
    }
