"""Hardware-free artifact checks for the opt-in pass-ball zero-pose profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def validate_artifacts(config: dict) -> dict:
    required = {
        "policy_name": "Pi_05",
        "protocol": "ws",
        "env_cfg_type": "tianji_dual",
        "action_type": "ee",
        "observation_profile": "tianji_taccap_pi05_zero_pose",
        "train_config_name": "pi05_pass_ball_hifi_umi_lora_zero_pose",
        "model_state_encoding": "zero_pose",
        "action_horizon": 32,
        "output_format": "xpolicylab",
        "action_semantics": "absolute_per_arm_base_xyz_wxyz",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"pass-ball requires {key}={expected!r}")
    checkpoint = Path(config["model_path"]).expanduser().resolve()
    stats = Path(config["norm_stats_path"]).expanduser().resolve() / "norm_stats.json"
    if not (checkpoint / "params/_METADATA").is_file():
        raise FileNotFoundError(f"Orbax Pi05 params metadata missing: {checkpoint}")
    if checkpoint.name != str(config["checkpoint_num"]):
        raise ValueError("checkpoint directory step does not match checkpoint_num")
    if checkpoint.parent.parent.name != config["checkpoint_source"]:
        raise ValueError("checkpoint export directory does not match checkpoint_source")
    expected_variant = f"pi05_pass_ball_zero_pose_step{config['checkpoint_num']}"
    if config["checkpoint_variant"] != expected_variant:
        raise ValueError("checkpoint_variant does not match selected step")
    digest = hashlib.sha256(stats.read_bytes()).hexdigest()
    if digest != config["norm_stats_sha256"] or config["norm_stats_source"] != "sha256_" + digest:
        raise ValueError("normalization checksum does not match the zero-pose recipe")
    normalization = json.loads(stats.read_text())["norm_stats"]
    for name in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(normalization[name][field], dtype=float)
            if value.shape != (20,) or not np.isfinite(value).all():
                raise ValueError(f"normalization {name}.{field} must be finite (20,)")
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
        "sampling_modes": ["default"],
    }
