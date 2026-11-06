"""Artifact identity for the isolated Tianji pack-plate Pi05 profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROFILE = "tianji_taccap_pi05_pack_plate"
JOINT_PROFILE = "tianji_taccap_pi05_pack_plate_joint"
TRAIN_CONFIG = "pi05_pack_plate_wrist_only_zero_pose_deploy"
SOURCE = "pi05-pack-plate-wrist-only-final-59999"
ASSET = "pack-plate-taccap-h32-zero-pose"
STEP = 59999

ARTIFACTS = {
    "pi05_pack_plate_umi_half_speed_h50_step29999": {
        "train_config_name": "pi05_pack_plate_umi_half_speed_h50_deploy",
        "checkpoint_source": "pi05-plate-umi-half-b64-h50-30k-dist-p4-20261105-152316-final-29999",
        "asset": "umi-half-speed-h50",
        "checkpoint_num": 29999,
        "action_horizon": 50,
        "model_variants": {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
        },
    },
    "pi05_pack_plate_umi_only_step40000": {
        "train_config_name": "pi05_pack_plate_curated_umi_only_full_60k",
        "checkpoint_source": "pi05-plate-umi-only-full-b32-h32-60k-p4-20261028-032801-step-40000",
        "asset": "umi_only-h32-zero-pose-all-train",
        "checkpoint_num": 40000,
        "model_variants": {"paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m"},
    },
    "pi05_pack_plate_umi_dagger_step40000": {
        "train_config_name": "pi05_pack_plate_curated_umi_dagger_full_60k",
        "checkpoint_source": "pi05-plate-umi-dagger-full-b32-h32-60k-p4-20261028-032801-step-40000",
        "asset": "umi_dagger-h32-zero-pose-all-train",
        "checkpoint_num": 40000,
        "model_variants": {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
        },
    },
    **{
        f"pi05_pack_plate_ratio_umi{umi:03d}_tele{tele:03d}_step29999": {
            "train_config_name": f"pi05_pack_plate_ratio_umi{umi:03d}_teleop{tele:03d}_full_30k",
            "checkpoint_source": (
                f"pi05-plate-ratio-umi{umi:03d}-tele{tele:03d}"
                f"-full-30k-p4-{run_stamp}-final-29999"
            ),
            "asset": "joint-fk-plus-taccap-natural-h32-zero-pose-terminal-aligned",
            "checkpoint_num": 29999,
            "model_variants": {
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
            },
        }
        for umi, tele, run_stamp in (
            (0, 100, "20261021-014111"),
            (25, 75, "20261021-014111"),
            (50, 50, "20261021-014111"),
            (75, 25, "20261021-014111"),
            (90, 10, "20261026-141546"),
            (95, 5, "20261026-141546"),
            (100, 0, "20261021-014111"),
        )
    },
    "pi05_pack_plate_joint_fk_tcp_zero_state_step29999": {
        "train_config_name": "pi05_pack_plate_joint_fk_tcp_wrist_zero_pose_full_30k",
        "checkpoint_source": "pi05-plate-joint-fk-tcp-zstate-full-30k-20261019-055917-r1-final-29999",
        "asset": "joint-fk-tcp-h32-zero-pose",
        "checkpoint_num": 29999,
        "model_variants": {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
        },
    },
    **{
        f"pi05_pack_plate_joint_{view}_step29999": {
            "train_config_name": f"pi05_pack_plate_joint_{train_view}_full_30k",
            "checkpoint_source": f"pi05-plate-joint-{view}-full-30k-20261019-055917-r1-final-29999",
            "asset": "joint-native-h32",
            "checkpoint_num": 29999,
            "model_state_encoding": "absolute",
            "control_space": "joint",
            "camera_mode": camera_mode,
            "model_variants": {
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
            },
        }
        for view, train_view, camera_mode in (
            ("wrist", "wrist", "wrist_only"),
            ("3cam", "three_camera", "wrist_main"),
        )
    },
    "pi05_pack_plate_joint_fk_taccap_mix_zero_state_step29999": {
        "train_config_name": "pi05_pack_plate_joint_fk_taccap_mix_wrist_zero_pose_full_30k",
        "checkpoint_source": "pi05-plate-joint-fk-taccap-mix-zstate-full-30k-20261019-055917-r1-final-29999",
        "asset": "joint-fk-plus-taccap-natural-h32-zero-pose-terminal-aligned",
        "checkpoint_num": 29999,
        "model_variants": {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
        },
    },
    "pi05_pack_plate_wrist_only_step59999": {
        "train_config_name": TRAIN_CONFIG,
        "checkpoint_source": SOURCE,
        "asset": ASSET,
        "checkpoint_num": STEP,
    },
    "pi05_pack_plate_full_zero_state_pack_instruction_step59999": {
        "train_config_name": "pi05_pack_plate_full_zero_state_pack_instruction_deploy",
        "checkpoint_source": "pi05-pack-plate-full-zero-state-pack-instruction-final-59999",
        "asset": "pack-plate-taccap-h32-zero-pose-pack-instruction",
        "checkpoint_num": STEP,
        "model_variants": {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
        },
    },
    "pi05_pack_plate_full_zero_state_pack_instruction_step20000": {
        "train_config_name": "pi05_pack_plate_full_zero_state_pack_instruction_deploy",
        "checkpoint_source": "pi05-pack-plate-full-zero-state-pack-instruction-20000",
        "asset": "pack-plate-taccap-h32-zero-pose-pack-instruction",
        "checkpoint_num": 20000,
        "model_variants": {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
        },
    },
    **{
        f"pi05_pack_plate_inter_hand_relative_step{step}": {
            "train_config_name": "pi05_pack_plate_inter_hand_relative_deploy",
            "checkpoint_source": (
                "pi05-pack-plate-inter-hand-relative-tracker-fixed-best-20000-final-59999"
            ),
            "checkpoint_subdir": f"checkpoints/{step}",
            "asset": "pack-plate-taccap-h32-inter-hand-relative",
            "checkpoint_num": step,
            "model_state_encoding": "inter_hand_relative",
            "model_variants": {
                "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora",
            },
        }
        for step in (20000, 59999)
    },
}


def validate_artifacts(config: dict) -> dict:
    variant = config.get("checkpoint_variant")
    if variant not in ARTIFACTS:
        raise ValueError(f"unsupported pack-plate checkpoint_variant={variant!r}")
    artifact = ARTIFACTS[variant]
    action_horizon = artifact.get("action_horizon", 32)
    joint = artifact.get("control_space") == "joint"
    data_dim = 16 if joint else 20
    required = {
        "policy_name": "Pi_05",
        "protocol": "ws",
        "task_name": "plate",
        "env_cfg_type": "tianji_dual",
        "action_type": "joint" if joint else "ee",
        "observation_profile": JOINT_PROFILE if joint else PROFILE,
        "model_state_encoding": artifact.get("model_state_encoding", "zero_pose"),
        "action_horizon": action_horizon,
        "output_format": "xpolicylab",
        "action_semantics": "absolute_joint_position" if joint else "absolute_per_arm_base_xyz_wxyz",
        "train_config_name": artifact["train_config_name"],
        "checkpoint_source": artifact["checkpoint_source"],
        "checkpoint_num": artifact["checkpoint_num"],
        **artifact.get("model_variants", {}),
        **({"camera_mode": artifact["camera_mode"]} if joint else {}),
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"pack-plate requires {key}={value!r}")
    checkpoint = Path(config["model_path"]).expanduser().resolve()
    stats_dir = Path(config["norm_stats_path"]).expanduser().resolve()
    stats = stats_dir / "norm_stats.json"
    step = artifact["checkpoint_num"]
    suffix = Path(artifact["checkpoint_source"]) / artifact.get(
        "checkpoint_subdir", f"checkpoint-{step}"
    )
    if checkpoint.parts[-len(suffix.parts) :] != suffix.parts:
        raise ValueError("pack-plate checkpoint step or export directory does not match recipe")
    if stats_dir != checkpoint / "assets" / artifact["asset"]:
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
            if value.shape != (data_dim,) or not np.isfinite(value).all():
                raise ValueError(f"pack-plate normalization {name}.{field} must be finite ({data_dim},)")
    if int(config["num_steps"]) <= 0:
        raise ValueError("num_steps must be positive")
    return {
        "contract_status": "ready",
        "inference_status": "not_run",
        "checkpoint": str(checkpoint),
        "norm_stats": str(stats),
        "norm_stats_sha256": digest,
        "action_horizon": action_horizon,
        "action_dt_s": 1 / 30,
        "sampling_modes": ["default", "rtc"],
    }
