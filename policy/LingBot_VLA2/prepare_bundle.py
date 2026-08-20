#!/usr/bin/env python3
"""Create a deployable bundle manifest from an official training run."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

BUNDLE_SCHEMA_VERSION = "xpolicylab.lingbot_vla2_yam_bundle.v1"
CAMERAS = ["camera_top", "camera_wrist_left", "camera_wrist_right"]


def _step(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("global_step_"))
    except ValueError:
        return -1


def _latest_hf_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("checkpoints/global_step_*/hf_ckpt"), key=_step)
    candidates = [
        path for path in candidates if (path / "model.safetensors.index.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no complete HuggingFace checkpoint under {run_dir}")
    return candidates[-1]


def _source_revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--native-hz", type=float, required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    training_config = run_dir / "lingbotvla_cli.yaml"
    if not training_config.is_file():
        raise FileNotFoundError(f"missing official training config: {training_config}")
    checkpoint = _latest_hf_checkpoint(run_dir)
    norm_stats = run_dir / "norm_stats.json"
    robot_config = run_dir / "robot_config.yaml"
    shutil.copy2(args.norm_stats.resolve(), norm_stats)
    shutil.copy2(args.robot_config.resolve(), robot_config)

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model": {
            "family": "lingbot-vla-v2",
            "official_source_revision": _source_revision(args.source_root.resolve()),
        },
        "artifacts": {
            "training_config": str(training_config.relative_to(run_dir)),
            "checkpoint": str(checkpoint.relative_to(run_dir)),
            "norm_stats": str(norm_stats.relative_to(run_dir)),
            "robot_config": str(robot_config.relative_to(run_dir)),
        },
        "control": {
            "native_hz": args.native_hz,
            "action_horizon": args.action_horizon,
            "action_space": "absolute_joint_position",
        },
        "embodiment": {
            "name": "yam_dual",
            "arm_dofs": [6, 6],
            "gripper_dofs": [1, 1],
            "cameras": CAMERAS,
        },
    }
    output_path = run_dir / "bundle.yaml"
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
