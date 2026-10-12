#!/usr/bin/env python3
"""Run the published C2R recipe with path-only relocation."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

from imagewam.runtime import run_training
from imagewam.utils.config_resolvers import register_default_resolvers


ADAPTER_DIR = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = ADAPTER_DIR / "PatchWAM"
CONFIG = ADAPTER_DIR / "assets" / "c2r_dr4_resolved.yaml"


def main() -> None:
    required = ["PATCHWAM_OUTPUT_DIR", "PATCHWAM_DATASET_DIR", "PATCHWAM_FLUX2_MODEL", "PATCHWAM_AE_MODEL"]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit("Missing required environment variables: " + ", ".join(missing))

    register_default_resolvers()
    cfg = OmegaConf.load(CONFIG)
    stats = ADAPTER_DIR / "assets" / "dataset_stats.json"
    nonidle = ADAPTER_DIR / "assets" / "nonidle_ranges.json"
    dataset = os.environ["PATCHWAM_DATASET_DIR"]

    overrides = {
        "output_dir": os.environ["PATCHWAM_OUTPUT_DIR"],
        "data.train.dataset_dirs": [dataset],
        "data.val.dataset_dirs": [dataset],
        "data.train.pretrained_norm_stats": str(stats),
        "data.val.pretrained_norm_stats": str(stats),
        "data.train.nonidle_filter_path": str(nonidle),
        "data.val.nonidle_filter_path": str(nonidle),
        "model.flux2_src_path": str(UPSTREAM_DIR / "vendor" / "flux2"),
        "model.flux2_model_path": os.environ["PATCHWAM_FLUX2_MODEL"],
        "model.ae_model_path": os.environ["PATCHWAM_AE_MODEL"],
        "model.qwen3_model_spec": os.environ.get("PATCHWAM_QWEN3_MODEL", "Qwen/Qwen3-4B"),
        "wandb.enabled": os.environ.get("PATCHWAM_WANDB", "false").lower() == "true",
        "wandb.workspace": os.environ.get("PATCHWAM_WANDB_ENTITY") or None,
    }
    if os.environ.get("PATCHWAM_RESUME"):
        overrides["resume"] = os.environ["PATCHWAM_RESUME"]
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    run_training(cfg)


if __name__ == "__main__":
    main()
