"""Resolve portable SAPolicy checkpoint bundles with XPolicyLab's shared resolver."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

POLICY_DIR = Path(__file__).resolve().parent


def resolve_assets(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    root = resolve_checkpoint_root(cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
    bundle = root if root.is_dir() else root.parent
    checkpoint = root / "model.ckpt" if root.is_dir() else root
    paths = {"model_path": checkpoint}
    for key, default in (
        ("cfg_file", "resolved_config.yaml"),
        ("normalizer_path", "action_normalizer.pt"),
        ("backbone_path", "backbone.pth"),
    ):
        value = cfg.get(key)
        path = Path(str(value)).expanduser() if value else bundle / default
        if not path.is_absolute():
            path = POLICY_DIR / path
        paths[key] = path.resolve()
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"SAPolicy {key} not found: {path}")
        cfg[key] = str(path)
    expected = cfg.get("checkpoint_source", "")
    if expected:
        if not str(expected).startswith("sha256:"):
            raise ValueError("SAPolicy checkpoint_source must be sha256:<digest>")
        with checkpoint.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != str(expected).removeprefix("sha256:"):
            raise ValueError("SAPolicy checkpoint does not match checkpoint_source SHA-256")
    loaded = yaml.safe_load(paths["cfg_file"].read_text())
    if not isinstance(loaded, dict) or not isinstance(loaded.get("model"), dict):
        raise ValueError("SAPolicy requires a checkpoint-matched resolved training YAML")
    # Use the recorded architecture and transforms; only deployment asset locations
    # and the explicitly selected weight variant differ from the training snapshot.
    from omegaconf import OmegaConf

    resolved = OmegaConf.create(loaded)
    if int(resolved.action_sequence_length) != int(cfg.get("action_horizon", 50)):
        raise ValueError("action_horizon must match the checkpoint training sequence length")
    if int(resolved.model.pipeline.action_cfg.action_dim) != 20:
        raise ValueError("This SAPolicy adapter requires the dual-arm pose18 + grip2 checkpoint")

    def enabled(value):
        return value is True or str(value).lower() == "true"

    if not enabled(resolved.get("use_relative_actions")) or not enabled(
        resolved.data.train_dataset.dataset_opts[0].get("body_frame_actions")
    ):
        raise ValueError(
            "This adapter requires the checkpoint's body-frame relative-action contract"
        )
    resolved.model.pipeline.load_pretrain_backbone = cfg["backbone_path"]
    resolved.model.use_ema = bool(cfg.get("use_ema", False))
    resolved.model.clear_output_dir = False
    resolved.ckpt_type = cfg.get("ckpt_type", "ema" if cfg.get("use_ema", False) else "raw")
    resolved.norm_stats_path = cfg["normalizer_path"]
    resolved.eval = {"normalizer_source": "path", "normalizer_path": cfg["normalizer_path"]}
    output = Path(str(cfg.get("workspace") or bundle / "inference-output")).expanduser().resolve()
    resolved.output_dir = str(output)
    resolved.model.output_dir = str(output / str(resolved.get("exp_name", "sapolicy")))
    cfg["resolved_cfg"] = resolved
    return cfg
