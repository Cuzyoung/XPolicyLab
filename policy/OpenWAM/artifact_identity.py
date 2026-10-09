"""Streaming deployment fingerprints; never deserialize model weights."""

import hashlib
import re
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_identity(checkpoint):
    root = Path(checkpoint).resolve()
    candidates = []
    for path in root.glob("checkpoint_step_*.safetensors"):
        match = re.fullmatch(r"checkpoint_step_(\d+)\.safetensors", path.name)
        if match:
            candidates.append((int(match[1]), path))
    if not candidates:
        raise ValueError("No numbered OpenWAM checkpoint found")
    _, weights = max(candidates, key=lambda item: item[0])
    stats = root / "normalization_stats.npy"
    return {
        "checkpoint_path": str(root),
        "checkpoint_file": weights.name,
        "checkpoint_sha256": sha256(weights),
        "training_config_sha256": sha256(root / "config.yaml"),
        "norm_stats_path": str(stats),
        "norm_stats_sha256": sha256(stats),
    }
