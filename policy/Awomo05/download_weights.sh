#!/usr/bin/env bash
# Download the released checkpoint and required base weights.
set -euo pipefail

WEIGHTS_DIR=${AWOMO05_WEIGHTS:-$(cd "$(dirname "$0")" && pwd)/weights}
mkdir -p "$WEIGHTS_DIR"
python -m pip install -q "huggingface_hub>=0.24"
python - "$WEIGHTS_DIR" "${1:-all}" <<'PY'
import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

weights_dir = sys.argv[1]
mode = sys.argv[2]
if mode not in {"all", "checkpoint", "base"}:
    raise SystemExit("Usage: download_weights.sh [all|checkpoint|base]")
if mode in {"all", "checkpoint"}:
    from pathlib import Path
    import hashlib
    snapshot_download(
        "Auwomo/Awomo-0.5-Robodojo",
        revision="5a3e043cba7932e55e759ccd3a7950340983bcae",
        allow_patterns=["model.pt", "dataset_stats.json", "SHA256SUMS", "LICENSE.md"],
        local_dir=weights_dir,
    )
    for line in (Path(weights_dir) / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        file = (Path(weights_dir) / name).resolve()
        if file.parent != Path(weights_dir).resolve():
            raise ValueError("Unexpected path in SHA256SUMS")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
    print("Released checkpoint checksums verified")
if mode == "checkpoint":
    raise SystemExit(0)
hf_hub_download(
    "black-forest-labs/FLUX.2-klein-base-4B",
    "flux-2-klein-base-4b.safetensors",
    local_dir=os.path.join(weights_dir, "flux2-klein-base-4b"),
)
hf_hub_download(
    "black-forest-labs/FLUX.2-dev",
    "ae.safetensors",
    local_dir=os.path.join(weights_dir, "flux2-dev"),
)
snapshot_download(
    "Qwen/Qwen3-VL-4B-Instruct",
    local_dir=os.path.join(weights_dir, "Qwen3-VL-4B-Instruct"),
)
PY
printf 'Requested weights downloaded to %s\n' "$WEIGHTS_DIR"
