"""Create a portable model.ckpt/config/backbone/normalizer bundle from local assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "backbone", "normalizer", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/mv51.yaml")
    parser.add_argument("--checkpoint-sha256", help="Expected SHA-256 from checkpoint provenance")
    args = parser.parse_args()
    sources = {
        "model.ckpt": args.checkpoint,
        "backbone.pth": args.backbone,
        "action_normalizer.pt": args.normalizer,
        "resolved_config.yaml": args.config,
    }
    records = {
        name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for name, path in sources.items()
    }
    if args.checkpoint_sha256 and records["model.ckpt"]["sha256"] != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 does not match provenance")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        target = args.output / name
        if target.exists():
            if sha256(target) != records[name]["sha256"]:
                raise FileExistsError(f"Refusing to replace different asset: {target}")
            continue
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=args.output)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            if sha256(Path(temporary)) != records[name]["sha256"]:
                raise OSError(f"Copied asset failed verification: {name}")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    (args.output / "assets.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Verified SAPolicy bundle: {args.output.resolve()}")


if __name__ == "__main__":
    main()
