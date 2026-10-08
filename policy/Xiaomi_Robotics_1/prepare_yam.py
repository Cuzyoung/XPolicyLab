"""Compose existing ManiMux recording conversion and the XR1 Hydra data binding."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a file basename")
    policy = Path(__file__).resolve().parent
    workspace = policy.parents[2]
    target = policy / "xiaomi_robotics_1/xr1/configs/data" / (args.name + ".yaml")
    if target.exists():
        raise FileExistsError(f"Use a new data config name; refusing to replace {target}")
    subprocess.run([sys.executable, str(workspace / "scripts/datasets/prepare_xr1_yam_dataset.py"),
                    "--episodes", str(args.source), "--output", str(args.dataset),
                    "--config-name", args.name, "--instruction", args.instruction,
                    "--batch-size", str(args.batch)], check=True, cwd=workspace)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    source = (args.dataset / manifest["config"]["path"]).resolve()
    source.relative_to(args.dataset.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)


if __name__ == "__main__":
    main()
