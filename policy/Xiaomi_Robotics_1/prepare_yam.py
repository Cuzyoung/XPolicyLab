"""Compatibility CLI delegating YAM preparation to process_data.sh."""

import argparse
import os
from pathlib import Path
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
    env = dict(os.environ, XR1_SOURCE_FORMAT="yam", RAW_DATA_ROOT=str(args.source.resolve()),
               OUTPUT_DIR=str(args.dataset.resolve()), DATA_CONFIG_NAME=args.name,
               XR1_INSTRUCTION=args.instruction, BATCH_SIZE=str(args.batch), XR1_PYTHON=sys.executable)
    subprocess.run(["bash", str(policy / "process_data.sh"),
                    "RoboDojo_real", args.name, "yam_dual", "ee"], check=True, env=env)


if __name__ == "__main__":
    main()
