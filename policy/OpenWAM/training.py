"""Translate XPolicy training arguments into the vendored OpenWAM trainer."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

POLICY = Path(__file__).resolve().parent


def build_command(args):
    if (args.bench, args.robot) not in {("RoboDojo_real", "yam_dual"), ("RoboDojo", "arx_x5")}:
        raise ValueError("Supported pairs: RoboDojo_real/yam_dual and RoboDojo/arx_x5")
    if args.action != "ee":
        raise ValueError("OpenWAM requires action_type=ee")
    if not args.gpu or any(not part.isdigit() for part in args.gpu.split(",")):
        raise ValueError("gpu_id must be comma-separated GPU indices")
    if args.finetune and args.resume:
        raise ValueError("finetune and resume are mutually exclusive")
    name = build_run_dir_name(
        dict(
            bench_name=args.bench,
            ckpt_name=args.run,
            env_cfg_type=args.robot,
            action_type=args.action,
            seed=args.seed,
        )
    )
    if not name or Path(name).name != name:
        raise ValueError("Training run identifiers must not contain path separators")
    output = (args.output or POLICY / "checkpoints" / name).resolve()
    if args.resume:
        output = args.resume.resolve()
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty training run: {output}")
    for path in (args.finetune, args.resume):
        if path and not (path / "config.yaml").is_file():
            raise ValueError(f"Checkpoint directory is missing config.yaml: {path}")
    if args.resume and not any(args.resume.glob("**/optimizer*")):
        # The upstream loader does the authoritative layout validation.
        print("[OpenWAM] resume requires saved optimizer/scheduler/RNG states", file=sys.stderr)
    if not args.dataset.is_dir():
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    variant = "real" if args.robot == "yam_dual" else "sim"
    overrides = list(args.overrides)
    protected = {
        "dataloader",
        "dataloader.embodiment",
        "dataloader.variant",
        "dataloader.action_mode",
        "dataloader.dataset_dir",
        "dataloader.num_frames",
        "dataloader.video_stride",
        "training.output_path",
        "training.finetune_ckpt_path",
        "training.resume_ckpt_path",
    }
    if any(item.split("=", 1)[0].lstrip("+~") in protected for item in overrides):
        raise ValueError("Hydra override cannot replace the XPolicy data/checkpoint contract")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(args.gpu.split(','))}",
        "scripts/train.py",
        "dataloader=robodojo",
        f"dataloader.embodiment={args.robot}",
        f"dataloader.variant={variant}",
        "dataloader.action_mode=eef",
        "dataloader.num_frames=33",
        "dataloader.video_stride=4",
        f"dataloader.dataset_dir={json.dumps(str(args.dataset.resolve()))}",
        f"training.output_path={json.dumps(str(output))}",
        f"project.seed={args.seed}",
    ]
    for field in ("finetune", "resume"):
        path = getattr(args, field)
        if path:
            command.append(f"training.{field}_ckpt_path={json.dumps(str(path.resolve()))}")
    return command + overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bench", "run", "robot", "action", "seed", "gpu"):
        parser.add_argument(name, type=int if name == "seed" else str)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dataset", type=Path, default=os.environ.get("OPENWAM_DATASET_DIR"))
    parser.add_argument("--output", type=Path, default=os.environ.get("OPENWAM_CHECKPOINT_DIR"))
    parser.add_argument(
        "--finetune", type=Path, default=os.environ.get("OPENWAM_FINETUNE_CKPT_PATH")
    )
    parser.add_argument("--resume", type=Path, default=os.environ.get("OPENWAM_RESUME_CKPT_PATH"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dataset is None:
        parser.error("OPENWAM_DATASET_DIR or --dataset is required")
    command = build_command(args)
    print(json.dumps({"command": command, "cuda_visible_devices": args.gpu}), flush=True)
    if not args.dry_run:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
        subprocess.run(command, cwd=POLICY / "OpenWAM", env=env, check=True)


if __name__ == "__main__":
    main()
