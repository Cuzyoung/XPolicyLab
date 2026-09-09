"""Validate native HDF5 and build checkpoint-compatible training statistics."""

import argparse
import json
import os

from XPolicyLab.policy.OpenWAM.model import _resolve_openwam_root


def prepare(root, robot):
    _resolve_openwam_root(None)
    from openwam.dataloader.robodojo import MultiTaskRoboDojoDataset

    dataset = MultiTaskRoboDojoDataset(
        root,
        embodiment=robot,
        variant="real" if robot == "yam_dual" else "sim",
        num_frames=33,
        video_stride=4,
        split="train",
        unify_action_map=["0-9", "34-43"],
        color_jitter={"enabled": False},
    )
    sample = dataset[0]
    return {
        "training_windows": len(dataset),
        "stats": dataset.normalization_stats_path,
        "sample_shapes": {
            key: list(value.shape) for key, value in sample.items() if hasattr(value, "shape")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench")
    parser.add_argument("run")
    parser.add_argument("robot", choices=["yam_dual", "arx_x5"])
    parser.add_argument("action", choices=["ee"])
    parser.add_argument("--dataset", default=os.environ.get("OPENWAM_DATASET_DIR"))
    args = parser.parse_args()
    if not args.dataset:
        parser.error("Set OPENWAM_DATASET_DIR to the native HDF5 dataset root")
    if args.bench != ("RoboDojo_real" if args.robot == "yam_dual" else "RoboDojo"):
        parser.error("bench does not match the robot profile")
    print(json.dumps(prepare(args.dataset, args.robot)))


if __name__ == "__main__":
    main()
