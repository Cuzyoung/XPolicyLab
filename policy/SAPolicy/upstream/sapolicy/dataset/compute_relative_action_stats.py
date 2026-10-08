"""
Compute per-timestep mean, std, and percentiles (0.02, 0.98) of relative action sequences
from RobomimicHDF5Dataset (supports CPGen / RoboSuite / RoboCasa)

Example:
    python compute_relative_action_stats.py \
        --hdf5 /mnt/nfs_client/minghuan/datasets/cpgen/datasets/generated/ThreePieceAssemblyWide/.../E73/tpawide-rgb-84-84-w-depth-ext-int.hdf5 \
        --dataset_type cpgen \
        --action_horizon 8 \
        --output stats_threepiecewide_h8.json
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from sapolicy.dataset.robomimic_hdf5 import RobomimicHDF5Dataset  # <-- 你的文件名

def compute_relative_action_stats_robomimic_hdf5(args):
    dataset = RobomimicHDF5Dataset(
        hdf5_path=args.hdf5,
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        split='train',
        use_train_val_split=False,
        train_val_split_ratio=0.9,
        action_sequence_length=args.action_horizon,
        normalize_actions=False,
        use_relative_actions=True,
        use_agentview=True,
        use_eye_in_hand=False,
        use_depth=False,
        use_proprioceptive=True,
        camera_name="agentview",
        min_depth=0.0,
        max_depth=5.0,
        max_episodes=args.max_episodes,
        action_orn_mode=args.action_orn_mode,
        observation_keys=[
            "image", "depth", "robot0_eye_in_hand_image", "robot0_eye_in_hand_depth",
            "robot0_eef_pos", "robot0_eef_quat", "robot0_eef_quat_site", "robot0_joint_pos", "robot0_gripper_qpos",
            "camera_intrinsics", "camera_extrinsics",
            "robot0_eye_in_hand_intrinsics", "robot0_eye_in_hand_extrinsics",
            "tcp_pixel_coords", "tcp_dir_x", "tcp_dir_y", "tcp_dir_z", "tcp_pos", "tcp_orn"
        ]
    )

    horizon = args.action_horizon
    sums = [None] * horizon
    sq_sums = [None] * horizon
    counts = [0] * horizon
    # Store list of actions for each timestep for percentile computation
    action_buffers = [list() for _ in range(horizon)]

    print(f"Computing per-timestep stats (horizon={horizon}) ...")
    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
            actions = sample["action"]  # shape (H, D)
            if actions.shape[0] < horizon:
                continue
        except Exception as e:
            print(f"⚠️ Error loading sample {i}: {e}")
            continue

        for t in range(horizon):
            a_t = actions[t]
            if sums[t] is None:
                sums[t] = np.zeros_like(a_t, dtype=np.float64)
                sq_sums[t] = np.zeros_like(a_t, dtype=np.float64)
            sums[t] += a_t
            sq_sums[t] += a_t ** 2
            counts[t] += 1
            action_buffers[t].append(a_t.copy())

    # Compute means, stds, and percentiles, min, max
    mean = {}
    std = {}
    perc_002 = {}
    perc_098 = {}
    min = {}
    max = {}
    for t in range(horizon):
        if counts[t] == 0:
            continue
        mean_t = sums[t] / counts[t]
        var_t = (sq_sums[t] / counts[t]) - (mean_t ** 2)
        std_t = np.sqrt(np.maximum(var_t, 1e-8))
        mean[f"a_t{t}"] = mean_t.tolist()
        std[f"a_t{t}"] = std_t.tolist()

        # Percentile computation
        per_t_actions = np.array(action_buffers[t])
        perc_002_t = np.percentile(per_t_actions, 2, axis=0).tolist()
        perc_098_t = np.percentile(per_t_actions, 98, axis=0).tolist()
        perc_002[f"a_t{t}"] = perc_002_t
        perc_098[f"a_t{t}"] = perc_098_t
        min[f"a_t{t}"] = np.min(per_t_actions, axis=0).tolist()
        max[f"a_t{t}"] = np.max(per_t_actions, axis=0).tolist()

    stats = {
        "mean": mean,
        "std": std,
        "percentile_0.02": perc_002,
        "percentile_0.98": perc_098,
        "min": min,
        "max": max,
        "total": int(sum(counts))
    }

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ Stats saved to {output_path}")
    print(f"Total samples used: {stats['total']}")
    print(f"Keys: {list(stats['mean'].keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-timestep relative action statistics")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--dataset_name", type=str, default="ThreePieceAssemblyWide")
    parser.add_argument("--dataset_type", type=str, default="cpgen", choices=["cpgen", "robosuite", "robocasa"])
    parser.add_argument("--action_horizon", type=int, default=16, help="Action sequence length")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--action_orn_mode", type=str, default="6d", choices=["6d", "quat", "euler", "rotmat"])

    args = parser.parse_args()
    if args.output is None:
        hdf5_path = Path(args.hdf5)
        args.output = f"{hdf5_path.parent}/relative_action_stats_{args.action_horizon}.json"
    compute_relative_action_stats_robomimic_hdf5(args)
