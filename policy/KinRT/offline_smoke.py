"""Reload a KinRT checkpoint and run one offline inference on an XPolicyLab observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from XPolicyLab.policy.KinRT.model import Model


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _synthetic_observation() -> dict:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "vision": {
            "cam_head": {"color": blank},
            "cam_left_wrist": {"color": blank.copy()},
            "cam_right_wrist": {"color": blank.copy()},
        },
        "state": {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.zeros(1, dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_ee_joint_state": np.zeros(1, dtype=np.float32),
        },
        "instruction": "stack the bowls",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=60000)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--repo-id", default="RoboDojo_lerobot_v30_video")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--action-chunk-size", type=int, default=50)
    parser.add_argument("--train-config-name", default="kinrt_full_robodojo")
    parser.add_argument("--actions-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_root is None:
        observation = _synthetic_observation()
        input_source = "synthetic"
    else:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError as error:
            if error.name not in {"lerobot.datasets", "lerobot.datasets.lerobot_dataset"}:
                raise
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        sample = LeRobotDataset(args.repo_id, root=args.dataset_root)[args.sample_index]
        state = _to_numpy(sample["observation.state"]).astype(np.float32).reshape(-1)
        observation = {
            "vision": {
                "cam_head": {"color": np.transpose(_to_numpy(sample["observation.images.cam_high"]), (1, 2, 0))},
                "cam_left_wrist": {
                    "color": np.transpose(_to_numpy(sample["observation.images.cam_left_wrist"]), (1, 2, 0))
                },
                "cam_right_wrist": {
                    "color": np.transpose(_to_numpy(sample["observation.images.cam_right_wrist"]), (1, 2, 0))
                },
            },
            "state": {
                "left_arm_joint_state": state[:6],
                "left_ee_joint_state": state[6:7],
                "right_arm_joint_state": state[7:13],
                "right_ee_joint_state": state[13:14],
            },
            "instruction": sample["task"],
        }
        input_source = str(args.dataset_root)

    load_started = time.perf_counter()
    model = Model(
        {
            "action_type": "joint",
            "env_cfg_type": "arx_x5",
            "checkpoint_path": str(args.checkpoint_root),
            "checkpoint_num": args.checkpoint_step,
            "train_config_name": args.train_config_name,
            "repo_id": args.repo_id,
            "action_chunk_size": args.action_chunk_size,
        }
    )
    model_load_seconds = time.perf_counter() - load_started
    model.update_obs(observation)
    inference_started = time.perf_counter()
    structured_actions = model.get_action()
    inference_seconds = time.perf_counter() - inference_started
    action_keys = (
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    )
    actions = np.stack(
        [np.concatenate([np.asarray(action[key]) for key in action_keys]) for action in structured_actions]
    )
    expected_shape = (args.action_chunk_size, 14)
    if actions.shape != expected_shape:
        raise RuntimeError(f"Expected action shape {expected_shape}, got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise RuntimeError("Inference returned NaN or infinite actions.")
    if args.actions_output is not None:
        args.actions_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.actions_output, actions)

    print(
        json.dumps(
            {
                "checkpoint_step": args.checkpoint_step,
                "train_config_name": args.train_config_name,
                "sample_index": args.sample_index,
                "instruction": observation["instruction"],
                "input_source": input_source,
                "action_shape": list(actions.shape),
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "actions_output": str(args.actions_output) if args.actions_output else None,
                "finite": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
