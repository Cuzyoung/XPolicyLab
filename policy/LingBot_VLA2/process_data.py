#!/usr/bin/env python3
"""Convert XPolicyLab HDF5 trajectories to LingBot-VLA2 LeRobot data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

POLICY_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = POLICY_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from XPolicyLab.utils.process_data import (  # noqa: E402
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
)

CAMERA_MAP = {
    "cam_head": "camera_top",
    "cam_left_wrist": "camera_wrist_left",
    "cam_right_wrist": "camera_wrist_right",
}


def _decode_text(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    return value


def _episode_instruction(root: h5py.File) -> str:
    key = "instruction" if "instruction" in root else "instructions"
    if key not in root:
        raise KeyError("trajectory is missing instruction/instructions")
    value = _decode_text(root[key][()])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        values = [str(_decode_text(item)).strip() for item in value]
        values = [item for item in values if item]
        if values:
            return values[0]
    text = str(value).strip()
    if not text:
        raise ValueError("trajectory instruction is empty")
    return text


def _group_arrays(root: h5py.File, singular: str, plural: str) -> dict[str, np.ndarray]:
    if singular in root:
        group = root[singular]
    elif plural in root:
        group = root[plural]
    else:
        raise KeyError(f"trajectory is missing {singular}/{plural}")
    values = {key: np.asarray(group[key][()]) for key in group}
    for key, value in tuple(values.items()):
        if not key.endswith("s"):
            values.setdefault(f"{key}s", value)
    return values


def split_yam_vector(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split packed [left arm, left grip, right arm, right grip] vectors."""

    packed = np.asarray(values, dtype=np.float32)
    if packed.ndim != 2 or packed.shape[1] != 14:
        raise ValueError(f"packed YAM data must be (T, 14), got {packed.shape}")
    arms = np.concatenate([packed[:, :6], packed[:, 7:13]], axis=1)
    effectors = np.concatenate([packed[:, 6:7], packed[:, 13:14]], axis=1)
    return arms, effectors


def _resize_frames(frames: Any, height: int, width: int) -> np.ndarray:
    decoded = np.asarray(decode_image_bit(frames), dtype=np.uint8)
    if decoded.ndim == 3:
        decoded = decoded[None, ...]
    if decoded.ndim != 4 or decoded.shape[-1] != 3:
        raise ValueError(f"camera frames must be (T, H, W, 3), got {decoded.shape}")
    if decoded.shape[1:3] != (height, width):
        decoded = np.stack(
            [
                cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                for frame in decoded
            ]
        )
    return np.ascontiguousarray(decoded)


def load_episode(
    episode_path: Path,
    *,
    robot_action_dim_info: dict[str, list[int]],
    height: int,
    width: int,
) -> dict[str, Any]:
    with h5py.File(episode_path, "r") as root:
        state_data = _group_arrays(root, "state", "states")
        action_data = _group_arrays(root, "action", "actions")
        state = pack_robot_state(
            {"state": state_data},
            "joint",
            robot_action_dim_info,
            source_type="dataset",
            state_type="state",
        )
        action = pack_robot_state(
            {"action": action_data},
            "joint",
            robot_action_dim_info,
            source_type="dataset",
            state_type="action",
        )
        state_arm, state_effector = split_yam_vector(state)
        action_arm, action_effector = split_yam_vector(action)
        instruction = _episode_instruction(root)

        vision = root.get("vision")
        if vision is None:
            raise KeyError("trajectory is missing vision")
        images: dict[str, np.ndarray] = {}
        for source_name, target_name in CAMERA_MAP.items():
            if source_name not in vision:
                raise KeyError(f"trajectory is missing vision.{source_name}")
            camera = vision[source_name]
            image_key = "colors" if "colors" in camera else "color"
            if image_key not in camera:
                raise KeyError(f"trajectory is missing vision.{source_name}.colors")
            images[target_name] = _resize_frames(camera[image_key][()], height, width)

    horizon = state_arm.shape[0]
    arrays = [state_effector, action_arm, action_effector, *images.values()]
    if any(array.shape[0] != horizon for array in arrays):
        raise ValueError(f"trajectory fields have inconsistent lengths: {episode_path}")
    return {
        "state_arm": state_arm,
        "state_effector": state_effector,
        "action_arm": action_arm,
        "action_effector": action_effector,
        "images": images,
        "instruction": instruction,
        "length": horizon,
    }


def _discover_episodes(
    source_root: Path,
    env_cfg_type: str,
    raw_task_dirs: str | None,
    expert_data_num: int | None,
) -> list[Path]:
    if raw_task_dirs:
        task_names = [name.strip() for name in raw_task_dirs.split(",") if name.strip()]
        task_dirs = [source_root / name for name in task_names]
    else:
        task_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())

    episodes: list[Path] = []
    for task_dir in task_dirs:
        data_dir = task_dir / env_cfg_type / "data"
        task_episodes = sorted(data_dir.glob("episode_*.hdf5"))
        task_episodes.extend(sorted(data_dir.glob("episode_*.h5")))
        if expert_data_num is not None:
            task_episodes = task_episodes[:expert_data_num]
        if not task_episodes:
            raise FileNotFoundError(f"no episodes found under {data_dir}")
        episodes.extend(task_episodes)
    if not episodes:
        raise FileNotFoundError(f"no trajectory task directories found under {source_root}")
    return episodes


def _create_dataset(
    *,
    repo_id: str,
    output_dir: Path,
    fps: int,
    height: int,
    width: int,
    mode: str,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image_dtype = "video" if mode == "video" else "image"
    features = {
        "observation.state.arm.position": {
            "dtype": "float32",
            "shape": (12,),
            "names": [f"arm_joint_{index}" for index in range(12)],
        },
        "observation.state.effector.position": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_gripper", "right_gripper"],
        },
        "action.arm.position": {
            "dtype": "float32",
            "shape": (12,),
            "names": [f"arm_joint_{index}" for index in range(12)],
        },
        "action.effector.position": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_gripper", "right_gripper"],
        },
    }
    for camera_name in CAMERA_MAP.values():
        features[f"observation.images.{camera_name}"] = {
            "dtype": image_dtype,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="yam_dual",
        features=features,
        root=output_dir,
        use_videos=mode == "video",
        image_writer_processes=4,
        image_writer_threads=4,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("action_type", choices=["joint"])
    parser.add_argument("expert_data_num", type=int, nargs="?", default=None)
    parser.add_argument("--raw-task-dirs")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.env_cfg_type != "yam_dual":
        raise ValueError("LingBot_VLA2 currently supports env_cfg_type=yam_dual only")
    source_root = (args.source_root or WORKSPACE_ROOT / "data" / args.bench_name).resolve()
    setting = f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.action_type}"
    output_dir = (args.output_dir or POLICY_DIR / "data" / setting).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output dataset already exists: {output_dir}")
    episodes = _discover_episodes(
        source_root,
        args.env_cfg_type,
        args.raw_task_dirs,
        args.expert_data_num,
    )
    robot_info = get_robot_action_dim_info(args.env_cfg_type)
    dataset = _create_dataset(
        repo_id=setting,
        output_dir=output_dir,
        fps=args.fps,
        height=args.height,
        width=args.width,
        mode=args.mode,
    )
    total_frames = 0
    try:
        for index, episode_path in enumerate(episodes, start=1):
            episode = load_episode(
                episode_path,
                robot_action_dim_info=robot_info,
                height=args.height,
                width=args.width,
            )
            for frame_index in range(episode["length"]):
                frame = {
                    "observation.state.arm.position": episode["state_arm"][frame_index],
                    "observation.state.effector.position": episode["state_effector"][frame_index],
                    "action.arm.position": episode["action_arm"][frame_index],
                    "action.effector.position": episode["action_effector"][frame_index],
                    "task": episode["instruction"],
                }
                for camera_name, images in episode["images"].items():
                    frame[f"observation.images.{camera_name}"] = images[frame_index]
                dataset.add_frame(frame)
            dataset.save_episode()
            total_frames += episode["length"]
            print(f"[LingBot_VLA2] converted {index}/{len(episodes)}: {episode_path}")
    finally:
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()
        if hasattr(dataset, "finalize"):
            dataset.finalize()
    print(f"[LingBot_VLA2] dataset={output_dir} episodes={len(episodes)} frames={total_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
