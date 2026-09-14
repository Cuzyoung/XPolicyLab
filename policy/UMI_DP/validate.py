"""Offline forward and optional numerical comparison against the source PolicyRunner.

The optional --legacy-root is only a validation reference. Production imports
and all serving/training code are fully contained in this policy directory.
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .debug_client import check_actions, synthetic_observation
from .model import Model
from .transforms import from_pose9, matrix_pose, pose_matrix, relative_condition


def recorded_observation(model, dataset_path, anchor=100):
    from .upstream.diffusion_policy.dataset.lerobot_image_dataset import LeRobotImageDataset

    cache = dataset_path / "cache/images_224"
    dataset = LeRobotImageDataset(
        dataset_path=str(dataset_path),
        shape_meta=model.shape_meta,
        image_cache_dir=str(cache) if cache.is_dir() else None,
        use_mmap=cache.is_dir(),
        action_mode="relative_trajectory",
        observation_mode="relative",
        pose_frame="world",
        position_unit="m",
        rotation_layout="column",
    )
    delta = round(model._metadata["observation_period_s"] * model._metadata["source_fps"])
    indices = [anchor - delta, anchor]
    if indices[0] < 0 or not any(
        ep.start <= indices[0] < indices[1] < ep.end for ep in dataset.episodes
    ):
        raise ValueError("Recorded validation window crosses an episode boundary")
    obs = synthetic_observation(model._metadata["observation_period_s"])
    obs["additional_info"]["umi_dp"]["frame_times_ns"] = [
        round(index / dataset.info["fps"] * 1e9) for index in indices
    ]
    for arm, side in enumerate(("left", "right")):
        rgb = dataset._video_source(dataset.DEFAULT_IMAGE_KEY_MAP[f"camera{arm}_rgb"]).read(indices)
        for j, suffix in enumerate(("_prev", "")):
            values = dataset.states[indices[j], arm * 10 : (arm + 1) * 10]
            obs["state"][f"{side}_ee_pose{suffix}"] = matrix_pose(from_pose9(values[:9]))
            obs["state"][f"{side}_ee_joint_state{suffix}"] = values[9:].copy()
            obs["vision"][f"cam_{side}_wrist{suffix}"]["color"] = rgb[j]
    return obs


def seed_all(seed):
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--source-fps", type=float, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rtc-guidance", choices=("pigdm", "soft_inpaint"))
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    model = Model(
        {
            "checkpoint_path": str(args.checkpoint),
            "source_fps": args.source_fps,
            "action_type": "ee",
            "env_cfg_type": "tianji_umi",
            "device": args.device,
            "rtc_guidance": args.rtc_guidance or "pigdm",
        }
    )
    report = model.runtime_metadata()
    obs = (
        recorded_observation(model, args.dataset)
        if args.dataset
        else synthetic_observation(report["observation_period_s"])
    )
    model.update_obs(obs)
    new_tensors = {key: value.copy() for key, value in model._observations[0][0].items()}
    seed_all(args.seed)
    start = time.perf_counter()
    actions = model.get_action()
    report["forward_ms"] = (time.perf_counter() - start) * 1000
    check_actions(actions, model.horizon)
    new_action = model.last_native_action.copy()
    report["input"] = (
        "recorded LeRobot window" if args.dataset else "explicit synthetic debug window"
    )
    report["action_shape"] = list(new_action.shape)
    report["finite"] = bool(np.isfinite(new_action).all())
    if args.rtc_guidance:
        absolute = np.array(
            [
                np.r_[
                    step["left_ee_pose"],
                    step["left_ee_joint_state"],
                    step["right_ee_pose"],
                    step["right_ee_joint_state"],
                ]
                for step in actions
            ]
        )
        weights = np.linspace(1, 0, model.horizon, dtype=np.float32)
        raw_condition = relative_condition(absolute, model._observations[0][1], weights)
        seed_all(args.seed)
        start = time.perf_counter()
        model.get_action_rtc(
            {"mode": "rtc", "action_condition": absolute, "condition_weights": weights, "beta": 5.0}
        )
        report["rtc_forward_ms"] = (time.perf_counter() - start) * 1000
        new_rtc = model.last_native_action.copy()
    if args.legacy_root:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        sys.path.insert(0, str(args.legacy_root / "policies/diffusion_policy"))
        sys.path.insert(0, str(args.legacy_root))
        from deploy.tianji.observation_adapter import ObservationAdapter
        from deploy.tianji.policy_runner import PolicyRunner

        runner = PolicyRunner(args.checkpoint, device=args.device)
        runner.load()
        frames = []
        for j, suffix in enumerate(("_prev", "")):
            frames.append(
                SimpleNamespace(
                    timestamp=obs["additional_info"]["umi_dp"]["frame_times_ns"][j] / 1e9,
                    tcp=np.stack(
                        [
                            pose_matrix(obs["state"][f"{side}_ee_pose{suffix}"])
                            for side in ("left", "right")
                        ]
                    ),
                    gripper=np.array(
                        [
                            obs["state"][f"{side}_ee_joint_state{suffix}"][0]
                            for side in ("left", "right")
                        ]
                    ),
                    rgb=[
                        obs["vision"][f"cam_{side}_wrist{suffix}"]["color"]
                        for side in ("left", "right")
                    ],
                )
            )
        window = ObservationAdapter(runner.shape_meta, period_s=runner.observation_period_s).build(
            frames
        )
        report["observation_max_abs_error"] = max(
            float(np.max(np.abs(new_tensors[key] - value))) for key, value in window.tensors.items()
        )
        for key, value in window.tensors.items():
            np.testing.assert_array_equal(new_tensors[key], value)
        seed_all(args.seed)
        old_action = runner.predict(window).action
        report["native_action_max_abs_error"] = float(np.max(np.abs(new_action - old_action)))
        np.testing.assert_allclose(new_action, old_action, atol=1e-6, rtol=1e-6)
        report["legacy_parity"] = "passed"
        if args.rtc_guidance:
            runner.enable_rtc(mode=args.rtc_guidance)
            seed_all(args.seed)
            old_rtc = runner.predict(window, condition=raw_condition, weights=weights).action
            report["rtc_native_action_max_abs_error"] = float(np.max(np.abs(new_rtc - old_rtc)))
            np.testing.assert_allclose(new_rtc, old_rtc, atol=1e-6, rtol=1e-6)
            report["legacy_rtc_parity"] = "passed"
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
