"""Run one pack-plate Pi05 forward with synthetic wrist RGB, without services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    checkpoint = args.checkpoint.expanduser().resolve()
    config["model_path"] = str(checkpoint)
    asset_name = Path(config["norm_stats_path"]).name
    config["norm_stats_path"] = str(checkpoint / "assets" / asset_name)

    from .model import Model

    model = Model(config)
    obs = {"instruction": "plate", "env_idx": 0, "state": {}, "vision": {}}
    for side, opening in (("left", 0.4), ("right", 0.6)):
        obs["state"][f"{side}_ee_pose"] = np.array([0.3, 0.1, 0.2, 1, 0, 0, 0], np.float32)
        obs["state"][f"{side}_ee_joint_state"] = np.array([opening], np.float32)
        obs["vision"][f"cam_{side}_wrist"] = {"color": np.full((224, 224, 3), 127, np.uint8)}
    model.update_obs(obs)
    steps = model.get_action()
    if len(steps) != 32:
        raise ValueError("pack-plate forward did not produce 32 steps")
    for step in steps:
        for side in ("left", "right"):
            for key, shape in ((f"{side}_ee_pose", (7,)), (f"{side}_ee_joint_state", (1,))):
                value = np.asarray(step[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid pack-plate output: {key}")
    model.reset()
    print(json.dumps({"offline_forward": "passed", "images": "synthetic constant RGB",
                      "robot_task_success": "not_evaluated", "horizon": len(steps),
                      "metadata": model.runtime_metadata()}, indent=2))


if __name__ == "__main__":
    main()
