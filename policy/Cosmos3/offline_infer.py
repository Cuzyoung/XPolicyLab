"""Run one official Cosmos3 inference through the XPolicy adapter, without a server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from XPolicyLab.policy.Cosmos3.model import Model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("deploy.yml"))
    parser.add_argument("--prompt", default="Pick up the object.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
        },
        "state": {
            "arm_joint_state": np.zeros(7, dtype=np.float32),
            "ee_joint_state": np.zeros(1, dtype=np.float32),
        },
        "instruction": args.prompt,
    }
    policy = Model(config)
    policy.update_obs(obs)
    action = policy.get_action()
    packed = np.stack(
        [np.concatenate([step["arm_joint_state"], step["ee_joint_state"]]) for step in action]
    )
    print(json.dumps({
        "status": "ok",
        "checkpoint": config["checkpoint_path"],
        "action_shape": list(packed.shape),
        "finite": bool(np.isfinite(packed).all()),
        "minimum": float(packed.min()),
        "maximum": float(packed.max()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
