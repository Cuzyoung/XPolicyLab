"""Hardware-free shared-server probe using real weights and explicit two-frame input."""

import argparse
import copy
import os

import numpy as np

from XPolicyLab.client_server.ws.model_client import WsModelClient
from XPolicyLab.utils.process_data import images_encoding


def synthetic_observation(period_s=0.1, seed=0):
    rng = np.random.default_rng(seed)
    obs = {
        "vision": {},
        "state": {},
        "instruction": "pass the ball",
        "data_format_version": "v1.0",
        "env_idx": 0,
        "additional_info": {
            "umi_dp": {"frame_times_ns": [1000000000, 1000000000 + round(period_s * 1e9)]}
        },
    }
    for side in ("left", "right"):
        for suffix in ("_prev", ""):
            obs["vision"][f"cam_{side}_wrist{suffix}"] = {
                "color": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
            }
            obs["state"][f"{side}_ee_pose{suffix}"] = np.array([0.3, 0.1, 0.4, 1, 0, 0, 0.0])
            obs["state"][f"{side}_ee_joint_state{suffix}"] = np.array([0.8], dtype=np.float32)
    return obs


def check_actions(actions, horizon):
    assert isinstance(actions, list) and len(actions) == horizon
    for step in actions:
        assert set(step) == {
            f"{side}_{key}" for side in ("left", "right") for key in ("ee_pose", "ee_joint_state")
        }
        for side in ("left", "right"):
            pose = np.asarray(step[f"{side}_ee_pose"])
            grip = np.asarray(step[f"{side}_ee_joint_state"])
            assert pose.shape == (7,) and grip.shape == (1,)
            assert np.isfinite(pose).all() and np.isfinite(grip).all()
            np.testing.assert_allclose(np.linalg.norm(pose[3:]), 1, atol=1e-6)


def run(server, seed=0, encoded=False):
    with WsModelClient(
        url=server, evaluation_id="umi-dp-offline", trial_id="probe", request_timeout_s=120
    ) as client:
        metadata = client.call("runtime_metadata")
        assert metadata["policy_family"] == "umi_dp"
        assert set(client.call("sampling_modes")) == {"default", "rtc"}
        obs = synthetic_observation(metadata["observation_period_s"], seed)
        if encoded:
            for camera in obs["vision"].values():
                camera["color"] = images_encoding([camera["color"]])[0][0]
        client.call("reset")
        actions = client.infer(obs)
        check_actions(actions, metadata["action_horizon"])
        # Model hooks and protocol payload exercise a real guided UNet forward.
        condition = np.array(
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
        weights = np.linspace(1, 0, len(actions), dtype=np.float32)
        actions = client.infer(
            obs,
            sampling={
                "mode": "rtc",
                "action_condition": condition,
                "condition_weights": weights,
                "beta": 5.0,
            },
        )
        check_actions(actions, metadata["action_horizon"])
        batch = [copy.deepcopy(obs), copy.deepcopy(obs)]
        batch[0]["env_idx"], batch[1]["env_idx"] = 7, 3
        client.call("update_obs_batch", batch)
        chunks = client.call("get_action_batch", [3, 7])
        assert len(chunks) == 2
        for chunk in chunks:
            check_actions(chunk, metadata["action_horizon"])
        client.call("reset")
        client.call("update_obs", obs)
        check_actions(client.call("get_action"), metadata["action_horizon"])
        print(
            f"UMI_DP shared server passed: H={metadata['action_horizon']}, "
            f"encoded={encoded}, default/rtc/batch/reset",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.server, args.seed, bool(int(os.environ.get("DEBUG_OBS_ENCODED", "0"))))


if __name__ == "__main__":
    main()
