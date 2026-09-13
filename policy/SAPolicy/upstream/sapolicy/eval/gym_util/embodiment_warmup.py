"""Physically warm up a non-Panda embodiment from its native reset pose to the
Panda-aligned START pose the policy was trained on, before eval rollout begins.

See embodiment_warmup_ref.py for HDF5 lookup helpers kept import-light for
Panda-only eval paths.
"""
import numpy as np

_GRIPPER_OPEN_ACTION = -1.0
_ROBOTIQ85_PATCHED = False


def _ensure_calibrated_robotiq85_patch():
    global _ROBOTIQ85_PATCHED
    if _ROBOTIQ85_PATCHED:
        return
    from mimicgen.models.robosuite.grippers import CalibratedRobotiq85Gripper

    CalibratedRobotiq85Gripper.NATIVE_OPEN_VEC = np.array([-1.0, -1.0])
    CalibratedRobotiq85Gripper.NATIVE_CLOSED_VEC = np.array([1.0, 1.0])
    _ROBOTIQ85_PATCHED = True


def warmup_to_aligned_start(env, raw_env, ref_table, num_interp_steps=30, num_fixed_steps=40):
    """Move `raw_env`'s arm from its just-reset native pose to the Panda-aligned
    start POSITION (mean of Panda's own frame-0 eef positions), then hand off to eval.
    """
    from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory
    from mimicgen.env_interfaces.robosuite import MG_Square

    _ensure_calibrated_robotiq85_patch()
    env_interface = MG_Square(raw_env)
    start_pose = env_interface.get_robot_eef_pose()

    target_eef_pose = start_pose.copy()
    target_eef_pose[:3, 3] = ref_table["eef_pos_mean"]

    gripper_action = np.array([_GRIPPER_OPEN_ACTION])
    traj = WaypointTrajectory()
    traj.add_waypoint_sequence(WaypointSequence.from_poses(
        poses=start_pose[None],
        gripper_actions=gripper_action[None],
        action_noise=0.0,
    ))
    traj.add_waypoint_sequence_for_target_pose(
        pose=target_eef_pose, gripper_action=gripper_action,
        num_steps=num_interp_steps, skip_interpolation=False,
    )
    traj.add_waypoint_sequence(WaypointSequence.from_poses(
        poses=np.repeat(target_eef_pose[None], num_fixed_steps, axis=0),
        gripper_actions=np.repeat(gripper_action[None], num_fixed_steps, axis=0),
        action_noise=0.0,
    ))
    traj.pop_first()
    _execute_absolute(env=env, env_interface=env_interface, traj=traj)


def _execute_absolute(env, env_interface, traj):
    """Run mimicgen waypoint trajectory with absolute world-frame actions."""
    for seq in traj.waypoint_sequences:
        for j in range(len(seq)):
            waypoint = seq[j]
            action_pose = env_interface.target_pose_to_action(target_pose=waypoint.pose, relative=False)
            play_action = np.concatenate([action_pose, waypoint.gripper_action], axis=0)
            env.step(play_action)
