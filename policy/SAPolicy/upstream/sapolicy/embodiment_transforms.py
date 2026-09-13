"""Shared train/eval transforms for robot embodiment data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Type

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import torch
except Exception:
    torch = None


# width_mm = offset + slope * qpos[..., 0]; see scripts/measure_robotiq_rad_to_width.py.
ROBOTIQ_ANGLE_TO_WIDTH_MM = {
    "robotiq140": (131.7, -219.4),
    "robotiq85": (105.3, -174.2),
}

NATIVE_GRIPPER_FAMILIES = {
    "iiwa": "robotiq140",
    "ur5e": "robotiq85",
    "panda": "parallel_jaw",
    "sawyer": "parallel_jaw",
    "rethink": "parallel_jaw",
}

# Per-embodiment gripper class, precise enough to key GRIPPER_TCP_OFFSET_M (unlike
# NATIVE_GRIPPER_FAMILIES, which collapses calibrated/native Robotiq variants).
NATIVE_GRIPPER_CLASS = {
    "iiwa": "Robotiq140Gripper",
    "ur5e": "Robotiq85Gripper",
    "panda": "PandaGripper",
    "sawyer": "RethinkGripper",
}

# Per-gripper-class translation (m) along grip_site +z aligning open-state fingertips
# to PandaGripper's; ported from mimicgen-x's GRIPPER_TCP_OFFSET. Don't hand-edit.
GRIPPER_TCP_OFFSET_M = {
    "PandaGripper": +0.00000,
    "RethinkGripper": +0.00550,
    "Robotiq85Gripper": +0.00187,
    "CalibratedRobotiq85Gripper": +0.00457,
    "Robotiq140Gripper": -0.00166,
    "CalibratedRobotiq140Gripper": +0.00347,
    "YamGripper": +0.00660,
    "RobotiqHandEGripper": +0.00560,
    "WSG50Gripper": +0.01160,
    "OnRobotRG2Gripper": +0.01260,
}


def _normalise_gripper_family(gripper_family: Optional[str]) -> Optional[str]:
    if gripper_family is None:
        return None
    value = str(gripper_family)
    value = value.removeprefix("Calibrated").removesuffix("Gripper")
    return value.lower()


def _numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _restore_type(value: np.ndarray, reference: Any) -> Any:
    value = np.asarray(value, dtype=np.float32)
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return value


def _observation(sample: Any) -> Optional[MutableMapping[str, Any]]:
    if not isinstance(sample, MutableMapping):
        return None
    observation = sample.get("observation", sample)
    return observation if isinstance(observation, MutableMapping) else None


def quaternion_xyzw_to_rotation_6d(quaternion: Any) -> np.ndarray:
    """Convert xyzw quaternions to the repository's column-based 6D format."""
    quaternion = _numpy(quaternion)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {quaternion.shape}")
    flat = quaternion.reshape(-1, 4)
    matrix = R.from_quat(flat).as_matrix().reshape(quaternion.shape[:-1] + (3, 3))
    return matrix[..., :, :2].swapaxes(-1, -2).reshape(quaternion.shape[:-1] + (6,)).astype(np.float32)


def rotation_6d_to_matrix(rotation_6d: Any) -> np.ndarray:
    """Convert the column-based 6D representation to rotation matrices."""
    value = _numpy(rotation_6d).astype(np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"Expected rotation-6D shape (..., 6), got {value.shape}")
    assert np.isfinite(value).all(), "Non-finite value in rotation-6D input"
    a1 = value[..., :3]
    a2 = value[..., 3:]
    a1_norm = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = a1 / np.maximum(a1_norm, 1e-12)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    a2_norm = np.linalg.norm(a2, axis=-1, keepdims=True)
    assert a1_norm.min() > 1e-3 and a2_norm.min() > 1e-3, "Near-singular rotation-6D input (Gram-Schmidt basis collapse)"
    b2 = a2 / np.maximum(a2_norm, 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1).astype(np.float32)


def _rotation_to_repr(rotation: R, mode: str) -> np.ndarray:
    if mode == "axis_angle":
        return rotation.as_rotvec()
    if mode == "quat":
        return rotation.as_quat()
    if mode == "euler":
        return rotation.as_euler("xyz", degrees=False)
    if mode == "rotmat":
        matrix = rotation.as_matrix()
        return matrix.reshape(matrix.shape[:-2] + (9,))
    if mode == "6d":
        matrix = rotation.as_matrix()
        return matrix[..., :, :2].swapaxes(-1, -2).reshape(matrix.shape[:-2] + (6,))
    raise ValueError(f"Invalid action orientation mode: {mode}")


def _repr_to_rotation(value: Any, mode: str) -> R:
    value = _numpy(value)
    if mode == "6d":
        return R.from_matrix(rotation_6d_to_matrix(value).reshape(-1, 3, 3))
    if mode == "quat":
        return R.from_quat(value.reshape(-1, 4))
    if mode == "euler":
        return R.from_euler("xyz", value.reshape(-1, 3), degrees=False)
    if mode == "axis_angle":
        return R.from_rotvec(value.reshape(-1, 3))
    if mode == "rotmat":
        return R.from_matrix(value.reshape(-1, 3, 3))
    raise ValueError(f"Invalid action orientation mode: {mode}")


class Transform:
    """One transform with optional input and output directions."""

    def inputs(self, sample: Any, context: Any = None) -> Any:
        return sample

    def outputs(self, action: Any, context: Any = None) -> Any:
        return action


class DataTransform:
    """Ordered transform collection shared by dataset and eval code."""

    def __init__(self, transforms: Sequence[Transform] = ()):
        self.transforms = tuple(transforms)

    def inputs(self, sample: Any, context: Any = None) -> Any:
        for transform in self.transforms:
            sample = transform.inputs(sample, context=context)
        return sample

    def outputs(self, action: Any, context: Any = None) -> Any:
        for transform in reversed(self.transforms):
            action = transform.outputs(action, context=context)
        return action

    def __repr__(self) -> str:
        names = ", ".join(type(transform).__name__ for transform in self.transforms)
        return f"DataTransform([{names}])"


TRANSFORM_REGISTRY: dict[str, Type[Transform]] = {}


def register_transform(name: str):
    """Register a transform class by a short configuration name."""
    if not name:
        raise ValueError("Transform name must be non-empty")

    def decorator(cls: Type[Transform]) -> Type[Transform]:
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform {name!r} is already registered")
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return decorator


@register_transform("gripper_width")
class GripperWidthTransform(Transform):
    """Map native gripper qpos to the one-dimensional model state value."""

    def __init__(self, gripper_family: Optional[str] = None):
        self.gripper_family = _normalise_gripper_family(gripper_family)

    def inputs(self, sample: Any, context: Any = None) -> Any:
        observation = _observation(sample)
        if observation is None:
            return sample

        for key, value in list(observation.items()):
            if not key.endswith("_gripper_qpos"):
                continue
            qpos = _numpy(value).astype(np.float32)
            if qpos.shape[-1] == 1:
                adapted = qpos
            elif qpos.shape[-1] == 2:
                adapted = qpos[..., :1]
            elif qpos.shape[-1] == 6:
                family = self.gripper_family
                if family is None and isinstance(context, Mapping):
                    family = _normalise_gripper_family(context.get("gripper_family"))
                if family not in ROBOTIQ_ANGLE_TO_WIDTH_MM:
                    raise ValueError(
                        f"Unknown gripper family {family!r} for 6-DOF qpos; "
                        f"expected one of {sorted(ROBOTIQ_ANGLE_TO_WIDTH_MM)}"
                    )
                offset, slope = ROBOTIQ_ANGLE_TO_WIDTH_MM[family]
                adapted = offset + slope * qpos[..., 0:1]
            else:
                raise ValueError(f"Undefined gripper qpos shape: {qpos.shape}")
            observation[key] = _restore_type(adapted, value)
        return sample


@register_transform("eef_rotation_6d")
class EefRotation6DTransform(Transform):
    """Add canonical 6D eef rotations while retaining raw quaternions."""

    suffix = "_eef_quat_site"
    output_suffix = "_eef_rot_6d"

    def inputs(self, sample: Any, context: Any = None) -> Any:
        observation = _observation(sample)
        if observation is None:
            return sample
        for key, value in list(observation.items()):
            if key.endswith(self.suffix):
                # some wrappers already store 6D under the quat-named key
                if _numpy(value).shape[-1] != 4:
                    continue
                output_key = key[: -len(self.suffix)] + self.output_suffix
                observation[output_key] = _restore_type(
                    quaternion_xyzw_to_rotation_6d(value), value
                )
        return sample


def _context_observation(context: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(context, Mapping):
        observation = context.get("observation", context)
        if isinstance(observation, Mapping):
            return observation
    return None


def _eef_pose(context: Any, prefix: str = "robot0") -> tuple[np.ndarray, np.ndarray]:
    observation = _context_observation(context)
    if observation is None:
        raise ValueError("Action transform requires context with eef position and quaternion")
    try:
        return (
            _numpy(observation[f"{prefix}_eef_pos"]),
            _numpy(observation[f"{prefix}_eef_quat_site"]),
        )
    except KeyError as exc:
        raise ValueError(f"Action transform context is missing {exc.args[0]!r}") from exc


@register_transform("tcp_alignment")
class TCPAlignmentTransform(Transform):
    """Shift eef position between the raw grip_site frame and a per-gripper TCP
    frame: p_tcp = p_site + delta_G * R_site[:, 2] (GRIPPER_TCP_OFFSET_M)."""

    def __init__(self, gripper_type: Optional[str] = None):
        self.gripper_type = gripper_type

    def _resolve_gripper_type(self, context: Any = None) -> str:
        gripper_type = self.gripper_type
        if isinstance(context, Mapping) and context.get("gripper_type") is not None:
            gripper_type = context.get("gripper_type")
        if gripper_type not in GRIPPER_TCP_OFFSET_M:
            raise KeyError(
                f"No TCP offset registered for gripper {gripper_type!r}. Measure it with "
                f"mimicgen.scripts.measure_gripper_geometry and add the printed entry to "
                f"GRIPPER_TCP_OFFSET_M in {__file__}, or pass a registered gripper_type."
            )
        return gripper_type

    def _offset(self, context: Any = None) -> float:
        return GRIPPER_TCP_OFFSET_M[self._resolve_gripper_type(context)]

    @staticmethod
    def _approach_axis(quat_or_rotvec: Any, from_rotvec: bool) -> np.ndarray:
        """Approach-axis (z) column of the rotation, for arbitrary leading dims."""
        value = _numpy(quat_or_rotvec).astype(np.float64)
        leading_shape = value.shape[:-1]
        flat = value.reshape(-1, value.shape[-1])
        rotation = R.from_rotvec(flat) if from_rotvec else R.from_quat(flat)
        z_col = rotation.as_matrix()[..., :, 2]
        return z_col.reshape(leading_shape + (3,))

    def _shift_observation(self, observation: MutableMapping[str, Any], offset: float) -> None:
        for key, value in list(observation.items()):
            if not key.endswith("_eef_pos"):
                continue
            quat_key = f"{key[: -len('_eef_pos')]}_eef_quat_site"
            if quat_key not in observation:
                continue
            quat_or_6d = _numpy(observation[quat_key])
            if quat_or_6d.shape[-1] == 4:
                z_col = self._approach_axis(quat_or_6d, from_rotvec=False)
            elif quat_or_6d.shape[-1] == 6:
                z_col = rotation_6d_to_matrix(quat_or_6d)[..., :, 2]
            else:
                continue
            pos = _numpy(value).astype(np.float64)
            observation[key] = _restore_type(pos + offset * z_col, value)

    def _shift_action(self, actions_ref: Any, offset: float) -> Any:
        actions = _numpy(actions_ref).astype(np.float64)
        if actions.shape[-1] % 7 != 0:
            raise ValueError(f"Expected absolute action dimension divisible by 7, got {actions.shape}")
        n_arms = actions.shape[-1] // 7
        blocks = actions.reshape(actions.shape[:-1] + (n_arms, 7))
        shifted_blocks = []
        for arm in range(n_arms):
            block = blocks[..., arm, :]
            position = block[..., :3]
            rotvec = block[..., 3:6]
            z_col = self._approach_axis(rotvec, from_rotvec=True)
            shifted_blocks.append(
                np.concatenate([position + offset * z_col, rotvec, block[..., 6:7]], axis=-1)
            )
        shifted = shifted_blocks[0] if n_arms == 1 else np.concatenate(shifted_blocks, axis=-1)
        return _restore_type(shifted, actions_ref)

    def inputs(self, sample: Any, context: Any = None) -> Any:
        offset = self._offset(context)

        observation = _observation(sample)
        if observation is not None:
            self._shift_observation(observation, offset)

        # also shift context["observation"] so a later ActionSE3Transform in the
        # same DataTransform list sees the shifted reference pose
        context_observation = _context_observation(context)
        if context_observation is not None and context_observation is not observation:
            self._shift_observation(context_observation, offset)

        if isinstance(sample, MutableMapping) and sample.get("action") is not None:
            sample["action"] = self._shift_action(sample["action"], offset)
        return sample

    def outputs(self, action: Any, context: Any = None) -> Any:
        offset = self._offset(context)
        return self._shift_action(action, -offset)


@register_transform("action_se3")
class ActionSE3Transform(Transform):
    """Convert recorded/model actions and environment actions in both directions."""

    def __init__(
        self,
        action_orn_mode: str = "6d",
        use_relative_actions: bool = False,
        body_frame_actions: bool = True,
        dataset_type: str = "robosuite",
        cpgen_action_pos_scale: float = 0.05,
        cpgen_action_rot_scale: float = 0.5,
        cpgen_absolute_actions: bool = False,
        rotation_backend: str = "scipy",
    ):
        self.action_orn_mode = action_orn_mode
        self.use_relative_actions = bool(use_relative_actions)
        self.body_frame_actions = bool(body_frame_actions)
        self.dataset_type = str(dataset_type).lower()
        self.cpgen_action_pos_scale = float(cpgen_action_pos_scale)
        self.cpgen_action_rot_scale = float(cpgen_action_rot_scale)
        self.cpgen_absolute_actions = bool(cpgen_absolute_actions)
        self.rotation_backend = str(rotation_backend).lower()
        if self.rotation_backend not in {"scipy", "models"}:
            raise ValueError(
                "rotation_backend must be 'scipy' or 'models', "
                f"got {rotation_backend!r}"
            )

    def inputs(self, sample: Any, context: Any = None) -> Any:
        if not isinstance(sample, MutableMapping) or sample.get("action") is None:
            return sample
        actions_ref = sample["action"]
        actions = _numpy(actions_ref).astype(np.float32)
        observation = _context_observation(context) or _observation(sample)

        if not self._cpgen_delta_actions and self.use_relative_actions:
            if observation is None:
                raise ValueError("Relative action transform requires an observation")
            actions = self._absolute_to_relative(actions, observation)
        else:
            actions = self._absolute_to_model(actions)

        sample["action"] = _restore_type(actions, actions_ref)
        return sample

    def outputs(self, action: Any, context: Any = None) -> Any:
        action_ref = action
        actions = _numpy(action).astype(np.float32)
        if not self._cpgen_delta_actions and self.use_relative_actions:
            actions = self._relative_to_absolute(actions, context)
        else:
            actions = self._model_to_absolute(actions)
        return _restore_type(actions, action_ref)

    @property
    def _cpgen_delta_actions(self) -> bool:
        return self.dataset_type == "cpgen" and not self.cpgen_absolute_actions

    def _absolute_to_model(self, actions: np.ndarray) -> np.ndarray:
        if actions.shape[-1] % 7 != 0:
            raise ValueError(f"Expected absolute action dimension divisible by 7, got {actions.shape}")
        n_arms = actions.shape[-1] // 7
        blocks = actions.reshape(actions.shape[:-1] + (n_arms, 7))
        converted = []
        for arm in range(n_arms):
            block = blocks[..., arm, :]
            position = block[..., :3]
            rotvec = block[..., 3:6]
            if self._cpgen_delta_actions:
                # cpgen deltas are normalised to [-1, 1]; rescale to physical units
                position = position * self.cpgen_action_pos_scale
                rotvec = rotvec * self.cpgen_action_rot_scale
            rotation = R.from_rotvec(rotvec.reshape(-1, 3))
            rotation_repr = _rotation_to_repr(rotation, self.action_orn_mode)
            rotation_repr = rotation_repr.reshape(block.shape[:-1] + (rotation_repr.shape[-1],))
            converted.append(np.concatenate([position, rotation_repr, block[..., 6:7]], axis=-1))
        return np.concatenate(converted, axis=-1).astype(np.float32)

    def _absolute_to_relative(self, actions: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray:
        if actions.shape[-1] % 7 != 0:
            raise ValueError(f"Expected absolute action dimension divisible by 7, got {actions.shape}")
        n_arms = actions.shape[-1] // 7
        blocks = actions.reshape(actions.shape[:-1] + (n_arms, 7))
        converted = []
        for arm in range(n_arms):
            prefix = "robot0" if n_arms == 1 else ("left", "right")[arm]
            eef_pos, eef_quat = _eef_pose({"observation": observation}, prefix=prefix)
            block = blocks[..., arm, :]
            target_pos = block[..., :3]
            target_rot = R.from_rotvec(block[..., 3:6].reshape(-1, 3))
            current_pos = np.broadcast_to(eef_pos, target_pos.shape)
            current_quat = np.broadcast_to(
                eef_quat, target_pos.shape[:-1] + (4,)
            )
            current_rot = R.from_quat(current_quat.reshape(-1, 4))
            rel_pos_world = target_pos - current_pos
            if self.body_frame_actions:
                rel_pos = current_rot.inv().apply(rel_pos_world.reshape(-1, 3)).reshape(target_pos.shape)
                rel_rot = current_rot.inv() * target_rot
            else:
                rel_pos = rel_pos_world
                rel_rot = target_rot * current_rot.inv()
            rel_repr = _rotation_to_repr(rel_rot, self.action_orn_mode)
            rel_repr = rel_repr.reshape(block.shape[:-1] + (rel_repr.shape[-1],))
            converted.append(np.concatenate([rel_pos, rel_repr, block[..., 6:7]], axis=-1))
        if n_arms == 1:
            return converted[0].astype(np.float32)
        return np.concatenate(
            [block[..., :-1] for block in converted] + [block[..., -1:] for block in converted],
            axis=-1,
        ).astype(np.float32)

    def _model_to_absolute(self, actions: np.ndarray) -> np.ndarray:
        if actions.shape[-1] < 4:
            raise ValueError(f"Invalid model action shape: {actions.shape}")
        n_arms = 1 if actions.shape[-1] < 20 else 2
        if n_arms == 1:
            blocks = [actions]
        else:
            blocks = [actions[..., arm * 10 : (arm + 1) * 10] for arm in range(n_arms)]
        converted = []
        for block in blocks:
            rot_dim = block.shape[-1] - 4
            rotation = self._repr_to_rotation(
                block[..., 3 : 3 + rot_dim], self.action_orn_mode
            )
            rotvec = rotation.as_rotvec().reshape(block.shape[:-1] + (3,))
            position = block[..., :3]
            if self._cpgen_delta_actions:
                position = position / self.cpgen_action_pos_scale
                rotvec = rotvec / self.cpgen_action_rot_scale
            converted.append(np.concatenate([position, rotvec, block[..., -1:]], axis=-1))
        return np.concatenate(converted, axis=-1).astype(np.float32)

    def _relative_to_absolute(self, actions: np.ndarray, context: Any) -> np.ndarray:
        observation = _context_observation(context)
        if observation is None:
            raise ValueError("Relative action output requires context with eef pose")
        n_arms = 1 if actions.shape[-1] < 20 else 2
        if n_arms == 1:
            blocks = [actions]
        else:
            pose_dim = 9 * n_arms
            blocks = []
            for arm in range(n_arms):
                pose = actions[..., arm * 9 : (arm + 1) * 9]
                grip = actions[..., pose_dim + arm : pose_dim + arm + 1]
                blocks.append(np.concatenate([pose, grip], axis=-1))

        converted = []
        for arm, block in enumerate(blocks):
            prefix = "robot0" if n_arms == 1 else ("left", "right")[arm]
            eef_pos, eef_quat = _eef_pose({"observation": observation}, prefix=prefix)
            target_shape = block.shape[:-1]
            cur_pos = np.broadcast_to(eef_pos, target_shape + (3,)).reshape(-1, 3)
            cur_quat = np.broadcast_to(eef_quat, target_shape + (4,)).reshape(-1, 4)
            cur_rot = R.from_quat(cur_quat)
            rel_pos = block[..., :3].reshape(-1, 3)
            rel_rot = self._repr_to_rotation(block[..., 3:-1], self.action_orn_mode)
            if self.body_frame_actions:
                abs_pos = cur_rot.apply(rel_pos) + cur_pos
                abs_rot = cur_rot * rel_rot
            else:
                abs_pos = rel_pos + cur_pos
                abs_rot = rel_rot * cur_rot
            abs_rotvec = abs_rot.as_rotvec().reshape(block.shape[:-1] + (3,))
            converted.append(np.concatenate([abs_pos.reshape(block.shape[:-1] + (3,)), abs_rotvec, block[..., -1:]], axis=-1))

        if n_arms == 1:
            return converted[0].astype(np.float32)
        return np.concatenate(converted, axis=-1).astype(np.float32)

    def _repr_to_rotation(self, value: Any, mode: str) -> R:
        """rotation_backend='models' decodes 6D via the torch model's own Gram-Schmidt."""
        if mode == "6d" and self.rotation_backend == "models":
            if torch is None:
                raise ImportError("rotation_backend='models' requires torch")
            from sapolicy.models.utils.rotation import rotation_6d_to_matrix

            value_np = _numpy(value).astype(np.float32)
            value_t = torch.as_tensor(value_np)
            matrix = rotation_6d_to_matrix(value_t).detach().cpu().numpy()
            return R.from_matrix(matrix.reshape(-1, 3, 3))
        return _repr_to_rotation(value, mode)


@dataclass(frozen=True)
class TransformSpec:
    name: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformConfig:
    dataset_type: str
    embodiment: Optional[str]
    gripper_family: Optional[str]
    transforms: tuple[TransformSpec, ...]
    gripper_type: Optional[str] = None


def _default_specs() -> tuple[TransformSpec, ...]:
    return (
        TransformSpec("gripper_width"),
        TransformSpec("eef_rotation_6d"),
        TransformSpec("action_se3"),
    )


def _tcp_aligned_specs() -> tuple[TransformSpec, ...]:
    return (
        TransformSpec("gripper_width"),
        TransformSpec("eef_rotation_6d"),
        TransformSpec("tcp_alignment"),
        TransformSpec("action_se3"),
    )


# This table is deliberately small. Add a row when a dataset/embodiment needs a
# different transform combination or a concrete transform argument.
TRANSFORM_CONFIGS = {
    ("robosuite", "iiwa"): TransformConfig(
        "robosuite", "iiwa", "robotiq140", _default_specs()
    ),
    ("robosuite", "ur5e"): TransformConfig(
        "robosuite", "ur5e", "robotiq85", _default_specs()
    ),
    ("robosuite", "panda"): TransformConfig(
        "robosuite", "panda", "parallel_jaw", _default_specs()
    ),
    ("robocasa", None): TransformConfig(
        "robocasa", None, None, _default_specs()
    ),
    ("cpgen", None): TransformConfig("cpgen", None, None, _default_specs()),
    # off-native gripper swaps: plain "iiwa"/"ur5e" would resolve the native gripper
    ("cpgen", "iiwa_robotiq85"): TransformConfig(
        "cpgen", "iiwa_robotiq85", "robotiq85", _default_specs()
    ),
    ("cpgen", "ur5e_robotiqhande"): TransformConfig(
        "cpgen", "ur5e_robotiqhande", "robotiqhande", _default_specs()
    ),
    # "<embodiment>_tcp" opts into TCPAlignmentTransform; same gripper/family otherwise
    ("robosuite", "iiwa_tcp"): TransformConfig(
        "robosuite", "iiwa_tcp", "robotiq140", _tcp_aligned_specs(),
        gripper_type=NATIVE_GRIPPER_CLASS["iiwa"],
    ),
    ("robosuite", "ur5e_tcp"): TransformConfig(
        "robosuite", "ur5e_tcp", "robotiq85", _tcp_aligned_specs(),
        gripper_type=NATIVE_GRIPPER_CLASS["ur5e"],
    ),
    ("robosuite", "panda_tcp"): TransformConfig(
        "robosuite", "panda_tcp", "parallel_jaw", _tcp_aligned_specs(),
        gripper_type=NATIVE_GRIPPER_CLASS["panda"],
    ),
    ("cpgen", "iiwa_tcp"): TransformConfig(
        "cpgen", "iiwa_tcp", "robotiq140", _tcp_aligned_specs(),
        gripper_type=NATIVE_GRIPPER_CLASS["iiwa"],
    ),
    ("cpgen", "ur5e_tcp"): TransformConfig(
        "cpgen", "ur5e_tcp", "robotiq85", _tcp_aligned_specs(),
        gripper_type=NATIVE_GRIPPER_CLASS["ur5e"],
    ),
    ("cpgen", "panda_tcp"): TransformConfig(
        "cpgen", "panda_tcp", "parallel_jaw", _tcp_aligned_specs(),
        gripper_type=NATIVE_GRIPPER_CLASS["panda"],
    ),
    ("cpgen", "iiwa_robotiq85_tcp"): TransformConfig(
        "cpgen", "iiwa_robotiq85_tcp", "robotiq85", _tcp_aligned_specs(),
        gripper_type="Robotiq85Gripper",
    ),
    ("cpgen", "ur5e_robotiqhande_tcp"): TransformConfig(
        "cpgen", "ur5e_robotiqhande_tcp", "robotiqhande", _tcp_aligned_specs(),
        gripper_type="RobotiqHandEGripper",
    ),
}


def resolve_transform_config(
    dataset_type: str,
    embodiment: Optional[str] = None,
    gripper_family: Optional[str] = None,
) -> TransformConfig:
    """Resolve a small default profile; callers can replace its specs explicitly."""
    dataset_type = str(dataset_type).lower()
    embodiment = embodiment.lower() if isinstance(embodiment, str) else embodiment
    gripper_family = _normalise_gripper_family(gripper_family)
    config = TRANSFORM_CONFIGS.get((dataset_type, embodiment))
    if config is None:
        config = TRANSFORM_CONFIGS.get((dataset_type, None))
    gripper_family = gripper_family or (config.gripper_family if config is not None else None)
    if gripper_family is None and embodiment is not None:
        gripper_family = NATIVE_GRIPPER_FAMILIES.get(embodiment)
    if config is not None:
        return TransformConfig(
            dataset_type=config.dataset_type,
            embodiment=embodiment,
            gripper_family=gripper_family,
            transforms=config.transforms,
            gripper_type=config.gripper_type,
        )
    return TransformConfig(dataset_type, embodiment, gripper_family, _default_specs())


def build_data_transform(config: TransformConfig, **overrides: Any) -> DataTransform:
    """Instantiate a transform config without sharing mutable instances."""
    transforms = []
    for spec in config.transforms:
        kwargs = dict(spec.kwargs)
        if spec.name == "gripper_width" and config.gripper_family is not None:
            kwargs.setdefault("gripper_family", config.gripper_family)
        if spec.name == "action_se3":
            kwargs.setdefault("dataset_type", config.dataset_type)
        if spec.name == "tcp_alignment" and config.gripper_type is not None:
            kwargs.setdefault("gripper_type", config.gripper_type)
        kwargs.update(overrides.get(spec.name, {}))
        try:
            transform_type = TRANSFORM_REGISTRY[spec.name]
        except KeyError as exc:
            raise KeyError(f"Unknown transform {spec.name!r}") from exc
        transforms.append(transform_type(**kwargs))
    return DataTransform(transforms)


__all__ = [
    "ActionSE3Transform",
    "DataTransform",
    "EefRotation6DTransform",
    "GRIPPER_TCP_OFFSET_M",
    "GripperWidthTransform",
    "NATIVE_GRIPPER_CLASS",
    "NATIVE_GRIPPER_FAMILIES",
    "ROBOTIQ_ANGLE_TO_WIDTH_MM",
    "TCPAlignmentTransform",
    "TRANSFORM_REGISTRY",
    "TRANSFORM_CONFIGS",
    "Transform",
    "TransformConfig",
    "TransformSpec",
    "build_data_transform",
    "quaternion_xyzw_to_rotation_6d",
    "register_transform",
    "resolve_transform_config",
    "rotation_6d_to_matrix",
]
