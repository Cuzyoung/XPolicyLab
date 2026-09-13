"""Sample-based Aloha observer-ring TCP HDF5 loader for cross-domain TCP training.

Public stores under ``.../public/rerender/aloha_ring24_{randompose,taskpose}.hdf5``
are *not* Robomimic demo trajectories: each ``samples/<id>`` entry is a single
RGB-D frame with dual-arm TCP labels and camera attrs (schema_version >= 5).
This dataset emits Robomimic-compatible batches for the joint CombinedLoader
TCP slot (``loss_mode="tcp"``), analogous to NutAssembly ``crossNutTcpOOD``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from sapolicy.embodiment_transforms import quaternion_xyzw_to_rotation_6d as _quaternion_xyzw_to_rotation_6d
from sapolicy.logger import Log


def _parse_literal(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def _as_4x4(mat: Any) -> np.ndarray:
    arr = np.asarray(_parse_literal(mat), dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Expected 4x4 matrix with 16 entries, got shape {arr.shape}")
    return arr.reshape(4, 4)


def _as_3x3_stack(mat: Any, n: int) -> np.ndarray:
    arr = np.asarray(_parse_literal(mat), dtype=np.float64)
    if arr.size == 9 * n:
        return arr.reshape(n, 3, 3)
    if arr.shape == (n, 3, 3):
        return arr
    raise ValueError(f"Expected {n} 3x3 matrices, got shape {arr.shape}")


def _rotmat_to_quat_xyzw(rotmats: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 4) xyzw."""
    return R.from_matrix(rotmats).as_quat().astype(np.float32)


class AlohaRingTcpSampleDataset(Dataset):
    """Cross-domain TCP samples from an Aloha observer-ring HDF5.

    Each ``__getitem__`` returns the same dict schema as ``RobomimicHDF5Dataset``:
    ``observation`` (agentview image/depth/TCP/state), dummy ``action``, ``prompt``.

    TCP tensors are stacked as ``[T, num_arms, C]`` to match RoboTwin bimanual.
    """

    def __init__(
        self,
        hdf5_path: str,
        dataset_name: str = "AlohaRingTcp24",
        task_description: str = "Aloha dual-arm TCP",
        observation_keys: Optional[List[str]] = None,
        action_sequence_length: int = 16,
        obs_hist_length: int = 3,
        use_depth: bool = True,
        use_state: bool = True,
        normalize_actions: bool = False,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
        transforms: Optional[List[Any]] = None,
        num_arms: int = 2,
        arm_obs_prefixes: Optional[List[str]] = None,
        tcp_orn_already_aligned: bool = True,
        action_dim: int = 20,
        ring_indices: Optional[Sequence[int]] = None,
        exclude_ring_indices: Optional[Sequence[int]] = None,
        max_samples: Optional[int] = None,
        split: str = "train",
        use_train_val_split: bool = False,
        train_val_split_ratio: float = 0.9,
        split_seed: int = 42,
        # Accepted for Hydra compatibility with RobomimicYAML knobs (unused).
        dataset_type: str = "cpgen",
        use_relative_actions: bool = True,
        action_orn_mode: str = "6d",
        norm_type: str = "minmax",
        use_task_description: bool = True,
        cpgen_absolute_actions: bool = True,
        body_frame_actions: bool = True,
        load_to_memory: bool = False,
        cache_dir: Optional[str] = None,
        cache_mmap: bool = False,
        camera_pair_choices: Optional[Dict[str, Any]] = None,
        prompt_prefix: str = "Aloha",
        tcp_shift_along_z_m: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs  # tolerate extra Hydra fields
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Aloha ring TCP HDF5 not found: {self.hdf5_path}")

        self.dataset_name = dataset_name
        self.dataset_type = str(dataset_type).lower()
        self.task_description = task_description
        self.use_task_description = bool(use_task_description)
        self.observation_keys = list(
            observation_keys
            or [
                "image",
                "depth",
                "robot0_eef_pos",
                "robot0_eef_quat_site",
                "robot0_gripper_qpos",
                "tcp_pixel_coords",
                "tcp_dir_x",
                "tcp_dir_y",
                "tcp_dir_z",
                "tcp_pos",
                "tcp_orn",
            ]
        )
        self.action_sequence_length = int(action_sequence_length)
        self.obs_hist_length = int(obs_hist_length)
        self.use_depth = bool(use_depth)
        self.use_state = bool(use_state)
        self.normalize_actions = bool(normalize_actions)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        raw_transforms = [] if transforms is None else list(transforms)
        self.num_arms = int(num_arms)
        self.arm_obs_prefixes = list(arm_obs_prefixes or ["left", "right"])
        if len(self.arm_obs_prefixes) != self.num_arms:
            raise ValueError(
                f"arm_obs_prefixes length {len(self.arm_obs_prefixes)} != num_arms {self.num_arms}"
            )
        self.tcp_orn_already_aligned = bool(tcp_orn_already_aligned)
        self.action_dim = int(action_dim)
        self.camera_names = ["agentview"]
        self.normalizer = None  # TCP-only branch; actions are dummy zeros
        self.prompt_prefix = str(prompt_prefix)
        # Shift the labelled TCP along its own z (approach) axis, in metres, before
        # projecting: lets a render set labelled at one site (e.g. YAM `pinch`, 0.138 m
        # on link_6 z) supervise a model trained on another (ABC `grasp_site`, 0.1347 m).
        self.tcp_shift_along_z_m = float(tcp_shift_along_z_m)

        include = None if ring_indices is None else {int(i) for i in ring_indices}
        exclude = set() if exclude_ring_indices is None else {int(i) for i in exclude_ring_indices}

        self._hdf5_file: Optional[h5py.File] = None
        with h5py.File(self.hdf5_path, "r") as f:
            if "samples" not in f:
                raise KeyError(f"{self.hdf5_path} missing top-level 'samples' group")
            sample_names = sorted(f["samples"].keys())
            # Avoid per-sample attr I/O when no ring filter is requested (10k opens
            # on network FS can take minutes).
            if include is None and not exclude:
                kept = list(sample_names)
            else:
                kept = []
                for name in sample_names:
                    ri = int(f["samples"][name].attrs["camera_ring_index"])
                    if include is not None and ri not in include:
                        continue
                    if ri in exclude:
                        continue
                    kept.append(name)
            if max_samples is not None:
                kept = kept[: int(max_samples)]

        if use_train_val_split and kept:
            rng = np.random.RandomState(int(split_seed))
            order = np.arange(len(kept))
            rng.shuffle(order)
            n_train = int(len(kept) * float(train_val_split_ratio))
            sel = order[:n_train] if split == "train" else order[n_train:]
            kept = [kept[i] for i in sel]

        self.sample_names = kept
        if not self.sample_names:
            raise RuntimeError(
                f"No Aloha ring samples kept from {self.hdf5_path} "
                f"(include={include}, exclude={exclude})"
            )

        if not raw_transforms:
            self.transforms = lambda x: x
        else:
            self.transforms = Compose(raw_transforms)

        Log.info(
            f"[AlohaRingTcp] {self.dataset_name}: {len(self.sample_names)} samples "
            f"from {self.hdf5_path.name} (depth={self.use_depth}, arms={self.num_arms})"
        )

    def _init_file(self) -> h5py.File:
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def __len__(self) -> int:
        return len(self.sample_names)

    def __del__(self):
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass
            self._hdf5_file = None

    def _tile_time(self, arr: np.ndarray) -> np.ndarray:
        """Broadcast a single-frame array to obs_hist_length on axis 0."""
        return np.repeat(arr[None, ...], self.obs_hist_length, axis=0)

    def _normalize_tcp_uvd(self, uvd: np.ndarray, height: int, width: int) -> np.ndarray:
        out = uvd.astype(np.float32).copy()
        out[..., 0] /= max(width - 1, 1)
        out[..., 1] /= max(height - 1, 1)
        d = np.clip(out[..., 2], self.min_depth, self.max_depth)
        out[..., 2] = (d - self.min_depth) / (self.max_depth - self.min_depth + 1e-8)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        f = self._init_file()
        name = self.sample_names[idx]
        sm = f["samples"][name]
        attrs = sm.attrs

        rgb = np.asarray(sm["rgb"], dtype=np.float32) / 255.0  # H, W, 3
        height, width = int(rgb.shape[0]), int(rgb.shape[1])

        observation: Dict[str, Any] = {
            "image": {"agentview": self._tile_time(rgb)},
        }

        if self.use_depth and "depth" in self.observation_keys and "depth" in sm:
            depth = np.asarray(sm["depth"], dtype=np.float32)
            if depth.ndim == 2:
                depth = depth[..., None]
            depth = np.clip(depth, self.min_depth, self.max_depth)
            depth = (depth - self.min_depth) / (self.max_depth - self.min_depth + 1e-8)
            observation["depth"] = {"agentview": self._tile_time(depth)}

        # Dual-arm TCP -> [T, num_arms, C]
        uvd = np.asarray(sm["tcp_uvd"], dtype=np.float32)
        if uvd.shape[0] != self.num_arms:
            raise ValueError(f"tcp_uvd arms={uvd.shape[0]} != num_arms={self.num_arms}")
        tcp_pos_cam = np.asarray(sm["tcp_pos"], dtype=np.float32)          # camera frame (OpenCV)
        tcp_rot_cam = np.asarray(sm["tcp_rotmat"], dtype=np.float32)       # (arms, 9) row-major
        if self.tcp_shift_along_z_m != 0.0:
            z_axis = tcp_rot_cam.reshape(self.num_arms, 3, 3)[:, :, 2]      # third column = site z
            tcp_pos_cam = tcp_pos_cam + self.tcp_shift_along_z_m * z_axis
            K_full = np.asarray(_parse_literal(attrs["K"]), dtype=np.float64).reshape(3, 3)
            proj = tcp_pos_cam @ K_full.T                                    # (arms, 3)
            uvd = np.stack([proj[:, 0] / proj[:, 2], proj[:, 1] / proj[:, 2], tcp_pos_cam[:, 2]], axis=-1).astype(np.float32)
        uvd_n = self._normalize_tcp_uvd(uvd, height, width)

        tcp_valid = np.asarray(sm["tcp_in_frame"], dtype=np.float32)
        if "tcp_visible" in sm:
            tcp_valid = tcp_valid * np.asarray(sm["tcp_visible"], dtype=np.float32)

        tcp_fields = {
            "tcp_pixel_coords": uvd_n,
            "tcp_dir_x": np.asarray(sm["tcp_dir_x"], dtype=np.float32),
            "tcp_dir_y": np.asarray(sm["tcp_dir_y"], dtype=np.float32),
            "tcp_dir_z": np.asarray(sm["tcp_dir_z"], dtype=np.float32),
            "tcp_pos": tcp_pos_cam,
            "tcp_orn": tcp_rot_cam,
            "tcp_valid": tcp_valid.astype(np.float32),
        }
        for key, val in tcp_fields.items():
            if key == "tcp_valid" or key in self.observation_keys:
                observation[key] = {"agentview": self._tile_time(val)}

        # Proprio from world attrs (same frame as tcp_pos_world / eef).
        eef_pos = np.asarray(_parse_literal(attrs["eef_pos_world"]), dtype=np.float32)
        eef_rot = _as_3x3_stack(attrs["eef_rotmat_world"], self.num_arms)
        eef_quat = _rotmat_to_quat_xyzw(eef_rot)  # (2, 4) xyzw

        grip_open = _parse_literal(attrs.get("gripper_openness_per_arm", [0.0] * self.num_arms))
        grip_open = np.asarray(grip_open, dtype=np.float32).reshape(self.num_arms)

        for arm_i, prefix in enumerate(self.arm_obs_prefixes):
            observation[f"{prefix}_eef_pos"] = self._tile_time(eef_pos[arm_i])
            observation[f"{prefix}_eef_quat_site"] = self._tile_time(eef_quat[arm_i])
            observation[f"{prefix}_gripper_qpos"] = self._tile_time(
                np.asarray([grip_open[arm_i]], dtype=np.float32)
            )

        # robot0_* aliases (left arm) for keys listed in observation_keys.
        observation["robot0_eef_pos"] = observation[f"{self.arm_obs_prefixes[0]}_eef_pos"]
        observation["robot0_eef_quat_site"] = observation[
            f"{self.arm_obs_prefixes[0]}_eef_quat_site"
        ]
        observation["robot0_gripper_qpos"] = observation[
            f"{self.arm_obs_prefixes[0]}_gripper_qpos"
        ]

        # Camera K / extrinsics under agentview.
        K = np.asarray(_parse_literal(attrs["K"]), dtype=np.float32).reshape(3, 3)
        E_c2w = _as_4x4(attrs["E_c2w"]).astype(np.float32)
        observation["camera_intrinsics"] = {"agentview": K}
        observation["camera_extrinsics"] = {"agentview": E_c2w}

        if self.use_state:
            blocks = []
            for prefix in self.arm_obs_prefixes:
                arm_pos = observation[f"{prefix}_eef_pos"]
                arm_rot = _quaternion_xyzw_to_rotation_6d(observation[f"{prefix}_eef_quat_site"])
                arm_grip = observation[f"{prefix}_gripper_qpos"][..., :1]
                blocks.extend([arm_pos, arm_rot, arm_grip])
            observation["state"] = np.concatenate(blocks, axis=-1)

        # Per-camera transforms (Resize / PrepareForNet), same contract as RobomimicHDF5.
        for camera_name in self.camera_names:
            camera_obs = {}
            for key, val in observation.items():
                if isinstance(val, dict) and camera_name in val:
                    camera_obs[key] = val[camera_name]
            camera_obs = self.transforms(camera_obs)
            for key, val in camera_obs.items():
                if isinstance(observation.get(key), dict):
                    observation[key][camera_name] = val

        prompt = self.task_description
        if self.use_task_description:
            task_name = str(attrs.get("task_name", "") or "").strip()
            if task_name:
                prompt = f"{self.prompt_prefix} {task_name}"

        actions = np.zeros(
            (self.action_sequence_length, self.action_dim), dtype=np.float32
        )
        return {
            "observation": observation,
            "action": actions,
            "prompt": prompt,
            "episode_id": name,
            "step_id": 0,
            "is_last_step": np.bool_(True),
            "camera_ring_index": int(attrs["camera_ring_index"]),
        }
