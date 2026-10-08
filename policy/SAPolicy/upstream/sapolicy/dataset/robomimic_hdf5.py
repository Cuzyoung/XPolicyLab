#!/usr/bin/env python3
"""
HDF5 Dataset Loader for Multiple Robotic Manipulation Datasets

Supports:
- CPGen: ThreePieceAssemblyWide, Coffee_D1, etc.
- RoboSuite: NutAssembly, PegInHole, etc.
- RoboCasa: PnPCounterToSink, OpenSingleDoor, etc.

Each dataset type has its own loading logic and camera parameters.
"""

import os
import sys
import numpy as np
import h5py
import json
import zarr
from pathlib import Path
from distutils.util import strtobool
from typing import Dict, List, Optional, Any, Tuple, Sequence
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import Compose

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sapolicy.logger import Log
from sapolicy.dataset.imagecodecs_numcodecs import register_codecs
from sapolicy.embodiment_transforms import build_data_transform, resolve_transform_config

register_codecs()

# CPGen / RoboSuite HDF5 tcp_orn is gripper body frame (robot0_eef_quat).
# State and body-frame actions use robot0_eef_quat_site — fixed R_z(-90°) offset on Panda.
_TCP_BODY_TO_SITE_ROT = R.from_euler('z', -90, degrees=True).as_matrix().astype(np.float32)

# Process-local registry so joint tcp/action datasets sharing the same cache_dir
# reuse one set of np.load / mmap handles instead of opening the cache twice.
# Key: (resolved_cache_dir, cache_mmap) -> shared payload dict.
_NUMPY_CACHE_REGISTRY: Dict[Tuple[str, bool], Dict[str, Any]] = {}

# Hardcoded camera intrinsics for RoboSuite and RoboCasa
ROBOSUITE_CAMERA_INTRINSICS = {
    'robot0_agentview': {
        'fovy': 45.0,
        'image_size': (256, 256),  # (H, W)
    },
    'agentview': {
        'fovy': 45.0,
        'image_size': (256, 256),
    },
    'robot0_eye_in_hand': {
        'fovy': 75.0,
        'image_size': (256, 256),
    }
}

ROBOCASA_CAMERA_INTRINSICS = {
    'robot0_agentview_left': {
        'fovy': 60.0,
        'image_size': (256, 256),
    },
    'robot0_agentview_right': {
        'fovy': 60.0,
        'image_size': (256, 256),
    },
    'robot0_eye_in_hand': {
        'fovy': 45.0,
        'image_size': (256, 256),
    }
}


def fovy_to_focal_length(fovy_deg: float, image_height: int) -> float:
    """Convert field of view to focal length in pixels"""
    fovy_rad = np.deg2rad(fovy_deg)
    return image_height / (2.0 * np.tan(fovy_rad / 2.0))


def get_camera_intrinsics(camera_name: str, dataset_type: str) -> np.ndarray:
    """
    Get camera intrinsics matrix for a given camera and dataset type

    Args:
        camera_name: Name of the camera (e.g., 'robot0_agentview')
        dataset_type: 'cpgen', 'robosuite', or 'robocasa'

    Returns:
        3x3 intrinsics matrix K
    """
    if dataset_type == 'robosuite':
        if camera_name not in ROBOSUITE_CAMERA_INTRINSICS:
            raise ValueError(f"Unknown RoboSuite camera: {camera_name}")
        params = ROBOSUITE_CAMERA_INTRINSICS[camera_name]
    elif dataset_type == 'robocasa':
        if camera_name not in ROBOCASA_CAMERA_INTRINSICS:
            raise ValueError(f"Unknown RoboCasa camera: {camera_name}")
        params = ROBOCASA_CAMERA_INTRINSICS[camera_name]
    else:
        # For CPGen, intrinsics are loaded from HDF5 file
        return None

    height, width = params['image_size']
    fy = fovy_to_focal_length(params['fovy'], height)
    fx = fy  # Square pixels

    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def _assemble_state(observation, arm_prefixes, fallback_prefix='robot0'):
    """Concatenate per-arm [eef_pos(3), eef_rot_6d(6), gripper_width(1)] blocks.

    Assumes GripperWidthTransform/EefRotation6DTransform already ran on
    `observation`, populating `<prefix>_eef_rot_6d` and adapting
    `<prefix>_gripper_qpos` in place. A per-arm key missing from `observation`
    (bimanual demos don't always carry a distinct pose per arm) falls back to
    `fallback_prefix`'s value; a missing gripper key falls back to zeros.
    """
    fallback_pos = observation.get(f"{fallback_prefix}_eef_pos")
    fallback_rot = observation.get(f"{fallback_prefix}_eef_rot_6d")
    fallback_grip = observation.get(f"{fallback_prefix}_gripper_qpos")
    blocks = []
    for prefix in arm_prefixes:
        pos = observation.get(f"{prefix}_eef_pos", fallback_pos)
        rot6d = observation.get(f"{prefix}_eef_rot_6d", fallback_rot)
        grip = observation.get(f"{prefix}_gripper_qpos", fallback_grip)
        grip = np.zeros(pos.shape[:-1] + (1,), dtype=np.float32) if grip is None else grip[..., :1]
        if not (pos.shape[0] == rot6d.shape[0] == grip.shape[0]):
            raise ValueError(
                f"State shape mismatch for prefix={prefix!r}: "
                f"eef_pos={pos.shape}, eef_rot={rot6d.shape}, gripper={grip.shape}"
            )
        blocks += [pos, rot6d, grip]
    return np.concatenate(blocks, axis=-1)


def _pad_action_window(actions: np.ndarray, target_len: int, fallback: np.ndarray) -> np.ndarray:
    """Right-pad by repeating the last action; use `fallback` if `actions` is empty."""
    if len(actions) >= target_len:
        return actions
    pad_len = target_len - len(actions)
    last_action = actions[-1:] if len(actions) > 0 else fallback
    return np.concatenate([actions, np.repeat(last_action, pad_len, axis=0)], axis=0)


def _as_bool(value) -> bool:
    return value if isinstance(value, bool) else strtobool(value)


class RobomimicHDF5Dataset(Dataset):
    """
    Dataset loader for robotic manipulation datasets in HDF5 format.

    Supports multiple dataset types:
    - CPGen: ThreePieceAssemblyWide, Coffee_D1, etc. (has TCP data in HDF5)
    - RoboSuite: NutAssembly, PegInHole, etc. (needs TCP preprocessing)
    - RoboCasa: PnPCounterToSink, OpenSingleDoor, etc. (needs TCP preprocessing)

    HDF5 Structure:
    - data/
      - demo_0/
        - actions: (T, action_dim)
        - states: (T, state_dim)
        - rewards: (T,)
        - dones: (T,)
        - obs/
          - {camera}_image: (T, H, W, 3)  # RGB
          - {camera}_depth: (T, H, W, 1) or (T, H, W)  # Depth (optional)
          - robot0_eef_pos: (T, 3)
          - robot0_eef_quat_site: (T, 4)
          - robot0_joint_pos: (T, 7)
          - ... (other observations)
        - next_obs/ (same structure as obs)
    """

    def _apply_tcp_occlusion_mask(self, tcp_in_frame, source, camera_prefix, indexer):
        """Mask tcp_in_frame using labels when present, otherwise TCP-vs-image depth."""
        valid = np.asarray(tcp_in_frame, dtype=np.float32)
        mask_keys = (
            (True, (
                f"{camera_prefix}_tcp_occluded",
                f"{camera_prefix}_occluded",
                f"{camera_prefix}_tcp_occlusion",
                f"{camera_prefix}_occlusion",
            )),
            (False, (
                f"{camera_prefix}_tcp_visibility",
                f"{camera_prefix}_visibility",
                f"{camera_prefix}_tcp_visible",
                f"{camera_prefix}_visible",
            )),
        )

        for is_occluded, keys in mask_keys:
            for mask_key in keys:
                if mask_key not in source:
                    continue
                mask = np.asarray(source[mask_key][indexer], dtype=np.float32)
                mask = mask <= 0.5 if is_occluded else mask > 0.5
                if mask.shape != valid.shape:
                    if mask.size == valid.size:
                        mask = mask.reshape(valid.shape)
                    elif valid.ndim == mask.ndim + 1 and valid.shape[-1] == 1 and mask.shape == valid.shape[:-1]:
                        mask = np.expand_dims(mask, axis=-1)
                    elif mask.ndim == valid.ndim + 1 and mask.shape[-1] == 1 and mask.shape[:-1] == valid.shape:
                        mask = mask[..., 0]
                    else:
                        continue
                return valid * mask.astype(valid.dtype)

        depth_key = f"{camera_prefix}_depth"
        tcp_key = f"{camera_prefix}_tcp_pixel_coords"
        if depth_key not in source or tcp_key not in source:
            return valid

        depth = np.asarray(source[depth_key][indexer], dtype=np.float32)
        tcp = np.asarray(source[tcp_key][indexer], dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None, ...]
        elif depth.ndim == 4:
            depth = depth[..., 0]
        if tcp.ndim == 1:
            tcp = tcp[None, :]
        if depth.ndim != 3 or tcp.shape[-1] < 3:
            return valid

        depth_flat = depth.reshape((-1,) + depth.shape[-2:])
        tcp_flat = tcp.reshape(-1, tcp.shape[-1])
        if depth_flat.shape[0] != tcp_flat.shape[0] or valid.size != tcp_flat.shape[0]:
            return valid

        h, w = depth_flat.shape[-2:]
        u = np.rint(tcp_flat[:, 0]).astype(np.int64)
        v = np.rint(tcp_flat[:, 1]).astype(np.int64)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        sampled_depth = np.zeros(tcp_flat.shape[0], dtype=np.float32)
        idxs = np.nonzero(in_bounds)[0]
        sampled_depth[idxs] = depth_flat[idxs, v[idxs], u[idxs]]

        tcp_depth = tcp_flat[:, 2]
        occluded = (
            (sampled_depth > 0.0) # depth check
            & (tcp_depth > sampled_depth + self.tcp_occlusion_depth_tolerance) # occusion check. TCP 比 depth map 表面更远，所以被前景挡住，也就是 occluded=True。
        )
        visible = (in_bounds & ~occluded).reshape(valid.shape)
        return valid * visible.astype(valid.dtype)

    def __init__(
        self,
        hdf5_path: str = None,
        zarr_path: str = None,
        dataset_name: str = "ThreePieceAssemblyWide",
        dataset_type: str = "cpgen",  # 'cpgen', 'robosuite', or 'robocasa'
        tcp_preprocessed_path: Optional[str] = None,  # Path to preprocessed TCP data
        split: str = 'train',
        use_train_val_split: bool = False,
        train_val_split_ratio: float = 0.9,
        split_seed: int = 42,
        observation_keys: Optional[List[str]] = None,
        obs_hist_length: int = 1,
        action_sequence_length: int = 4,
        normalize_actions: bool = True,
        use_task_description: bool = True,
        task_description: Optional[str] = None,
        transforms: Optional[List] = None,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
        use_agentview: bool = True,
        use_eye_in_hand: bool = False,
        use_depth: bool = True,
        use_state: bool = True,
        max_episodes: Optional[int] = None,
        camera_names: Optional[List[str]] = None,  # Override camera names
        use_relative_actions: bool = False,
        action_orn_mode: str = '6d',
        relative_action_stats_path: Optional[str] = None,
        load_to_memory: bool = False,
        cache_dir: Optional[str] = None,  # preprocessed numpy cache directory
        cache_mmap: bool = False,  # share read-only cache pages across DDP ranks
        norm_type: str = 'percentile_0.02_0.98', # 'mean_std' or 'percentile_0.02_0.98'
        cpgen_action_pos_scale: float = 0.05,
        cpgen_action_rot_scale: float = 0.5,
        cpgen_absolute_actions: bool = False,
        body_frame_actions: bool = True,
        random_camera_pairs: Optional[List[List[str]]] = None,  # R54: random 2-of-4 camera selection per sample
        camera_pair_choices: Optional[Dict[str, Any]] = None,  # coupled camera pair selection per sample
        load_future_tcp: bool = False,
        future_tcp_horizons: Optional[List[int]] = None,
        load_future_images: bool = False,
        future_image_horizons: Optional[List[int]] = None,
        tcp_occlusion_depth_tolerance: float = 0.20, # 20cm to prevent occlusion
        num_arms: int = 1,  # >1: bimanual data (RoboTwin); actions are per-arm blocks
        arm_obs_prefixes: Optional[List[str]] = None,  # per-arm obs key prefixes, e.g. ["left", "right"]
        tcp_orn_already_aligned: bool = False,  # dataset tcp_orn is already in the state/action rotation frame
        embodiment: Optional[str] = None,  # grouping key for balanced sampling across embodiments
        **kwargs
    ):
        """
        Initialize HDF5 dataset, now supports both HDF5 and Zarr files.

        Args:
            hdf5_path: Path to HDF5 file
            zarr_path: Path to Zarr file
            dataset_name: Name of the dataset (for logging)
            dataset_type: Type of dataset ('cpgen', 'robosuite', 'robocasa')
            tcp_preprocessed_path: Path to preprocessed TCP data HDF5 file
            split: 'train' or 'val'
            use_train_val_split: Whether to split train/val
            train_val_split_ratio: Ratio for train/val split
            split_seed: Random seed for splitting
            observation_keys: List of observation keys to use
            action_sequence_length: Number of future actions to return
            normalize_actions: Whether to normalize actions
            use_task_description: Whether to include task description
            task_description: Custom task description (if None, use dataset name)
            transforms: List of transforms to apply to images
            min_depth: Minimum depth value for normalization
            max_depth: Maximum depth value for normalization
            use_agentview: Use agentview camera
            use_eye_in_hand: Use eye-in-hand camera
            use_depth: Include depth images
            use_state: Include proprioceptive observations
            max_episodes: Maximum number of episodes to load
            camera_name: Override camera name (default: auto-detect from dataset_type)
        """
        super().__init__()

        assert hdf5_path is not None or zarr_path is not None, "Either hdf5_path or zarr_path must be provided"

        self.hdf5_path = Path(hdf5_path) if hdf5_path else None
        self.zarr_path = Path(zarr_path) if zarr_path else None
        if self.hdf5_path and not self.hdf5_path.exists():
            print(f"HDF5 file not found: {self.hdf5_path}, using Zarr file instead")
        if self.zarr_path and not self.zarr_path.exists():
            print(f"Zarr file not found: {self.zarr_path}, using HDF5 file instead")
            
        self.hdf5_file = None
        self.zarr_file = None
        self.tcp_file = None
        self.cache_mmap = bool(cache_mmap)

        if self.zarr_path and self.zarr_file is None:
            # store = zarr.DirectoryStore(self.zarr_path)
            # cache = zarr.LRUStoreCache(store=store, max_size=2**28)
            # self.zarr_file = zarr.open(cache, "r")
            self.zarr_file = zarr.open(self.zarr_path, "r")
            if load_to_memory:
                self.load_to_memory()

        self.dataset_name = dataset_name
        self.embodiment = embodiment
        self.dataset_type = dataset_type.lower()
        self.tcp_preprocessed_path = Path(tcp_preprocessed_path) if tcp_preprocessed_path else None
        self.split = split
        self.use_train_val_split = use_train_val_split
        self.train_val_split_ratio = train_val_split_ratio
        self.split_seed = split_seed
        self.observation_keys = observation_keys
        self.action_sequence_length = int(action_sequence_length)
        self.obs_hist_length = int(obs_hist_length)
        self.normalize_actions = normalize_actions
        self.use_task_description = use_task_description
        self.task_description = task_description or f"Perform {dataset_name} task"
        self.transforms = [] if transforms is None else list(transforms)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.use_agentview = _as_bool(use_agentview)
        self.use_eye_in_hand = _as_bool(use_eye_in_hand)
        self.use_depth = use_depth
        self.use_state = _as_bool(use_state)
        self.max_episodes = max_episodes
        self.use_relative_actions = _as_bool(use_relative_actions)
        self.action_orn_mode = action_orn_mode
        if self.use_relative_actions:
            assert self.action_orn_mode in ['6d'], "only 6d representation is supported for relative actions"
        self.relative_action_stats_path = None
        # CPGen datasets often store controller *input* deltas in [-1, 1].
        # Convert them to physical deltas with controller output scales.
        self.cpgen_action_pos_scale = float(cpgen_action_pos_scale)
        self.cpgen_action_rot_scale = float(cpgen_action_rot_scale)
        self.cpgen_absolute_actions = cpgen_absolute_actions
        self.body_frame_actions = body_frame_actions

        transform_config = resolve_transform_config(self.dataset_type, self.embodiment)
        action_se3_kwargs = dict(
            action_orn_mode=self.action_orn_mode,
            use_relative_actions=self.use_relative_actions,
            body_frame_actions=self.body_frame_actions,
            cpgen_action_pos_scale=self.cpgen_action_pos_scale,
            cpgen_action_rot_scale=self.cpgen_action_rot_scale,
            cpgen_absolute_actions=self.cpgen_absolute_actions,
        )
        self.embodiment_transform = build_data_transform(transform_config, action_se3=action_se3_kwargs)

        self.load_future_tcp = _as_bool(load_future_tcp)
        self.load_future_images = _as_bool(load_future_images)
        if future_image_horizons is None:
            self.future_image_horizons = (1, 2, 4) if self.load_future_images else ()
        else:
            self.future_image_horizons = tuple(int(h) for h in future_image_horizons)
        if future_tcp_horizons is None:
            self.future_tcp_horizons = (1, 2, 4) if self.load_future_tcp else ()
        else:
            self.future_tcp_horizons = tuple(int(h) for h in future_tcp_horizons)

        def _validate_horizons(name, enabled, horizons):
            if not enabled:
                return
            if not horizons or any(h < 1 for h in horizons):
                raise ValueError(f"{name} must be positive when enabled, got {horizons}")
            if max(horizons) > self.action_sequence_length:
                raise ValueError(
                    f"max({name})={max(horizons)} exceeds "
                    f"action_sequence_length={self.action_sequence_length}"
                )

        _validate_horizons("future_image_horizons", self.load_future_images, self.future_image_horizons)
        _validate_horizons("future_tcp_horizons", self.load_future_tcp, self.future_tcp_horizons)
        expected_horizon = (int(self.action_sequence_length),)
        if self.load_future_images and self.future_image_horizons != expected_horizon:
            raise ValueError(
                f"future_image_horizons must be [{self.action_sequence_length}] "
                f"(action chunk size T), got {self.future_image_horizons}"
            )
        if self.load_future_tcp and self.future_tcp_horizons != expected_horizon:
            raise ValueError(
                f"future_tcp_horizons must be [{self.action_sequence_length}] "
                f"(action chunk size T), got {self.future_tcp_horizons}"
            )
        self.tcp_occlusion_depth_tolerance = float(tcp_occlusion_depth_tolerance)

        # Bimanual support. num_arms=1 keeps the original single-arm code path
        self.num_arms = int(num_arms)
        if arm_obs_prefixes is None:
            arm_obs_prefixes = ["robot0"] if self.num_arms == 1 else ["left", "right"]
        self.arm_obs_prefixes = list(arm_obs_prefixes)
        # _align_tcp_to_eef_site_frame post-multiplies every tcp_orn by a fixed
        # Rz(-90 deg) encoding robosuite's gripper-body -> eef-site offset. RoboTwin
        # conversions already write tcp_orn in the same frame as the state/action
        # rotations, so that extra rotation would leave TCP rotation supervision off
        # from the action frame by a constant Rz(-90).
        self.tcp_orn_already_aligned = _as_bool(tcp_orn_already_aligned)

        self.data = None
        self.actions = None
        self.states = None
        self._load_actions()
        self._load_states()

        # Determine camera names.
        # Supports:
        #   - explicit list, e.g. ['agentview', 'robot0_eye_in_hand', ...]
        #   - 'auto' / ['auto']: detect all available camera image streams from dataset
        #   - fallback from use_agentview / use_eye_in_hand flags
        if camera_names:
            if isinstance(camera_names, str):
                req_camera_names = [camera_names]
            else:
                req_camera_names = list(camera_names)

            if len(req_camera_names) == 1 and str(req_camera_names[0]).lower() == 'auto':
                self.camera_names = self._detect_available_camera_names()
            else:
                self.camera_names = req_camera_names
        else:
            selected = []
            if self.use_agentview:
                if self.dataset_type == 'cpgen':
                    selected.append('agentview')
                elif self.dataset_type == 'robocasa':
                    selected.append('robot0_agentview_left')
                else:  # robosuite
                    selected.append('robot0_agentview')
            if self.use_eye_in_hand:
                selected.append('robot0_eye_in_hand')

            if not selected:
                if self.dataset_type == 'cpgen':
                    selected = ['agentview']
                elif self.dataset_type == 'robocasa':
                    selected = ['robot0_agentview_left']
                else:
                    selected = ['robot0_agentview']

            # De-duplicate while preserving order.
            seen = set()
            self.camera_names = [c for c in selected if not (c in seen or seen.add(c))]

        # R54: Random camera pair selection (e.g., pick 1 of 2 third-person + 1 of 2 wrist cameras)
        self.random_camera_pairs = random_camera_pairs
        self._camera_pair_groups = None
        # Coupled camera pair selection (per-sample atomic choice of a (cam_a, cam_b) pair)
        self.camera_pair_choices = None
        self._pair_canonical_names: Optional[List[str]] = None
        self._pair_choices: Optional[List[List[str]]] = None
        self._n_pairs = 1  # used by __len__ to enumerate (sample, pair_idx) pairs

        if random_camera_pairs is not None and camera_pair_choices is not None:
            raise ValueError(
                "random_camera_pairs and camera_pair_choices are mutually exclusive"
            )
        if self.random_camera_pairs:
            self._camera_pair_groups = [list(group) for group in self.random_camera_pairs]
            # Canonical names are the first camera in each group
            self.camera_names = [group[0] for group in self._camera_pair_groups]
            # All cameras that need to be present in dataset
            self._all_camera_names_pairs = [cam for group in self._camera_pair_groups for cam in group]
            Log.info(f"[R54] Random camera pairs mode:")
            for i, group in enumerate(self._camera_pair_groups):
                Log.info(f"  Group {i}: {group} → canonical '{group[0]}'")
            Log.info(f"  Model sees cameras as: {self.camera_names}")
        elif camera_pair_choices is not None:
            self.camera_pair_choices = camera_pair_choices
            canonical = list(camera_pair_choices["canonical_names"])
            choices = [list(pair) for pair in camera_pair_choices["choices"]]
            arity = len(canonical)
            if arity < 1:
                raise ValueError("canonical_names must be non-empty")
            for i, pair in enumerate(choices):
                if len(pair) != arity:
                    raise ValueError(
                        f"camera_pair_choices choice[{i}]={pair} has {len(pair)} cameras, expected {arity}"
                    )
            self._pair_canonical_names = canonical
            self._pair_choices = choices
            self.camera_names = canonical  # model-facing canonical names
            self._all_camera_names_pairs = sorted({c for pair in choices for c in pair})
            # Strategy A: enumerate (sample, pair_idx) as distinct indices so the
            # dataloader visits every (demo, step, pair) combo exactly once per
            # epoch instead of randomly sampling 1/n_pairs per sample.
            # Set enumerate_all: false to disable (random.choice per sample, __len__ unchanged).
            if camera_pair_choices.get("enumerate_all", True):
                self._n_pairs = len(choices)
            else:
                self._n_pairs = 1  # random pair per sample, no __len__ expansion
            Log.info(f"[coupled] camera_pair_choices mode: canonical={canonical}")
            for i, pair in enumerate(choices):
                Log.info(f"  choice[{i}] -> {dict(zip(canonical, pair))}")
            Log.info(f"  backing cameras: {self._all_camera_names_pairs}")
            Log.info(f"  enumerate pairs: __len__ x {self._n_pairs} ({'deterministic' if self._n_pairs > 1 else 'random'})")

        Log.info(f"Using camera(s): {self.camera_names}")
        try:
            available_cameras = set(self._detect_available_camera_names())
            # Validate ALL cameras (including those used in random/coupled selection)
            if self.random_camera_pairs or self._pair_choices:
                cameras_to_check = self._all_camera_names_pairs
            else:
                cameras_to_check = self.camera_names
            missing_cameras = [c for c in cameras_to_check if c not in available_cameras]
            if missing_cameras:
                raise ValueError(
                    f"Configured camera_names {missing_cameras} not found in dataset. "
                    f"Available cameras: {sorted(available_cameras)}"
                )
        except Exception as e:
            raise ValueError(f"Failed camera validation: {e}") from e
        # Cache for parsed camera_info JSON (keyed by demo_key)
        self._cpgen_camera_key_map = {
            'agentview': 'agentview',
            'robot0_agentview': 'agentview',
            'robot0_eye_in_hand': 'robot0_eye_in_hand',
        }
        # Register identity entries for any extra cameras referenced by the
        # random/coupled pair selection, so camera_info lookups resolve without
        # relying on the .get(cn, cn) fallback.
        if self._pair_choices or self.random_camera_pairs:
            for cam in self._all_camera_names_pairs:
                self._cpgen_camera_key_map.setdefault(cam, cam)
        self._cpgen_camera_info_cache = {}

        # Load TCP preprocessed data if provided
        self.tcp_data = {}
        self.tcp_file = None
        self._loaded_tcp_data = False

        # Get hardcoded camera intrinsics for RoboSuite/RoboCasa
        self.hardcoded_intrinsics = {}
        if self.dataset_type in ['robosuite', 'robocasa']:
            if self._pair_choices is not None:
                raise NotImplementedError(
                    "camera_pair_choices is only supported for CPGen datasets; "
                    "robosuite/robocasa hardcoded intrinsics use static camera_names."
                )
            for camera_name in self.camera_names:
                self.hardcoded_intrinsics[camera_name] = get_camera_intrinsics(camera_name, self.dataset_type)
                Log.info(f"Using hardcoded camera intrinsics for {self.dataset_type}/{camera_name}")
        else:
            self.hardcoded_intrinsics = {}

        self.build_transforms()

        # Load episode list and create index
        self._load_episode_index()

        # Preload all observation data into memory to avoid per-sample h5py
        # gzip decompression overhead. Each experiment needs ~58GB RAM.
        self._obs_cache = None
        self._numpy_cache = None  # preprocessed numpy memmap cache
        self._cache_demo_boundaries: dict | None = None
        self._cache_demo_attrs = None
        self._cache_images_resized = False

        if cache_dir and Path(cache_dir).exists() and (Path(cache_dir) / "meta.json").exists():
            self._load_numpy_cache(cache_dir)
        elif load_to_memory and self.hdf5_path:
            self._preload_obs_to_memory()

        Log.info(f"Initialized {self.dataset_name} ({self.dataset_type}) dataset with {len(self.episode_indices)} steps")

    def _resolve_sample_cameras(self, pair_idx: Optional[int] = None):
        """Return [(actual_camera, output_canonical), ...] for the current sample.

        Single source of truth for per-sample camera selection. Used by both
        the non-cache __getitem__ path and the numpy-cache fast path so their
        (actual, canonical) pairings stay in sync.

        When ``pair_idx`` is provided (strategy A: dataloader enumerates every
        (sample, pair) combo), the pair is selected deterministically by index
        so each epoch visits every (sample, pair) exactly once. When not given
        the legacy random.choice path is used (only for random_camera_pairs).
        Call exactly ONCE per sample; the caller must use the returned list
        consistently for image / depth / TCP / camera_info loading in that
        sample.
        """
        if self._pair_choices:
            if pair_idx is None:
                import random
                pair = random.choice(self._pair_choices)
            else:
                pair = self._pair_choices[pair_idx % len(self._pair_choices)]
            return list(zip(pair, self._pair_canonical_names))
        if self._camera_pair_groups:
            import random
            out = []
            for group in self._camera_pair_groups:
                out.append((random.choice(group), group[0]))
            return out
        return [(cn, cn) for cn in self.camera_names]

    @staticmethod
    def _cameras_requiring_cpgen_k(
        cameras_to_load: Sequence[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """Drop RGB-only wrists: RoboTwin camera_info has no eye_in_hand K/ext."""
        return [
            (actual_cn, output_cn)
            for actual_cn, output_cn in cameras_to_load
            if "eye_in_hand" not in actual_cn and "eye_in_hand" not in output_cn
        ]

    def _load_cpgen_k_and_ext(
        self,
        cameras_to_load: Sequence[Tuple[str, str]],
        camera_info: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load K/ext for cameras that are supposed to have them.

        Missing entries raise. Do not reuse another camera's K — wrist and ring
        pinholes are not interchangeable, and a silent fallback hides dataset bugs.
        """
        intrinsics: Dict[str, np.ndarray] = {}
        extrinsics: Dict[str, np.ndarray] = {}
        available = sorted(camera_info.keys())
        for actual_cn, output_cn in cameras_to_load:
            ck = self._cpgen_camera_key_map.get(actual_cn, actual_cn)
            entry = camera_info.get(ck)
            if not isinstance(entry, dict) or entry.get("intrinsics") is None:
                raise ValueError(
                    f"Missing camera intrinsics for camera {actual_cn} "
                    f"(lookup key={ck!r}, camera_info keys={available}). "
                    "Refusing to reuse another camera's K."
                )
            intrinsics[output_cn] = np.array(entry["intrinsics"], dtype=np.float32)
            if entry.get("extrinsics") is not None:
                extrinsics[output_cn] = np.array(entry["extrinsics"], dtype=np.float32)
        return intrinsics, extrinsics

    def _resolve_primary_alt_cameras(self, pair_idx: Optional[int] = None):
        """Return primary camera mappings; alternate-view output is retired."""
        return self._resolve_sample_cameras(pair_idx=pair_idx), []

    def _detect_available_camera_names(self) -> List[str]:
        """Detect camera names from observation image keys of the first demo.

        Camera names are inferred by stripping the `_image` suffix from keys like:
          - agentview_image -> agentview
          - third_view_0_image -> third_view_0
        """
        opened_hdf5 = None
        if self.data is not None:
            data_group = self.data['data']
        else:
            if self.zarr_path and self.zarr_path.exists():
                zarr_file = zarr.open(self.zarr_path, 'r')
                data_group = zarr_file['data']
            elif self.hdf5_path and self.hdf5_path.exists():
                opened_hdf5 = h5py.File(self.hdf5_path, 'r', swmr=True)
                data_group = opened_hdf5['data']
            else:
                raise FileNotFoundError(
                    f"Cannot auto-detect camera names: no readable dataset at "
                    f"hdf5_path={self.hdf5_path}, zarr_path={self.zarr_path}"
                )

        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        if not demo_keys:
            if opened_hdf5 is not None:
                opened_hdf5.close()
            raise ValueError("Cannot auto-detect camera names: dataset has no 'demo_*' entries.")

        obs = data_group[demo_keys[0]]['obs']
        camera_names = sorted([k[:-6] for k in obs.keys() if k.endswith('_image')])

        if opened_hdf5 is not None:
            opened_hdf5.close()

        if not camera_names:
            raise ValueError(
                "Cannot auto-detect camera names: no observation keys matching '*_image' were found."
            )
        return camera_names

    def _preload_obs_to_memory(self):
        """Preload all observation data (images, depth, TCP, proprio) into RAM.

        HDF5 with gzip compression is very slow for random access due to
        per-chunk decompression.  Loading everything once into numpy arrays
        makes __getitem__ use simple numpy indexing instead of h5py reads.
        """
        Log.info(f"Preloading all observations into memory from {self.hdf5_path} ...")
        cache = {}
        with h5py.File(self.hdf5_path, 'r', swmr=True) as f:
            data_group = f['data']
            demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
            for di, demo_key in enumerate(demo_keys):
                obs = data_group[demo_key]['obs']
                cache[demo_key] = {}
                for key in obs.keys():
                    cache[demo_key][key] = obs[key][:]
                # Also cache demo attrs (camera_info etc.)
                cache[demo_key]['_attrs'] = dict(data_group[demo_key].attrs)
                if (di + 1) % 20 == 0 or di == len(demo_keys) - 1:
                    Log.info(f"  Preloaded {di+1}/{len(demo_keys)} demos")
        self._obs_cache = cache
        Log.info(f"Preloading complete. {len(cache)} demos in memory.")

    def _compute_rel_geometry(self, eef_pos, eef_quat, obj_state):
        """R119: Compute view-invariant relative geometry GT from proprioceptive state.

        Args:
            eef_pos: (To, 3) gripper position
            eef_quat: (To, 4) gripper quaternion (xyzw convention)
            obj_state: (To, 14) = [nut_pos(3), nut_quat(4), handle_pos(3), handle_quat(4)]
        Returns:
            dict with rel_pos (To,3), rel_dist (To,1), rel_ori (To,3), contact (To,1)
        """
        from scipy.spatial.transform import Rotation as R
        nut_pos = obj_state[..., :3]         # (To, 3)
        nut_quat_wxyz = obj_state[..., 3:7]  # (To, 4) wxyz (MuJoCo body_xquat convention)
        # Convert nut quaternion from wxyz (MuJoCo) to xyzw (scipy)
        nut_quat = np.concatenate([nut_quat_wxyz[..., 1:], nut_quat_wxyz[..., :1]], axis=-1)

        # Relative position in gripper frame (view-invariant)
        rel_pos_world = nut_pos - eef_pos  # (To, 3)
        r_eef = R.from_quat(eef_quat)     # eef_quat_site is already xyzw
        rel_pos_gripper = r_eef.inv().apply(rel_pos_world)  # (To, 3)

        # Euclidean distance
        rel_dist = np.linalg.norm(rel_pos_world, axis=-1, keepdims=True)  # (To, 1)

        # Relative orientation as axis-angle
        r_nut = R.from_quat(nut_quat)     # now correctly xyzw
        rel_rot = r_eef.inv() * r_nut
        rel_ori = rel_rot.as_rotvec().astype(np.float32)  # (To, 3)

        # Contact: binary (distance < 3cm)
        contact = (rel_dist < 0.03).astype(np.float32)  # (To, 1)

        return {
            'rel_pos': rel_pos_gripper.astype(np.float32),
            'rel_dist': rel_dist.astype(np.float32),
            'rel_ori': rel_ori,
            'contact': contact,
        }

    def _future_horizon_steps(self, step_idx: int, traj_len: int, horizons: Sequence[int]):
        """Return (clamped_steps, valid_mask) for skip horizons."""
        steps = []
        valid = []
        for k in horizons:
            raw = int(step_idx) + int(k)
            ok = raw < int(traj_len)
            steps.append(min(raw, int(traj_len) - 1))
            valid.append(ok)
        return steps, np.asarray(valid, dtype=np.float32)

    def _future_image_steps(self, step_idx: int, traj_len: int):
        """Return (clamped_steps, valid_mask) for configured future_image_horizons."""
        return self._future_horizon_steps(step_idx, traj_len, self.future_image_horizons)

    def _store_future_tcp(
        self,
        observation: Dict[str, Any],
        output_cam: str,
        camera_prefix: str,
        source,
        fut_idxs,
        fut_valid: np.ndarray,
    ) -> None:
        """Write multi-horizon future TCP tensors shaped ``[K, ...]`` into observation."""
        observation.setdefault("future_tcp_temporal_valid", {})[output_cam] = fut_valid
        for out_key, tcp_key in {
            "future_tcp_pixel_coords": f"{camera_prefix}_tcp_pixel_coords",
            "future_tcp_pos": f"{camera_prefix}_tcp_pos",
            "future_tcp_orn": f"{camera_prefix}_tcp_orn",
        }.items():
            observation.setdefault(out_key, {})[output_cam] = np.array(
                source[tcp_key][fut_idxs],
                dtype=np.float32,
            )
        valid_key = f"{camera_prefix}_tcp_in_frame"
        if valid_key in source:
            future_valid = np.array(source[valid_key][fut_idxs], dtype=np.float32)
            future_valid = self._apply_tcp_occlusion_mask(
                future_valid,
                source,
                camera_prefix,
                fut_idxs,
            )
            observation.setdefault("future_tcp_valid", {})[output_cam] = np.array(
                future_valid,
                dtype=np.float32,
            )

    @staticmethod
    def _transform_tcp_orn_body_to_site(tcp_orn: np.ndarray) -> np.ndarray:
        """Rotate camera-frame tcp_orn from gripper body to eef site frame."""
        orn = np.asarray(tcp_orn, dtype=np.float32)
        if orn.shape[-1] == 9:
            R_body = orn.reshape(*orn.shape[:-1], 3, 3)
            R_site = R_body @ _TCP_BODY_TO_SITE_ROT
            return R_site.reshape(*orn.shape[:-1], 9)
        if orn.shape[-2:] == (3, 3):
            return (orn @ _TCP_BODY_TO_SITE_ROT).astype(np.float32)
        raise ValueError(f"Unexpected tcp_orn shape: {orn.shape}")

    @staticmethod
    def _recompute_tcp_dirs_from_orn(
        tcp_pos: np.ndarray,
        tcp_orn: np.ndarray,
        intrinsics: np.ndarray,
        axis_length: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalized 2D axis directions from camera-frame pos + orn (matches preprocess_tcp_data_robomimic)."""
        pos = np.asarray(tcp_pos, dtype=np.float64)
        orn = np.asarray(tcp_orn, dtype=np.float64)
        single_step = pos.ndim == 1
        if single_step:
            pos = pos.reshape(1, 3)
        if orn.ndim == 1:
            orn = orn.reshape(1, 9)
        if orn.shape[-1] == 9:
            orn = orn.reshape(*orn.shape[:-1], 3, 3)

        K = np.asarray(intrinsics, dtype=np.float64)
        # Index the coordinate axis with an ellipsis: bimanual tensors carry an
        # extra [.., num_tcp, ..] axis, and `[:, :2]` would slice that instead.
        center_uvw = pos @ K.T
        center_uv = center_uvw[..., :2] / np.maximum(center_uvw[..., 2:3], 1e-8)

        dirs = []
        for i in range(3):
            endpoint = pos + orn[..., :3, i] * axis_length
            ep_uvw = endpoint @ K.T
            ep_uv = ep_uvw[..., :2] / np.maximum(ep_uvw[..., 2:3], 1e-8)
            delta = ep_uv - center_uv
            norm = np.linalg.norm(delta, axis=-1, keepdims=True)
            delta = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-8)
            dirs.append(delta.astype(np.float32))

        if single_step:
            return dirs[0][0], dirs[1][0], dirs[2][0]
        return dirs[0], dirs[1], dirs[2]

    @staticmethod
    def _camera_image_key(camera: str) -> str:
        return f"{camera}_image"

    @staticmethod
    def _camera_depth_key(camera: str) -> str:
        return f"{camera}_depth"

    def _align_tcp_to_eef_site_frame(self, observation: Dict[str, Any]) -> None:
        """Align TCP supervision with state/action eef site frame."""
        if getattr(self, 'tcp_orn_already_aligned', False):
            return
        if 'tcp_orn' not in observation and 'future_tcp_orn' not in observation:
            return

        intrinsics = observation.get('camera_intrinsics', {})
        tcp_specs = [
            ('tcp_orn', 'tcp_pos', ('tcp_dir_x', 'tcp_dir_y', 'tcp_dir_z')),
            ('future_tcp_orn', 'future_tcp_pos', None),
        ]
        for orn_key, pos_key, dir_keys in tcp_specs:
            orn_by_cam = observation.get(orn_key)
            if not orn_by_cam:
                continue
            pos_by_cam = observation.get(pos_key, {})
            for cam, orn_val in orn_by_cam.items():
                orn_by_cam[cam] = self._transform_tcp_orn_body_to_site(orn_val)
                if dir_keys is None:
                    continue
                if not all(k in observation for k in dir_keys):
                    continue
                tcp_pos = pos_by_cam.get(cam)
                K = intrinsics.get(cam)
                if tcp_pos is None or K is None:
                    continue
                dir_x, dir_y, dir_z = self._recompute_tcp_dirs_from_orn(
                    tcp_pos, orn_by_cam[cam], K
                )
                observation['tcp_dir_x'][cam] = dir_x
                observation['tcp_dir_y'][cam] = dir_y
                observation['tcp_dir_z'][cam] = dir_z

    def _normalize_tcp_pixel_coords_dict(self, observation: Dict[str, Any], key: str):
        coords_by_camera = observation.get(key)
        if not coords_by_camera:
            return

        images_by_camera = observation.get('image', {})
        for camera_name, coords in coords_by_camera.items():
            # img shape is (To, H, W, C) before transforms.
            H, W = images_by_camera[camera_name].shape[1:3]

            coords[..., 0] /= max(W - 1, 1)
            coords[..., 1] /= max(H - 1, 1)
            if self.dataset_type == 'cpgen':
                d = coords[..., 2]
                d = np.clip(d, self.min_depth, self.max_depth)
                coords[..., 2] = (d - self.min_depth) / (self.max_depth - self.min_depth + 1e-8)

    def _apply_per_camera_transforms(self, observation: Dict[str, Any]) -> None:
        for camera_name in self.camera_names:
            camera_obs = {}
            for key, val in observation.items():
                if isinstance(val, dict) and (camera_name in val):
                    camera_obs[key] = val[camera_name]
            if self._cache_images_resized and hasattr(self.transforms, "transforms"):
                for tr in self.transforms.transforms:
                    if tr.__class__.__name__ != 'Resize':
                        camera_obs = tr(camera_obs)
            else:
                camera_obs = self.transforms(camera_obs)
            for key, val in camera_obs.items():
                if isinstance(observation.get(key), dict):
                    observation[key][camera_name] = val

    def _getitem_from_numpy_cache(self, index_info, pair_idx=None):
        """Fast __getitem__ using preprocessed numpy memmap cache."""
        demo_key = index_info['demo_key']
        step_idx = index_info['step_idx']
        traj_len = index_info['traj_len']
        cache = self._numpy_cache
        demo_start, demo_end = self._cache_demo_boundaries[demo_key]
        start_offset = index_info.get('start_offset', 0)

        # Observation history indices (global). Clamped to `start_offset` (not 0)
        # so history windows near the episode start never reach back into the
        # skipped alignment segment — see _load_episode_index.
        end_obs_idx = step_idx + 1
        start_obs_idx = max(start_offset, end_obs_idx - self.obs_hist_length)
        obs_idxs = list(range(start_obs_idx, end_obs_idx))
        if len(obs_idxs) < self.obs_hist_length:
            obs_idxs = [start_offset] * (self.obs_hist_length - len(obs_idxs)) + obs_idxs
        global_idxs = np.array([demo_start + i for i in obs_idxs])

        observation = {}

        # Determine which cameras to load and how to name them (shared helper
        # covers static / R54 random / coupled-pair modes in one place).
        cameras_to_load, _ = self._resolve_primary_alt_cameras(pair_idx=pair_idx)

        for actual_cam, output_cam in cameras_to_load:
            camera_prefix = actual_cam

            # Images — already resized in cache
            if 'image' in self.observation_keys:
                image_key = self._camera_image_key(actual_cam)
                if image_key in cache:
                    img = np.array(cache[image_key][global_idxs], dtype=np.float32) / 255.0
                    if 'image' not in observation:
                        observation['image'] = {}
                    observation['image'][output_cam] = img

            if self.load_future_images and 'image' in self.observation_keys:
                image_key = self._camera_image_key(actual_cam)
                if image_key in cache:
                    fut_steps, fut_valid = self._future_image_steps(step_idx, traj_len)
                    fut_global = np.array([demo_start + s for s in fut_steps])
                    fut_img = np.array(cache[image_key][fut_global], dtype=np.float32) / 255.0
                    observation.setdefault('future_image', {})[output_cam] = fut_img
                    observation.setdefault('future_image_valid', {})[output_cam] = fut_valid

            # Depth — already resized in cache, needs normalization
            if self.use_depth and 'depth' in self.observation_keys:
                depth_key = self._camera_depth_key(actual_cam)
                if depth_key in cache:
                    camera_depth = np.array(cache[depth_key][global_idxs], dtype=np.float32)
                    # Normalize depth to [0, 1] using min/max depth (same as non-cache path)
                    if self.dataset_type == 'cpgen':
                        camera_depth = np.clip(camera_depth, self.min_depth, self.max_depth)
                        camera_depth = (camera_depth - self.min_depth) / (self.max_depth - self.min_depth + 1e-8)
                    if 'depth' not in observation:
                        observation['depth'] = {}
                    observation['depth'][output_cam] = camera_depth

            # Proprioceptive (shared across cameras, only write once)
            for key in ['robot0_eef_pos', 'robot0_eef_quat_site', 'robot0_joint_pos', 'robot0_gripper_qpos', 'object']:
                if key in self.observation_keys and key in cache and key not in observation:
                    observation[key] = np.array(cache[key][global_idxs], dtype=np.float32)

            # Bimanual: per-arm proprio backs the dual-arm state vector (same as
            # non-cache __getitem__). Load whenever present in the numpy cache.
            if self.num_arms > 1:
                for prefix in self.arm_obs_prefixes:
                    for suffix in ('_eef_pos', '_eef_quat_site', '_gripper_qpos'):
                        arm_key = f"{prefix}{suffix}"
                        if arm_key in cache and arm_key not in observation:
                            observation[arm_key] = np.array(cache[arm_key][global_idxs], dtype=np.float32)

            # TCP data (CPGen)
            if self.dataset_type == 'cpgen':
                for key in ['tcp_pixel_coords', 'tcp_dir_x', 'tcp_dir_y', 'tcp_dir_z', 'tcp_pos', 'tcp_orn', 'tcp_valid']:
                    should_load = (key in self.observation_keys) or (key == 'tcp_valid' and 'tcp_pixel_coords' in self.observation_keys)
                    if not should_load:
                        continue
                    suffix = '_tcp_in_frame' if key == 'tcp_valid' else '_' + key
                    if self.num_arms > 1:
                        # Stack per-arm TCP -> [T, num_arms, C], matching non-cache path.
                        arm_keys = [f"{camera_prefix}_{p}{suffix}" for p in self.arm_obs_prefixes]
                        if not all(k in cache for k in arm_keys):
                            continue
                        stacked = np.stack(
                            [np.array(cache[k][global_idxs], dtype=np.float32) for k in arm_keys],
                            axis=1,
                        )
                        if key not in observation:
                            observation[key] = {}
                        observation[key][output_cam] = stacked
                        continue
                    tcp_key = camera_prefix + suffix
                    if tcp_key in cache:
                        if key not in observation:
                            observation[key] = {}
                        tcp_values = np.array(cache[tcp_key][global_idxs], dtype=np.float32)
                        if key == 'tcp_valid':
                            tcp_values = self._apply_tcp_occlusion_mask(
                                tcp_values,
                                cache,
                                camera_prefix,
                                global_idxs,
                            )
                        observation[key][output_cam] = tcp_values

                if self.load_future_tcp:
                    fut_steps, fut_valid = self._future_horizon_steps(
                        step_idx, traj_len, self.future_tcp_horizons
                    )
                    fut_global = np.array([demo_start + s for s in fut_steps])
                    self._store_future_tcp(
                        observation, output_cam, camera_prefix, cache, fut_global, fut_valid
                    )

        # Camera info (from cached attrs)
        demo_attrs = self._cache_demo_attrs.get(demo_key, {})
        if self.dataset_type == 'cpgen' and 'camera_info' in demo_attrs:
            # Use the same (actual, canonical) pairs we just picked for images
            cameras_for_info = cameras_to_load
            # Only cache when cameras are deterministic (per-demo, not per-sample)
            is_dynamic_cameras = bool(self._camera_pair_groups or self._pair_choices)
            use_cache = not is_dynamic_cameras
            if use_cache:
                cached_cam = self._cpgen_camera_info_cache.get(demo_key, None)
            else:
                cached_cam = None
            if cached_cam is None:
                camera_info = json.loads(demo_attrs['camera_info'])
                cached_cam = self._load_cpgen_k_and_ext(
                    self._cameras_requiring_cpgen_k(cameras_for_info), camera_info
                )
                if use_cache:
                    self._cpgen_camera_info_cache[demo_key] = cached_cam
            intrinsics_cache, extrinsics_cache = cached_cam
            if intrinsics_cache:
                observation['camera_intrinsics'] = dict(intrinsics_cache)
                # Scale intrinsics to match cache target resolution (images are pre-resized)
                if self._cache_images_resized and hasattr(self, '_cache_target_size'):
                    from sapolicy.dataset.transform import scale_intrinsics_to_resolution

                    tgt_h, tgt_w = self._cache_target_size
                    for cn, K in observation['camera_intrinsics'].items():
                        observation['camera_intrinsics'][cn] = scale_intrinsics_to_resolution(
                            K, tgt_h, tgt_w
                        )
            if extrinsics_cache:
                observation['camera_extrinsics'] = dict(extrinsics_cache)
        elif self.dataset_type in ['robosuite', 'robocasa'] and hasattr(self, 'hardcoded_intrinsics'):
            for cn in self.camera_names:
                if 'camera_intrinsics' not in observation:
                    observation['camera_intrinsics'] = {}
                observation['camera_intrinsics'][cn] = self.hardcoded_intrinsics[cn].astype(np.float32)

        # Actions (raw, unconverted -- self.actions holds raw per-demo arrays)
        end_act = min(step_idx + self.action_sequence_length, traj_len)
        if self.actions is None:
            raise RuntimeError("Numpy cache mode requires preloaded actions")
        raw_action_window = self.actions[demo_key][step_idx:end_act].astype(np.float32).copy()
        raw_action_window = _pad_action_window(
            raw_action_window, self.action_sequence_length,
            fallback=self.actions[demo_key][traj_len - 1:traj_len].astype(np.float32),
        )

        is_last_step = step_idx >= (traj_len - 1)

        self._align_tcp_to_eef_site_frame(observation)

        # Normalize TCP pixel coords
        self._normalize_tcp_pixel_coords_dict(observation, 'tcp_pixel_coords')
        self._normalize_tcp_pixel_coords_dict(observation, 'future_tcp_pixel_coords')

        # One combined call: GripperWidthTransform/EefRotation6DTransform convert the
        # (possibly obs_hist_length-windowed) observation in place; ActionSE3Transform
        # converts the action window using a single current-step reference pose passed
        # via context -- context takes priority over sample["observation"] in
        # ActionSE3Transform specifically (see embodiment_transforms.py), so the
        # obs_hist_length window never gets misbroadcast against the action window.
        context = None
        if self.use_relative_actions:
            context = {"observation": {
                key: value[-1] for key, value in observation.items()
                if key.endswith('_eef_pos') or key.endswith('_eef_quat_site')
            }}
        actions = self.embodiment_transform.inputs(
            {"observation": observation, "action": raw_action_window}, context=context
        )["action"]
        if self.use_state:
            arm_prefixes = self.arm_obs_prefixes if self.num_arms > 1 else ['robot0']
            try:
                observation['state'] = _assemble_state(observation, arm_prefixes)
            except ValueError as exc:
                raise ValueError(f"{exc} (demo={demo_key}, step={step_idx})") from exc

        raw_relative_action = actions.copy() if self.use_relative_actions else None

        # R119: Compute relative geometry GT (view-invariant)
        if 'object' in observation and 'robot0_eef_pos' in observation and 'robot0_eef_quat_site' in observation:
            observation['rel_geometry'] = self._compute_rel_geometry(
                observation['robot0_eef_pos'], observation['robot0_eef_quat_site'], observation['object']
            )

        self._apply_per_camera_transforms(observation)

        ret = {
            'observation': observation,
            'action': actions,
            'prompt': self.task_description,
            'episode_id': demo_key,
            'step_id': step_idx,
            'is_last_step': np.bool_(is_last_step),
        }
        if raw_relative_action is not None:
            ret['raw_relative_action'] = raw_relative_action
        return ret

    def _needed_view_keys(self, meta):
        """Which image_keys/depth_keys this instance can ever touch.

        meta["image_keys"]/["depth_keys"] enumerate every physical camera the
        cache was built with (e.g. all enumerated third-person views), but a
        given experiment only ever selects a subset via camera_names /
        random_camera_pairs / camera_pair_choices. Loading the rest wastes
        memory (each is a full (N, H, W, C) array), so filter to just the
        cameras this instance's config can select from.
        """
        if self._pair_choices or self.random_camera_pairs:
            needed_cameras = set(self._all_camera_names_pairs)
        else:
            needed_cameras = set(self.camera_names)

        needed_image_keys = {self._camera_image_key(cam) for cam in needed_cameras}
        needed_depth_keys = {self._camera_depth_key(cam) for cam in needed_cameras}

        image_keys = [k for k in meta["image_keys"] if k in needed_image_keys]
        depth_keys = [k for k in meta.get("depth_keys", []) if k in needed_depth_keys]
        return image_keys, depth_keys

    def _load_numpy_cache(self, cache_dir):
        """Load preprocessed numpy memmap cache (built by build_dataset_cache.py).

        Within one process, datasets that share ``(cache_dir, cache_mmap)`` reuse
        the same arrays / mmap handles (important for joint tcp+action loaders).
        Only image/depth arrays for cameras this instance can actually select
        (see ``_needed_view_keys``) are loaded — the cache may hold many more
        physical views than any one experiment uses.
        """
        cache_dir = Path(cache_dir)
        registry_key = (str(cache_dir.resolve()), bool(self.cache_mmap))
        shared = _NUMPY_CACHE_REGISTRY.get(registry_key)

        with open(cache_dir / "meta.json", "r") as fp:
            meta = json.load(fp)
        image_keys, depth_keys = self._needed_view_keys(meta)
        n_skipped = (
            len(meta["image_keys"]) - len(image_keys)
            + len(meta.get("depth_keys", [])) - len(depth_keys)
        )

        if shared is not None:
            missing = [k for k in image_keys + depth_keys if k not in shared["cache"]]
            if missing:
                Log.info(
                    f"Extending shared numpy cache from {cache_dir} with "
                    f"{len(missing)} additional view key(s) needed by this instance: {missing}"
                )
                for k in missing:
                    path = cache_dir / f"{k}.npy"
                    shared["cache"][k] = np.load(
                        str(path),
                        mmap_mode="r" if self.cache_mmap else None,
                    )
            self._numpy_cache = shared["cache"]
            self._cache_demo_boundaries = shared["demo_boundaries"]
            self._cache_demo_attrs = shared["demo_attrs"]
            self._cache_images_resized = True
            self._cache_target_size = shared["target_size"]
            Log.info(
                f"Reusing in-process numpy cache from {cache_dir} "
                f"(mmap={self.cache_mmap}, shared with prior dataset)"
            )
            return

        Log.info(f"Loading numpy cache from {cache_dir} ({meta['total_steps']} steps, "
                 f"{meta['target_size'][0]}x{meta['target_size'][1]})")
        if n_skipped:
            Log.info(
                f"Skipping {n_skipped} unused view key(s) not reachable by this "
                f"dataset's camera selection (loading {len(image_keys)} image + "
                f"{len(depth_keys)} depth key(s) of {len(meta['image_keys'])} + "
                f"{len(meta.get('depth_keys', []))} available)"
            )
        if (
            any(k.endswith("_tcp_pixel_coords") for k in meta.get("scalar_keys", []))
            and not meta.get("tcp_pixel_coords_scaled_to_target", False)
        ):
            Log.warn(
                "Numpy cache contains *_tcp_pixel_coords but meta.json has no "
                "tcp_pixel_coords_scaled_to_target flag. If this cache was built at a "
                "different resolution than the source HDF5, TCP UV labels may be "
                "mis-scaled; rebuild the cache with the current build_dataset_cache.py."
            )

        cache = {}
        for ik in image_keys:
            path = cache_dir / f"{ik}.npy"
            cache[ik] = np.load(
                str(path),
                mmap_mode="r" if self.cache_mmap else None,
            )
        for dk in depth_keys:
            path = cache_dir / f"{dk}.npy"
            if path.exists():
                cache[dk] = np.load(
                    str(path),
                    mmap_mode="r" if self.cache_mmap else None,
                )
        for sk in meta["scalar_keys"]:
            path = cache_dir / f"{sk}.npy"
            if path.exists():
                cache[sk] = np.load(
                    str(path),
                    mmap_mode="r" if self.cache_mmap else None,
                )

        demo_boundaries = {
            dk: tuple(v) for dk, v in meta["demo_boundaries"].items()
        }
        demo_attrs = meta.get("demo_attrs", {})
        target_size = tuple(meta["target_size"])  # (height, width)

        self._numpy_cache = cache
        self._cache_demo_boundaries = demo_boundaries
        self._cache_demo_attrs = demo_attrs
        self._cache_images_resized = True
        self._cache_target_size = target_size
        Log.info(f"Numpy cache loaded: {len(image_keys)} image keys, "
                 f"{len(depth_keys)} depth keys, {len(meta['scalar_keys'])} scalar keys, "
                 f"mmap={self.cache_mmap}")

        _NUMPY_CACHE_REGISTRY[registry_key] = {
            "cache": cache,
            "demo_boundaries": demo_boundaries,
            "demo_attrs": demo_attrs,
            "target_size": target_size,
        }

    def _init_file(self):
        if self._obs_cache is not None or self._numpy_cache is not None:
            return  # Data is in memory/cache, no need for file handle
        if (self.hdf5_file is None) and (self.zarr_path is None):  # 每个 worker 只打开一次
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)

    def load_to_memory(self):
        if isinstance(self.zarr_file, zarr.Array):
            # single array
            self.data = self.zarr_file[:]
        else:
            # group with multiple arrays
            self.data = self._load_group(self.zarr_file)

    def _load_group(self, group):
        mem = {}
        for k, v in group.items():
            if isinstance(v, zarr.Array):
                mem[k] = v[:]
            else:
                mem[k] = self._load_group(v)
        return mem

    def build_transforms(self):
        transforms = self.transforms
        if len(transforms) == 0:
            self.transforms = lambda x: x
            return
        log_str = f"{self.dataset_name} transform layers: \n"
        for idx, transform in enumerate(transforms):
            log_str += (
                (str(transform) + "\n")
                if idx != len(transforms) - 1
                else str(transform)
            )
        Log.info(log_str)
        self.transforms = Compose(transforms)

    def _open_data_group(self):

        f = None
        if self.data is not None:
            data_group = self.data['data']
        else:
            if self.zarr_path and self.zarr_path.exists():
                f = zarr.open(self.zarr_path, 'r')
            elif self.hdf5_path:
                f = h5py.File(self.hdf5_path, 'r', swmr=True)
            else:
                raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path} and Zarr file not found: {self.zarr_path}")
            data_group = f['data']

        return data_group, f

    def _load_actions(self):
        """Raw per-demo actions straight from the dataset, unconverted.

        Embodiment conversion (rotation representation, relative-to-eef SE3, CPGen
        delta-scale) happens later, once per consumer: per-item in __getitem__/
        _getitem_from_numpy_cache via self.embodiment_transform, and once more in
        sapolicy.entrys.normalizer_utils when pooling for the training normalizer.
        """
        data_group, f = self._open_data_group()

        self.actions = {}
        for demo_key in data_group.keys():
            if not demo_key.startswith('demo_'):
                continue
            self.actions[demo_key] = data_group[demo_key]['actions'][:]
        Log.info(f"Loaded {len(self.actions)} actions")

        # Close HDF5 file now that all action data has been read
        if self.data is None and isinstance(f, h5py.File):
            f.close()

    def _load_states(self):
        """Raw per-demo proprio (eef_pos/eef_quat_site/gripper_qpos), unconverted.

        Feeds sapolicy.entrys.normalizer_utils, which pools this across demos/datasets
        and runs it through self.embodiment_transform once when fitting the training
        normalizer. Per-item state assembly in __getitem__/_getitem_from_numpy_cache
        reads directly from the observation already loaded there instead of this store.
        """
        self.states = {}
        if not self.use_state:
            return

        data_group, f = self._open_data_group()
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])

        for demo_key in demo_keys:
            obs_group = data_group[demo_key]["obs"]
            if not all(
                f"{p}_eef_pos" in obs_group and f"{p}_eef_quat_site" in obs_group
                for p in self.arm_obs_prefixes
            ):
                continue
            proprio = {}
            for p in self.arm_obs_prefixes:
                proprio[f"{p}_eef_pos"] = obs_group[f"{p}_eef_pos"][:].astype(np.float32)
                proprio[f"{p}_eef_quat_site"] = obs_group[f"{p}_eef_quat_site"][:].astype(np.float32)
                grip_key = f"{p}_gripper_qpos"
                if grip_key in obs_group:
                    proprio[grip_key] = obs_group[grip_key][:].astype(np.float32)
            self.states[demo_key] = proprio

        if self.data is None and isinstance(f, h5py.File):
            f.close()

    def _load_tcp_preprocessed_data(self):
        """Load preprocessed TCP data from HDF5 file (supports multi-camera with prefixed keys)"""
        if not (self.tcp_preprocessed_path and self.tcp_preprocessed_path.exists()):
            return
        if self._loaded_tcp_data:
            return

        self._loaded_tcp_data = True
        Log.info(f"Loading TCP preprocessed data from {self.tcp_preprocessed_path}")

        with h5py.File(self.tcp_preprocessed_path, 'r', swmr=True) as f:
            data_group = f['data']

            for demo_key in data_group.keys():
                if not demo_key.startswith('demo_'):
                    continue

                demo = data_group[demo_key]
                if 'obs' not in demo:
                    continue

                obs = demo['obs']

                # Initialize TCP data dict for this demo
                self.tcp_data[demo_key] = {}

                # Load all TCP data (camera-prefixed keys)
                # Keys format: {camera_name}_tcp_pixel_coords, {camera_name}_tcp_dir_x, etc.
                for obs_key in obs.keys():
                    if 'tcp_' in obs_key:  # Load any TCP-related key
                        self.tcp_data[demo_key][obs_key] = obs[obs_key][:]

                # Load camera info if available
                if 'tcp_camera_info' in demo.attrs:
                    camera_info = json.loads(demo.attrs['tcp_camera_info'])
                    self.tcp_data[demo_key]['camera_info'] = camera_info

        # Log loaded cameras
        if self.tcp_data:
            first_demo = next(iter(self.tcp_data.values()))
            tcp_keys = [k for k in first_demo.keys() if k != 'camera_info']
            cameras = set()
            for key in tcp_keys:
                # Extract camera name from key (e.g., robot0_agentview_tcp_pixel_coords -> robot0_agentview)
                if '_tcp_' in key:
                    camera_name = key.split('_tcp_')[0]
                    cameras.add(camera_name)

            Log.info(f"Loaded TCP data for {len(self.tcp_data)} demonstrations")
            if cameras:
                Log.info(f"Available cameras in preprocessed data: {sorted(cameras)}")
        else:
            Log.warn("No TCP data loaded!")

    def _load_episode_index(self):
        """Load and index all episodes from HDF5 file"""
        if self.zarr_path and self.zarr_path.exists():
            f = zarr.open(self.zarr_path, 'r')
        elif self.hdf5_path.exists():
            f = h5py.File(self.hdf5_path, 'r', swmr=True)
        else:
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path} and Zarr file not found: {self.zarr_path}")

        data_group = f['data']

        # Get all demo keys
        all_demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])

        # Limit episodes if requested
        if self.max_episodes:
            all_demo_keys = all_demo_keys[:self.max_episodes]

        # Split train/val if requested
        if self.use_train_val_split:
            np.random.seed(self.split_seed)
            n_train = int(len(all_demo_keys) * self.train_val_split_ratio)
            indices = np.random.permutation(len(all_demo_keys))

            if self.split == 'train':
                selected_indices = indices[:n_train]
            else:
                selected_indices = indices[n_train:]

            all_demo_keys = [all_demo_keys[i] for i in selected_indices]

        # One index entry per (demo, step). Per-step TCP validity is not
        # filtered out here — invalid TCP is masked (not dropped) via the
        # tcp_valid field computed lazily in __getitem__.
        #
        # `align_start_offset` (HDF5 demo-group attr, default 0) skips the
        # leading embodiment-specific "move to Panda-aligned start pose"
        # frames that mimicgen bakes into non-Panda cross-embodiment demos
        # (see scripts/compare_embodiment_alignment.py). Absent for Panda and
        # any dataset generated before this field existed, so behavior is
        # unchanged unless the attr is explicitly present.
        self.episode_indices = []
        self._demo_start_offsets = {}
        for demo_key in all_demo_keys:
            traj_len = data_group[demo_key]['actions'].shape[0]
            start_offset = int(data_group[demo_key].attrs.get('align_start_offset', 0))
            start_offset = min(max(start_offset, 0), max(traj_len - 1, 0))
            self._demo_start_offsets[demo_key] = start_offset
            for step_idx in range(start_offset, traj_len):
                self.episode_indices.append({
                    'demo_key': demo_key,
                    'step_idx': step_idx,
                    'traj_len': traj_len,
                    'start_offset': start_offset,
                })

        if isinstance(f, h5py.File):
            f.close()

        Log.info(f"Loaded {len(self.episode_indices)} steps from {len(all_demo_keys)} episodes")

    def __len__(self):
        # Strategy A: when camera_pair_choices is enabled, enumerate every
        # (episode_idx, pair_idx) combination so the dataloader sees each pair
        # for every sample exactly once per epoch.
        return len(self.episode_indices) * self._n_pairs

    def _safe_close_handle(self, attr_name):
        handle = getattr(self, attr_name, None)
        if handle is None:
            return
        try:
            handle.close()
        except Exception:
            # Best-effort close for interpreter shutdown / partially-destructed h5py objects.
            pass
        finally:
            setattr(self, attr_name, None)

    def __del__(self):
        self._safe_close_handle('hdf5_file')
        self._safe_close_handle('tcp_file')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._safe_close_handle('hdf5_file')
        self._safe_close_handle('tcp_file')

    def __getitem__(self, idx):
        """Get a single step from the dataset"""
        self._init_file()
        self._load_tcp_preprocessed_data()

        # Strategy A: dataset length is len(episode_indices) * n_pairs.
        # Decompose idx into (base_idx, pair_idx) so each (sample, pair) combo
        # is enumerated exactly once per epoch.
        n_base = len(self.episode_indices)
        base_idx = idx % n_base
        pair_idx = idx // n_base if self._n_pairs > 1 else None

        index_info = self.episode_indices[base_idx]
        demo_key = index_info['demo_key']
        step_idx = index_info['step_idx']
        traj_len = index_info['traj_len']
        start_offset = index_info.get('start_offset', 0)

        # Use preprocessed numpy cache (memmap) — fastest path
        if self._numpy_cache is not None:
            return self._getitem_from_numpy_cache(index_info, pair_idx=pair_idx)

        # Use preloaded cache if available (bypasses h5py gzip decompression)
        if self._obs_cache is not None:
            obs = self._obs_cache[demo_key]
            demo_attrs = self._obs_cache[demo_key].get('_attrs', {})
        elif self.data is not None:
            demo = self.data['data'][demo_key]
            obs = demo['obs']
            demo_attrs = {}
        else:
            if self.zarr_path:
                file = self.zarr_file
            elif self.hdf5_path:
                file = self.hdf5_file
            else:
                raise ValueError(f"Invalid file type: {type(self.hdf5_file)}")
            demo = file['data'][demo_key]
            obs = demo['obs']
            demo_attrs = dict(demo.attrs)
        observation = {}

        # Get visual observations - unified camera-based loading
        # Load camera-specific images/depth and return with generic keys based on camera_name

        # Align observation history with the current step.
        #
        # We want the last `obs_hist_length` observation frames ending at
        # `step_idx` (inclusive). This matches the standard robomimic
        # convention where `actions[step_idx]` is conditioned on `obs[step_idx]`.
        #
        # NOTE: Previous versions of this dataset used `end_obs_idx = step_idx`
        # (exclusive), which shifted observations by -1 (obs[t-1] → action[t]).
        # That can lead to good training loss but 0 success at rollout time when
        # the policy is conditioned on the current observation.
        end_obs_idx = step_idx + 1
        start_obs_idx = max(start_offset, end_obs_idx - self.obs_hist_length)
        obs_idxs = list(range(start_obs_idx, end_obs_idx))
        # Pad if necessary (when start_obs_idx == start_offset and obs_hist_length > 1)
        if len(obs_idxs) < self.obs_hist_length:
            padding_needed = self.obs_hist_length - len(obs_idxs)
            obs_idxs = [start_offset] * padding_needed + obs_idxs
        obs_idxs = np.array(obs_idxs)

        # HDF5 fancy indexing requires strictly increasing indices.
        # When padding duplicates index 0 (e.g. obs_idxs=[0,0] for step_idx=0),
        # we must read unique indices first, then reconstruct the full sequence.
        unique_idxs, inverse_map = np.unique(obs_idxs, return_inverse=True)

        # Determine primary and optional alternate cameras. In
        # camera_pair_choices mode the alternate is another enumerated view of
        # the same (demo, step), which is the positive pair for contrastive loss.
        cameras_to_load, _ = self._resolve_primary_alt_cameras(pair_idx=pair_idx)

        for actual_camera_name, output_camera_name in cameras_to_load:
            camera_name = actual_camera_name  # used for data lookup
            camera_prefix = camera_name
            # Load image for the specified camera
            if 'image' in self.observation_keys:
                image_key = self._camera_image_key(camera_name)
                if image_key and image_key in obs.keys():
                    # # save fig for debug
                    # plt.imshow(obs[image_key][step_idx])
                    # plt.savefig(f'./tmp/debug_image_{camera_name}_{demo_key}_{step_idx}.png')
                    # plt.close()
                    camera_image = np.array(obs[image_key][unique_idxs], dtype=np.float32) / 255.0
                    camera_image = camera_image[inverse_map]  # (To, H, W, 3) — expand padding frames
                    # images are already in RGB format (verified visually)
                    # Just need normalization from uint8 [0, 255] to float32 [0, 1]

                    if 'image' not in observation:
                        observation['image'] = {}
                    observation['image'][output_camera_name] = camera_image  # Return as generic 'image' key

                    if self.load_future_images:
                        fut_steps, fut_valid = self._future_image_steps(step_idx, traj_len)
                        # Horizons clamp to the last frame near the end of a trajectory,
                        # so fut_steps can repeat (e.g. [77,77,77]). h5py fancy indexing
                        # requires strictly increasing indices -- fine when the file is
                        # preloaded into numpy, but not when reading the dataset directly
                        # (load_to_memory=false). Deduplicate then expand, mirroring the
                        # unique_idxs/inverse_map pattern used for the observation frames.
                        fut_unique, fut_inverse = np.unique(fut_steps, return_inverse=True)
                        fut_img = np.array(obs[image_key][fut_unique], dtype=np.float32)[fut_inverse] / 255.0
                        observation.setdefault('future_image', {})[output_camera_name] = fut_img
                        observation.setdefault('future_image_valid', {})[output_camera_name] = fut_valid

            # Load depth for the specified camera (if available)
            if self.use_depth and 'depth' in self.observation_keys:
                depth_key = self._camera_depth_key(camera_name)
                if depth_key and depth_key in obs.keys():
                    camera_depth = np.array(obs[depth_key][unique_idxs], dtype=np.float32)
                    camera_depth = camera_depth[inverse_map]  # expand padding frames

                    # Ensure depth has consistent shape.
                    # Common cases:
                    # - (To, H, W) from CPGen -> expand to (To, H, W, 1)
                    # - (To, H, W, 1) already ok
                    # - (H, W) -> (H, W, 1) (rare; if obs_hist_length indexing is scalar)
                    if len(camera_depth.shape) == 2:
                        camera_depth = camera_depth[:, :, np.newaxis]
                    elif len(camera_depth.shape) == 3:
                        camera_depth = camera_depth[:, :, :, np.newaxis]
                    # Normalize depth to [0, 1] range using min/max depth
                    if self.dataset_type == 'cpgen':
                        # CPGen depth is typically float32 in meters
                        # Clip and normalize to [0, 1]
                        camera_depth = np.clip(camera_depth, self.min_depth, self.max_depth)
                        camera_depth = (camera_depth - self.min_depth) / (self.max_depth - self.min_depth + 1e-8)

                    if 'depth' not in observation:
                        observation['depth'] = {}
                    observation['depth'][output_camera_name] = camera_depth  # Return as generic 'depth' key
        
            # Get proprioceptive observations
            # if self.use_state:
            for key in ['robot0_eef_pos', 'robot0_eef_quat_site', 'robot0_joint_pos', 'robot0_gripper_qpos', 'object']:
                if key in self.observation_keys and key in obs.keys() and key not in observation:
                    observation[key] = obs[key][unique_idxs].astype(np.float32)[inverse_map]

            # Bimanual: per-arm proprio backs the dual-arm state vector, so it is
            # loaded whenever present rather than gated on observation_keys.
            if self.num_arms > 1:
                for prefix in self.arm_obs_prefixes:
                    for suffix in ('_eef_pos', '_eef_quat_site', '_gripper_qpos'):
                        arm_key = f"{prefix}{suffix}"
                        if arm_key in obs.keys():
                            observation[arm_key] = obs[arm_key][unique_idxs].astype(np.float32)[inverse_map]

            # Load TCP data - different handling for CPGen vs RoboSuite/RoboCasa
            if self.dataset_type == 'cpgen':
                # CPGen has TCP data directly in HDF5
                for key in ['tcp_pixel_coords', 'tcp_dir_x', 'tcp_dir_y', 'tcp_dir_z', 'tcp_pos', 'tcp_orn', 'tcp_valid']:
                    should_load = (key in self.observation_keys) or (key == 'tcp_valid' and 'tcp_pixel_coords' in self.observation_keys)
                    if not should_load:
                        continue
                    suffix = '_tcp_in_frame' if key == 'tcp_valid' else '_' + key
                    if self.num_arms > 1:
                        # Bimanual: stack the per-arm keys on a new axis so every
                        # TCP tensor becomes [T, num_arms, C]. The aux head emits the
                        # same axis (LatentAuxiliaryModel.num_tcp) and the L1/BCE
                        # losses reduce over it without further changes.
                        arm_keys = [f"{camera_prefix}_{p}{suffix}" for p in self.arm_obs_prefixes]
                        if not all(k in obs.keys() for k in arm_keys):
                            continue
                        stacked = np.stack(
                            [obs[k][unique_idxs].astype(np.float32) for k in arm_keys], axis=1
                        )
                        if key not in observation:
                            observation[key] = {}
                        observation[key][output_camera_name] = stacked[inverse_map]
                        continue
                    tcp_key = camera_prefix + suffix
                    if tcp_key in obs.keys():
                        if key not in observation:
                            observation[key] = {}
                        tcp_values = obs[tcp_key][unique_idxs].astype(np.float32)
                        if key == 'tcp_valid':
                            tcp_values = self._apply_tcp_occlusion_mask(
                                tcp_values,
                                obs,
                                camera_prefix,
                                unique_idxs,
                            )
                        observation[key][output_camera_name] = tcp_values[inverse_map]
                        # else:
                        #     raise KeyError(f"TCP key '{tcp_key}' not found in observation for dataset '{self.dataset_name}'. "
                        #                 f"Available keys: {list(obs.keys())}")
                if self.load_future_tcp:
                    fut_steps, fut_valid = self._future_horizon_steps(
                        step_idx, traj_len, self.future_tcp_horizons
                    )
                    self._store_future_tcp(
                        observation,
                        output_camera_name,
                        camera_prefix,
                        obs,
                        np.asarray(fut_steps),
                        fut_valid,
                    )
            else:
                # RoboSuite/RoboCasa: load from preprocessed data with camera-prefixed keys
                if demo_key in self.tcp_data:
                    tcp_demo_data = self.tcp_data[demo_key]

                    # Construct camera-prefixed keys
                    # e.g., robot0_agentview_tcp_pixel_coords, robot0_eye_in_hand_tcp_pixel_coords
                    # Load TCP pixel coords for the current camera
                    if 'tcp_pixel_coords' in self.observation_keys:
                            tcp_key = f'{camera_prefix}_tcp_pixel_coords'
                            if tcp_key in tcp_demo_data:
                                if 'tcp_pixel_coords' not in observation:
                                    observation['tcp_pixel_coords'] = {}
                                observation['tcp_pixel_coords'][output_camera_name] = tcp_demo_data[tcp_key][unique_idxs].astype(np.float32)[inverse_map]
                            else:
                                raise KeyError(f"TCP key '{tcp_key}' not found in preprocessed data for dataset '{self.dataset_name}'. "
                                            f"Available TCP keys: {list(tcp_demo_data.keys())}")

                    # Load TCP direction vectors for the current camera
                    for key in ['tcp_dir_x', 'tcp_dir_y', 'tcp_dir_z', 'tcp_pos', 'tcp_orn', 'tcp_valid']:
                        should_load = (key in self.observation_keys) or (key == 'tcp_valid' and 'tcp_pixel_coords' in self.observation_keys)
                        if should_load:
                            tcp_key = f'{camera_prefix}_tcp_in_frame' if key == 'tcp_valid' else f'{camera_prefix}_{key}'
                            if tcp_key in tcp_demo_data:
                                if key not in observation:
                                    observation[key] = {}
                                tcp_values = tcp_demo_data[tcp_key][unique_idxs].astype(np.float32)
                                if key == 'tcp_valid':
                                    tcp_values = self._apply_tcp_occlusion_mask(
                                        tcp_values,
                                        tcp_demo_data,
                                        camera_prefix,
                                        unique_idxs,
                                    )
                                observation[key][output_camera_name] = tcp_values[inverse_map]
                            else:
                                raise KeyError(f"TCP key '{tcp_key}' not found in preprocessed data for dataset '{self.dataset_name}'. "
                                            f"Available TCP keys: {list(tcp_demo_data.keys())}")

                    if self.load_future_tcp:
                        fut_steps, fut_valid = self._future_horizon_steps(
                            step_idx, traj_len, self.future_tcp_horizons
                        )
                        self._store_future_tcp(
                            observation,
                            output_camera_name,
                            camera_prefix,
                            tcp_demo_data,
                            np.asarray(fut_steps),
                            fut_valid,
                        )
                else:
                    # If TCP data is required but not available, raise an error
                    tcp_keys = [k for k in self.observation_keys if 'tcp_' in k]
                    if tcp_keys or self.load_future_tcp:
                        raise ValueError(f"TCP data is required but no preprocessed TCP data found for demo_key '{demo_key}' "
                                    f"in dataset '{self.dataset_name}'. Required TCP keys: {tcp_keys}")

        # Get camera info - unified camera-based loading
        # Always return as generic 'camera_intrinsics' and 'camera_extrinsics' for the selected camera
        # R54: When random camera modes, load intrinsics from actual camera, store under output name
        if self.dataset_type == 'cpgen':
            # CPGen: load from HDF5 attributes for the specified camera (cached)
            if 'camera_info' in demo_attrs:
                # Disable demo-level cache whenever the (actual, canonical)
                # mapping can change per sample — otherwise the cached dict
                # (keyed by canonical name) would lock in the first-sampled
                # actual camera's intrinsics and silently mis-route later samples.
                is_dynamic_cameras = bool(self._camera_pair_groups or self._pair_choices)
                if is_dynamic_cameras:
                    camera_info = json.loads(demo_attrs['camera_info'])
                    intrinsics_dict, extrinsics_dict = self._load_cpgen_k_and_ext(
                        self._cameras_requiring_cpgen_k(cameras_to_load), camera_info
                    )
                    if intrinsics_dict:
                        observation['camera_intrinsics'] = intrinsics_dict
                    if extrinsics_dict:
                        observation['camera_extrinsics'] = extrinsics_dict
                else:
                    cached = self._cpgen_camera_info_cache.get(demo_key, None)
                    if cached is None:
                        camera_info = json.loads(demo_attrs['camera_info'])
                        cached = self._load_cpgen_k_and_ext(
                            self._cameras_requiring_cpgen_k(
                                [(cn, cn) for cn in self.camera_names]
                            ),
                            camera_info,
                        )
                        self._cpgen_camera_info_cache[demo_key] = cached
                    intrinsics_cache, extrinsics_cache = cached
                    if intrinsics_cache:
                        observation['camera_intrinsics'] = dict(intrinsics_cache)
                    if extrinsics_cache:
                        observation['camera_extrinsics'] = dict(extrinsics_cache)

        else:
            # RoboSuite/RoboCasa: use hardcoded intrinsics for the selected camera
            if hasattr(self, 'hardcoded_intrinsics'):
                for actual_cn, output_cn in cameras_to_load:
                    if 'camera_intrinsics' not in observation:
                        observation['camera_intrinsics'] = {}
                    if actual_cn in self.hardcoded_intrinsics:
                        observation['camera_intrinsics'][output_cn] = self.hardcoded_intrinsics[actual_cn].astype(np.float32)

            # Check if preprocessed data has camera info with extrinsics
            if demo_key in self.tcp_data and 'camera_info' in self.tcp_data[demo_key]:
                camera_info = self.tcp_data[demo_key]['camera_info']
                for actual_cn, output_cn in cameras_to_load:
                    if actual_cn in camera_info and 'extrinsics' in camera_info[actual_cn]:
                        if 'camera_extrinsics' not in observation:
                            observation['camera_extrinsics'] = {}
                        observation['camera_extrinsics'][output_cn] = np.array(
                            camera_info[actual_cn]['extrinsics'], dtype=np.float32
                        )

        # Get action sequence (raw, unconverted -- self.actions holds raw per-demo arrays)
        # HDF5 requires indices to be in increasing order, so we handle padding separately
        end_idx = min(step_idx + self.action_sequence_length, traj_len)
        if self.actions is not None:
            raw_action_window = self.actions[demo_key][step_idx:end_idx].astype(np.float32).copy()
        else:
            raw_action_window = demo['actions'][step_idx:end_idx].astype(np.float32)

        raw_action_window = _pad_action_window(
            raw_action_window, self.action_sequence_length,
            fallback=demo['actions'][traj_len - 1:traj_len].astype(np.float32),
        )

        is_last_step = step_idx >= (traj_len - 1)

        self._align_tcp_to_eef_site_frame(observation)

        # Normalize TCP pixel coordinates if present
        self._normalize_tcp_pixel_coords_dict(observation, 'tcp_pixel_coords')
        self._normalize_tcp_pixel_coords_dict(observation, 'future_tcp_pixel_coords')

        # One combined call: GripperWidthTransform/EefRotation6DTransform convert the
        # (possibly obs_hist_length-windowed) observation in place; ActionSE3Transform
        # converts the action window using a single current-step reference pose passed
        # via context -- context takes priority over sample["observation"] in
        # ActionSE3Transform specifically (see embodiment_transforms.py), so the
        # obs_hist_length window (and, for bimanual, per-arm keys) never gets
        # misbroadcast against the action window.
        context = None
        if self.use_relative_actions:
            context = {"observation": {
                key: value[-1] for key, value in observation.items()
                if key.endswith('_eef_pos') or key.endswith('_eef_quat_site')
            }}
        actions = self.embodiment_transform.inputs(
            {"observation": observation, "action": raw_action_window}, context=context
        )["action"]

        # Save raw relative action BEFORE normalization for consistency loss.
        # Only meaningful when use_relative_actions=True (actual SE(3) deltas).
        raw_relative_action = actions.copy() if self.use_relative_actions else None

        if self.use_state:
            arm_prefixes = self.arm_obs_prefixes if self.num_arms > 1 else ['robot0']
            try:
                observation['state'] = _assemble_state(observation, arm_prefixes)
            except ValueError as exc:
                raise ValueError(f"{exc} (demo={demo_key}, step={step_idx})") from exc

        # R119: Compute relative geometry GT (view-invariant)
        if 'object' in observation and 'robot0_eef_pos' in observation and 'robot0_eef_quat_site' in observation:
            observation['rel_geometry'] = self._compute_rel_geometry(
                observation['robot0_eef_pos'], observation['robot0_eef_quat_site'], observation['object']
            )

        # Apply per-camera transforms. Be robust to partially-missing camera keys.
        if os.environ.get("SAPOLICY_DEBUG_MISSING_CAMERA_KEYS", "0") == "1":
            for cam in self.camera_names:
                missing = []
                for key, val in observation.items():
                    if isinstance(val, dict) and cam not in val:
                        missing.append(key)
                if missing:
                    Log.warn(f"[dataset] Missing camera keys for cam={cam}: {missing}")

        self._apply_per_camera_transforms(observation)

        ret = {
            'observation': observation,
            'action': actions,
            'prompt': self.task_description,
            'episode_id': demo_key,
            'step_id': step_idx,
            'is_last_step': np.bool_(is_last_step),
        }
        if raw_relative_action is not None:
            ret['raw_relative_action'] = raw_relative_action
        return ret


class CPGenIterableDataset(IterableDataset):
    """
    Iterable version of CPGen dataset for use with DataLoader
    Supports multi-worker and distributed training
    """

    def __init__(self, base_dataset: RobomimicHDF5Dataset, shuffle: bool = True, seed: int = 42):
        """
        Args:
            base_dataset: RobomimicHDF5Dataset instance
            shuffle: Whether to shuffle indices
            seed: Random seed for shuffling
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single worker
            iter_start = 0
            iter_end = len(self.base_dataset)
        else:
            # Multiple workers - split data
            per_worker = int(np.ceil(len(self.base_dataset) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.base_dataset))

        # Get indices
        indices = np.arange(iter_start, iter_end)

        # Shuffle if requested
        if self.shuffle:
            # Use worker-specific seed
            rng = np.random.RandomState(self.seed + worker_info.id if worker_info else self.seed)
            rng.shuffle(indices)

        # Yield data
        for idx in indices:
            yield self.base_dataset[idx]


def create_cpgen_dataloader(
    hdf5_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create a DataLoader for CPGen dataset

    Args:
        hdf5_path: Path to HDF5 file
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for RobomimicHDF5Dataset

    Returns:
        DataLoader instance
    """
    # Create base dataset
    base_dataset = RobomimicHDF5Dataset(hdf5_path=hdf5_path, **dataset_kwargs)

    # Create iterable dataset
    dataset = CPGenIterableDataset(base_dataset, shuffle=shuffle)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == "__main__":
    # Test the dataset
    hdf5_path = "/home/mhliu/projects/datasets/cpgen/datasets/generated/NutAssemlySquare/nutassembly-rgb-w-depth-ext-int.hdf5"

    # import zarr

    # f = h5py.File(hdf5_path, "r")
    # z = zarr.open(hdf5_path.replace(".hdf5", ".zarr"), mode="w")
    # zarr.convenience.copy(f, z, name="/")

    # f.close()

    # exit(0)

    print("Creating dataset...")
    dataset = RobomimicHDF5Dataset(
        hdf5_path=hdf5_path,
        zarr_path=hdf5_path.replace(".hdf5", ".zarr"),
        dataset_name="ThreePieceAssemblyWide",
        split='train',
        use_train_val_split=True,
        train_val_split_ratio=0.9,
        action_sequence_length=4,
        normalize_actions=True,
        use_agentview=True,
        use_eye_in_hand=False,
        use_depth=True,
        max_episodes=10,  # Test with 10 episodes
        observation_keys=['image', 'depth', 'robot0_eef_pos', 'robot0_eef_quat_site', 'robot0_joint_pos', 'robot0_gripper_qpos', 'tcp_pixel_coords', 'tcp_dir_x', 'tcp_dir_y', 'tcp_dir_z', 'tcp_pos', 'tcp_orn'],
    )

    print(f"\nDataset size: {len(dataset)} steps")

    print("\nTesting data loading...")
    sample = dataset[0]

    print("\nObservation keys:", sample['observation'].keys())
    print("\nShapes:")
    for key, val in sample['observation'].items():
        if isinstance(val, dict):
            for camera_name, camera_val in val.items():
                print(f"  {key}[{camera_name}]: {camera_val.shape} dtype: {camera_val.dtype}")
        else:
            print(f"  {key}: {val.shape} dtype: {val.dtype}")

    print(f"\nAction shape: {sample['action'].shape}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Episode ID: {sample['episode_id']}")
    print(f"Step ID: {sample['step_id']}")

    print("\n✅ Dataset test successful!")
