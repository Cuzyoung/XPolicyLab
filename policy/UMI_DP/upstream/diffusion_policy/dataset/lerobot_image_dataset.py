"""A small, explicit adapter from a LeRobot dataset root to Diffusion Policy.

The adapter performs only the documented representation conversion needed by
this repository: TacCap stores the first two *columns* of a world-from-body
rotation, while the bundled UMI utilities consume the first two *rows*.  The
conversion reconstructs the rotation matrix and re-serializes those rows; it
does not change the physical pose or coordinate frame.

The first implementation reads the standard LeRobot parquet files and the
per-feature MP4 files directly.  It uses the parquet ``index`` as the global
video frame index and derives episode boundaries from ``episode_index``.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.dataset.base_dataset import BaseImageDataset
from XPolicyLab.policy.UMI_DP.upstream.diffusion_policy.model.common.normalizer import LinearNormalizer


def _pose10d_to_mat(pose10d: np.ndarray) -> np.ndarray:
    """Vectorized row-layout rotation-6D to homogeneous matrices."""
    pose10d = np.asarray(pose10d, dtype=np.float32)
    if pose10d.shape[-1] != 9:
        raise ValueError(f"pose10d must have width 9, got {pose10d.shape}")
    d6 = pose10d[..., 3:]
    b1 = d6[..., :3]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True).clip(min=1e-12)
    a2 = d6[..., 3:]
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    mat = np.zeros(pose10d.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, :3] = np.stack((b1, b2, b3), axis=-2)
    mat[..., :3, 3] = pose10d[..., :3]
    mat[..., 3, 3] = 1
    return mat


def _mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """Vectorized homogeneous matrices to row-layout pose-9 vectors."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape[-2:] != (4, 4):
        raise ValueError(f"mat must have shape (..., 4, 4), got {mat.shape}")
    return np.concatenate(
        [mat[..., :3, 3], mat[..., :2, :3].reshape(mat.shape[:-2] + (6,))],
        axis=-1,
    ).astype(np.float32, copy=False)


def _pose10d_to_relative(pose10d: np.ndarray, base_pose10d: np.ndarray) -> np.ndarray:
    """Express one or more 9-D pose vectors in the base pose frame.

    The 9-D vector is ``[x, y, z, rotation6d]``.  The operation is carried
    out on full SE(3) matrices so that rotation-6D is never subtracted or
    otherwise treated as a Euclidean vector.
    """
    pose10d = np.asarray(pose10d, dtype=np.float32)
    base_pose10d = np.asarray(base_pose10d, dtype=np.float32)
    pose_mat = _pose10d_to_mat(pose10d)
    base_mat = _pose10d_to_mat(base_pose10d)
    inv_base = np.linalg.inv(base_mat)
    # Broadcast a per-batch base over a trajectory dimension when needed.
    while inv_base.ndim < pose_mat.ndim:
        inv_base = np.expand_dims(inv_base, axis=-3)
    return _mat_to_pose10d(inv_base @ pose_mat).astype(np.float32, copy=False)


def convert_action_to_relative(action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    """Convert a packed bimanual absolute action to a UMI relative trajectory.

    Parameters
    ----------
    action:
        Array with shape ``(..., 20)`` and layout
        ``[arm0 pose9, gripper0, arm1 pose9, gripper1]``.
    current_state:
        Current packed state with shape ``(..., 20)`` or ``(20,)``.  A single
        state is broadcast over all action trajectory dimensions.

    Returns
    -------
    np.ndarray
        Same shape as ``action``.  Each arm's pose is
        ``T_current_ee^-1 @ T_target_ee``; gripper values are unchanged.
    """
    action = np.asarray(action, dtype=np.float32)
    current_state = np.asarray(current_state, dtype=np.float32)
    if action.shape[-1] != 20 or current_state.shape[-1] != 20:
        raise ValueError("action and current_state must have packed width 20")
    result = action.copy()
    for arm in range(2):
        start = arm * 10
        result[..., start:start + 9] = _pose10d_to_relative(
            action[..., start:start + 9],
            current_state[..., start:start + 9],
        )
    return result


@dataclass(frozen=True)
class _Episode:
    episode_index: int
    start: int
    end: int  # exclusive


class _VideoSource:
    """Lazy, process-local random access to one LeRobot video feature."""

    FRAME_CACHE_SIZE = 32

    def __init__(self, paths: Sequence[str], expected_frames: int):
        try:
            import av
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "LeRobotImageDataset requires PyAV. Install the diffusion-policy "
                "extras (av) before loading video observations."
            ) from exc

        self._av = av
        self.paths = tuple(paths)
        self._segments = []
        total = 0
        for path in self.paths:
            container = av.open(path)
            try:
                stream = container.streams.video[0]
                frames = int(stream.frames)
                if frames <= 0:
                    raise ValueError(f"video stream has no frames: {path}")
                self._segments.append((path, total, frames, float(stream.average_rate or 30.0)))
                total += frames
            finally:
                container.close()
        if total != expected_frames:
            raise ValueError(
                f"video frame count mismatch for {os.path.basename(os.path.dirname(paths[0]))}: "
                f"{total} frames, parquet has {expected_frames}"
            )
        self._containers: Dict[str, object] = {}
        self._cache: OrderedDict[Tuple[str, int], np.ndarray] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_containers"] = {}
        state["_cache"] = OrderedDict()
        return state

    def close(self):
        for container in self._containers.values():
            try:
                container.close()
            except Exception:
                pass
        self._containers.clear()

    def __del__(self):  # pragma: no cover - interpreter shutdown ordering
        self.close()

    def _segment_for(self, index: int):
        for path, start, frames, fps in self._segments:
            if start <= index < start + frames:
                return path, start, frames, fps
        raise IndexError(f"video frame index out of range: {index}")

    def read(self, indices: Sequence[int]) -> list[np.ndarray]:
        cache_limit = max(self.FRAME_CACHE_SIZE, len(indices))
        missing_by_path: Dict[str, list[int]] = {}
        for index in indices:
            path, start, _, _ = self._segment_for(int(index))
            local = int(index) - start
            key = (path, local)
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                missing_by_path.setdefault(path, []).append(local)

        for path, requests in missing_by_path.items():
            # Seeking to the earliest requested frame and decoding until the
            # latest requested frame is considerably cheaper than decoding the
            # whole MP4 while still working with ordinary H.264 keyframes.
            first = min(requests)
            last = max(requests)
            container = self._containers.get(path)
            if container is None:
                container = self._av.open(path)
                self._containers[path] = container
            stream = container.streams.video[0]
            fps = float(stream.average_rate or 30.0)
            container.seek(int((first / fps) / float(stream.time_base)), stream=stream,
                           backward=True, any_frame=False)
            wanted = set(requests)
            for frame in container.decode(stream):
                local = int(round(float(frame.time) * fps))
                if local in wanted:
                    self._cache[(path, local)] = frame.to_ndarray(format="rgb24")
                    self._cache.move_to_end((path, local))
                    while len(self._cache) > cache_limit:
                        self._cache.popitem(last=False)
                if local >= last:
                    break
            missing = [x for x in wanted if (path, x) not in self._cache]
            if missing:
                raise RuntimeError(f"could not decode frames {missing} from {path}")

        return [self._cache[(self._segment_for(int(i))[0],
                             int(i) - self._segment_for(int(i))[1])]
                for i in indices]

class _CachedVideoSource:
    """Memory-mapped, pre-decoded RGB frames written by build_video_cache.py."""
    def __init__(self, path: str, expected_frames: int, use_mmap: bool):
        self._frames = np.load(path, mmap_mode="r" if use_mmap else None)
        if self._frames.shape[0] > expected_frames or self._frames.shape[1:] != (224, 224, 3):
            raise ValueError(f"invalid cached video shape {self._frames.shape}")
    def read(self, indices: Sequence[int]) -> list[np.ndarray]:
        return [np.asarray(self._frames[int(i)]) for i in indices]


class LeRobotImageDataset(BaseImageDataset):
    """Expose one LeRobot root through the Diffusion Policy dataset API.

    The expected action is the 20-D packed vector
    ``[left(3 pos, 6 rot6d, 1 gripper), right(...)]``.  TacCap's documented
    world-frame/column-layout pose is converted to the row-layout expected by
    ``umi.common.pose_util``.  By default samples are emitted in UMI's
    current-EE relative-trajectory representation.
    """

    DEFAULT_IMAGE_KEY_MAP = {
        "camera0_rgb": "observation.images.left_wrist",
        "camera1_rgb": "observation.images.right_wrist",
    }

    def __init__(
        self,
        dataset_path: str,
        shape_meta: dict,
        *,
        image_key_map: Optional[Mapping[str, str]] = None,
        output_resolution: Tuple[int, int] = (224, 224),
        action_padding: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        action_mode: str = "relative_trajectory",
        observation_mode: str = "relative",
        pose_frame: str = "world",
        position_unit: str = "m",
        rotation_layout: str = "column",
        verify_semantics: bool = True,
        image_cache_dir: Optional[str] = None,
        use_mmap: bool = False,
    ):
        self.dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        self.image_cache_dir = os.path.abspath(os.path.expanduser(image_cache_dir)) if image_cache_dir else None
        self.use_mmap = bool(use_mmap)
        if self.use_mmap and not self.image_cache_dir:
            raise ValueError("use_mmap=True requires image_cache_dir")
        self.shape_meta = copy.deepcopy(shape_meta)
        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(self.dataset_path)
        if action_mode not in {"absolute_next_state", "relative_trajectory"}:
            raise ValueError(
                "action_mode must be 'relative_trajectory' or 'absolute_next_state'"
            )
        if observation_mode not in {"relative", "absolute"}:
            raise ValueError("observation_mode must be 'relative' or 'absolute'")
        if pose_frame != "world":
            raise ValueError("TacCap tcp poses are documented in the world frame; pose_frame must be 'world'")
        if position_unit != "m":
            raise ValueError("TacCap tcp positions are documented in meters; position_unit must be 'm'")
        if len(output_resolution) != 2 or min(output_resolution) <= 0:
            raise ValueError("output_resolution must be (height, width) with positive values")
        self.output_resolution = tuple(int(x) for x in output_resolution)
        self.action_padding = bool(action_padding)
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.rotation_layout = rotation_layout
        self.semantic_contract = {
            "action_mode": action_mode,
            "observation_mode": observation_mode,
            "action_reference": "last_observation_per_arm" if action_mode == "relative_trajectory" else None,
            "action_source_semantics": "absolute_next_state",
            "pose_frame": pose_frame,
            "world_axes": "FLU",
            "rotation_direction": "world_from_body",
            "position_unit": position_unit,
            "rotation_layout": rotation_layout,
            "output_rotation_layout": "row" if rotation_layout == "column" else rotation_layout,
            "output_pose_frame": "current_ee" if observation_mode == "relative" or action_mode == "relative_trajectory" else "world",
        }
        if rotation_layout not in {"unknown", "row", "column"}:
            raise ValueError("rotation_layout must be 'unknown', 'row', or 'column'")

        self._load_metadata(verify_semantics=verify_semantics)
        self._configure_keys(image_key_map)
        self._configure_sampling(seed=seed, val_ratio=val_ratio)
        self._video_sources: Dict[str, _VideoSource] = {}

    def _load_metadata(self, verify_semantics: bool):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "LeRobotImageDataset requires pyarrow. Install the diffusion-policy "
                "extras (pyarrow) before loading parquet data."
            ) from exc

        info_path = os.path.join(self.dataset_path, "meta", "info.json")
        with open(info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)
        expected_names = [
            "left_tcp.x", "left_tcp.y", "left_tcp.z",
            "left_tcp.r1", "left_tcp.r2", "left_tcp.r3", "left_tcp.r4", "left_tcp.r5", "left_tcp.r6",
            "left_gripper.pos",
            "right_tcp.x", "right_tcp.y", "right_tcp.z",
            "right_tcp.r1", "right_tcp.r2", "right_tcp.r3", "right_tcp.r4", "right_tcp.r5", "right_tcp.r6",
            "right_gripper.pos",
        ]
        for feature_name in ("action", "observation.state"):
            feature = self.info.get("features", {}).get(feature_name, {})
            if feature.get("shape") != [20] or feature.get("names") != expected_names:
                raise ValueError(
                    f"{feature_name} metadata is not the expected TacCap 20-D bimanual layout"
                )
        parquet_paths = sorted(glob.glob(os.path.join(self.dataset_path, "data", "chunk-*", "*.parquet")))
        if not parquet_paths:
            raise FileNotFoundError(f"no parquet files under {self.dataset_path}/data")
        tables = [pq.read_table(path) for path in parquet_paths]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)

        def fixed_array(name: str, width: int):
            if name not in table.column_names:
                raise ValueError(f"missing parquet column: {name}")
            column = table[name].combine_chunks()
            if not str(column.type).startswith("fixed_size_list"):
                raise ValueError(f"{name} must be a fixed-size list column, got {column.type}")
            values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32)
            if values.size != len(column) * width:
                raise ValueError(f"{name} has unexpected width")
            return values.reshape(len(column), width)

        n = table.num_rows
        self.actions = fixed_array("action", 20)
        self.states = fixed_array("observation.state", 20)
        self.episode_index = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        self.frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        self.global_index = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        self.timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        if not (len(self.actions) == len(self.states) == n == len(self.episode_index)):
            raise ValueError("parquet columns have inconsistent lengths")
        order = np.argsort(self.global_index, kind="stable")
        if not np.array_equal(order, np.arange(n)):
            self.actions = self.actions[order]
            self.states = self.states[order]
            self.episode_index = self.episode_index[order]
            self.frame_index = self.frame_index[order]
            self.timestamps = self.timestamps[order]
            self.global_index = self.global_index[order]
        if not np.array_equal(self.global_index, np.arange(n)):
            raise ValueError("parquet index must be contiguous from 0 for video alignment")
        if not np.isfinite(self.actions).all() or not np.isfinite(self.states).all():
            raise ValueError("state/action contains non-finite values")

        boundaries = np.flatnonzero(np.r_[True, self.episode_index[1:] != self.episode_index[:-1], True])
        self.episodes = tuple(
            _Episode(int(self.episode_index[boundaries[i]]), int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(len(boundaries) - 1)
        )
        if verify_semantics:
            for ep in self.episodes:
                sl = slice(ep.start, ep.end)
                if not np.array_equal(self.frame_index[sl], np.arange(ep.end - ep.start)):
                    raise ValueError(f"frame_index is not contiguous in episode {ep.episode_index}")
                if ep.end - ep.start > 1 and np.any(np.diff(self.timestamps[sl]) <= 0):
                    raise ValueError(f"timestamps are not increasing in episode {ep.episode_index}")
                if ep.end - ep.start > 1 and not np.allclose(
                    self.actions[ep.start:ep.end - 1], self.states[ep.start + 1:ep.end], atol=1e-6, rtol=0
                ):
                    raise ValueError(
                        "action is not the next-frame absolute state; pass verify_semantics=False "
                        "only after checking the source export"
                    )

        if self.rotation_layout == "column":
            self.actions = self._convert_rotation_layout(self.actions)
            self.states = self._convert_rotation_layout(self.states)

        self.task_names = self._read_tasks(pq)

    @staticmethod
    def _convert_rotation_layout(values: np.ndarray) -> np.ndarray:
        """Convert TacCap [R[:,0], R[:,1]] to UMI [R[0,:], R[1,:]]."""
        result = values.copy()
        for arm in range(2):
            start = arm * 10 + 3
            d6 = values[:, start:start + 6]
            b1 = d6[:, :3].copy()
            b1 /= np.linalg.norm(b1, axis=-1, keepdims=True).clip(min=1e-12)
            a2 = d6[:, 3:].copy()
            b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
            b2 /= np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-12)
            b3 = np.cross(b1, b2)
            rotation = np.stack((b1, b2, b3), axis=-1)
            result[:, start:start + 6] = np.concatenate((rotation[:, 0, :], rotation[:, 1, :]), axis=-1)
        return result

    def _read_tasks(self, pq):
        path = os.path.join(self.dataset_path, "meta", "tasks.parquet")
        if not os.path.exists(path):
            return tuple()
        table = pq.read_table(path)
        column = "task" if "task" in table.column_names else table.column_names[0]
        return tuple(str(x) for x in table[column].to_pylist())

    def _configure_keys(self, image_key_map):
        self.rgb_keys = []
        self.lowdim_keys = []
        self._image_sources = {}
        for key, attr in self.shape_meta.get("obs", {}).items():
            typ = attr.get("type", "low_dim")
            if typ == "rgb":
                self.rgb_keys.append(key)
                source = (image_key_map or {}).get(key, self.DEFAULT_IMAGE_KEY_MAP.get(key))
                if source is None:
                    raise ValueError(f"no source video mapping for RGB key {key}; pass image_key_map")
                self._image_sources[key] = source
                expected = tuple(attr.get("shape", ()))
                if expected != (3, *self.output_resolution):
                    raise ValueError(
                        f"RGB key {key} shape must be (3, H, W) = (3, {self.output_resolution[0]}, "
                        f"{self.output_resolution[1]})"
                    )
            elif typ == "low_dim":
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"unsupported observation type {typ!r} for {key}")
        action_shape = tuple(self.shape_meta.get("action", {}).get("shape", (20,)))
        if action_shape != (20,):
            raise ValueError(f"the first LeRobot adapter requires action shape [20], got {action_shape}")

    @staticmethod
    def _integer_steps(attr: Mapping, name: str) -> int:
        value = attr.get(name, 1)
        if abs(float(value) - round(float(value))) > 1e-6 or int(round(float(value))) <= 0:
            raise ValueError(f"{name} must be a positive integer in the first adapter")
        return int(round(float(value)))

    def _configure_sampling(self, seed: int, val_ratio: float):
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")
        obs_attrs = self.shape_meta.get("obs", {})
        self._obs_specs = {
            key: (int(attr.get("horizon", 1)), self._integer_steps(attr, "down_sample_steps"),
                  int(round(float(attr.get("latency_steps", 0)))))
            for key, attr in obs_attrs.items()
        }
        action_attr = self.shape_meta.get("action", {})
        self._action_horizon = int(action_attr.get("horizon", 1))
        self._action_downsample = self._integer_steps(action_attr, "down_sample_steps")
        if self._action_horizon <= 0:
            raise ValueError("action horizon must be positive")
        self._sample_refs = []
        for ep_idx, ep in enumerate(self.episodes):
            history = max((h - 1) * ds + max(0, latency) for h, ds, latency in self._obs_specs.values()) if self._obs_specs else 0
            future = (self._action_horizon - 1) * self._action_downsample
            last_anchor = ep.end - 1 if self.action_padding else ep.end - 1 - future
            for anchor in range(ep.start + history, last_anchor + 1):
                self._sample_refs.append((ep_idx, anchor))

        rng = np.random.default_rng(seed)
        val_eps = set()
        if val_ratio > 0 and len(self.episodes) > 1:
            count = max(1, int(round(len(self.episodes) * val_ratio)))
            val_eps = set(rng.choice(len(self.episodes), size=min(count, len(self.episodes) - 1), replace=False).tolist())
        self._val_indices = [i for i, (ep, _) in enumerate(self._sample_refs) if ep in val_eps]
        self._indices = [i for i in range(len(self._sample_refs)) if i not in set(self._val_indices)]

    def _video_source(self, source: str) -> _VideoSource:
        if source not in self._video_sources:
            if self.image_cache_dir:
                cache_path = os.path.join(self.image_cache_dir, source.replace("/", "__") + ".npy")
                if os.path.isfile(cache_path):
                    self._video_sources[source] = _CachedVideoSource(cache_path, len(self.states), self.use_mmap)
                    return self._video_sources[source]
            paths = sorted(glob.glob(os.path.join(self.dataset_path, "videos", source, "chunk-*", "*.mp4")))
            if not paths:
                raise FileNotFoundError(f"no MP4 files for video feature {source}")
            self._video_sources[source] = _VideoSource(paths, len(self.states))
        return self._video_sources[source]

    def _read_image(self, key: str, indices: Sequence[int]) -> np.ndarray:
        source = self._image_sources[key]
        frames = self._video_source(source).read(indices)
        h, w = self.output_resolution
        result = np.empty((len(frames), 3, h, w), dtype=np.float32)
        for i, frame in enumerate(frames):
            input_h, input_w = frame.shape[:2]
            scale = max(w / input_w, h / input_h)
            resized_w = math.ceil(input_w * scale)
            resized_h = math.ceil(input_h * scale)
            image = Image.fromarray(frame).resize(
                (resized_w, resized_h), Image.Resampling.BILINEAR
            )
            left = (resized_w - w) // 2
            top = (resized_h - h) // 2
            image = image.crop((left, top, left + w, top + h))
            result[i] = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return result

    @staticmethod
    def _state_slice(key: str) -> slice:
        arm = 0 if key.startswith("robot0_") else 1 if key.startswith("robot1_") else -1
        if arm < 0:
            raise ValueError(f"unsupported low-dimensional key {key}")
        base = arm * 10
        if key.endswith("eef_pos"):
            return slice(base, base + 3)
        if key.endswith("eef_rot_axis_angle") or key.endswith("eef_rot_6d"):
            return slice(base + 3, base + 9)
        if key.endswith("gripper_width"):
            return slice(base + 9, base + 10)
        raise ValueError(
            f"unsupported low-dimensional key {key}; use robot{{0,1}}_eef_pos, "
            "robot{{0,1}}_eef_rot_axis_angle/eef_rot_6d, or robot{{0,1}}_gripper_width"
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set._indices = list(self._val_indices)
        val_set._video_sources = {}
        return val_set

    @staticmethod
    def _relative_position(position: np.ndarray, base_pose10d: np.ndarray) -> np.ndarray:
        """Express position vectors in the base pose's body frame."""
        position = np.asarray(position, dtype=np.float32)
        base_mat = _pose10d_to_mat(np.asarray(base_pose10d, dtype=np.float32))
        delta = position - base_mat[..., :3, 3]
        rotation = base_mat[..., :3, :3]
        while rotation.ndim < position.ndim + 1:
            rotation = np.expand_dims(rotation, axis=-3)
        return np.einsum("...j,...jk->...k", delta, rotation).astype(np.float32, copy=False)

    @staticmethod
    def _relative_rotation(rotation6d: np.ndarray, base_pose10d: np.ndarray) -> np.ndarray:
        """Express rotation-6D values in the base pose's body frame."""
        rotation6d = np.asarray(rotation6d, dtype=np.float32)
        zeros = np.zeros(rotation6d.shape[:-1] + (3,), dtype=np.float32)
        pose = np.concatenate([zeros, rotation6d], axis=-1)
        return _pose10d_to_relative(pose, np.asarray(base_pose10d, dtype=np.float32))[..., 3:]

    def _sample_plan(self, sample_ref_index: int):
        ep_idx, anchor = self._sample_refs[sample_ref_index]
        ep = self.episodes[ep_idx]
        obs_indices = {}
        for key, (horizon, ds, latency) in self._obs_specs.items():
            obs_indices[key] = [
                max(ep.start, min(ep.end - 1, anchor + latency - (horizon - 1 - j) * ds))
                for j in range(horizon)
            ]
        action_indices = [
            min(ep.end - 1, anchor + j * self._action_downsample)
            for j in range(self._action_horizon)
        ]
        # UMI uses the latest synchronized observation as the reference.  A
        # separate reference is retained per arm to support asymmetric sensor
        # latency, while normal bimanual data will use the same index.
        base_indices = []
        for arm in range(2):
            pos_key = f"robot{arm}_eef_pos"
            if pos_key in obs_indices:
                base_indices.append(obs_indices[pos_key][-1])
            else:
                base_indices.append(anchor)
        base_state = self.states[anchor].copy()
        for arm, base_index in enumerate(base_indices):
            start = arm * 10
            base_state[start:start + 10] = self.states[base_index, start:start + 10]
        return ep, obs_indices, action_indices, base_state

    def _sample_lowdim_action(self, sample_ref_index: int):
        _, obs_indices, action_indices, base_state = self._sample_plan(sample_ref_index)
        obs = {}
        for key in self.lowdim_keys:
            values = self.states[obs_indices[key], self._state_slice(key)].astype(np.float32)
            if self.observation_mode == "relative":
                arm = 0 if key.startswith("robot0_") else 1
                base_pose = base_state[arm * 10:arm * 10 + 9]
                if key.endswith("eef_pos"):
                    values = self._relative_position(values, base_pose)
                elif key.endswith("eef_rot_axis_angle") or key.endswith("eef_rot_6d"):
                    values = self._relative_rotation(values, base_pose)
            obs[key] = values

        action = self.actions[action_indices].astype(np.float32)
        if self.action_mode == "relative_trajectory":
            action = convert_action_to_relative(action, base_state)
        return obs, action

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # Relative poses depend on each sample's current EE base, so fit
        # statistics on the actual transformed training windows rather than
        # on the absolute source table.
        action_values = []
        lowdim_values = {key: [] for key in self.lowdim_keys}
        for sample_ref_index in self._indices:
            sample_obs, sample_action = self._sample_lowdim_action(sample_ref_index)
            action_values.append(sample_action)
            for key in self.lowdim_keys:
                lowdim_values[key].append(sample_obs[key])
        action_values = np.concatenate(action_values, axis=0)

        action_parts = []
        for arm in range(2):
            values = action_values[:, arm * 10:(arm + 1) * 10]
            action_parts.extend([
                get_range_normalizer_from_stat(array_to_stats(values[:, :3])),
                get_identity_normalizer_from_stat(array_to_stats(values[:, 3:9])),
                get_range_normalizer_from_stat(array_to_stats(values[:, 9:10])),
            ])
        normalizer["action"] = concatenate_normalizer(action_parts)
        for key in self.lowdim_keys:
            values = np.concatenate(lowdim_values[key], axis=0)
            stat = array_to_stats(values)
            if key.endswith("eef_rot_axis_angle") or key.endswith("eef_rot_6d"):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                normalizer[key] = get_range_normalizer_from_stat(stat)
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        if self.action_mode == "absolute_next_state":
            return torch.from_numpy(self.actions.copy())
        # Relative action depends on the sample's current EE reference.  As
        # with other replay-buffer datasets, return a flat collection of
        # action points; overlapping windows are intentionally retained.
        actions = [self._sample_lowdim_action(i)[1] for i in self._indices]
        return torch.from_numpy(np.concatenate(actions, axis=0).astype(np.float32, copy=False))

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_ref_index = self._indices[idx]
        _, obs_indices, _, _ = self._sample_plan(sample_ref_index)
        obs, action = self._sample_lowdim_action(sample_ref_index)
        for key in self.rgb_keys:
            obs[key] = self._read_image(key, obs_indices[key])
        return {
            "obs": {key: torch.from_numpy(value) for key, value in obs.items()},
            "action": torch.from_numpy(action),
        }
