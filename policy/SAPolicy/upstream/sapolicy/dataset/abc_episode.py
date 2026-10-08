#!/usr/bin/env python3
"""Direct-read PyTorch dataset for the ABC-130k official training release.

Reads the released per-episode format *as-is* (mentor requirement: no image
cache, no format conversion)::

    <data_root>/<split>/episode_<uuid>/
        states_actions.bin              float64 [T, 28] = state14 + action14
                                        14 = [L arm6, L grip, R arm6, R grip]
                                        (joints rad; gripper 0=closed..1=open;
                                        action = commanded, same layout)
        combined_camera-images-rgb.mp4  224x224 cameras stacked vertically
                                        (e.g. top/left/right -> 672x224), 30 fps,
                                        IDR every 30 frames, timebase 1/15360,
                                        integer PTS of 512 per frame
        episode_metadata.json           task_name, cameras, t0_ns,
                                        tick_ns=33333333, num_steps

plus optional TCP-label sidecars (produced by tcp_tools/annotate_tcp.py)::

    <tcp_labels_root>/<split>/<uuid>.npz
        frame_ts [T_lab] int64 (native top-camera frame timestamps, ns)
        tcp_pos [T_lab,2,3] / tcp_orn [T_lab,2,3,3] / tcp_pixel_coords
        [T_lab,2,3] / tcp_valid [T_lab,2]  (RealSense camera frame, 640x480 K)
        K [3,3], D [5], image_wh, extrinsics [4,4], needs_extrinsics

Sim episodes carry the same sidecar schema from tcp_tools/annotate_sim_tcp.py
(MJCF-exact top camera, K from fovy at the 224x168 render, D=0). Their
metadata has no ``t0_ns``/``tick_ns``; label row k is training step k, and
``_join_tcp_labels`` aligns them by identity in that case.

Episodes without a sidecar (or with ``needs_extrinsics=True``, e.g. ZED
stations) still yield samples: TCP keys are emitted as zeros with
``tcp_valid`` all 0 so the action branch trains and TCP losses are masked.

The sample contract mirrors ``RobomimicHDF5Dataset.__getitem__`` for
``dataset_type='cpgen'``/``num_arms=2`` (see that class for the reference
implementation the action/state/TCP construction below is copied from):

    {'observation': {'image': {cam: [To,H,W,3]->transforms},
                     'state': [To,20], 'tcp_pixel_coords': {cam: [To,2,3]},
                     'tcp_valid': {cam: [To,2]}, 'tcp_pos': {cam: [To,2,3]},
                     'tcp_orn': {cam: [To,2,9]},
                     'camera_intrinsics': {cam: K'}, ...
                     + with load_future_images / load_future_tcp (dynamics head):
                     'future_image': {cam: [1,H,W,3]}, 'future_image_valid': {cam: [1]},
                     'future_tcp_{pixel_coords,pos,orn,valid}': {cam: [1,2,*]},
                     'future_tcp_temporal_valid': {cam: [1]}  (slot = step t+T)},
     'action': [action_sequence_length, 20] float32,
     'prompt': str, 'episode_id': str, 'step_id': int, 'is_last_step': bool}

Key conversions (all copied from validated pipelines, not re-derived):

* 14D joints -> 20D EE actions: MuJoCo FK of the *commanded* joints through the
  official station MJCF (``{left,right}_grasp_site``), exactly the
  ``--action_source command`` path of scripts/yam/yam_mcap_to_robomimic.py.
  Absolute layout per arm [pos3, rot6d, grip1] (interleaved L,R -> 20);
  relative layout [poseL9, poseR9, gripL, gripR] with body-frame right-deltas,
  matching RobomimicHDF5Dataset._load_actions (num_arms=2 branch).
* State 20D: FK of the measured state joints, [pos3, rot6d, grip1] per arm,
  via the same `_quaternion_xyzw_to_rotation_6d` column-6d convention.
* npz->tick alignment: the release resamples the native camera stream onto a
  causal floor grid ticks[k] = t0_ns + (k+1)*tick_ns (see export_mcap.py), so
  label row of training step k is
  ``clip(searchsorted(frame_ts, ticks, 'right')-1, 0, len-1)``.
* Pixel labels: the released mp4 is the native 640x480 image passed through
  resize_with_pad to 224x224 (scale 0.35, 28 px letterbox top/bottom), so TCP
  pixels are *re-projected* from camera-frame ``tcp_pos`` with the padded
  intrinsics K' (fx'=fx*s, fy'=fy*s, cx'=cx*s+pad_w, cy'=cy*s+pad_h).
  Lens distortion D is ignored in this re-projection (sub-pixel to ~2 px
  approximation at image borders for the D405).

Optional frame cache (``frame_cache_dir``): ``<dir>/<split>/episode_<uuid>.npy``
uint8 ``[T, H_total, W, 3]`` written once by scripts/yam/teleop/build_frame_cache.py
(sequential decode of the same mp4, so pixels are identical) and memory-mapped
per worker; episodes without a cache file fall back to video decoding. Use it
for our own small teleop sets, where PyAV seek+decode (up to ~80 frames per
sample with obs history + a t+50 future frame) is the training bottleneck.

Video decoding: torchcodec ``VideoDecoder`` with a synthesized CFR frame map
(production path; no per-file probing) when available, otherwise a PyAV
keyframe-seek fallback with identical semantics (dev machines without
torchcodec). Select via ``video_backend='auto'|'torchcodec'|'pyav'``.
"""

import hashlib
import functools
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from sapolicy.logger import Log
from sapolicy.dataset.normalizer import LinearNormalizer
from sapolicy.embodiment_transforms import (
    quaternion_xyzw_to_rotation_6d as _quaternion_xyzw_to_rotation_6d,
)
from sapolicy.models.utils.rotation import axis_angle_to_matrix, matrix_to_rotation_6d


def _convert_actions(raw_actions, abs_action, action_orn_mode="6d"):
    """Absolute [pos3, rotvec3, grip1] x arms -> [pos3, col-6d, grip1] x arms (float32).

    Verbatim copy of the helper RobomimicHDF5Dataset used until 2026-09 (removed
    upstream in the transform refactor); kept here so ABC label/action math is
    unchanged. 14D dual-arm input -> 20D, 7D single-arm -> 10D.
    """
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True
        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        if action_orn_mode == "6d":
            rot_tensor = torch.from_numpy(rot).float().reshape(-1, 3)
            rot_mat = axis_angle_to_matrix(rot_tensor)
            rot_6d = matrix_to_rotation_6d(rot_mat)
            rot = rot_6d.numpy().reshape(rot.shape[:-1] + (6,))
        else:
            raise ValueError(f"Unsupported action_orn_mode: {action_orn_mode}")
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions

ROW_DIM = 28  # state14 + action14
ROW_BYTES = ROW_DIM * 8  # float64
PTS_PER_FRAME = 512  # 1/15360 timebase at 30 fps
GOP_FRAMES = 30  # IDR every 30 frames (official release); our exports may carry `gop_frames` in metadata
SIDES = ("left", "right")
DEFAULT_MJCF = "/home/jw/proj/workspace/models/yam_station/put_bottle.xml"


# --------------------------------------------------------------------------- FK


class _Station:
    """MuJoCo FK of the official YAM station MJCF (both arms in one call).

    Mirrors tcp_tools/annotate_tcp.py `Station` / yam_mcap_to_robomimic.py
    `FKSolver`: drive the 6 joints per side, read the `{side}_grasp_site`
    world pose. mj_kinematics is ~5 us/step, so each dataloader worker keeps
    one instance and runs FK online per sample.
    """

    def __init__(self, mjcf_path: str = DEFAULT_MJCF):
        import mujoco  # deferred: only FK needs it

        self._mj = mujoco
        try:
            self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        except ValueError as e:
            if "keyframe" not in str(e):
                raise
            # Older/newer mujoco versions disagree on the scene keyframe's
            # qpos length (free-joint objects). FK only needs the kinematic
            # tree, so strip <keyframe> and load a sibling copy (same dir so
            # relative mesh paths still resolve).
            import re
            import tempfile
            xml = open(mjcf_path).read()
            xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
            with tempfile.NamedTemporaryFile(
                "w", suffix=".xml", dir=os.path.dirname(mjcf_path), delete=False
            ) as f:
                f.write(xml)
                tmp = f.name
            try:
                self.model = mujoco.MjModel.from_xml_path(tmp)
            finally:
                os.unlink(tmp)
        self.data = mujoco.MjData(self.model)
        self.qadr = {
            s: [self.model.joint(f"{s}_joint{i}").qposadr[0] for i in range(1, 7)]
            for s in SIDES
        }
        self.site = {s: self.model.site(f"{s}_grasp_site").id for s in SIDES}

    def fk(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """joints [T,2,6] -> world grasp_site pos [T,2,3], rot [T,2,3,3]."""
        T = joints.shape[0]
        pos = np.empty((T, 2, 3))
        rot = np.empty((T, 2, 3, 3))
        for k in range(T):
            for j, s in enumerate(SIDES):
                self.data.qpos[self.qadr[s]] = joints[k, j]
            self._mj.mj_kinematics(self.model, self.data)
            for j, s in enumerate(SIDES):
                pos[k, j] = self.data.site_xpos[self.site[s]]
                rot[k, j] = self.data.site_xmat[self.site[s]].reshape(3, 3)
        return pos, rot


# ------------------------------------------------------------------- video I/O


def _probe_torchcodec() -> bool:
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _torchcodec_has_frame_mappings() -> bool:
    """torchcodec >= 0.7 accepts VideoDecoder(custom_frame_mappings=...); 0.6 does not."""
    try:
        import inspect
        from torchcodec.decoders import VideoDecoder
        return "custom_frame_mappings" in inspect.signature(VideoDecoder.__init__).parameters
    except Exception:
        return False


@functools.lru_cache(maxsize=256)
def _frame_mapping_json(num_frames: int, gop: int) -> str:
    """Synthesized CFR frame map for torchcodec (pts = 512*k, IDR every `gop`), cached per
    (num_frames, gop): building ~3000 dicts + json.dumps per sample was measurable loader cost."""
    frames = [
        {"pts": PTS_PER_FRAME * i, "duration": PTS_PER_FRAME, "key_frame": 1 if i % gop == 0 else 0}
        for i in range(num_frames)
    ]
    return json.dumps({"frames": frames})


def _decode_frames_torchcodec(path: str, indices: Sequence[int], num_frames: int,
                              gop: int = GOP_FRAMES):
    """Production path: torchcodec VideoDecoder with a synthesized CFR frame
    map (pts = 512*k, keyframe every 30) -- no per-file probing. Copied from
    the validated abc_minimal/train_loop.py `decode_frame`.

    Returns {idx: HWC uint8 ndarray of the full stacked frame}.
    """
    from torchcodec.decoders import VideoDecoder

    if _torchcodec_has_frame_mappings():
        decoder = VideoDecoder(path, custom_frame_mappings=_frame_mapping_json(int(num_frames), int(gop)))
    else:
        # torchcodec < 0.7 (e.g. 0.6 on the NGC torch 2.8 images): no custom_frame_mappings. Approximate seek
        # trusts the container index, which is exact for our strict-CFR exports (pts = 512*k, checked frame-identical
        # to the PyAV path on GOP-30 and GOP-10 files, 2026-09-06) and is ~2-3x faster than the PyAV fallback.
        decoder = VideoDecoder(path, seek_mode="approximate", num_ffmpeg_threads=1)
    out = {}
    for idx in sorted(set(int(i) for i in indices)):
        frame = decoder[idx]  # (C, H_total, W) uint8
        out[idx] = frame.permute(1, 2, 0).contiguous().numpy()
    return out


def _decode_frames_pyav(path: str, indices: Sequence[int]):
    """Fallback for machines without torchcodec: PyAV keyframe seek + forward
    decode. Same frame indexing (pts = 512*idx); slower but exact. Production
    should use the torchcodec path above.

    Returns {idx: HWC uint8 ndarray of the full stacked frame}.
    """
    import av

    out = {}
    wanted = sorted(set(int(i) for i in indices))
    with av.open(path) as container:
        stream = container.streams.video[0]
        # Single-threaded decode: ffmpeg thread pools x forked DataLoader workers
        # oversubscribe the CPU quota and can deadlock after fork; 224x504 frames
        # decode cheaply without threads.
        stream.thread_type = "NONE"
        stream.thread_count = 1
        frame_iter = None
        last_pts = None
        for idx in wanted:
            target = PTS_PER_FRAME * idx
            # Re-seek when moving backwards or skipping far ahead (> 2 GOPs).
            if (
                frame_iter is None
                or last_pts is None
                or target < last_pts
                or target - last_pts > 2 * GOP_FRAMES * PTS_PER_FRAME
            ):
                container.seek(target, stream=stream, any_frame=False, backward=True)
                frame_iter = container.decode(stream)
                last_pts = None
            got = None
            for frame in frame_iter:
                if frame.pts is None:
                    continue
                last_pts = frame.pts
                if frame.pts >= target:
                    got = frame
                    break
            if got is None:
                raise RuntimeError(f"{path}: could not decode frame {idx}")
            out[idx] = got.to_ndarray(format="rgb24")
    return out


def _slice_camera(frame_hwc: np.ndarray, source_cameras: Sequence[str], cam: str,
                  ep_uuid: str) -> np.ndarray:
    """Cut one camera band out of the vertically stacked frame.

    Stereo-top episodes (cameras like top_left/top_right, ZED stations) alias
    one eye to `top` deterministically by uuid hash, matching the production
    export (train_loop.py `decode_frame`).
    """
    n = len(source_cameras)
    h = frame_hwc.shape[0] // n
    if cam in source_cameras:
        i = source_cameras.index(cam)
        return frame_hwc[i * h:(i + 1) * h]
    if cam == "top" and "top_left" in source_cameras and "top_right" in source_cameras:
        digest = hashlib.sha1(ep_uuid.encode("utf-8")).digest()[0]
        pick = "top_left" if digest % 2 == 0 else "top_right"
        i = source_cameras.index(pick)
        return frame_hwc[i * h:(i + 1) * h]
    raise KeyError(f"camera {cam!r} not in episode cameras {source_cameras}")


# ---------------------------------------------------------------- geometry


def _resize_with_pad_intrinsics(K: np.ndarray, src_wh: Sequence[int],
                                dst_wh) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Adjust K for resize_with_pad(src -> dst_w x dst_h).

    Copied math from abc_minimal/preprocess.py `resize_with_pad`, generalized
    to a rectangular destination: uniform scale s = 1/max(w/dst_w, h/dst_h),
    then centered zero padding per axis. Official release: dst 224x224
    (640x480 -> s=0.35, content band v in [28, 196)). Our own exports use
    224x168, where 640x480 fills the frame exactly (no letterbox).

    Returns (K', (pad_w0, pad_h0, new_w, new_h)).
    """
    if isinstance(dst_wh, (int, np.integer)):
        dst_w = dst_h = int(dst_wh)
    else:
        dst_w, dst_h = int(dst_wh[0]), int(dst_wh[1])
    w, h = int(src_wh[0]), int(src_wh[1])
    ratio = max(w / dst_w, h / dst_h)
    new_w = max(1, int(round(w / ratio)))
    new_h = max(1, int(round(h / ratio)))
    pad_w0 = (dst_w - new_w) // 2
    pad_h0 = (dst_h - new_h) // 2
    Kp = np.asarray(K, dtype=np.float64).copy()
    Kp[0, 0] /= ratio
    Kp[1, 1] /= ratio
    Kp[0, 2] = Kp[0, 2] / ratio + pad_w0
    Kp[1, 2] = Kp[1, 2] / ratio + pad_h0
    return Kp.astype(np.float32), (pad_w0, pad_h0, new_w, new_h)


def _project_pinhole(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame points [...,3] -> pixel (u,v,z) [...,3] with pinhole K.

    Distortion D is intentionally ignored (documented approximation): the
    released 224x224 mp4 was produced by resizing the (distorted) native
    image, so re-projecting undistorted rays with the padded K' is exact up
    to the lens distortion residual (<~2 px for the D405 at image borders).
    """
    z = pts_cam[..., 2]
    z_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * pts_cam[..., 0] / z_safe + K[0, 2]
    v = K[1, 1] * pts_cam[..., 1] / z_safe + K[1, 2]
    return np.stack([u, v, z], axis=-1)


def _rotmat_to_col6d(rot: np.ndarray) -> np.ndarray:
    """[..., 3, 3] -> [..., 6] using the repo's column-6d convention
    (rotmat[..., :, :2].swapaxes(-1, -2), i.e. [col1, col2])."""
    return rot[..., :, :2].swapaxes(-1, -2).reshape(*rot.shape[:-2], 6)


def _relative_action_windows(
    p_cur: np.ndarray,
    R_cur: np.ndarray,
    p_win: np.ndarray,
    R_win: np.ndarray,
    g_win: np.ndarray,
    body_frame: bool,
) -> np.ndarray:
    """Vectorized copy of RobomimicHDF5Dataset._load_actions's bimanual
    relative-action construction (`convert_actions_to_relative` per arm, then
    regroup [poseL9, poseR9, gripL, gripR]).

    Args:
        p_cur [T,2,3], R_cur [T,2,3,3]: current EE pose (from *state* FK).
        p_win [T,H,2,3], R_win [T,H,2,3,3], g_win [T,H,2]: absolute action
            windows (already padded to H by repeating the last row).
        body_frame: True -> rel = R_cur^{-1}(p_t - p_cur), R_cur^{-1} R_t;
            False -> world-frame deltas (p_t - p_cur, R_t R_cur^{-1}).
    Returns:
        [T,H,20] float32.
    """
    diff = p_win - p_cur[:, None]  # [T,H,2,3]
    if body_frame:
        rel_pos = np.einsum("taki,thak->thai", R_cur, diff)
        rel_rot = np.einsum("taki,thakj->thaij", R_cur, R_win)
    else:
        rel_pos = diff
        rel_rot = np.einsum("thail,tajl->thaij", R_win, R_cur)
    rel6 = _rotmat_to_col6d(rel_rot)  # [T,H,2,6]
    out = np.concatenate(
        [
            rel_pos[:, :, 0], rel6[:, :, 0],  # left  [pos3, rot6d]
            rel_pos[:, :, 1], rel6[:, :, 1],  # right [pos3, rot6d]
            g_win[:, :, 0:1], g_win[:, :, 1:2],  # grippers grouped last
        ],
        axis=-1,
    )
    return out.astype(np.float32)


def _abs14_from_fk(pos: np.ndarray, rot: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """FK pose -> the converter's absolute action rows
    [L pos3, L rotvec3, L grip, R pos3, R rotvec3, R grip] (T,14), exactly
    yam_mcap_to_robomimic.py `arm_actions` with --action_source command."""
    T = pos.shape[0]
    rv = R.from_matrix(rot.reshape(-1, 3, 3)).as_rotvec().reshape(T, 2, 3)
    blocks = []
    for a in range(2):
        blocks += [pos[:, a], rv[:, a], grip[:, a:a + 1]]
    return np.concatenate(blocks, axis=-1)


# ------------------------------------------------------------------- dataset


class AbcEpisodeDataset(Dataset):
    """Direct-read dataset over the ABC-130k official training release.

    __getitem__ output matches RobomimicHDF5Dataset (cpgen, num_arms=2); see
    module docstring. Heavy per-sample work (video decode, MuJoCo FK) happens
    lazily in the dataloader worker; init only scans lightweight metadata
    (bin sizes, episode json, npz frame_ts for tick alignment) into memory --
    nothing is written to disk.
    """

    def __init__(
        self,
        data_root: str,
        tcp_labels_root: Optional[str] = None,
        dataset_name: str = "AbcEpisodes",
        dataset_type: str = "cpgen",  # kept for parity; controls cpgen-style depth norm
        split: str = "train",
        use_train_val_split: bool = False,
        train_val_split_ratio: float = 0.9,
        split_seed: int = 42,
        observation_keys: Optional[List[str]] = None,
        obs_hist_length: int = 1,
        action_sequence_length: int = 16,
        normalize_actions: bool = True,
        use_task_description: bool = True,
        task_description: Optional[str] = None,
        transforms: Optional[List] = None,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
        use_agentview: bool = True,
        use_eye_in_hand: bool = False,
        use_depth: bool = False,
        use_state: bool = True,
        max_episodes: Optional[int] = None,
        camera_names: Optional[List[str]] = None,
        use_relative_actions: bool = False,
        action_orn_mode: str = "6d",
        norm_type: str = "percentile_0.02_0.98",
        cpgen_absolute_actions: bool = True,  # ABC actions are absolute EE poses
        body_frame_actions: bool = True,
        camera_pair_choices: Optional[Dict[str, Any]] = None,
        num_arms: int = 2,
        arm_obs_prefixes: Optional[List[str]] = None,
        tcp_orn_already_aligned: bool = True,  # FK site frame == action/state frame
        tcp_arm: str = "left",  # robot0_* alias arm (converter default)
        mjcf_path: str = DEFAULT_MJCF,
        video_backend: str = "auto",  # 'auto' | 'torchcodec' | 'pyav'
        stats_num_episodes: int = 100,  # parity with parent's ~100-episode sample
        label_cache_episodes: int = 32,  # per-worker LRU of decoded npz labels
        image_size: int = 224,  # released frame size (per camera band)
        normalizer_load_path: Optional[str] = None,  # reuse another dataset's fitted stats (e.g. sim <- real)
        crop_rows: Optional[Sequence[int]] = None,  # e.g. (28, 196): drop the letterbox rows of the official 224x224 release -> 224x168
        normalizer_save_path: Optional[str] = None,  # persist fitted stats for other datasets
        load_future_images: bool = False,  # future_image[cam] at t+T for the dynamics head (semantic)
        future_image_horizons: Optional[List[int]] = None,  # must be [action_sequence_length]
        load_future_tcp: bool = False,  # future_tcp_* at t+T for the dynamics head (predict_tcp)
        future_tcp_horizons: Optional[List[int]] = None,  # must be [action_sequence_length]
        frame_cache_dir: Optional[str] = None,  # <dir>/<split>/episode_<uuid>.npy uint8 [T,H_total,W,3] from scripts/yam/teleop/build_frame_cache.py; skips video decode
        tcp_label_camera: str = "top",  # sidecar TCP is expressed in THIS camera; other backings get tcp_valid=0
        require_tcp_labels: bool = False,  # drop episodes with no usable TCP sidecar
        **kwargs,  # swallow harness extras (check_mask, ...)
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.tcp_labels_root = Path(tcp_labels_root) if tcp_labels_root else None
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type.lower()
        self.split = split
        self.use_train_val_split = bool(use_train_val_split)
        self.train_val_split_ratio = float(train_val_split_ratio)
        self.split_seed = int(split_seed)
        self.observation_keys = list(observation_keys) if observation_keys else ["image"]
        self.obs_hist_length = int(obs_hist_length)
        self.action_sequence_length = int(action_sequence_length)
        self.normalize_actions = bool(normalize_actions)
        self.use_task_description = bool(use_task_description)
        self.task_description = task_description
        self.transforms = [] if transforms is None else list(transforms)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.use_agentview = bool(use_agentview)
        self.use_eye_in_hand = bool(use_eye_in_hand)
        self.use_depth = bool(use_depth)
        self.use_state = bool(use_state)
        self.max_episodes = max_episodes
        self.use_relative_actions = bool(use_relative_actions)
        self.action_orn_mode = action_orn_mode
        self.norm_type = norm_type
        assert self.norm_type in ["mean_std", "percentile_0.02_0.98", "minmax"], (
            f"Invalid norm type: {self.norm_type}"
        )
        self.cpgen_absolute_actions = bool(cpgen_absolute_actions)
        self.body_frame_actions = bool(body_frame_actions)
        self.num_arms = int(num_arms)
        assert self.num_arms == 2, "ABC episodes are bimanual (num_arms=2)"
        self.arm_obs_prefixes = list(arm_obs_prefixes) if arm_obs_prefixes else list(SIDES)
        self.tcp_orn_already_aligned = bool(tcp_orn_already_aligned)
        self.tcp_arm = tcp_arm
        self.mjcf_path = mjcf_path
        self.stats_num_episodes = int(stats_num_episodes)
        self.label_cache_episodes = int(label_cache_episodes)
        self.image_size = int(image_size)
        self.normalizer_load_path = normalizer_load_path
        self.crop_rows = tuple(int(v) for v in crop_rows) if crop_rows else None
        self.normalizer_save_path = normalizer_save_path
        # Dynamics-head targets (parity with RobomimicHDF5Dataset): one future slot
        # at t+T, T = action chunk size; keys are [K=1, ...] per camera.
        self.load_future_images = bool(load_future_images)
        self.load_future_tcp = bool(load_future_tcp)
        T = int(action_sequence_length)
        for name, on, hs in (("future_image_horizons", self.load_future_images, future_image_horizons),
                             ("future_tcp_horizons", self.load_future_tcp, future_tcp_horizons)):
            if on and tuple(int(h) for h in (hs or [T])) != (T,):
                raise ValueError(f"{name} must be [{T}] (= action_sequence_length), got {hs}")
        self.future_horizon = T
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self._frame_mmaps: Dict[int, np.ndarray] = {}  # per-worker open memmaps
        if self.use_relative_actions:
            assert self.action_orn_mode == "6d", (
                "only 6d representation is supported for relative actions"
            )

        # -- video backend ----------------------------------------------------
        if video_backend == "auto":
            self.video_backend = "torchcodec" if _probe_torchcodec() else "pyav"
        else:
            self.video_backend = video_backend
        if self.video_backend == "torchcodec" and not _probe_torchcodec():
            raise ImportError("video_backend='torchcodec' but torchcodec is not importable")
        Log.info(f"[{dataset_name}] video backend: {self.video_backend}")

        # -- camera selection (parity with RobomimicHDF5Dataset) -------------
        self.camera_pair_choices = None
        self._pair_canonical_names: Optional[List[str]] = None
        self._pair_choices: Optional[List[List[str]]] = None
        self._n_pairs = 1
        if camera_pair_choices is not None:
            self.camera_pair_choices = camera_pair_choices
            canonical = list(camera_pair_choices["canonical_names"])
            choices = [list(pair) for pair in camera_pair_choices["choices"]]
            arity = len(canonical)
            if arity < 1:
                raise ValueError("canonical_names must be non-empty")
            for i, pair in enumerate(choices):
                if len(pair) != arity:
                    raise ValueError(
                        f"camera_pair_choices choice[{i}]={pair} has {len(pair)} "
                        f"cameras, expected {arity}"
                    )
            self._pair_canonical_names = canonical
            self._pair_choices = choices
            self.camera_names = canonical
            self._all_camera_names_pairs = sorted({c for pair in choices for c in pair})
            if camera_pair_choices.get("enumerate_all", True):
                self._n_pairs = len(choices)
            Log.info(f"[coupled] camera_pair_choices: canonical={canonical}, "
                     f"choices={choices}, n_pairs={self._n_pairs}")
        elif camera_names:
            self.camera_names = [camera_names] if isinstance(camera_names, str) else list(camera_names)
            self._all_camera_names_pairs = list(self.camera_names)
        else:
            self.camera_names = ["top"]
            self._all_camera_names_pairs = ["top"]
        Log.info(f"Using camera(s): {self.camera_names}")

        # -- episode scan (lightweight; kept in memory, never written) --------
        self._scan_episodes()

        # -- TCP label join + tick-alignment precompute -----------------------
        self._join_tcp_labels()
        self.tcp_label_camera = str(tcp_label_camera)
        self.require_tcp_labels = bool(require_tcp_labels) if not isinstance(require_tcp_labels, str) \
            else require_tcp_labels.strip().lower() in ("1", "true", "yes", "y", "t")
        if self.require_tcp_labels:
            # Episodes without a usable sidecar are emitted with zero position +
            # identity orientation and tcp_valid=0. The loss now masks them, but for a
            # clean TCP-supervision study drop them from the index entirely.
            keep = [ep for ep in self.episodes if ep.get("label") is not None]
            dropped = len(self.episodes) - len(keep)
            if not keep:
                raise ValueError(
                    f"[{self.dataset_name}] require_tcp_labels=true left 0 episodes "
                    f"(of {len(self.episodes)}): check tcp_labels_root")
            if dropped:
                steps_before = int(self.cum_steps[-1])
                self.episodes = keep
                self.cum_steps = np.cumsum(
                    np.asarray([int(ep["num_steps"]) for ep in self.episodes], dtype=np.int64))
                Log.info(
                    f"[{self.dataset_name}] require_tcp_labels: dropped {dropped} unlabeled "
                    f"episodes, {steps_before - int(self.cum_steps[-1])} steps "
                    f"({100.0 * (steps_before - int(self.cum_steps[-1])) / max(steps_before, 1):.1f}%)")
        self._label_cache: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()

        # -- FK station: lazily constructed per worker process ----------------
        self._station: Optional[_Station] = None

        self.build_transforms()

        # -- action/state normalization stats: fitted (or loaded) here so train_net
        #    can bind them to the model; NOT applied to samples (see __getitem__) --
        self.normalizer = None
        if self.normalize_actions:
            lp = self.normalizer_load_path
            if lp and os.path.exists(lp):
                # Shared stats across datasets of one ConcatDataset (real+sim):
                # every member must normalize with the same limits.
                self.normalizer = LinearNormalizer()
                self.normalizer.load_state_dict(torch.load(lp, map_location="cpu"))
                Log.info(f"[{self.dataset_name}] loaded normalizer stats from {lp}")
            else:
                if lp:
                    Log.warn(f"[{self.dataset_name}] normalizer_load_path {lp} missing; fitting locally")
                self._compute_action_stats()
                if self.normalizer_save_path:
                    os.makedirs(os.path.dirname(self.normalizer_save_path), exist_ok=True)
                    torch.save(self.normalizer.state_dict(), self.normalizer_save_path)
                    Log.info(f"[{self.dataset_name}] saved normalizer stats to {self.normalizer_save_path}")

        Log.info(
            f"Initialized {self.dataset_name} split={self.split}: "
            f"{len(self.episodes)} episodes, {int(self.cum_steps[-1])} steps, "
            f"{self._n_labeled} with TCP labels"
        )

    # ------------------------------------------------------------------ setup

    def _scan_episodes(self):
        ep_root = self.data_root / self.split
        if not ep_root.exists():
            raise FileNotFoundError(f"episode root not found: {ep_root}")
        dirs = sorted(d for d in ep_root.iterdir()
                      if d.is_dir() and d.name.startswith("episode_"))
        if self.max_episodes:
            dirs = dirs[: self.max_episodes]
        if self.use_train_val_split:
            # Parity with RobomimicHDF5Dataset: permutation split of the scanned
            # list (normally unused here -- the release already ships train/val).
            rng = np.random.RandomState(self.split_seed)
            n_train = int(len(dirs) * self.train_val_split_ratio)
            order = rng.permutation(len(dirs))
            sel = order[:n_train] if self.split == "train" else order[n_train:]
            dirs = [dirs[i] for i in sorted(sel)]

        self.episodes: List[Dict[str, Any]] = []
        lengths = []
        for d in dirs:
            bin_path = d / "states_actions.bin"
            if not bin_path.exists():
                continue
            T = bin_path.stat().st_size // ROW_BYTES
            if T <= 0:
                continue
            meta = {}
            meta_path = d / "episode_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            if meta.get("num_steps") not in (None, T):
                Log.warn(f"{d.name}: bin has {T} rows but metadata says "
                         f"{meta.get('num_steps')}; using the smaller of the two")
                T = min(T, int(meta["num_steps"]))
            self.episodes.append({
                "dir": d,
                "uuid": d.name[len("episode_"):],
                "num_steps": int(T),
                "cameras": tuple(meta.get("cameras") or ("top", "left", "right")),
                # Per-camera released frame size from metadata (falls back to
                # the square legacy default). Official release: [224,224];
                # our own 4:3-native exports: [224,168].
                "frame_wh_raw": tuple(
                    (meta.get("camera_resolutions") or {}).get("top")
                    or (self.image_size, self.image_size)
                ),
                "task_name": meta.get("task_name", ""),
                "frame_wh": None,  # filled below (crop-aware)
                "t0_ns": meta.get("t0_ns"),
                "tick_ns": meta.get("tick_ns"),
                # IDR interval of combined.mp4 (export_mcap.py ABC_GOP); official release = 30.
                "gop_frames": int(meta.get("gop_frames") or GOP_FRAMES),
                "label": None,  # filled by _join_tcp_labels
            })
            lengths.append(T)
        if not self.episodes:
            raise ValueError(f"no episodes found under {ep_root}")
        n_cached = 0
        for ep in self.episodes:
            cache = (self.frame_cache_dir / self.split / f"{ep['dir'].name}.npy") if self.frame_cache_dir else None
            ep["frame_cache"] = str(cache) if (cache is not None and cache.exists()) else None
            n_cached += ep["frame_cache"] is not None
        if self.frame_cache_dir:
            Log.info(f"[{self.dataset_name}] frame cache {self.frame_cache_dir}: "
                     f"{n_cached}/{len(self.episodes)} episodes cached (others decode video)")
        for ep in self.episodes:
            fw, fh = ep["frame_wh_raw"]
            if self.crop_rows and 0 <= self.crop_rows[0] < self.crop_rows[1] <= fh:
                ep["crop_rows"] = self.crop_rows
                ep["frame_wh"] = (fw, self.crop_rows[1] - self.crop_rows[0])
            else:
                ep["crop_rows"] = None
                ep["frame_wh"] = (fw, fh)
        self.cum_steps = np.cumsum(np.asarray(lengths, dtype=np.int64))

    def _join_tcp_labels(self):
        """uuid -> npz join + per-episode tick-alignment index (int32 [T]).

        Only tiny arrays (frame_ts, K, extrinsics) are decompressed here;
        the bulky TCP arrays stay on disk until a worker touches the episode.
        """
        self._n_labeled = 0
        fallback_K224 = None
        fallback_ext = None
        lab_dir = (self.tcp_labels_root / self.split) if self.tcp_labels_root else None
        for ep in self.episodes:
            if lab_dir is None:
                continue
            npz_path = lab_dir / f"{ep['uuid']}.npz"
            if not npz_path.exists():
                continue
            try:
                with np.load(npz_path) as z:
                    files = set(z.files)
                    has_cam = (
                        not bool(z["needs_extrinsics"])
                        and {"tcp_pos", "tcp_orn", "tcp_pixel_coords", "tcp_valid",
                             "K", "image_wh"}.issubset(files)
                    )
                    if not has_cam:
                        continue  # world-frame-only label (e.g. ZED): treat as unlabeled
                    frame_ts = z["frame_ts"]
                    K = np.asarray(z["K"], dtype=np.float64)
                    image_wh = np.asarray(z["image_wh"], dtype=np.int64)
                    ext = np.asarray(z["extrinsics"], dtype=np.float32) if "extrinsics" in files else None
            except Exception as e:
                Log.warn(f"failed to read TCP label {npz_path}: {e}")
                continue
            T = ep["num_steps"]
            if ep["t0_ns"] is None or ep["tick_ns"] is None:
                # No clock in the metadata: only tick-exact labels can be used,
                # i.e. sim exports (tcp_tools/annotate_sim_tcp.py) where label
                # row k *is* training step k. Anything shorter is unusable.
                if len(frame_ts) < T:
                    Log.warn(f"{ep['dir'].name}: metadata lacks t0_ns/tick_ns and the "
                             f"label has {len(frame_ts)} rows < {T} steps; skipping label")
                    continue
                idx = np.arange(T, dtype=np.int32)
            else:
                # Causal floor grid: training step k shows the native frame at
                # ticks[k] = t0 + (k+1)*tick (export_mcap.py convention).
                ticks = ep["t0_ns"] + np.arange(1, T + 1, dtype=np.int64) * int(ep["tick_ns"])
                idx = np.clip(
                    np.searchsorted(frame_ts, ticks, side="right") - 1, 0, len(frame_ts) - 1
                ).astype(np.int32)
            K224, pad_info = _resize_with_pad_intrinsics(
                K, image_wh, ep.get("frame_wh_raw") or self.image_size)
            if ep.get("crop_rows"):
                # Letterbox rows removed from the frame: shift the principal point
                # and the content band accordingly (band is clamped to the crop).
                top, bot = ep["crop_rows"]
                pad_w0, pad_h0, new_w, new_h = pad_info
                band_top = max(pad_h0 - top, 0)
                band_bot = min(pad_h0 + new_h - top, bot - top)
                K224 = K224.copy(); K224[1, 2] -= top
                pad_info = (pad_w0, band_top, new_w, max(band_bot - band_top, 1))
            ep["label"] = {
                "path": str(npz_path),
                "tick_idx": idx,
                "K224": K224,
                "pad_info": pad_info,  # (pad_w0, pad_h0, new_w, new_h)
                "extrinsics": ext,
            }
            self._n_labeled += 1
            if fallback_K224 is None:
                fallback_K224 = K224
                fallback_ext = ext
        # Fallback intrinsics for unlabeled episodes: needed only so that every
        # sample carries a `camera_intrinsics` entry (default_collate requires
        # consistent keys); TCP supervision of those samples is masked by
        # tcp_valid == 0. Nominal K when no labels exist at all.
        if fallback_K224 is None:
            fw, fh = (self.episodes[0].get("frame_wh")
                      if self.episodes else (self.image_size, self.image_size))
            f = float(max(fw, fh))
            fallback_K224 = np.array(
                [[f, 0, fw / 2], [0, f, fh / 2], [0, 0, 1]], dtype=np.float32
            )
        if fallback_ext is None:
            fallback_ext = np.eye(4, dtype=np.float32)
        self._fallback_K224 = fallback_K224
        self._fallback_ext = fallback_ext

    def build_transforms(self):
        # Photocopied from RobomimicHDF5Dataset.build_transforms.
        transforms = self.transforms
        if len(transforms) == 0:
            self.transforms = lambda x: x
            return
        log_str = f"{self.dataset_name} transform layers: \n"
        for idx, transform in enumerate(transforms):
            log_str += (
                (str(transform) + "\n") if idx != len(transforms) - 1 else str(transform)
            )
        Log.info(log_str)
        self.transforms = Compose(transforms)

    # ------------------------------------------------------------- FK helpers

    def _get_station(self) -> _Station:
        if self._station is None:
            self._station = _Station(self.mjcf_path)
        return self._station

    @staticmethod
    def _split_rows(rows: np.ndarray) -> Dict[str, np.ndarray]:
        """[N,28] float64 -> joints/grippers of state and action halves."""
        return {
            "state_joints": np.stack([rows[:, 0:6], rows[:, 7:13]], axis=1),   # [N,2,6]
            "state_grip": rows[:, [6, 13]],                                    # [N,2]
            "action_joints": np.stack([rows[:, 14:20], rows[:, 21:27]], axis=1),
            "action_grip": rows[:, [20, 27]],
        }

    def _read_rows(self, ep: Dict[str, Any], start: int, end: int) -> np.ndarray:
        with open(ep["dir"] / "states_actions.bin", "rb") as f:
            f.seek(start * ROW_BYTES)
            raw = f.read((end - start) * ROW_BYTES)
        return np.frombuffer(raw, dtype=np.float64).reshape(-1, ROW_DIM)

    # ------------------------------------------------------------- TCP labels

    def _get_episode_labels(self, ep_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Tick-aligned TCP label arrays for one episode (worker-local LRU).

        Arrays are pre-gathered onto the training tick grid, with pixel
        coordinates re-projected into the released 224x224 frame via K'.
        """
        ep = self.episodes[ep_idx]
        lab = ep["label"]
        if lab is None:
            return None
        cached = self._label_cache.get(ep_idx)
        if cached is not None:
            self._label_cache.move_to_end(ep_idx)
            return cached
        with np.load(lab["path"]) as z:
            idx = lab["tick_idx"]
            tcp_pos = np.asarray(z["tcp_pos"], dtype=np.float32)[idx]      # [T,2,3] cam frame
            tcp_orn = np.asarray(z["tcp_orn"], dtype=np.float32)[idx]      # [T,2,3,3]
            src_valid = np.asarray(z["tcp_valid"], dtype=bool)[idx]        # [T,2]
        K224 = lab["K224"]
        pad_w0, pad_h0, new_w, new_h = lab["pad_info"]
        uvz = _project_pinhole(tcp_pos.astype(np.float64), K224.astype(np.float64))
        u, v, zc = uvz[..., 0], uvz[..., 1], uvz[..., 2]
        # Valid = source-frame validity AND in the letterboxed content band of
        # the 224x224 frame (padding rows can never contain the TCP).
        in_band = (
            (zc > 0)
            & (u >= pad_w0) & (u < pad_w0 + new_w)
            & (v >= pad_h0) & (v < pad_h0 + new_h)
        )
        entry = {
            "tcp_pos": tcp_pos,
            "tcp_orn": tcp_orn.reshape(tcp_orn.shape[0], 2, 9),  # model expects flat 9
            "tcp_pixel_coords": uvz.astype(np.float32),
            "tcp_valid": (src_valid & in_band).astype(np.float32),
        }
        self._label_cache[ep_idx] = entry
        while len(self._label_cache) > self.label_cache_episodes:
            self._label_cache.popitem(last=False)
        return entry

    # ------------------------------------------------------------ action math

    def _episode_abs_and_state(self, rows: np.ndarray):
        """FK both halves of [N,28] rows -> dict of poses (float64)."""
        parts = self._split_rows(rows)
        station = self._get_station()
        s_pos, s_rot = station.fk(parts["state_joints"])
        a_pos, a_rot = station.fk(parts["action_joints"])
        return {
            "s_pos": s_pos, "s_rot": s_rot, "s_grip": parts["state_grip"],
            "s_joints": parts["state_joints"],
            "a_pos": a_pos, "a_rot": a_rot, "a_grip": parts["action_grip"],
        }

    def _absolute_actions20(self, a_pos, a_rot, a_grip) -> np.ndarray:
        """Absolute 20D via the shared `_convert_actions` (rotvec -> col-6d),
        interleaved per arm [L pos3+rot6d+grip, R pos3+rot6d+grip]."""
        abs14 = _abs14_from_fk(a_pos, a_rot, a_grip)
        return _convert_actions(abs14, True, self.action_orn_mode)

    @staticmethod
    def _pad_window(arr: np.ndarray, H: int) -> np.ndarray:
        """Repeat-last padding along axis 0 to length H (parent convention)."""
        if arr.shape[0] >= H:
            return arr[:H]
        pad = np.repeat(arr[-1:], H - arr.shape[0], axis=0)
        return np.concatenate([arr, pad], axis=0)

    # ------------------------------------------------------------ norm stats

    def _compute_action_stats(self):
        """Photocopy of RobomimicHDF5Dataset._compute_action_stats semantics:
        sample ~stats_num_episodes episodes, build the exact action tensors the
        loader emits ([T,20] absolute or [T,H,20] relative windows) plus the
        20D state, then LinearNormalizer.fit(mode='limits', last_n_dims=1,
        horizon_stats=use_relative_actions)."""
        Log.info("Computing action statistics...")
        n_eps = len(self.episodes)
        stride = max(1, n_eps // self.stats_num_episodes)
        sample_idxs = list(range(0, n_eps, stride))
        H = self.action_sequence_length
        all_actions, all_states = [], []
        for i in sample_idxs:
            ep = self.episodes[i]
            rows = self._read_rows(ep, 0, ep["num_steps"])
            fkres = self._episode_abs_and_state(rows)
            if self.use_relative_actions:
                T = rows.shape[0]
                widx = np.minimum(
                    np.arange(T)[:, None] + np.arange(H)[None, :], T - 1
                )  # [T,H] repeat-last window padding, same as per-sample path
                rel = _relative_action_windows(
                    fkres["s_pos"], fkres["s_rot"],
                    fkres["a_pos"][widx], fkres["a_rot"][widx], fkres["a_grip"][widx],
                    self.body_frame_actions,
                )
                all_actions.append(rel)
            else:
                all_actions.append(
                    self._absolute_actions20(fkres["a_pos"], fkres["a_rot"], fkres["a_grip"])
                )
            if self.use_state:
                quat = R.from_matrix(fkres["s_rot"].reshape(-1, 3, 3)).as_quat().reshape(
                    rows.shape[0], 2, 4
                )
                blocks = []
                for a in range(2):
                    rot6 = _quaternion_xyzw_to_rotation_6d(quat[:, a])
                    blocks += [
                        fkres["s_pos"][:, a].astype(np.float32),
                        rot6,
                        fkres["s_grip"][:, a:a + 1].astype(np.float32),
                    ]
                all_states.append(np.concatenate(blocks, axis=-1))

        all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
        Log.info(f"Action stats from {len(sample_idxs)} episodes, "
                 f"tensor {all_actions.shape}")
        normalizer_data = {"action": all_actions}
        if self.use_state and all_states:
            all_states = np.concatenate(all_states, axis=0).astype(np.float32)
            normalizer_data["state"] = all_states
        self.normalizer = LinearNormalizer()
        self.normalizer.fit(
            data=normalizer_data, last_n_dims=1, mode="limits",
            horizon_stats=self.use_relative_actions,
        )
        for k, v in self.normalizer.params_dict.items():
            Log.info(
                f"Normalizer {k} - min: {v['input_stats'].min}, max: {v['input_stats'].max}"
            )

    # ---------------------------------------------------------------- getitem

    def __len__(self):
        return int(self.cum_steps[-1]) * self._n_pairs

    def _resolve_sample_cameras(self, pair_idx: Optional[int] = None):
        # Parity with RobomimicHDF5Dataset._resolve_sample_cameras.
        if self._pair_choices:
            if pair_idx is None:
                import random

                pair = random.choice(self._pair_choices)
            else:
                pair = self._pair_choices[pair_idx % len(self._pair_choices)]
            return list(zip(pair, self._pair_canonical_names))
        return [(cn, cn) for cn in self.camera_names]

    def _normalize_tcp_pixel_coords_dict(self, observation: Dict[str, Any], key: str):
        # Photocopy of RobomimicHDF5Dataset._normalize_tcp_pixel_coords_dict
        # (cpgen depth handling): uv / (W-1, H-1), depth clip-normalized.
        coords_by_camera = observation.get(key)
        if not coords_by_camera:
            return
        images_by_camera = observation.get("image", {})
        for camera_name, coords in coords_by_camera.items():
            H, W = images_by_camera[camera_name].shape[1:3]
            coords[..., 0] /= max(W - 1, 1)
            coords[..., 1] /= max(H - 1, 1)
            if self.dataset_type == "cpgen":
                d = coords[..., 2]
                d = np.clip(d, self.min_depth, self.max_depth)
                coords[..., 2] = (d - self.min_depth) / (self.max_depth - self.min_depth + 1e-8)

    def _decode_images(self, ep: Dict[str, Any], obs_idxs: np.ndarray,
                       cameras_to_load, future_idx: Optional[int] = None):
        """Decode the requested frames once, slice per camera band.

        Returns ({canonical_cam: [To, H, W, 3] float32 in [0, 1]},
                 {canonical_cam: [1, H, W, 3]} or None) -- the same pre-transform
        layout RobomimicHDF5Dataset emits for image / future_image.
        """
        path = str(ep["dir"] / "combined_camera-images-rgb.mp4")
        unique_idxs = sorted(set(int(i) for i in obs_idxs)
                             | ({int(future_idx)} if future_idx is not None else set()))
        if ep.get("frame_cache"):
            mm = self._frame_mmaps.get(id(ep))
            if mm is None:
                mm = np.load(ep["frame_cache"], mmap_mode="r")
                self._frame_mmaps[id(ep)] = mm
            frames = {i: np.asarray(mm[i]) for i in unique_idxs}
        elif self.video_backend == "torchcodec":
            frames = _decode_frames_torchcodec(path, unique_idxs, ep["num_steps"], ep.get("gop_frames", GOP_FRAMES))
        else:
            frames = _decode_frames_pyav(path, unique_idxs)
        out = {}
        fut = {} if future_idx is not None else None
        for actual_cam, output_cam in cameras_to_load:
            def band(i):
                arr = _slice_camera(frames[int(i)], list(ep["cameras"]), actual_cam, ep["uuid"])
                if ep.get("crop_rows"):
                    top, bot = ep["crop_rows"]
                    arr = arr[top:bot]
                return arr
            out[output_cam] = np.stack([band(i) for i in obs_idxs]).astype(np.float32) / 255.0
            if fut is not None:
                fut[output_cam] = band(future_idx)[None].astype(np.float32) / 255.0
        return out, fut

    def __getitem__(self, idx):
        # Strategy A pair enumeration, same decomposition as the parent class.
        n_base = int(self.cum_steps[-1])
        base_idx = idx % n_base
        pair_idx = idx // n_base if self._n_pairs > 1 else None

        ep_idx = int(np.searchsorted(self.cum_steps, base_idx, side="right"))
        prev = int(self.cum_steps[ep_idx - 1]) if ep_idx > 0 else 0
        step_idx = int(base_idx - prev)
        ep = self.episodes[ep_idx]
        traj_len = ep["num_steps"]

        # Observation history indices ending at step_idx inclusive, padded by
        # repeating index 0 (parent convention).
        end_obs_idx = step_idx + 1
        start_obs_idx = max(0, end_obs_idx - self.obs_hist_length)
        obs_idxs = list(range(start_obs_idx, end_obs_idx))
        if len(obs_idxs) < self.obs_hist_length:
            obs_idxs = [0] * (self.obs_hist_length - len(obs_idxs)) + obs_idxs
        obs_idxs = np.array(obs_idxs)

        cameras_to_load = self._resolve_sample_cameras(pair_idx=pair_idx)

        # Dynamics-head target slot: step t+T, clamped to the last frame (temporal
        # validity carried separately), exactly RobomimicHDF5Dataset._future_horizon_steps.
        fut_step, fut_ok = None, None
        if self.load_future_images or self.load_future_tcp:
            raw = step_idx + self.future_horizon
            fut_ok = np.array([raw < traj_len], dtype=np.float32)  # [K=1]
            fut_step = min(raw, traj_len - 1)

        # One contiguous bin read covers state history and the action window.
        read_start = int(obs_idxs.min())
        read_end = min(step_idx + self.action_sequence_length, traj_len)
        rows = self._read_rows(ep, read_start, read_end)
        parts = self._split_rows(rows)
        station = self._get_station()

        # --- state FK on the measured joints (unique history rows) ----------
        rel_obs = obs_idxs - read_start
        uniq_rel, inverse_map = np.unique(rel_obs, return_inverse=True)
        s_pos_u, s_rot_u = station.fk(parts["state_joints"][uniq_rel])
        s_pos = s_pos_u[inverse_map]                       # [To,2,3]
        s_rot = s_rot_u[inverse_map]                       # [To,2,3,3]
        s_quat = R.from_matrix(s_rot.reshape(-1, 3, 3)).as_quat().reshape(
            len(obs_idxs), 2, 4
        )
        s_grip = parts["state_grip"][rel_obs]              # [To,2]
        s_joints = parts["state_joints"][rel_obs]          # [To,2,6]

        observation: Dict[str, Any] = {}

        # --- images ----------------------------------------------------------
        if "image" in self.observation_keys:
            imgs, fut_imgs = self._decode_images(
                ep, obs_idxs, cameras_to_load, fut_step if self.load_future_images else None)
            observation["image"] = imgs
            if fut_imgs is not None:
                observation["future_image"] = fut_imgs
                observation["future_image_valid"] = {cam: fut_ok.copy() for cam in fut_imgs}

        # --- proprio keys (parent naming: per-arm always, robot0_* aliased) --
        for a, prefix in enumerate(self.arm_obs_prefixes):
            observation[f"{prefix}_eef_pos"] = s_pos[:, a].astype(np.float32)
            observation[f"{prefix}_eef_quat_site"] = s_quat[:, a].astype(np.float32)
            observation[f"{prefix}_gripper_qpos"] = s_grip[:, a:a + 1].astype(np.float32)
            observation[f"{prefix}_joint_pos"] = s_joints[:, a].astype(np.float32)
        alias_arm = self.arm_obs_prefixes.index(self.tcp_arm) if self.tcp_arm in self.arm_obs_prefixes else 0
        robot0_alias = {
            "robot0_eef_pos": s_pos[:, alias_arm].astype(np.float32),
            "robot0_eef_quat": s_quat[:, alias_arm].astype(np.float32),
            "robot0_eef_quat_site": s_quat[:, alias_arm].astype(np.float32),
            "robot0_joint_pos": s_joints[:, alias_arm].astype(np.float32),
            "robot0_gripper_qpos": s_grip[:, alias_arm:alias_arm + 1].astype(np.float32),
        }
        for k, v in robot0_alias.items():
            if k in self.observation_keys or (self.use_state and k in (
                    "robot0_eef_pos", "robot0_eef_quat_site", "robot0_gripper_qpos")):
                observation[k] = v

        # --- TCP labels -------------------------------------------------------
        want_tcp = any(k in self.observation_keys
                       for k in ("tcp_pixel_coords", "tcp_pos", "tcp_orn"))
        labels = self._get_episode_labels(ep_idx) if (want_tcp or self.load_future_tcp) else None
        if self.load_future_tcp:
            if labels is not None:
                fut = {k: labels[k][[fut_step]] for k in
                       ("tcp_pixel_coords", "tcp_pos", "tcp_orn", "tcp_valid")}
            else:
                fut = {
                    "tcp_pixel_coords": np.zeros((1, 2, 3), dtype=np.float32),
                    "tcp_pos": np.zeros((1, 2, 3), dtype=np.float32),
                    "tcp_orn": np.tile(np.eye(3, dtype=np.float32).reshape(1, 1, 9), (1, 2, 1)),
                    "tcp_valid": np.zeros((1, 2), dtype=np.float32),
                }
            for _, output_cam in cameras_to_load:
                for key in ("tcp_pixel_coords", "tcp_pos", "tcp_orn", "tcp_valid"):
                    observation.setdefault(f"future_{key}", {})[output_cam] = fut[key].copy()
                observation.setdefault("future_tcp_temporal_valid", {})[output_cam] = fut_ok.copy()
        if want_tcp:
            To = len(obs_idxs)
            if labels is not None:
                tcp_slice = {k: labels[k][obs_idxs] for k in
                             ("tcp_pixel_coords", "tcp_pos", "tcp_orn", "tcp_valid")}
            else:
                # Unlabeled episode: emit zeros with tcp_valid=0 so batches
                # collate and TCP losses mask out these samples.
                tcp_slice = {
                    "tcp_pixel_coords": np.zeros((To, 2, 3), dtype=np.float32),
                    "tcp_pos": np.zeros((To, 2, 3), dtype=np.float32),
                    "tcp_orn": np.tile(np.eye(3, dtype=np.float32).reshape(1, 1, 9),
                                       (To, 2, 1)),
                    "tcp_valid": np.zeros((To, 2), dtype=np.float32),
                }
            # The sidecar expresses TCP in ``tcp_label_camera``'s frame. Under
            # camera_pair_choices a canonical slot can be backed by a different,
            # uncalibrated camera (e.g. the side-mounted Gemini views, which have
            # no extrinsics); copying the top-camera labels onto those would be
            # geometrically wrong, so they ride along with tcp_valid=0 and the
            # masked TCP loss ignores them.
            for key in ("tcp_pixel_coords", "tcp_pos", "tcp_orn"):
                if key in self.observation_keys:
                    observation.setdefault(key, {})
                    for actual_cam, output_cam in cameras_to_load:
                        observation[key][output_cam] = tcp_slice[key].copy()
            # tcp_valid rides along whenever tcp_pixel_coords is requested
            # (parent behavior).
            observation.setdefault("tcp_valid", {})
            for actual_cam, output_cam in cameras_to_load:
                v = tcp_slice["tcp_valid"].copy()
                if actual_cam != self.tcp_label_camera:
                    v[:] = 0.0
                observation["tcp_valid"][output_cam] = v

        # --- camera info ------------------------------------------------------
        lab = ep["label"]
        K224 = lab["K224"] if lab is not None else self._fallback_K224
        ext = (lab["extrinsics"] if (lab is not None and lab["extrinsics"] is not None)
               else self._fallback_ext)
        observation["camera_intrinsics"] = {}
        observation["camera_extrinsics"] = {}
        for _, output_cam in cameras_to_load:
            observation["camera_intrinsics"][output_cam] = K224.astype(np.float32).copy()
            observation["camera_extrinsics"][output_cam] = ext.astype(np.float32).copy()

        # --- actions ----------------------------------------------------------
        H = self.action_sequence_length
        win_rel = np.arange(step_idx - read_start, rows.shape[0])
        a_joints_win = parts["action_joints"][win_rel]
        a_grip_win = parts["action_grip"][win_rel]
        a_pos_win, a_rot_win = station.fk(a_joints_win)
        a_pos_win = self._pad_window(a_pos_win, H)
        a_rot_win = self._pad_window(a_rot_win, H)
        a_grip_win = self._pad_window(a_grip_win, H)

        if self.use_relative_actions:
            cur = len(obs_idxs) - 1  # current step = last history frame
            actions = _relative_action_windows(
                s_pos[cur:cur + 1], s_rot[cur:cur + 1],
                a_pos_win[None], a_rot_win[None], a_grip_win[None],
                self.body_frame_actions,
            )[0]
        else:
            actions = self._absolute_actions20(a_pos_win, a_rot_win, a_grip_win)
        actions = actions.astype(np.float32)

        raw_relative_action = actions.copy() if self.use_relative_actions else None
        is_last_step = step_idx >= (traj_len - 1)

        # Actions/state leave the dataset RAW. Since main 2026-09 (4e7221e, f3d3f06)
        # normalization lives in the model: train_net binds this dataset's fitted
        # normalizer to pipeline.normalizer and SAPolicy.forward_test normalizes
        # action + state itself (the eval server does the same explicitly).
        # Normalizing here as well would double-normalize (action range -> [-4, 9]).

        # tcp_orn is already in the grasp-site frame (same frame as state and
        # action rotations) -> tcp_orn_already_aligned=True, no body->site fix.

        self._normalize_tcp_pixel_coords_dict(observation, "tcp_pixel_coords")
        self._normalize_tcp_pixel_coords_dict(observation, "future_tcp_pixel_coords")

        if self.use_state:
            # Photocopy of the parent's bimanual state assembly.
            eef_rot = _quaternion_xyzw_to_rotation_6d(observation["robot0_eef_quat_site"])
            gripper_qpos = observation["robot0_gripper_qpos"][..., :1]
            blocks = []
            for prefix in self.arm_obs_prefixes:
                arm_pos = observation.get(f"{prefix}_eef_pos", observation["robot0_eef_pos"])
                arm_quat = observation.get(f"{prefix}_eef_quat_site")
                arm_rot = eef_rot if arm_quat is None else _quaternion_xyzw_to_rotation_6d(arm_quat)
                arm_grip = observation.get(f"{prefix}_gripper_qpos", gripper_qpos)[..., :1]
                blocks += [arm_pos, arm_rot, arm_grip]
            observation["state"] = np.concatenate(blocks, axis=-1)  # raw; model normalizes

        # --- per-camera transforms (photocopied application loop) -------------
        for camera_name in self.camera_names:
            camera_obs = {}
            for key, val in observation.items():
                if isinstance(val, dict) and (camera_name in val):
                    camera_obs[key] = val[camera_name]
            camera_obs = self.transforms(camera_obs)
            for key, val in camera_obs.items():
                if isinstance(observation.get(key), dict):
                    observation[key][camera_name] = val

        prompt = self.task_description
        if prompt is None:
            prompt = (ep["task_name"] or self.dataset_name).replace("_", " ")

        ret = {
            "observation": observation,
            "action": actions,
            "prompt": prompt,
            "episode_id": ep["dir"].name,
            "step_id": step_idx,
            "is_last_step": np.bool_(is_last_step),
        }
        if raw_relative_action is not None:
            ret["raw_relative_action"] = raw_relative_action
        return ret
