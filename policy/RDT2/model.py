"""XPolicyLab adapter for RDT2 (thu-ml/RDT2) on a bimanual UMI rig.

Observation profile
-------------------
The UMI bench has **two wrist fisheye cameras and no third-person camera**.
That is the profile ``policy/Pi_05/model.py`` calls ``umi_native``
(``encode_umi_obs``, line 657); the camera-key candidate lists below are copied
from it verbatim so both adapters accept the same environment observations.

RDT2 does not take three image slots.  ``RDTInferencer.step`` receives two
separate ``(384, 384, 3)`` uint8 RGB frames keyed ``left_stereo`` /
``right_stereo`` and concatenates them itself into ``(384, 768, 3)`` with the
**left wrist on the left half**
(``RDT2/models/rdt_inferencer.py:180-186``, ``:266-268``; camera order from
``RDT2/configs/rdt/post_train.yaml:24``).  :func:`concat_wrist_images` builds
that same side-by-side frame -- it is what the offline shard format stores
(``RDT2/README.md:354``) and what the unit test pins down.

The policy server has already decoded every camera colour, so this module only
reshapes / casts / resizes.  It never decodes and never swaps channels
(AGENTS.md, "Images are RGB from end to end").  Upstream is RGB too: the
HikRobot driver converts once at the source
(``RDT2/deploy/umi/real_world/camera/mvs_cam.py:308``) and the resize helper is
built with ``bgr_to_rgb=False``
(``RDT2/deploy/umi/real_world/bimanual_umi_env.py:90-97``).

Layout conventions (the ones that fail silently)
------------------------------------------------
1. **Arm order.**  Our data and the XPolicyLab action dicts are *left-arm
   first*::

       [ L_xyz(3) L_rot6d(6) L_grip(1) | R_xyz(3) R_rot6d(6) R_grip(1) ]

   RDT2 is *right-arm first*: ``RDT2/README.md:267-276`` documents
   ``[0-2] RIGHT pos, [3-8] RIGHT rot6d, [9] RIGHT grip, [10-12] LEFT pos,
   [13-18] LEFT rot6d, [19] LEFT grip``, and the robot config comments agree
   (``RDT2/configs/robots/eval_bimanual_fr3_config.yaml:18``: ``0->right
   1->left``).  :func:`swap_arm_order` is its own named, involutive function
   precisely because getting it wrong raises nothing at all.

2. **6-D rotation layout.**  Upstream packs the **first two columns** of R:
   ``RDT2/data/umi/pose_util.py:152-156`` (``col0 = mat[..., :, 0]``), and
   decodes by Gram-Schmidt stacking ``(b1, b2, b3)`` along the *last* axis
   (``pose_util.py:98-105``).  The other common convention (pytorch3d /
   ``policy/GR00T_N17``) uses the first two *rows*; the two differ by a
   transpose and both produce a valid rotation, so this is the deploy.yml key
   ``rot6d_layout`` and it defaults to ``cols`` to match upstream.

3. **Gripper unit.**  Our ``*_ee_joint_state`` is a **normalized fraction of
   the gripper slide stroke in [0, 1]** -- not metres.  Evidence:
   ``manimux/src/manimux/kinematics/yam.py:135`` ("The gripper is a normalized
   fraction of the slide joint's stroke", followed by ``np.clip(grip, 0, 1)``),
   the gripper is ``linear_4310``
   (``manimux/src/manimux/robots/yam/arm.py:22``) whose stroke is 0.096 m
   (``i2rt/i2rt/robots/config/linear_4310.yml:43``), and
   ``exchange_ball_v0/meta/stats.json`` has ``right_gripper.pos`` topping out
   at exactly 1.0.  RDT2's channel is *"gripper width, normalized to
   [0, 0.088], 0.088 means fully open"*
   (``RDT2/ckpt/RVQ/README.md:59`` and ``:62``).  See :class:`GripperScale`.

4. **Image halves are ordered the OPPOSITE way to the action vector.**  The
   action is right-arm-first, but the side-by-side image is left-wrist-on-the-
   left.  ``left_stereo`` is fed from ``camera1_rgb``
   (``RDT2/deploy/inference_real_fm.py:390-391``) and ``configs/robots/
   eval_bimanual_fr3_config.yaml:18`` says ``0->right 1->left``, so
   ``left_stereo`` really is the *left* arm.  ``RDTInferencer`` then
   concatenates ``[left_stereo, right_stereo]``
   (``RDT2/models/rdt_inferencer.py:180-186, 266-268`` over
   ``configs/rdt/post_train.yaml:24``).  **Never apply the arm swap to the
   images.**  :data:`IMAGE_HALVES_ARE_LEFT_FIRST` states this in one place.

Action side
-----------
``RDTInferencer.step`` returns a CPU float32 ``(24, 20)`` tensor, **already
unnormalized** (``RDT2/models/rdt_inferencer.py:324-330``) and **relative** to
the current EEF pose.  Upstream rebuilds absolute poses with
``T_abs = T_anchor @ T_rel``
(``RDT2/data/umi/common/pose_repr_util.py:37-38``, driven from
``RDT2/deploy/umi/real_world/real_inference_util.py:165-183``), i.e.::

    abs_pos  = anchor_pos + anchor_rotm @ rel_pos
    abs_rotm = anchor_rotm @ rel_rotm

with the gripper channel passed through **absolutely**
(``real_inference_util.py:185``).  ``policy/Xiaomi_Robotics_1`` does the same
job for its 60-D delta chunk and this mirrors its ``_restore_abs_ee``.

Emitted action dicts use the XPolicyLab EE keys with a **7-D** pose
``[x, y, z, qw, qx, qy, qz]``.  That width is fixed by the environment-side
validator (``XPolicyLab/debug_env_client.py:258``:
``"left_ee_pose": 7``) independently of ``arm_dim`` -- which is 9 for
``umi_dual`` because the *observation* pose carries xyz + rot6d.  The two
contracts genuinely disagree; ``ee_pose_format`` lets you switch to the 9-D
``pack_robot_state`` form for an environment that wants it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# The policy server imports this module as XPolicyLab.policy.RDT2.model, so the
# importable root is the parent of the XPolicyLab checkout (AGENTS.md).
_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info

# Camera-key candidates, identical to policy/Pi_05/model.py::encode_umi_obs.
CAM_LEFT_WRIST = ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"]
CAM_RIGHT_WRIST = ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"]

# Per-arm slot widths inside the 20-D UMI vector.
ARM_BLOCK = 10
POS_SLICE = slice(0, 3)
ROT6D_SLICE = slice(3, 9)
GRIP_INDEX = 9
UMI_STATE_DIM = 2 * ARM_BLOCK

# XPolicyLab EE action poses are [x, y, z, qw, qx, qy, qz] (7).
EE_POSE_QUAT_DIM = 7
# ...but the umi_dual *observation* pose is xyz + rot6d (9), matching arm_dim.
EE_POSE_ROT6D_DIM = 9

# RDT2 concatenates the two wrist frames as [left, right]; keys and order come
# from configs/rdt/post_train.yaml:24 (camera_names: [left_stereo, right_stereo])
# and models/rdt_inferencer.py:266-268.
RDT2_CAMERA_KEYS = ("left_stereo", "right_stereo")

# The single most confusable fact in this adapter: the IMAGE halves and the
# ACTION vector use OPPOSITE arm orders.
#
#   image (384, 768, 3):  [ LEFT wrist | RIGHT wrist ]   <- natural order
#   action (20,):         [ RIGHT arm  | LEFT arm    ]   <- right-arm-first
#
# Upstream evidence: robot index 0 is the right arm and 1 the left
# (RDT2/configs/robots/eval_bimanual_fr3_config.yaml:18, "0->right 1->left"),
# and the deploy loop feeds left_stereo from camera1_rgb, i.e. the LEFT arm
# (RDT2/deploy/inference_real_fm.py:390-391). RDTInferencer then concatenates in
# camera_names order (models/rdt_inferencer.py:180-186, 266-268).
#
# Consequence: swap_arm_order applies to the 20-D state/action vectors ONLY.
# Applying it to the images too would silently train/evaluate on mirrored views.
IMAGE_HALVES_ARE_LEFT_FIRST = True
ACTION_VECTOR_IS_RIGHT_FIRST = True

# The 6-D "no rotation" value is the identity's first two columns,
# [1, 0, 0, 0, 1, 0] -- NOT a zero vector. A zero 6-D block is degenerate and
# rot6d_to_rotm rejects it.
IDENTITY_ROT6D_COLS = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


# ----------------------------------------------------------------------------
# Arm order
# ----------------------------------------------------------------------------
def swap_arm_order(vec: np.ndarray) -> np.ndarray:
    """Swap the two 10-D arm blocks of a 20-D UMI vector.

    ``[A(10) B(10)] -> [B(10) A(10)]``.  Used in both directions:

    * ours (left-first) -> RDT2 (right-first) on the way into the model, and
    * RDT2 -> ours on the way out.

    The function is its own inverse, which is what the round-trip unit test
    pins down.  Works on ``(20,)`` and on any ``(..., 20)`` stack.
    """
    arr = np.asarray(vec)
    if arr.shape[-1] != UMI_STATE_DIM:
        raise ValueError(
            f"swap_arm_order expects a trailing dim of {UMI_STATE_DIM}, got {arr.shape}"
        )
    return np.concatenate([arr[..., ARM_BLOCK:], arr[..., :ARM_BLOCK]], axis=-1)


# ----------------------------------------------------------------------------
# Rotation helpers
# ----------------------------------------------------------------------------
def rot6d_to_rotm(rot6d: np.ndarray, layout: str = "cols") -> np.ndarray:
    """6-D rotation -> 3x3 rotation matrix via Gram-Schmidt.

    ``layout='cols'`` reproduces ``RDT2/data/umi/pose_util.py:98-105``: read the
    6 numbers as ``(a1, a2)``, orthonormalize, and stack ``(b1, b2, b3)`` as the
    **columns** of R.  ``layout='rows'`` is the pytorch3d / GR00T convention and
    stacks them as rows.  The two differ by a transpose; both yield a valid
    rotation, so a mismatch is silent.
    """
    values = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    a1, a2 = values[0], values[1]

    n1 = np.linalg.norm(a1)
    if n1 < 1e-12:
        raise ValueError("rot6d first vector is degenerate")
    b1 = a1 / n1

    a2_orth = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2_orth)
    if n2 < 1e-12:
        raise ValueError("rot6d vectors are parallel; cannot orthonormalize")
    b2 = a2_orth / n2

    b3 = np.cross(b1, b2)
    if layout == "cols":
        return np.stack([b1, b2, b3], axis=-1)
    if layout == "rows":
        return np.stack([b1, b2, b3], axis=0)
    raise ValueError(f"unknown rot6d layout: {layout!r} (expected 'cols' or 'rows')")


def rotm_to_rot6d(rotm: np.ndarray, layout: str = "cols") -> np.ndarray:
    """3x3 rotation matrix -> 6-D rotation, inverse of :func:`rot6d_to_rotm`.

    ``layout='cols'`` reproduces ``RDT2/data/umi/pose_util.py:152-156``
    (``concat(mat[..., :, 0], mat[..., :, 1])``).
    """
    matrix = np.asarray(rotm, dtype=np.float64).reshape(3, 3)
    if layout == "cols":
        return np.concatenate([matrix[:, 0], matrix[:, 1]])
    if layout == "rows":
        return np.concatenate([matrix[0, :], matrix[1, :]])
    raise ValueError(f"unknown rot6d layout: {layout!r} (expected 'cols' or 'rows')")


def quat_wxyz_to_rotm(quaternion: np.ndarray) -> np.ndarray:
    """Quaternion ``[w, x, y, z]`` -> 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("end-effector quaternion cannot be zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotm_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion ``[w, x, y, z]`` (Shepperd's method).

    Sign is canonicalised to ``w >= 0``.  Same implementation as
    ``policy/Xiaomi_Robotics_1/model.py::_rotm_to_quat_wxyz`` (lines 99-145) --
    reused rather than reinvented, and unit-tested here as a round trip.
    """
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0.0 else -quaternion


def rot6d_to_quat_wxyz(rot6d: np.ndarray, layout: str = "cols") -> np.ndarray:
    """Convenience: 6-D rotation -> quaternion ``[w, x, y, z]``."""
    return rotm_to_quat_wxyz(rot6d_to_rotm(rot6d, layout=layout))


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------
def extract_image(observation: dict, candidate_names: list[str]) -> np.ndarray:
    """Pull one camera colour frame out of a v1.0 observation.

    Mirrors ``policy/Pi_05/model.py::extract_image`` (line 712), including the
    ``vision`` -> ``images`` fallback that lets unit tests pass plain dicts.
    The server has already decoded, so the value under ``color`` is an array.
    """
    vision = observation.get("vision", observation.get("images", {}))
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_hwc_uint8(image: Any) -> np.ndarray:
    """Coerce a decoded frame to contiguous HWC uint8, channel order untouched.

    Decoded arrays arriving from the server are read-only views, so this always
    hands back a fresh contiguous copy: ``cv2.resize`` rejects some non-
    contiguous inputs and nothing downstream may write through a server view.
    """
    array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)

    if array.shape[-1] in (1, 3):
        hwc = array
    elif array.shape[0] in (1, 3):
        hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if hwc.shape[-1] == 1:
        hwc = np.repeat(hwc, 3, axis=-1)

    return np.ascontiguousarray(hwc)


def resize_square(image: Any, image_size: int = 384) -> np.ndarray:
    """One wrist frame -> ``(image_size, image_size, 3)`` uint8, RGB preserved."""
    hwc = ensure_hwc_uint8(image)
    height, width = hwc.shape[:2]
    interpolation = (
        cv2.INTER_AREA
        if (height > image_size or width > image_size)
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(hwc, (image_size, image_size), interpolation=interpolation)
    return np.ascontiguousarray(resized.astype(np.uint8))


def concat_wrist_images(
    left: np.ndarray,
    right: np.ndarray,
    image_size: int = 384,
) -> np.ndarray:
    """Two wrist frames -> one ``(image_size, 2 * image_size, 3)`` uint8 image.

    Left wrist occupies the left half, right wrist the right half -- the order
    ``RDTInferencer`` builds internally (``models/rdt_inferencer.py:180-186``
    over ``camera_names = [left_stereo, right_stereo]``) and the order the
    training shards store (``RDT2/README.md:354``).  No channel conversion:
    frames are RGB in and RGB out.
    """
    left_resized = resize_square(left, image_size)
    right_resized = resize_square(right, image_size)
    return np.ascontiguousarray(
        np.concatenate([left_resized, right_resized], axis=1).astype(np.uint8)
    )


# ----------------------------------------------------------------------------
# State packing (ours, left-first, 20-D)
# ----------------------------------------------------------------------------
def _as_pos_rot6d(pose: np.ndarray, rot6d_layout: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one EE pose slot as ``(xyz, rot6d)``.

    ``umi_dual`` has ``arm_dim = 9``, so a real UMI observation carries
    ``left_ee_pose`` as ``xyz + rot6d`` (9).  The debug environment client is
    robot-agnostic and always emits a 7-D ``xyz + quat_wxyz`` pose
    (``debug_env_client.py:135``).  Both are accepted so the debug loop and the
    real rig exercise the same code path.
    """
    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.shape[0] == EE_POSE_ROT6D_DIM:
        return values[POS_SLICE].copy(), values[ROT6D_SLICE].copy()
    if values.shape[0] == EE_POSE_QUAT_DIM:
        rotm = quat_wxyz_to_rotm(values[3:7])
        return values[POS_SLICE].copy(), rotm_to_rot6d(rotm, layout=rot6d_layout)
    raise ValueError(
        f"Unsupported EE pose width {values.shape[0]}: expected "
        f"{EE_POSE_ROT6D_DIM} (xyz+rot6d) or {EE_POSE_QUAT_DIM} (xyz+quat)"
    )


def pack_umi_state(observation: dict, rot6d_layout: str = "cols") -> np.ndarray:
    """Build the 20-D left-first UMI state vector from a v1.0 observation.

    ``[L_xyz(3) L_rot6d(6) L_grip(1) R_xyz(3) R_rot6d(6) R_grip(1)]``, with the
    gripper still in *our* normalized [0, 1] stroke fraction.  This is the same
    interleaved ``[arm_0, ee_0, arm_1, ee_1]`` order
    ``XPolicyLab.utils.process_data.pack_robot_state`` produces; it is spelled
    out here only because that helper hard-fails on the debug client's 7-D
    pose.
    """
    if "observation/state" in observation:
        packed = np.asarray(observation["observation/state"], dtype=np.float64).reshape(-1)
        if packed.shape[0] != UMI_STATE_DIM:
            raise ValueError(f"observation/state must be 20-D, got {packed.shape}")
        return packed

    state = observation.get("state")
    if state is None:
        raise KeyError("observation is missing 'state'")
    if not isinstance(state, dict):
        packed = np.asarray(state, dtype=np.float64).reshape(-1)
        if packed.shape[0] != UMI_STATE_DIM:
            raise ValueError(f"flat state must be 20-D, got {packed.shape}")
        return packed

    parts = []
    for side in ("left", "right"):
        pose_key = f"{side}_ee_pose"
        grip_key = f"{side}_ee_joint_state"
        if pose_key not in state:
            raise KeyError(f"observation['state'] is missing '{pose_key}'")
        if grip_key not in state:
            raise KeyError(f"observation['state'] is missing '{grip_key}'")
        xyz, rot6d = _as_pos_rot6d(state[pose_key], rot6d_layout)
        grip = np.asarray(state[grip_key], dtype=np.float64).reshape(-1)[:1]
        parts.append(np.concatenate([xyz, rot6d, grip]))
    return np.concatenate(parts)


# ----------------------------------------------------------------------------
# Gripper unit conversion
# ----------------------------------------------------------------------------
class GripperScale:
    """Our normalized [0, 1] stroke fraction <-> RDT2's gripper channel.

    **What RDT2's channel actually is.**  Verbatim from the checkpoint card,
    ``RDT2/ckpt/RVQ/README.md:59`` (and ``:62`` for the left arm)::

        - [9]: RIGHT ARM gripper width, normalized to [0, 0.088],
               0.088 means fully open

    So the model's units are metres on the UMI-gripper scale, with 0.088 m as
    full open, and that is exactly what the training shards contain (the
    dataset slices the raw action through without any rescale,
    ``RDT2/data/umi_video_dataset.py:423``).

    **The ``/0.088*0.10`` rescale is NOT part of this.**  It appears only in the
    deploy scripts' *output* path -- ``RDT2/deploy/inference_real_fm.py:408``,
    ``:414``, ``:423`` and the same three lines in ``inference_real_vq.py`` --
    where it converts the model's width to the *specific* Franka/zhixing
    gripper those rigs use, whose ``open_width`` is 0.12 m
    (``RDT2/configs/robots/eval_bimanual_fr3_config.yaml:57``).  It is a
    per-robot calibration, not a unit conversion, and it does not appear
    anywhere in ``data/``, ``rdt/`` or ``train.py``.  Applying it here as well
    would scale our range by 1.14x for no reason -- so this class does not
    apply it at all.  :meth:`to_metres_franka` exists only to document it.

    **Our mapping.**  ``model_gripper = pos * full_open_model_value``, with two
    choices selected by ``gripper_mapping`` in deploy.yml:

    ``full_open_normalized`` (default)
        ``full_open_model_value = rdt2_gripper_width_max_m`` (0.088).  Our
        fully-open state is declared to be RDT2's fully-open state.  Every
        value stays inside the [0, 0.088] range the pre-training saw.

    ``stroke_absolute``
        ``full_open_model_value = gripper_stroke_m`` (0.096).  Our travel is
        taken literally as metres.  Physically honest about absolute width, but
        ``pos > 0.917`` lands **outside** the training range.

    **Neither is a fact.**  Our YAM ``linear_4310`` gripper (0.096 m stroke,
    ``i2rt/i2rt/robots/config/linear_4310.yml:43``; ``crank_4310`` is 0.071, so
    the constant is a config key and never a literal in this file) is not the
    UMI gripper RDT2 assumes.  Mapping full-open to full-open compresses 0.096 m
    of real travel into 0.088 m of model units -- a modelling choice, and how
    much of it fine-tuning absorbs is unknown.  ``full_open_normalized`` is the
    default only because staying in-distribution matters more for a fine-tune
    than a nominally-correct millimetre.  **The offline shard converter must
    make the same choice**; nothing here can detect a mismatch.
    """

    MAPPINGS = ("full_open_normalized", "stroke_absolute")

    def __init__(
        self,
        stroke_m: float,
        mapping: str = "full_open_normalized",
        rdt2_width_max_m: float = 0.088,
        rdt2_deploy_open_width_m: float = 0.1,
    ):
        if mapping not in self.MAPPINGS:
            raise ValueError(
                f"gripper_mapping must be one of {self.MAPPINGS}, got {mapping!r}"
            )
        if not stroke_m > 0.0:
            raise ValueError(f"gripper_stroke_m must be positive, got {stroke_m!r}")
        if not rdt2_width_max_m > 0.0:
            raise ValueError(
                f"rdt2_gripper_width_max_m must be positive, got {rdt2_width_max_m!r}"
            )
        self.stroke_m = float(stroke_m)
        self.mapping = mapping
        self.rdt2_width_max_m = float(rdt2_width_max_m)
        self.rdt2_deploy_open_width_m = float(rdt2_deploy_open_width_m)

    @property
    def full_open_model_value(self) -> float:
        """Model-space gripper value corresponding to our ``pos = 1.0``."""
        if self.mapping == "stroke_absolute":
            return self.stroke_m
        return self.rdt2_width_max_m

    def to_rdt2(self, normalized: np.ndarray | float) -> np.ndarray:
        """Our [0, 1] fraction -> the gripper width RDT2 works in (metres)."""
        fraction = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
        return fraction * self.full_open_model_value

    def from_rdt2(self, model_value: np.ndarray | float) -> np.ndarray:
        """RDT2's gripper width -> our [0, 1] fraction (inverse of to_rdt2)."""
        value = np.asarray(model_value, dtype=np.float64)
        return np.clip(value / self.full_open_model_value, 0.0, 1.0)

    def to_metres_franka(self, model_value: np.ndarray | float) -> np.ndarray:
        """Upstream's deploy-side rescale, for reference only.

        ``RDT2/deploy/inference_real_fm.py:408``: ``/ 0.088 * 0.10``.  This
        adapter never calls it -- our downstream consumes our normalized
        fraction, not a Franka width.  Kept so the constant has one documented
        home and the unit test can assert we are *not* applying it.
        """
        value = np.asarray(model_value, dtype=np.float64)
        return value / self.rdt2_width_max_m * self.rdt2_deploy_open_width_m


# ----------------------------------------------------------------------------
# Relative <-> absolute chunk
# ----------------------------------------------------------------------------
def relative_chunk_to_absolute(
    rel_chunk: np.ndarray,
    anchor_state: np.ndarray,
    rot6d_layout: str = "cols",
    translation_frame: str = "anchor",
) -> np.ndarray:
    """Relative ``(T, 20)`` chunk -> absolute ``(T, 20)`` in the anchor's frame.

    Both arrays are in *our* left-first layout; swap the arm order before
    calling if the chunk is still RDT2-ordered.

    Per arm, with ``translation_frame='anchor'`` (upstream's ``T_abs =
    T_anchor @ T_rel``, ``RDT2/data/umi/common/pose_repr_util.py:37-38``)::

        abs_pos  = anchor_pos + anchor_rotm @ rel_pos
        abs_rotm = anchor_rotm @ rel_rotm

    ``translation_frame='world'`` instead adds the translation directly
    (``abs_pos = anchor_pos + rel_pos``), which is what a converter that took
    plain coordinate differences would produce; keep it only if the offline
    shard converter is written that way.  The gripper channel is **absolute**
    and passes through unchanged -- upstream copies it straight across
    (``RDT2/deploy/umi/real_world/real_inference_util.py:185``), and that is
    also the runbook / HiFi-UMI convention.
    """
    rel = np.asarray(rel_chunk, dtype=np.float64)
    if rel.ndim != 2 or rel.shape[-1] != UMI_STATE_DIM:
        raise ValueError(f"rel_chunk must be (T, 20), got {rel.shape}")
    anchor = np.asarray(anchor_state, dtype=np.float64).reshape(-1)
    if anchor.shape[0] != UMI_STATE_DIM:
        raise ValueError(f"anchor_state must be (20,), got {anchor.shape}")
    if translation_frame not in {"anchor", "world"}:
        raise ValueError(
            f"translation_frame must be 'anchor' or 'world', got {translation_frame!r}"
        )

    out = np.empty_like(rel)
    for arm in range(2):
        base = arm * ARM_BLOCK
        anchor_pos = anchor[base : base + 3]
        anchor_rotm = rot6d_to_rotm(anchor[base + 3 : base + 9], layout=rot6d_layout)

        for t in range(rel.shape[0]):
            rel_pos = rel[t, base : base + 3]
            rel_rotm = rot6d_to_rotm(rel[t, base + 3 : base + 9], layout=rot6d_layout)

            if translation_frame == "anchor":
                abs_pos = anchor_pos + anchor_rotm @ rel_pos
            else:
                abs_pos = anchor_pos + rel_pos
            abs_rotm = anchor_rotm @ rel_rotm

            out[t, base : base + 3] = abs_pos
            out[t, base + 3 : base + 9] = rotm_to_rot6d(abs_rotm, layout=rot6d_layout)
            out[t, base + GRIP_INDEX] = rel[t, base + GRIP_INDEX]

    return out


def absolute_chunk_to_relative(
    abs_chunk: np.ndarray,
    anchor_state: np.ndarray,
    rot6d_layout: str = "cols",
    translation_frame: str = "anchor",
) -> np.ndarray:
    """Inverse of :func:`relative_chunk_to_absolute`.

    Upstream's training-side transform is ``T_rel = inv(T_anchor) @ T_abs``
    (``RDT2/data/umi/common/pose_repr_util.py:13-15``).  Only the unit test and
    the offline shard converter need this direction, but it lives next to its
    inverse so the two cannot drift apart.
    """
    absolute = np.asarray(abs_chunk, dtype=np.float64)
    if absolute.ndim != 2 or absolute.shape[-1] != UMI_STATE_DIM:
        raise ValueError(f"abs_chunk must be (T, 20), got {absolute.shape}")
    anchor = np.asarray(anchor_state, dtype=np.float64).reshape(-1)
    if anchor.shape[0] != UMI_STATE_DIM:
        raise ValueError(f"anchor_state must be (20,), got {anchor.shape}")

    out = np.empty_like(absolute)
    for arm in range(2):
        base = arm * ARM_BLOCK
        anchor_pos = anchor[base : base + 3]
        anchor_rotm = rot6d_to_rotm(anchor[base + 3 : base + 9], layout=rot6d_layout)
        anchor_rotm_t = anchor_rotm.T

        for t in range(absolute.shape[0]):
            abs_pos = absolute[t, base : base + 3]
            abs_rotm = rot6d_to_rotm(absolute[t, base + 3 : base + 9], layout=rot6d_layout)

            if translation_frame == "anchor":
                rel_pos = anchor_rotm_t @ (abs_pos - anchor_pos)
            else:
                rel_pos = abs_pos - anchor_pos
            rel_rotm = anchor_rotm_t @ abs_rotm

            out[t, base : base + 3] = rel_pos
            out[t, base + 3 : base + 9] = rotm_to_rot6d(rel_rotm, layout=rot6d_layout)
            out[t, base + GRIP_INDEX] = absolute[t, base + GRIP_INDEX]

    return out


def chunk_to_action_dicts(
    abs_chunk: np.ndarray,
    rot6d_layout: str = "cols",
    ee_pose_format: str = "quat",
) -> list[dict]:
    """Absolute left-first ``(T, 20)`` chunk -> XPolicyLab ``action_type='ee'`` dicts.

    ``ee_pose_format='quat'`` (default) emits the 7-D
    ``[x, y, z, qw, qx, qy, qz]`` pose the environment-side validator demands
    (``XPolicyLab/debug_env_client.py:258``).  ``'rot6d'`` emits the 9-D
    ``xyz + rot6d`` pose that matches ``arm_dim = 9`` and what
    ``unpack_robot_state`` would produce (the form ``policy/Pi_05`` returns).
    The two contracts disagree in this repo; see the module docstring.
    """
    chunk = np.asarray(abs_chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[-1] != UMI_STATE_DIM:
        raise ValueError(f"abs_chunk must be (T, 20), got {chunk.shape}")
    if ee_pose_format not in {"quat", "rot6d"}:
        raise ValueError(
            f"ee_pose_format must be 'quat' or 'rot6d', got {ee_pose_format!r}"
        )

    actions: list[dict] = []
    for t in range(chunk.shape[0]):
        row = chunk[t]
        action: dict[str, np.ndarray] = {}
        for arm, side in enumerate(("left", "right")):
            base = arm * ARM_BLOCK
            xyz = row[base : base + 3]
            rot6d = row[base + 3 : base + 9]
            if ee_pose_format == "quat":
                rotation = rot6d_to_quat_wxyz(rot6d, layout=rot6d_layout)
            else:
                rotation = rot6d
            action[f"{side}_ee_pose"] = np.concatenate([xyz, rotation]).astype(np.float32)
            action[f"{side}_ee_joint_state"] = np.array(
                [row[base + GRIP_INDEX]], dtype=np.float32
            )
        actions.append(action)
    return actions


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Model(ModelTemplate):
    """RDT2 flow-matching action expert, driven from XPolicyLab observations."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)

        self.action_type = self.model_cfg.get("action_type") or "ee"
        if self.action_type != "ee":
            raise ValueError(
                "RDT2 predicts end-effector poses; this adapter supports only "
                f"action_type='ee', got {self.action_type!r}."
            )

        self.env_cfg_type = self.model_cfg.get("env_cfg_type") or self.model_cfg.get("env_cfg")
        if not self.env_cfg_type:
            raise ValueError("RDT2 requires env_cfg_type (e.g. umi_dual).")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        arm_dim = list(self.robot_action_dim_info["arm_dim"])
        ee_dim = list(self.robot_action_dim_info["ee_dim"])
        if len(arm_dim) != 2 or len(ee_dim) != 2:
            raise ValueError(
                f"RDT2 is bimanual; {self.env_cfg_type!r} has arm_dim={arm_dim}, ee_dim={ee_dim}."
            )
        self.action_dim = sum(arm_dim) + sum(ee_dim)
        if self.action_dim != UMI_STATE_DIM:
            raise ValueError(
                "RDT2 expects a 20-D bimanual UMI state (3 xyz + 6 rot6d + 1 gripper "
                f"per arm); {self.env_cfg_type!r} gives {self.action_dim}."
            )
        if ee_dim != [1, 1]:
            raise ValueError(f"RDT2 expects one gripper DOF per arm, got ee_dim={ee_dim}.")

        try:
            self.batch_size = get_batch_size(self.env_cfg_type)
        except (KeyError, FileNotFoundError):
            self.batch_size = 1

        # --- layout conventions (every one a silent-failure risk) ---
        self.rot6d_layout = str(self.model_cfg.get("rot6d_layout", "cols"))
        self.translation_frame = str(
            self.model_cfg.get("relative_translation_frame", "anchor")
        )
        self.swap_arms = bool(self.model_cfg.get("swap_arm_order", True))
        self.ee_pose_format = str(self.model_cfg.get("ee_pose_format", "quat"))
        self.gripper = GripperScale(
            stroke_m=float(self.model_cfg.get("gripper_stroke_m", 0.096)),
            mapping=str(self.model_cfg.get("gripper_mapping", "full_open_normalized")),
            rdt2_width_max_m=float(self.model_cfg.get("rdt2_gripper_width_max_m", 0.088)),
            rdt2_deploy_open_width_m=float(
                self.model_cfg.get("rdt2_gripper_deploy_open_width_m", 0.1)
            ),
        )
        # Upstream's FM deploy loop feeds a zero state and says so in the README
        # ("use zero input current state for currently", RDT2/README.md:311-313),
        # so the checkpoint has never seen a real proprio vector. Sending one
        # would be off-distribution; keep it opt-in.
        self.feed_zero_state = bool(self.model_cfg.get("feed_zero_state", True))

        # --- image / chunk shape ---
        self.image_size = int(self.model_cfg.get("image_size", 384))
        self.action_chunk_size = int(self.model_cfg.get("action_chunk_size", 24))
        self.exc_action_size = int(self.model_cfg.get("exc_action_size", 12))
        self.exc_action_interval = int(self.model_cfg.get("exc_action_interval", 1))
        if self.exc_action_interval < 1:
            raise ValueError("exc_action_interval must be >= 1")
        if self.exc_action_size < 1:
            raise ValueError("exc_action_size must be >= 1")
        if self.exc_action_size * self.exc_action_interval > self.action_chunk_size:
            raise ValueError(
                "exc_action_size * exc_action_interval "
                f"({self.exc_action_size} * {self.exc_action_interval}) exceeds "
                f"action_chunk_size ({self.action_chunk_size})."
            )

        self.default_prompt = str(
            self.model_cfg.get("default_prompt")
            or self.model_cfg.get("prompt")
            or self.model_cfg.get("task_name")
            or "Exchange the ball."
        )

        # --- runtime state ---
        self._encoded_obs: dict[int, dict[str, Any]] = {}
        self._latest_env_idx_list: list[int] = [0]

        # --- weights ---
        self.rdt2_root = self._resolve_rdt2_root()
        self.checkpoint_root = self._resolve_checkpoint_root()
        self.normalizer_path = self._resolve_asset_path("normalizer_path")
        self.vlm_path = self._resolve_asset_path("vlm_path")
        self.model_config_path = self._resolve_asset_path("rdt2_model_config")
        self.policy = None
        self._stub_reason: str | None = None
        self._load_policy()

        print(
            f"[RDT2] ready: env_cfg_type={self.env_cfg_type}, action_type={self.action_type}, "
            f"chunk={self.action_chunk_size}, exec={self.exc_action_size}x{self.exc_action_interval}, "
            f"rot6d_layout={self.rot6d_layout}, translation_frame={self.translation_frame}, "
            f"swap_arm_order={self.swap_arms}, ee_pose_format={self.ee_pose_format}, "
            f"gripper_mapping={self.gripper.mapping} (stroke={self.gripper.stroke_m} m, "
            f"full-open model value={self.gripper.full_open_model_value:.5f})",
            flush=True,
        )

    # -- resolution -----------------------------------------------------------
    def _resolve_rdt2_root(self) -> Path | None:
        """Locate the upstream RDT2 tree, kept deliberately outside this repo."""
        raw = self.model_cfg.get("rdt2_root") or os.environ.get("RDT2_ROOT")
        candidates: list[Path] = []
        if raw:
            path = Path(os.path.expanduser(str(raw)))
            candidates.append(path if path.is_absolute() else (_CUR_DIR / path))
        candidates.append(_REPO_ROOT.parent / "RDT2")

        for candidate in candidates:
            if (candidate / "models" / "rdt_inferencer.py").is_file():
                resolved = candidate.resolve()
                if str(resolved) not in sys.path:
                    sys.path.insert(0, str(resolved))
                return resolved
        return None

    def _resolve_checkpoint_root(self) -> Path:
        # Shared precedence (AGENTS.md): explicit path keys > ckpt_name-as-path
        # > the concatenated 5-tuple run dir > checkpoints/<ckpt_name>.
        return resolve_checkpoint_root(
            self.model_cfg,
            _CHECKPOINTS_DIR,
            policy_dir=_CUR_DIR,
            explicit_keys=("checkpoint_path", "model_path", "ckpt_path", "model_dir"),
            must_exist=False,
        )

    def _resolve_asset_path(self, key: str) -> Path | None:
        """Resolve one auxiliary asset path against the policy dir, then RDT2."""
        raw = self.model_cfg.get(key)
        if not raw:
            return None
        path = Path(os.path.expanduser(str(raw)))
        if path.is_absolute():
            return path
        for base in (_CUR_DIR, self.rdt2_root):
            if base is None:
                continue
            candidate = (Path(base) / path).resolve()
            if candidate.exists():
                return candidate
        return (_CUR_DIR / path).resolve()

    # -- weights --------------------------------------------------------------
    def _load_policy(self) -> None:
        """Build the RDT2 flow-matching policy, or fall back to stub mode.

        Stub mode is entered only under ``EVAL_ENV_TYPE=debug``.  It exists so
        the debug closed loop can verify imports, server startup, observation
        serialization, action keys and dimensions -- none of which need real
        weights -- and it prints a red banner so nobody mistakes its output for
        a policy.

        Under ``EVAL_ENV_TYPE=debug`` the model is **not loaded at all** by
        default, even when the weights are present.  The FM path pulls a full
        Qwen2.5-VL-7B into memory (``RDT2/models/rdt_inferencer.py:115-124``,
        ~16.5 GB in bf16) plus ``torch.compile`` workers; on a 30 GB host that
        is an OOM risk, and an OOM here takes the user's input method down with
        it.  A wiring check has no business paying that cost.  Set
        ``debug_load_model: true`` in deploy.yml to override.
        """
        problems: list[str] = []
        if _eval_env_type() == "debug" and not bool(
            self.model_cfg.get("debug_load_model", False)
        ):
            self._stub_reason = (
                "EVAL_ENV_TYPE=debug and debug_load_model is false, so the "
                "Qwen2.5-VL-7B backbone was deliberately not loaded"
            )
            self._print_stub_banner()
            return

        if self.rdt2_root is None:
            problems.append(
                "upstream RDT2 tree not found (set rdt2_root in deploy.yml or $RDT2_ROOT); "
                "it is deliberately NOT vendored into XPolicyLab"
            )
        if not self.checkpoint_root.exists():
            problems.append(f"RDT2-FM checkpoint not found: {self.checkpoint_root}")
        if self.normalizer_path is None:
            problems.append("normalizer_path is unset in deploy.yml")
        elif not self.normalizer_path.exists():
            problems.append(f"normalizer not found: {self.normalizer_path}")
        if self.vlm_path is None:
            problems.append("vlm_path is unset in deploy.yml")
        elif not self.vlm_path.exists():
            problems.append(f"VLM backbone not found: {self.vlm_path}")

        if not problems:
            try:
                self.policy = self._build_upstream_policy()
                self.model = self.policy
                return
            except Exception as exc:  # noqa: BLE001 - re-raised below unless debugging
                problems.append(f"model construction failed: {exc!r}")

        reason = "; ".join(problems)
        if _eval_env_type() != "debug":
            raise FileNotFoundError(
                "[RDT2] cannot load the policy: "
                + reason
                + "\nSet rdt2_root / checkpoint_path / normalizer_path / vlm_path in "
                "deploy.yml. Weights: robotics-diffusion-transformer/RDT2-FM (the "
                "action expert), robotics-diffusion-transformer/RVQActionTokenizer "
                "(the normalizer .pt), and robotics-diffusion-transformer/RDT2-VQ "
                "(the Qwen2.5-VL-7B backbone the FM path also loads)."
            )

        self._stub_reason = reason
        self._print_stub_banner()

    def _print_stub_banner(self) -> None:
        print(
            "\033[31m"
            "[RDT2] ############################################################\n"
            "[RDT2] # STUB MODE - NO MODEL LOADED. Output is NOT a policy.      #\n"
            "[RDT2] # Every action holds the current observed pose.             #\n"
            "[RDT2] # Reason:                                                   #\n"
            f"[RDT2] #   {self._stub_reason}\n"
            "[RDT2] ############################################################"
            "\033[0m",
            flush=True,
        )

    def _build_upstream_policy(self):
        """Construct ``RDTInferencer`` from the upstream tree.

        Mirrors ``RDT2/deploy/inference_real_fm.py:204-216``::

            model = RDTInferencer(
                config=model_config, pretrained_path=input,
                normalizer_path=normalizer_path,
                pretrained_vision_language_model_name_or_path=...,
                device=device, dtype=torch.bfloat16)

        Kept in one method so the whole upstream-API surface this adapter
        depends on is a single reviewable place.

        WARNING (host RAM/VRAM): the FM path is not small.
        ``RDT2/models/rdt_inferencer.py:115-124`` loads a full
        **Qwen2.5-VL-7B** in bf16 straight onto the GPU (~16.5 GB VRAM); the
        4.2 M-parameter RDT action expert is the cheap part.  It also calls
        ``torch.compile`` (``rdt_inferencer.py:154``), whose Inductor workers
        are host-RAM hungry on the first step -- set ``disable_torch_compile:
        true`` on a memory-constrained machine.
        """
        import torch
        import yaml

        from models.rdt_inferencer import RDTInferencer  # upstream, sys.path'd above

        config_path = self.model_config_path
        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                "rdt2_model_config must point at the upstream FM config "
                "(configs/rdt/post_train.yaml); got "
                f"{config_path!r}"
            )
        with open(config_path, "r", encoding="utf-8") as handle:
            rdt2_config = yaml.safe_load(handle)

        upstream_chunk = int(rdt2_config["common"]["action_chunk_size"])
        if upstream_chunk != self.action_chunk_size:
            raise ValueError(
                f"action_chunk_size mismatch: deploy.yml says {self.action_chunk_size}, "
                f"{config_path} says {upstream_chunk}."
            )
        self.state_dim = int(rdt2_config["common"]["state_dim"])

        dtype_name = str(self.model_cfg.get("dtype", "bfloat16"))
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            dtype_name
        ]
        device = torch.device(str(self.model_cfg.get("device", "cuda")))

        if bool(self.model_cfg.get("disable_torch_compile", False)):
            # rdt_inferencer.reset() unconditionally torch.compile()s the policy;
            # make it a no-op rather than patching the upstream tree.
            torch.compile = lambda module, *args, **kwargs: module  # type: ignore[assignment]

        return RDTInferencer(
            config=rdt2_config,
            pretrained_path=str(self.checkpoint_root),
            normalizer_path=str(self.normalizer_path),
            pretrained_vision_language_model_name_or_path=str(self.vlm_path),
            device=device,
            dtype=dtype,
        )

    # -- observations ---------------------------------------------------------
    def encode_obs(self, obs: dict) -> dict[str, Any]:
        """v1.0 observation -> RDT2 inputs, plus the anchor needed to decode."""
        left = resize_square(extract_image(obs, CAM_LEFT_WRIST), self.image_size)
        right = resize_square(extract_image(obs, CAM_RIGHT_WRIST), self.image_size)

        state_ours = pack_umi_state(obs, rot6d_layout=self.rot6d_layout)

        # To RDT2 conventions: gripper in its 0.088-scale unit, right arm first.
        state_rdt2 = state_ours.copy()
        state_rdt2[GRIP_INDEX] = self.gripper.to_rdt2(state_ours[GRIP_INDEX])
        state_rdt2[ARM_BLOCK + GRIP_INDEX] = self.gripper.to_rdt2(
            state_ours[ARM_BLOCK + GRIP_INDEX]
        )
        if self.swap_arms:
            state_rdt2 = swap_arm_order(state_rdt2)

        instruction = obs.get("instruction") or obs.get("instructions") or self.default_prompt
        if isinstance(instruction, (list, tuple)) and instruction:
            instruction = instruction[0]

        return {
            "image_left": left,             # (S, S, 3) uint8 RGB
            "image_right": right,           # (S, S, 3) uint8 RGB
            "state_rdt2": state_rdt2,       # (20,) right-first, RDT2 gripper unit
            "anchor_state": state_ours,     # (20,) left-first, our gripper unit
            "instruction": str(instruction),
        }

    def update_obs(self, obs):
        if "env_idx" not in obs:
            obs = {**obs, "env_idx": 0}
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [
            obs.get("env_idx", index) for index, obs in enumerate(obs_list)
        ]
        for env_idx, obs in zip(self._latest_env_idx_list, obs_list):
            self._encoded_obs[env_idx] = self.encode_obs(obs)

    # -- actions --------------------------------------------------------------
    def get_action(self, **kwargs):
        if not self._encoded_obs:
            raise AssertionError("[RDT2] call update_obs before get_action.")
        return self._chunk_for_env(self._latest_env_idx_list[0])

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._encoded_obs:
            raise AssertionError("[RDT2] call update_obs_batch before get_action_batch.")
        if env_idx_list is None:
            env_idx_list = kwargs.get("obs")
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        elif isinstance(env_idx_list, np.ndarray):
            env_idx_list = env_idx_list.reshape(-1).tolist()
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        else:
            env_idx_list = list(env_idx_list)
        return [self._chunk_for_env(env_idx) for env_idx in env_idx_list]

    def _chunk_for_env(self, env_idx: int) -> list[dict]:
        encoded = self._encoded_obs.get(env_idx)
        if encoded is None:
            raise KeyError(f"[RDT2] no observation stored for env_idx={env_idx}")

        rel_rdt2 = np.asarray(self._forward(encoded), dtype=np.float64)
        if rel_rdt2.ndim != 2 or rel_rdt2.shape[-1] != UMI_STATE_DIM:
            raise ValueError(
                f"[RDT2] expected a (T, {UMI_STATE_DIM}) chunk, got {rel_rdt2.shape}"
            )

        rel_ours = swap_arm_order(rel_rdt2) if self.swap_arms else rel_rdt2
        rel_ours = np.array(rel_ours, dtype=np.float64, copy=True)

        # The gripper channel is absolute inside the chunk; convert its unit,
        # never integrate it.
        rel_ours[:, GRIP_INDEX] = self.gripper.from_rdt2(rel_ours[:, GRIP_INDEX])
        rel_ours[:, ARM_BLOCK + GRIP_INDEX] = self.gripper.from_rdt2(
            rel_ours[:, ARM_BLOCK + GRIP_INDEX]
        )

        abs_ours = relative_chunk_to_absolute(
            rel_ours,
            encoded["anchor_state"],
            rot6d_layout=self.rot6d_layout,
            translation_frame=self.translation_frame,
        )

        selected = abs_ours[:: self.exc_action_interval][: self.exc_action_size]
        return chunk_to_action_dicts(
            selected,
            rot6d_layout=self.rot6d_layout,
            ee_pose_format=self.ee_pose_format,
        )

    def _forward(self, encoded: dict[str, Any]) -> np.ndarray:
        """One RDT2 forward -> ``(action_chunk_size, 20)`` relative, RDT2 layout."""
        if self.policy is None:
            return self._stub_chunk(encoded)

        state = (
            np.zeros(self.state_dim, dtype=np.float32)
            if self.feed_zero_state
            else encoded["state_rdt2"].astype(np.float32)
        )
        observations = {
            "images": {
                RDT2_CAMERA_KEYS[0]: encoded["image_left"],
                RDT2_CAMERA_KEYS[1]: encoded["image_right"],
            },
            "state": state,
        }
        result = self.policy.step(
            observations=observations, instruction=encoded["instruction"]
        )
        return result.detach().cpu().numpy()

    def _stub_chunk(self, encoded: dict[str, Any]) -> np.ndarray:
        """Identity relative chunk: hold the current pose, keep the gripper.

        Deliberately routed through the *real* decode path so the debug loop
        still exercises the arm swap, the gripper rescale, the relative ->
        absolute reconstruction and the 6D -> quaternion conversion.

        Note the rotation block is the **identity** 6-D value, not zeros: a
        zero 6-D block is degenerate and rot6d_to_rotm rejects it.
        """
        chunk = np.zeros((self.action_chunk_size, UMI_STATE_DIM), dtype=np.float64)
        identity_rot6d = rotm_to_rot6d(np.eye(3), layout=self.rot6d_layout)
        for arm in range(2):
            base = arm * ARM_BLOCK
            chunk[:, base + 3 : base + 9] = identity_rot6d
            chunk[:, base + GRIP_INDEX] = encoded["state_rdt2"][base + GRIP_INDEX]
        return chunk

    def reset(self):
        self._encoded_obs.clear()
        self._latest_env_idx_list = [0]


def _eval_env_type() -> str:
    return (os.environ.get("EVAL_ENV_TYPE") or "sim").strip().lower()
