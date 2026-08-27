"""Unit tests for the RDT2 UMI adapter's silent-failure conventions.

Every conversion covered here fails *quietly* when it is wrong -- a swapped arm
order, a transposed 6-D rotation, a mis-scaled gripper and a mis-anchored
relative chunk all produce well-formed numbers of the right shape. These tests
are the only thing that pins them down without a real checkpoint.

Style follows tests/unit/test_pi05_umi_encode.py: import module-level functions
(never the ``Model`` class, which would try to load weights).
"""

from __future__ import annotations

import numpy as np

from XPolicyLab.policy.RDT2.model import (
    ARM_BLOCK,
    GRIP_INDEX,
    IDENTITY_ROT6D_COLS,
    ACTION_VECTOR_IS_RIGHT_FIRST,
    IMAGE_HALVES_ARE_LEFT_FIRST,
    RDT2_CAMERA_KEYS,
    UMI_STATE_DIM,
    GripperScale,
    absolute_chunk_to_relative,
    chunk_to_action_dicts,
    concat_wrist_images,
    pack_umi_state,
    quat_wxyz_to_rotm,
    relative_chunk_to_absolute,
    rot6d_to_quat_wxyz,
    rot6d_to_rotm,
    rotm_to_quat_wxyz,
    rotm_to_rot6d,
    swap_arm_order,
)

UMI_DIM_INFO = {"arm_dim": [9, 9], "ee_dim": [1, 1]}


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    """A generic, non-symmetric rotation matrix built from two Euler angles."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rz @ ry


def _state(
    left_pos: np.ndarray,
    left_rotm: np.ndarray,
    left_grip: float,
    right_pos: np.ndarray,
    right_rotm: np.ndarray,
    right_grip: float,
    layout: str = "cols",
) -> np.ndarray:
    """Build a 20-D left-first UMI state vector."""
    return np.concatenate(
        [
            left_pos,
            rotm_to_rot6d(left_rotm, layout=layout),
            [left_grip],
            right_pos,
            rotm_to_rot6d(right_rotm, layout=layout),
            [right_grip],
        ]
    )


def _raw_observation(fill: float) -> dict:
    """A v1.0-shaped observation with the 9-D (xyz + rot6d) umi_dual pose."""
    return {
        "state": {
            "left_ee_pose": np.full(9, fill, dtype=np.float32),
            "left_ee_joint_state": np.array([0.25], dtype=np.float32),
            "right_ee_pose": np.full(9, fill, dtype=np.float32),
            "right_ee_joint_state": np.array([0.75], dtype=np.float32),
        },
        "images": {
            "left_camera": np.zeros((4, 6, 3), dtype=np.uint8),
            "right_camera": np.full((4, 6, 3), 255, dtype=np.uint8),
        },
        "instruction": "Exchange the ball.",
    }


# ---------------------------------------------------------------------------
# 1. Arm order
# ---------------------------------------------------------------------------
def test_swap_arm_order_is_its_own_inverse() -> None:
    vector = np.arange(UMI_STATE_DIM, dtype=np.float64)
    np.testing.assert_array_equal(swap_arm_order(swap_arm_order(vector)), vector)


def test_swap_arm_order_moves_the_right_block_to_the_front() -> None:
    # Ours is left-first; RDT2 is right-first (RDT2/README.md:267-276).
    vector = np.arange(UMI_STATE_DIM, dtype=np.float64)
    swapped = swap_arm_order(vector)
    np.testing.assert_array_equal(swapped[:ARM_BLOCK], vector[ARM_BLOCK:])
    np.testing.assert_array_equal(swapped[ARM_BLOCK:], vector[:ARM_BLOCK])
    # The right gripper (index 19 for us) lands on RDT2's index 9.
    assert swapped[GRIP_INDEX] == vector[ARM_BLOCK + GRIP_INDEX]


def test_swap_arm_order_works_on_a_chunk_and_round_trips() -> None:
    chunk = np.arange(5 * UMI_STATE_DIM, dtype=np.float64).reshape(5, UMI_STATE_DIM)
    np.testing.assert_array_equal(swap_arm_order(swap_arm_order(chunk)), chunk)
    np.testing.assert_array_equal(swap_arm_order(chunk)[:, :ARM_BLOCK], chunk[:, ARM_BLOCK:])


def test_swap_arm_order_rejects_a_wrong_width() -> None:
    try:
        swap_arm_order(np.zeros(19))
    except ValueError:
        return
    raise AssertionError("swap_arm_order accepted a 19-D vector")


# ---------------------------------------------------------------------------
# 2. 6-D rotation <-> matrix <-> quaternion
# ---------------------------------------------------------------------------
def test_rot6d_column_layout_matches_upstream_mat_to_rot6d() -> None:
    # RDT2/data/umi/pose_util.py:152-156 concatenates mat[:, 0] and mat[:, 1].
    rotm = _rotation(0.4, -0.9)
    np.testing.assert_allclose(
        rotm_to_rot6d(rotm, layout="cols"),
        np.concatenate([rotm[:, 0], rotm[:, 1]]),
    )


def test_rot6d_row_layout_is_the_transpose_of_the_column_layout() -> None:
    # The silent-failure case: both are valid rotations, one is wrong.
    rotm = _rotation(1.1, 0.3)
    as_cols = rot6d_to_rotm(rotm_to_rot6d(rotm, layout="cols"), layout="cols")
    as_rows = rot6d_to_rotm(rotm_to_rot6d(rotm, layout="cols"), layout="rows")
    np.testing.assert_allclose(as_cols, rotm, atol=1e-12)
    np.testing.assert_allclose(as_rows, rotm.T, atol=1e-12)


def test_rot6d_to_rotm_gram_schmidt_orthonormalizes_a_noisy_input() -> None:
    rotm = _rotation(-0.6, 0.25)
    noisy = rotm_to_rot6d(rotm, layout="cols") + 0.05
    recovered = rot6d_to_rotm(noisy, layout="cols")
    np.testing.assert_allclose(recovered @ recovered.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(recovered) > 0.0  # right-handed, not a reflection


def test_rot6d_to_quat_round_trips_through_the_matrix() -> None:
    for layout in ("cols", "rows"):
        for yaw, pitch in ((0.0, 0.0), (0.7, -1.2), (2.9, 0.05), (-1.4, 1.5)):
            rotm = _rotation(yaw, pitch)
            quat = rot6d_to_quat_wxyz(rotm_to_rot6d(rotm, layout=layout), layout=layout)
            assert abs(np.linalg.norm(quat) - 1.0) < 1e-12
            assert quat[0] >= 0.0  # canonical sign
            np.testing.assert_allclose(quat_wxyz_to_rotm(quat), rotm, atol=1e-10)


def test_rotm_to_quat_handles_the_trace_negative_branches() -> None:
    # 180-degree turns drive trace <= 0 and exercise each Shepperd branch.
    for diagonal in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        rotm = np.diag(np.array(diagonal, dtype=np.float64))
        quat = rotm_to_quat_wxyz(rotm)
        np.testing.assert_allclose(quat_wxyz_to_rotm(quat), rotm, atol=1e-10)


def test_identity_rotation_maps_to_the_identity_quaternion() -> None:
    quat = rot6d_to_quat_wxyz(rotm_to_rot6d(np.eye(3), layout="cols"), layout="cols")
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Images
# ---------------------------------------------------------------------------
def test_concat_wrist_images_shape_dtype_and_halves() -> None:
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.full((480, 640, 3), 200, dtype=np.uint8)

    concatenated = concat_wrist_images(left, right, image_size=384)

    # RDT2 wants (384, 768, 3) uint8 (RDT2/README.md:354).
    assert concatenated.shape == (384, 768, 3)
    assert concatenated.dtype == np.uint8
    # Left wrist on the LEFT half (RDT2/models/rdt_inferencer.py:180-186).
    assert concatenated[:, :384].max() == 0
    assert concatenated[:, 384:].min() == 200


def test_image_halves_and_action_vector_use_opposite_arm_orders() -> None:
    """The single most confusable fact in this adapter.

    image  (384, 768, 3): [ LEFT wrist | RIGHT wrist ]  -- natural order
    action (20,):         [ RIGHT arm  | LEFT arm    ]  -- right-arm-first

    Upstream: robot 0 is the right arm and robot 1 the left
    (RDT2/configs/robots/eval_bimanual_fr3_config.yaml:18), and the deploy loop
    feeds left_stereo from camera1_rgb, i.e. the LEFT arm
    (RDT2/deploy/inference_real_fm.py:390-391). So swap_arm_order applies to the
    20-D vectors ONLY -- swapping the images too would mirror the views.
    """
    assert IMAGE_HALVES_ARE_LEFT_FIRST is True
    assert ACTION_VECTOR_IS_RIGHT_FIRST is True
    # The camera keys are in the order RDTInferencer concatenates them
    # (configs/rdt/post_train.yaml:24).
    assert RDT2_CAMERA_KEYS == ("left_stereo", "right_stereo")

    left = np.zeros((32, 32, 3), dtype=np.uint8)
    right = np.full((32, 32, 3), 255, dtype=np.uint8)
    concatenated = concat_wrist_images(left, right, image_size=16)

    # Left wrist really is on the left half, with no arm swap applied.
    assert concatenated[:, :16].max() == 0
    assert concatenated[:, 16:].min() == 255


def test_the_six_d_zero_rotation_is_the_identity_not_a_zero_vector() -> None:
    # Measured on a real UR5e example shard: chunk row 0's d3-d8 is
    # [1, ~0, ~0, ~0, 1, ~0], not all zeros.
    np.testing.assert_allclose(
        rotm_to_rot6d(np.eye(3), layout="cols"), IDENTITY_ROT6D_COLS
    )
    np.testing.assert_allclose(
        rot6d_to_rotm(IDENTITY_ROT6D_COLS, layout="cols"), np.eye(3), atol=1e-12
    )
    # A zero 6-D block is degenerate and must be rejected, not silently accepted.
    try:
        rot6d_to_rotm(np.zeros(6), layout="cols")
    except ValueError:
        return
    raise AssertionError("rot6d_to_rotm accepted a zero 6-D vector")


def test_concat_wrist_images_accepts_read_only_and_chw_inputs() -> None:
    # Server-decoded arrays arrive as read-only views; CHW must be accepted too.
    left = np.zeros((3, 120, 160), dtype=np.uint8)
    right = np.full((120, 160, 3), 7, dtype=np.uint8)
    right.setflags(write=False)

    concatenated = concat_wrist_images(left, right, image_size=64)

    assert concatenated.shape == (64, 128, 3)
    assert concatenated.dtype == np.uint8
    assert concatenated.flags["WRITEABLE"]  # a copy, not a view on the input
    assert not right.flags["WRITEABLE"]     # the caller's array was not mutated


def test_concat_wrist_images_upcasts_float_frames() -> None:
    left = np.zeros((32, 32, 3), dtype=np.float32)
    right = np.ones((32, 32, 3), dtype=np.float32)

    concatenated = concat_wrist_images(left, right, image_size=16)

    assert concatenated.dtype == np.uint8
    assert concatenated[:, 16:].min() == 255


# ---------------------------------------------------------------------------
# 4. State packing
# ---------------------------------------------------------------------------
def test_pack_umi_state_reads_the_9d_umi_dual_pose() -> None:
    packed = pack_umi_state(_raw_observation(0.5))
    assert packed.shape == (UMI_STATE_DIM,)
    # Interleaved [arm_0, ee_0, arm_1, ee_1], like pack_robot_state.
    assert packed[GRIP_INDEX] == 0.25
    assert packed[ARM_BLOCK + GRIP_INDEX] == 0.75


def test_pack_umi_state_also_accepts_the_debug_clients_7d_quat_pose() -> None:
    # debug_env_client.py:135 emits a 7-D xyz+quat pose regardless of arm_dim,
    # so the debug loop and the real rig must share one code path.
    rotm = _rotation(0.3, 0.8)
    quat = rotm_to_quat_wxyz(rotm)
    observation = {
        "state": {
            "left_ee_pose": np.concatenate([[1.0, 2.0, 3.0], quat]).astype(np.float32),
            "left_ee_joint_state": np.array([0.4], dtype=np.float32),
            "right_ee_pose": np.concatenate([[4.0, 5.0, 6.0], quat]).astype(np.float32),
            "right_ee_joint_state": np.array([0.6], dtype=np.float32),
        },
    }

    packed = pack_umi_state(observation, rot6d_layout="cols")

    assert packed.shape == (UMI_STATE_DIM,)
    np.testing.assert_allclose(packed[:3], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        rot6d_to_rotm(packed[3:9], layout="cols"), rotm, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 5. Gripper
# ---------------------------------------------------------------------------
def test_gripper_full_open_normalized_maps_full_open_to_rdt2s_0088() -> None:
    # RDT2/ckpt/RVQ/README.md:59 -- "normalized to [0, 0.088], 0.088 means
    # fully open". Our pos=1.0 is also fully open, so they map onto each other.
    scale = GripperScale(stroke_m=0.096, mapping="full_open_normalized")

    assert np.isclose(scale.to_rdt2(1.0), 0.088)
    assert np.isclose(scale.to_rdt2(0.0), 0.0)
    for fraction in (0.0, 0.19496, 0.5389, 0.9143, 1.0):  # real stats.json values
        assert np.isclose(scale.from_rdt2(scale.to_rdt2(fraction)), fraction)


def test_gripper_stroke_absolute_uses_our_literal_metres() -> None:
    scale = GripperScale(stroke_m=0.096, mapping="stroke_absolute")

    assert np.isclose(scale.to_rdt2(1.0), 0.096)
    # ...which is outside RDT2's [0, 0.088] training range. That is the cost of
    # this mapping and the reason it is not the default.
    assert scale.to_rdt2(1.0) > 0.088
    assert np.isclose(scale.from_rdt2(scale.to_rdt2(0.3)), 0.3)


def test_the_two_gripper_mappings_actually_disagree() -> None:
    # If they ever agreed, the deploy.yml key would be decoration.
    stroke = GripperScale(stroke_m=0.096, mapping="stroke_absolute")
    normalized = GripperScale(stroke_m=0.096, mapping="full_open_normalized")
    assert not np.isclose(stroke.to_rdt2(1.0), normalized.to_rdt2(1.0))


def test_the_deploy_side_rescale_is_not_applied_to_our_values() -> None:
    """The /0.088*0.10 rescale is upstream's per-robot calibration, not ours.

    It lives only in RDT2/deploy/inference_real_*.py (output side, three times
    over) and nowhere in data/ or train.py. Applying it here too would stretch
    the range by 1.14x. This test fails the moment someone folds it in.
    """
    scale = GripperScale(stroke_m=0.096, mapping="full_open_normalized")

    assert np.isclose(scale.to_rdt2(1.0), 0.088)
    # The Franka-specific value, kept for documentation only.
    assert np.isclose(scale.to_metres_franka(scale.to_rdt2(1.0)), 0.10)
    assert not np.isclose(scale.to_rdt2(1.0), 0.10)


def test_gripper_stroke_is_configurable_for_other_gripper_types() -> None:
    # crank_4310 is 0.071 m, so the stroke can never be a literal in model.py.
    crank = GripperScale(stroke_m=0.071, mapping="stroke_absolute")
    assert np.isclose(crank.to_rdt2(1.0), 0.071)


def test_gripper_clamps_out_of_range_values() -> None:
    scale = GripperScale(stroke_m=0.096)
    assert np.isclose(scale.to_rdt2(1.5), scale.to_rdt2(1.0))
    assert np.isclose(scale.to_rdt2(-0.2), 0.0)
    assert np.isclose(scale.from_rdt2(10.0), 1.0)


def test_gripper_rejects_an_unknown_mapping() -> None:
    try:
        GripperScale(stroke_m=0.096, mapping="guess")
    except ValueError:
        return
    raise AssertionError("GripperScale accepted an unknown mapping")


# ---------------------------------------------------------------------------
# 6. Relative <-> absolute chunk
# ---------------------------------------------------------------------------
def test_relative_to_absolute_round_trips_against_its_inverse() -> None:
    rng = np.random.default_rng(0)
    anchor = _state(
        np.array([0.12, 0.18, -0.80]), _rotation(0.4, -0.2), 0.5,
        np.array([0.16, -0.14, -0.82]), _rotation(-1.1, 0.7), 0.6,
    )
    absolute = np.stack(
        [
            _state(
                anchor[:3] + rng.normal(scale=0.02, size=3),
                _rotation(0.4 + 0.01 * step, -0.2 + 0.02 * step),
                0.5,
                anchor[ARM_BLOCK : ARM_BLOCK + 3] + rng.normal(scale=0.02, size=3),
                _rotation(-1.1 - 0.01 * step, 0.7 + 0.015 * step),
                0.6,
            )
            for step in range(24)
        ]
    )

    relative = absolute_chunk_to_relative(absolute, anchor)
    recovered = relative_chunk_to_absolute(relative, anchor)

    np.testing.assert_allclose(recovered, absolute, atol=1e-9)


def test_a_zero_relative_chunk_reproduces_the_anchor_exactly() -> None:
    # The identity relative action must be a no-op; this is what stub mode emits.
    anchor = _state(
        np.array([0.1, 0.2, -0.8]), _rotation(0.9, 0.3), 0.42,
        np.array([0.3, -0.1, -0.9]), _rotation(-0.5, 1.4), 0.84,
    )
    identity = np.zeros((24, UMI_STATE_DIM))
    identity_rot6d = rotm_to_rot6d(np.eye(3), layout="cols")
    for arm in range(2):
        base = arm * ARM_BLOCK
        identity[:, base + 3 : base + 9] = identity_rot6d
        identity[:, base + GRIP_INDEX] = anchor[base + GRIP_INDEX]

    absolute = relative_chunk_to_absolute(identity, anchor)

    for step in range(24):
        np.testing.assert_allclose(absolute[step], anchor, atol=1e-12)


def test_relative_translation_is_expressed_in_the_anchor_frame() -> None:
    # Upstream composes T_abs = T_anchor @ T_rel, so a pure +x relative step
    # moves along the anchor's own x axis, not the world x axis.
    anchor_rotm = _rotation(np.pi / 2, 0.0)  # world x -> anchor y
    anchor = _state(
        np.zeros(3), anchor_rotm, 0.0, np.zeros(3), np.eye(3), 0.0
    )
    relative = np.zeros((1, UMI_STATE_DIM))
    identity_rot6d = rotm_to_rot6d(np.eye(3), layout="cols")
    for arm in range(2):
        base = arm * ARM_BLOCK
        relative[:, base + 3 : base + 9] = identity_rot6d
    relative[0, 0] = 0.1  # +0.1 along the left arm's local x

    in_anchor = relative_chunk_to_absolute(relative, anchor, translation_frame="anchor")
    in_world = relative_chunk_to_absolute(relative, anchor, translation_frame="world")

    np.testing.assert_allclose(in_anchor[0, :3], anchor_rotm @ [0.1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(in_world[0, :3], [0.1, 0.0, 0.0], atol=1e-12)
    # The two conventions genuinely differ, which is why it is a config key.
    assert not np.allclose(in_anchor[0, :3], in_world[0, :3])


def test_the_gripper_channel_is_absolute_and_passes_through_untouched() -> None:
    # RDT2/deploy/umi/real_world/real_inference_util.py:185 copies it straight
    # across -- it is never composed with the anchor.
    anchor = _state(
        np.array([1.0, 2.0, 3.0]), _rotation(0.2, 0.2), 0.11,
        np.array([4.0, 5.0, 6.0]), _rotation(0.5, -0.3), 0.22,
    )
    relative = np.zeros((3, UMI_STATE_DIM))
    identity_rot6d = rotm_to_rot6d(np.eye(3), layout="cols")
    for arm in range(2):
        relative[:, arm * ARM_BLOCK + 3 : arm * ARM_BLOCK + 9] = identity_rot6d
    relative[:, GRIP_INDEX] = [0.3, 0.4, 0.5]
    relative[:, ARM_BLOCK + GRIP_INDEX] = [0.9, 0.8, 0.7]

    absolute = relative_chunk_to_absolute(relative, anchor)

    np.testing.assert_allclose(absolute[:, GRIP_INDEX], [0.3, 0.4, 0.5])
    np.testing.assert_allclose(absolute[:, ARM_BLOCK + GRIP_INDEX], [0.9, 0.8, 0.7])


# ---------------------------------------------------------------------------
# 7. Action dicts
# ---------------------------------------------------------------------------
def test_chunk_to_action_dicts_emits_the_7d_quat_pose_the_env_validates() -> None:
    anchor = _state(
        np.array([0.1, 0.2, -0.8]), _rotation(0.9, 0.3), 0.42,
        np.array([0.3, -0.1, -0.9]), _rotation(-0.5, 1.4), 0.84,
    )
    actions = chunk_to_action_dicts(anchor[None, :])

    assert len(actions) == 1
    action = actions[0]
    assert set(action) == {
        "left_ee_pose",
        "right_ee_pose",
        "left_ee_joint_state",
        "right_ee_joint_state",
    }
    for key in ("left_ee_pose", "right_ee_pose"):
        # debug_env_client.py:258 hard-codes 7 for *_ee_pose.
        assert action[key].shape == (7,)
        assert action[key].dtype == np.float32
        assert abs(np.linalg.norm(action[key][3:]) - 1.0) < 1e-6
    for key in ("left_ee_joint_state", "right_ee_joint_state"):
        assert action[key].shape == (1,)
        assert action[key].dtype == np.float32
    assert np.isclose(action["left_ee_joint_state"][0], 0.42, atol=1e-6)
    assert np.isclose(action["right_ee_joint_state"][0], 0.84, atol=1e-6)


def test_chunk_to_action_dicts_can_emit_the_9d_rot6d_pose() -> None:
    # The umi_dual arm_dim=9 form, what unpack_robot_state / Pi_05 produce.
    anchor = _state(
        np.array([0.1, 0.2, -0.8]), _rotation(0.9, 0.3), 0.42,
        np.array([0.3, -0.1, -0.9]), _rotation(-0.5, 1.4), 0.84,
    )
    action = chunk_to_action_dicts(anchor[None, :], ee_pose_format="rot6d")[0]

    assert action["left_ee_pose"].shape == (9,)
    np.testing.assert_allclose(action["left_ee_pose"], anchor[:9], atol=1e-6)


def test_action_pose_matches_the_state_it_came_from() -> None:
    # End to end: pack a 9-D observation pose, decode it to a quaternion action,
    # and confirm both describe the same rotation.
    rotm = _rotation(1.3, -0.4)
    observation = {
        "state": {
            "left_ee_pose": np.concatenate(
                [[0.1, 0.2, 0.3], rotm_to_rot6d(rotm, layout="cols")]
            ).astype(np.float32),
            "left_ee_joint_state": np.array([0.5], dtype=np.float32),
            "right_ee_pose": np.concatenate(
                [[0.4, 0.5, 0.6], rotm_to_rot6d(rotm, layout="cols")]
            ).astype(np.float32),
            "right_ee_joint_state": np.array([0.5], dtype=np.float32),
        },
    }

    packed = pack_umi_state(observation, rot6d_layout="cols")
    action = chunk_to_action_dicts(packed[None, :], rot6d_layout="cols")[0]

    np.testing.assert_allclose(action["left_ee_pose"][:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(
        quat_wxyz_to_rotm(action["left_ee_pose"][3:]), rotm, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 8. The full inbound/outbound convention round trip
# ---------------------------------------------------------------------------
def test_arm_swap_and_gripper_scaling_survive_a_full_round_trip() -> None:
    """ours -> RDT2 conventions -> back, the exact path get_action walks.

    Catches a one-sided swap (swap in, forget to swap out) and a gripper
    conversion applied in the wrong direction -- both of which leave shapes and
    dtypes perfectly intact.
    """
    scale = GripperScale(stroke_m=0.096, mapping="full_open_normalized")
    ours = _state(
        np.array([0.12, 0.18, -0.80]), _rotation(0.4, -0.2), 0.2261,
        np.array([0.16, -0.14, -0.82]), _rotation(-1.1, 0.7), 0.9143,
    )

    # Outbound: gripper unit, then arm order.
    to_model = ours.copy()
    to_model[GRIP_INDEX] = scale.to_rdt2(ours[GRIP_INDEX])
    to_model[ARM_BLOCK + GRIP_INDEX] = scale.to_rdt2(ours[ARM_BLOCK + GRIP_INDEX])
    to_model = swap_arm_order(to_model)

    # The right arm's block (ours index 10..19) is now first, as RDT2 wants.
    np.testing.assert_allclose(to_model[:3], ours[ARM_BLOCK : ARM_BLOCK + 3])

    # Inbound: arm order, then gripper unit.
    back = swap_arm_order(to_model).copy()
    back[GRIP_INDEX] = scale.from_rdt2(back[GRIP_INDEX])
    back[ARM_BLOCK + GRIP_INDEX] = scale.from_rdt2(back[ARM_BLOCK + GRIP_INDEX])

    np.testing.assert_allclose(back, ours, atol=1e-12)
