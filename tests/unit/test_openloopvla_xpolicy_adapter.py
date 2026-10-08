import numpy as np

from XPolicyLab.policy.OpenLoopVLA import model


def test_state_action_permutations_are_exact_inverses():
    values = np.arange(14)
    np.testing.assert_array_equal(values[model.ENV_TO_TRAIN][model.TRAIN_TO_ENV], values)
    np.testing.assert_array_equal(values[model.TRAIN_TO_ENV][model.ENV_TO_TRAIN], values)


def test_rgb_contract_rejects_resize_or_channel_layout_changes():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    assert model._rgb(image, "head") is image
    for invalid in (
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros((240, 320, 3), dtype=np.float32),
        np.zeros((3, 240, 320), dtype=np.uint8),
    ):
        try:
            model._rgb(invalid, "head")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RGB input was accepted")


def test_execution_chunk_is_twelve_and_returns_xpolicy_named_groups():
    adapter = model.Model.__new__(model.Model)
    adapter.execute_steps = 12
    adapter.action_type = "joint"
    adapter.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}

    actions_train = np.tile(np.arange(14, dtype=np.float32), (50, 1))
    chunk = adapter._unpack_actions(actions_train)

    assert len(chunk) == 12
    first = chunk[0]
    np.testing.assert_array_equal(first["left_arm_joint_state"], np.arange(6))
    np.testing.assert_array_equal(first["left_ee_joint_state"], [12])
    np.testing.assert_array_equal(first["right_arm_joint_state"], np.arange(6, 12))
    np.testing.assert_array_equal(first["right_ee_joint_state"], [13])

