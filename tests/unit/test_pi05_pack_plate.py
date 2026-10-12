"""Pack-plate artifact isolation without loading a checkpoint or hardware."""

import hashlib
import json

import numpy as np
import pytest

from XPolicyLab.policy.Pi_05.model import Model
from XPolicyLab.policy.Pi_05.pack_plate_contract import (
    ASSET,
    PROFILE,
    SOURCE,
    TRAIN_CONFIG,
    validate_artifacts,
)
from XPolicyLab.policy.Pi_05.pack_plate_model import (
    PackPlateZeroPoseModel,
    encode_rtc_condition,
    make_pack_plate_train_config,
)
from XPolicyLab.policy.Pi_05.pass_ball_model import PassBallInputs
from XPolicyLab.policy.Pi_05.pass_ball_pose import (
    OPTICAL_FROM_FLU,
    relative_action_chunk,
    rotation_to_6d,
)


def test_pack_plate_dispatch_does_not_change_pass_ball(monkeypatch):
    from XPolicyLab.policy.Pi_05 import pack_plate_model

    sentinel = object()
    monkeypatch.setattr(pack_plate_model, "PackPlateZeroPoseModel", lambda config: sentinel)
    assert Model.__new__(Model, {"observation_profile": PROFILE}) is sentinel
    assert type(Model.__new__(Model, {"observation_profile": "yam_native"})) is Model


def test_full_checkpoint_architecture_preserves_old_lora_profile():
    pytest.importorskip("openpi")
    common = {"repo_id": "pack-plate", "train_config_name": TRAIN_CONFIG}
    old = make_pack_plate_train_config(common)
    full = make_pack_plate_train_config({
        **common,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
    })
    assert old.model.paligemma_variant == "gemma_2b_lora"
    assert old.model.action_expert_variant == "gemma_300m_lora"
    assert full.model.paligemma_variant == "gemma_2b"
    assert full.model.action_expert_variant == "gemma_300m"
    assert full.model.pi05 and full.model.action_dim == 32
    assert full.model.action_horizon == 32


def test_pack_plate_requires_its_own_checkpoint_and_stats(tmp_path):
    checkpoint = tmp_path / SOURCE / "checkpoint-59999"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params/_METADATA").write_text("{}")
    stats_dir = checkpoint / "assets" / ASSET
    stats_dir.mkdir(parents=True)
    stats = {
        name: {field: [0.0] * 20 for field in ("mean", "std", "q01", "q99")}
        for name in ("state", "actions")
    }
    path = stats_dir / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": stats}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {
        "policy_name": "Pi_05", "protocol": "ws", "task_name": "plate",
        "env_cfg_type": "tianji_dual", "action_type": "ee",
        "observation_profile": PROFILE, "train_config_name": TRAIN_CONFIG,
        "model_state_encoding": "zero_pose", "action_horizon": 32,
        "output_format": "xpolicylab",
        "action_semantics": "absolute_per_arm_base_xyz_wxyz",
        "checkpoint_source": SOURCE, "checkpoint_num": 59999,
        "checkpoint_variant": "pi05_pack_plate_wrist_only_step59999",
        "model_path": str(checkpoint), "norm_stats_path": str(stats_dir),
        "norm_stats_sha256": digest, "norm_stats_source": "sha256_" + digest,
        "num_steps": 10,
    }
    contract = validate_artifacts(config)
    assert contract["contract_status"] == "ready"
    assert contract["sampling_modes"] == ["default", "rtc"]
    with pytest.raises(ValueError, match="normalization must come from"):
        validate_artifacts({**config, "norm_stats_path": str(tmp_path)})
    with pytest.raises(ValueError, match="checksum"):
        validate_artifacts({**config, "norm_stats_sha256": "bad"})
    with pytest.raises(ValueError, match="task_name"):
        validate_artifacts({**config, "task_name": "pass_ball"})


def _anchor() -> np.ndarray:
    anchor = np.zeros(20, dtype=np.float32)
    for offset in (0, 10):
        anchor[offset + 3 : offset + 9] = rotation_to_6d(OPTICAL_FROM_FLU.T)
        anchor[offset + 9] = 0.5
    return anchor


def test_pack_plate_rtc_converts_absolute_condition_to_training_layout():
    anchor = _anchor()
    condition = np.zeros((32, 16), dtype=np.float32)
    condition[:2, 3] = condition[:2, 11] = 1.0
    condition[0, :3] = (0.1, 0.2, 0.3)
    condition[0, 8:11] = (-0.1, -0.2, -0.3)
    condition[0, 7] = 0.25
    condition[0, 15] = 0.75
    condition[1] = condition[0]
    weights = np.zeros(32, dtype=np.float32)
    weights[:2] = (1.0, 0.4)

    native = encode_rtc_condition(condition, weights, anchor)

    assert native.shape == (32, 20)
    np.testing.assert_allclose(native[0, :3], (-0.1, -0.2, -0.3))
    np.testing.assert_allclose(native[0, 10:13], (0.1, 0.2, 0.3))
    np.testing.assert_allclose(native[0, [9, 19]], (0.75, 0.25))
    np.testing.assert_allclose(native[0, 3:9], rotation_to_6d(OPTICAL_FROM_FLU.T))
    np.testing.assert_allclose(native[2:], np.repeat(anchor[None, :], 30, axis=0))
    relative = relative_action_chunk(anchor, native)
    np.testing.assert_allclose(relative_action_chunk(anchor, relative, inverse=True), native)
    np.testing.assert_allclose(relative[2:, :3], 0.0)
    transformed = PassBallInputs()(
        {
            "state": anchor,
            "actions": native,
            "images": {
                "left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
                "right_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
            },
            "prompt": "plate",
        }
    )
    np.testing.assert_allclose(transformed["actions"], relative)


def test_pack_plate_rtc_calls_real_sampler_hook_with_native_condition():
    anchor = _anchor()
    calls = []

    class FakePolicy:
        def infer(self, observation, **kwargs):
            calls.append((observation, kwargs))
            return {"actions": relative_action_chunk(anchor, np.tile(anchor, (32, 1)))}

    model = object.__new__(PackPlateZeroPoseModel)
    model.policy = FakePolicy()
    model.num_steps = 10
    model.observations = {0: {"state": anchor}}
    model.env_indices = [0]
    condition = np.zeros((32, 16), dtype=np.float32)
    condition[0, [3, 11]] = 1.0
    condition[0, [7, 15]] = (0.3, 0.7)
    weights = np.zeros(32, dtype=np.float32)
    weights[0] = 1.0

    actions = model.get_action_rtc(
        {"action_condition": condition, "condition_weights": weights, "beta": 5.0}
    )

    assert model.sampling_modes() == ["default", "rtc"]
    assert len(calls) == 1
    assert calls[0][1]["action_condition"].shape == (32, 20)
    np.testing.assert_allclose(calls[0][1]["condition_weights"], weights)
    assert calls[0][1]["rtc_beta"] == 5.0
    assert len(actions) == 32
    assert set(actions[0]) == {
        "left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state"
    }


def test_pack_plate_rtc_rejects_bad_condition_and_beta():
    anchor = _anchor()
    with pytest.raises(ValueError, match="requires \\(32,16\\)"):
        encode_rtc_condition(np.zeros((31, 16)), np.zeros(32), anchor)
    model = object.__new__(PackPlateZeroPoseModel)
    model.observations = {0: {"state": anchor}}
    model.env_indices = [0]
    with pytest.raises(ValueError, match="beta"):
        model.get_action_rtc(
            {"action_condition": np.zeros((32, 16)), "condition_weights": np.zeros(32), "beta": 0}
        )
