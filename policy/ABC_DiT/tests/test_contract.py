"""CPU contract checks with a tiny random ABC-DiT checkpoint (no 8 GB download needed)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from XPolicyLab.policy.ABC_DiT.model import Model, _ensure_upstream_on_path, letterbox

CLIP_DIR = Path.home() / ".cache" / "clip"
needs_clip = pytest.mark.skipif(
    not (CLIP_DIR / "ViT-B-32.pt").is_file(), reason="CLIP text assets not cached"
)


def test_letterbox_matches_ffmpeg_geometry():
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    out = letterbox(image)
    assert out.shape == (3, 224, 224) and out.dtype == np.uint8
    # 640x480 -> 224x168, centered: 28 zero rows above and below.
    assert not out[:, :28].any() and not out[:, 196:].any()
    assert (out[:, 28:196] == 200).all()
    square = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    assert np.array_equal(letterbox(square), square.transpose(2, 0, 1))


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    _ensure_upstream_on_path()
    from abc_minimal.config import DiTConfig
    from abc_minimal.dit import DiTPolicy

    torch.manual_seed(0)
    model = DiTPolicy(DiTConfig(hidden_size=128, depth=2, num_heads=2))
    stats = {"mean": [0.1] * 14, "std": [0.5] * 14}
    path = tmp_path_factory.mktemp("abc") / "tiny.pt"
    torch.save(
        {"model": model.state_dict(), "step": 7, "norm_stats": {"state": stats, "actions": stats}},
        path,
    )
    return path


def config(path, **kwargs):
    return {
        "task_name": "put_the_plastic_bottles_in_the_bin",
        "robot_action_dim_info": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
        "action_type": "joint",
        "model_path": str(path),
        "device": "cpu",
        "num_steps": 2,
        **kwargs,
    }


def observation(value=0.2):
    rng = np.random.default_rng(1)
    return {
        "vision": {
            name: {"color": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)}
            for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
        "state": {
            "left_arm_joint_state": np.full(6, value, np.float32),
            "left_ee_joint_state": np.array([1.0], np.float32),
            "right_arm_joint_state": np.full(6, -value, np.float32),
            "right_ee_joint_state": np.array([0.0], np.float32),
        },
        "instruction": "put the plastic bottles in the bin",
    }


@needs_clip
def test_actions_are_seeded_absolute_joint_chunks(tiny_checkpoint):
    model = Model(config(tiny_checkpoint))
    model.update_obs(observation())
    first = model.get_action()
    assert len(first) == 30
    assert {k: v.shape for k, v in first[0].items()} == {
        "left_arm_joint_state": (6,),
        "left_ee_joint_state": (1,),
        "right_arm_joint_state": (6,),
        "right_ee_joint_state": (1,),
    }
    grippers = np.array([[a["left_ee_joint_state"][0], a["right_ee_joint_state"][0]] for a in first])
    assert ((grippers >= 0) & (grippers <= 1)).all()
    second = model.get_action()
    model.reset()
    model.update_obs(observation())
    again = model.get_action()
    assert all(np.array_equal(a[k], b[k]) for a, b in zip(first, again) for k in a)
    assert not all(np.array_equal(a[k], b[k]) for a, b in zip(first, second) for k in a)
    metadata = model.runtime_metadata()
    assert metadata["checkpoint_step"] == 7 and metadata["action_horizon"] == 30


@needs_clip
def test_rtc_validates_and_returns_a_chunk(tiny_checkpoint):
    model = Model(config(tiny_checkpoint))
    model.update_obs(observation())
    condition = np.zeros((30, 14), np.float32)
    weights = np.linspace(1, 0, 30, dtype=np.float32)
    actions = model.get_action_rtc({"action_condition": condition, "condition_weights": weights, "beta": 5.0})
    assert len(actions) == 30
    with pytest.raises(ValueError, match="action_condition"):
        model.get_action_rtc({"action_condition": condition[:10], "condition_weights": weights, "beta": 5.0})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.get_action_rtc({"action_condition": condition, "condition_weights": weights + 1, "beta": 5.0})


@needs_clip
def test_paint_uses_abc_prefix_conditioning(tiny_checkpoint):
    model = Model(config(tiny_checkpoint))
    model.update_obs(observation())
    prefix = np.random.default_rng(2).normal(size=(7, 14)).astype(np.float32)
    prefix[:, [6, 13]] = 0.5
    result = model.get_action_paint({"action_prefix": prefix, "delay_steps": 7})
    actions = np.stack([np.concatenate([a["left_arm_joint_state"], a["left_ee_joint_state"],
                                        a["right_arm_joint_state"], a["right_ee_joint_state"]])
                        for a in result["actions"]])
    assert actions.shape == (30, 14)
    assert np.allclose(actions[:7], prefix)
    assert result["paint"]["prefix_length"] == 4 and result["paint"]["method"] == "abc_action_prefix"
    with pytest.raises(ValueError, match="action_prefix"):
        model.get_action_paint({"action_prefix": prefix[:3], "delay_steps": 7})


@needs_clip
def test_rejects_mismatched_contracts(tiny_checkpoint):
    with pytest.raises(ValueError, match="action_type"):
        Model(config(tiny_checkpoint, action_type="ee"))
    with pytest.raises(ValueError, match="camera_map"):
        Model(config(tiny_checkpoint, camera_map={"top": "cam_head"}))
    with pytest.raises(ValueError, match="robot_action_dim_info"):
        Model(config(tiny_checkpoint, robot_action_dim_info={"arm_dim": [6], "ee_dim": [1]}))
