from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def gr00t_model_module(monkeypatch: pytest.MonkeyPatch):
    gr00t = ModuleType("gr00t")
    gr00t_data = ModuleType("gr00t.data")
    embodiment_tags = ModuleType("gr00t.data.embodiment_tags")
    gr00t_policy = ModuleType("gr00t.policy")
    embodiment_tags.EmbodimentTag = type("EmbodimentTag", (), {})
    gr00t_policy.Gr00tPolicy = type("Gr00tPolicy", (), {})
    for name, module in {
        "gr00t": gr00t,
        "gr00t.data": gr00t_data,
        "gr00t.data.embodiment_tags": embodiment_tags,
        "gr00t.policy": gr00t_policy,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).parents[2] / "policy/GR00T_N17/model.py"
    spec = importlib.util.spec_from_file_location("gr00t_yam_adapter_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yam_observation_uses_checkpoint_modality_keys(gr00t_model_module) -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    observation = {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image + 1},
            "cam_right_wrist": {"color": image + 2},
        },
        "state": {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": np.array([0.1], dtype=np.float32),
            "right_arm_joint_state": np.arange(6, dtype=np.float32) + 10,
            "right_ee_joint_state": np.array([0.9], dtype=np.float32),
        },
        "instruction": "pick the block",
    }

    encoded = gr00t_model_module._encode_observation(
        observation,
        "fallback",
        "yam_bimanual",
        (6, 6),
        (1, 1),
    )

    assert tuple(encoded["video"]) == (
        "base_view",
        "left_wrist_view",
        "right_wrist_view",
    )
    assert np.array_equal(encoded["video"]["base_view"][0, 0], image)
    assert tuple(encoded["state"]) == (
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
    )
    assert encoded["state"]["left_arm"].shape == (1, 1, 6)
    assert encoded["state"]["left_gripper"].shape == (1, 1, 1)


def test_yam_absolute_actions_decode_to_xpolicy_keys(gr00t_model_module) -> None:
    horizon = 16
    action = {
        "left_arm": np.zeros((1, horizon, 6), dtype=np.float32),
        "left_gripper": np.full((1, horizon, 1), 0.2, dtype=np.float32),
        "right_arm": np.ones((1, horizon, 6), dtype=np.float32),
        "right_gripper": np.full((1, horizon, 1), 0.8, dtype=np.float32),
    }

    decoded = gr00t_model_module._gr00t_action_to_env(
        action,
        "joint",
        "yam_bimanual",
        (6, 6),
        (1, 1),
    )

    assert len(decoded) == horizon
    assert decoded[0]["left_arm_joint_state"].shape == (6,)
    assert decoded[0]["left_ee_joint_state"].tolist() == pytest.approx([0.2])
    assert decoded[0]["right_arm_joint_state"].tolist() == pytest.approx([1.0] * 6)
    assert decoded[0]["right_ee_joint_state"].tolist() == pytest.approx([0.8])


def test_consolidated_checkpoint_root_is_accepted(
    gr00t_model_module, tmp_path: Path
) -> None:
    root = tmp_path / "gr00t-yam"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    assert gr00t_model_module._resolve_checkpoint_dir({"model_dir": str(root)}) == root
