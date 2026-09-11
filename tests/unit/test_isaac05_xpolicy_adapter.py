from __future__ import annotations

import numpy as np
import pytest

from XPolicyLab.policy.Isaac_05.model import (
    _weight_package_status,
    decode_actions,
    encode_observation,
)


def _wire_observation() -> dict[str, object]:
    return {
        "vision": {
            "primary": {"color": np.zeros((8, 10, 3), dtype=np.uint8)},
            "wrist": {"color": np.ones((8, 10, 3), dtype=np.uint8)},
        },
        "state": {"observation.state": np.arange(8, dtype=np.float32)},
        "instruction": "Pick up the object.",
        "additional_info": {"timestep": 24},
    }


def test_encode_observation_uses_official_lerobot_keys() -> None:
    encoded, instruction, timestep = encode_observation(
        _wire_observation(),
        camera_map={"image": "primary", "wrist_image": "wrist"},
        state_keys=("unused",),
        proprio_dim=8,
        default_prompt="fallback",
    )
    assert set(encoded) == {
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.state",
    }
    assert encoded["observation.state"].shape == (8,)
    assert instruction == "Pick up the object."
    assert timestep == 24


def test_encode_observation_requires_exact_camera_roles() -> None:
    with pytest.raises(ValueError, match="camera_map keys"):
        encode_observation(
            _wire_observation(),
            camera_map={"image": "primary"},
            state_keys=("unused",),
            proprio_dim=8,
            default_prompt="fallback",
        )


def test_decode_actions_keeps_eight_by_seven_contract() -> None:
    actions = np.arange(56, dtype=np.float32).reshape(8, 7)
    decoded = decode_actions(actions, action_dim=7, horizon=8)
    assert len(decoded) == 8
    np.testing.assert_allclose(decoded[3]["action"], actions[3])


def test_weight_package_requires_every_indexed_shard(tmp_path) -> None:
    assert _weight_package_status(tmp_path) == (False, None)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    assert _weight_package_status(tmp_path) == (False, 1)
    (tmp_path / "model-00002-of-00002.safetensors").touch()
    assert _weight_package_status(tmp_path) == (True, 0)
