from __future__ import annotations

from types import SimpleNamespace

import pytest

from XPolicyLab.policy.Pi_05 import model as pi05


@pytest.mark.parametrize("seed", [None, 0, 17])
def test_model_applies_deployment_seed_at_load_and_episode_reset(monkeypatch, tmp_path, seed):
    resets = []
    policy = SimpleNamespace(reset_rng=resets.append)

    def load_policy(model, model_cfg):
        model.model_root = tmp_path
        model.norm_stats_path = None
        return policy

    monkeypatch.setattr(
        pi05,
        "_resolve_train_config",
        lambda _cfg: SimpleNamespace(
            name="test_pi05", data=None, model=SimpleNamespace(action_horizon=3, action_dim=2)
        ),
    )
    monkeypatch.setattr(pi05.Model, "get_model", load_policy)
    config = {"task_name": "pick", "seed": 99}
    if seed is not None:
        config["inference_seed"] = seed
    expected_seed = 0 if seed is None else seed
    model = pi05.Model(config)
    assert resets == [expected_seed]
    assert model.runtime_metadata()["inference_seed"] == expected_seed

    model.observation_window = {"previous": True}
    model.reset()
    assert resets == [expected_seed, expected_seed]
    assert model.observation_window is None


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "0", None])
def test_invalid_inference_seed_is_rejected_before_checkpoint_loading(monkeypatch, seed):
    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("invalid seeds must fail before loading model assets")

    monkeypatch.setattr(pi05.Model, "get_model", unexpected_load)
    with pytest.raises(ValueError, match="inference_seed"):
        pi05.Model({"task_name": "pick", "inference_seed": seed})
