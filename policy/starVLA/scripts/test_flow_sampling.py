"""Numerical sampler checks on real small action heads, without VLM weights."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "source_starvla"))
    return import_module("starVLA.model.modules.action_model.runtime_sampling")


def test_paint_inverts_constant_velocity_and_preserves_free_suffix(modules):
    noise = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    condition = torch.zeros_like(noise)
    result = modules.integrate(
        lambda x, t: torch.ones_like(x),
        noise,
        4,
        dict(mode="paint", action_condition=condition, delay_steps=2),
    )
    torch.testing.assert_close(result["actions"][:, :2], condition[:, :2])
    torch.testing.assert_close(result["actions"][:, 2:], noise[:, 2:] + 1)
    assert result["paint"]["model_evaluations"] == 12


def test_rtc_vjp_has_the_correct_sign_and_zero_weights_are_neutral(modules):
    noise = torch.zeros(1, 4, 3)
    parameters = dict(
        mode="rtc",
        action_condition=torch.ones_like(noise),
        condition_weights=torch.ones(1, 4, 1),
        beta=1.0,
    )
    guided = modules.integrate(lambda x, t: x * 0, noise, 4, parameters)["actions"]
    torch.testing.assert_close(guided, torch.full_like(noise, 1 - 0.75**4))
    parameters["condition_weights"].zero_()
    neutral = modules.integrate(lambda x, t: x * 0, noise, 4, parameters)["actions"]
    torch.testing.assert_close(neutral, noise)


def test_dvac_uses_clean_estimates_and_population_variance(modules):
    noise = torch.ones(1, 4, 3)
    result = modules.integrate(lambda x, t: x * 0 + t, noise, 4, dict(mode="dvac", tail_steps=3))
    x, estimates = 1.0, []
    for step in range(4):
        t = step / 4
        estimates.append(x + (1 - t) * t)
        x += 0.25 * t
    np.testing.assert_allclose(result["dvac"]["variance"], np.var(estimates[-3:]) * 3)


@pytest.fixture(params=[False, True], ids=["groot", "pi_v3"])
def head(request, modules, monkeypatch):
    layerwise = request.param
    module = import_module(
        "starVLA.model.modules.action_model."
        + ("LayerwiseFM_ActionHeader" if layerwise else "GR00T_ActionHeader")
    )
    if not layerwise:
        monkeypatch.setitem(
            module.DiTConfig,
            "tiny",
            dict(input_embedding_dim=32, attention_head_dim=8, num_attention_heads=4),
        )
    action = dict(
        action_model_type="tiny",
        input_embedding_dim=32,
        hidden_size=32,
        action_dim=7,
        state_dim=7,
        action_horizon=8,
        num_inference_timesteps=4,
        num_target_vision_tokens=2,
        add_pos_embed=True,
        max_seq_len=32,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        diffusion_model_cfg=dict(
            input_embedding_dim=32,
            num_attention_heads=4,
            attention_head_dim=8,
            cross_attention_dim=32,
            num_layers=2,
            output_dim=32,
            dropout=0.0,
            final_dropout=False,
            interleave_self_attention=True,
        ),
    )
    config = OmegaConf.create(dict(framework=dict(action_model=action)))
    model = (
        module.LayerwiseFlowmatchingActionHead(config)
        if layerwise
        else module.FlowmatchingActionHead(config)
    ).eval()
    features = [torch.randn(1, 5, 32) for _ in range(2)] if layerwise else torch.randn(1, 5, 32)
    return model, features, layerwise


def test_opt_in_sampler_matches_original_euler_head(head):
    model, features, _ = head
    state = torch.randn(1, 1, 7)
    mask = torch.ones(1, 5, dtype=torch.bool)
    torch.manual_seed(123)
    expected = model.predict_action(features, state, encoder_attention_mask=mask)
    torch.manual_seed(123)
    actual = model.predict_action(
        features, state, encoder_attention_mask=mask, sampling=dict(mode="default")
    )["actions"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["rtc", "paint", "aac", "dvac", "autohorizon"])
def test_real_head_hooks_work_inside_framework_inference_mode(head, mode):
    model, features, _ = head
    parameters = {"mode": mode}
    if mode in {"rtc", "paint"}:
        parameters["action_condition"] = np.zeros((8, 7), dtype=np.float32)
    if mode == "rtc":
        parameters.update(condition_weights=np.ones(8, dtype=np.float32), beta=1.0)
    if mode == "paint":
        parameters["delay_steps"] = 2
    if mode == "aac":
        parameters["num_samples"] = 3
    if mode == "dvac":
        parameters["tail_steps"] = 3
    with torch.inference_mode():
        result = model.predict_action(
            features,
            torch.zeros(1, 1, 7),
            encoder_attention_mask=torch.ones(1, 5, dtype=torch.bool),
            sampling=parameters,
        )
    assert result["actions"].shape == (3 if mode == "aac" else 1, 8, 7)
    assert torch.isfinite(result["actions"]).all()
    assert all(parameter.grad is None for parameter in model.parameters())
    if mode == "aac":
        assert not torch.equal(result["actions"][0], result["actions"][1])
    if mode == "autohorizon":
        assert 1 <= result[mode]["execution_steps"] <= 8
        assert all(not module._forward_pre_hooks for module in model.modules())
