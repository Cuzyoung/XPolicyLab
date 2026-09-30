# ruff: noqa: SLF001

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _noise_policy(monkeypatch, backend, *, rng=None):
    class NoiseModel:
        def __init__(self):
            if backend == "torch":
                self.config = SimpleNamespace(action_horizon=3, action_dim=2)
            else:
                self.action_horizon = 3
                self.action_dim = 2

        def to(self, _device):
            return self

        def eval(self):
            return None

        def sample_actions(self, key_or_device, observation, *, noise=None, **_kwargs):
            if backend == "torch":
                assert noise is not None, "seeded PyTorch inference must supply the sampler noise"
                assert noise.shape == (observation.state.shape[0], self.config.action_horizon, self.config.action_dim)
                assert noise.dtype == torch.float32
                assert noise.device == observation.state.device
                return noise
            if noise is not None:
                return noise
            return jax.random.normal(key_or_device, (observation.state.shape[0], 3, 2))

    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method, **_kwargs: method)
    return _policy.Policy(NoiseModel(), rng=rng, is_pytorch=backend == "torch", pytorch_device="cpu")


def _noise_observation():
    return {
        "image": {"base": np.zeros((2, 2, 3), dtype=np.uint8)},
        "image_mask": {"base": np.ones((), dtype=np.bool_)},
        "state": np.zeros(2, dtype=np.float32),
    }


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_seeded_inference_advances_and_reset_repeats_the_sequence(monkeypatch, backend):
    policy = _noise_policy(monkeypatch, backend)
    policy.reset_rng(17)
    first = policy.infer(_noise_observation())["actions"]
    # An unrelated global Torch RNG consumer must not perturb this policy's stream.
    with torch.random.fork_rng(devices=[]):
        torch.rand(19)
        second = policy.infer(_noise_observation())["actions"]
    assert not np.array_equal(first, second)

    policy.reset_rng(17)
    np.testing.assert_array_equal(policy.infer(_noise_observation())["actions"], first)
    np.testing.assert_array_equal(policy.infer(_noise_observation())["actions"], second)
    policy.reset_rng(23)
    assert not np.array_equal(policy.infer(_noise_observation())["actions"], first)


def test_jax_constructor_preserves_an_explicit_array_key(monkeypatch):
    key = jax.random.key(7)
    policy = _noise_policy(monkeypatch, "jax", rng=key)
    _, sample_key = jax.random.split(key)
    expected = np.asarray(jax.random.normal(sample_key, (1, 3, 2)))[0]
    np.testing.assert_array_equal(policy.infer(_noise_observation())["actions"], expected)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_explicit_noise_overrides_seeded_noise(monkeypatch, backend):
    policy = _noise_policy(monkeypatch, backend)
    policy.reset_rng(17)
    noise = np.arange(6, dtype=np.float32).reshape(3, 2)
    np.testing.assert_array_equal(policy.infer(_noise_observation(), noise=noise)["actions"], noise)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


def test_infer_preserves_multi_sample_axis_for_output_transforms():
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = type("FakeModel", (), {"action_horizon": 3, "action_dim": 2})()
    policy._input_transform = lambda values: values
    policy._output_transform = lambda values: values
    policy._sample_kwargs = {}
    policy._is_pytorch_model = False
    policy._rng = jax.random.key(0)
    calls = []

    def sample_actions(_rng, _observation, **kwargs):
        calls.append(kwargs)
        return jnp.zeros((kwargs["num_samples"], 3, 2), dtype=jnp.float32)

    policy._sample_actions = lambda *_args, **_kwargs: None
    policy._sample_actions_multi = sample_actions
    observation = {
        "image": {"base": np.zeros((2, 2, 3), dtype=np.uint8)},
        "image_mask": {"base": np.ones((), dtype=np.bool_)},
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
    }

    result = policy.infer(observation, num_samples=4)

    assert result["actions"].shape == (4, 3, 2)
    assert result["state"].shape == (4, 2)
    assert calls == [{"num_samples": 4}]


def test_infer_passes_transformed_paint_condition_to_jax_sampler():
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = type("FakeModel", (), {"action_horizon": 3, "action_dim": 2})()
    policy._input_transform = lambda values: values
    policy._output_transform = lambda values: values
    policy._sample_kwargs = {}
    policy._is_pytorch_model = False
    policy._rng = jax.random.key(0)
    calls = []

    def sample_actions(_rng, _observation, **kwargs):
        calls.append(kwargs)
        return jnp.zeros((1, 3, 2), dtype=jnp.float32)

    policy._sample_actions = sample_actions
    policy._sample_actions_multi = lambda *_args, **_kwargs: None
    observation = {
        "image": {"base": np.zeros((2, 2, 3), dtype=np.uint8)},
        "image_mask": {"base": np.ones((), dtype=np.bool_)},
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
    }
    condition = np.arange(6, dtype=np.float32).reshape(3, 2)

    result = policy.infer(
        observation,
        paint_action_condition=condition,
        paint_delay_steps=1,
    )

    assert result["actions"].shape == (3, 2)
    assert len(calls) == 1
    assert calls[0]["paint_delay_steps"] == 1
    np.testing.assert_array_equal(
        np.asarray(calls[0]["paint_action_condition"]),
        condition[None, ...],
    )
