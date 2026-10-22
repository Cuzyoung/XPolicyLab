# ruff: noqa: SLF001

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


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
