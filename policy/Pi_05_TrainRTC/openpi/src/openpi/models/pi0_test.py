import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_repeat_sample_batch_reuses_one_prefix_cache():
    observation = _model.Observation(
        images={"base": jnp.arange(12, dtype=jnp.float32).reshape(1, 2, 2, 3)},
        image_masks={"base": jnp.ones((1,), dtype=jnp.bool_)},
        state=jnp.arange(4, dtype=jnp.float32).reshape(1, 4),
    )
    prefix_mask = jnp.asarray([[True, False, True]])
    kv_cache = (
        jnp.arange(24, dtype=jnp.float32).reshape(2, 1, 3, 2, 2),
        jnp.arange(24, 48, dtype=jnp.float32).reshape(2, 1, 3, 2, 2),
    )

    repeated_observation, repeated_mask, repeated_cache = pi0._repeat_sample_batch(  # noqa: SLF001
        observation,
        prefix_mask,
        kv_cache,
        3,
    )

    assert repeated_observation.state.shape == (3, 4)
    assert repeated_observation.images["base"].shape == (3, 2, 2, 3)
    assert repeated_mask.shape == (3, 3)
    assert repeated_cache[0].shape == (2, 3, 3, 2, 2)
    assert jnp.array_equal(repeated_cache[0][:, 0], kv_cache[0][:, 0])
    assert jnp.array_equal(repeated_cache[0][:, 2], kv_cache[0][:, 0])


def test_paint_euler_recovers_prefix_and_preserves_naive_suffix():
    free_noise = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    action_condition = jnp.full_like(free_noise, -7.0)
    velocity = jnp.full_like(free_noise, 2.0)

    actions = pi0._paint_euler_sample(  # noqa: SLF001
        lambda _sample, _time: velocity,
        free_noise,
        action_condition,
        delay_steps=2,
        num_steps=5,
    )

    np_actions = np.asarray(actions)
    np.testing.assert_allclose(np_actions[:, :2], -7.0, atol=1e-6)
    np.testing.assert_allclose(
        np_actions[:, 2:],
        np.asarray(free_noise - velocity)[:, 2:],
        atol=3e-6,
    )
