import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import continuation as continuation_lib
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def _repeat_sample_batch(observation, prefix_mask, kv_cache, num_samples: int):
    observation = jax.tree.map(
        lambda value: jnp.repeat(value, num_samples, axis=0),
        observation,
    )
    prefix_mask = jnp.repeat(prefix_mask, num_samples, axis=0)
    kv_cache = jax.tree.map(
        lambda value: jnp.repeat(value, num_samples, axis=1),
        kv_cache,
    )
    return observation, prefix_mask, kv_cache


def _paint_euler_sample(
    predict_velocity,
    free_noise,
    action_condition,
    delay_steps,
    num_steps,
):
    """Algorithm 1 in the Pi0 time convention: noise at 1, actions at 0."""
    step_size = 1.0 / num_steps

    def forward_step(_index, carry):
        x_t, time = carry
        return x_t - step_size * predict_velocity(x_t, time), time - step_size

    def integrate_forward(initial_noise):
        actions, _ = jax.lax.fori_loop(
            0,
            num_steps,
            forward_step,
            (initial_noise, 1.0),
        )
        return actions

    naive_actions = integrate_forward(free_noise)
    paint_mask = (
        jnp.arange(free_noise.shape[1])[None, :, None]
        < jnp.asarray(delay_steps)
    )
    target_actions = jnp.where(paint_mask, action_condition, naive_actions)

    def inverse_step(_index, carry):
        x_t, time = carry
        return x_t + step_size * predict_velocity(x_t, time), time + step_size

    inverted_noise, _ = jax.lax.fori_loop(
        0,
        num_steps,
        inverse_step,
        (target_actions, 0.0),
    )
    repainted_noise = jnp.where(paint_mask, inverted_noise, free_noise)
    return integrate_forward(repainted_noise)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, "*batch"], embedding_dim: int, min_period: float, max_period: float) -> at.Float[at.Array, "*batch {embedding_dim}"]:
    if embedding_dim % 2:
        raise ValueError("embedding_dim must be even")
    fraction = jnp.linspace(0., 1., embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    phase = jnp.einsum("...i,j->...ij", pos, 1. / period * 2 * jnp.pi, precision=jax.lax.Precision.HIGHEST)
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.continuation = config.continuation
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim + int(config.continuation.method == "legato"), action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b ah"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b s emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        if self.continuation.method != "none" and self.continuation.time_distribution == "uniform":
            time = jax.random.uniform(time_rng, batch_shape)
        omega = continuation_lib.sample_prefix(
            jax.random.fold_in(rng, 173), actions.shape[0], self.action_horizon, self.continuation
        ) if self.continuation.method != "none" else jnp.zeros((*actions.shape[:2], 1))
        x_t, time, u_t = continuation_lib.training_inputs(actions, noise, time, omega, self.continuation)

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return continuation_lib.loss_per_token(v_t, u_t, omega, self.continuation)

    @override
    def sample_actions(self, rng, observation, *, num_steps=10, noise=None,
                       action_condition=None, condition_weights=None, rtc_beta=5.,
                       num_samples=1, return_attention=False):
        del rtc_beta  # Policy transport compatibility; no VJP guidance.
        if num_samples != 1 or return_attention:
            raise ValueError("Continuation supports one sample and no attention extraction")
        if (action_condition is None) != (condition_weights is None):
            raise ValueError("Condition and weights must be provided together")
        observation = _model.preprocess_observation(None, observation, train=False)
        batch = observation.state.shape[0]
        shape = (batch, self.action_horizon, self.action_dim)
        x = jax.random.normal(rng, shape) if noise is None else noise
        target = jnp.zeros(shape) if action_condition is None else action_condition
        omega = jnp.zeros((*shape[:2], 1)) if condition_weights is None else condition_weights
        if x.shape != shape or target.shape != shape or omega.shape != (*shape[:2], 1):
            raise ValueError("Continuation noise, target or weights have incompatible shapes")
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=make_attn_mask(prefix_mask, prefix_ar),
            positions=jnp.cumsum(prefix_mask, axis=1)-1)

        def velocity(model_input, time):
            tokens, mask, ar, cond = self.embed_suffix(observation, model_input, time)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=tokens.shape[1])
            full_mask = jnp.concatenate([prefix_attn, make_attn_mask(mask, ar)], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1)-1
            (_, output), _ = self.PaliGemma.llm(
                [None, tokens], mask=full_mask, positions=positions, kv_cache=cache,
                adarms_cond=[None, cond])
            return self.action_out_proj(output[:, -self.action_horizon:])
        return continuation_lib.integrate(velocity, x, target, omega, self.continuation, num_steps)
