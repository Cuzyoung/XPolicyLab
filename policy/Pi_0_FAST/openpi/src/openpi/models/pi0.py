import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
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
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
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
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
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
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
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
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

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

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        num_samples: int = 1,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_condition: at.Float[at.Array, "b ah ad"] | None = None,
        condition_weights: at.Float[at.Array, "b ah 1"] | None = None,
        rtc_beta: float = 5.0,
        paint_action_condition: at.Float[at.Array, "b ah ad"] | None = None,
        paint_delay_steps: int | at.Int[at.Array, ""] | None = None,
        return_attention: bool = False,
        return_denoising_variance: bool = False,
        dvac_tail_steps: int = 5,
    ):
        if (action_condition is None) != (condition_weights is None):
            raise ValueError("action_condition and condition_weights must be provided together")
        if (paint_action_condition is None) != (paint_delay_steps is None):
            raise ValueError(
                "paint_action_condition and paint_delay_steps must be provided together"
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_samples > 1 and action_condition is not None:
            raise ValueError("multi-sample inference cannot be combined with RTC conditioning")
        if num_samples > 1 and paint_action_condition is not None:
            raise ValueError("multi-sample inference cannot be combined with PAINT")
        if action_condition is not None and paint_action_condition is not None:
            raise ValueError("RTC conditioning cannot be combined with PAINT")
        if return_attention and num_samples > 1:
            raise ValueError("AutoHorizon attention requires one action sample")
        if return_attention and action_condition is not None:
            raise ValueError("AutoHorizon attention cannot be combined with RTC conditioning")
        if return_attention and paint_action_condition is not None:
            raise ValueError("AutoHorizon attention cannot be combined with PAINT")
        if return_denoising_variance and num_samples > 1:
            raise ValueError("DVAC requires one action sample")
        if return_denoising_variance and action_condition is not None:
            raise ValueError("DVAC cannot be combined with RTC conditioning")
        if return_denoising_variance and paint_action_condition is not None:
            raise ValueError("DVAC cannot be combined with PAINT")
        if return_denoising_variance and return_attention:
            raise ValueError("DVAC cannot be combined with AutoHorizon attention")
        if dvac_tail_steps <= 0:
            raise ValueError("DVAC tail steps must be positive")
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        input_batch_size = observation.state.shape[0]
        if num_samples > 1 and input_batch_size != 1:
            raise ValueError(
                "multi-sample inference requires one observation, got batch size "
                f"{input_batch_size}"
            )

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if num_samples > 1:
            observation, prefix_mask, kv_cache = _repeat_sample_batch(
                observation,
                prefix_mask,
                kv_cache,
                num_samples,
            )
        batch_size = input_batch_size * num_samples
        expected_noise_shape = (batch_size, self.action_horizon, self.action_dim)
        if noise is None:
            noise = jax.random.normal(rng, expected_noise_shape)
        elif noise.shape != expected_noise_shape:
            raise ValueError(
                f"noise must have shape {expected_noise_shape}, got {noise.shape}"
            )
        if (
            paint_action_condition is not None
            and paint_action_condition.shape != expected_noise_shape
        ):
            raise ValueError(
                "paint_action_condition must have shape "
                f"{expected_noise_shape}, got {paint_action_condition.shape}"
            )

        def predict_velocity(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            llm_outputs = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                return_attention=return_attention,
            )
            if return_attention:
                (prefix_out, suffix_out), _, attention = llm_outputs
            else:
                (prefix_out, suffix_out), _ = llm_outputs
            assert prefix_out is None
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if not return_attention:
                return velocity
            action_attention = attention[..., -self.action_horizon :]
            action_attention = jnp.mean(action_attention, axis=(0, 1, 2)) / num_steps
            return velocity, action_attention

        def forward_step(carry):
            x_t, time = carry
            if action_condition is None:
                v_t = predict_velocity(x_t, time)
            else:

                def clean_estimate(sample):
                    velocity = predict_velocity(sample, time)
                    return sample - time * velocity, velocity

                clean, pullback, v_t = jax.vjp(clean_estimate, x_t, has_aux=True)
                weighted_error = (action_condition - clean) * condition_weights
                guidance = pullback(weighted_error)[0]
                tau = 1.0 - time
                denominator = jnp.maximum(time * tau, jnp.finfo(x_t.dtype).eps)
                raw_scale = (time**2 + tau**2) / denominator
                guidance_scale = jnp.minimum(jnp.asarray(rtc_beta), raw_scale)
                v_t = v_t - guidance_scale * guidance

            return x_t + dt * v_t, time + dt

        def forward_cond(carry):
            x_t, time = carry
            del x_t
            # robust to floating-point error
            return time >= -dt / 2

        def integrate_forward(initial_noise):
            if return_attention:
                def attention_step(carry):
                    x_t, time, step_index, selected_attention = carry
                    v_t, action_attention = predict_velocity(x_t, time)
                    selected_attention = jnp.where(
                        step_index == 2,
                        action_attention,
                        selected_attention,
                    )
                    return x_t + dt * v_t, time + dt, step_index + 1, selected_attention

                def attention_cond(carry):
                    _, time, _, _ = carry
                    return time >= -dt / 2

                actions, _, _, action_attention = jax.lax.while_loop(
                    attention_cond,
                    attention_step,
                    (
                        initial_noise,
                        1.0,
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.zeros(
                            (self.action_horizon, self.action_horizon),
                            dtype=prefix_tokens.dtype,
                        ),
                    ),
                )
                return actions, action_attention
            if return_denoising_variance:
                def dvac_step(carry):
                    x_t, time, clean_tail = carry
                    v_t = predict_velocity(x_t, time)
                    clean_estimate = x_t - time * v_t
                    clean_tail = jnp.roll(clean_tail, -1, axis=0)
                    clean_tail = clean_tail.at[-1].set(clean_estimate)
                    return x_t + dt * v_t, time + dt, clean_tail

                def dvac_cond(carry):
                    _, time, _ = carry
                    return time >= -dt / 2

                actions, _, clean_tail = jax.lax.while_loop(
                    dvac_cond,
                    dvac_step,
                    (
                        initial_noise,
                        1.0,
                        jnp.zeros(
                            (dvac_tail_steps, *initial_noise.shape),
                            dtype=initial_noise.dtype,
                        ),
                    ),
                )
                return actions, clean_tail
            actions, _ = jax.lax.while_loop(
                forward_cond,
                forward_step,
                (initial_noise, 1.0),
            )
            return actions

        if paint_action_condition is None:
            return integrate_forward(noise)

        return _paint_euler_sample(
            predict_velocity,
            noise,
            paint_action_condition,
            paint_delay_steps,
            num_steps,
        )
