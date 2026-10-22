"""Continuation flow matching, using OpenPI time sigma=1 (noise) -> 0 (data).

References are pinned in the policy README. omega denotes the fixed prefix;
Legato's published code calls its complement w. No inference-time VJP is used.
"""
import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ContinuationConfig:
    method: Literal['none', 'train_time_rtc', 'legato'] = 'none'
    prefix_min_steps: int = 0
    prefix_max_steps: int = 6  # Inclusive, unlike RTC's exclusive simulated_delay.
    prefix_distribution: Literal['exp', 'bell', 'uniform'] = 'exp'
    num_denoise_steps: int = 10
    time_distribution: Literal['native_beta', 'uniform'] = 'native_beta'
    rtc_loss_reduction: Literal['native_mean', 'official_token_sum'] = 'native_mean'

    def validate(self, horizon, pi05=True):
        if self.method not in {'none', 'train_time_rtc', 'legato'}:
            raise ValueError('Unknown continuation method')
        if self.method == 'none':
            return
        if not pi05 or not 0 <= self.prefix_min_steps <= self.prefix_max_steps < horizon:
            raise ValueError('Continuation requires Pi05 and 0 <= min <= max < horizon')
        if self.num_denoise_steps < 1:
            raise ValueError('num_denoise_steps must be positive')
        if self.prefix_distribution not in {'exp', 'bell', 'uniform'}:
            raise ValueError('Unknown prefix distribution')
        if self.time_distribution not in {'native_beta', 'uniform'}:
            raise ValueError('Unknown time distribution')
        if self.rtc_loss_reduction not in {'native_mean', 'official_token_sum'}:
            raise ValueError('Unknown RTC loss reduction')


def sample_prefix(rng, batch_size, horizon, config):
    lo, hi = config.prefix_min_steps, config.prefix_max_steps
    if config.prefix_distribution == 'exp':
        candidates = jnp.arange(lo, hi + 1)
        lengths = jax.random.choice(rng, candidates, (batch_size,),
                                    p=jax.nn.softmax(-(candidates - lo).astype(jnp.float32)))
    elif config.prefix_distribution == 'bell':
        # Official Legato bell distribution with peak=warmup_min: Beta(1, 4).
        values = jax.random.beta(rng, 1., 4., (batch_size,))
        lengths = jnp.clip(jnp.floor(lo + values * (hi - lo + 1)).astype(jnp.int32), lo, hi)
    else:
        lengths = jax.random.randint(rng, (batch_size,), lo, hi + 1)
    return (jnp.arange(horizon)[None, :] < lengths[:, None]).astype(jnp.float32)[..., None]


def training_inputs(actions, noise, sigma, omega, config):
    """Return model action input, time condition and velocity target."""
    s = sigma[:, None, None]
    x = s * noise + (1 - s) * actions
    velocity = noise - actions
    if config.method == 'none':
        return x, sigma, velocity
    guided = omega * actions + (1 - omega) * x
    if config.method == 'train_time_rtc':
        return guided, jnp.where(omega[..., 0] > 0, 0., sigma[:, None]), velocity
    # Published Legato code's PLUS target, after reversing the flow time.
    target = velocity * (1 + omega * config.num_denoise_steps * s)
    return jnp.concatenate([guided, 1 - omega], axis=-1), sigma, target


def loss_per_token(prediction, target, omega, config):
    error = jnp.square(prediction - target)
    loss = jnp.mean(error, axis=-1)
    if config.method == 'train_time_rtc':
        valid = 1 - omega[..., 0]
        # Native trainer subsequently calls .mean(). Correct for masked tokens.
        loss = loss * valid * valid.size / jnp.maximum(jnp.sum(valid), 1)
        if config.rtc_loss_reduction == 'official_token_sum':
            loss = loss * error.shape[-1]
    return loss


def sampling_input(x, target, omega, sigma, config):
    guided = omega * target + (1 - omega) * x
    time = jnp.broadcast_to(sigma, (x.shape[0],))
    if config.method == 'train_time_rtc':
        time = jnp.where(omega[..., 0] > 0, 0., time[:, None])
        return guided, guided, time
    if config.method == 'legato':
        return guided, jnp.concatenate([guided, 1 - omega], axis=-1), time
    return x, x, time


def integrate(predict_velocity, noise, target, omega, config, num_steps):
    """Uniform Euler updates from the guided state, as in both public samplers."""
    if num_steps < 1:
        raise ValueError('num_steps must be positive')
    if config.method == 'legato' and num_steps != config.num_denoise_steps:
        raise ValueError('Legato inference grid must match its training target')
    def step(index, x):
        sigma = 1. - index / num_steps
        guided, model_input, time = sampling_input(x, target, omega, sigma, config)
        return guided - predict_velocity(model_input, time) / num_steps
    return jax.lax.fori_loop(0, num_steps, step, noise)
