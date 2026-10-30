"""Training-only continuation objectives in OpenWAM's noise-to-data convention.

OpenWAM uses sigma=1 for noise, sigma=0 for data and predicts noise-action.
The public Legato Kinetix implementation uses the opposite time/velocity
direction. Upstream 5caa033 corrects the target to
(1 - omega*N*sigma)*(noise-action) here. Historical reference names retain
their original meaning so old saved configurations are not reinterpreted.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ContinuationConfig:
    method: str = "none"
    prefix_min_steps: int = 0
    prefix_max_steps: int = 6
    prefix_distribution: str = "exponential_decay"
    schedule: str = "step"
    ramp_steps: int = 0
    reference: str = "official_20261020"
    num_denoise_steps: int = 10

    @classmethod
    def parse(cls, values=None):
        cfg = cls(**dict(values or {}))
        if cfg.method not in {"none", "train_time_rtc", "legato"}:
            raise ValueError(f"Unknown continuation method: {cfg.method}")
        for key in ("prefix_min_steps", "prefix_max_steps", "ramp_steps", "num_denoise_steps"):
            if type(getattr(cfg, key)) is not int:
                raise ValueError(f"continuation.{key} must be an integer")
        if not 0 <= cfg.prefix_min_steps <= cfg.prefix_max_steps:
            raise ValueError("Require 0 <= prefix_min_steps <= prefix_max_steps (inclusive)")
        if cfg.prefix_distribution not in {"uniform", "exponential_decay"}:
            raise ValueError("prefix_distribution must be uniform or exponential_decay")
        if cfg.reference not in {"official_20261020", "official_code", "paper_v2"}:
            raise ValueError("reference must be official_20261020, official_code (legacy PLUS), or paper_v2")
        if cfg.schedule not in {"step", "linear"} or cfg.ramp_steps < 0:
            raise ValueError("schedule must be step or linear; ramp_steps must be nonnegative")
        if cfg.schedule == "step" and cfg.ramp_steps != 0:
            raise ValueError("step schedule requires ramp_steps=0")
        if cfg.schedule == "linear" and cfg.ramp_steps == 0:
            raise ValueError("linear schedule requires ramp_steps>0")
        if cfg.method == "train_time_rtc" and cfg.schedule != "step":
            raise ValueError("Training-time RTC requires a hard prefix, not a ramp")
        if cfg.num_denoise_steps <= 0:
            raise ValueError("num_denoise_steps must be positive")
        return cfg

    def sample_lengths(self, batch_size, horizon, device):
        if self.prefix_max_steps + self.ramp_steps >= horizon:
            raise ValueError("prefix_max_steps + ramp_steps must leave at least one free action")
        candidates = torch.arange(self.prefix_min_steps, self.prefix_max_steps + 1, device=device)
        if self.prefix_distribution == "uniform":
            indexes = torch.randint(len(candidates), (batch_size,), device=device)
        else:
            probs = torch.softmax(-(candidates - self.prefix_min_steps).float(), dim=0)
            indexes = torch.multinomial(probs, batch_size, replacement=True)
        return candidates[indexes]


def configure_architecture(architecture, values=None):
    """Called by the factory before checkpoint loading / optimizer creation."""
    cfg = ContinuationConfig.parse(values)
    architecture.continuation = cfg
    if cfg.method == "none":
        return
    framework = architecture._cfg_get(architecture.cfg, "framework")
    variant = architecture._cfg_get(architecture.cfg, "variant")
    if framework != "dual_system" or variant not in {"joint_self_attn", "joint_cross_attn"}:
        raise ValueError("Continuation currently supports dual_system joint_self_attn/joint_cross_attn only")
    if cfg.method == "legato":
        # Nonuniform action grids need a different target/step-size contract.
        if float(architecture._cfg_get(architecture.cfg, "shift_action", 5.0)) != 1.0:
            raise ValueError("Legato requires model.action_backbone.shift_action=1.0 (uniform action grid)")
        old = architecture.action_backbone.action_encoder
        # Avoid consuming RNG or reinitializing any pretrained action columns.
        expanded = nn.Linear(old.in_features + 1, old.out_features,
                             bias=old.bias is not None, device="meta", dtype=old.weight.dtype)
        expanded.weight = nn.Parameter(torch.cat((old.weight.detach(), old.weight.new_zeros(old.out_features, 1)), dim=1))
        if old.bias is not None:
            expanded.bias = nn.Parameter(old.bias.detach().clone())
        architecture.action_backbone.action_encoder = expanded


def prepare_action_training(cfg, actions, noise, sigma, timesteps, noisy_actions, target,
                            action_is_pad=None, *, prefix_lengths=None):
    """Return model input, time condition, velocity target, and loss padding.

No-op mode returns the original objects and consumes no random numbers.
GT and dataset masks are never modified in place. Schedule is appended to
MODEL INPUT only; target/output/normalization/action_dim remain D-dimensional.
"""
    if cfg.method == "none":
        return noisy_actions, timesteps, target, action_is_pad
    batch, horizon, _ = actions.shape
    if prefix_lengths is None:
        prefix_lengths = cfg.sample_lengths(batch, horizon, actions.device)
    elif prefix_lengths.shape != (batch,):
        raise ValueError("prefix_lengths must have shape [B]")
    positions = torch.arange(horizon, device=actions.device)[None, :]
    prefix = positions < prefix_lengths[:, None]
    if cfg.method == "train_time_rtc":
        model_input = torch.where(prefix[..., None], actions, noisy_actions)
        token_time = torch.where(prefix, 0, timesteps[:, None])
        if action_is_pad is None:
            padding = prefix
        else:
            original = action_is_pad.to(device=actions.device, dtype=torch.bool)
            padding = original | (prefix[..., None] if original.ndim == 3 else prefix)
        return model_input, token_time, target, padding

    omega = prefix.to(actions.dtype)
    if cfg.schedule == "linear":
        # d full-guidance rows, then r strictly intermediate weights, then zero.
        ramp = (prefix_lengths[:, None] + cfg.ramp_steps - positions) / (cfg.ramp_steps + 1)
        omega = torch.where(prefix, 1, ramp.clamp(0, 1)).to(actions.dtype)
    omega = omega[..., None]
    guided = omega * actions + (1 - omega) * noisy_actions
    sign = 1.0 if cfg.reference == "official_code" else -1.0
    factor = 1 + sign * omega.float() * cfg.num_denoise_steps * sigma.reshape(batch, 1, 1).float()
    legato_target = factor * (noise.float() - actions.float())
    return torch.cat((guided, 1 - omega), dim=-1), timesteps, legato_target, action_is_pad


def expand_base_action_weight(state_dict, expected_weight):
    """Explicit warm-start migration of the ONE added Legato input column."""
    key = "action_backbone.action_encoder.weight"
    if key not in state_dict:
        raise RuntimeError(f"Base checkpoint is missing {key}")
    weight = state_dict[key]
    if weight.shape != (expected_weight.shape[0], expected_weight.shape[1] - 1):
        raise RuntimeError(f"Unexpected base action input shape {tuple(weight.shape)}")
    state_dict = dict(state_dict)
    state_dict[key] = torch.cat((weight, weight.new_zeros(weight.shape[0], 1)), dim=1)
    return state_dict
