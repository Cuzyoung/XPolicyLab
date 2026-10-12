"""Inference-only flow algorithms, independent of model tokenization and transforms.

Time is noise=0, clean=1. Adapters with the reverse convention must map both
time and velocity. PAINT follows arXiv:2606.19774 Algorithm 1; RTC uses a
clean-estimate VJP. This module never normalizes physical actions.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch


class EulerSampler:
    def __init__(self, num_steps: int):
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(self, velocity, noise):
        x = noise.clone()
        for i in range(self.num_steps):
            x = x + velocity(x, i / self.num_steps) / self.num_steps
        return x


class PaintSampler(EulerSampler):
    @torch.no_grad()
    def sample(self, velocity, noise, *, target, mask):
        if target.shape != noise.shape or mask.shape != noise.shape:
            raise ValueError("PAINT target and mask must match the normalized action shape")
        if mask.dtype != torch.bool or not torch.isfinite(target).all():
            raise ValueError("PAINT requires a boolean mask and finite normalized targets")
        naive = super().sample(velocity, noise)
        x = torch.where(mask, target, naive)
        for i in range(self.num_steps):
            x = x - velocity(x, 1.0 - i / self.num_steps) / self.num_steps
        return super().sample(velocity, torch.where(mask, x, noise))


class RtcSampler(EulerSampler):
    def sample(self, velocity, noise, *, target, weights, beta):
        if target.shape != noise.shape or weights.shape != noise.shape:
            raise ValueError("RTC target and weights must match normalized action shape")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("RTC beta must be finite and positive")
        if not torch.isfinite(target).all() or not torch.isfinite(weights).all():
            raise ValueError("RTC condition must be finite")
        if torch.any((weights < 0) | (weights > 1)):
            raise ValueError("RTC weights must be in [0, 1]")
        x = noise.clone()
        # inference_mode must be disabled by the caller; no_grad is supported.
        with torch.enable_grad():
            for i in range(self.num_steps):
                t = i / self.num_steps
                sample = x.detach().requires_grad_(True)
                v = velocity(sample, t)
                clean = sample + (1 - t) * v
                error = ((target - clean) * weights).detach()
                correction = torch.autograd.grad(clean, sample, grad_outputs=error)[0]
                scale = beta if t == 0 else min(beta, (t * t + (1 - t) ** 2) / (t * (1 - t)))
                x = (sample + (v + scale * correction) / self.num_steps).detach()
        return x


def validate_samples(sampling):
    if set(sampling) - {"mode", "num_samples"}:
        raise ValueError("AAC only accepts num_samples")
    count = sampling.get("num_samples", 20)
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("AAC num_samples must be an integer greater than one")
    return count


def validate_prefix(sampling, horizon, width):
    import numpy as np

    if set(sampling) - {"mode", "action_prefix", "delay_steps"}:
        raise ValueError("PAINT only accepts action_prefix and delay_steps")
    delay = sampling.get("delay_steps")
    if isinstance(delay, bool) or not isinstance(delay, int) or not 0 < delay < horizon:
        raise ValueError("PAINT delay_steps must be an integer in (0, horizon)")
    prefix = np.asarray(sampling.get("action_prefix"), dtype=np.float32)
    if prefix.shape != (delay, width) or not np.isfinite(prefix).all():
        raise ValueError(f"PAINT prefix must be finite with shape {(delay, width)}")
    return prefix, delay


def action_attention(query, key, *, horizon, mask=None):
    """Capture trailing action queries/keys, after RoPE, from [B,H,S,D].

    Softmax includes ALL visible keys before selecting the action block. This
    preserves attention mass across heads/layers for the official aggregation.
    Only action query rows are materialized, avoiding a full vision S-by-S map.
    """
    q = query[..., -horizon:, :].float()
    k = key.float()
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1]:
            raise ValueError("Attention heads must be divisible by KV heads")
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        mask = mask[..., -horizon:, :]
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, float("-inf"))
        else:
            logits = logits + mask
    return logits.softmax(-1)[..., -horizon:].mean(dim=(0, 1)).detach()


class AutoHorizonSampler(EulerSampler):
    """Unmodified Euler actions; official selection from the third evaluation."""

    @torch.no_grad()
    def sample(self, velocity, noise, *, attention_owners):
        if self.num_steps < 3 or noise.shape[-2] < 2:
            raise ValueError("AutoHorizon requires at least 3 steps and 2 action tokens")
        capture = ActionAttentionCapture(noise.shape[-2])
        x = noise.clone()
        for i in range(self.num_steps):
            if i == 2:
                with capture.attach(attention_owners):
                    v = velocity(x, i / self.num_steps)
            else:
                v = velocity(x, i / self.num_steps)
            x = x + v / self.num_steps
        return x, capture.metadata()


class ActionAttentionCapture:
    def __init__(self, horizon):
        self.horizon = horizon
        self.maps = []

    def __call__(self, query, key, mask=None):
        self.maps.append(action_attention(query, key, horizon=self.horizon, mask=mask))

    @contextmanager
    def attach(self, owners):
        owners = list(owners)
        if not owners or any(
            getattr(o, "_action_attention_capture", None) is not None for o in owners
        ):
            raise ValueError("AutoHorizon requires available attention capture hooks")
        try:
            for owner in owners:
                owner._action_attention_capture = self
            yield self
        finally:
            for owner in owners:
                owner._action_attention_capture = None

    def metadata(self):
        from XPolicyLab.utils.autohorizon_official import UPSTREAM_COMMIT, bidir_soft_pointer

        if not self.maps:
            raise RuntimeError("AutoHorizon sampler did not capture action attention")
        matrix = torch.stack(self.maps).mean(0)
        if matrix.shape != (self.horizon, self.horizon) or not torch.isfinite(matrix).all():
            raise ValueError("AutoHorizon received invalid action attention")
        selected, diag = bidir_soft_pointer(matrix)
        return {
            "execution_steps": int(selected),
            "denoising_step": 3,
            "method": "bidir_soft_pointer",
            "upstream_commit": UPSTREAM_COMMIT,
            "N_forward": diag["N_forward"],
            "N_backward": diag["N_backward"],
        }
