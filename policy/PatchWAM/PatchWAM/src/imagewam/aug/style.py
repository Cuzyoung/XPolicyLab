"""Style randomization via AdaIN (C-tier).

Per-clip-consistent "swap-texture-keep-shape" augmentation for
clean-to-randomization (C2R) visual generalization. Operates on float frames in
[0, 1] with shape [T, C, H, W]; the SAME sampled style parameters are applied to
every frame in the clip to preserve temporal consistency.

AdaIN idea: whiten each channel's spatial statistics (per-frame mean/std) and
re-color them with a randomly sampled target (mean, std). This changes global
color/contrast "style" while preserving spatial structure (shape/layout), unlike
photometric jitter which is a fixed parametric family. No external network or
weights: the target statistics are sampled procedurally (optionally biased
towards the clip's own statistics for realism).

NO geometric ops. Requires no masks.
"""
from __future__ import annotations

import torch
from torch import nn


def _uniform(low: float, high: float, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())


class StyleRandomize(nn.Module):
    """AdaIN-style channel-statistics randomization (texture swap, shape keep)."""

    def __init__(
        self,
        p: float = 0.5,
        target_mean=(0.2, 0.8),
        target_std=(0.05, 0.4),
        # how strongly to move towards the random target (0 = identity, 1 = full swap)
        strength=(0.4, 1.0),
        eps: float = 1e-5,
    ):
        super().__init__()
        self.p = float(p)
        self.target_mean = tuple(target_mean)
        self.target_std = tuple(target_std)
        self.strength = tuple(strength)
        self.eps = float(eps)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [T, C, H, W] float in [0, 1]. Returns same shape, one style per clip."""
        dev, dt = frames.device, frames.dtype
        if _uniform(0.0, 1.0, dev) >= self.p:
            return frames

        T, C, H, W = frames.shape
        # per-channel source statistics over the WHOLE clip (spatial + temporal),
        # so the style transform is identical across frames -> temporally consistent.
        flat = frames.permute(1, 0, 2, 3).reshape(C, -1)  # [C, T*H*W]
        src_mean = flat.mean(dim=1).view(1, C, 1, 1)
        src_std = flat.std(dim=1).view(1, C, 1, 1).clamp_min(self.eps)

        # random target statistics (one draw per channel, shared across the clip)
        tgt_mean = torch.empty(C, device=dev, dtype=dt).uniform_(
            self.target_mean[0], self.target_mean[1]
        ).view(1, C, 1, 1)
        tgt_std = torch.empty(C, device=dev, dtype=dt).uniform_(
            self.target_std[0], self.target_std[1]
        ).view(1, C, 1, 1)

        s = _uniform(self.strength[0], self.strength[1], dev)
        # interpolate target stats between source (identity) and random target
        eff_mean = src_mean + s * (tgt_mean - src_mean)
        eff_std = src_std + s * (tgt_std - src_std)

        normed = (frames - src_mean) / src_std  # whiten
        out = normed * eff_std + eff_mean       # re-color
        return out.clamp(0.0, 1.0)
