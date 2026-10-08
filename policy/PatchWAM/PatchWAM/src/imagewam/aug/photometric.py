"""Photometric domain randomization (A-tier).

Per-clip-consistent photometric augmentation for clean-to-randomization (C2R)
visual generalization. Operates on float frames in [0, 1] with shape
[T, C, H, W]; the SAME sampled parameters are applied to every frame in the clip
to preserve temporal consistency (the model predicts future frames, so an
inconsistent per-frame transform would corrupt the video target).

NO geometric ops (crop / rotate / flip): actions are in absolute coordinates and
must not be desynchronized from the pixels.
"""
from __future__ import annotations

import torch
from torch import nn
import torchvision.transforms.functional as F


def _uniform(low: float, high: float, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())


class PhotometricRandomize(nn.Module):
    """Strong photometric randomization (background/lighting invariance)."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        gamma=(0.7, 1.4),
        exposure=(-0.5, 0.5),
        color_temperature: float = 0.15,
        gaussian_noise_std: float = 0.03,
        blur_sigma=(0.0, 1.5),
        blur_prob: float = 0.3,
    ):
        super().__init__()
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.gamma = tuple(gamma) if gamma else None
        self.exposure = tuple(exposure) if exposure else None
        self.color_temperature = float(color_temperature)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.blur_sigma = tuple(blur_sigma) if blur_sigma else None
        self.blur_prob = float(blur_prob)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [T, C, H, W] float in [0, 1]. Returns same shape."""
        dev = frames.device

        ops = []
        if self.brightness > 0:
            f = _uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness, dev)
            ops.append(lambda x, f=f: F.adjust_brightness(x, f))
        if self.contrast > 0:
            f = _uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast, dev)
            ops.append(lambda x, f=f: F.adjust_contrast(x, f))
        if self.saturation > 0:
            f = _uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation, dev)
            ops.append(lambda x, f=f: F.adjust_saturation(x, f))
        if self.hue > 0:
            f = _uniform(-min(self.hue, 0.5), min(self.hue, 0.5), dev)
            ops.append(lambda x, f=f: F.adjust_hue(x, f))
        if ops:
            for i in torch.randperm(len(ops)).tolist():
                frames = ops[i](frames)

        if self.gamma:
            g = _uniform(self.gamma[0], self.gamma[1], dev)
            frames = F.adjust_gamma(frames.clamp(0.0, 1.0), gamma=g)

        if self.exposure:
            ev = _uniform(self.exposure[0], self.exposure[1], dev)
            frames = frames * (2.0 ** ev)

        if self.color_temperature > 0:
            # per-channel gain: warm (R up / B down) <-> cool (R down / B up)
            t = self.color_temperature
            gains = torch.tensor(
                [1.0 + _uniform(-t, t, dev),
                 1.0 + _uniform(-0.5 * t, 0.5 * t, dev),
                 1.0 + _uniform(-t, t, dev)],
                device=dev, dtype=frames.dtype,
            ).view(1, 3, 1, 1)
            frames = frames * gains

        frames = frames.clamp(0.0, 1.0)

        if self.blur_sigma and _uniform(0.0, 1.0, dev) < self.blur_prob:
            sig = _uniform(max(1e-3, self.blur_sigma[0]), self.blur_sigma[1], dev)
            if sig > 1e-2:
                k = int(2 * round(3 * sig) + 1)
                frames = F.gaussian_blur(frames, kernel_size=[k, k], sigma=[sig, sig])

        if self.gaussian_noise_std > 0:
            noise = torch.randn(
                (1,) + tuple(frames.shape[1:]), device=dev, dtype=frames.dtype
            ) * self.gaussian_noise_std
            frames = frames + noise

        return frames.clamp(0.0, 1.0)
