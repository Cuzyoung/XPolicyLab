"""Domain randomization for clean-to-randomization (C2R) generalization.

Drop-in replacement for
``imagewam.datasets.lerobot.transforms.image.VideoAugmentation`` in the data
config's ``video_augmentation`` field. Same call signature ``forward(video)``
where ``video`` is [num_cameras, T, C, H, W] float in [0, 1] (also supports
[T, C, H, W] and [C, H, W]).

Tiers (each clip-consistent, no geometric ops):
* A photometric  : brightness/contrast/gamma/color/noise/blur. Always available.
* C style        : AdaIN channel-statistics swap (texture swap, shape keep).
* D fourier      : amplitude-spectrum randomization, phase preserved.
* B background   : mask-guided background replacement, ONLY when a foreground
  mask is passed via ``forward(video, masks=...)``.

Order per clip: photometric -> style -> fourier -> (mask) background.
Each sub-tier fires independently by its own ``p`` (except photometric which is
gated only by the outer ``p``). Style/fourier/background default OFF, so passing
no ``style``/``fourier``/``background`` config reproduces the previous behavior
exactly (backward compatible).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .photometric import PhotometricRandomize
from .style import StyleRandomize
from .fourier import FourierAmplitudeRandomize
from .background import BackgroundBank, MaskGuidedBackground


class DomainRandomization(nn.Module):
    def __init__(
        self,
        p: float = 0.8,
        photometric: Optional[dict] = None,
        style: Optional[dict] = None,
        fourier: Optional[dict] = None,
        background: Optional[dict] = None,
    ):
        super().__init__()
        self.p = float(p)

        self.photometric = PhotometricRandomize(**(photometric or {}))

        # --- C-tier: AdaIN style (default OFF -> backward compatible) ---
        style = style or {}
        self.style_enabled = bool(style.get("enabled", False))
        if self.style_enabled:
            style_kwargs = {k: v for k, v in style.items() if k != "enabled"}
            self.style = StyleRandomize(**style_kwargs)
        else:
            self.style = None

        # --- D-tier: Fourier amplitude (default OFF -> backward compatible) ---
        fourier = fourier or {}
        self.fourier_enabled = bool(fourier.get("enabled", False))
        if self.fourier_enabled:
            fourier_kwargs = {k: v for k, v in fourier.items() if k != "enabled"}
            self.fourier = FourierAmplitudeRandomize(**fourier_kwargs)
        else:
            self.fourier = None

        # --- B-tier: mask-guided background (default OFF -> backward compatible) ---
        background = background or {}
        self.background_enabled = bool(background.get("enabled", False))
        self.background_p = float(background.get("p", 1.0))
        if self.background_enabled:
            bank = BackgroundBank(
                asset_dirs=background.get("asset_dirs"),
                procedural=background.get("procedural", True),
                asset_prob=background.get("asset_prob", 0.7),
            )
            self.bg = MaskGuidedBackground(
                bank=bank,
                feather=background.get("feather", 1.5),
                distractor_prob=background.get("distractor_prob", 0.0),
                distractor_count=background.get("distractor_count", (1, 3)),
            )
        else:
            self.bg = None

    def _augment_clip(self, frames: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        frames = self.photometric(frames)
        if self.style is not None:
            frames = self.style(frames)
        if self.fourier is not None:
            frames = self.fourier(frames)
        if self.bg is not None and mask is not None and torch.rand(()).item() < self.background_p:
            frames = self.bg(frames, mask)
        return frames

    def forward(self, video: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not torch.is_floating_point(video):
            raise TypeError(f"DomainRandomization expects float frames, got {video.dtype}")
        if torch.rand(()).item() >= self.p:
            return video

        if video.dim() == 5:  # [cam, T, C, H, W]
            out = video.clone()
            for c in range(video.shape[0]):
                m = None if masks is None else masks[c]
                out[c] = self._augment_clip(video[c], m)
            return out
        if video.dim() == 4:  # [T, C, H, W]
            return self._augment_clip(video, masks)
        if video.dim() == 3:  # [C, H, W]
            return self._augment_clip(video.unsqueeze(0), masks).squeeze(0)
        raise ValueError(f"DomainRandomization expects 3/4/5-D, got {tuple(video.shape)}")
