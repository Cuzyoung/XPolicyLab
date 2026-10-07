"""Mask-guided background replacement (B-tier).

Given foreground masks, replaces the BACKGROUND region of a camera clip with a
random texture / image (and optional distractor blobs) to force background
invariance. Per-clip consistent: one background plate is sampled per clip and
reused across all frames.

Foreground masks are NOT present in the LeRobot data; they must be precomputed
offline (see ``precompute_masks.py`` and ``mask_provider.py``). Without a mask
this module is a no-op.
"""
from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import torch
from torch import nn
import torchvision.transforms.functional as F


def _uniform(low: float, high: float, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())


class BackgroundBank:
    """Samples random background plates.

    Uses images/textures from ``asset_dirs`` when available, otherwise generates
    procedural plates (solid color / gradient / low-frequency noise). Procedural
    plates need no external assets, so B-tier works even without a texture set.
    """

    def __init__(
        self,
        asset_dirs: Optional[Sequence[str]] = None,
        procedural: bool = True,
        asset_prob: float = 0.7,
        exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        self.procedural = procedural
        self.asset_prob = float(asset_prob)
        self.files = []
        if asset_dirs:
            if isinstance(asset_dirs, str):
                asset_dirs = [asset_dirs]
            for d in asset_dirs:
                for e in exts:
                    self.files.extend(glob.glob(os.path.join(d, "**", "*" + e), recursive=True))
        self.files = sorted(self.files)

    def _procedural(self, h, w, device, dtype) -> torch.Tensor:
        mode = int(torch.randint(0, 3, ()).item())
        if mode == 0:  # solid color
            c = torch.rand(3, device=device, dtype=dtype).view(3, 1, 1)
            return c.expand(3, h, w).clone()
        if mode == 1:  # gradient between two colors
            c0 = torch.rand(3, device=device, dtype=dtype).view(3, 1, 1)
            c1 = torch.rand(3, device=device, dtype=dtype).view(3, 1, 1)
            axis_w = bool(torch.randint(0, 2, ()).item())
            n = w if axis_w else h
            t = torch.linspace(0, 1, n, device=device, dtype=dtype)
            t = t.view(1, 1, n) if axis_w else t.view(1, n, 1)
            return (c0 + (c1 - c0) * t).expand(3, h, w).clone()
        # low-frequency noise upsampled to full res
        s = int(torch.randint(4, 17, ()).item())
        low = torch.rand(3, s, s, device=device, dtype=dtype)
        return F.resize(low, [h, w], antialias=True).clamp(0.0, 1.0)

    def sample(self, h, w, device, dtype) -> torch.Tensor:
        use_asset = self.files and (not self.procedural or torch.rand(()).item() < self.asset_prob)
        if use_asset:
            try:
                from PIL import Image
                import numpy as np

                idx = int(torch.randint(0, len(self.files), ()).item())
                img = Image.open(self.files[idx]).convert("RGB")
                arr = np.asarray(img, dtype="float32") / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=dtype)
                t = F.resize(t, [h, w], antialias=True)
                if bool(torch.randint(0, 2, ()).item()):
                    t = torch.flip(t, dims=[-1])
                return t.clamp(0.0, 1.0)
            except Exception:
                pass
        return self._procedural(h, w, device, dtype)


class MaskGuidedBackground(nn.Module):
    """Composite a random background behind the (masked) foreground."""

    def __init__(
        self,
        bank: Optional[BackgroundBank] = None,
        feather: float = 1.5,
        distractor_prob: float = 0.0,
        distractor_count=(1, 3),
    ):
        super().__init__()
        self.bank = bank or BackgroundBank()
        self.feather = float(feather)
        self.distractor_prob = float(distractor_prob)
        self.distractor_count = tuple(distractor_count)

    def forward(self, frames: torch.Tensor, fg_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """frames: [T, C, H, W] float[0,1]; fg_mask: [T,1,H,W] or [1,1,H,W]/[T,H,W],
        value 1 = foreground (kept), 0 = background (replaced)."""
        if fg_mask is None:
            return frames
        T, C, H, W = frames.shape
        dev, dt = frames.device, frames.dtype

        if fg_mask.dim() == 3:  # [T,H,W]
            fg_mask = fg_mask.unsqueeze(1)
        if fg_mask.shape[0] == 1 and T > 1:
            fg_mask = fg_mask.expand(T, 1, H, W)
        fg = fg_mask.to(device=dev, dtype=dt).clamp(0.0, 1.0)

        if self.feather > 0:
            k = int(2 * round(3 * self.feather) + 1)
            fg = F.gaussian_blur(fg, kernel_size=[k, k], sigma=[self.feather, self.feather])

        bg = self.bank.sample(H, W, dev, dt).unsqueeze(0)  # [1,3,H,W], shared across frames
        if self.distractor_prob > 0 and _uniform(0.0, 1.0, dev) < self.distractor_prob:
            bg = self._add_distractors(bg, H, W, dev, dt)

        out = fg * frames + (1.0 - fg) * bg
        return out.clamp(0.0, 1.0)

    def _add_distractors(self, bg, H, W, dev, dt) -> torch.Tensor:
        n = int(torch.randint(self.distractor_count[0], self.distractor_count[1] + 1, ()).item())
        bg = bg.clone()
        for _ in range(n):
            ph = max(1, int(_uniform(0.08, 0.25, dev) * H))
            pw = max(1, int(_uniform(0.08, 0.25, dev) * W))
            top = int(_uniform(0, max(1, H - ph), dev))
            left = int(_uniform(0, max(1, W - pw), dev))
            color = torch.rand(3, device=dev, dtype=dt).view(1, 3, 1, 1)
            bg[..., top:top + ph, left:left + pw] = color
        return bg
