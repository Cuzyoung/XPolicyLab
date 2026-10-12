"""Fourier amplitude-spectrum randomization (D-tier).

Per-clip-consistent frequency-domain augmentation for clean-to-randomization
(C2R). For each frame we take the 2D FFT, perturb ONLY the amplitude spectrum
while keeping the PHASE spectrum intact (phase carries object shape/structure),
then iFFT back to the spatial domain. This changes texture/contrast statistics
without moving edges/geometry, which is complementary to photometric and AdaIN.

The SAME random amplitude perturbation is sampled once per clip and applied to
every frame (temporal consistency). Operates on float frames in [0, 1] with
shape [T, C, H, W]. No external assets/weights; cheap.

NO geometric ops. Requires no masks.
"""
from __future__ import annotations

import torch
from torch import nn


def _uniform(low: float, high: float, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())


class FourierAmplitudeRandomize(nn.Module):
    """Randomize the amplitude spectrum, preserve the phase spectrum."""

    def __init__(
        self,
        p: float = 0.5,
        # per-frequency multiplicative jitter of the amplitude spectrum
        amp_jitter=(0.5, 1.5),
        # blend factor towards a random low-frequency amplitude envelope
        envelope_strength=(0.0, 0.6),
        # low-frequency grid size for the random envelope (upsampled to H,W)
        envelope_grid: int = 8,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.p = float(p)
        self.amp_jitter = tuple(amp_jitter)
        self.envelope_strength = tuple(envelope_strength)
        self.envelope_grid = int(envelope_grid)
        self.eps = float(eps)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [T, C, H, W] float in [0, 1]. Returns same shape."""
        dev, dt = frames.device, frames.dtype
        if _uniform(0.0, 1.0, dev) >= self.p:
            return frames

        T, C, H, W = frames.shape

        # FFT in float32 for numerical stability, then cast back.
        x = frames.to(torch.float32)
        fft = torch.fft.fft2(x, dim=(-2, -1))
        amp = torch.abs(fft)
        phase = torch.angle(fft)

        # --- build one random amplitude modulation, shared across the clip ---
        # (a) global per-frequency multiplicative jitter, same H x W map per channel
        g = int(self.envelope_grid)
        lo = torch.empty(1, 1, g, g, device=dev, dtype=torch.float32).uniform_(
            self.amp_jitter[0], self.amp_jitter[1]
        )
        jitter = torch.nn.functional.interpolate(
            lo, size=(H, W), mode="bilinear", align_corners=False
        )  # [1,1,H,W]

        # (b) blend towards a random smooth low-frequency envelope
        es = _uniform(self.envelope_strength[0], self.envelope_strength[1], dev)
        env_lo = torch.empty(1, 1, g, g, device=dev, dtype=torch.float32).uniform_(0.5, 1.5)
        envelope = torch.nn.functional.interpolate(
            env_lo, size=(H, W), mode="bilinear", align_corners=False
        )
        mod = jitter * (1.0 - es) + envelope * es  # [1,1,H,W], broadcast over T,C

        new_amp = amp * mod.clamp_min(0.0)
        new_fft = torch.polar(new_amp, phase)
        out = torch.fft.ifft2(new_fft, dim=(-2, -1)).real

        return out.to(dt).clamp(0.0, 1.0)
