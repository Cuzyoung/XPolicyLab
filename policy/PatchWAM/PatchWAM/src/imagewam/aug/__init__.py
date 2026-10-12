"""Domain-randomization augmentation for clean-to-randomization (C2R).

- ``DomainRandomization``: drop-in for the data config's ``video_augmentation``
  field (A-tier photometric out of the box; C/D style+fourier when enabled;
  B-tier background when masks given).
- ``PhotometricRandomize``: A-tier photometric randomization.
- ``StyleRandomize``: C-tier AdaIN style randomization (texture swap, shape keep).
- ``FourierAmplitudeRandomize``: D-tier amplitude-spectrum randomization.
- ``MaskGuidedBackground`` / ``BackgroundBank``: B-tier background replacement.
- ``PrecomputedMaskProvider`` / ``NullMaskProvider``: foreground-mask sources.
"""
from .photometric import PhotometricRandomize
from .style import StyleRandomize
from .fourier import FourierAmplitudeRandomize
from .background import BackgroundBank, MaskGuidedBackground
from .mask_provider import NullMaskProvider, PrecomputedMaskProvider
from .domain_randomization import DomainRandomization

__all__ = [
    "DomainRandomization",
    "PhotometricRandomize",
    "StyleRandomize",
    "FourierAmplitudeRandomize",
    "BackgroundBank",
    "MaskGuidedBackground",
    "NullMaskProvider",
    "PrecomputedMaskProvider",
]
