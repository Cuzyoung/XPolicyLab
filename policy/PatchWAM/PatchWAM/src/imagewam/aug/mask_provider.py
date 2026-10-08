"""Foreground-mask providers for B-tier background replacement.

Masks are NOT stored in the LeRobot data. They must be precomputed offline
(see ``precompute_masks.py``; e.g. SAM or RoboTwin sim segmentation) and loaded
here. The dataset supplies the lookup keys (episode id, frame indices, camera).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import torch


class NullMaskProvider:
    """Always returns None -> B-tier becomes a no-op (A-tier still runs)."""

    def get(self, *args, **kwargs) -> Optional[torch.Tensor]:
        return None


class PrecomputedMaskProvider:
    """Loads foreground masks from disk.

    Layout (either extension): ``<root>/<episode_id>/<key>.png`` or ``.npy`` where
    ``<key>`` is ``<camera>_<frame_idx>`` when a camera is given, else ``<frame_idx>``.
    Returns [T, 1, H, W] float in {0, 1}. Returns None if any frame is missing.
    """

    def __init__(self, root: str, threshold: float = 0.5):
        self.root = str(root)
        self.threshold = float(threshold)

    def _load_one(self, path: str) -> Optional[torch.Tensor]:
        if not os.path.exists(path):
            return None
        if path.endswith(".npy"):
            import numpy as np

            return torch.from_numpy(np.load(path)).float()
        from PIL import Image
        import numpy as np

        m = np.asarray(Image.open(path).convert("L"), dtype="float32") / 255.0
        return torch.from_numpy(m)

    def get(
        self,
        episode_id,
        frame_indices: Sequence[int],
        camera: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        masks = []
        for fi in frame_indices:
            sub = f"{camera}_{fi}" if camera is not None else str(fi)
            base = os.path.join(self.root, str(episode_id), sub)
            m = None
            for ext in (".png", ".npy"):
                m = self._load_one(base + ext)
                if m is not None:
                    break
            if m is None:
                return None
            masks.append((m > self.threshold).float())
        return torch.stack(masks, 0).unsqueeze(1)  # [T, 1, H, W]
