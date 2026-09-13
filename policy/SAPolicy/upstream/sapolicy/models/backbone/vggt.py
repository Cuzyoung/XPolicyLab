"""VGGT (Visual Geometry Grounded Transformer) backbone for multi-view fusion.

Wraps VGGT's Aggregator as a frozen visual backbone. Unlike DA3 which processes
each camera independently, VGGT uses alternating frame/global attention to fuse
cross-view information at every layer, producing view-aware features.

Key differences from DA3:
- DA3: ViT-B (768 dim), output 1536 (local+global concat), per-view independent
- VGGT: ViT-L (1024 dim), output 2048 (frame+global concat), cross-view fused
"""

import torch
import torch.nn as nn
import contextlib
from typing import Tuple, Optional


class VGGTBackbone(nn.Module):
    """
    VGGT Aggregator as frozen visual backbone.

    Takes multi-view images [B, S, 3, H, W] and produces per-view tokens
    that already contain cross-view information through global attention.

    Args:
        pretrained: If True, load pretrained weights from HuggingFace
        img_size: Expected image size (must be multiple of 14)
        feature_level: Which aggregator output layer to use (-1 = last)
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        pretrained: bool = True,
        img_size: int = 518,
        feature_level: int = -1,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        if pretrained:
            from vggt.models.vggt import VGGT as VGGTModel
            # Prefer an explicit local snapshot for offline clusters.
            path = pretrained_path if pretrained_path else "facebook/VGGT-1B"
            kwargs = {}
            if isinstance(path, str) and (path.startswith("/") or path.startswith(".")):
                kwargs["local_files_only"] = True
            full_model = VGGTModel.from_pretrained(path, **kwargs)
            self.aggregator = full_model.aggregator
            del full_model
        else:
            from vggt.models.aggregator import Aggregator
            self.aggregator = Aggregator(img_size=img_size, patch_size=self.PATCH_SIZE, embed_dim=1024)

        # Freeze completely
        self.aggregator.eval()
        for param in self.aggregator.parameters():
            param.requires_grad = False

        self.feature_level = feature_level
        self.embed_dim = 1024  # ViT-L
        self.hidden_size = 2048  # frame + global concat
        self.patch_size = self.PATCH_SIZE

        n_params = sum(p.numel() for p in self.aggregator.parameters())
        print(f"[VGGTBackbone] Loaded: {n_params / 1e6:.1f}M params, "
              f"embed_dim={self.embed_dim}, output_dim={self.hidden_size}")

    def train(self, mode=True):
        """Override train to keep aggregator in eval mode."""
        super().train(mode)
        self.aggregator.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        feature_levels=None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Forward pass through VGGT aggregator.

        Args:
            images: [B, S, 3, H, W] in range [0, 1], where S = number of views.
                    Note: VGGT normalizes internally, so pass raw [0,1] images.
            feature_levels: Optional list of aggregator output indices to
                channel-concat (DINO-style multilayer). ``None`` uses
                ``self.feature_level`` only (default last / -1).

        Returns:
            tokens: [B, S, N_patches, 2048 * K] patch tokens (special tokens excluded)
            patch_start_idx: index where patch tokens start in raw output
        """
        if images.ndim == 4:
            # Single view: [B, 3, H, W] -> [B, 1, 3, H, W]
            images = images.unsqueeze(1)

        input_dtype = images.dtype
        # VGGT requires float32 internally.
        grad_enabled = any(param.requires_grad for param in self.aggregator.parameters())
        grad_context = contextlib.nullcontext() if grad_enabled else torch.no_grad()
        with grad_context:
            output_list, patch_start_idx = self.aggregator(images.float())

        if feature_levels is None:
            levels = [self.feature_level]
        else:
            levels = [int(i) for i in feature_levels]
            if not levels:
                raise ValueError("feature_levels must be a non-empty list when provided")

        patch_list = []
        for level in levels:
            tokens = output_list[level]  # [B, S, P_total, 2048]
            patch_list.append(tokens[:, :, patch_start_idx:, :])  # [B, S, N_patches, 2048]

        if len(patch_list) == 1:
            patch_tokens = patch_list[0]
        else:
            patch_tokens = torch.cat(patch_list, dim=-1)  # [B, S, N, 2048 * K]

        # Cast back to input dtype (important for fp32 training)
        return patch_tokens.to(input_dtype), patch_start_idx
