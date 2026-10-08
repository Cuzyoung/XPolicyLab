"""Shared spatial latent trunk (geo inject + transformer → attended patch tokens).

No task-specific readout; consumers are LatentAuxiliaryModel / DynamicsHead / DiT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sapolicy.models.transformer import TransformerBlock, RMSNorm
from sapolicy.models.utils.rope import rope_params


def build_ray_maps(depth, intrinsics, geo_mode="ray_depth"):
    """
    depth:      [B, T, 1, H, W]
    intrinsics: [B, T, 4]  -> (fx, fy, cx, cy)

    return:
        geo_input: [B, 3, T, H, W] = [depth, ray_x, ray_y] or xyz
    """
    B, T, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    fx = intrinsics[..., 0].view(B, T, 1, 1, 1)
    fy = intrinsics[..., 1].view(B, T, 1, 1, 1)
    cx = intrinsics[..., 2].view(B, T, 1, 1, 1)
    cy = intrinsics[..., 3].view(B, T, 1, 1, 1)

    grid_u, grid_v = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing="xy",
    )
    u = grid_u.view(1, 1, 1, H, W)
    v = grid_v.view(1, 1, 1, H, W)

    ray_x = (u - cx) / (fx + 1e-6)
    ray_y = (v - cy) / (fy + 1e-6)

    if geo_mode == "ray_depth":
        geo = torch.cat([depth, ray_x, ray_y], dim=2)
    elif geo_mode == "xyz":
        geo = torch.cat([ray_x * depth, ray_y * depth, depth], dim=2)
    else:
        raise ValueError(f"Unknown geo_mode: {geo_mode}")
    return geo.permute(0, 2, 1, 3, 4).contiguous()


class LatentTrunk(nn.Module):
    """Geo-injected transformer over patch tokens → attended_patch Z."""

    def __init__(
        self,
        in_dim,
        embed_dim,
        patch_size,
        num_heads=4,
        num_layers=4,
        use_depth=True,
        use_camera_intrinsics=True,
        geo_mode="ray_depth",
        final_norm=True,
        geo_gate_init=0.0,
        geo_embed_init_std=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.geo_gate_init = float(geo_gate_init)
        self.geo_embed_init_std = float(geo_embed_init_std)

        self.patch_embedding = nn.Conv3d(in_dim, embed_dim, kernel_size=1, stride=1)
        assert (embed_dim % num_heads) == 0 and (embed_dim // num_heads) % 2 == 0
        d = embed_dim // num_heads
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]

        if use_depth and use_camera_intrinsics:
            geo_in_channels = 3
        elif use_depth:
            geo_in_channels = 1
        else:
            geo_in_channels = 0
        self.geo_in_channels = geo_in_channels
        self.geo_mode = geo_mode

        if geo_in_channels > 0:
            hidden = embed_dim // 4
            self.geo_stem = nn.Sequential(
                nn.Conv3d(geo_in_channels, hidden, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
                nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
            self.geo_embedding_1 = nn.Conv3d(
                in_channels=hidden, out_channels=embed_dim,
                kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size),
            )
            nn.init.normal_(self.geo_embedding_1.weight, std=self.geo_embed_init_std)
            nn.init.constant_(self.geo_embedding_1.bias, 0.0)
            self.geo_embedding_2 = nn.Conv3d(
                in_channels=hidden, out_channels=embed_dim,
                kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size),
            )
            nn.init.normal_(self.geo_embedding_2.weight, std=self.geo_embed_init_std)
            nn.init.constant_(self.geo_embedding_2.bias, 0.0)
            self.geo_gate_1 = nn.Parameter(torch.full((1, embed_dim, 1, 1, 1), self.geo_gate_init))
            self.geo_gate_2 = nn.Parameter(torch.full((1, 1, embed_dim), self.geo_gate_init))
            self.norm_after_inject_1 = nn.GroupNorm(1, embed_dim)
            self.norm_after_inject_2 = RMSNorm(embed_dim, eps=1e-6)

        # Nodepth TCP/auxbypass: do not add RGB-only GroupNorm/RMSNorm here.
        # Unlike DiT notcp (where those two norms consistently help), tcp depth vs
        # nodepth is mixed (lift slightly worse; handover collapsed either way).
        # Inject norms stay gated behind geo_in_channels > 0.

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4,
                dropout=0.1,
                qkv_bias=True,
                proj_bias=False,
                norm_eps=1e-6,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim, eps=1e-6) if final_norm else nn.Identity()

    def _create_freqs(self, grid_size, start_frame: int):
        """grid_size: (f, h, w) as plain ints.

        It used to arrive as a CPU tensor built from x.shape and be converted back with
        .tolist(): a pointless alloc+sync per forward, and the source of two torch.compile
        graph breaks (data-dependent `u0 < 0` plus step_unsupported) in forward_camera.
        A tensor is still accepted so external callers keep working.
        """
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if torch.is_tensor(grid_size):
            grid_size = grid_size.tolist()
        f, h, w = (int(v) for v in grid_size)
        return torch.cat(
            [
                self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

    def forward_camera(self, visual_tokens, depths=None, camera_intrinsics=None):
        """
        visual_tokens: [B, C, T, H, W]
        depths:        [B, 1, T, H, W] optional
        camera_intrinsics: [B, T, 4] optional

        Returns attended_patch: [B, embed_dim, T, H, W]
        """
        B, _, T, H, W = visual_tokens.shape
        x = self.patch_embedding(visual_tokens)
        geo_feat = None

        if self.geo_in_channels > 0:
            if self.geo_in_channels == 1:
                assert depths is not None
                geo = depths
            elif self.geo_in_channels == 3:
                assert depths is not None and camera_intrinsics is not None
                depth_bt1hw = depths.permute(0, 2, 1, 3, 4).contiguous()
                geo = build_ray_maps(depth_bt1hw, camera_intrinsics, self.geo_mode)
            else:
                raise ValueError(f"Unknown geo_in_channels: {self.geo_in_channels}")

            geo_feat = self.geo_stem(geo)
            geo_tokens_1 = self.geo_embedding_1(geo_feat)
            if geo_tokens_1.shape[2:] != x.shape[2:]:
                geo_tokens_1 = F.interpolate(
                    geo_tokens_1, size=x.shape[2:], mode="trilinear", align_corners=False
                )
            x = self.norm_after_inject_1(x + self.geo_gate_1 * geo_tokens_1)

        freqs = self._create_freqs(tuple(x.shape[2:]), start_frame=0)   # static ints: no tensor round-trip
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H * W, self.embed_dim)

        for layer in self.blocks[: self.num_layers // 2]:
            x = layer(x, cond=None, freqs=freqs)

        if self.geo_in_channels > 0:
            geo_tokens_2 = self.geo_embedding_2(geo_feat)
            if geo_tokens_2.shape[2:] != (T, H, W):
                geo_tokens_2 = F.interpolate(
                    geo_tokens_2, size=(T, H, W), mode="trilinear", align_corners=False
                )
            geo_tokens_2 = geo_tokens_2.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H * W, self.embed_dim)
            x = self.norm_after_inject_2(x + self.geo_gate_2 * geo_tokens_2)

        for layer in self.blocks[self.num_layers // 2 :]:
            x = layer(x, cond=None, freqs=freqs)
        x = self.norm(x)
        return x.view(B, T, H, W, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()
