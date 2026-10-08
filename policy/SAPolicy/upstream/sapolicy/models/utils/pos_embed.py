# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def position_grid_to_embed(
    pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100
) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions. # noqa

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid


def add_pos_embed(x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
    D = x.shape[-1]
    L = x.shape[1]
    HW = W * H
    pe = create_uv_grid(W, H, aspect_ratio=W / H, dtype=x.dtype, device=x.device)  # [H, W, 2]
    pe = position_grid_to_embed(pe, D) * ratio  # [H, W, D]
    pe_flat = pe.reshape(HW, D).unsqueeze(0)  # [1, HW, D]
    return x + pe_flat.repeat(1, L // HW, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    1D sinusoidal positional embeddings (Attention is All You Need).
    Works for both:
    - Continuous time in [0,1] for Flow Matching
    - Discrete timesteps [0, 1, 2, ..., T] for Diffusion
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):  # [B] or [N]
        """
        Args:
            t: (B,) tensor of time values (continuous or discrete)
        Returns:
            (B, dim) tensor of sinusoidal embeddings
        """
        half_dim = self.dim // 2
        device = t.device
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)  # [half_dim]
        args = t.unsqueeze(-1) * freqs  # [B, half_dim]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


class SinusoidalSequencePosEnc(nn.Module):
    """Transformer 的可泛化正弦位置编码（对序列长度 T 任意）"""

    def __init__(self, embed_dim: int, max_len: int = 4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        pe = self._build_pe(max_len, embed_dim)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(T: int, D: int, device=None):
        position = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / D)
        )
        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [T, D]

    def forward(self, T: int, device: torch.device):
        if T > self.max_len:
            extra = self._build_pe(T, self.embed_dim, device=device)
            self.pe = extra
            self.max_len = T
        return self.pe[:T, :]  # [T, D]
