# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import os
from einops import repeat

ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"


def rope_params(max_seq_len, dim, theta=10000):
    if ENABLE_TENSORRT:
        return rope_params_no_polar(max_seq_len, dim, theta)
    else:
        return rope_params_polar(max_seq_len, dim, theta)


# @amp.autocast(enabled=False)
def rope_params_polar(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def rope_params_no_polar(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    inv_freq = 1.0 / torch.pow(
        theta,
        torch.arange(0, dim, 2).to(torch.float32) / dim
    )
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.stack((freqs.cos(), freqs.sin()), dim=-1).flatten(-2)
    return emb

def rope_apply(x, grid_sizes, freqs):
    if ENABLE_TENSORRT:
        return rope_apply_no_polar(x, freqs)
    else:
        return rope_apply_polar(x, freqs)

# @amp.autocast(enabled=False)
def rope_apply_polar(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x

def rope_apply_no_polar(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, seq_len, n, D = x.shape

    # Reshape freqs to be broadcastable: (1, seq_len, 1, D)
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    x0, x1 = x.chunk(2, dim=-1)
    freqs_cos, freqs_sin = freqs.chunk(2, dim=-1)

    rotated_x0 = x0 * freqs_cos - x1 * freqs_sin
    rotated_x1 = x1 * freqs_cos + x0 * freqs_sin
    x_rotated = torch.cat((rotated_x0, rotated_x1), dim=-1)
    return x_rotated