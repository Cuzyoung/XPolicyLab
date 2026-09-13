import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sapolicy.models.utils.rope import rope_apply
from sapolicy.models.wan2_1_attention import AttentionModule


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w1(x)) * self.w2(x))


class SelfAttention(nn.Module):
    """
    Encoder self-attention with:
    - QK-Norm (RMSNorm on each head)
    - scaled_dot_product_attention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=out_bias)

        # QK-Norm is typically applied per-head on the last dim (head_dim).
        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.attn = AttentionModule(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_p=dropout,
            causal=False,
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, T, H, Hd)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, T, D)
        attn_mask:
            optional attention mask broadcastable to (B, H, T, T) or (T, T)
            - bool mask: True means keep / attend, False means masked out
            - float mask: additive bias (e.g. 0 for keep, -inf for mask)
        key_padding_mask:
            optional bool tensor of shape (B, T)
            True means "this token is padding and should be masked out"
        """
        B, T, D = x.shape

        q = self._shape(self.q_proj(x))  # (B, T, H, Hd)
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        if freqs is not None:
            # rope_apply expects [B, seq_len, n_heads, head_dim]
            q = rope_apply(q, None, freqs).type_as(q)
            k = rope_apply(k, None, freqs).type_as(k)

        attn_out = self.attn(q, k, v)

        attn_out = attn_out.contiguous().view(B, T, D)
        if attn_out.dtype != self.out_proj.weight.dtype:
            attn_out = attn_out.to(self.out_proj.weight.dtype)
        return self.out_proj(attn_out)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=out_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)

        self.proj_dropout = nn.Dropout(dropout)
        self.attn = AttentionModule(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_p=dropout,
            causal=False,
        )

    def project_kv(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project static conditioning once for iterative denoising."""
        B, S, C = cond.shape
        if C != self.dim:
            raise ValueError(f"CrossAttention cond dim {C} != configured dim {self.dim}")
        k = self.k_proj(cond).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(cond).view(B, S, self.num_heads, self.head_dim)
        return self.k_norm(k), v

    def forward(
        self,
        x: torch.Tensor,                  # [B, T, C]  query
        cond: Optional[torch.Tensor] = None,  # [B, S, C] key/value source
        kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        if (cond is None) == (kv is None):
            raise ValueError("CrossAttention requires exactly one of cond or precomputed kv")

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)   # [B, T, H, D]
        q = self.q_norm(q)
        if kv is None:
            k, v = self.project_kv(cond)
        else:
            k, v = kv
            if (
                k.ndim != 4
                or v.ndim != 4
                or k.shape != v.shape
                or k.shape[0] != B
                or k.shape[2:] != (self.num_heads, self.head_dim)
            ):
                raise ValueError(
                    "CrossAttention cached kv must be matching [B,S,H,D] tensors, "
                    f"got {tuple(k.shape)} and {tuple(v.shape)}"
                )

        out = self.attn(q, k, v)   # [B, T, H, D]
        out = out.contiguous().view(B, T, C)
        if out.dtype != self.out_proj.weight.dtype:
            out = out.to(self.out_proj.weight.dtype)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    """
    A modern transformer block:
    - Pre RMSNorm
    - QK-Norm attention
    - SwiGLU MLP
    - Conditional attention if conditional is True
    - residual connections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 8 / 3,   # common with SwiGLU-ish setups
        dropout: float = 0.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        # Often rounded for efficiency
        hidden_dim = (hidden_dim + 255) // 256 * 256

        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=proj_bias,
            qk_norm_eps=norm_eps,
        )

        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim=dim, hidden_dim=hidden_dim, bias=proj_bias)

        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self-attention
        x = x + self.resid_dropout(
            self.self_attn(
                self.norm1(x),
                freqs=freqs,
            )
        )
        x = x + self.resid_dropout(self.mlp(self.norm2(x)))
        return x


class DiTBlock(nn.Module):
    """
    DiT-style block with AdaLN modulation and residual gating.

    This block keeps only global non-causal self-attention + MLP and
    uses timestep/context embedding `e` to modulate both branches.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        norm_eps: float = 1e-6,
        conditional: bool = False,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        hidden_dim = (hidden_dim + 255) // 256 * 256

        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=proj_bias,
            qk_norm_eps=norm_eps,
        )

        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.mlp = SwiGLU(dim=dim, hidden_dim=hidden_dim, bias=proj_bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.conditional = conditional
        if self.conditional:
            self.norm_cross = RMSNorm(dim, eps=norm_eps)
            self.norm_cond = RMSNorm(dim, eps=norm_eps)
            self.cross_attn = CrossAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                qkv_bias=qkv_bias,
                out_bias=proj_bias,
                qk_norm_eps=norm_eps,
            )

        # Six-way modulation: [sa_shift, sa_scale, sa_gate, mlp_shift, mlp_scale, mlp_gate]
        self.modulation = nn.Parameter(torch.randn(1, 1, 6, dim) / dim**0.5)
        if self.conditional:
            # [shift, scale, gate]
            self.modulation_cross = nn.Parameter(torch.randn(1, 1, 3, dim) / dim**0.5)

    @staticmethod
    def _align_modulation(part: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Align modulation sequence length with token sequence length.
        part: [B, Le, 1, D]
        """
        le = part.shape[1]
        if le == target_len:
            return part
        if le > target_len:
            return part[:, :target_len]
        repeat = (target_len + le - 1) // le
        return part.repeat_interleave(repeat, dim=1)[:, :target_len]

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        cross_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        x: [B, L, D]
        e: [B, Le, 6, D] if ``conditional`` is False; [B, Le, 9, D] if True (6 for self-attn/MLP AdaLN + 3 for cross).
        cond: [B, S, D] key/value context when ``conditional`` is True (e.g. vision + TCP tokens).
        freqs: optional RoPE freqs for self-attention
        """
        if e.dim() != 4 or e.shape[3] != x.shape[-1]:
            raise ValueError(
                f"Expected e as [B, Le, *, D] with D matching x, got {tuple(e.shape)} for x {tuple(x.shape)}"
            )
        if e.shape[2] < 9:
            raise ValueError(
                f"Expected e.shape[2]>=9, got {e.shape[2]}"
            )
        if self.conditional and cond is None and cross_kv is None:
            raise ValueError(
                "DiTBlock(conditional=True) requires `cond` or cached `cross_kv`."
            )

        # Learnable base modulation + external modulation embedding.
        mods = (self.modulation + e[:, :, :6, :]).chunk(6, dim=2)
        l = x.shape[1]
        mods = tuple(self._align_modulation(m, l).squeeze(2) for m in mods)  # 6 x [B, L, D]
        sa_shift, sa_scale, sa_gate, mlp_shift, mlp_scale, mlp_gate = mods

        # Self-attention branch with AdaLN + residual gating.
        h = self.norm1(x) * (1 + sa_scale) + sa_shift
        h = self.self_attn(h, freqs=freqs)
        x = x + self.resid_dropout(h * sa_gate)

        if self.conditional:
            # Last 3 of 9 timestep channels for cross AdaLN (indices 6..8), matching modulation_cross [1, 1, 3, D].
            e_cross = e[:, :, 6:, :]
            mods_cross = (self.modulation_cross + e_cross).chunk(3, dim=2)
            mods_cross = tuple(self._align_modulation(m, l).squeeze(2) for m in mods_cross)
            shift_cross, scale_cross, gate_cross = mods_cross
            h = self.norm_cross(x) * (1 + scale_cross) + shift_cross
            if cross_kv is None:
                h = self.cross_attn(h, self.norm_cond(cond))
            else:
                h = self.cross_attn(h, kv=cross_kv)
            x = x + self.resid_dropout(h * gate_cross)

        # MLP branch with AdaLN + residual gating.
        h = self.norm2(x) * (1 + mlp_scale) + mlp_shift
        h = self.mlp(h)
        x = x + self.resid_dropout(h * mlp_gate)
        return x

    def project_cross_kv(
        self, cond: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return this block's normalized/projected conditioning cache."""
        if not self.conditional:
            return None
        return self.cross_attn.project_kv(self.norm_cond(cond))
