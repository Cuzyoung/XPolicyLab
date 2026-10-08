"""ConceptBottleneck: compress the reference-image tokens into K concept slots
for the FLUX.2 action expert (CC-WAM Stage 1).

Design (see 实验计划 —— Stage 1 代码实现细节):
  - K learnable slot queries (in the action expert's hidden dim `da`).
  - They cross-attend the video expert's reference-image tokens (the `cond`
    segment of `video_pre["tokens"]["img"]`, i.e. its first `cond_len` tokens,
    dim `dv`).
  - Output: c_t [B, K, da]. In Stage 1 these are PREPENDED to the action token
    sequence (extra information the action expert can attend to). It is NOT a
    true bottleneck yet — the action expert still sees the image directly; that
    restriction is Stage 2.

Kept deliberately small and self-contained: it uses its own MultiheadAttention
(da must be divisible by n_heads) and does not touch the frozen experts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConceptBottleneck(nn.Module):
    def __init__(self, dv: int = 3072, da: int = 1024, K: int = 16, n_heads: int = 8):
        super().__init__()
        if da % n_heads != 0:
            raise ValueError(f"da={da} must be divisible by n_heads={n_heads}")
        self.dv, self.da, self.K, self.n_heads = dv, da, K, n_heads
        # learnable concept slot queries, in the action hidden dim
        self.slots = nn.Parameter(torch.randn(K, da) * 0.02)
        # project reference-image tokens (video dim dv) into action dim da
        self.kv_proj = nn.Linear(dv, da)
        self.attn = nn.MultiheadAttention(da, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(da)
        # CC-WAM Stage 1 方向1a: 强制注入,不再用 gate。
        # (zero-init gate 在 ID 目标下被优化器压回 0、concept 被忽略——见 gate 实验存档:
        #  B1 loss 0.00279 vs B0 0.00272, sign-test z=0.28 打平。)
        # 去掉 gate 这个"免费忽略"出口,concept 强制进入 action 序列,由训练决定它有害/无害/有益。

    def forward(self, ref_img_tokens: torch.Tensor) -> torch.Tensor:
        """ref_img_tokens: [B, cond_len, dv] (video expert's cond segment).
        returns concept slots: [B, K, da]."""
        if ref_img_tokens.ndim != 3:
            raise ValueError(f"`ref_img_tokens` must be [B,cond_len,dv], got {tuple(ref_img_tokens.shape)}")
        if ref_img_tokens.shape[-1] != self.dv:
            raise ValueError(f"expected dv={self.dv}, got {ref_img_tokens.shape[-1]}")
        B = ref_img_tokens.shape[0]
        # keep dtype consistent with the (bf16) model weights
        dtype = self.kv_proj.weight.dtype
        kv = self.kv_proj(ref_img_tokens.to(dtype))            # [B, cond_len, da]
        q = self.slots.to(dtype).unsqueeze(0).expand(B, -1, -1)  # [B, K, da]
        c, _ = self.attn(q, kv, kv, need_weights=False)         # [B, K, da]
        return self.norm(c)

    @torch.no_grad()
    def build_concept_ids(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Position ids for the K concept slots, matching the action expert's
        4-dim id layout (dim0=modality tag, dim1=position). Uses modality tag
        3.0 to distinguish concept slots from action tokens (which use 2.0)."""
        ids = torch.zeros(batch_size, self.K, 4, device=device, dtype=dtype)
        ids[..., 0] = 3.0
        ids[..., 1] = torch.arange(self.K, device=device, dtype=dtype)[None, :]
        return ids


if __name__ == "__main__":
    # ---- unit test: shapes + grad flow ----
    torch.manual_seed(0)
    B, cond_len, dv, da, K = 2, 64, 3072, 1024, 16
    cb = ConceptBottleneck(dv=dv, da=da, K=K)
    x = torch.randn(B, cond_len, dv, requires_grad=True)
    c = cb(x)
    assert c.shape == (B, K, da), c.shape
    ids = cb.build_concept_ids(B, x.device, torch.float32)
    assert ids.shape == (B, K, 4), ids.shape
    # grad flows back to input and params
    loss = c.pow(2).mean()
    loss.backward()
    assert x.grad is not None and cb.kv_proj.weight.grad is not None
    n_params = sum(p.numel() for p in cb.parameters())
    print(f"[ok] c_t {tuple(c.shape)}, ids {tuple(ids.shape)}, params={n_params:,}, "
          f"grad(slots)={cb.slots.grad is not None}, ids[...,0].unique={ids[...,0].unique().tolist()}")
