"""Current-frame TCP pose readout (obs-only, horizon k=0)."""

from __future__ import annotations

import torch
import torch.nn as nn


def remap_legacy_tcp_pool_state_dict(state_dict, prefix: str = ""):
    """Map pre-refactor LatentAux pool keys onto ``pose_head.*``.

    Old checkpoints stored bimanual pool weights at the LatentAux root:
    ``tcp_queries``, ``tcp_pool_attn.*``, ``tcp_pool_norm.*``.
    """
    if not prefix:
        rel_keys = list(state_dict.keys())
    else:
        rel_keys = [k for k in state_dict if k.startswith(prefix)]

    remapped = dict(state_dict)
    for key in rel_keys:
        rel = key[len(prefix):] if prefix else key
        if rel == "tcp_queries" or rel.startswith("tcp_pool_attn.") or rel.startswith("tcp_pool_norm."):
            new_key = f"{prefix}pose_head.{rel}"
            if new_key not in remapped:
                remapped[new_key] = remapped.pop(key)
    return remapped


class TCPPoseHead(nn.Module):
    """Patch-token pooling + per-frame TCP pose readout."""

    def __init__(self, embed_dim: int, num_tcp: int = 1, pool_num_heads: int = 4):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_tcp = int(num_tcp)
        if self.num_tcp < 1:
            raise ValueError(f"num_tcp must be >= 1, got {num_tcp}")

        # num_tcp>1 (bimanual): learned query attention pool per arm.
        # num_tcp==1 keeps mean pool with no extra parameters.
        if self.num_tcp > 1:
            self.tcp_queries = nn.Parameter(torch.randn(self.num_tcp, self.embed_dim) * 0.02)
            self.tcp_pool_attn = nn.MultiheadAttention(
                self.embed_dim, int(pool_num_heads), batch_first=True
            )
            self.tcp_pool_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.tcp_queries = None
            self.tcp_pool_attn = None
            self.tcp_pool_norm = None

        self.uv_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, 2), nn.Sigmoid(),
        )
        self.pos3d_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, 3),
        )
        self.rot6d_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, 6),
        )
        self.valid_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[BT, HW, D]`` -> ``[BT, D]`` or ``[BT, num_tcp, D]``."""
        if self.num_tcp > 1:
            q = self.tcp_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            pooled, _ = self.tcp_pool_attn(q, tokens, tokens, need_weights=False)
            return self.tcp_pool_norm(pooled)
        return tokens.mean(dim=1)

    def forward_tokens(self, tokens: torch.Tensor) -> dict:
        """``[BT, HW, D]`` patch tokens -> TCP pose dict."""
        return self.forward_pooled(self.pool_tokens(tokens))

    def forward_pooled(self, x_bt_d: torch.Tensor) -> dict:
        """``[BT, D]`` or ``[BT, num_tcp, D]`` -> dict of pose tensors."""
        valid_logit = self.valid_head(x_bt_d)
        return {
            "tcp_uv": self.uv_head(x_bt_d),
            "tcp_3d": self.pos3d_head(x_bt_d),
            "tcp_6d": self.rot6d_head(x_bt_d),
            "tcp_valid_logit": valid_logit,
            "tcp_valid": torch.sigmoid(valid_logit),
        }
