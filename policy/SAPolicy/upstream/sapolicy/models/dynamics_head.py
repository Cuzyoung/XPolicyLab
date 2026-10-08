"""Observation-conditioned dynamics head for future TCP + semantic prediction.

Predicts state at horizon ``T`` (= action chunk size) directly from current
``attended_patch`` tokens (+ stopgrad backbone features for semantic cond).
No explicit action input and no separate dynamics encoder.

Two decoders:
  - ``future_tcp_decoder`` (``TCPPoseHead``): future TCP pose at ``t+T``.
  - ``future_semantic`` (RAE-style wide DiT + DDT head): flow-matching diffusion on future
    DINO features at ``t+T`` (see Representation Autoencoders, arXiv:2510.11690).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sapolicy.models.tcp_pose_head import TCPPoseHead
from sapolicy.models.transformer import DiTBlock, RMSNorm
from sapolicy.models.utils.pos_embed import SinusoidalPositionEmbeddings, add_pos_embed


def _patch_tokens(fmap: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` -> ``[B, H*W, C]``."""
    if fmap.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(fmap.shape)}")
    return fmap.flatten(2).transpose(1, 2).contiguous()


class FutureSemanticDDT(nn.Module):
    """RAE-style wide DiT + lightweight wide DDT head for feature-space flow matching."""

    def __init__(
        self,
        feat_dim: int,
        q_dim: int,
        wide_dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
        num_inference_steps: int = 4,
    ):
        super().__init__()
        if wide_dim % num_heads != 0:
            raise ValueError(f"wide_dim={wide_dim} must be divisible by num_heads={num_heads}")
        self.feat_dim = int(feat_dim)
        self.q_dim = int(q_dim)
        self.wide_dim = int(wide_dim)
        self.num_inference_steps = int(num_inference_steps)

        self.in_proj = nn.Linear(self.feat_dim, self.wide_dim)
        self.q_cond_proj = nn.Linear(self.q_dim, self.wide_dim)
        self.feat_cond_proj = nn.Linear(self.feat_dim, self.wide_dim)
        self.timestep_embed = SinusoidalPositionEmbeddings(self.wide_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(self.wide_dim, self.wide_dim),
            nn.SiLU(),
            nn.Linear(self.wide_dim, self.wide_dim * 9),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=self.wide_dim,
                num_heads=num_heads,
                dropout=dropout,
                conditional=True,
            )
            for _ in range(num_layers)
        ])
        ddt_hidden = self.wide_dim * 2
        self.ddt_norm = RMSNorm(self.wide_dim)
        self.ddt_fc1 = nn.Linear(self.wide_dim, ddt_hidden, bias=False)
        self.ddt_fc2 = nn.Linear(self.wide_dim, ddt_hidden, bias=False)
        self.ddt_out = nn.Linear(ddt_hidden, self.feat_dim, bias=False)

    def _build_cond(self, q_tokens: torch.Tensor, feat_t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        q = _patch_tokens(q_tokens)
        feat = _patch_tokens(feat_t.detach())
        if q.shape[1] != feat.shape[1]:
            raise ValueError(
                f"Spatial token count mismatch: q={q.shape[1]} feat={feat.shape[1]}"
            )
        cond = self.q_cond_proj(q) + self.feat_cond_proj(feat)
        return add_pos_embed(cond, w, h)

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        t_scalar: torch.Tensor,
        cond: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """``x_t`` is feat-dim tokens ``[B,HW,feat_dim]``; returns velocity in the same space."""
        t_emb = self.timestep_embed(t_scalar)
        e = self.timestep_mlp(t_emb).view(x_t.shape[0], 1, 9, self.wide_dim)
        x = add_pos_embed(self.in_proj(x_t), w, h)
        for block in self.blocks:
            x = block(x, e, cond=cond)
        h_dec = self.ddt_norm(x)
        return self.ddt_out(F.silu(self.ddt_fc1(h_dec)) * self.ddt_fc2(h_dec))

    def forward_training(
        self,
        feat_future: torch.Tensor,
        q_tokens: torch.Tensor,
        feat_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Flow match future DINO features ``feat_future`` at ``[B,C,H,W]``."""
        if feat_future.ndim != 4:
            raise ValueError(f"feat_future must be [B,C,H,W], got {tuple(feat_future.shape)}")
        b, c, h, w = feat_future.shape
        if c != self.feat_dim:
            raise ValueError(f"feat_future C={c} != feat_dim={self.feat_dim}")

        # Flow-match in feat_dim; Wide DiT is only the internal backbone.
        x0 = _patch_tokens(feat_future)
        cond = self._build_cond(q_tokens, feat_t, h, w)

        t = torch.rand(b, device=x0.device, dtype=x0.dtype)
        noise = torch.randn_like(x0)
        t_view = t.view(b, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * noise
        target = noise - x0
        pred = self._predict_velocity(x_t, t, cond, h, w)
        loss = F.mse_loss(pred, target)
        stats = {
            "loss_future_semantic": loss.detach(),
            "loss_future_semantic_baseline_zero": target.pow(2).mean().detach(),
        }
        return loss, stats


class DynamicsHead(nn.Module):
    """Future TCP + semantic dynamics at a single horizon ``T``."""

    def __init__(
        self,
        q_dim: int,
        feat_dim: int,
        prediction_horizon: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        predict_tcp: bool = True,
        predict_semantic: bool = True,
        num_tcp: int = 1,
        pool_num_heads: int = 4,
        wide_dim: int = 2048,
        num_dit_layers: int = 4,
        num_inference_steps: int = 4,
        tcp_uv_weight: float = 1.0,
        tcp_pos3d_weight: float = 1.0,
        tcp_rot6d_weight: float = 1.0,
        tcp_valid_weight: float = 0.2,
        embed_dim: int = 512,
        num_cross_layers: int = 2,
    ):
        super().__init__()
        del embed_dim, num_cross_layers  # legacy cfg keys; direct decode only
        self.q_dim = int(q_dim)
        self.feat_dim = int(feat_dim)
        self.prediction_horizon = int(prediction_horizon)
        if self.prediction_horizon < 1:
            raise ValueError(f"prediction_horizon must be >= 1, got {prediction_horizon}")
        self.predict_tcp = bool(predict_tcp)
        self.predict_semantic = bool(predict_semantic)
        if not self.predict_tcp and not self.predict_semantic:
            raise ValueError("DynamicsHead requires at least one of predict_tcp / predict_semantic")

        self.future_tcp_decoder = None
        if self.predict_tcp:
            self.future_tcp_decoder = TCPPoseHead(
                self.q_dim,
                num_tcp=int(num_tcp),
                pool_num_heads=int(pool_num_heads),
            )

        self.future_semantic = None
        if self.predict_semantic:
            self.future_semantic = FutureSemanticDDT(
                feat_dim=self.feat_dim,
                q_dim=self.q_dim,
                wide_dim=int(wide_dim),
                num_heads=int(num_heads),
                num_layers=int(num_dit_layers),
                dropout=float(dropout),
                num_inference_steps=int(num_inference_steps),
            )

        self.tcp_uv_weight = float(tcp_uv_weight)
        self.tcp_pos3d_weight = float(tcp_pos3d_weight)
        self.tcp_rot6d_weight = float(tcp_rot6d_weight)
        self.tcp_valid_weight = float(tcp_valid_weight)

    def forward(
        self,
        q_tokens: torch.Tensor,
        feat_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference-style forward (TCP predictions only; semantic uses ``forward_training``)."""
        del feat_t
        q_patch = _patch_tokens(q_tokens)
        out: Dict[str, torch.Tensor] = {"q_patch_tokens": q_patch}
        if self.future_tcp_decoder is not None:
            tcp = self.future_tcp_decoder.forward_tokens(q_patch)
            out.update({f"future_{k}": v for k, v in tcp.items()})
        return out

    def compute_losses(
        self,
        q_tokens: torch.Tensor,
        feat_t: torch.Tensor,
        *,
        feat_future: Optional[torch.Tensor] = None,
        feat_future_valid: Optional[torch.Tensor] = None,
        gt_tcp_uv: Optional[torch.Tensor] = None,
        gt_tcp_3d: Optional[torch.Tensor] = None,
        gt_tcp_6d: Optional[torch.Tensor] = None,
        gt_tcp_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        device = q_tokens.device
        dtype = q_tokens.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        stats: Dict[str, torch.Tensor] = {
            "loss_future_tcp": zero,
            "loss_future_semantic": zero,
        }
        tcp_loss = zero
        semantic_loss = zero

        q_patch = _patch_tokens(q_tokens)

        if self.future_tcp_decoder is not None and gt_tcp_3d is not None:
            pred = self.future_tcp_decoder.forward_tokens(q_patch)
            # Drop trailing singleton dims left by horizon stacking ([B,1,*] -> [B,*]).
            def _match(pred_t: torch.Tensor, gt_t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if gt_t is None:
                    return None
                while gt_t.ndim > pred_t.ndim:
                    gt_t = gt_t.squeeze(1)
                if gt_t.shape != pred_t.shape:
                    gt_t = gt_t.reshape_as(pred_t)
                return gt_t

            gt_tcp_uv = _match(pred["tcp_uv"], gt_tcp_uv)
            gt_tcp_3d = _match(pred["tcp_3d"], gt_tcp_3d)
            gt_tcp_6d = _match(pred["tcp_6d"], gt_tcp_6d)
            loss_uv = F.l1_loss(pred["tcp_uv"], gt_tcp_uv) if gt_tcp_uv is not None else zero
            loss_3d = F.l1_loss(pred["tcp_3d"], gt_tcp_3d)
            loss_6d = F.l1_loss(pred["tcp_6d"], gt_tcp_6d)
            if gt_tcp_valid is not None and "tcp_valid_logit" in pred:
                logit = pred["tcp_valid_logit"].squeeze(-1)
                while gt_tcp_valid.ndim > logit.ndim:
                    gt_tcp_valid = gt_tcp_valid.squeeze(-1 if gt_tcp_valid.shape[-1] == 1 else 1)
                if gt_tcp_valid.shape != logit.shape:
                    gt_tcp_valid = gt_tcp_valid.reshape_as(logit)
                loss_valid = F.binary_cross_entropy_with_logits(
                    logit,
                    gt_tcp_valid.to(dtype=logit.dtype),
                )
            else:
                loss_valid = zero
            tcp_loss = (
                self.tcp_uv_weight * loss_uv
                + self.tcp_pos3d_weight * loss_3d
                + self.tcp_rot6d_weight * loss_6d
                + self.tcp_valid_weight * loss_valid
            )
            stats["loss_future_tcp"] = tcp_loss.detach()
            stats["loss_future_tcp_uv"] = loss_uv.detach()
            stats["loss_future_tcp_3d"] = loss_3d.detach()
            stats["loss_future_tcp_6d"] = loss_6d.detach()
            stats["loss_future_tcp_valid"] = loss_valid.detach()

        if self.future_semantic is not None and feat_future is not None:
            if feat_future_valid is not None and not bool(feat_future_valid.any()):
                semantic_loss = zero
                stats["loss_future_semantic"] = zero
            else:
                semantic_loss, sem_stats = self.future_semantic.forward_training(
                    feat_future,
                    q_tokens,
                    feat_t,
                )
                stats.update(sem_stats)

        stats["loss_future_dynamics"] = (tcp_loss + semantic_loss).detach()
        stats["loss_future_video"] = stats.get("loss_future_semantic", zero)
        return tcp_loss, semantic_loss, stats
