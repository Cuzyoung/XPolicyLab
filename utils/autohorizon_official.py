"""AutoHorizon selector vendored from the pinned official implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


UPSTREAM_COMMIT = "c7504f1756109103f2cfcc2e23f1b1a23841c885"


@torch.no_grad()
def _soft_pointer_prefix(
    A: torch.Tensor,
    hold_thr: float = 0.3,
    run_len: int = 1,
    max_entropy_q: float = 0.9,
):
    """
    One-way soft pointer that advances until a plateau (prefix horizon).
    Returns: mu [T], dmu [T], reliable mask [T], stop_row (int)
    """
    assert A.ndim == 2 and A.size(0) == A.size(1)
    dev, T = A.device, A.size(0)
    eps = 1e-12

    A = A / (A.sum(-1, keepdim=True) + eps)

    idx = torch.arange(T, device=dev, dtype=A.dtype)
    mu = (A * idx).sum(-1)

    mu = torch.maximum(mu, torch.cummax(mu, dim=0).values)

    ent = -(A.clamp_min(eps) * (A.clamp_min(eps)).log()).sum(-1) / math.log(T)
    thr_ent = torch.quantile(ent.float(), q=min(max_entropy_q, 0.999))
    reliable = ent <= thr_ent

    prev = torch.cat([mu.new_tensor([0.0]), mu[:-1]])
    dmu = mu - prev
    dmu[0] = mu[0] - 0.0

    is_hold = (dmu < hold_thr) & reliable
    win = F.conv1d(
        is_hold.float()[None, None, :],
        torch.ones(1, 1, run_len, device=dev),
    ).squeeze()
    if win.numel() and (win >= run_len - 1e-6).any():
        stop_row = int(torch.nonzero(win >= run_len, as_tuple=False)[0].item())
    else:
        stop_row = T - 1

    return mu, dmu, reliable, stop_row


@torch.no_grad()
def bidir_soft_pointer(
    attn: torch.Tensor,
    hold_thr: float = 0.3,
    run_len: int = 1,
    max_entropy_q: float = 0.9,
):
    """
    Forward + backward soft pointers. If they overlap, stitch them to cover all actions.
    Returns:
        j_hat: [T] estimated key index per row (float, 0-based)
        join_row: int where we stitch (or None if no overlap)
        diags: dict with forward/backward mu, horizons, etc.
    """
    assert attn.ndim == 2 and attn.size(0) == attn.size(1)
    dev, T = attn.device, attn.size(0)
    eps = 1e-12

    A = attn / (attn.sum(-1, keepdim=True) + eps)

    mu_f, dmu_f, rel_f, stop_f = _soft_pointer_prefix(
        A,
        hold_thr=hold_thr,
        run_len=run_len,
        max_entropy_q=max_entropy_q,
    )
    j_f = mu_f
    N_f = int(torch.clamp(torch.floor(mu_f[min(stop_f, T - 1)]) + 1, 1, T).item())

    A_rev = torch.flip(A, dims=(0, 1))
    mu_b_rev, dmu_b_rev, rel_b_rev, stop_b_rev = _soft_pointer_prefix(
        A_rev,
        hold_thr=hold_thr,
        run_len=run_len,
        max_entropy_q=max_entropy_q,
    )
    mu_b = torch.flip(T - 1 - mu_b_rev, dims=(0,))
    j_b = mu_b
    N_b = int(torch.clamp(torch.floor(mu_b[max(T - 1 - stop_b_rev, 0)]) + 1, 1, T).item())

    gap = j_b - j_f
    meet_mask = gap <= 1.0
    join_row = None
    if meet_mask.any():
        join_row = int(torch.nonzero(meet_mask, as_tuple=False)[0].item())

    diags = dict(
        mu_forward=mu_f,
        dmu_forward=dmu_f,
        reliable_forward=rel_f,
        stop_forward=stop_f,
        N_forward=N_f,
        mu_backward=mu_b,
        stop_backward=T - 1 - stop_b_rev,
        N_backward=N_b,
        join_row=join_row,
        gap=gap,
        method="bidir_soft_pointer",
    )
    N_f = diags["N_forward"]
    N_b = diags["N_backward"]

    if (N_f + N_b >= T) and (join_row is not None):
        N = T
    else:
        N = N_f
    return torch.tensor(N), diags
