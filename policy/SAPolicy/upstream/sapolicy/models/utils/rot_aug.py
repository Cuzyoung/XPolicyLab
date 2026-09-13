"""Top-camera roll augmentation with geometry-consistent label updates.

Rotating the image by phi about the principal point is exactly a camera roll:
P_cam' = Rz(phi) P_cam, K unchanged.  The same phi is therefore applied to
    * the image (bilinear resample, zero fill; depth: nearest),
    * TCP pixel coords (normalized by (W-1, H-1), pixel map derived from K),
    * camera-frame TCP position / orientation (Rz(phi) P, Rz(phi) R),
    * camera-to-world extrinsics (T_c2w Rz(-phi)),
    * future TCP pixel coords (same phi per sample).
Mirrors the official ABC train-time +-5 deg top-camera rotation (abc_minimal/train_loop.py
augment_and_normalize), which has no labels to keep consistent.
"""
import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


_LOGGED = False


def _default_cam_filter(name: str) -> bool:
    return "top" in name and "eye_in_hand" not in name


def _k_params(K: torch.Tensor):
    if K.ndim == 4:  # [B, T, 3, 3] -> per-episode constant, take t=0
        K = K[:, 0]
    return K[:, 0, 0].float(), K[:, 1, 1].float(), K[:, 0, 2].float(), K[:, 1, 2].float()


def _rz(phi: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(phi), torch.sin(phi)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def rotate_uv(uv: torch.Tensor, phi: torch.Tensor, fx, fy, cx, cy, W: int, H: int) -> torch.Tensor:
    """uv [..., >=2] normalized by (W-1, H-1); phi [B]; returns same shape/normalization (z untouched)."""
    B = uv.shape[0]
    flat = uv.reshape(B, -1, uv.shape[-1]).clone()
    u = flat[..., 0] * (W - 1) - cx[:, None]
    v = flat[..., 1] * (H - 1) - cy[:, None]
    c, s = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
    r = (fx / fy)[:, None]
    u2 = cx[:, None] + c * u - s * r * v
    v2 = cy[:, None] + s / r * u + c * v
    flat[..., 0] = u2 / (W - 1)
    flat[..., 1] = v2 / (H - 1)
    return flat.reshape(uv.shape)


def apply_top_rotation_aug(
    x: Dict[str, torch.Tensor],
    depths: Optional[Dict[str, torch.Tensor]],
    camera_intrinsics: Dict[str, torch.Tensor],
    tcp_pixel_coords: Optional[Dict[str, torch.Tensor]],
    tcp_pos: Optional[Dict[str, torch.Tensor]],
    tcp_orn: Optional[Dict[str, torch.Tensor]],
    future_tcp_pixel_coords: Optional[Dict[str, torch.Tensor]],
    camera_extrinsics: Optional[Dict[str, torch.Tensor]],
    max_deg: float,
    cam_filter: Callable[[str], bool] = _default_cam_filter,
    angles: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[dict, Optional[dict], Optional[dict], Optional[dict], Optional[dict], Optional[dict], Optional[dict], Dict[str, torch.Tensor]]:
    """Returns rotated copies of the dicts (untouched entries are the same objects) and the angles used (rad)."""
    x = dict(x)
    depths = dict(depths) if depths is not None else None
    tcp_pixel_coords = dict(tcp_pixel_coords) if tcp_pixel_coords is not None else None
    tcp_pos = dict(tcp_pos) if tcp_pos is not None else None
    tcp_orn = dict(tcp_orn) if tcp_orn is not None else None
    future_tcp_pixel_coords = dict(future_tcp_pixel_coords) if future_tcp_pixel_coords is not None else None
    camera_extrinsics = dict(camera_extrinsics) if camera_extrinsics is not None else None
    used: Dict[str, torch.Tensor] = {}
    global _LOGGED
    for cam, img in x.items():
        if not cam_filter(cam) or camera_intrinsics is None or camera_intrinsics.get(cam) is None:
            continue
        if img.ndim != 5:
            raise ValueError(f"rotation aug expects x[{cam!r}] as [B, T, C, H, W]; got {tuple(img.shape)}")
        B, T, C, H, W = img.shape
        dev = img.device
        fx, fy, cx, cy = _k_params(camera_intrinsics[cam].to(dev))
        if angles is not None and cam in angles:
            phi = angles[cam].to(dev).float()
        else:
            phi = (torch.rand(B, device=dev) * 2.0 - 1.0) * (max_deg * math.pi / 180.0)
        used[cam] = phi
        c, s = torch.cos(phi), torch.sin(phi)
        # Inverse map: output pixel (u', v') -> source pixel, i.e. rotation by -phi about (cx, cy).
        vs, us = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        du = us[None] - cx[:, None, None]
        dv = vs[None] - cy[:, None, None]
        r = (fx / fy)[:, None, None]
        u_src = cx[:, None, None] + c[:, None, None] * du + s[:, None, None] * r * dv
        v_src = cy[:, None, None] - s[:, None, None] / r * du + c[:, None, None] * dv
        grid = torch.stack([(u_src + 0.5) / W * 2.0 - 1.0, (v_src + 0.5) / H * 2.0 - 1.0], dim=-1)  # [B, H, W, 2]
        grid_bt = grid.unsqueeze(1).expand(B, T, H, W, 2).reshape(B * T, H, W, 2)
        img_bt = img.reshape(B * T, C, H, W)
        out = F.grid_sample(img_bt.float(), grid_bt, mode="bilinear", padding_mode="zeros", align_corners=False)
        x[cam] = out.to(img.dtype).reshape(B, T, C, H, W)
        if depths is not None and depths.get(cam) is not None:
            d = depths[cam]
            d5 = d if d.ndim == 5 else d.unsqueeze(2)
            Bd, Td, Cd, Hd, Wd = d5.shape
            if (Hd, Wd) == (H, W) and Bd == B:
                gd = grid.unsqueeze(1).expand(B, Td, H, W, 2).reshape(B * Td, H, W, 2)
                d_out = F.grid_sample(d5.reshape(B * Td, Cd, H, W).float(), gd, mode="nearest", padding_mode="zeros", align_corners=False)
                d_out = d_out.to(d.dtype).reshape(B, Td, Cd, H, W)
                depths[cam] = d_out if d.ndim == 5 else d_out.squeeze(2)
        for dct in (tcp_pixel_coords, future_tcp_pixel_coords):
            if dct is not None and dct.get(cam) is not None:
                dct[cam] = rotate_uv(dct[cam].to(dev), phi, fx, fy, cx, cy, W, H)
        R = _rz(phi)  # [B, 3, 3]
        if tcp_pos is not None and tcp_pos.get(cam) is not None:
            p = tcp_pos[cam].to(dev)
            pf = p.reshape(B, -1, 3)
            tcp_pos[cam] = torch.einsum("bij,bnj->bni", R.to(pf.dtype), pf).reshape(p.shape)
        if tcp_orn is not None and tcp_orn.get(cam) is not None:
            o = tcp_orn[cam].to(dev)
            of = o.reshape(B, -1, 3, 3)
            tcp_orn[cam] = torch.einsum("bij,bnjk->bnik", R.to(of.dtype), of).reshape(o.shape)
        if camera_extrinsics is not None and camera_extrinsics.get(cam) is not None:
            e = camera_extrinsics[cam].to(dev)  # camera-to-world
            Rinv4 = torch.eye(4, device=dev, dtype=e.dtype).repeat(B, 1, 1)
            Rinv4[:, :3, :3] = _rz(-phi).to(e.dtype)
            ef = e.reshape(B, -1, 4, 4)
            camera_extrinsics[cam] = torch.einsum("bnij,bjk->bnik", ef, Rinv4).reshape(e.shape)
    if used and not _LOGGED:
        _LOGGED = True
        print(f"[rot_aug] top-camera roll augmentation active: cams={list(used)} max_deg={max_deg} "
              f"first-batch |phi| mean={torch.cat([v.abs() for v in used.values()]).mean().item()*180/math.pi:.2f} deg", flush=True)
    return x, depths, tcp_pixel_coords, tcp_pos, tcp_orn, future_tcp_pixel_coords, camera_extrinsics, used
