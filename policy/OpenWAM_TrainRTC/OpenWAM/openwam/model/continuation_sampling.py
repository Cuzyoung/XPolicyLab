"""Continuation sampling in OpenWAM's decreasing noise-time convention.

The native video schedule and first-frame constraint are retained. Action
updates start from the conditioned input at EVERY step, as in the references.
No final hard clamp is added: the caller owns the already committed prefix.
"""
import math
import torch


class ContinuationFlowSampler:
    def __init__(self, architecture, *, target=None, prefix_steps=0):
        self.arch = architecture
        self.cfg = architecture.continuation
        if self.cfg.method not in {"train_time_rtc", "legato"}:
            raise ValueError("Continuation sampling requires a continuation-trained checkpoint")
        if type(prefix_steps) is not int or prefix_steps < 0:
            raise ValueError("prefix_steps must be a nonnegative integer")
        self.target = target
        self.prefix_steps = prefix_steps
        self.metadata = {}

    @torch.no_grad()
    def __call__(self, schedule, inputs, noise, *, dit_cache=None,
                 cfg_scale_f=1.0, cfg_merge=False, inactive_action_dims=None):
        if dit_cache is not None or cfg_scale_f != 1.0:
            raise ValueError("Continuation sampling requires cache disabled and CFG scale 1")
        cfg, arch = self.cfg, self.arch
        batch, horizon, width = noise.shape
        if self.prefix_steps + (cfg.ramp_steps if self.prefix_steps else 0) >= horizon:
            raise ValueError("Continuation prefix and ramp must leave a free suffix")
        if self.prefix_steps > cfg.prefix_max_steps:
            raise ValueError("prefix_steps exceeds the checkpoint training range")
        nfa = float(arch.action_scheduler.num_train_timesteps)
        nfv = float(arch.video_scheduler.num_train_timesteps)
        grid = [(float(v)/nfv, float(a)/nfa) for v, a in schedule]
        if len(grid) < 2 or grid[-1] != (0.0, 0.0) or not all(
            math.isfinite(v) and math.isfinite(a) and 0 <= v <= 1 and 0 <= a <= 1 for v,a in grid
        ) or not math.isclose(grid[0][1], 1.0, abs_tol=1e-6):
            raise ValueError("Continuation requires a complete finite noise-to-data grid")
        if any(an >= a or vn > v for (v,a),(vn,an) in zip(grid, grid[1:])):
            raise ValueError("Continuation requires strictly decreasing action noise")
        if cfg.method == "legato":
            expected = cfg.num_denoise_steps
            if len(grid) - 1 != expected or any(
                not math.isclose(a, 1-i/expected, abs_tol=2e-6) for i, (_,a) in enumerate(grid)
            ):
                raise ValueError("Legato inference grid must match the trained uniform num_denoise_steps")
        if self.prefix_steps:
            if self.target is None:
                raise ValueError("A nonempty prefix requires normalized action conditions")
            target = torch.as_tensor(self.target, dtype=noise.dtype, device=noise.device)
            if target.ndim == 2:
                target = target[None].expand(batch, -1, -1)
            if target.shape != noise.shape or not torch.isfinite(target).all():
                raise ValueError("Normalized action conditions must be finite [B,H,D] or [H,D]")
        else:
            if self.target is not None:
                raise ValueError("An empty prefix must not supply a target")
            target = torch.zeros_like(noise)
        positions = torch.arange(horizon, device=noise.device)[None, :]
        prefix = positions < self.prefix_steps
        omega = prefix.to(noise.dtype)
        if cfg.schedule == "linear" and self.prefix_steps:
            ramp = (self.prefix_steps + cfg.ramp_steps - positions) / (cfg.ramp_steps + 1)
            omega = torch.where(prefix, 1, ramp.clamp(0, 1)).to(noise.dtype)
        omega = omega[..., None].expand(batch, -1, -1)
        initial_noise = noise.clone()
        x = noise.clone()
        for (tv,ta), (vn,an) in zip(schedule, grid[1:]):
            sv,sa = float(tv)/nfv, float(ta)/nfa
            # This addition is to the guided input, not to the old unconditioned x.
            guided = omega * target + (1 - omega) * x
            at = torch.full((batch,), float(ta), dtype=x.dtype, device=x.device)
            if cfg.method == "train_time_rtc":
                at = torch.where(prefix, 0, at[:, None])
                model_input = guided
            else:
                model_input = torch.cat((guided, 1-omega), dim=-1)
            vt = torch.full((batch,), float(tv), dtype=x.dtype, device=x.device)
            vv, va = arch.forward(model_input, at, **inputs, timestep=vt)
            if va is None or va.shape != noise.shape:
                raise RuntimeError("Continuation model must output D-dimensional action velocity")
            x = guided + (an-sa)*va
            if inactive_action_dims is not None:
                x[..., inactive_action_dims] = initial_noise[..., inactive_action_dims]*an
            if vn != sv:
                video = inputs["latents"] + (vn-sv)*vv
                ref = inputs.get("first_frame_latents")
                if ref is not None:
                    video = video.clone()
                    video[:, :, :ref.shape[2]] = ref
                inputs["latents"] = video
        self.metadata = {"method": cfg.method, "reference": cfg.reference,
                         "prefix_steps": self.prefix_steps, "num_steps": len(grid)-1,
                         "model_evaluations": len(grid)-1, "schedule": cfg.schedule,
                         "ramp_steps": cfg.ramp_steps, "final_prefix_clamped": False}
        return x
