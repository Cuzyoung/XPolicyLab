"""Action conditioning while preserving OpenWAM's joint video/action flow grid.

PAINT integrates both streams for the naive/inverse/final passes. Repainting
retains free video noise and replaces only the conditioned action-noise rows.
RTC differentiates the clean action estimate at each native action noise level.
"""

from contextlib import nullcontext
import math

import numpy as np
import torch

from XPolicyLab.utils.flow_sampling import ActionAttentionCapture


def normalize_action(architecture, actions):
    normalizer = architecture.normalizer
    if normalizer is None:
        return np.asarray(actions)
    # Unified state and action maps need not be equal. normalize() is proprio-only.
    if hasattr(normalizer, "_action_dst_index"):
        raw = np.asarray(actions)
        if normalizer._inner is not None:
            raw = normalizer._inner.normalize(raw)
        return normalizer._map_to_unify(raw, normalizer._action_dst_index, normalizer._unify_dim)[0]
    return normalizer.normalize(np.asarray(actions))


class JointFlowSampler:
    def __init__(self, architecture, mode, *, target=None, weights=None, delay=None, beta=5.0):
        self.arch = architecture
        self.mode = mode
        self.target = target
        self.weights = weights
        self.delay = delay
        self.beta = beta
        self.metadata = {}

    def __call__(
        self,
        schedule,
        inputs,
        noise,
        *,
        dit_cache=None,
        cfg_scale_f=1.0,
        cfg_merge=False,
        inactive_action_dims=None,
    ):
        arch = self.arch
        if cfg_scale_f != 1.0:
            raise ValueError("Conditioned OpenWAM sampling currently requires cfg_scale=1")
        if len(schedule) < 2 or schedule[-1] != (0.0, 0.0):
            raise ValueError("OpenWAM sampling requires a grid ending at zero noise")
        if self.mode == "autohorizon" and len(schedule) < 4:
            raise ValueError("AutoHorizon requires at least three denoising evaluations")
        target = (
            None
            if self.target is None
            else torch.as_tensor(self.target, device=noise.device, dtype=noise.dtype)[None]
        )
        if target is not None and target.shape != noise.shape:
            raise ValueError("Normalized OpenWAM condition must match model action shape")
        if target is not None and not torch.isfinite(target).all():
            raise ValueError("OpenWAM condition must be finite")
        initial_sigma = float(schedule[0][1]) / float(arch.action_scheduler.num_train_timesteps)
        if initial_sigma <= 0:
            raise ValueError("OpenWAM sampling must start at positive action noise")
        inactive_noise = noise / initial_sigma
        valid = torch.ones_like(noise, dtype=torch.bool)
        if inactive_action_dims is not None:
            valid[..., inactive_action_dims] = False
        weights = None
        if self.weights is not None:
            w = np.asarray(self.weights)
            if (
                w.shape != (noise.shape[1],)
                or not np.isfinite(w).all()
                or np.any((w < 0) | (w > 1))
            ):
                raise ValueError("RTC weights must be finite [H] values in [0,1]")
            if not math.isfinite(self.beta) or self.beta <= 0:
                raise ValueError("RTC beta must be finite and positive")
            weights = (
                torch.as_tensor(w, device=noise.device, dtype=noise.dtype)[None, :, None] * valid
            )
        original_video = inputs["latents"].clone()
        capture = ActionAttentionCapture(noise.shape[1])
        nfv = float(arch.video_scheduler.num_train_timesteps)
        nfa = float(arch.action_scheduler.num_train_timesteps)

        def integrate(initial, grid, *, guided=False, capture_third=False):
            x = initial.clone()
            for i, ((tv, ta), (tv_next, ta_next)) in enumerate(zip(grid, grid[1:])):
                sv, sa = float(tv) / nfv, float(ta) / nfa
                dsv, dsa = float(tv_next) / nfv - sv, float(ta_next) / nfa - sa
                vt = torch.full((len(x),), float(tv), device=x.device, dtype=x.dtype)
                at = torch.full((len(x),), float(ta), device=x.device, dtype=x.dtype)
                attention_context = (
                    capture.attach([arch.mot_driver]) if capture_third and i == 2 else nullcontext()
                )
                with attention_context, torch.set_grad_enabled(guided):
                    sample = x.detach().requires_grad_(guided)
                    vvideo, vaction = arch.forward(sample, at, **inputs, timestep=vt)
                    if vaction is None:
                        raise RuntimeError("OpenWAM did not produce action velocity")
                    if guided:
                        clean = sample - sa * vaction
                        error = ((target - clean) * weights).detach()
                        correction = torch.autograd.grad(clean, sample, grad_outputs=error)[0]
                        time = 1 - sa
                        scale = (
                            self.beta
                            if time <= 0
                            else min(self.beta, (time * time + sa * sa) / max(time * sa, 1e-12))
                        )
                        vaction = vaction - scale * correction
                x = (x + dsa * vaction.detach()).detach()
                if inactive_action_dims is not None:
                    x[..., inactive_action_dims] = inactive_noise[..., inactive_action_dims] * (
                        float(ta_next) / nfa
                    )
                video = inputs["latents"] + dsv * vvideo.detach()
                ref = inputs.get("first_frame_latents")
                if ref is not None:
                    video = video.clone()
                    video[:, :, : ref.shape[2]] = ref
                inputs["latents"] = video.detach()
            return x

        if self.mode == "paint":
            naive = integrate(noise, schedule)
            mask = valid.clone()
            mask[:, self.delay :] = False
            inverse = integrate(torch.where(mask, target, naive), list(reversed(schedule)))
            inputs["latents"] = original_video
            result = integrate(torch.where(mask, inverse, noise), schedule)
            self.metadata = {
                "delay_steps": self.delay,
                "num_steps": len(schedule) - 1,
                "model_evaluations": 3 * (len(schedule) - 1),
                "inversion": "backward_euler_joint_flow",
            }
            return result
        result = integrate(
            noise, schedule, guided=self.mode == "rtc", capture_third=self.mode == "autohorizon"
        )
        if self.mode == "autohorizon":
            self.metadata = capture.metadata()
        return result
