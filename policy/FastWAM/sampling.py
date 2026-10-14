"""XPolicyLab inference extensions on FastWAM's native action-only flow grid.

The frozen first-frame video KV cache is shared across evaluations. Native time
goes from noise to data and velocity is noise minus clean. These are local
RTC/PAINT/AutoHorizon adaptations, not upstream FastWAM algorithms.
"""

from contextlib import nullcontext
import math

import numpy as np
import torch

from XPolicyLab.utils.flow_sampling import ActionAttentionCapture, rtc_correction


def normalize_actions(policy, actions):
    """Use action statistics, never the proprio/state statistics."""
    processor = policy.processor
    if processor.action_state_transforms or processor.delta_action_dim_mask is not None:
        raise ValueError("FastWAM sampling currently requires absolute untransformed actions")
    meta = processor.shape_meta["action"]
    if len(meta) != 1:
        raise ValueError("FastWAM sampling requires one merged action key")
    normalizer = processor.normalizer.normalizers["action"][meta[0]["key"]]
    return normalizer.forward(torch.as_tensor(actions, dtype=torch.float32)[None])


class ActionFlowSampler:
    def __init__(self, mode, *, target=None, weights=None, beta=5.0, delay=None):
        if mode not in {"rtc", "paint", "autohorizon"}:
            raise ValueError(f"Unsupported FastWAM sampling mode: {mode}")
        self.mode, self.target, self.weights = mode, target, weights
        self.beta, self.delay = beta, delay
        self.metadata = {}

    def __call__(self, noise, timesteps, deltas, *, velocity, scheduler, attention_owner):
        if timesteps.ndim != 1 or timesteps.shape != deltas.shape or not len(timesteps):
            raise ValueError("FastWAM requires a nonempty native action schedule")
        target = None if self.target is None else torch.as_tensor(self.target).to(noise)
        if target is not None and (target.shape != noise.shape or not torch.isfinite(target).all()):
            raise ValueError("FastWAM normalized action condition must match [1,H,D] and be finite")
        weights = None
        if self.mode == "rtc":
            w = np.asarray(self.weights, dtype=np.float32)
            if (w.shape != (noise.shape[1],) or not np.isfinite(w).all()
                    or np.any((w < 0) | (w > 1))):
                raise ValueError("RTC weights must be finite [H] values in [0,1]")
            if not math.isfinite(self.beta) or self.beta <= 0 or target is None:
                raise ValueError("RTC requires a finite positive beta and an action condition")
            weights = torch.as_tensor(w, device=noise.device, dtype=noise.dtype)[None, :, None]
        if self.mode == "paint" and (
            target is None or isinstance(self.delay, bool) or not isinstance(self.delay, int)
            or not 0 < self.delay < noise.shape[1]
        ):
            raise ValueError("PAINT requires a target and 0 < delay_steps < horizon")
        if self.mode == "autohorizon" and (len(timesteps) < 3 or noise.shape[1] < 2):
            raise ValueError("AutoHorizon requires at least three evaluations and two action tokens")
        capture = ActionAttentionCapture(noise.shape[1])

        def integrate(initial, times, steps, *, guided=False, capture_third=False):
            x = initial.detach().clone()
            for i, (time, delta) in enumerate(zip(times, steps)):
                context = capture.attach([attention_owner]) if capture_third and i == 2 else nullcontext()
                with context, torch.set_grad_enabled(guided):
                    sample = x.detach().requires_grad_(guided)
                    v = velocity(sample, time)
                    if guided:
                        sigma = time / float(scheduler.num_train_timesteps)
                        clean = sample - sigma * v
                        correction = rtc_correction(clean, sample, target, weights)
                        t = 1 - sigma
                        scale = ((t * t + sigma * sigma) / (t * sigma).clamp_min(1e-12)).clamp_max(self.beta)
                        v = v - scale * correction
                x = scheduler.step(v.detach(), delta, x).detach()
            return x

        if self.mode == "paint":
            naive = integrate(noise, timesteps, deltas)
            mask = torch.zeros_like(noise, dtype=torch.bool)
            mask[:, :self.delay] = True
            # Reverse the native endpoints/deltas, including clean time zero.
            reverse_times = torch.cat([timesteps.new_zeros(1), timesteps[1:].flip(0)])
            inverse = integrate(torch.where(mask, target, naive), reverse_times, -deltas.flip(0))
            result = integrate(torch.where(mask, inverse, noise), timesteps, deltas)
            self.metadata = {
                "delay_steps": self.delay, "num_steps": len(timesteps),
                "model_evaluations": 3 * len(timesteps),
                "inversion": "backward_euler_native_action_flow",
            }
            return result
        # With zero guidance use exactly the native no-grad execution path.
        guided = self.mode == "rtc" and bool(torch.any(weights != 0))
        result = integrate(noise, timesteps, deltas, guided=guided,
                           capture_third=self.mode == "autohorizon")
        if self.mode == "autohorizon":
            self.metadata = capture.metadata()
        return result
