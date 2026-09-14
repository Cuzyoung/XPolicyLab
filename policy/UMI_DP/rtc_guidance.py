"""Inpainting guidance for the DDIM sampler of ``DiffusionUnetTimmPolicy``.

The published RTC implementations all guide a *flow* policy: the sampler
integrates ``x += v * dt`` from noise at ``t=0`` to actions at ``t=1``, and
Eq. 2 adds ``min(beta, (1-t) / (t * r_t^2)) * g`` to the velocity, where
``g`` is the vector-Jacobian product of the weighted inpainting error
through the clean-sample estimate ``f(x) = x + (1-t) v``.

This checkpoint is a DDIM epsilon-prediction diffusion policy, so the same
guidance has to be restated for a variance-preserving forward process.  With
``s_t = sqrt(alpha_bar_t)`` the signal scale and ``sigma_t =
sqrt(1 - alpha_bar_t)`` the noise scale:

* the clean-sample estimate is ``x0 = (x_t - sigma_t * eps) / s_t``, which is
  the exact analogue of ``f(x)`` above and is what ``scheduler.step`` itself
  re-derives internally;
* PiGDM's guidance on the score is ``(d x0 / d x_t)^T [W * (Y - x0)] / r_t^2``
  with ``r_t^2 = sigma_t^2 / (sigma_t^2 + s_t^2)``.  A VP process has
  ``s_t^2 + sigma_t^2 = 1``, so ``r_t^2 = sigma_t^2``;
* ``eps = -sigma_t * score``, so folding that guidance into the model output
  gives ``eps_guided = eps - (sigma_t / r_t^2) * g = eps - g / sigma_t``.

The paper's ``beta`` clip carries over unchanged as
``min(beta, 1 / sigma_t)``: the coefficient diverges as ``sigma_t -> 0``
exactly like ``(1-t)/(t r_t^2)`` diverges at the end of a flow, and beta is
what keeps the last denoising steps from snapping the whole chunk onto the
old plan.

Guiding costs one backward pass through the U-Net per denoising step, so a
guided chunk is roughly three times the wall clock of an unguided one.
``soft_inpaint`` is the cheap alternative: it blends the clean estimate
towards the target with the same weights and needs no backward pass.  It is
what the guided update converges to as ``beta -> inf``, without the
distribution-aware correction, and exists so a cell whose inference budget
cannot absorb the backward pass can still run real-time chunking.

Nothing here imports the runtime: the sampler is wrapped in place, the
condition lives on the wrapper, and a policy that is never handed a
condition samples exactly as it did before.
"""
import contextlib
import math

import numpy as np

GUIDANCE_MODES = ('pigdm', 'soft_inpaint')
DEFAULT_BETA = 5.0


def guidance_scale(sigma, beta):
    """``min(beta, 1 / sigma_t)``, the VP restatement of Eq. 2's clip."""
    if sigma <= 0.0:
        return float(beta)
    return min(float(beta), 1.0 / float(sigma))


class RtcGuidance:
    """A DDIM policy whose sampler accepts one RTC condition at a time.

    The policy object is patched, not subclassed: ``predict_action`` calls
    ``self.conditional_sample`` and the runtime holds the policy instance
    that ``policy_runner`` built from the checkpoint.  Wrapping keeps the
    unconditioned path bit-identical -- with no active condition the wrapper
    delegates straight to the original bound method.
    """

    def __init__(self, policy, mode='pigdm', beta=DEFAULT_BETA):
        import torch  # imported lazily: mask/frames stay torch-free

        if mode not in GUIDANCE_MODES:
            raise ValueError(f'unknown RTC guidance mode {mode!r}')
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f'rtc beta must be finite and positive, got {beta}')
        self._torch = torch
        self.policy = policy
        self.mode = mode
        self.beta = float(beta)
        self._active = None
        self._original = policy.conditional_sample

        def wrapped(condition_data, condition_mask, local_cond=None,
                    global_cond=None, generator=None, **kwargs):
            if self._active is None:
                return self._original(condition_data, condition_mask,
                                      local_cond=local_cond,
                                      global_cond=global_cond,
                                      generator=generator, **kwargs)
            return self._guided_sample(condition_data, condition_mask,
                                       local_cond, global_cond, generator,
                                       **kwargs)

        policy.conditional_sample = wrapped

    def detach(self):
        """Restore the checkpoint's own sampler."""
        self.policy.conditional_sample = self._original
        self._active = None

    @contextlib.contextmanager
    def condition(self, action_condition, condition_weights, beta=None):
        """Activate one condition for the duration of a single predict call.

        ``action_condition`` is ``(H, D)`` in RAW action units -- the same
        units ``ActionAdapter.decode`` consumes -- already rebased into the
        frame of the observation being sent (see ``rtc/frames.py``).  It is
        normalised here with the checkpoint's own action normaliser, because
        the sampler works in normalised space and the runtime must never
        carry a second copy of those statistics.

        Passing ``None`` for both yields ``False`` and leaves the sampler
        untouched, so a caller can use one code path for the first chunk of
        a run and every chunk after it.
        """
        torch = self._torch
        if action_condition is None and condition_weights is None:
            yield False
            return
        if action_condition is None or condition_weights is None:
            raise ValueError('action_condition and condition_weights must be sent together')
        if self._active is not None:
            raise RuntimeError('an RTC condition is already active')
        beta = self.beta if beta is None else float(beta)
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f'rtc beta must be finite and positive, got {beta}')

        horizon = int(self.policy.action_horizon)
        dim = int(self.policy.action_dim)
        target = np.asarray(action_condition, dtype=np.float32)
        if target.shape != (horizon, dim):
            raise ValueError(
                f'action_condition must have shape {(horizon, dim)}, got {target.shape}')
        if not np.isfinite(target).all():
            raise ValueError('action_condition contains non-finite values')
        weights = np.array(condition_weights, dtype=np.float32).reshape(-1)
        if weights.shape != (horizon,):
            raise ValueError(
                f'condition_weights must have shape {(horizon,)}, got {weights.shape}')
        if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError('condition_weights must be finite values in [0, 1]')

        device = self.policy.device
        dtype = self.policy.dtype
        raw = torch.from_numpy(target).to(device=device, dtype=dtype).unsqueeze(0)
        normalized = self.policy.normalizer['action'].normalize(raw)
        self._active = (
            normalized.detach(),
            torch.from_numpy(weights).to(device=device, dtype=dtype).view(1, -1, 1),
            beta,
        )
        try:
            yield True
        finally:
            self._active = None

    def _guided_sample(self, condition_data, condition_mask, local_cond,
                       global_cond, generator, **kwargs):
        torch = self._torch
        target, weights, beta = self._active
        model = self.policy.model
        scheduler = self.policy.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype,
            device=condition_data.device, generator=generator)
        if target.shape[0] != trajectory.shape[0]:
            target = target.expand(trajectory.shape[0], -1, -1)
        if target.shape != trajectory.shape:
            raise ValueError(
                f'normalised condition {tuple(target.shape)} does not match the '
                f'sampled chunk {tuple(trajectory.shape)}')

        scheduler.set_timesteps(self.policy.num_inference_steps)
        alphas = scheduler.alphas_cumprod.to(device=trajectory.device,
                                             dtype=trajectory.dtype)
        for t in scheduler.timesteps:
            # Kept for parity with the checkpoint's own sampler; the RTC path
            # never sets a hard mask, it is the soft weights that condition.
            trajectory[condition_mask] = condition_data[condition_mask]
            alpha_bar = alphas[int(t)]
            signal = torch.sqrt(alpha_bar)
            sigma = torch.sqrt(1.0 - alpha_bar)

            if self.mode == 'pigdm':
                # predict_action runs under no_grad, not inference_mode, so
                # the chunk can be made a leaf here.  Only the U-Net's action
                # path is differentiated, and only with respect to the noisy
                # chunk -- no parameter gradients are ever produced.
                with torch.enable_grad():
                    noisy = trajectory.detach().requires_grad_(True)
                    epsilon = model(noisy, t, local_cond=local_cond,
                                    global_cond=global_cond)
                    clean = (noisy - sigma * epsilon) / signal
                    error = ((target - clean) * weights).detach()
                    correction = torch.autograd.grad(
                        clean, noisy, grad_outputs=error,
                        create_graph=False, retain_graph=False)[0]
                model_output = epsilon.detach() - guidance_scale(
                    float(sigma), beta) * correction
            else:
                epsilon = model(trajectory, t, local_cond=local_cond,
                                global_cond=global_cond)
                clean = (trajectory - sigma * epsilon) / signal
                blended = (1.0 - weights) * clean + weights * target
                # Back to the epsilon the scheduler expects, so the DDIM step
                # itself stays the checkpoint's own arithmetic.
                model_output = (trajectory - signal * blended) / sigma

            trajectory = scheduler.step(model_output, t, trajectory,
                                        generator=generator, **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory
