"""VP-diffusion form of pseudoinverse guidance with RTC's soft mask and cap.

For x_t=sqrt(alpha)*x0+sigma*epsilon, use the PiGDM Gaussian approximation
r_t^2=sigma^2. The score correction is J_x0^T W(y-x0)/r_t^2;
epsilon correction is -sigma times that score, capped by beta. The native
DDPM/DDIM scheduler still owns every step, variance injection and clipping.
This is a VP sampler adaptation, not the flow-Euler implementation in Pi05.
"""

import math
import torch


def guided_prediction(model, sample, timestep, scheduler, *, target, weights, beta, **model_kwargs):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("RTC beta must be finite and positive")
    if target.shape != sample.shape or weights.shape != sample.shape:
        raise ValueError("RTC target/weights must match the diffusion trajectory")
    if (
        not torch.isfinite(target).all()
        or not torch.isfinite(weights).all()
        or torch.any((weights < 0) | (weights > 1))
    ):
        raise ValueError("RTC requires finite targets and weights in [0,1]")
    prediction_type = scheduler.config.prediction_type
    if prediction_type not in {"epsilon", "sample", "v_prediction"}:
        raise ValueError(f"Unsupported RTC diffusion prediction type: {prediction_type}")
    with torch.enable_grad():
        x = sample.detach().requires_grad_(True)
        prediction = model(x, timestep, **model_kwargs)
        if prediction.shape != x.shape:
            raise ValueError("RTC does not support learned-variance diffusion heads")
        alpha = scheduler.alphas_cumprod[int(timestep)].to(x)
        a, sigma = alpha.sqrt(), (1 - alpha).sqrt().clamp_min(1e-8)
        if prediction_type == "epsilon":
            clean, eps = (x - sigma * prediction) / a, prediction
        elif prediction_type == "sample":
            clean, eps = prediction, (x - a * prediction) / sigma
        else:
            clean, eps = a * x - sigma * prediction, a * prediction + sigma * x
        correction = torch.autograd.grad(
            clean, x, grad_outputs=((target - clean) * weights).detach()
        )[0]
        eps = eps - torch.minimum(sigma.reciprocal(), sigma.new_tensor(beta)) * correction
        if prediction_type == "epsilon":
            guided = eps
        elif prediction_type == "sample":
            guided = (x - sigma * eps) / a
        else:
            guided = (eps - sigma * x) / a
    return guided.detach()
