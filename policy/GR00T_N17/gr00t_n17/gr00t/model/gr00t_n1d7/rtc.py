"""Pi-style guided inpainting for GR00T's noise-to-data flow convention."""

from collections.abc import Callable

import torch


def guided_velocity(
    actions: torch.Tensor,
    predict_velocity: Callable[[torch.Tensor], torch.Tensor],
    condition: torch.Tensor,
    weights: torch.Tensor,
    time: float,
    beta: float,
) -> torch.Tensor:
    """Differentiate the clean estimate only with respect to the noisy actions.

    GR00T integrates t=0 (noise) to t=1 (data), opposite to Pi05. Thus
    clean = x + (1-t)*v and the correction is added to v. Compute the residual
    and guidance scale in float32, including at the capped t=0 endpoint.
    No parameter gradients or graphs are retained between denoising steps.
    """
    with torch.enable_grad():
        sample = actions.detach().requires_grad_(True)
        velocity = predict_velocity(sample)
        clean = sample.float() + (1.0 - time) * velocity.float()
        residual = (condition.float() - clean) * weights.float()
        correction = torch.autograd.grad(clean, sample, grad_outputs=residual.detach())[0]
    denominator = max(time * (1.0 - time), torch.finfo(torch.float32).eps)
    scale = min(beta, (time**2 + (1.0 - time)**2) / denominator)
    return (velocity.float() + scale * correction.float()).to(actions.dtype).detach()
