"""RTC inpainting for SA's noise-to-data flow (t=1 to t=0).

The VJP corrects the estimated clean action, as in the Pi05 RTC sampler.
Only the action input needs gradients; model parameter gradients are not stored.
"""

import torch


def guided_velocity(predict_velocity, actions, time, condition, weights, beta):
    with torch.enable_grad():
        sample = actions.detach().requires_grad_(True)
        velocity = predict_velocity(sample)
        clean = sample - time * velocity
        error = ((condition - clean) * weights).detach()
        correction = torch.autograd.grad(clean, sample, grad_outputs=error)[0]
        scale = min(beta, (time**2 + (1.0 - time)**2) / max(time * (1.0 - time), 1e-6))
        return (velocity - scale * correction).detach()
