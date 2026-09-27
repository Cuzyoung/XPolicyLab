"""Opt-in flow samplers for the GR00T and layer-wise PI action heads.

Time follows StarVLA training: 0 is noise and 1 is clean action. The default
head path is unchanged. RTC and PAINT equations are ported from the XPolicyLab
OpenPI integration (Apache-2.0); DVAC follows arXiv:2606.03847v1. This is an
inference implementation, not a claim of reproducing their task results.
"""

import math

import torch


def integrate(velocity, noise, num_steps, sampling, *, attention=None):
    """Euler integration with explicit conditioning/diagnostics at the sampler."""
    mode = sampling["mode"]
    if mode not in {"default", "rtc", "paint", "aac", "dvac", "autohorizon"}:
        raise ValueError(f"Unsupported flow sampling mode: {mode}")
    if type(num_steps) is not int or num_steps <= 0:
        raise ValueError("num_steps must be positive")
    dt = 1.0 / num_steps
    metadata = {}

    def forward(initial, *, diagnostics=False):
        actions = initial
        clean_tail = []
        for step in range(num_steps):
            time = step / float(num_steps)
            if mode == "rtc":
                with torch.enable_grad():
                    sample = actions.detach().requires_grad_(True)
                    prediction = velocity(sample, time)
                    clean = sample + (1.0 - time) * prediction
                    error = (sampling["action_condition"] - clean).detach() * sampling["condition_weights"]
                    guidance = torch.autograd.grad(clean, sample, grad_outputs=error)[0]
                    scale = min(
                        sampling["beta"],
                        (time**2 + (1 - time) ** 2) / max(time * (1 - time), torch.finfo(sample.dtype).eps),
                    )
                    prediction = prediction + scale * guidance
                actions = (actions + dt * prediction).detach()
                continue
            if diagnostics and mode == "autohorizon" and step == 2:
                with attention:
                    prediction = velocity(actions, time)
            else:
                prediction = velocity(actions, time)
            if diagnostics and mode == "dvac":
                clean_tail.append((actions + (1 - time) * prediction).float())
            actions = actions + dt * prediction
        if diagnostics and mode == "dvac":
            tail = torch.stack(clean_tail[-sampling["tail_steps"] :])
            variance = tail.var(dim=0, correction=0).sum(dim=-1)
            metadata["dvac"] = {
                "variance": variance[0].cpu().tolist(),
                "total_variance": float(variance[0].sum()),
                "tail_steps": sampling["tail_steps"],
                "num_steps": num_steps,
                "variance_space": "normalized_valid_action",
            }
        return actions

    actions = forward(noise, diagnostics=True)
    if mode == "paint":
        delay = sampling["delay_steps"]
        inverted = actions.clone()
        inverted[:, :delay] = sampling["action_condition"][:, :delay]
        for step in range(num_steps):
            time = 1.0 - step / float(num_steps)
            inverted = inverted - dt * velocity(inverted, time)
        repainted = noise.clone()
        repainted[:, :delay] = inverted[:, :delay]
        actions = forward(repainted)
        metadata["paint"] = {
            "delay_steps": delay,
            "num_steps": num_steps,
            "model_evaluations": 3 * num_steps,
            "inversion": "backward_euler",
        }
    if mode == "autohorizon":
        from .autohorizon_official import UPSTREAM_COMMIT, bidir_soft_pointer

        matrix = attention.mean()
        execution, diagnostics = bidir_soft_pointer(matrix)
        metadata["autohorizon"] = {
            "execution_steps": int(execution),
            "attention_step": 3,
            "hold_threshold": 0.3,
            "entropy_quantile": 0.9,
            "run_length": 1,
            "method": diagnostics["method"],
            "forward_horizon": diagnostics["N_forward"],
            "backward_horizon": diagnostics["N_backward"],
            "join_row": diagnostics["join_row"],
            "framework": "starvla_flow_attention_port",
            "upstream_commit": UPSTREAM_COMMIT,
        }
    if not torch.isfinite(actions).all():
        raise ValueError("Flow sampler returned non-finite actions")
    return {"actions": actions, **metadata}


class ActionAttention:
    """Observe action self-attention without replacing the model's attention kernel."""

    def __init__(self, model, horizon):
        self.modules = [
            module for name, module in model.named_modules() if name.endswith("attn1") and not module.is_cross_attention
        ]
        if not self.modules:
            raise ValueError("AutoHorizon requires action self-attention layers")
        self.horizon = horizon
        self.handles = []
        self.matrices = []

    def _capture(self, module, args, kwargs):
        if kwargs.get("encoder_hidden_states") is not None or kwargs.get("attention_mask") is not None:
            raise ValueError("AutoHorizon requires unmasked action self-attention")
        hidden = args[0]
        batch, length, _ = hidden.shape
        q = module.to_q(hidden).view(batch, length, module.heads, -1).transpose(1, 2)
        k = module.to_k(hidden).view(batch, length, module.heads, -1).transpose(1, 2)
        if module.norm_q is not None:
            q = module.norm_q(q)
        if module.norm_k is not None:
            k = module.norm_k(k)
        scores = (q.float() @ k.float().transpose(-1, -2)) * module.scale
        # Normalize over all keys before taking the action-to-action submatrix.
        weights = scores.softmax(dim=-1)[..., -self.horizon :, -self.horizon :]
        self.matrices.append(weights.mean(dim=(0, 1)))

    def __enter__(self):
        self.handles = [module.register_forward_pre_hook(self._capture, with_kwargs=True) for module in self.modules]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def mean(self):
        if len(self.matrices) != len(self.modules):
            raise ValueError("Did not capture every selected action attention layer")
        return torch.stack(self.matrices).mean(dim=0)


def sample_head(head, embeddings, state, encoder_attention_mask, sampling, *, layerwise=False):
    """One VLM encoding, then head-only sampling; no model weights are replaced."""
    mode = sampling["mode"]
    count = sampling.get("num_samples", 1)
    if type(count) is not int or count < 1 or (mode == "aac" and count <= 1):
        raise ValueError("AAC requires an integer num_samples greater than one")
    if mode != "aac" and count != 1:
        raise ValueError("Multiple samples cannot be combined with another mode")
    num_steps = int(head.num_inference_timesteps)
    if mode == "autohorizon" and num_steps < 3:
        raise ValueError("AutoHorizon requires at least three denoising steps")
    if mode == "dvac" and (type(sampling.get("tail_steps")) is not int or not 1 < sampling["tail_steps"] <= num_steps):
        raise ValueError("DVAC requires 1 < tail_steps <= num_inference_timesteps")
    # Framework predict_action may run in inference_mode. RTC needs normal
    # tensors for the action-head VJP, but never differentiates through the VLM.
    with torch.inference_mode(False), torch.no_grad():
        features = [
            item.detach().clone().repeat_interleave(count, dim=0) for item in (embeddings if layerwise else [embeddings])
        ]
        if features[0].shape[0] != count:
            raise ValueError("Specialized sampling requires exactly one observation")
        device, dtype = features[0].device, features[0].dtype
        state = state.detach().clone().repeat_interleave(count, dim=0) if state is not None else None
        state_features = head.state_encoder(state) if state is not None else None
        mask = (
            encoder_attention_mask.detach().clone().repeat_interleave(count, dim=0)
            if encoder_attention_mask is not None
            else None
        )
        future_tokens = head.future_tokens.weight.detach().unsqueeze(0).expand(count, -1, -1)
        position = None
        if head.config.add_pos_embed:
            position = head.position_embedding(torch.arange(head.action_horizon, device=device)).detach().unsqueeze(0)

        def velocity(actions, time):
            buckets = torch.full((count,), int(time * head.num_timestep_buckets), device=device, dtype=torch.long)
            action_features = head.action_encoder(actions, buckets)
            if position is not None:
                action_features = action_features + position
            parts = [future_tokens, action_features]
            if state_features is not None:
                parts.insert(0, state_features)
            options = dict(
                hidden_states=torch.cat(parts, dim=1),
                encoder_hidden_states=features if layerwise else features[0],
                timestep=buckets,
                encoder_attention_mask=mask,
            )
            if layerwise:
                options["return_pre_output"] = True
            output = head.model(**options)
            return head.action_decoder(output)[:, -head.action_horizon :]

        parameters = dict(sampling)
        if mode in {"rtc", "paint"}:
            parameters["action_condition"] = torch.tensor(sampling["action_condition"], device=device, dtype=dtype)[None]
            if parameters["action_condition"].shape != (1, head.action_horizon, head.action_dim):
                raise ValueError("Normalized condition must match the action head horizon and width")
        if mode == "rtc":
            parameters["condition_weights"] = torch.tensor(sampling["condition_weights"], device=device, dtype=dtype)[
                None, :, None
            ]
            if parameters["condition_weights"].shape != (1, head.action_horizon, 1):
                raise ValueError("RTC weights must match the action horizon")
            if not math.isfinite(parameters["beta"]) or parameters["beta"] <= 0:
                raise ValueError("RTC beta must be finite and positive")
        noise = torch.randn((count, head.action_horizon, head.action_dim), device=device, dtype=dtype)
        attention = ActionAttention(head.model, head.action_horizon) if mode == "autohorizon" else None
        return integrate(velocity, noise, num_steps, parameters, attention=attention)
