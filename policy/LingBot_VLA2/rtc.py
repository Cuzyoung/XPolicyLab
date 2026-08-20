"""Sampler-level PI-guided RTC for the official LingBot-VLA2 flow model.

The upstream sampler exposes the prefix cache and ``predict_velocity`` pieces
needed by RTC but has no conditioning API. This module owns the integration
without monkeypatching upstream classes: it runs the same Euler flow loop and
applies VJP guidance at every denoising step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

RTC_MODE = "pi_guided_v1"


def guidance_scale(time: torch.Tensor, beta: float) -> torch.Tensor:
    """Return the bounded PI RTC guidance schedule for one flow time."""

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"RTC beta must be finite and positive, got {beta}")
    tau = 1.0 - time
    denominator = torch.clamp(time * tau, min=torch.finfo(time.dtype).eps)
    raw_scale = (time.square() + tau.square()) / denominator
    return torch.minimum(torch.as_tensor(beta, device=time.device, dtype=time.dtype), raw_scale)


def guided_velocity(
    predict_velocity: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    time: torch.Tensor,
    action_condition: torch.Tensor,
    condition_weights: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Apply clean-action VJP guidance to one denoising step."""

    if action_condition.shape != x_t.shape:
        raise ValueError(
            f"RTC action_condition must have shape {tuple(x_t.shape)}, "
            f"got {tuple(action_condition.shape)}"
        )
    if condition_weights.shape != x_t.shape:
        raise ValueError(
            f"RTC condition_weights must have shape {tuple(x_t.shape)}, "
            f"got {tuple(condition_weights.shape)}"
        )
    if not torch.isfinite(action_condition).all() or not torch.isfinite(
        condition_weights
    ).all():
        raise ValueError("RTC conditioning tensors must be finite")
    if torch.any(condition_weights < 0) or torch.any(condition_weights > 1):
        raise ValueError("RTC condition_weights must be in [0, 1]")

    with torch.enable_grad():
        sample = x_t.detach().requires_grad_(True)
        velocity = predict_velocity(sample)
        clean = sample - time * velocity
        weighted_error = (action_condition - clean) * condition_weights
        guidance = torch.autograd.grad(
            clean,
            sample,
            grad_outputs=weighted_error,
            create_graph=False,
            retain_graph=False,
        )[0]
    return velocity.detach() - guidance_scale(time, beta) * guidance.detach()


@torch.no_grad()
def sample_actions_rtc(
    flow_model: Any,
    images: torch.Tensor,
    img_masks: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    *,
    action_condition: torch.Tensor,
    condition_weights: torch.Tensor,
    beta: float,
    noise: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    _make_masks: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run LingBot's native flow loop with RTC guidance on every step."""

    if _make_masks is None:
        from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks

        _make_masks = make_att_2d_masks

    batch_size = state.shape[0]
    device = state.device
    dtype = state.dtype
    actions_shape = (
        batch_size,
        flow_model.config.n_action_steps,
        flow_model.config.max_action_dim,
    )
    if noise is None:
        noise = torch.randn(actions_shape, device=device, dtype=dtype)
    if noise.shape != actions_shape:
        raise ValueError(f"RTC noise must have shape {actions_shape}, got {tuple(noise.shape)}")

    target = action_condition.to(device=device, dtype=dtype)
    weights = condition_weights.to(device=device, dtype=dtype)
    if target.shape != actions_shape or weights.shape != actions_shape:
        raise ValueError(
            "RTC normalized condition and weights must match model action shape "
            f"{actions_shape}, got {tuple(target.shape)} and {tuple(weights.shape)}"
        )

    (
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    ) = flow_model.embed_prefix(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw=image_grid_thw,
    )
    prefix_att_2d_masks = _make_masks(prefix_pad_masks, prefix_att_masks)
    _, past_key_values, _ = flow_model.qwenvl_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        vlm_position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=flow_model.config.use_cache,
        fill_kv_cache=True,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )

    dt = torch.tensor(-1.0 / flow_model.config.num_steps, dtype=dtype, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=dtype, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(batch_size)

        def predict(
            sample: torch.Tensor,
            expanded_time: torch.Tensor = expanded_time,
        ) -> torch.Tensor:
            return flow_model.predict_velocity(
                state,
                prefix_pad_masks,
                past_key_values,
                sample,
                expanded_time,
                prefix_position_ids=prefix_position_ids,
            )

        velocity = guided_velocity(predict, x_t, time, target, weights, beta)
        x_t = x_t + dt * velocity
        time = time + dt
    return x_t


def encode_raw_condition(action_condition: np.ndarray) -> dict[str, np.ndarray]:
    """Map ManiMux [left 6+1, right 6+1] rows to LingBot action features."""

    condition = np.asarray(action_condition, dtype=np.float32)
    if condition.ndim != 2 or condition.shape[1] != 14:
        raise ValueError(f"RTC raw condition must be (H, 14), got {condition.shape}")
    return {
        "action.arm.position": np.concatenate(
            [condition[:, :6], condition[:, 7:13]], axis=-1
        ),
        "action.effector.position": np.stack(
            [condition[:, 6], condition[:, 13]], axis=-1
        ),
    }


def normalize_condition(
    feature_transform: Any,
    raw_actions: Mapping[str, np.ndarray],
    condition_weights: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and pad one raw condition into LingBot's 55-D model space."""

    if any(feature_transform.action_subtract_state.values()):
        raise NotImplementedError(
            "RTC requires an absolute-action robot config; "
            "delta actions need state-aware conversion"
        )
    condition_item = {
        key: torch.as_tensor(value, dtype=torch.float32) for key, value in raw_actions.items()
    }
    converted = feature_transform.convert_features(condition_item, w_action=True)
    normalized = feature_transform.normalizer.normalize(converted)
    horizon = next(iter(normalized.values())).shape[0]
    chunks: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for joint in feature_transform.feature_config.joints:
        key = f"action.{joint}"
        max_dim = feature_transform.feature_config.joints_max_dim[joint]
        if key in normalized:
            values = normalized[key].to(torch.float32)
            pad = max_dim - values.shape[-1]
            if pad < 0:
                raise ValueError(f"RTC feature {key} exceeds configured max dim {max_dim}")
            chunks.append(F.pad(values, (0, pad)))
            masks.append(F.pad(torch.ones(values.shape[-1]), (0, pad)))
        else:
            chunks.append(torch.zeros(horizon, max_dim))
            masks.append(torch.zeros(max_dim))
    target = torch.cat(chunks, dim=-1)
    action_mask = torch.cat(masks, dim=-1).to(torch.float32)
    max_action_dim = int(feature_transform.model_config.max_action_dim)
    tail_pad = max_action_dim - target.shape[-1]
    if tail_pad < 0:
        raise ValueError(
            f"RTC packed condition exceeds model max_action_dim {max_action_dim}"
        )
    target = F.pad(target, (0, tail_pad))
    action_mask = F.pad(action_mask, (0, tail_pad))
    weights = torch.as_tensor(condition_weights, dtype=torch.float32)
    if weights.shape != (horizon,):
        raise ValueError(f"RTC condition_weights must have shape {(horizon,)}, got {weights.shape}")
    padded_weights = weights[:, None] * action_mask[None, :]
    return target, padded_weights


class LingBotRtcBridge:
    """Run sampler-level RTC through an initialized official server instance."""

    def __init__(self, server: Any) -> None:
        self.server = server

    def infer(
        self,
        observation: Mapping[str, Any],
        action_condition: np.ndarray,
        condition_weights: np.ndarray,
        beta: float,
    ) -> dict[str, np.ndarray]:
        raw_actions = encode_raw_condition(action_condition)
        target, weights = normalize_condition(
            self.server.vla.feature_transform, raw_actions, condition_weights
        )

        item = dict(observation)
        self.server.resize_image(item)
        for key, value in list(item.items()):
            if isinstance(value, np.ndarray):
                item[key] = torch.from_numpy(value)
        transformed = self.server.vla.feature_transform.apply(item, policy_eval=True)

        dtype = torch.bfloat16 if self.server.use_bf16 else torch.float32
        images = transformed["images"]
        img_masks = transformed["img_masks"]
        if images.ndim == 4:
            images = images.unsqueeze(0)
            img_masks = img_masks.unsqueeze(0)
        lang_tokens = transformed["lang_tokens"]
        lang_masks = transformed["lang_masks"]
        state = transformed["state"]
        if lang_tokens.ndim == 1:
            lang_tokens = lang_tokens.unsqueeze(0)
            lang_masks = lang_masks.unsqueeze(0)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        grid = transformed.get("image_grid_thw")
        if grid is not None:
            grid = grid.to(device="cuda", dtype=torch.long)

        actions = sample_actions_rtc(
            self.server.vla.model,
            images.to(device="cuda", dtype=dtype),
            img_masks.to(device="cuda"),
            lang_tokens.to(device="cuda"),
            lang_masks.to(device="cuda"),
            state.to(device="cuda", dtype=dtype),
            action_condition=target.unsqueeze(0),
            condition_weights=weights.unsqueeze(0),
            beta=beta,
            image_grid_thw=grid,
        )
        transformed["actions"] = actions[0].to(dtype=torch.float32, device="cpu")
        if self.server.use_bf16:
            transformed["state"] = transformed["state"].to(torch.float32)
        result = self.server.vla.feature_transform.unapply(transformed)
        output: dict[str, np.ndarray] = {}
        for key in self.server.action_key:
            output[key] = np.asarray(result[key], dtype=np.float32)
            if self.server.use_length > 0:
                output[key] = output[key][: self.server.use_length]
        return output
