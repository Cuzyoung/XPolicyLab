"""ImageWAMActionPatch: single-stream "action-as-image-patch" variant of ImageWAM.

Idea (exploration/action_img_patch): actions are just another modality of patch.
Instead of a separate action expert (MoT) or an in-block action stream
(three-stream), the action chunk is mapped into the SAME 128-d packed-latent
token space as the FLUX.2 image tokens, appended to the noisy target sequence,
and denoised by the unmodified FLUX.2 backbone exactly like image patches:

    joint sequence: [txt(+proprio) | ref(clean) | target(noisy) | action(noisy)]
                                                  \_________ one sigma ________/

  * ONE scheduler / ONE sigma for image and action tokens (single-stream flow
    matching); the loss is the same velocity MSE, evaluated on image tokens
    (loss_video) and action tokens (loss_action) with the usual lambdas.
  * The visibility mask is the stock ImageWAM flux2 mask with `action_len=0`:
    action tokens live INSIDE the noisy-target block, so the only distinction
    left is clean-prefix vs noisy tokens (clean never attends noisy).
  * The action<->token codec is FIXED (zero parameters): each of the A action
    dims is tiled 128//A times (126 dims used for A=14) and scaled by
    `action_patch_scale`; decoding averages the tiles. z-scored actions are
    ~unit-variance, matching the latent statistics the scheduler assumes.
    A fixed codec cannot collapse, so no auxiliary reconstruction loss is
    needed and the training objective stays purely flow matching.
  * ZERO new trainable modules: everything trainable is the FLUX.2 transformer
    itself (plus the shared proprio encoder, same as the MoT baseline).

Construction mirrors `ImageWAM.from_flux2_klein_pretrained` minus the action
DiT: MoT wraps only the video expert (its `_mixed_attention` /
`_flux2_video_single_io` helpers and gradient checkpointing are reused by the
video-only forward below, which is `MoT._forward_flux2` with the action parts
removed). `concept_k=0` so no concept bottleneck. Checkpoint payload stays
{"mot", "proprio_encoder", ...} -> trainer save/load works unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)

# RoPE time-axis coordinate for action tokens (target frame uses 0.0, the
# clean reference frame uses 10.0); axis 3 carries the chunk index, mirroring
# how txt ids use axis 3 for sequence position.
ACTION_TOKEN_TIME_VALUE = 20.0
FLUX2_TOKEN_DIM = 128


class ImageWAMActionPatch(ImageWAM):
    """ImageWAM whose action head is the FLUX.2 image stream itself."""

    def __init__(self, *args, action_patch_dim: int, action_patch_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_patch_dim = int(action_patch_dim)
        self.action_patch_scale = float(action_patch_scale)
        if self.action_patch_dim <= 0 or self.action_patch_dim > FLUX2_TOKEN_DIM:
            raise ValueError(f"`action_patch_dim` must be in [1, {FLUX2_TOKEN_DIM}], got {action_patch_dim}")
        if self.action_patch_scale <= 0:
            raise ValueError(f"`action_patch_scale` must be positive, got {action_patch_scale}")
        self.action_patch_reps = FLUX2_TOKEN_DIM // self.action_patch_dim
        self.action_patch_used = self.action_patch_reps * self.action_patch_dim
        logger.info(
            "ImageWAMActionPatch: action dim %d tiled x%d -> %d/%d token dims, scale %.3f (fixed codec, 0 params)",
            self.action_patch_dim,
            self.action_patch_reps,
            self.action_patch_used,
            FLUX2_TOKEN_DIM,
            self.action_patch_scale,
        )

    # ------------------------------------------------------------- fixed codec
    def _encode_action_tokens(self, action: torch.Tensor) -> torch.Tensor:
        """[B, H, A] actions -> [B, H, 128] latent-space tokens (fixed tiling)."""
        if action.shape[-1] != self.action_patch_dim:
            raise ValueError(f"Expected action dim {self.action_patch_dim}, got {action.shape[-1]}")
        tiled = action.repeat_interleave(self.action_patch_reps, dim=-1) * self.action_patch_scale
        if self.action_patch_used < FLUX2_TOKEN_DIM:
            pad = tiled.new_zeros(*tiled.shape[:-1], FLUX2_TOKEN_DIM - self.action_patch_used)
            tiled = torch.cat([tiled, pad], dim=-1)
        return tiled

    def _decode_action_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, H, 128] tokens -> [B, H, A] actions (average the tiles)."""
        used = tokens[..., : self.action_patch_used]
        b, h = used.shape[0], used.shape[1]
        return used.reshape(b, h, self.action_patch_dim, self.action_patch_reps).mean(dim=-1) / self.action_patch_scale

    @staticmethod
    def _build_action_token_ids(
        batch_size: int,
        horizon: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, horizon, 4, device=device, dtype=dtype)
        ids[..., 0] = ACTION_TOKEN_TIME_VALUE
        ids[..., 3] = torch.arange(horizon, device=device, dtype=dtype)[None, :]
        return ids

    # ------------------------------------------------------- video-only forward
    def _forward_flux2_video_only(self, video_pre: dict, attention_mask: dict) -> dict:
        """`MoT._forward_flux2` with the action-expert parts removed.

        Reuses the MoT helpers so gradient checkpointing / GQA settings apply
        identically to the baseline.
        """
        from flux2.model import apply_rope

        mot = self.mot
        video_expert = mot.mixtures["video"]
        txt = video_pre["tokens"]["txt"]
        img = video_pre["tokens"]["img"]
        txt_pe = video_pre["freqs"]["txt"]
        img_pe = video_pre["freqs"]["img"]
        t_mod = video_pre["t_mod"]

        for layer_idx in range(int(video_expert.double_layers)):
            block = video_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                t_mod["double_img"],
                t_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe_full)
            mixed = mot._mixed_attention(
                mot._flux2_flatten_heads(q),
                mot._flux2_flatten_heads(k),
                mot._flux2_flatten_heads(v),
                attention_mask["double_joint"],
            )
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(video_expert.single_layers)):
            block = video_expert.single_blocks[layer_idx]
            state = mot._flux2_video_single_io(block, video_stream, stream_pe, t_mod["single"])
            mixed = mot._mixed_attention(state["q"], state["k"], state["v"], attention_mask["single"])
            video_stream = block._out(state["residual_x"], mixed, state["mlp"], state["gate"])

        txt_len = int(txt.shape[1])
        return {"txt": video_stream[:, :txt_len], "img": video_stream[:, txt_len:]}

    # ---------------------------------------------------------- training loss
    def _training_loss_flux2(self, sample, tiled: bool = False):
        inputs = self.build_inputs_flux2(sample, tiled=tiled)
        target_latent = inputs["target_latent"]  # [B, S, 128]
        action = inputs["action"]                # [B, H, A]
        batch_size = int(target_latent.shape[0])
        target_len = int(target_latent.shape[1])
        horizon = int(action.shape[1])

        action_latent = self._encode_action_tokens(action)  # [B, H, 128]

        # ONE sigma for image and action tokens: single-stream flow matching.
        timestep = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noise_img = torch.randn_like(target_latent)
        noise_act = torch.randn_like(action_latent)
        noisy_img = self.train_video_scheduler.add_noise(target_latent, noise_img, timestep)
        noisy_act = self.train_video_scheduler.add_noise(action_latent, noise_act, timestep)
        target_v_img = self.train_video_scheduler.training_target(target_latent, noise_img, timestep)
        target_v_act = self.train_video_scheduler.training_target(action_latent, noise_act, timestep)

        target_img_ids = inputs["target_img_ids"]
        action_ids = self._build_action_token_ids(
            batch_size,
            horizon,
            device=target_img_ids.device,
            dtype=target_img_ids.dtype,
        )
        combined = torch.cat([noisy_img, noisy_act], dim=1)
        combined_ids = torch.cat([target_img_ids, action_ids], dim=1)

        video_pre = self.video_expert.pre_dit(
            x=combined,
            timestep=self._scheduler_timestep_to_unit(timestep, self.train_video_scheduler),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=combined_ids,
            ref_img_ids=inputs["ref_img_ids"],
        )
        # action_len=0: action tokens sit inside the noisy-target block, so the
        # stock mask degenerates to the pure clean/noisy rule.
        attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=combined.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens_out = self._forward_flux2_video_only(video_pre, attention_mask)
        pred = self.video_expert.post_dit(tokens_out, video_pre)  # [B, S+H, 128]
        pred_img = pred[:, :target_len]
        pred_act = pred[:, target_len:]

        weight = self.train_video_scheduler.training_weight(timestep)

        video_loss_per_sample = (
            F.mse_loss(pred_img.float(), target_v_img.float(), reduction="none").flatten(1).mean(dim=1)
        )
        video_weight = weight.to(video_loss_per_sample.device, dtype=video_loss_per_sample.dtype)
        loss_video = (video_loss_per_sample * video_weight).mean()

        # Same velocity MSE on the action tokens (token-space, pad-masked).
        # `action_dim_is_pad` is not applicable in tiled token space (RoboTwin
        # does not use it anyway).
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_act,
            target_action=target_v_act,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=None,
        )
        action_weight = weight.to(action_loss_per_sample.device, dtype=action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def infer_action_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        """Joint denoising of [future-image ; action] tokens, then decode actions.

        Unlike the MoT KV-cache action-only decode, the single-stream model
        denoises the future image alongside the actions (actions are grounded
        in the predicted future by construction).
        """
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="action-patch inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(input_image, time_value=10.0)
        batch_size = int(ref_tokens.shape[0])
        target_len = int(ref_tokens.shape[1])
        horizon = int(action_horizon)

        target_img_ids = ref_img_ids.clone()
        target_img_ids[..., 0] = 0.0
        action_ids = self._build_action_token_ids(
            batch_size,
            horizon,
            device=ref_img_ids.device,
            dtype=ref_img_ids.dtype,
        )
        combined_ids = torch.cat([target_img_ids, action_ids], dim=1)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=rand_device)
            generator.manual_seed(int(seed))
        latents = torch.randn(
            (batch_size, target_len + horizon, FLUX2_TOKEN_DIM),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        scheduler = self.infer_video_scheduler
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=self.torch_dtype,
            shift_override=sigma_shift,
        )

        attention_mask = None
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.expand(batch_size).to(device=self.device, dtype=latents.dtype)
            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=self._scheduler_timestep_to_unit(timestep, scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                target_img_ids=combined_ids,
                ref_img_ids=ref_img_ids,
            )
            if attention_mask is None:
                attention_mask = self._build_mot_attention_mask_flux2(
                    batch_size=batch_size,
                    txt_len=int(video_pre["txt_len"]),
                    target_len=int(video_pre["target_len"]),
                    cond_len=int(video_pre["cond_len"]),
                    action_len=0,
                    device=self.device,
                    text_attention_mask=video_pre["text_mask"],
                )
            tokens_out = self._forward_flux2_video_only(video_pre, attention_mask)
            pred = self.video_expert.post_dit(tokens_out, video_pre)
            latents = scheduler.step(pred, step_delta, latents)

        action = self._decode_action_tokens(latents[:, target_len:])
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}

    # ------------------------------------------------------------------ build
    @classmethod
    def from_flux2_klein_actionpatch_pretrained(
        cls,
        flux2_model_path: str,
        ae_model_path: str,
        flux2_src_path: str | None = None,
        variant: str = "klein-base-4b",
        qwen3_model_spec: str | None = None,
        qwen_context_len: int = 128,
        proprio_dim: Optional[int] = None,
        load_text_encoder: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        pack_proprio_after_text: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 0.5,
        loss_lambda_action: float = 1.0,
        action_patch_dim: int = 14,
        action_patch_scale: float = 1.0,
    ):
        """Mirror of `ImageWAM.from_flux2_klein_pretrained` without the action DiT."""
        from safetensors.torch import load_file as load_sft

        from imagewam.models.backbones.flux2_imports import ensure_flux2_importable
        from imagewam.models.backbones.flux2_video_expert import Flux2VideoExpert
        from imagewam.models.backbones.mot import MoT

        ensure_flux2_importable(flux2_src_path)
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams

        key = str(variant).lower().replace("_", "-")
        if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
            text_dim = 7680
            default_qwen3_model_spec = "Qwen/Qwen3-4B"
        elif key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
            text_dim = 12288
            default_qwen3_model_spec = "Qwen/Qwen3-8B"
        else:
            raise ValueError(f"Unsupported FLUX.2 Klein variant: {variant!r}")

        video_expert = Flux2VideoExpert.from_pretrained(
            flux2_model_path=flux2_model_path,
            variant=key,
            flux2_src_path=flux2_src_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        video_expert.flux2_lora_enabled = False

        # MoT with the video expert only: its mixed-attention helper (with
        # gradient checkpointing) is what `_forward_flux2_video_only` uses.
        mot = MoT(
            mixtures={"video": video_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )

        with torch.device("meta"):
            ae = AutoEncoder(AutoEncoderParams())
        ae_state = load_sft(str(ae_model_path), device=str(device))
        ae.load_state_dict(ae_state, strict=True, assign=True)
        ae = ae.to(device=device, dtype=torch_dtype).eval()

        if load_text_encoder:
            from types import SimpleNamespace

            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_spec = qwen3_model_spec or default_qwen3_model_spec
            qwen3_model = AutoModelForCausalLM.from_pretrained(
                model_spec,
                torch_dtype=torch_dtype,
            ).to(device).eval()
            text_encoder = SimpleNamespace(
                model=qwen3_model,
                tokenizer=AutoTokenizer.from_pretrained(model_spec),
                max_length=int(qwen_context_len),
            )
        else:
            text_encoder = None

        model = cls(
            video_expert=video_expert,
            action_expert=None,
            mot=mot,
            vae=ae,
            text_encoder=text_encoder,
            tokenizer=None,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="flux2",
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=bool(pack_proprio_after_text),
            concept_k=0,
            lambda_concept=0.0,
            action_patch_dim=action_patch_dim,
            action_patch_scale=action_patch_scale,
        )
        model.model_paths = {
            "flux2": flux2_model_path,
            "flux2_src": flux2_src_path,
            "ae": ae_model_path,
            "action_dit": None,
            "qwen3": qwen3_model_spec or default_qwen3_model_spec,
        }
        model.flux2_qwen3_model_spec = qwen3_model_spec or default_qwen3_model_spec
        model.save_lora_merged = False
        model.save_trainable_only = False
        return model
