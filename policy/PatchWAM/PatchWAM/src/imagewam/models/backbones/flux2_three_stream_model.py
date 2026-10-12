"""ImageWAMThreeStream: top-level model that swaps the MoT (video-expert + action-DiT,
separate-block joint attention) core for the single self-contained three-stream
`Flux2ActionTransformer2DModel` (image + text + action, in-block joint attention on the
FLUX.2 hybrid backbone), while reusing every piece of the ImageWAM FLUX.2 training path:

  * VAE video->latent encoding (`_encode_flux2_image_tokens`, packed [B,S,128] tokens);
  * the LatentReasoner (Qwen3 Coconut) call -> encoder_hidden_states (prompt+16 latent);
  * the continuous flow-matching schedulers (independent video / action sigma);
  * the per-sample video / action loss helpers and the dual-loss contract;
  * `training_loss(sample) -> (loss, {loss_video, loss_action})`.

The ONLY structural change vs. the base `ImageWAM` FLUX.2 stack is the DiT forward: instead
of `video_expert.pre_dit` -> `MoT._forward_flux2` -> `post_dit`, we feed the raw packed
latents straight into the three-stream model:

    target packed latent   -> hidden_states            (noisy, real video sigma)
    ref packed latent       -> context_hidden_states    (clean history, zero_cond_t)
    reasoner output         -> encoder_hidden_states     (text stream: prompt + latent)
    proprio[:, 0]           -> state_tokens              (single state token in text stream)
    action                  -> noisy_actions             (independent action sigma)
    video sigma (unit)      -> timestep
    action sigma (unit)     -> action_timestep

The three-stream builds its own RoPE ids / modulation / masks internally; we hand it the
packed FLUX img_ids (target then context) and txt_ids sized to the reasoner length + the
appended state token.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from imagewam.utils.logging_config import get_logger

from .imagewam import ImageWAM

logger = get_logger(__name__)


_KLEIN_4B_CONFIG = {
    "patch_size": 1,
    "in_channels": 128,
    "out_channels": 128,
    "num_layers": 5,
    "num_single_layers": 20,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 7680,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "guidance_embeds": False,
}


def _read_flux2_transformer_config(flux2_transformer_dir: str) -> dict:
    """Read the diffusers `<dir>/config.json` and extract the fields the three-stream needs.

    Falls back to the known klein-4b constants for any field the config omits so that the
    instantiated model's parameter shapes match the pretrained safetensors exactly (an empty
    `unexpected` set on load).
    """
    import json
    import os

    cfg_path = os.path.join(flux2_transformer_dir, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as fh:
            cfg = json.load(fh)
    out = dict(_KLEIN_4B_CONFIG)
    for src_key, dst_key in [
        ("in_channels", "in_channels"),
        ("num_layers", "num_layers"),
        ("num_single_layers", "num_single_layers"),
        ("attention_head_dim", "attention_head_dim"),
        ("num_attention_heads", "num_attention_heads"),
        ("joint_attention_dim", "joint_attention_dim"),
        ("timestep_guidance_channels", "timestep_guidance_channels"),
        ("mlp_ratio", "mlp_ratio"),
        ("patch_size", "patch_size"),
        ("rope_theta", "rope_theta"),
        ("guidance_embeds", "guidance_embeds"),
    ]:
        if src_key in cfg and cfg[src_key] is not None:
            out[dst_key] = cfg[src_key]
    if "axes_dims_rope" in cfg and cfg["axes_dims_rope"] is not None:
        out["axes_dims_rope"] = tuple(cfg["axes_dims_rope"])
    if cfg.get("out_channels"):
        out["out_channels"] = cfg["out_channels"]
    else:
        out["out_channels"] = out["in_channels"]
    return out


class ImageWAMThreeStream(ImageWAM):
    """ImageWAM whose FLUX.2 core is the single three-stream Flux2ActionTransformer2DModel."""

    def __init__(self, *args, three_stream=None, **kwargs):
        if three_stream is None:
            raise ValueError("ImageWAMThreeStream requires a `three_stream` model.")
        # Reuse ImageWAM.__init__ for schedulers / vae / proprio / loss lambdas etc.
        # ImageWAM.__init__ registers whatever is passed as `mot` (== `dit`) and
        # `video_expert`/`action_expert` as submodules; we pass the three-stream for all so the
        # canonical registered submodule is `self.mot`. To avoid registering the SAME nn.Module
        # under 5 different names (which would 5x the top-level state_dict), we then drop the
        # duplicate module registrations and keep `dit`/`video_expert`/`action_expert`/
        # `three_stream` as plain aliases to the single registered `self.mot`.
        super().__init__(*args, **kwargs)
        # `mot` is the single registered submodule (== the three-stream). Deregister the aliases
        # that ImageWAM.__init__ created as submodules, then re-point them as plain attributes.
        for alias in ("dit", "video_expert", "action_expert"):
            if alias in self._modules:
                del self._modules[alias]
        object.__setattr__(self, "dit", self.mot)
        object.__setattr__(self, "video_expert", self.mot)
        object.__setattr__(self, "action_expert", self.mot)
        object.__setattr__(self, "three_stream", self.mot)
        self.to(self.device)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_flux2_klein_threestream_pretrained(
        cls,
        flux2_transformer_dir: str,
        ae_model_path: str,
        flux2_src_path: str | None = None,
        variant: str = "klein-base-4b",
        action_dim: int = 14,
        action_horizon: int = 64,
        state_dim: int = 14,
        proprio_dim: Optional[int] = None,
        qwen3_model_spec: str | None = None,
        qwen_context_len: int = 128,
        load_flux_weights: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 0.5,
        loss_lambda_action: float = 1.0,
        latent_reasoner_config=None,
    ):
        from safetensors.torch import load_file as load_sft

        from .flux2_action_three_stream import (
            Flux2ActionTransformer2DModel,
            init_action_from_flux,
            load_flux2_pretrained,
        )
        from .flux2_imports import ensure_flux2_importable

        ensure_flux2_importable(flux2_src_path)
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams

        # ---- three-stream model, dims aligned to the pretrained FLUX transformer config ----
        ts_cfg = _read_flux2_transformer_config(flux2_transformer_dir)
        inner_dim = int(ts_cfg["num_attention_heads"]) * int(ts_cfg["attention_head_dim"])
        model_ts = Flux2ActionTransformer2DModel(
            patch_size=int(ts_cfg["patch_size"]),
            in_channels=int(ts_cfg["in_channels"]),
            out_channels=int(ts_cfg["out_channels"]),
            num_layers=int(ts_cfg["num_layers"]),
            num_single_layers=int(ts_cfg["num_single_layers"]),
            attention_head_dim=int(ts_cfg["attention_head_dim"]),
            num_attention_heads=int(ts_cfg["num_attention_heads"]),
            joint_attention_dim=int(ts_cfg["joint_attention_dim"]),
            timestep_guidance_channels=int(ts_cfg["timestep_guidance_channels"]),
            mlp_ratio=float(ts_cfg["mlp_ratio"]),
            axes_dims_rope=tuple(ts_cfg["axes_dims_rope"]),
            rope_theta=int(ts_cfg["rope_theta"]),
            guidance_embeds=bool(ts_cfg["guidance_embeds"]),
            action_dim=int(action_dim),
            action_horizon=int(action_horizon),
            state_dim=int(state_dim),
            # action_inner_dim == inner_dim so init_action_from_flux can copy-init.
            action_inner_dim=inner_dim,
        )
        if load_flux_weights:
            missing, unexpected = load_flux2_pretrained(model_ts, flux2_transformer_dir, device="cpu")
            if unexpected:
                raise ValueError(
                    f"load_flux2_pretrained produced {len(unexpected)} unexpected keys; "
                    f"first: {list(unexpected)[:8]}. Model config must mismatch the pretrained transformer."
                )
            logger.info(
                "load_flux2_pretrained: missing=%d (action/state/view fresh), unexpected=%d",
                len(missing),
                len(unexpected),
            )
            copied = init_action_from_flux(model_ts)
            logger.info("init_action_from_flux copied %d tensors (action warm-start).", copied)
        model_ts = model_ts.to(device=device, dtype=torch_dtype)

        # ---- VAE (same source as the base FLUX.2 factory) ----
        with torch.device("meta"):
            ae = AutoEncoder(AutoEncoderParams())
        ae_state = load_sft(str(ae_model_path), device=str(device))
        ae.load_state_dict(ae_state, strict=True, assign=True)
        ae = ae.to(device=device, dtype=torch_dtype).eval()

        key = str(variant).lower().replace("_", "-")
        if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
            text_dim = int(ts_cfg["joint_attention_dim"])
            default_qwen3_model_spec = "Qwen/Qwen3-4B"
        elif key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
            text_dim = int(ts_cfg["joint_attention_dim"])
            default_qwen3_model_spec = "Qwen/Qwen3-8B"
        else:
            raise ValueError(f"Unsupported FLUX.2 Klein variant: {variant!r}")

        model = cls(
            video_expert=model_ts,
            action_expert=model_ts,
            mot=model_ts,
            vae=ae,
            text_encoder=None,
            tokenizer=None,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="flux2_threestream",
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=False,
            three_stream=model_ts,
        )
        model.model_paths = {
            "flux2_transformer": flux2_transformer_dir,
            "flux2_src": flux2_src_path,
            "ae": ae_model_path,
            "qwen3": qwen3_model_spec or default_qwen3_model_spec,
        }
        model.flux2_qwen3_model_spec = qwen3_model_spec or default_qwen3_model_spec
        # The three-stream is fully fine-tuned (no LoRA); keep full-state checkpoints.
        model.save_lora_merged = False
        model.save_trainable_only = False

        lr_cfg = latent_reasoner_config
        if lr_cfg is not None and bool(lr_cfg.get("enabled", False)):
            from .latent_reasoner import LatentReasoner

            lora_cfg = dict(lr_cfg.get("lora") or {})
            model.latent_reasoner = LatentReasoner(
                qwen_path=str(lr_cfg.get("qwen_path", qwen3_model_spec or default_qwen3_model_spec)),
                d_out=int(lr_cfg.get("d_out", text_dim)),
                mode=str(lr_cfg.get("mode", "latent_loop")),
                num_latent=int(lr_cfg.get("num_latent", 16)),
                qwen_layers=tuple(lr_cfg.get("qwen_layers", (9, 18, 27))),
                system_prompt=lr_cfg.get("system_prompt"),
                max_prompt_len=int(lr_cfg.get("max_prompt_len", qwen_context_len)),
                lora_rank=int(lora_cfg.get("rank", 16)),
                lora_alpha=float(lora_cfg.get("alpha", 16.0)),
                lora_dropout=float(lora_cfg.get("dropout", 0.0)),
                lora_target_suffixes=tuple(
                    lora_cfg.get(
                        "target_suffixes",
                        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                    )
                ),
                torch_dtype=torch_dtype,
            ).to(device=device)
        return model

    # ------------------------------------------------------ trainable policy
    def apply_trainable_policy(self) -> None:
        """Full fine-tune of the three-stream (no video-expert LoRA branch here).

        The trainer already sets `model.dit.requires_grad_(True)`; the three-stream IS the
        dit, so nothing extra is required. Overriding prevents the base ImageWAM policy from
        poking at `self.mot.mixtures` (which does not exist for the three-stream).
        """
        self.three_stream.train()
        self.three_stream.requires_grad_(True)

    # --------------------------------------------------------------- forward
    def _build_threestream_rope_ids(self, inputs) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Assemble unbatched FLUX 4-axis RoPE ids for the three-stream.

        img_ids = [target_img_ids ; ref_img_ids]  (target first, matching the model's
        img = cat([hidden_states, context_hidden_states])). txt_ids covers the reasoner
        tokens PLUS the single appended state token. Ids are identical across the batch,
        so we take the [0] slice -> [N, 4].
        """
        target_img_ids = inputs["target_img_ids"]  # [B, target_len, 4]
        ref_img_ids = inputs["ref_image_ids"]       # [B, cond_len, 4]
        img_ids = torch.cat([target_img_ids, ref_img_ids], dim=1)[0]  # [target+cond, 4]

        txt_len = int(inputs["text_hidden_states"].shape[1])
        # The three-stream ALWAYS appends exactly one state token to the text stream.
        total_txt = txt_len + 1
        # Text ids: FLUX text axis = L, other axes 0.
        txt_ids = torch.zeros(total_txt, img_ids.shape[-1], device=img_ids.device, dtype=img_ids.dtype)
        txt_ids[:, -1] = torch.arange(total_txt, device=img_ids.device, dtype=img_ids.dtype)
        return img_ids, txt_ids, total_txt

    def _build_inputs_threestream(self, sample, tiled: bool = False):
        """Like ImageWAM.build_inputs_flux2 but keeps proprio SEPARATE (for state_tokens)
        and returns ref ids under `ref_image_ids` (build_inputs_flux2 stores them as
        `ref_img_ids`). We reuse the exact VAE-encode + reasoner-call logic."""
        video = sample.get("video")
        if "target_latent" in sample and "target_img_ids" in sample:
            target_tokens = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            target_img_ids = sample["target_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("three-stream sample requires `next_frame`/`target_image` or target token fields.")
            target_tokens, target_img_ids = self._encode_flux2_image_tokens(next_frame, time_value=0.0)

        if "ref_image_latents" in sample and "ref_img_ids" in sample:
            ref_tokens = sample["ref_image_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ref_img_ids = sample["ref_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("three-stream sample requires `current_frame`/`input_image` or ref token fields.")
            ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(current_frame, time_value=10.0)

        if getattr(self, "latent_reasoner", None) is not None:
            prompts = sample.get("instruction", sample.get("prompt"))
            if prompts is None:
                raise ValueError("LatentReasoner requires raw `instruction`/`prompt` strings in the sample.")
            text_hidden_states, text_attention_mask = self.latent_reasoner(
                prompts, device=self.device, dtype=self.torch_dtype
            )
        else:
            text_hidden_states, text_attention_mask = self._encode_flux2_text(sample)

        # proprio -> single state token vector [B, state_dim]; NOT concatenated into text.
        proprio = sample.get("proprio")
        state_tokens = None
        if proprio is not None:
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            if proprio.ndim == 3:
                proprio = proprio[:, 0, :]
            elif proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            state_tokens = proprio

        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_image_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "state_tokens": state_tokens,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    def _training_loss_threestream(self, sample, tiled: bool = False):
        inputs = self._build_inputs_threestream(sample, tiled=tiled)
        target_latent = inputs["target_latent"]        # [B, target_len, 128] noisy target base
        context_latent = inputs["ref_image_latents"]   # [B, cond_len, 128]   clean history
        action = inputs["action"]                       # [B, H, action_dim]
        batch_size = int(target_latent.shape[0])

        # ---- video: real sigma; flow-matching noise on the target latent ----
        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=target_latent.dtype
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        # ---- action: INDEPENDENT sigma ----
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # ---- state token (proprio) ----
        state_tokens = inputs["state_tokens"]
        if state_tokens is None:
            state_dim = int(self.three_stream.config.state_dim)
            state_tokens = torch.zeros(batch_size, state_dim, device=self.device, dtype=self.torch_dtype)

        # ---- RoPE ids + view ids ----
        img_ids, txt_ids, _ = self._build_threestream_rope_ids(inputs)
        cond_len = int(context_latent.shape[1])
        context_view_ids = torch.zeros(batch_size, cond_len, dtype=torch.long, device=self.device)

        # text validity mask: reasoner padding (left-pad) + always-valid appended state token
        text_valid_mask = inputs.get("text_attention_mask")
        if text_valid_mask is not None:
            text_valid_mask = text_valid_mask.to(device=self.device, dtype=torch.bool)
            state_col = text_valid_mask.new_ones(text_valid_mask.shape[0], 1)
            text_valid_mask = torch.cat([text_valid_mask, state_col], dim=1)  # [B, seq_txt]

        # ---- three-stream forward: dual flow-matching outputs ----
        image_output, action_output = self.three_stream(
            hidden_states=noisy_latent,
            encoder_hidden_states=inputs["text_hidden_states"],
            timestep=self._scheduler_timestep_to_unit(timestep_video, self.train_video_scheduler),
            img_ids=img_ids,
            txt_ids=txt_ids,
            noisy_actions=noisy_action,
            state_tokens=state_tokens,
            context_hidden_states=context_latent,
            context_view_ids=context_view_ids,
            action_timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
            text_valid_mask=text_valid_mask,
            return_dict=False,
        )

        # ---- video (image) flow-matching loss ----
        video_loss_per_sample = F.mse_loss(
            image_output.float(), target_video.float(), reduction="none"
        ).flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
        )
        loss_video = (video_loss_per_sample * video_weight).mean()

        # ---- action flow-matching loss (reuses ImageWAM helper: pad masking) ----
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=action_output,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def training_loss(self, sample, tiled: bool = False):
        return self._training_loss_threestream(sample, tiled=tiled)
