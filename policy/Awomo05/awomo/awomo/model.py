from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from awomo.awomo.models.backbones.imagewam import ImageWAM
from awomo.awomo.utils.logging_config import get_logger

logger = get_logger(__name__)

ACTION_TOKEN_TIME_VALUE = 20.0
FLUX2_TOKEN_DIM = 128
REF_TOKEN_TIME_VALUE = 10.0
TARGET_TOKEN_TIME_VALUE = 0.0
HISTORY_POOL_GRID = (4, 4)


class ImageWAMActionPatch(ImageWAM):
    def __init__(
        self,
        *args,
        action_patch_dim: int,
        action_patch_scale: float = 1.0,
        action_attn_isolate: bool = False,
        view_time_stride: float = 1.0,
        target_views: Optional[Sequence[int]] = None,
        history_max_slots: Optional[int] = None,
        history_pool_grid: Sequence[int] = HISTORY_POOL_GRID,
        history_vae_micro_batch: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_patch_dim = int(action_patch_dim)
        self.action_patch_scale = float(action_patch_scale)
        self.action_attn_isolate = bool(action_attn_isolate)
        self.view_time_stride = float(view_time_stride)
        self.target_views = None if target_views is None else [int(v) for v in target_views]
        self.history_max_slots = None if history_max_slots is None else int(history_max_slots)
        self.history_pool_grid = (int(history_pool_grid[0]), int(history_pool_grid[1]))
        self.history_tokens_per_frame = self.history_pool_grid[0] * self.history_pool_grid[1]
        self.history_vae_micro_batch = int(history_vae_micro_batch)
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
        logger.info(
            "ImageWAMActionPatch: action_attn_isolate=%s (%s)",
            self.action_attn_isolate,
            "noisy action <-/-> noisy image (stock MoT mask)" if self.action_attn_isolate
            else "full attention inside noisy block (original AP)",
        )
        logger.info(
            "ImageWAMActionPatch: multi-view target_views=%s, view_time_stride=%.1f",
            "all" if self.target_views is None else self.target_views,
            self.view_time_stride,
        )
        logger.info(
            "ImageWAMActionPatch: history pool %dx%d -> %d tokens/frame, max_slots=%s, "
            "vae_micro_batch=%d (0 new parameters)",
            self.history_pool_grid[0], self.history_pool_grid[1], self.history_tokens_per_frame,
            self.history_max_slots if self.history_max_slots is not None else "from tensor",
            self.history_vae_micro_batch,
        )

    def _mask_lens(self, video_pre: dict, img_len: int, horizon: int) -> tuple[int, int]:
        total = int(video_pre["target_len"])
        if total != img_len + horizon:
            raise ValueError(f"target_len mismatch: video_pre={total}, img_len+horizon={img_len + horizon}")
        if self.action_attn_isolate:
            return img_len, horizon
        return total, 0

    # ------------------------------------------------------ multi-view images
    def _view_time_value(self, view_index: int, base_time: float) -> float:
        return float(base_time) + float(view_index) * self.view_time_stride

    def _select_target_views(self, num_views: int) -> list[int]:
        views = list(range(num_views)) if self.target_views is None else list(self.target_views)
        for v in views:
            if not 0 <= v < num_views:
                raise ValueError(f"`target_views` index {v} out of range for {num_views} views")
        return views

    def _encode_flux2_view_tokens(
        self,
        frames: torch.Tensor,
        *,
        base_time: float,
        view_indices: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)
        if frames.ndim != 5 or int(frames.shape[2]) != 3:
            raise ValueError(f"`frames` must be [B,V,3,H,W] or [B,3,H,W], got {tuple(frames.shape)}")
        idx = list(range(int(frames.shape[1]))) if view_indices is None else list(view_indices)
        tokens, ids = [], []
        for slot, view in enumerate(idx):
            tok, tok_ids = self._encode_flux2_image_tokens(
                frames[:, slot], time_value=self._view_time_value(view, base_time)
            )
            tokens.append(tok)
            ids.append(tok_ids)
        return torch.cat(tokens, dim=1), torch.cat(ids, dim=1)

    def _build_flux2_view_img_ids(
        self,
        batch_size: int,
        height: int,
        width: int,
        view_indices: Sequence[int],
        base_time: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Token ids for `view_indices` at `base_time` without running the VAE."""
        from awomo.awomo.models.backbones.flux2_video_expert import Flux2VideoExpert

        return torch.cat(
            [
                Flux2VideoExpert.build_img_ids(
                    batch_size=int(batch_size),
                    token_height=int(height) // 16,
                    token_width=int(width) // 16,
                    time_value=self._view_time_value(v, base_time),
                    device=device,
                    dtype=dtype,
                )
                for v in view_indices
            ],
            dim=1,
        )

    # -------------------------------------------------------- pooled history
    def _pool_history_tokens(self, tokens: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
        m, n, c = tokens.shape
        if n != latent_h * latent_w:
            raise ValueError(f"history tokens {n} != {latent_h}x{latent_w}")
        grid = tokens.view(m, latent_h, latent_w, c).permute(0, 3, 1, 2)      # [M,C,h,w]
        pooled = F.adaptive_avg_pool2d(grid.float(), self.history_pool_grid).to(tokens.dtype)
        return pooled.permute(0, 2, 3, 1).reshape(m, self.history_tokens_per_frame, c)

    def _build_history_img_ids(
        self,
        batch_size: int,
        num_slots: int,
        max_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        from awomo.awomo.models.backbones.flux2_video_expert import Flux2VideoExpert

        pool_h, pool_w = self.history_pool_grid
        return torch.cat(
            [
                Flux2VideoExpert.build_img_ids(
                    batch_size=int(batch_size),
                    token_height=pool_h,
                    token_width=pool_w,
                    time_value=float(n - int(max_slots)),
                    device=device,
                    dtype=dtype,
                )
                for n in range(int(num_slots))
            ],
            dim=1,
        )

    def _encode_flux2_history_tokens(
        self,
        frames: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if frames.ndim != 5 or int(frames.shape[2]) != 3:
            raise ValueError(f"`history_frames` must be [B,N,3,H,W], got {tuple(frames.shape)}")
        b, n, _, height, width = frames.shape
        frames = frames.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if valid_mask is None:
            valid = torch.ones(b, n, dtype=torch.bool, device=frames.device)
        else:
            valid = valid_mask.to(device=frames.device, dtype=torch.bool)
            if tuple(valid.shape) != (b, n):
                raise ValueError(f"`history_valid_mask` must be [B,N]={(b, n)}, got {tuple(valid.shape)}")

        tpf = self.history_tokens_per_frame
        pooled = frames.new_zeros(b * n, tpf, FLUX2_TOKEN_DIM)
        flat_valid = valid.reshape(-1)
        if bool(flat_valid.any()):
            flat = frames.reshape(b * n, 3, height, width)[flat_valid]
            chunks = []
            step = max(1, self.history_vae_micro_batch)
            for i in range(0, int(flat.shape[0]), step):
                tok, _ = self._encode_flux2_image_tokens(flat[i: i + step], time_value=0.0)
                chunks.append(tok)
            tok = torch.cat(chunks, dim=0)
            pooled[flat_valid] = self._pool_history_tokens(tok, height // 16, width // 16)
        tokens = pooled.view(b, n * tpf, FLUX2_TOKEN_DIM)

        max_slots = self.history_max_slots if self.history_max_slots is not None else n
        ids = self._build_history_img_ids(b, n, max_slots, tokens.device, tokens.dtype)
        return tokens, ids, valid.repeat_interleave(tpf, dim=1)


    # ------------------------------------------------------------- fixed codec

    def _decode_action_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
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
        from awomo.flux2.model import apply_rope

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
        history_frames: Optional[torch.Tensor] = None,
        history_valid_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim == 4:
            input_image = input_image.unsqueeze(1)          # [B,3,H,W] -> [B,1,3,H,W]
        if input_image.ndim != 5 or input_image.shape[0] != 1 or input_image.shape[2] != 3:
            raise ValueError(
                f"`input_image` must be [1,V,3,H,W], [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="action-patch inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        num_views = int(input_image.shape[1])
        target_views = self._select_target_views(num_views)
        ref_tokens, ref_img_ids = self._encode_flux2_view_tokens(
            input_image, base_time=REF_TOKEN_TIME_VALUE
        )
        batch_size = int(ref_tokens.shape[0])
        horizon = int(action_horizon)


        target_img_ids = self._build_flux2_view_img_ids(
            batch_size=batch_size,
            height=int(input_image.shape[-2]),
            width=int(input_image.shape[-1]),
            view_indices=target_views,
            base_time=TARGET_TOKEN_TIME_VALUE,
            device=ref_img_ids.device,
            dtype=ref_img_ids.dtype,
        )
        target_len = int(target_img_ids.shape[1])

        cond_attention_mask = None
        if history_frames is not None:
            hist_tokens, hist_ids, hist_valid = self._encode_flux2_history_tokens(
                history_frames, history_valid_mask
            )
            ref_tokens = torch.cat([hist_tokens, ref_tokens.to(hist_tokens.dtype)], dim=1)
            ref_img_ids = torch.cat([hist_ids, ref_img_ids.to(hist_ids.dtype)], dim=1)
            cond_attention_mask = torch.cat(
                [
                    hist_valid,
                    torch.ones(
                        ref_tokens.shape[0],
                        int(ref_img_ids.shape[1]) - int(hist_ids.shape[1]),
                        dtype=torch.bool,
                        device=hist_valid.device,
                    ),
                ],
                dim=1,
            )
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
                mask_target_len, mask_action_len = self._mask_lens(video_pre, target_len, horizon)
                attention_mask = self._build_mot_attention_mask_flux2(
                    batch_size=batch_size,
                    txt_len=int(video_pre["txt_len"]),
                    target_len=mask_target_len,
                    cond_len=int(video_pre["cond_len"]),
                    action_len=mask_action_len,
                    device=self.device,
                    text_attention_mask=video_pre["text_mask"],
                    cond_attention_mask=cond_attention_mask,
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
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_patch_dim: int = 14,
        action_patch_scale: float = 1.0,
        action_attn_isolate: bool = False,
        view_time_stride: float = 1.0,
        target_views: Optional[Sequence[int]] = None,
        history_max_slots: Optional[int] = None,
        history_pool_grid: Sequence[int] = HISTORY_POOL_GRID,
        history_vae_micro_batch: int = 16,
    ):
        from safetensors.torch import load_file as load_sft

        from awomo.awomo.models.backbones.flux2_imports import ensure_flux2_importable
        from awomo.awomo.models.backbones.flux2_video_expert import Flux2VideoExpert
        from awomo.awomo.models.backbones.mot import MoT

        ensure_flux2_importable(flux2_src_path)
        from awomo.flux2.autoencoder import AutoEncoder, AutoEncoderParams

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
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            stack="flux2",
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=bool(pack_proprio_after_text),
            concept_k=0,
            lambda_concept=0.0,
            action_patch_dim=action_patch_dim,
            action_patch_scale=action_patch_scale,
            action_attn_isolate=action_attn_isolate,
            view_time_stride=view_time_stride,
            target_views=target_views,
            history_max_slots=history_max_slots,
            history_pool_grid=history_pool_grid,
            history_vae_micro_batch=history_vae_micro_batch,
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
