"""Factory for the PatchWAM Action-as-Patch model."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf


def _to_dict(value, name: str) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must resolve to a dict, got {type(value)}")
    return value


def create_imagewam_flux2_klein_actionpatch(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    qwen3_model_spec: str | None = None,
    qwen_context_len: int = 128,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    video_scheduler=None,
    loss=None,
    action_patch_dim: int = 14,
    action_patch_scale: float = 1.0,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from exploration.action_img_patch.model import ImageWAMActionPatch

    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    loss = _to_dict(loss, "loss")
    return ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=qwen3_model_spec,
        qwen_context_len=int(qwen_context_len),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        load_text_encoder=bool(load_text_encoder),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 0.5)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_patch_dim=int(action_patch_dim),
        action_patch_scale=float(action_patch_scale),
    )
