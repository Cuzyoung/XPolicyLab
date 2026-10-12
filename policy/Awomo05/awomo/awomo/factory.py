from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from awomo.awomo.utils.logging_config import get_logger

logger = get_logger(__name__)


def _to_dict(value, name: str) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value




def create_imagewam_flux2_klein_actionpatch_vl(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    vl_model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    vl_qwen_layers=(9, 18, 27),
    vl_max_prompt_len: int = 768,
    vl_lora=None,                       # {enabled, r, alpha, dropout, target_modules}
    vl_attn_implementation: str = "sdpa",
    vl_ar=None,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    video_scheduler=None,
    action_patch_dim: int = 14,
    action_patch_scale: float = 1.0,
    # Multi-view input and long-horizon history.
    target_views=None,
    view_time_stride: float = 1.0,
    history_max_slots: int | None = None,
    history_pool_grid=(4, 4),
    history_vae_micro_batch: int = 16,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from awomo.awomo.model_vl import ImageWAMActionPatchVL, VLPromptEncoder

    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    ar_cfg = dict(_to_dict(vl_ar, "vl_ar"))
    for legacy in ("enabled", "boundary_threshold"):
        if legacy in ar_cfg:
            logger.warning("vl_ar.%s is a v1 option and is ignored in VL-v2.", legacy)
            ar_cfg.pop(legacy)

    model = ImageWAMActionPatchVL.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=None,
        qwen_context_len=int(vl_max_prompt_len),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_text_encoder=False,
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_patch_dim=int(action_patch_dim),
        action_patch_scale=float(action_patch_scale),
        target_views=(None if target_views is None else [int(v) for v in target_views]),
        view_time_stride=float(view_time_stride),
        history_max_slots=(None if history_max_slots is None else int(history_max_slots)),
        history_pool_grid=tuple(int(v) for v in history_pool_grid),
        history_vae_micro_batch=int(history_vae_micro_batch),
    )
    enc = VLPromptEncoder(
        vl_model_path,
        qwen_layers=tuple(int(l) for l in vl_qwen_layers),
        summary_layers=tuple(int(l) for l in ar_cfg.get("summary_layers", (18, 27, 36))),
        max_prompt_len=int(vl_max_prompt_len),
        device=device,
        torch_dtype=model_dtype,
        lora=_to_dict(vl_lora, "vl_lora"),
        attn_implementation=str(vl_attn_implementation),
        image_size=tuple(ar_cfg.get("image_size", (448, 448))),
        history_image_size=tuple(ar_cfg.get("history_image_size", (448, 448))),
        history_pool=tuple(ar_cfg.get("history_pool", (4, 4))),
        history_num_slots=int(ar_cfg.get("history_num_slots", 20)),
        history_period_s=float(ar_cfg.get("history_period_s", 1.0)),
        history_fps=float(ar_cfg.get("history_fps", 25.0)),
        max_new_tokens=int(ar_cfg.get("max_new_tokens", 24)),
        decode_every=int(ar_cfg.get("decode_every", 4)),
    )
    model.attach_vl_encoder(enc)
    return model
