"""Policy adapter for closed-loop RoboMimic evaluation."""

from __future__ import annotations

import glob
import os

import torch

class SAPolicyEvalAdapter:
    def __init__(self, pipeline: torch.nn.Module, device: torch.device):
        self.pipeline = pipeline
        self.device = device
        first_param = next(self.pipeline.parameters())
        self.dtype = first_param.dtype

    def eval(self):
        self.pipeline.eval()

    def reset(self):
        if hasattr(self.pipeline, "reset"):
            self.pipeline.reset()


def _instantiate_vla_pipeline(
    ckpt_path: str,
    device: torch.device,
    encoder: str,
    backbone_type: str,
    load_pretrain_backbone: str = "",
    use_action_head: bool = False,
    action_length: int = 16,
    obs_hist_length: int = 1,
    use_camera_intrinsics: bool = True,
    use_state: bool = True,
    state_dim: int = 7,
    action_orn_mode: str = "6d",
    num_cameras: int = 1,
    use_latent_aux_model: bool = True,
) -> torch.nn.Module:
    from sapolicy.models.sa_policy import SAPolicy

    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} missing 'state_dict' entry.")

    state_dict = ckpt["state_dict"]
    has_action_head_weights = any(
        key.startswith("pipeline.action_head") for key in state_dict.keys()
    )

    include_action_head = use_action_head or has_action_head_weights
    if use_action_head and not has_action_head_weights:
        print(
            "[ERROR] --use-action-head was specified, but the checkpoint does not "
            "contain any 'pipeline.action_head' weights. The action head will be randomly initialised.",
            flush=True,
        )
    elif has_action_head_weights and not use_action_head:
        print(
            "[INFO] Detected action head weights in checkpoint; enabling action head for evaluation.",
            flush=True,
        )

    state_dict = {
        key[len("pipeline.") :]: value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("pipeline.")
    }

    action_cfg = {
        # "action_head_class": "TransformerFlowMatchingHead",
        "action_head_class": "UNetDiffusionHead",
        "sequence_length": action_length,
        "obs_hist_length": obs_hist_length,
        "embed_dim": 256,
        "num_inference_steps": 10,
        "action_orn_mode": action_orn_mode,
        "num_cameras": num_cameras,
    }

    pipeline = SAPolicy(
        encoder=encoder,
        load_pretrain_backbone=load_pretrain_backbone,
        freeze_rgb=True,
        backbone_type=backbone_type,
        use_action_head=include_action_head,
        action_cfg=action_cfg,
        use_camera_intrinsics=use_camera_intrinsics,
        use_state=use_state,
        state_dim=state_dim,
        use_latent_aux_model=use_latent_aux_model,
    )

    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing}", flush=True)
        exit(0)
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}", flush=True)
        exit(0)

    if include_action_head and not has_action_head_weights:
        print(
            "[WARN] Action head is enabled but no pretrained weights were found. "
            "Actions will be generated from randomly initialised parameters.",
            flush=True,
        )
    if not include_action_head and not has_action_head_weights:
        print(
            "[INFO] This checkpoint was trained without an action head. "
            "Only TCP predictions will be available.",
            flush=True,
        )

    pipeline.to(device)
    pipeline.eval()
    return pipeline


def _find_latest_checkpoint(ckpt_path):
    """
    Find the latest checkpoint file if ckpt_path is a directory.
    Returns the original path if it's already a file.
    """
    if os.path.isfile(ckpt_path):
        return ckpt_path

    if os.path.isdir(ckpt_path):
        # Search for .ckpt files in the directory and subdirectories
        pattern = os.path.join(ckpt_path, "**", "*.ckpt")
        ckpt_files = glob.glob(pattern, recursive=True)

        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt files found in directory: {ckpt_path}")

        # Sort by modification time, newest first
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        print(f"[INFO] Found {len(ckpt_files)} checkpoint files, using latest: {latest_ckpt}")
        return latest_ckpt

    raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")


# Backward-compatible aliases
_SAPolicyWrapper = SAPolicyEvalAdapter
instantiate_vla_pipeline = _instantiate_vla_pipeline
find_latest_checkpoint = _find_latest_checkpoint
