"""LatentAuxiliaryModel: config-driven bundle of LatentTrunk + task heads.

Always:
  - LatentTrunk → attended_patch Z
  - TCPPoseHead → current-frame TCP (k=0, obs-only)

Optional (``dynamics_cfg``):
  - DynamicsHead → future TCP + RAE-style semantic diffusion at horizon ``T``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from sapolicy.models.latent_trunk import LatentTrunk
from sapolicy.models.dynamics_head import DynamicsHead

# Canonical third-view output name in coupled camera_pair_choices recipes.
_DYNAMICS_DEFAULT_CAMERA = "agentview"
_DYNAMICS_FORBIDDEN_CAMERAS = frozenset(
    {"robot0_eye_in_hand", "left_camera", "right_camera"}
)
from sapolicy.models.tcp_pose_head import TCPPoseHead, remap_legacy_tcp_pool_state_dict


class LatentAuxiliaryModel(nn.Module):
    """Shared latent trunk + current-pose and optional video-dynamics heads."""

    def __init__(
        self,
        in_dim,
        embed_dim,
        patch_size,
        num_heads=4,
        num_layers=4,
        use_depth=True,
        use_camera_intrinsics=True,
        geo_mode="ray_depth",
        final_norm=True,
        geo_gate_init=0.0,
        geo_embed_init_std=0.0,
        *,
        dynamics_cfg: Optional[Dict[str, Any]] = None,
        feat_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        num_tcp: int = 1,
        pool_num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_tcp = int(num_tcp)
        if self.num_tcp < 1:
            raise ValueError(f"num_tcp must be >= 1, got {num_tcp}")

        self.trunk = LatentTrunk(
            in_dim=in_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            use_depth=use_depth,
            use_camera_intrinsics=use_camera_intrinsics,
            geo_mode=geo_mode,
            final_norm=final_norm,
            geo_gate_init=geo_gate_init,
            geo_embed_init_std=geo_embed_init_std,
        )
        self.pose_head = TCPPoseHead(
            embed_dim,
            num_tcp=self.num_tcp,
            pool_num_heads=int(pool_num_heads),
        )

        self.dynamics_head = None
        self.dynamics_semantic_loss_weight = 0.0
        self.dynamics_tcp_loss_weight = 0.0
        self.dynamics_camera = None
        if dynamics_cfg:
            self._init_dynamics(dynamics_cfg, feat_dim=feat_dim, action_dim=action_dim)

    def load_state_dict(self, state_dict, strict=True):
        state_dict = remap_legacy_tcp_pool_state_dict(state_dict, prefix="")
        return super().load_state_dict(state_dict, strict=strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        remapped = {}
        keys_to_delete = []
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            rel = key[len(prefix):]
            if rel == "tcp_queries" or rel.startswith("tcp_pool_attn.") or rel.startswith("tcp_pool_norm."):
                new_key = prefix + "pose_head." + rel
                if new_key not in state_dict:
                    remapped[new_key] = state_dict[key]
                    keys_to_delete.append(key)
        for key in keys_to_delete:
            del state_dict[key]
        state_dict.update(remapped)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_dynamics(
        self,
        dynamics_cfg: Dict[str, Any],
        *,
        feat_dim: Optional[int],
        action_dim: Optional[int] = None,
    ) -> None:
        del action_dim  # dynamics no longer consumes explicit action tokens
        dyn_cfg = dict(dynamics_cfg)
        if feat_dim is None:
            raise ValueError(
                "LatentAuxiliaryModel dynamics_cfg requires feat_dim (backbone fused dim)"
            )
        predict_semantic = bool(
            dyn_cfg.get("predict_semantic", dyn_cfg.get("predict_video", True))
        )
        predict_tcp = bool(dyn_cfg.get("predict_tcp", False))
        if not predict_semantic and not predict_tcp:
            raise ValueError("dynamics_cfg requires predict_semantic and/or predict_tcp")
        prediction_horizon = int(dyn_cfg.get("prediction_horizon", 16))
        self.dynamics_head = DynamicsHead(
            q_dim=self.embed_dim,
            feat_dim=int(feat_dim),
            prediction_horizon=prediction_horizon,
            embed_dim=int(dyn_cfg.get("embed_dim", 512)),
            num_heads=int(dyn_cfg.get("num_heads", 8)),
            num_cross_layers=int(dyn_cfg.get("num_cross_layers", dyn_cfg.get("num_layers", 2))),
            dropout=float(dyn_cfg.get("dropout", 0.0)),
            predict_tcp=predict_tcp,
            predict_semantic=predict_semantic,
            num_tcp=int(dyn_cfg.get("num_tcp", 1)),
            pool_num_heads=int(dyn_cfg.get("pool_num_heads", 4)),
            wide_dim=int(dyn_cfg.get("wide_dim", 2048)),
            num_dit_layers=int(dyn_cfg.get("num_dit_layers", 4)),
            num_inference_steps=int(dyn_cfg.get("num_inference_steps", 4)),
            tcp_uv_weight=float(dyn_cfg.get("tcp_uv_weight", 1.0)),
            tcp_pos3d_weight=float(dyn_cfg.get("tcp_pos3d_weight", 1.0)),
            tcp_rot6d_weight=float(dyn_cfg.get("tcp_rot6d_weight", 1.0)),
            tcp_valid_weight=float(dyn_cfg.get("tcp_valid_weight", 0.2)),
        )
        self.dynamics_semantic_loss_weight = float(
            dyn_cfg.get(
                "semantic_loss_weight",
                dyn_cfg.get("video_loss_weight", 0.1 if predict_semantic else 0.0),
            )
        )
        self.dynamics_tcp_loss_weight = float(dyn_cfg.get("tcp_loss_weight", 1.0 if predict_tcp else 0.0))
        if predict_semantic and self.dynamics_semantic_loss_weight <= 0:
            raise ValueError("predict_semantic=true requires semantic_loss_weight > 0")
        if predict_tcp and self.dynamics_tcp_loss_weight <= 0:
            raise ValueError("predict_tcp=true requires tcp_loss_weight > 0")
        cam = dyn_cfg.get("camera_name", _DYNAMICS_DEFAULT_CAMERA)
        if cam in (None, "", "null"):
            cam = _DYNAMICS_DEFAULT_CAMERA
        cam = str(cam)
        if cam in _DYNAMICS_FORBIDDEN_CAMERAS or "eye_in_hand" in cam:
            raise ValueError(
                f"dynamics_cfg.camera_name={cam!r} must be the third-view canonical "
                f"camera ({_DYNAMICS_DEFAULT_CAMERA}), not a wrist/aux camera."
            )
        self.dynamics_camera = cam

    def forward(self, visual_tokens, depths=None, camera_intrinsics=None):
        """
        visual_tokens / depths / camera_intrinsics: dicts keyed by camera name.

        Returns dict with tcp_uv/3d/6d/valid(+logit) and attended_patch.
        """
        tcp_uv, tcp_3d, tcp_6d, tcp_valid, tcp_valid_logit, attended_patch = {}, {}, {}, {}, {}, {}
        for camera_name in visual_tokens.keys():
            attended = self.trunk.forward_camera(
                visual_tokens[camera_name],
                None if depths is None else depths.get(camera_name),
                None if camera_intrinsics is None else camera_intrinsics.get(camera_name),
            )
            attended_patch[camera_name] = attended
            B, D, T, H, W = attended.shape
            tokens = (
                attended.permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(B * T, H * W, D)
            )
            pose = self.pose_head.forward_tokens(tokens)
            tcp_uv[camera_name] = pose["tcp_uv"]
            tcp_3d[camera_name] = pose["tcp_3d"]
            tcp_6d[camera_name] = pose["tcp_6d"]
            tcp_valid[camera_name] = pose["tcp_valid"]
            tcp_valid_logit[camera_name] = pose["tcp_valid_logit"]

        return {
            "tcp_uv": tcp_uv,
            "tcp_3d": tcp_3d,
            "tcp_6d": tcp_6d,
            "tcp_valid": tcp_valid,
            "tcp_valid_logit": tcp_valid_logit,
            "attended_patch": attended_patch,
        }
