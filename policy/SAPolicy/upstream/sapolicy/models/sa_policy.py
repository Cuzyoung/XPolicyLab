import os
import random

_DEBUG_NONFINITE = os.environ.get("SA_DEBUG_NONFINITE", "0") == "1"  # host-syncing NaN guards, off by default
import contextlib
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from sapolicy.logger import Log
from collections.abc import Sequence
from typing import List, Optional, Tuple
from distutils.util import strtobool

# Backbones are imported lazily inside __init__ so environments without a full
# dependency chain for one backbone (e.g. Stampede3 without depth_anything_3) can
# still import sa_policy to use another backbone (VGGT / DINOv2).
from .utils.rotation import matrix_to_rotation_6d
from sapolicy.models.action_head.dit import DiTActionHead
from sapolicy.models.action_head.transformer import (
    TransformerFlowMatchingHead,
    TransformerDiffusionHead,
)
from sapolicy.models.backbone.checkpoint import (
    _load_pretrained_weights,
    _prepare_state_dict,
)

from sapolicy.dataset.normalizer import LinearNormalizer
from sapolicy.models.utils.crop_randomizer import CropRandomizer


def _as_hw(value):
    """Normalize image size to (h, w). Scalar means square."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"image size must be scalar or (h, w), got {value!r}")
        return int(value[0]), int(value[1])
    v = int(value)
    return v, v


_COMPILED_FNS = {}  # (id(module), key) -> torch.compile'd bound method (see SAPolicy._compiled_fn)


@dataclass
class ObservationFeatures:
    """Observation encoder result consumed by TCP and action stages."""

    visual_tokens: dict
    depth_inputs: dict
    intrinsics_inputs: dict
    tcp_valid_inputs: dict
    future_tcp_uv_targets: dict
    future_tcp_valid_inputs: dict
    input_hw: tuple
    batch_size: int
    time_steps: int
    crop_offsets: dict = field(default_factory=dict)  # cam -> (offsets_bt [B*T,2], h_pre, w_pre)


class SAPolicy(nn.Module):
    """
    Spatial Alignment Policy (SAPolicy).

    Vision-based policy for robotic manipulation: predicts TCP (Tool Center Point)
    3D pose from RGB-D images and generates robot action sequences.
    Supports multiple vision backbones including DINOv2 and DINOv3.

    Args:
        encoder (str): Encoder architecture size. One of ['vits', 'vitb', 'vitl', 'vitg']
            - vits: ViT-Small (384 dim, 12 layers)
            - vitb: ViT-Base (768 dim, 12 layers)
            - vitl: ViT-Large (1024 dim, 24 layers)
            - vitg: ViT-Giant (1536 dim, 40 layers)
        load_pretrain_backbone (str): Path to pretrained backbone weights
        load_pretrain_net (str): Path to pretrained full network weights
        freeze_rgb (bool): Whether to freeze RGB encoder during training (default: True)
        backbone_type (str): Vision backbone type. One of ['dinov2', 'dinov3', 'da3', 'vggt'] (default: 'dinov2')
            - 'dinov2': DINOv2 vision transformer (stable, well-tested)
            - 'dinov3': DINOv3 vision transformer (improved features, better zero-shot)
        use_registers (bool): Whether to use register tokens (DINOv3 feature) (default: True)
            Register tokens can improve model performance but add computational cost.
            Only applicable when backbone_type='dinov3'
        use_action_head (bool): Whether to include action prediction head (default: False)
        action_cfg (dict): Configuration for action head (optional)

    Example:
        # Using DINOv2 (default)
        model = SAPolicy(encoder='vitl', backbone_type='dinov2')

        # Using DINOv3 with register tokens and action head
        model = SAPolicy(encoder='vitl', backbone_type='dinov3', use_registers=True,
                     use_action_head=True, action_cfg={'action_dim': 7, 'sequence_length': 15})
    """
    def __init__(
        self,
        encoder='vitl',
        load_pretrain_backbone=None, # Path to pretrained backbone weights, for DA3 and DINOv2 (both used Dino)
        load_pretrain_net=None,
        freeze_rgb=True,
        freeze_action_head=False,
        backbone_type='dinov2',  # one of: 'dinov2', 'dinov3', 'vggt', 'da3'
        use_registers=True,  # Official DINOv3 backbones use four register/storage tokens
        use_action_head=False,  # Whether to use action head
        action_cfg=None,  # Action head configuration
        use_state=False,
        state_dim=7,
        use_random_crop=False,  # random crop augmentation (84->70)
        use_central_crop=False,  # if True, force center crop both train+eval (no random aug) (84->70)
        crop_height=70,
        crop_width=70,
        input_image_size=84,  # expected input H,W before crop
        use_depth=True,  # Enable depth-conditioned geometry by default
        gt_depth_min=0.1,  # min depth in meters (for GT depth denormalization)
        gt_depth_max=5.0,  # max depth in meters
        use_camera_intrinsics=True,  # Enable camera intrinsics conditioning by default
        cross_view_fuse=False,  # DA3/VGGT: fuse multi-view tokens in a single backbone forward
        temporal_fuse=False,  # DA3/VGGT: fuse T_obs frames per-camera via S-dim attention
        # === V2 Architecture ===
        use_latent_aux_model=False,  # LatentAuxiliaryModel: TCPPoseHead + attended_patch (+ optional Dynamics)
        tcp_cfg={},  # Config for LatentAuxiliaryModel
        aux_bypass_cameras=None,  # Cameras routed from backbone to action head without auxiliary heads
        encode_only_consumed_frames=False,  # Backbone: skip obs-history frames nobody consumes (see _frames_to_encode)
        compile_backbone=False,  # torch.compile the frozen backbone's get_intermediate_layers (fuses LN/GELU/residual)
        compile_action_head=False,  # torch.compile DiTActionHead._forward_cond (training step + sampling step)
        compile_dynamics_head=False,  # torch.compile the semantic flow-matching velocity net (eager 4-layer DiT is ~40% of a futsem step)
        compile_latent_trunk=False,  # torch.compile LatentTrunk.forward_camera (eager today; needs the _create_freqs shape fix)
        compile_mode="default",  # torch.compile mode; keep 'default' on sm_120 (max-autotune crashes there)
        top_rotation_deg=0.0,  # train-time roll aug for third-person 'top' cameras (+-deg, official ABC uses 5); labels rotate consistently
        aux_bypass_mlp_hidden_dim=None,  # Hidden width of the token-wise action adapter
        tcp_loss_weight=1.0,  # Weight for current-frame TCP auxiliary loss
        action_consistency_weight=0.0,  # Weight for action consistency loss (R68/ACL)
        action_consistency_uv_weight=0.0,  # Extra future-UV term inside action consistency loss
        backbone_feat_mode='last',  # 'last' | 'concat' (channel-concat selected intermediate layers; DINO/DA3/VGGT)
        backbone_feat_layers=None,  # Optional override for intermediate_layer_idx[encoder] (DINO blocks / DA3 OUT_LAYERS / VGGT aggregator levels)
        use_dynamics_head=False,
        dynamics_cfg=None,
    ):
        super(SAPolicy, self).__init__()
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.freeze_rgb = freeze_rgb
        self.backbone_type = backbone_type.lower()
        self.backbone_feat_mode = str(backbone_feat_mode).lower()
        if self.backbone_feat_mode not in ('last', 'concat'):
            raise ValueError(
                f"backbone_feat_mode must be 'last' or 'concat', got {backbone_feat_mode!r}"
            )
        use_registers = (
            use_registers
            if isinstance(use_registers, bool)
            else bool(strtobool(use_registers))
        )

        self.encoder = 'vitl' if self.backbone_type == 'vggt' else ('vitb' if self.backbone_type == 'da3' else encoder)
        self.raw_backbone_dim = {'vits': 384, 'vitb': 768, 'vitl': 1024, 'vitg': 1536}.get(self.encoder, 384)
        if backbone_feat_layers is not None:
            self.intermediate_layer_idx = dict(self.intermediate_layer_idx)
            self.intermediate_layer_idx[self.encoder] = [int(x) for x in backbone_feat_layers]
        elif self.backbone_type == 'da3':
            # DA3 OUT_LAYERS = [5, 7, 9, 11] (not DINOv2-B's [2, 5, 8, 11]).
            self.intermediate_layer_idx = dict(self.intermediate_layer_idx)
            self.intermediate_layer_idx['vitb'] = [5, 7, 9, 11]
        # VGGT aggregator depth=24 → reuse vitl defaults [4, 11, 17, 23].
        self.use_state = use_state if isinstance(use_state, bool) else strtobool(use_state)
        self.normalizer = LinearNormalizer()

        action_cfg = dict(action_cfg) if action_cfg else {}
        state_dim = int(state_dim) if self.use_state else 0
        tcp_loss_kwargs = dict(
            uv_weight=float(tcp_cfg.get('uv_weight', 1.0)),
            pos3d_weight=float(tcp_cfg.get('pos3d_weight', 1.0)),
            rot6d_weight=float(tcp_cfg.get('rot6d_weight', 1.0)),
            proj_gt_weight=float(tcp_cfg.get('proj_gt_weight', 1.0)),
            proj_self_weight=float(tcp_cfg.get('proj_self_weight', 1.0)),
        )

        self.use_depth = use_depth
        self.use_latent_aux_model = (
            use_latent_aux_model
            if isinstance(use_latent_aux_model, bool)
            else bool(strtobool(use_latent_aux_model))
        )
        self.use_camera_intrinsics = use_camera_intrinsics
        self.use_action_head = use_action_head
        _use_dyn = (
            use_dynamics_head
            if isinstance(use_dynamics_head, bool)
            else bool(strtobool(use_dynamics_head))
        )
        if _use_dyn and not self.use_latent_aux_model:
            raise ValueError(
                "use_dynamics_head=true requires use_latent_aux_model=true "
                "(DynamicsHead Q is LatentAuxiliaryModel attended_patch)."
            )
        # Stash for DynamicsHead init after tcp_embed_dim is known.
        self._dynamics_init_requested = _use_dyn
        self._dynamics_cfg_raw = dict(dynamics_cfg) if dynamics_cfg else {}
        self.use_oracle_tcp = False  # eval diagnostic: replace pred TCP with GT from state
        self.cross_view_fuse = cross_view_fuse if isinstance(cross_view_fuse, bool) else strtobool(cross_view_fuse)
        self.temporal_fuse = temporal_fuse if isinstance(temporal_fuse, bool) else strtobool(temporal_fuse)
        if use_camera_intrinsics:
            Log.info(f"[Camera Intrinsics] Enabled: 4-dim normalized intrinsics")

        # Create backbone based on type
        Log.info(f"Using backbone: {backbone_type}")
        Log.info(f"Using depth: {use_depth}")

        self.latent_aux_model = None
        self.tcp_aux_loss = None
        if isinstance(aux_bypass_cameras, str):
            aux_bypass_cameras = (aux_bypass_cameras,)
        self.aux_bypass_cameras = frozenset(aux_bypass_cameras or ())
        self.encode_only_consumed_frames = (
            encode_only_consumed_frames if isinstance(encode_only_consumed_frames, bool)
            else bool(strtobool(encode_only_consumed_frames))
        )
        self.compile_backbone = compile_backbone if isinstance(compile_backbone, bool) else bool(strtobool(compile_backbone))
        self.compile_action_head = compile_action_head if isinstance(compile_action_head, bool) else bool(strtobool(compile_action_head))
        self.compile_mode = str(compile_mode)
        self.top_rotation_deg = float(top_rotation_deg)
        self.compile_dynamics_head = bool(compile_dynamics_head) if not isinstance(compile_dynamics_head, str) else compile_dynamics_head.strip().lower() in ('1','true','yes','y','t')
        self.compile_latent_trunk = bool(compile_latent_trunk) if not isinstance(compile_latent_trunk, str) else compile_latent_trunk.strip().lower() in ('1','true','yes','y','t')
        self.action_bypass_adapters = nn.ModuleDict()

        backbone_log_name = None

        if self.backbone_type == 'da3':
            from .backbone.da3 import DA3Encoder  # lazy: requires depth_anything_3
            # Dual-stream (local+global) per level; optionally channel-concat levels.
            self.encoder = 'vitb'  # DA3 uses ViT-B architecture
            per_level_dim = self.raw_backbone_dim * 2
            layer_list = self.intermediate_layer_idx.get(self.encoder, DA3Encoder.OUT_LAYERS)
            n_levels = len(layer_list) if self.backbone_feat_mode == 'concat' else 1
            self.fused_feat_dim = per_level_dim * n_levels
            self.pretrained = DA3Encoder(pretrained_path=encoder)
            Log.info(f"DA3 backbone loaded and frozen: "
                     f"{sum(p.numel() for p in self.pretrained.parameters())/1e6:.1f}M params, "
                     f"embed_dim={self.pretrained.embed_dim}")
            Log.info(
                f"[RGB backbone] feat_mode={self.backbone_feat_mode}, "
                f"layers={layer_list}, patch_dim={self.fused_feat_dim}"
            )
            backbone_log_name = "DA3"

        elif self.backbone_type == 'vggt':
            from .backbone.vggt import VGGTBackbone  # lazy: requires vggt pkg
            # Dual-stream (frame+global) per level; optionally channel-concat levels.
            per_level_dim = self.raw_backbone_dim * 2
            layer_list = self.intermediate_layer_idx.get(self.encoder, [4, 11, 17, 23])
            n_levels = len(layer_list) if self.backbone_feat_mode == 'concat' else 1
            self.fused_feat_dim = per_level_dim * n_levels

            vggt_ckpt = load_pretrain_backbone or encoder or None
            if isinstance(vggt_ckpt, str) and vggt_ckpt.strip() == "":
                vggt_ckpt = None
            self.pretrained = VGGTBackbone(
                pretrained=True, img_size=84, pretrained_path=vggt_ckpt
            )
            Log.info(f"VGGT backbone loaded and frozen: "
                     f"{sum(p.numel() for p in self.pretrained.parameters())/1e6:.1f}M params, "
                     f"embed_dim={self.pretrained.embed_dim}, output_dim={self.pretrained.hidden_size}")
            Log.info(
                f"[RGB backbone] feat_mode={self.backbone_feat_mode}, "
                f"layers={layer_list}, patch_dim={self.fused_feat_dim}"
            )
            backbone_log_name = "VGGT"

        elif self.backbone_type == 'dinov2':
            from .backbone.dinov2 import DINOv2  # lazy
            # Keep the visual backbone RGB-only; depth is reserved for the TCP head.
            self.fused_feat_dim = self.raw_backbone_dim

            self.pretrained = DINOv2(model_name=encoder)
            Log.info(f"DINOv2 backbone loaded: {encoder}, "
                     f"{sum(p.numel() for p in self.pretrained.parameters())/1e6:.1f}M params, "
                     f"embed_dim={self.raw_backbone_dim}")

            # Load pretrained backbone weights
            if load_pretrain_backbone is not None and load_pretrain_backbone != "":
                assert os.path.exists(load_pretrain_backbone), f"Pretrained backbone not found: {load_pretrain_backbone}"
                Log.info("Load pretrain backbone from {}".format(load_pretrain_backbone))
                checkpoint = torch.load(load_pretrain_backbone, map_location="cpu")
                checkpoint = _prepare_state_dict(checkpoint)

                # Detect CDM-format checkpoint (dual-branch keys with prefixes)
                has_rgb_prefix = any(k.startswith('pretrained.') for k in checkpoint.keys())
                if has_rgb_prefix:
                    Log.info("[CDM] Detected CDM-format checkpoint with prefixed keys")
                    rgb_state = {k[len('pretrained.'):]: v
                                for k, v in checkpoint.items() if k.startswith('pretrained.')}
                    _load_pretrained_weights(
                        self.pretrained, rgb_state,
                        allow_partial=False, log_prefix="[CDM][RGB] "
                    )
                else:
                    log_prefix = f"[{self.backbone_type.upper()}][RGB] "
                    _load_pretrained_weights(
                        self.pretrained,
                        checkpoint,
                        allow_partial=False,
                        log_prefix=log_prefix,
                    )

            backbone_log_name = "DINOv2"

        elif self.backbone_type == 'dinov3':
            from .backbone.dinov3 import DINOv3  # lazy
            # Keep the visual backbone RGB-only; depth is reserved for the TCP head.
            self.fused_feat_dim = self.raw_backbone_dim

            self.pretrained = DINOv3(
                model_name=encoder,
                # DINOv3 sizes its pos-enc table from one scalar and interpolates
                # per forward, so the longer side is the safe choice for non-square input.
                img_size=max(_as_hw(input_image_size)),
                use_registers=use_registers,
            )
            Log.info(
                f"DINOv3 backbone loaded: {encoder}, "
                f"{sum(p.numel() for p in self.pretrained.parameters())/1e6:.1f}M params, "
                f"embed_dim={self.raw_backbone_dim}, "
                f"storage_tokens={self.pretrained.n_storage_tokens}"
            )

            if load_pretrain_backbone is not None and load_pretrain_backbone != "":
                assert os.path.exists(load_pretrain_backbone), f"Pretrained backbone not found: {load_pretrain_backbone}"
                Log.info("Load pretrain backbone from {}".format(load_pretrain_backbone))
                checkpoint = torch.load(load_pretrain_backbone, map_location="cpu")
                checkpoint = _prepare_state_dict(checkpoint)
                _load_pretrained_weights(
                    self.pretrained,
                    checkpoint,
                    allow_partial=False,
                    log_prefix="[DINOV3][RGB] ",
                    fallback_to_partial=False,
                )

            backbone_log_name = "DINOv3"
        else:
            raise ValueError(
                f"Unknown backbone_type: {backbone_type}. "
                "Choose 'dinov2', 'dinov3', 'da3', or 'vggt'"
            )

        if self.backbone_type in ("dinov2", "dinov3"):
            self.fused_feat_dim = self._resolve_visual_patch_dim()
            layer_list = self.intermediate_layer_idx.get(self.encoder, [])
            Log.info(
                f"[RGB backbone] feat_mode={self.backbone_feat_mode}, "
                f"layers={layer_list}, patch_dim={self.fused_feat_dim}"
            )

        if hasattr(self, "pretrained"):
            self._apply_backbone_freeze_policy()

        # Crop/randomizer must be set BEFORE action head init because
        # _dit_num_spatial_patches_per_cam reads self.randomizer / crop_height / crop_width.
        self.use_random_crop = bool(use_random_crop) if isinstance(use_random_crop, bool) else bool(strtobool(use_random_crop))
        self.use_central_crop = bool(use_central_crop) if isinstance(use_central_crop, bool) else bool(strtobool(use_central_crop))
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        # input_image_size is an int (square) or a (h, w) pair -- the ring cameras
        # render at the sensor's native 4:3, so the pipeline must not assume h == w.
        self.input_image_h, self.input_image_w = _as_hw(input_image_size)
        self.input_image_size = self.input_image_h  # legacy scalar alias
        if self.use_random_crop or self.use_central_crop:
            self.randomizer = CropRandomizer(
                input_shape=(3, self.input_image_h, self.input_image_w),
                crop_height=crop_height,
                crop_width=crop_width,
                num_crops=1,
                central_crop=self.use_central_crop,
            )
        else:
            self.randomizer = None

        if self.backbone_type in ("dinov2", "dinov3"):
            patch_size = int(getattr(self.pretrained, "patch_size", 1))
            image_h = self.crop_height if self.randomizer is not None else self.input_image_h
            image_w = self.crop_width if self.randomizer is not None else self.input_image_w
            if image_h % patch_size != 0 or image_w % patch_size != 0:
                raise ValueError(
                    f"{self.backbone_type} input after crop is {image_h}x{image_w}, "
                    f"which is not divisible by patch size {patch_size}. "
                    "Adjust input_image_size/crop_height/crop_width to exact multiples."
                )

        default_embed_dim = 2048
        tcp_patch_size = self.pretrained.patch_size if hasattr(self, 'pretrained') else 1
        tcp_embed_dim = int(tcp_cfg.get('embed_dim', default_embed_dim))
        if self.use_latent_aux_model:
            action_dim_for_dyn = int(action_cfg.get("action_dim", 10)) if action_cfg else 10
            dynamics_cfg_payload = None
            if getattr(self, "_dynamics_init_requested", False):
                dynamics_cfg_payload = dict(getattr(self, "_dynamics_cfg_raw", {}) or {})
                if action_cfg:
                    dynamics_cfg_payload.setdefault(
                        "prediction_horizon",
                        int(action_cfg.get("sequence_length", 16)),
                    )
                dynamics_cfg_payload.setdefault(
                    "num_tcp",
                    int(tcp_cfg.get("num_tcp", 1)),
                )
            self._init_latent_aux_model(
                in_dim=self.fused_feat_dim,
                patch_size=tcp_patch_size,
                tcp_cfg=tcp_cfg,
                tcp_loss_weight=tcp_loss_weight,
                default_embed_dim=default_embed_dim,
                log_name=backbone_log_name,
                loss_kwargs=tcp_loss_kwargs,
                dynamics_cfg=dynamics_cfg_payload,
                feat_dim=int(self.fused_feat_dim),
                action_dim=action_dim_for_dyn,
            )

        if self.aux_bypass_cameras:
            if not self.use_latent_aux_model:
                raise ValueError("aux_bypass_cameras requires use_latent_aux_model=true")
            dynamics_camera = self._dynamics_cfg_raw.get("camera_name")
            if dynamics_camera in self.aux_bypass_cameras:
                raise ValueError(
                    "dynamics_cfg.camera_name cannot be an aux_bypass_cameras entry."
                )
            hidden_dim = int(aux_bypass_mlp_hidden_dim or tcp_embed_dim)
            for camera_name in sorted(self.aux_bypass_cameras):
                self.action_bypass_adapters[camera_name] = nn.Sequential(
                    nn.Linear(self.fused_feat_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, tcp_embed_dim),
                )
            Log.info(
                f"[Action bypass] cameras={sorted(self.aux_bypass_cameras)}, "
                f"per-camera adapters={self.fused_feat_dim}->{hidden_dim}->{tcp_embed_dim}"
            )

        if self.use_action_head:
            # DiT/FM consume LatentAuxiliaryModel attended_patch (tcp_embed_dim) when
            # enabled; otherwise raw backbone features (fused_feat_dim).
            patch_in_dim = tcp_embed_dim if self.use_latent_aux_model else self.fused_feat_dim
            self._init_action_head(
                patch_in_dim=patch_in_dim,
                action_cfg=action_cfg,
                default_embed_dim=default_embed_dim,
                log_name=backbone_log_name,
                state_dim=state_dim,
            )

        self.freeze_action_head = freeze_action_head
        if self.use_action_head:
            if freeze_action_head:
                for param in self.action_head.parameters():
                    param.requires_grad = False

        else:
            self.action_head = None

        # Dynamics lives under LatentAuxiliaryModel when enabled.
        self.use_dynamics_head = bool(
            self.use_latent_aux_model
            and self.latent_aux_model is not None
            and getattr(self.latent_aux_model, "dynamics_head", None) is not None
        )
        self.dynamics_head = (
            self.latent_aux_model.dynamics_head if self.use_dynamics_head else None
        )
        if self.use_dynamics_head:
            self._dynamics_semantic_loss_weight = float(
                self.latent_aux_model.dynamics_semantic_loss_weight
            )
            self._dynamics_tcp_loss_weight = float(
                self.latent_aux_model.dynamics_tcp_loss_weight
            )
            self._dynamics_camera = self.latent_aux_model.dynamics_camera
            Log.info(
                f"[LatentAux/Dynamics] horizon={self.dynamics_head.prediction_horizon}, "
                f"predict_tcp={self.dynamics_head.predict_tcp}, "
                f"predict_semantic={self.dynamics_head.predict_semantic}, "
                f"tcp_w={self._dynamics_tcp_loss_weight}, "
                f"semantic_w={self._dynamics_semantic_loss_weight}"
            )
        else:
            self._dynamics_semantic_loss_weight = 0.0
            self._dynamics_tcp_loss_weight = 0.0
            self._dynamics_camera = None

        if load_pretrain_net is not None:
            if (".pt" not in load_pretrain_net) and (".ckpt" not in load_pretrain_net):
                # find the latest pt file in the load_pretrain_net folder
                load_pretrain_net = os.path.join(
                    load_pretrain_net,
                    sorted(
                        [
                            f
                            for f in os.listdir(load_pretrain_net)
                            if (f.endswith(".pt") or f.endswith(".ckpt")) and "best" not in f
                        ],
                        key=lambda x: int(x.split("-")[-1].split(".")[0][1:]),
                        reverse=True,
                    )[0],
                )

            Log.info("Load pretrain network from {}".format(load_pretrain_net))
            assert os.path.exists(load_pretrain_net)
            model = torch.load(load_pretrain_net, "cpu")
            strict = False
            if "model" in model:
                model = model["model"]
                # Strip "module." prefix (DataParallel/DDP) if present
                prefix = "module."
                if any(k.startswith(prefix) for k in model):
                    model = {k[len(prefix):]: v for k, v in model.items()}
                strict = False
            elif "state_dict" in model:
                model = model["state_dict"]
                # Strip "pipeline." prefix (PyTorch Lightning) if present
                prefix = "pipeline."
                if any(k.startswith(prefix) for k in model):
                    model = {k[len(prefix):]: v for k, v in model.items()}
                strict = False
            else:
                model = model
                strict = False
            self.load_state_dict(model, strict=strict)

        self.register_buffer(
            "_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # (randomizer/crop_height/crop_width already initialized earlier, before action head)

        self.action_consistency_weight = float(action_consistency_weight)
        self.action_consistency_uv_weight = float(action_consistency_uv_weight)
        if self.action_consistency_weight > 0:
            if getattr(self.action_head, "num_diffusion_draws", 1) != 1:
                # The consistency loss pairs per-sample flow outputs with batch-B TCP
                # predictions; with k draws the head returns k*B rows.
                raise ValueError(
                    "action_consistency_weight > 0 is not supported with "
                    "action_cfg.num_diffusion_draws > 1"
                )
            from sapolicy.models.loss import ActionConsistencyLoss
            self.action_consistency_loss_fn = ActionConsistencyLoss()
            Log.info(
                f"[ActionConsistency] weight={self.action_consistency_weight}, "
                f"uv_weight={self.action_consistency_uv_weight}"
            )

    def _resolve_visual_patch_dim(self):
        """Channel dimension of per-patch visual tokens fed to TCP/DiT (before projection)."""
        layer_list = self.intermediate_layer_idx.get(self.encoder)
        if self.backbone_feat_mode == "concat" and layer_list:
            return self.raw_backbone_dim * len(layer_list)
        return self.raw_backbone_dim

    def _apply_backbone_freeze_policy(self):
        if not self.freeze_rgb or not hasattr(self, "pretrained"):
            return
        for param in self.pretrained.parameters():
            param.requires_grad = False

    def _init_latent_aux_model(
        self,
        *,
        in_dim,
        patch_size,
        tcp_cfg,
        tcp_loss_weight,
        default_embed_dim,
        log_name,
        loss_kwargs,
        dynamics_cfg=None,
        feat_dim=None,
        action_dim=None,
    ):
        from sapolicy.models.latent_aux import LatentAuxiliaryModel
        from sapolicy.models.loss import TCPAuxiliaryLoss

        tcp_embed_dim = int(tcp_cfg.get('embed_dim', default_embed_dim))
        self.latent_aux_model = LatentAuxiliaryModel(
            in_dim=in_dim,
            embed_dim=tcp_embed_dim,
            patch_size=patch_size,
            num_heads=int(tcp_cfg.get('num_heads', 4)),
            num_layers=int(tcp_cfg.get('num_layers', 4)),
            use_depth=self.use_depth,
            use_camera_intrinsics=self.use_camera_intrinsics,
            geo_mode=tcp_cfg.get('geo_mode', "ray_depth"),
            final_norm=tcp_cfg.get('final_norm', True),
            geo_gate_init=float(tcp_cfg.get('geo_gate_init', 0.0)),
            geo_embed_init_std=float(tcp_cfg.get('geo_embed_init_std', 0.0)),
            # >1 for bimanual data; must match the dataset's num_arms.
            num_tcp=int(tcp_cfg.get('num_tcp', 1)),
            pool_num_heads=int(tcp_cfg.get('pool_num_heads', 4)),
            dynamics_cfg=dynamics_cfg,
            feat_dim=feat_dim,
            action_dim=action_dim,
        )
        self.tcp_aux_loss = TCPAuxiliaryLoss(**(loss_kwargs or {}))
        self._tcp_loss_weight = float(tcp_loss_weight)
        self._tcp_valid_loss_weight = float(tcp_cfg.get('valid_weight', 0.2))
        Log.info(
            f"[{log_name}] LatentAuxiliaryModel: in={in_dim}, embed={tcp_embed_dim}, "
            f"num_heads={tcp_cfg.get('num_heads', 4)}, num_layers={tcp_cfg.get('num_layers', 4)}, "
            f"patch_size={patch_size}, tcp_loss_weight={tcp_loss_weight}, "
            f"tcp_valid_weight={self._tcp_valid_loss_weight}, "
            f"dynamics={'on' if dynamics_cfg else 'off'}, "
            f"geo_mode={tcp_cfg.get('geo_mode', 'ray_depth')}, "
            f"geo_gate_init={tcp_cfg.get('geo_gate_init', 0.0)}, "
            f"geo_embed_init_std={tcp_cfg.get('geo_embed_init_std', 0.0)}, "
        )

    def _init_action_head(
        self,
        *,
        patch_in_dim,
        action_cfg,
        default_embed_dim,
        log_name,
        state_dim,
    ):
        """Instantiate action head based on action_cfg.action_head_class.

        Dispatches to DiTActionHead, TransformerFlowMatchingHead, or TransformerDiffusionHead.
        """
        action_head_class = action_cfg.get("action_head_class", "DiTActionHead")
        self._action_head_type = action_head_class

        if action_head_class == "DiTActionHead":
            self._init_dit_action_head(
                patch_in_dim=patch_in_dim,
                action_cfg=action_cfg,
                default_embed_dim=default_embed_dim,
                log_name=log_name,
                state_dim=state_dim,
            )
        elif action_head_class in ("TransformerFlowMatchingHead", "TransformerDiffusionHead"):
            if int(action_cfg.get("num_diffusion_draws", 1)) != 1:
                raise ValueError("action_cfg.num_diffusion_draws > 1 is only implemented for DiTActionHead")
            self._init_fm_action_head(
                patch_in_dim=patch_in_dim,
                action_cfg=action_cfg,
                default_embed_dim=default_embed_dim,
                log_name=log_name,
                state_dim=state_dim,
            )
        else:
            raise ValueError(f"Unknown action_head_class: {action_head_class}")

    def _dit_num_spatial_patches_per_cam(self, action_cfg) -> int:
        """Grid tokens H*W per camera per obs frame."""
        explicit = action_cfg.get("num_spatial_patches_per_cam")
        if explicit is not None:
            return int(explicit)
        if self.randomizer is not None:
            h_i, w_i = int(self.crop_height), int(self.crop_width)
        else:
            h_i, w_i = int(self.input_image_h), int(self.input_image_w)
        if self.backbone_type in ("da3", "dinov2", "dinov3"):
            ps = int(getattr(self.pretrained, "patch_size", 14))
            return (h_i // ps) * (w_i // ps)
        if self.backbone_type == "vggt":
            return (h_i // 14) * (w_i // 14)
        ps = 14
        return (h_i // ps) * (w_i // ps)

    def _init_dit_action_head(
        self,
        *,
        patch_in_dim,
        action_cfg,
        default_embed_dim,
        log_name,
        state_dim,
    ):
        adim = int(action_cfg.get("action_dim", 10))
        apd = action_cfg.get("action_part_dims", (9, 1))
        # Duck-typed: OmegaConf hands this over as a ListConfig, which is neither
        # list nor tuple -- the old isinstance check silently discarded the
        # configured value and fell back to (9, 1).
        try:
            action_part_dims = tuple(int(x) for x in apd)
        except TypeError:
            action_part_dims = (9, 1)
        dit_e = int(action_cfg.get("embed_dim", default_embed_dim))
        dit_heads = int(action_cfg.get("num_heads", 8))
        dit_layers = int(action_cfg.get("num_layers", 8))
        dit_ps = int(action_cfg.get("patch_size", 14))
        seq_len = int(action_cfg.get("sequence_length", 10))
        n_inf = int(action_cfg.get("num_inference_steps", 10))
        geo = action_cfg.get("geo_mode", "ray_depth")
        geo_embed_init_std = float(action_cfg.get("geo_embed_init_std", 0.0))
        drop = float(action_cfg.get("dropout", 0.1))
        skip_proj = bool(action_cfg.get("skip_patch_kv_proj", False)) and self.use_latent_aux_model
        num_cameras = int(action_cfg.get("num_cameras", 2))
        num_tcp_cameras = num_cameras - len(self.aux_bypass_cameras)
        if num_tcp_cameras < 0:
            raise ValueError("aux_bypass_cameras has more entries than action_cfg.num_cameras")
        if num_tcp_cameras == 0 and not bool(action_cfg.get("disable_tcp_kv", False)):
            raise ValueError("DiT TCP KV requires at least one non-bypass camera")
        num_spatial = self._dit_num_spatial_patches_per_cam(action_cfg)
        num_tcp = (
            int(getattr(self.latent_aux_model, "num_tcp", 1))
            if self.latent_aux_model is not None
            else int(action_cfg.get("num_tcp", 1))
        )

        self.action_head = DiTActionHead(
            patch_in_dim=patch_in_dim,
            action_dim=adim,
            embed_dim=dit_e,
            num_heads=dit_heads,
            num_layers=dit_layers,
            patch_size=dit_ps,
            use_depth=self.use_depth,
            use_camera_intrinsics=self.use_camera_intrinsics,
            geo_mode=geo,
            geo_embed_init_std=geo_embed_init_std,
            dropout=drop,
            sequence_length=seq_len,
            num_inference_steps=n_inf,
            action_part_dims=action_part_dims,
            use_tcp_valid=bool(action_cfg.get("use_tcp_valid", True)),
            skip_patch_kv_proj=skip_proj,
            disable_tcp_kv=bool(action_cfg.get("disable_tcp_kv", False)),
            disable_tcp_additive_bias=bool(action_cfg.get("disable_tcp_additive_bias", False)),
            use_state=self.use_state,
            state_dim=state_dim,
            obs_hist_length=int(action_cfg.get("obs_hist_length", 1)),
            full_conditional=bool(action_cfg.get("full_conditional", False)),
            num_cameras=num_cameras,
            num_tcp_cameras=num_tcp_cameras,
            num_tcp=num_tcp,
            num_spatial_patches_per_cam=num_spatial,
            use_last_frame_visual=bool(action_cfg.get("use_last_frame_visual", False)),
            num_diffusion_draws=int(action_cfg.get("num_diffusion_draws", 1)),
            use_rgb_kv_norm=bool(action_cfg.get("use_rgb_kv_norm", True)),
        )
        self._dit_use_last_frame_visual = bool(action_cfg.get("use_last_frame_visual", False))
        if self.compile_action_head:
            self.action_head.compile_step = True
            self.action_head.compile_mode = self.compile_mode
        self._dit_tcp_kv_curriculum = bool(action_cfg.get("tcp_kv_curriculum", False))
        self._dit_tcp_curriculum_start_step = int(action_cfg.get("tcp_kv_curriculum_start_step", 0))
        end_step = action_cfg.get("tcp_kv_curriculum_end_step")
        self._dit_tcp_curriculum_end_step = int(end_step) if end_step is not None else None
        self._dit_tcp_curriculum_step = 0
        if self._dit_tcp_kv_curriculum:
            if self._dit_tcp_curriculum_end_step is None:
                raise ValueError(
                    "action_cfg.tcp_kv_curriculum=true requires tcp_kv_curriculum_end_step "
                    "(global training step when TCP KV becomes 100% pred)."
                )
            if self._dit_tcp_curriculum_end_step <= self._dit_tcp_curriculum_start_step:
                raise ValueError(
                    "tcp_kv_curriculum_end_step must be > tcp_kv_curriculum_start_step "
                    f"(got end={self._dit_tcp_curriculum_end_step}, "
                    f"start={self._dit_tcp_curriculum_start_step})."
                )
        Log.info(
            f"[{log_name}][DiTActionHead] action_dim={adim}, action_part_dims={action_part_dims}, embed_dim={dit_e}, "
            f"layers={dit_layers}, heads={dit_heads}, seq_len={seq_len}, infer_steps={n_inf}, "
            f"visual_cameras={num_cameras}, tcp_cameras={num_tcp_cameras}, "
            f"num_tcp={num_tcp}, "
            f"disable_tcp_kv={bool(action_cfg.get('disable_tcp_kv', False))}, "
            f"use_last_frame_visual={bool(action_cfg.get('use_last_frame_visual', False))}, "
            f"tcp_kv_curriculum={self._dit_tcp_kv_curriculum}, "
            f"num_diffusion_draws={self.action_head.num_diffusion_draws}"
        )

    def _init_fm_action_head(
        self,
        *,
        patch_in_dim,
        action_cfg,
        default_embed_dim,
        log_name,
        state_dim,
    ):
        """Instantiate a transformer-based flow-matching or diffusion action head."""
        cls_name = action_cfg.get("action_head_class")
        fm_heads = {
            "TransformerFlowMatchingHead": TransformerFlowMatchingHead,
            "TransformerDiffusionHead": TransformerDiffusionHead,
        }
        HeadClass = fm_heads.get(cls_name)
        if HeadClass is None:
            raise ValueError(
                f"Unknown action_head_class '{cls_name}'. Available: {list(fm_heads)}"
            )

        embed_dim = int(action_cfg.get("embed_dim", default_embed_dim))
        num_cameras = int(action_cfg.get("num_cameras", 2))

        self.action_head = HeadClass(
            obs_in_channels=patch_in_dim,
            obs_pyramid_channels=int(action_cfg.get("obs_pyramid_channels", patch_in_dim)),
            sequence_length=int(action_cfg.get("sequence_length", 16)),
            obs_hist_length=int(action_cfg.get("obs_hist_length", 1)),
            embed_dim=embed_dim,
            num_heads=int(action_cfg.get("num_heads", 12)),
            num_layers=int(action_cfg.get("num_layers", 12)),
            num_inference_steps=int(action_cfg.get("num_inference_steps", 20)),
            action_orn_mode=str(action_cfg.get("action_orn_mode", "6d")),
            num_cameras=num_cameras,
            use_state=self.use_state,
            state_dim=state_dim,
            encoder_type=str(action_cfg.get("encoder_type", "transformer")),
            fpn_last_scale_only=bool(action_cfg.get("fpn_last_scale_only", True)),
            num_spatial_patches_per_cam=int(action_cfg.get("num_spatial_patches_per_cam", 25)),
            # None keeps the single-arm width implied by action_orn_mode.
            action_dim=action_cfg.get("action_dim", None),
        )
        Log.info(
            f"[{log_name}][{cls_name}] embed_dim={embed_dim}, "
            f"obs_in={patch_in_dim}, cameras={num_cameras}, "
            f"encoder={action_cfg.get('encoder_type', 'transformer')}"
        )

    
    def _separate_cls(self, z: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate CLS token from patch features."""
        _, n, _ = z.shape
        
        if grid_size is not None:
            expected_patches = grid_size[0] * grid_size[1]
            if n == expected_patches + 1:
                return z[:, 1:], z[:, 0]
            if n == expected_patches:
                return z, None

        side = int(sqrt(n))
        if side * side == n - 1:
            return z[:, 1:], z[:, 0]
        if side * side == n:
            return z, None
        
        return z, None
    
    def _reshape_to_2d(self, z: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Reshape (B*V, N, C) -> (B*V, C, H, W)"""
        bv, n, c = z.shape
        
        if grid_size is not None:
            h, w = grid_size
            if n != h * w:
                h = w = int(sqrt(n))
        else:
            h = w = int(sqrt(n))
                 
        return z.transpose(1, 2).reshape(bv, c, h, w)

    @staticmethod
    def _tcp_stacked_per_cam_dict(outputs, key_suffix, camera_names):
        """tcp_* ``[B, num_cam, T, (num_tcp,) D]`` -> per-camera ``[B, T, (num_tcp,) D]`` for DiT."""
        t = outputs[f"{key_suffix}"]
        return {cam: t[:, i] for i, cam in enumerate(camera_names)}

    def set_tcp_kv_curriculum_step(self, step: int) -> None:
        """Update global training step for DiT TCP KV GT→pred curriculum."""
        self._dit_tcp_curriculum_step = int(step)

    def _dit_tcp_kv_pred_prob(self) -> float:
        """Curriculum probability of using pred TCP KV (batch-level Bernoulli switch)."""
        if not getattr(self, "_dit_tcp_kv_curriculum", False):
            return 0.0
        step = int(getattr(self, "_dit_tcp_curriculum_step", 0))
        start = int(self._dit_tcp_curriculum_start_step)
        end = int(self._dit_tcp_curriculum_end_step)
        if step <= start:
            return 0.0
        if step >= end:
            return 1.0
        return float(step - start) / float(end - start)

    @staticmethod
    def _select_tcp_kv_by_curriculum(
        gt_uv: dict,
        gt_3d: dict,
        gt_6d: dict,
        gt_valid: Optional[dict],
        pred_uv: dict,
        pred_3d: dict,
        pred_6d: dict,
        pred_valid: Optional[dict],
        p_pred: float,
        device: torch.device,
    ) -> Tuple[dict, dict, dict, Optional[dict], float]:
        """Batch-level Bernoulli: with prob ``p_pred`` use pred TCP KV, else GT (no interpolation)."""
        if p_pred <= 0.0:
            return gt_uv, gt_3d, gt_6d, gt_valid, 0.0
        if p_pred >= 1.0:
            return pred_uv, pred_3d, pred_6d, pred_valid, 1.0
        use_pred = random.random() < p_pred  # CPU draw; the CUDA .item() was a host sync
        if use_pred:
            return pred_uv, pred_3d, pred_6d, pred_valid, 1.0
        return gt_uv, gt_3d, gt_6d, gt_valid, 0.0

    @staticmethod
    def _extract_gt_tcp_camera_dicts(camera_names, tcp_pos, tcp_orn, t_steps=None, dtype=None):
        """Return per-camera GT TCP pose dicts in camera frame.

        Expects dataset-provided ``observation['tcp_pos']`` (``[B, T, 3]`` or
        ``[B, T, num_tcp, 3]``) and ``observation['tcp_orn']`` (``[..., 9]`` flat rotmat).
        """
        if tcp_pos is None or not isinstance(tcp_pos, dict):
            raise ValueError("Need camera-frame GT TCP dict from observation['tcp_pos'].")
        if tcp_orn is None or not isinstance(tcp_orn, dict):
            raise ValueError("Need camera-frame GT TCP dict from observation['tcp_orn'].")

        # Expect [B, T, D] or [B, T, num_tcp, D] from dataset.
        pos_dict, rot6d_dict = {}, {}
        for cam in camera_names:
            pos = tcp_pos[cam]
            if t_steps is not None:
                pos = pos[:, -t_steps:]
            if dtype is not None:
                pos = pos.to(dtype=dtype)
            pos_dict[cam] = pos

            orn = tcp_orn[cam]
            if t_steps is not None:
                orn = orn[:, -t_steps:]
            orn = orn.reshape(*orn.shape[:-1], 3, 3)
            if dtype is not None:
                orn = orn.to(dtype=dtype)
            rot6d_dict[cam] = matrix_to_rotation_6d(orn)

        return pos_dict, rot6d_dict

    def _da3_out_layer_level(self, layer_id: int) -> int:
        """Map DA3 OUT_LAYERS block id to DA3Encoder level index (-1 = deepest)."""
        from .backbone.da3 import DA3Encoder

        if layer_id not in DA3Encoder.OUT_LAYERS:
            raise ValueError(
                f"DA3 backbone_feat_layers entry {layer_id} not in OUT_LAYERS={DA3Encoder.OUT_LAYERS}"
            )
        level_idx = DA3Encoder.OUT_LAYERS.index(layer_id)
        return level_idx - len(DA3Encoder.OUT_LAYERS)

    def _da3_encode_tokens(self, rgb, grad_context):
        """Run DA3 encoder; optionally channel-concat multiple OUT_LAYERS (DINO-style).

        Returns:
            tok: [B*S, N, D] where D=1536 (last) or 1536*K (concat).
        """
        from .backbone.da3 import DA3Encoder

        if self.backbone_feat_mode == "concat":
            layer_ids = list(self.intermediate_layer_idx.get(self.encoder, DA3Encoder.OUT_LAYERS))
            with grad_context():
                all_levels = self.pretrained.forward(rgb, mode="all")  # level_idx -> [B*S, N, 1536]
            selected = []
            for lid in layer_ids:
                level_idx = DA3Encoder.OUT_LAYERS.index(lid)
                selected.append(all_levels[level_idx])
            return torch.cat(selected, dim=-1)

        layer_ids = list(self.intermediate_layer_idx.get(self.encoder, DA3Encoder.OUT_LAYERS))
        last_level = self._da3_out_layer_level(layer_ids[-1])
        with grad_context():
            return self.pretrained.forward(rgb, mode="single", level=last_level)

    def _vggt_feature_levels(self):
        """Aggregator output indices for VGGT; None → single last level."""
        if self.backbone_feat_mode == "concat":
            return list(self.intermediate_layer_idx.get(self.encoder, [4, 11, 17, 23]))
        return None

    def _obtain_da3_visual_tokens(self, rgb_per_cam, camera_names, b, t_steps):
        """Build DA3 per-camera visual tokens from preprocessed per-camera RGB tensors.

        DA3 is built on DINOv2 backbone weights — expects ImageNet-normalized
        inputs like the DINOv2 paths do. `_prepare_per_camera_backbone_inputs`
        returns raw [0,1] RGB and delegates per-backbone normalization here.

        With ``backbone_feat_mode='concat'``, channel-concats selected OUT_LAYERS
        (default ``[5, 7, 9, 11]``) like DINOv2 multilayer concat.
        """
        # ImageNet normalization (DINOv2/DA3 was pretrained with it).
        rgb_per_cam = [(rgb - self._mean) / self._std for rgb in rgb_per_cam]

        visual_tokens = {}
        num_cam = len(camera_names)

        grad_enabled = any(param.requires_grad for param in self.pretrained.backbone.parameters())
        grad_context = contextlib.nullcontext if grad_enabled else torch.no_grad

        bt = b * t_steps
        if self.temporal_fuse and t_steps > 1:
            # Temporal fuse: feed T_obs frames as the S sequence dim per camera so DA3
            # cross-attention fuses information across time. Each camera independent.
            rgb_tokens_per_cam = []
            ch, ih, iw = rgb_per_cam[0].shape[-3:]
            for rgb in rgb_per_cam:
                rgb_temporal = rgb.view(b, t_steps, ch, ih, iw)  # [B, T_obs, 3, H, W] chronological [t-T+1, ..., t]
                rgb_temporal = rgb_temporal.flip(dims=[1])  # [t, t-1, ..., t-T+1]
                tok = self._da3_encode_tokens(rgb_temporal, grad_context)  # [B*T_obs, N, D]
                tok = tok.view(b, t_steps, tok.shape[1], tok.shape[2])
                tok = tok.flip(dims=[1])
                tok = tok.reshape(b * t_steps, tok.shape[2], tok.shape[3])
                rgb_tokens_per_cam.append(tok)
            rgb_tokens = torch.stack(rgb_tokens_per_cam, dim=1)  # [BT, num_cam, N, D]
        elif self.cross_view_fuse:
            # [BT, num_cam, 3, H, W]: DA3 fuses across views in one pass.
            rgb_stacked = torch.stack(rgb_per_cam, dim=1)
            rgb_tokens = self._da3_encode_tokens(rgb_stacked, grad_context)
            if rgb_tokens.ndim == 4:  # [B, V, N, D] etc.
                b_out, v_out, n, c_out = rgb_tokens.shape
                rgb_tokens = rgb_tokens.reshape(b_out * v_out, n, c_out)
            # [BT*num_cam, N, D] -> [BT, num_cam, N, D]
            rgb_tokens = rgb_tokens.view(bt, num_cam, rgb_tokens.shape[1], rgb_tokens.shape[2])
        else:
            # No cross-view fusion: run each camera independently through DA3 backbone.
            rgb_tokens_per_cam = []
            for rgb in rgb_per_cam:
                tok = self._da3_encode_tokens(rgb, grad_context)  # [BT,N,D]
                if tok.ndim == 4:  # conservative reshape if backend returns [BT,1,N,D]
                    bt_out, v_out, n, c_out = tok.shape
                    tok = tok.reshape(bt_out * v_out, n, c_out)
                rgb_tokens_per_cam.append(tok)
            rgb_tokens = torch.stack(rgb_tokens_per_cam, dim=1)  # [BT, num_cam, N, D]

        h_i, w_i = rgb_per_cam[0].shape[-2:]
        h_p, w_p = h_i // 14, w_i // 14
        for cam_idx, camera_name in enumerate(camera_names):
            tok = rgb_tokens[:, cam_idx].contiguous()
            tok, cls_tok = self._separate_cls(tok, grid_size=(h_p, w_p))
            tok = self._reshape_to_2d(tok, grid_size=(h_p, w_p))
            tok = tok.reshape(b, t_steps, *tok.shape[1:])
            tok = tok.permute(0, 2, 1, 3, 4)
            visual_tokens[camera_name] = tok

        return visual_tokens, cls_tok, h_p, w_p

    def _obtain_vggt_visual_tokens(self, rgb_per_cam, camera_names, b, t_steps):
        """Build VGGT per-camera visual tokens from independent single-view forwards.

        Unlike the original multi-view fused VGGT path, this treats each camera the
        same way as DINO/DA3(fuse=false): each view is encoded independently,
        while reusing the same frozen VGGT backbone weights for all cameras.

        Args:
            rgb_per_cam: list of [BT, 3, H, W] raw [0, 1] RGB tensors (NOT ImageNet
                normalized — VGGT normalizes internally).
            camera_names: list of camera name strings
            b: batch size
            t_steps: number of timesteps

        Returns:
            visual_tokens: dict camera_name -> [B, C, T, h_p, w_p]
            h_p, w_p: patch grid dimensions
        """
        visual_tokens = {}
        h_i, w_i = rgb_per_cam[0].shape[-2:]
        h_p, w_p = h_i // 14, w_i // 14

        if self.temporal_fuse and self.cross_view_fuse and t_steps > 1:
            # Combined fuse: feed all S cameras × T_obs frames in a single VGGT forward.
            # VGGT does cross-view + cross-time attention together; output is split back per cam.
            ch, ih, iw = rgb_per_cam[0].shape[-3:]
            S = len(rgb_per_cam)
            per_cam_BTCHW = [r.view(b, t_steps, ch, ih, iw) for r in rgb_per_cam]
            rgb_st = torch.stack(per_cam_BTCHW, dim=1)  # [B, S, T, 3, H, W]
            rgb_st = rgb_st.flip(dims=[2])
            rgb_flat = rgb_st.permute(0, 2, 1, 3, 4, 5).contiguous().view(b, t_steps * S, ch, ih, iw)
            patch_tokens, _ = self.pretrained(rgb_flat, feature_levels=self._vggt_feature_levels())
            tok_tsn = patch_tokens.view(b, t_steps, S, patch_tokens.shape[-2], patch_tokens.shape[-1])
            tok_tsn = tok_tsn.flip(dims=[1])  # [B, T(chrono), S, N, D]
            for cam_idx, camera_name in enumerate(camera_names):
                tok = tok_tsn[:, :, cam_idx]  # [B, T, N, D]
                flat = tok.reshape(b * t_steps, tok.shape[-2], tok.shape[-1])
                flat_2d = self._reshape_to_2d(flat, grid_size=(h_p, w_p))
                flat_2d = flat_2d.reshape(b, t_steps, *flat_2d.shape[1:])
                flat_2d = flat_2d.permute(0, 2, 1, 3, 4)
                visual_tokens[camera_name] = flat_2d
        elif self.temporal_fuse and t_steps > 1:
            # Temporal fuse: VGGT aggregator's alternating frame/global attention across S
            # operates on T_obs frames per camera. Each camera independent.
            ch, ih, iw = rgb_per_cam[0].shape[-3:]
            vggt_levels = self._vggt_feature_levels()
            for cam_idx, camera_name in enumerate(camera_names):
                rgb_temporal = rgb_per_cam[cam_idx].view(b, t_steps, ch, ih, iw)  # [B, T_obs, 3, H, W] chronological [t-T+1, ..., t]
                rgb_temporal = rgb_temporal.flip(dims=[1])  # [t, t-1, ..., t-T+1]
                patch_tokens, _ = self.pretrained(rgb_temporal, feature_levels=vggt_levels)  # [B, T_obs, N_patches, D]
                patch_tokens = patch_tokens.flip(dims=[1])  # back to [t-T+1, ..., t]
                tok = patch_tokens.reshape(b * t_steps, patch_tokens.shape[2], patch_tokens.shape[3])
                tok = self._reshape_to_2d(tok, grid_size=(h_p, w_p))
                tok = tok.reshape(b, t_steps, *tok.shape[1:])
                tok = tok.permute(0, 2, 1, 3, 4)
                visual_tokens[camera_name] = tok
        elif self.cross_view_fuse:
            rgb_stacked = torch.stack(rgb_per_cam, dim=1)  # [BT, S, 3, H, W]
            # VGGT aggregator: cross-view fusion via alternating frame/global attention
            patch_tokens, _ = self.pretrained(
                rgb_stacked, feature_levels=self._vggt_feature_levels()
            )  # [BT, S, N_patches, D]
            for cam_idx, camera_name in enumerate(camera_names):
                tok = patch_tokens[:, cam_idx].contiguous()
                tok = self._reshape_to_2d(tok, grid_size=(h_p, w_p))
                tok = tok.reshape(b, t_steps, *tok.shape[1:])
                tok = tok.permute(0, 2, 1, 3, 4)
                visual_tokens[camera_name] = tok
        else:
            vggt_levels = self._vggt_feature_levels()
            for cam_idx, camera_name in enumerate(camera_names):
                patch_tokens, _ = self.pretrained(
                    rgb_per_cam[cam_idx], feature_levels=vggt_levels
                )
                tok = patch_tokens[:, 0].contiguous()  # [BT, N_patches, D]
                tok = self._reshape_to_2d(tok, grid_size=(h_p, w_p))
                tok = tok.reshape(b, t_steps, *tok.shape[1:])
                tok = tok.permute(0, 2, 1, 3, 4)
                visual_tokens[camera_name] = tok

        return visual_tokens, h_p, w_p

    def _aligned_crop(self, cam_rgb, depth, b, t_steps):
        """Apply self.randomizer with one crop offset shared across all T_obs frames
        of the same trajectory.

        Required for temporal_fuse=True correctness (cross-frame attention assumes
        patch (i,j) refers to the same image region across time). Harmless for
        temporal_fuse=False — enforces temporal-coherent augmentation, which is the
        expected behavior for sequence models even without explicit fusion.

        Args:
            cam_rgb: [B*T, 3, H, W] flattened over batch and time.
            depth: [B*T, C_d, H, W] or None.
            b: trajectory batch size.
            t_steps: T_obs.

        Returns:
            cam_rgb_out: [B*T, 3, Hc, Wc]
            depth_out: [B*T, C_d, Hc, Wc] or None
            offsets_bt: [B*T, 2] (top, left), each row replicated across the T_obs
                frames of the same trajectory.
            h_pre, w_pre: source spatial size before crop.
        """
        assert getattr(self.randomizer, "num_crops", 1) == 1, (
            "_aligned_crop assumes num_crops=1; multi-crop would change leading dim."
        )
        h_pre, w_pre = cam_rgb.shape[-2:]
        if depth is not None:
            combined = torch.cat([cam_rgb, depth], dim=1)
        else:
            combined = cam_rgb
        total_c = combined.shape[1]
        grouped = combined.reshape(b, t_steps * total_c, h_pre, w_pre)
        cropped = self.randomizer(grouped)
        hc, wc = cropped.shape[-2:]
        cropped = cropped.reshape(b * t_steps, total_c, hc, wc)
        if depth is not None:
            cam_rgb_out, depth_out = cropped[:, :3], cropped[:, 3:]
        else:
            cam_rgb_out, depth_out = cropped, None
        offset_b = self.randomizer.last_crop_top_left  # [B, 2]
        offsets_bt = offset_b.unsqueeze(1).expand(-1, t_steps, -1).reshape(
            b * t_steps, 2
        )
        return cam_rgb_out, depth_out, offsets_bt, h_pre, w_pre

    def _prepare_per_camera_backbone_inputs(self, rgb, depths, camera_names, b, t_steps):
        """Reshape RGB and apply synchronized random crop to RGB/depth per camera.

        Returns raw [0, 1] RGB — per-backbone normalization (ImageNet mean/std for
        DA3/DINOv2, internal for VGGT) is applied inside each
        `_obtain_*_visual_tokens` method. Depth is carried through so the TCP head
        can receive a crop-aligned depth tensor.
        """
        bt = b * t_steps
        rgb_per_cam = []
        depth_per_cam = []
        crop_offsets = {}
        ref_hw = None
        need_depth = self.use_depth and depths is not None

        for camera_name in camera_names:
            cam_rgb = rgb[camera_name].reshape(bt, *rgb[camera_name].shape[2:])
            # Wrist cameras stay RGB-only; ray_depth geo is third-person only.
            use_cam_depth = need_depth and "eye_in_hand" not in camera_name

            if use_cam_depth:
                d = depths.get(camera_name)
                if d is None:
                    raise ValueError(f"Depth is not found for camera: {camera_name}")
                d = d.reshape(bt, *d.shape[2:])
                # Eval may hand in [BT,1,1,H,W] (PrepareForNet CHW + extra
                # unsqueeze) or [BT,H,W]; crop/interpolate need [BT,C,H,W].
                if d.ndim == 5 and d.shape[2] == 1:
                    d = d.squeeze(2)
                elif d.ndim == 3:
                    d = d.unsqueeze(1)
                if d.ndim != 4:
                    raise ValueError(
                        f"depth must be [BT,C,H,W] before crop, got {tuple(d.shape)}"
                    )
                if d.shape[-2:] != cam_rgb.shape[-2:]:
                    # Closed-loop RGB may already be training-resized (168x126)
                    # while RoboTwin depth is still native (322x238).
                    d = F.interpolate(
                        d.float(),
                        size=cam_rgb.shape[-2:],
                        mode="nearest",
                    ).to(dtype=cam_rgb.dtype)
                if self.randomizer is not None:
                    cam_rgb, d, offsets_bt, h_pre, w_pre = self._aligned_crop(
                        cam_rgb, d, b, t_steps
                    )
                    crop_offsets[camera_name] = (offsets_bt, h_pre, w_pre)
                depth_per_cam.append(d)
            else:
                if self.randomizer is not None:
                    cam_rgb, _, offsets_bt, h_pre, w_pre = self._aligned_crop(
                        cam_rgb, None, b, t_steps
                    )
                    crop_offsets[camera_name] = (offsets_bt, h_pre, w_pre)

            if ref_hw is None:
                ref_hw = cam_rgb.shape[-2:]
            elif cam_rgb.shape[-2:] != ref_hw:
                raise ValueError(
                    "Multi-view forward requires identical H,W across cameras after preprocessing; "
                    f"got {ref_hw} vs {tuple(cam_rgb.shape[-2:])} for {camera_name}"
                )
            rgb_per_cam.append(cam_rgb)

        return rgb_per_cam, depth_per_cam, crop_offsets

    @staticmethod
    def _build_depth_inputs_from_crops(depth_per_cam, camera_names, b, t_steps):
        depth_inputs = {}
        depth_cams = [cn for cn in camera_names if "eye_in_hand" not in cn]
        if len(depth_per_cam) != len(depth_cams):
            raise ValueError(
                f"depth crops ({len(depth_per_cam)}) != non-wrist cameras ({len(depth_cams)})"
            )
        for cam_idx, camera_name in enumerate(depth_cams):
            d = depth_per_cam[cam_idx]
            d = d.reshape(b, t_steps, *d.shape[1:])
            d = d.permute(0, 2, 1, 3, 4).contiguous()
            depth_inputs[camera_name] = d
        return depth_inputs

    def _build_intrinsics_inputs(self, camera_intrinsics, camera_names, t_steps, crop_offsets=None):
        intrinsics_inputs = {}
        if not self.use_camera_intrinsics or camera_intrinsics is None:
            return intrinsics_inputs

        for camera_name in camera_names:
            K = camera_intrinsics.get(camera_name)
            if K is None:
                continue
            if K.ndim == 3:
                K = K.unsqueeze(1).expand(-1, t_steps, -1, -1)
            fx = K[..., 0, 0]
            fy = K[..., 1, 1]
            cx = K[..., 0, 2]
            cy = K[..., 1, 2]
            if crop_offsets and camera_name in crop_offsets:
                offsets_bt = crop_offsets[camera_name][0]
                bsz = K.shape[0]
                cx = cx.clone() - offsets_bt[:, 1].float().reshape(bsz, t_steps)
                cy = cy.clone() - offsets_bt[:, 0].float().reshape(bsz, t_steps)
            intrinsics_inputs[camera_name] = torch.stack([fx, fy, cx, cy], dim=-1)
        return intrinsics_inputs

    @staticmethod
    def _format_tcp_valid_mask(source_valid, camera_name, b, t_steps, device):
        if not isinstance(source_valid, dict) or camera_name not in source_valid:
            return None
        valid = source_valid[camera_name]
        if valid is None:
            return None
        if torch.is_tensor(valid):
            valid = valid.to(device=device)
        else:
            valid = torch.as_tensor(valid, device=device)
        if valid.ndim == 3 and valid.shape[-1] == 1:
            valid = valid[..., 0]
        elif valid.ndim == 1:
            if b == 1:
                valid = valid.unsqueeze(0)
            else:
                valid = valid.reshape(b, -1)
        elif valid.ndim > 2:
            valid = valid.reshape(valid.shape[0], -1)
        if valid.ndim != 2 or valid.shape[0] != b or valid.shape[1] < t_steps:
            raise ValueError(
                f"tcp_valid[{camera_name!r}] must be [B, T] or [B, T, 1] "
                f"with B={b}, T>={t_steps}; got {tuple(valid.shape)}"
            )
        valid = valid[:, -t_steps:]
        if valid.dtype == torch.bool:
            return valid
        if valid.is_floating_point():
            return valid > 0.5
        return valid.bool()

    def _adjust_tcp_pixel_coords_for_crop(self, tcp_pixel_coords, camera_names, crop_offsets, b, t_steps, source_valid=None):
        tcp_valid_inputs = {}
        if tcp_pixel_coords is None:
            return tcp_valid_inputs

        bt = b * t_steps
        crop_h = self.randomizer.crop_height if crop_offsets else None
        crop_w = self.randomizer.crop_width if crop_offsets else None
        for camera_name in camera_names:
            uv = tcp_pixel_coords.get(camera_name, None)
            if uv is None:
                continue
            if uv.ndim == 2:
                uv = uv.unsqueeze(1)
            # bimanual tensors are [B, T, num_tcp, 3], and the
            # positional `[:, :, :2]` would slice the num_tcp axis there -- folding
            # the depth channel into the validity test and making the crop reshape
            uv_xy = uv[..., :2]
            raw_valid = ((uv_xy >= 0) & (uv_xy <= 1)).all(dim=-1) # For consistency, should already contained in source_valid_mask
            source_valid_mask = self._format_tcp_valid_mask(
                source_valid,
                camera_name,
                b,
                t_steps,
                uv_xy.device,
            )
            if source_valid_mask is not None:
                if raw_valid.ndim == source_valid_mask.ndim + 1:
                    source_valid_mask = source_valid_mask.unsqueeze(-1)
                raw_valid = raw_valid & source_valid_mask
            if crop_offsets and camera_name in crop_offsets:
                offsets_bt, h_pre, w_pre = crop_offsets[camera_name]
                top = offsets_bt[:, 0].float()
                left = offsets_bt[:, 1].float()
                uv_flat = uv_xy.reshape(bt, -1, 2).clone()  # [bt, num_tcp or 1, 2]
                uv_flat[..., 0] = (uv_flat[..., 0] * (w_pre - 1) - left[:, None]) / max(crop_w - 1, 1)
                uv_flat[..., 1] = (uv_flat[..., 1] * (h_pre - 1) - top[:, None]) / max(crop_h - 1, 1)
                uv_xy = uv_flat.reshape(uv.shape[:-1] + (2,))
                tcp_pixel_coords[camera_name] = torch.cat([uv_xy, uv[..., 2:]], dim=-1)
            crop_valid = ((uv_xy >= 0) & (uv_xy <= 1)).all(dim=-1)
            tcp_valid_inputs[camera_name] = raw_valid & crop_valid
        return tcp_valid_inputs

    def _adjust_tcp_pixel_coords_for_crop_target(
        self,
        tcp_pixel_coords,
        camera_names,
        crop_offsets,
        b,
        obs_t_steps,
        source_valid=None,
    ):
        adjusted_targets = {}
        tcp_valid_inputs = {}
        if tcp_pixel_coords is None:
            return adjusted_targets, tcp_valid_inputs

        crop_h = self.randomizer.crop_height if crop_offsets else None
        crop_w = self.randomizer.crop_width if crop_offsets else None
        for camera_name in camera_names:
            uv = tcp_pixel_coords[camera_name]  # [B, 1, 3] or [B, 1, num_tcp, 3]
            cam_target_t = uv.shape[1]

            # Same ellipsis discipline as _adjust_tcp_pixel_coords_for_crop: keep an
            # optional num_tcp axis intact and slice channels on the LAST dim only.
            uv_xy = uv[..., :2]
            raw_valid = ((uv_xy >= 0) & (uv_xy <= 1)).all(dim=-1)
            source_valid_mask = self._format_tcp_valid_mask(
                source_valid,
                camera_name,
                b,
                cam_target_t,
                uv_xy.device,
            )
            if source_valid_mask is not None:
                if raw_valid.ndim == source_valid_mask.ndim + 1:
                    source_valid_mask = source_valid_mask.unsqueeze(-1)  # broadcast over arms
                raw_valid = raw_valid & source_valid_mask
            if crop_offsets and camera_name in crop_offsets:
                offsets_bt, h_pre, w_pre = crop_offsets[camera_name]
                offsets_bt = offsets_bt.to(device=uv.device)
                offsets_b = offsets_bt.reshape(b, obs_t_steps, 2)[:, -1]
                offsets_target = offsets_b.unsqueeze(1).expand(-1, cam_target_t, -1)
                offsets_target = offsets_target.reshape(b * cam_target_t, 2)
                top = offsets_target[:, 0].float()
                left = offsets_target[:, 1].float()
                uv_flat = uv_xy.reshape(b * cam_target_t, -1, 2).clone()
                uv_flat[..., 0] = (uv_flat[..., 0] * (w_pre - 1) - left[:, None]) / max(crop_w - 1, 1)
                uv_flat[..., 1] = (uv_flat[..., 1] * (h_pre - 1) - top[:, None]) / max(crop_h - 1, 1)
                uv_xy = uv_flat.reshape(uv.shape[:-1] + (2,))

            adjusted_targets[camera_name] = torch.cat([uv_xy, uv[..., 2:]], dim=-1)
            crop_valid = ((uv_xy >= 0) & (uv_xy <= 1)).all(dim=-1)
            tcp_valid_inputs[camera_name] = raw_valid & crop_valid
        return adjusted_targets, tcp_valid_inputs

    def _compiled_fn(self, key, fn):
        """Lazily torch.compile a bound method, cached per module instance.

        Kept outside the module tree on purpose: state_dict keys stay unchanged and a
        deepcopy (EMA pipeline) compiles its own copy instead of inheriting a callable
        bound to the original module's weights.
        """
        k = (id(self), key)
        if k not in _COMPILED_FNS:
            _COMPILED_FNS[k] = torch.compile(fn, dynamic=False, mode=self.compile_mode)
        return _COMPILED_FNS[k]

    def _frames_to_encode(self, camera_names, t_obs):
        """Trailing observation frames the backbone must encode, per camera.

        With ``encode_only_consumed_frames`` a camera keeps all ``t_obs`` frames only if the
        latent-aux (TCP) model reads it; cameras that feed the action head alone are encoded
        at the last frame only, because ``DiTActionHead(use_last_frame_visual=True)`` discards
        the others (``_build_patch_kv`` slices ``[:, :, -1:]``). Same outputs, fewer backbone
        forwards (3-camera lowtcp: 9 -> 5 per sample; notcp: 9 -> 3).
        """
        n = len(camera_names)
        if not self.encode_only_consumed_frames or t_obs <= 1:
            return [t_obs] * n
        if getattr(self, "_action_head_type", None) != "DiTActionHead" or not getattr(self, "_dit_use_last_frame_visual", False):
            raise ValueError("encode_only_consumed_frames requires DiTActionHead with action_cfg.use_last_frame_visual=true")
        if getattr(self, "use_dynamics_head", False) or getattr(self, "action_consistency_weight", 0.0) > 0:
            raise ValueError("encode_only_consumed_frames is incompatible with the dynamics head / action-consistency loss")
        return [
            t_obs if (self.use_latent_aux_model and cam not in self.aux_bypass_cameras) else 1
            for cam in camera_names
        ]

    def _obtain_dinov2_visual_tokens(self, rgb_per_cam, camera_names, b, t_steps):
        """Build DINOv2/DINOv3 per-camera visual tokens from RGB only."""
        # ImageNet normalization used by DINOv2 and DINOv3.
        rgb_per_cam = [(rgb - self._mean) / self._std for rgb in rgb_per_cam]

        camera_features = {}
        # t_steps: int (all cameras) or per-camera list (encode_only_consumed_frames).
        t_list = list(t_steps) if isinstance(t_steps, (list, tuple)) else [int(t_steps)] * len(camera_names)
        row_offsets = [0]
        for t in t_list:
            row_offsets.append(row_offsets[-1] + b * t)
        patch_size = int(getattr(self.pretrained, "patch_size", 14))
        image_h, image_w = rgb_per_cam[0].shape[-2:]
        if image_h % patch_size != 0 or image_w % patch_size != 0:
            raise ValueError(
                f"{self.backbone_type} received {image_h}x{image_w} input, "
                f"which is not divisible by patch size {patch_size}."
            )
        patch_h, patch_w = image_h // patch_size, image_w // patch_size

        batched_rgb = torch.cat(rgb_per_cam, dim=0)
        layer_idx = self.intermediate_layer_idx.get(self.encoder, None)
        layers_fn = (
            self._compiled_fn("backbone_layers", self.pretrained.get_intermediate_layers)
            if self.compile_backbone else self.pretrained.get_intermediate_layers
        )
        batched_features_rgb = layers_fn(
            batched_rgb,
            n=layer_idx,
            return_class_token=True,
        )

        for cam_idx, camera_name in enumerate(camera_names):
            features = []
            r0, r1, t_cam = row_offsets[cam_idx], row_offsets[cam_idx + 1], t_list[cam_idx]
            for feat, cls_t in batched_features_rgb:
                cam_feat = feat[r0:r1]
                cam_feat = cam_feat.reshape(b, t_cam, *cam_feat.shape[1:])
                cam_cls_t = cls_t[r0:r1]
                cam_cls_t = cam_cls_t.reshape(b, t_cam, *cam_cls_t.shape[1:])
                features.append((cam_feat, cam_cls_t))
            camera_features[camera_name] = features

        visual_tokens = {}
        for cam_idx, camera_name in enumerate(camera_names):
            t_cam = t_list[cam_idx]
            if self.backbone_feat_mode == "concat":
                layer_feats = [camera_features[camera_name][i][0] for i in range(len(camera_features[camera_name]))]
                feat = torch.cat(layer_feats, dim=-1)  # [B, T, N, C * num_layers]
            else:
                feat = camera_features[camera_name][-1][0]  # [B, T, N, C]
            c_feat = feat.shape[-1]
            feat_spatial = feat.reshape(b, t_cam, patch_h, patch_w, c_feat)
            feat_spatial = feat_spatial.permute(0, 4, 1, 2, 3).contiguous()
            visual_tokens[camera_name] = feat_spatial

        return camera_features, visual_tokens, patch_h, patch_w

    def _compute_oracle_tcp(self, state, camera_extrinsics, camera_intrinsics, camera_names, B, T, image_hw=None):
        """Compute GT TCP (uv, 3d, 6d) from env state + camera extrinsics for oracle eval.

        Args:
            state: [B, T, 7+] — first 3 = eef_pos (world), next 4 = eef_quat (world, wxyz)
            camera_extrinsics: dict cam -> [B, T, 4, 4] world-to-camera transform
            camera_intrinsics: dict cam -> [B, T, 4] (fx, fy, cx, cy) normalized to [0,1]
            camera_names: list of camera names
            B, T: batch size and obs timesteps
            image_hw: optional current preprocessed image size ``(H, W)`` used to
                normalize projected pixel coordinates to ``[0, 1]``.

        Returns:
            oracle_uv, oracle_3d, oracle_6d: dicts cam -> [B, T, D]
        """
        from scipy.spatial.transform import Rotation as Rot
        device = state.device
        dtype = state.dtype

        # state: [B, T, D] where D >= 7; first 3 = pos_world, 3:7 = quat (xyzw for scipy)
        if state.ndim == 2:
            state = state.unsqueeze(1).expand(-1, T, -1)
        eef_pos_world = state[:, :, :3]  # [B, T, 3]
        eef_quat = state[:, :, 3:7]  # [B, T, 4]

        oracle_uv, oracle_3d, oracle_6d = {}, {}, {}

        for cam in camera_names:
            # Skip wrist cameras — their extrinsics are static from env init
            # but the camera moves each step, so oracle would be wrong.
            if "eye_in_hand" in cam:
                continue
            ext = camera_extrinsics.get(cam)
            if ext is None:
                continue

            # ext: [B, T, 4, 4] or [B, 4, 4] — camera-to-world transform (from robosuite)
            if ext.ndim == 3:
                ext = ext.unsqueeze(1).expand(-1, T, -1, -1)

            # Camera-to-world: ext = [R_c2w | t_c2w; 0 0 0 1]
            # World-to-camera: inv(ext)
            # For 4x4 SE(3): inv = [R^T | -R^T @ t; 0 0 0 1]
            R_c2w = ext[:, :, :3, :3]  # [B, T, 3, 3]
            t_c2w = ext[:, :, :3, 3]   # [B, T, 3] — camera position in world
            R_w2c = R_c2w.transpose(-1, -2)  # [B, T, 3, 3]

            # TCP position in camera frame: p_cam = R_w2c @ (p_world - t_c2w)
            # Note: extrinsics from eval runner already have OpenGL→CV correction applied
            tcp_cam = torch.einsum('btij,btj->bti', R_w2c, eef_pos_world - t_c2w)  # [B, T, 3]
            oracle_3d[cam] = tcp_cam

            # UV projection: u_px = fx * X/Z + cx, v_px = fy * Y/Z + cy
            # intrinsics are in pixel coords (after crop); normalize by crop size
            K = camera_intrinsics.get(cam)
            if K is not None:
                if K.ndim == 2:
                    K = K.unsqueeze(1).expand(-1, T, -1)
                fx, fy, cx, cy = K[:, :, 0], K[:, :, 1], K[:, :, 2], K[:, :, 3]  # each [B, T]
                Z = tcp_cam[:, :, 2].clamp(min=1e-4)
                u_px = fx * tcp_cam[:, :, 0] / Z + cx
                v_px = fy * tcp_cam[:, :, 1] / Z + cy
                # Normalize with the same convention as dataset tcp_pixel_coords:
                # u /= (W - 1), v /= (H - 1), using the actual preprocessed image size.
                if image_hw is not None:
                    image_h, image_w = image_hw
                elif self.randomizer is not None:
                    image_h, image_w = self.randomizer.crop_height, self.randomizer.crop_width
                else:
                    image_h = image_w = 84
                u_norm = u_px / max(image_w - 1, 1)
                v_norm = v_px / max(image_h - 1, 1)
                oracle_uv[cam] = torch.stack([u_norm, v_norm], dim=-1).clamp(0, 1)  # [B, T, 2]
            else:
                oracle_uv[cam] = torch.zeros(B, T, 2, device=device, dtype=dtype)

            # 6D rotation: transform eef rotation from world to camera frame
            # eef_quat is [B, T, 4] in some convention; convert to rotation matrix
            # then apply R_w2c to get camera-frame rotation
            eef_q_np = eef_quat.float().cpu().numpy().reshape(-1, 4)
            # robosuite uses (x,y,z,w) convention
            try:
                eef_rot_world = torch.tensor(
                    Rot.from_quat(eef_q_np).as_matrix(),
                    device=device, dtype=dtype
                ).reshape(B, T, 3, 3)
            except Exception:
                eef_rot_world = torch.eye(3, device=device, dtype=dtype).expand(B, T, 3, 3)

            rot_cam = torch.einsum('btij,btjk->btik', R_w2c, eef_rot_world)  # [B, T, 3, 3]
            # Convert to 6D: first two columns
            rot6d = torch.cat([rot_cam[:, :, :, 0], rot_cam[:, :, :, 1]], dim=-1)  # [B, T, 6]
            oracle_6d[cam] = rot6d

        return oracle_uv, oracle_3d, oracle_6d

    def _build_dit_gt_tcp_dicts(self, camera_names, b, t_steps, tcp_pos, tcp_orn, tcp_pixel_coords, tcp_valid=None):
        """GT TCP dicts for DiTActionHead.forward_training (per-camera ``[B, T, *]`` or ``[B, T, num_tcp, *]``).

        Args:
            camera_names: keys for each output dict.
            b: batch size ``B``.
            t_steps: observation frames ``T``.
            tcp_pos: GT TCP position in camera frame from dataset ``observation['tcp_pos']``.
            tcp_orn: GT TCP rotation matrix in camera frame from dataset ``observation['tcp_orn']``.
            tcp_pixel_coords: per-camera ``[B, T', 2+]`` or ``[B, T', num_tcp, 2+]`` normalized UV
                (uses ``[:, -T:][..., :2]``).

        Returns:
            Tuple of four dicts (keys = ``camera_names``):

            - ``gt_tcp_uv[cam]``: ``[B, T, 2]`` or ``[B, T, num_tcp, 2]``
            - ``gt_tcp_3d[cam]``: ``[B, T, 3]`` or ``[B, T, num_tcp, 3]``
            - ``gt_tcp_6d[cam]``: ``[B, T, 6]`` or ``[B, T, num_tcp, 6]`` (rot6d from rotmat)
            - ``gt_tcp_valid[cam]``: ``[B, T]`` or ``[B, T, num_tcp]`` bool

            UV is clipped to ``[0, 1]`` for supervision stability; visibility is carried
            explicitly in ``gt_tcp_valid`` instead of zeroing the TCP pose tokens.
        """
        sample_uv = tcp_pixel_coords[camera_names[0]]
        dtype = sample_uv.dtype
        gt_pos_dict, gt_6d_dict = self._extract_gt_tcp_camera_dicts(
            camera_names,
            tcp_pos=tcp_pos,
            tcp_orn=tcp_orn,
            t_steps=t_steps,
            dtype=dtype,
        )
        gt_tcp_3d, gt_tcp_6d, gt_tcp_uv, gt_tcp_valid = {}, {}, {}, {}
        for cam in camera_names:
            if tcp_pixel_coords is not None and tcp_pixel_coords.get(cam, None) is not None:
                uv = tcp_pixel_coords[cam]
                if uv.ndim == 2:
                    uv = uv.unsqueeze(1)
                # Slice time on dim 1 and uv channels on the LAST dim, so a bimanual
                # [B,T,num_tcp,3] tensor keeps its arm axis intact for DiT TCP-KV slots.
                uv = uv[:, -t_steps:][..., :2].to(dtype=dtype)
                valid = (
                    tcp_valid[cam].bool()
                    if tcp_valid is not None and cam in tcp_valid
                    else ((uv >= 0) & (uv <= 1)).all(dim=-1)
                )
                gt_tcp_uv[cam] = uv.clamp(0.0, 1.0)
                gt_tcp_3d[cam] = gt_pos_dict[cam]
                gt_tcp_6d[cam] = gt_6d_dict[cam]
                gt_tcp_valid[cam] = valid
            else:
                raise ValueError(f"TCP pixel coordinates are not found for camera: {cam}")
        return gt_tcp_uv, gt_tcp_3d, gt_tcp_6d, gt_tcp_valid

    def _encode_observation(
        self,
        x,
        depths,
        camera_intrinsics,
        camera_names,
        tcp_pixel_coords,
        tcp_valid,
        future_tcp_pixel_coords,
        future_tcp_valid,
    ):
        """Preprocess and encode observations for the TCP and action stages."""
        B_, To_ = x[camera_names[0]].shape[:2]
        patch_h, patch_w = None, None
        depth_inputs = {}
        intrinsics_inputs = {}
        tcp_valid_inputs = {}
        future_tcp_uv_targets = {}
        future_tcp_valid_inputs = {}
        rgb_per_cam = None

        if "dinov" in self.backbone_type:
            rgb_per_cam, depth_per_cam, crop_offsets = self._prepare_per_camera_backbone_inputs(
                x,
                depths,
                camera_names,
                B_,
                To_,
            )
            # Crop first (keeps crop_offsets at [B*T] rows for the K/uv adjustments below),
            # then drop the history frames no module consumes before the backbone.
            t_per_cam = self._frames_to_encode(camera_names, To_)
            if any(t != To_ for t in t_per_cam):
                if depth_per_cam:
                    raise NotImplementedError("encode_only_consumed_frames does not support use_depth=true")
                rgb_per_cam = [
                    rgb if t == To_ else rgb.reshape(B_, To_, *rgb.shape[1:])[:, -t:].reshape(B_ * t, *rgb.shape[1:])
                    for rgb, t in zip(rgb_per_cam, t_per_cam)
                ]
            camera_features, visual_tokens, patch_h, patch_w = self._obtain_dinov2_visual_tokens(
                rgb_per_cam,
                camera_names,
                B_,
                t_per_cam,
            )

            if depth_per_cam:
                depth_inputs = self._build_depth_inputs_from_crops(
                    depth_per_cam,
                    camera_names,
                    B_,
                    To_,
                )

            intrinsics_inputs = self._build_intrinsics_inputs(
                camera_intrinsics,
                camera_names,
                To_,
                crop_offsets,
            )
            tcp_valid_inputs = self._adjust_tcp_pixel_coords_for_crop(
                tcp_pixel_coords,
                camera_names,
                crop_offsets,
                B_,
                To_,
                source_valid=tcp_valid,
            )
                        
        elif self.backbone_type == 'da3':
            rgb_per_cam = []
            depth_per_cam = []
            crop_offsets = {}  # camera_name -> [BT, 2] (top, left) pixel offsets
            need_depth = self.use_depth and depths is not None
            ref_hw = None
            for camera_name in camera_names:
                rgb = x[camera_name].reshape(B_ * To_, *x[camera_name].shape[2:])
                use_cam_depth = need_depth and "eye_in_hand" not in camera_name

                if use_cam_depth:
                    d = depths.get(camera_name)
                    if d is None:
                        raise ValueError(f"Depth is not found for camera: {camera_name}")
                    d = d.reshape(B_ * To_, *d.shape[2:])  # [BT, C, H, W]
                    if self.randomizer is not None:
                        rgb, d, offsets_bt, H_pre, W_pre = self._aligned_crop(
                            rgb, d, B_, To_
                        )
                        crop_offsets[camera_name] = (offsets_bt, H_pre, W_pre)
                    depth_per_cam.append(d)
                else:
                    if self.randomizer is not None:
                        rgb, _, offsets_bt, H_pre, W_pre = self._aligned_crop(
                            rgb, None, B_, To_
                        )
                        crop_offsets[camera_name] = (offsets_bt, H_pre, W_pre)

                if ref_hw is None:
                    ref_hw = rgb.shape[-2:]
                elif rgb.shape[-2:] != ref_hw:
                    raise ValueError(
                        "DA3 multi-view forward requires identical H,W across cameras; "
                        f"got {ref_hw} vs {tuple(rgb.shape[-2:])} for {camera_name}"
                    )
                rgb_per_cam.append(rgb)

            visual_tokens, _cls_tok, h_p, w_p = self._obtain_da3_visual_tokens(
                rgb_per_cam, camera_names, B_, To_
            )

            if depth_per_cam:
                depth_cams = [cn for cn in camera_names if "eye_in_hand" not in cn]
                if len(depth_per_cam) != len(depth_cams):
                    raise ValueError(
                        f"depth crops ({len(depth_per_cam)}) != non-wrist cameras ({len(depth_cams)})"
                    )
                for cam_idx, camera_name in enumerate(depth_cams):
                    d = depth_per_cam[cam_idx]  # [BT, C, H_crop, W_crop]
                    d = d.reshape(B_, To_, *d.shape[1:])  # [B, T, C, H, W]
                    d = d.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                    depth_inputs[camera_name] = d

            tcp_valid_inputs = self._adjust_tcp_pixel_coords_for_crop(
                tcp_pixel_coords,
                camera_names,
                crop_offsets,
                B_,
                To_,
                source_valid=tcp_valid,
            )

            # Camera intrinsics → [B, T, 4] for TCP head / DiT, then adjust for crop
            if self.use_camera_intrinsics and camera_intrinsics is not None:
                for camera_name in camera_names:
                    K = camera_intrinsics.get(camera_name)
                    if K is None:
                        continue
                    # Normalize to [B, T, 3, 3]
                    if K.ndim == 3:  # [B, 3, 3]
                        K = K.unsqueeze(1).expand(-1, To_, -1, -1)
                    fx = K[..., 0, 0]; fy = K[..., 1, 1]
                    cx = K[..., 0, 2]; cy = K[..., 1, 2]
                    # Adjust cx/cy for crop offset
                    if camera_name in crop_offsets:
                        offsets_bt = crop_offsets[camera_name][0]  # [BT, 2] = (top, left)
                        cx = cx.clone() - offsets_bt[:, 1].float().reshape(B_, To_)
                        cy = cy.clone() - offsets_bt[:, 0].float().reshape(B_, To_)
                    intrinsics_inputs[camera_name] = torch.stack([fx, fy, cx, cy], dim=-1)  # [B, T, 4]

            patch_h = visual_tokens[camera_names[0]].shape[-2]
            patch_w = visual_tokens[camera_names[0]].shape[-1]

        elif self.backbone_type == 'vggt':
            rgb_per_cam, depth_per_cam, crop_offsets = self._prepare_per_camera_backbone_inputs(
                x,
                depths,
                camera_names,
                B_,
                To_,
            )
            visual_tokens, patch_h, patch_w = self._obtain_vggt_visual_tokens(
                rgb_per_cam,
                camera_names,
                B_,
                To_,
            )

            if depth_per_cam:
                depth_inputs = self._build_depth_inputs_from_crops(
                    depth_per_cam,
                    camera_names,
                    B_,
                    To_,
                )

            intrinsics_inputs = self._build_intrinsics_inputs(
                camera_intrinsics,
                camera_names,
                To_,
                crop_offsets,
            )
            tcp_valid_inputs = self._adjust_tcp_pixel_coords_for_crop(
                tcp_pixel_coords,
                camera_names,
                crop_offsets,
                B_,
                To_,
                source_valid=tcp_valid,
            )

        else:
            raise ValueError(
                f"Unknown backbone_type: {self.backbone_type}. "
                "Choose 'dinov2', 'dinov3', 'da3', or 'vggt'"
            )

        if future_tcp_pixel_coords is not None and crop_offsets is not None:
            future_tcp_uv_targets, future_tcp_valid_inputs = self._adjust_tcp_pixel_coords_for_crop_target(
                future_tcp_pixel_coords,
                camera_names,
                crop_offsets,
                B_,
                To_,
                source_valid=future_tcp_valid,
            )

        input_hw = tuple(rgb_per_cam[0].shape[-2:]) if rgb_per_cam else None
        return ObservationFeatures(
            visual_tokens=visual_tokens,
            depth_inputs=depth_inputs,
            intrinsics_inputs=intrinsics_inputs,
            tcp_valid_inputs=tcp_valid_inputs,
            future_tcp_uv_targets=future_tcp_uv_targets,
            future_tcp_valid_inputs=future_tcp_valid_inputs,
            input_hw=input_hw,
            batch_size=B_,
            time_steps=To_,
            crop_offsets=crop_offsets,
        )

    def _run_latent_aux_model(self, features, camera_names, outputs):
        """Run LatentAuxiliaryModel (current-frame TCPPoseHead + attended_patch)."""
        if not self.use_latent_aux_model:
            return None

        aux_camera_names = [
            camera_name for camera_name in camera_names
            if camera_name not in self.aux_bypass_cameras
        ]
        outputs["_aux_camera_names"] = aux_camera_names
        if not aux_camera_names:
            return {"attended_patch": {}}

        if self.compile_latent_trunk and self.latent_aux_model is not None \
                and not getattr(self.latent_aux_model.trunk, '_forward_camera_compiled', False):
            # Compile LatentTrunk.forward_camera. Same bound-method cache as the backbone /
            # action head / dynamics head, so state_dict keys are untouched and the EMA
            # deepcopy compiles its own copy.
            trunk = self.latent_aux_model.trunk
            trunk.forward_camera = self._compiled_fn('latent_trunk', trunk.forward_camera)
            trunk._forward_camera_compiled = True
        latent_aux_out = self.latent_aux_model(
            {camera_name: features.visual_tokens[camera_name] for camera_name in aux_camera_names},
            {camera_name: features.depth_inputs[camera_name] for camera_name in aux_camera_names
             if camera_name in features.depth_inputs},
            {camera_name: features.intrinsics_inputs[camera_name] for camera_name in aux_camera_names
             if camera_name in features.intrinsics_inputs},
        )
        for key in [
            'tcp_uv', 'tcp_3d', 'tcp_6d', 'tcp_valid', 'tcp_valid_logit',
        ]:
            if key not in latent_aux_out:
                continue
            num_tcp = getattr(self.latent_aux_model, "num_tcp", 1)
            per_cam = []
            for camera_name in aux_camera_names:
                cam_val = latent_aux_out[key][camera_name]
                if num_tcp > 1:
                    # [B*T, num_tcp, C] -> [B, T, num_tcp, C]; the extra axis rides
                    # through the L1/BCE losses untouched since they reduce over all dims.
                    cam_val = cam_val.view(
                        features.batch_size, features.time_steps, num_tcp, -1
                    )
                elif cam_val.dim() == 2:
                    cam_val = cam_val.view(features.batch_size, features.time_steps, -1)
                elif cam_val.dim() == 3:
                    cam_val = cam_val.view(features.batch_size, cam_val.shape[1], -1)
                else:
                    raise ValueError(
                        f"Unexpected {key}[{camera_name}] shape: {tuple(cam_val.shape)}"
                    )
                per_cam.append(cam_val)
            outputs[key] = torch.stack(per_cam, dim=1)

        if features.tcp_valid_inputs:
            outputs["tcp_valid_target"] = torch.stack(
                [features.tcp_valid_inputs[camera_name].unsqueeze(-1) for camera_name in aux_camera_names],
                dim=1,
            )
        if features.future_tcp_uv_targets:
            # Preserve crop-adjusted future-TCP targets for future auxiliary objectives.
            outputs["future_tcp_uv_target"] = torch.stack(
                [features.future_tcp_uv_targets[camera_name][..., :2] for camera_name in aux_camera_names],
                dim=1,
            )
            outputs["future_tcp_valid_target"] = torch.stack(
                [features.future_tcp_valid_inputs[camera_name].unsqueeze(-1) for camera_name in aux_camera_names],
                dim=1,
            )
        return latent_aux_out

    def _bypass_aux_for_action(self, camera_name, visual_tokens):
        """Map a bypass camera's backbone tokens into the action-head token space."""
        tokens = visual_tokens.permute(0, 2, 3, 4, 1)
        tokens = self.action_bypass_adapters[camera_name](tokens)
        return tokens.permute(0, 4, 1, 2, 3).contiguous()

    def _dispatch_action_head(
        self,
        features,
        latent_aux_out,
        outputs,
        actions,
        state,
        camera_names,
        tcp_pos,
        tcp_orn,
        tcp_pixel_coords,
        camera_extrinsics,
        run_action_head,
    ):
        """Dispatch the configured action head after observation and latent-aux stages."""
        if not run_action_head or self.action_head is None:
            return

        if self.use_latent_aux_model and latent_aux_out is not None and 'attended_patch' in latent_aux_out:
            action_visual_tokens = {}
            for camera_name in camera_names:
                if camera_name in self.aux_bypass_cameras:
                    action_visual_tokens[camera_name] = self._bypass_aux_for_action(
                        camera_name,
                        features.visual_tokens[camera_name]
                    )
                else:
                    action_visual_tokens[camera_name] = latent_aux_out['attended_patch'][camera_name]
        else:
            action_visual_tokens = features.visual_tokens # From visual backbones directly
        head_type = getattr(self, '_action_head_type', 'DiTActionHead')
        if head_type == 'DiTActionHead':
            self._forward_dit_action(
                outputs,
                action_visual_tokens,
                actions,
                state,
                features.depth_inputs,
                features.intrinsics_inputs,
                camera_names,
                latent_aux_out,
                tcp_pos,
                tcp_orn,
                tcp_pixel_coords,
                features.tcp_valid_inputs,
                features.batch_size,
                features.time_steps,
                camera_extrinsics,
                features.input_hw,
            )
        else:
            self._forward_fm_action(
                outputs,
                action_visual_tokens,
                actions,
                state,
                features.intrinsics_inputs,
                camera_names,
                latent_aux_out,
            )

    def forward(self, x, depths, camera_intrinsics=None, state=None, camera_names=None, actions=None, tcp_pixel_coords=None, tcp_pos=None, tcp_orn=None, future_tcp_pixel_coords=None, _camera_extrinsics=None, run_action_head=True, tcp_valid=None, future_tcp_valid=None, **kwargs):
        if self.training and self.top_rotation_deg > 0 and camera_intrinsics is not None:
            from sapolicy.models.utils.rot_aug import apply_top_rotation_aug
            (x, depths, tcp_pixel_coords, tcp_pos, tcp_orn, future_tcp_pixel_coords,
             _camera_extrinsics, _) = apply_top_rotation_aug(
                x, depths, camera_intrinsics, tcp_pixel_coords, tcp_pos, tcp_orn,
                future_tcp_pixel_coords, _camera_extrinsics, self.top_rotation_deg)
        features = self._encode_observation(
            x=x,
            depths=depths,
            camera_intrinsics=camera_intrinsics,
            camera_names=camera_names,
            tcp_pixel_coords=tcp_pixel_coords,
            tcp_valid=tcp_valid,
            future_tcp_pixel_coords=future_tcp_pixel_coords,
            future_tcp_valid=future_tcp_valid,
        )
        outputs = {
            "_camera_intrinsics_inputs": features.intrinsics_inputs,
            "_input_hw": features.input_hw,
            "_visual_tokens": features.visual_tokens,
            "_crop_offsets": features.crop_offsets,
            "_batch_size": features.batch_size,
            "_time_steps": features.time_steps,
        }
        latent_aux_out = self._run_latent_aux_model(features, camera_names, outputs)
        if latent_aux_out is not None and "attended_patch" in latent_aux_out:
            outputs["_attended_patch"] = latent_aux_out["attended_patch"]
        self._dispatch_action_head(
            features,
            latent_aux_out,
            outputs,
            actions,
            state,
            camera_names,
            tcp_pos,
            tcp_orn,
            tcp_pixel_coords,
            _camera_extrinsics,
            run_action_head,
        )
        return outputs

    def _forward_dit_action(
        self, outputs, dit_visual_tokens, actions, state,
        depth_inputs, intrinsics_inputs, camera_names,
        latent_aux_out, tcp_pos, tcp_orn,
        tcp_pixel_coords, tcp_valid_inputs,
        B_, To_, _camera_extrinsics, input_hw,
    ):
        """Forward path for DiTActionHead."""
        dit_depths = depth_inputs if self.use_depth else None
        dit_K = intrinsics_inputs if self.use_camera_intrinsics else None
        tcp_camera_names = outputs.get("_aux_camera_names", camera_names)
        if self.training and actions is not None:
            if not getattr(self.action_head, 'disable_tcp_kv', False):
                dit_uv, dit_3d, dit_6d, dit_valid = self._build_dit_gt_tcp_dicts(
                    tcp_camera_names, B_, To_,
                    tcp_pos, tcp_orn,
                    tcp_pixel_coords, tcp_valid=tcp_valid_inputs,
                )
                device = next(iter(dit_visual_tokens.values())).device
                p_pred = self._dit_tcp_kv_pred_prob()
                outputs["dit_tcp_kv_pred_prob"] = torch.full((), float(p_pred), device=device)
                if p_pred > 0.0:
                    if "tcp_uv" not in outputs or "tcp_3d" not in outputs or "tcp_6d" not in outputs:
                        raise ValueError(
                            "DiT tcp_kv_curriculum requires TCP head outputs in training forward."
                        )
                    pred_uv = self._tcp_stacked_per_cam_dict(outputs, "tcp_uv", tcp_camera_names)
                    pred_3d = self._tcp_stacked_per_cam_dict(outputs, "tcp_3d", tcp_camera_names)
                    pred_6d = self._tcp_stacked_per_cam_dict(outputs, "tcp_6d", tcp_camera_names)
                    pred_valid = (
                        self._tcp_stacked_per_cam_dict(outputs, "tcp_valid", tcp_camera_names)
                        if "tcp_valid" in outputs else None
                    )
                    dit_uv, dit_3d, dit_6d, dit_valid, used_pred = self._select_tcp_kv_by_curriculum(
                        dit_uv,
                        dit_3d,
                        dit_6d,
                        dit_valid,
                        pred_uv,
                        pred_3d,
                        pred_6d,
                        pred_valid,
                        p_pred,
                        device,
                    )
                    outputs["dit_tcp_kv_used_pred"] = torch.full((), float(used_pred), device=device)
                else:
                    outputs["dit_tcp_kv_used_pred"] = torch.zeros((), device=device)
            else:
                dit_uv, dit_3d, dit_6d, dit_valid = None, None, None, None
            outputs["actions"] = self.action_head.forward_training(
                actions,
                dit_visual_tokens,
                dit_depths,
                dit_K,
                dit_uv,
                dit_3d,
                dit_6d,
                gt_tcp_valid=dit_valid,
                state=state if self.use_state else None
            )
        else:
            if not getattr(self.action_head, 'disable_tcp_kv', False):
                if "tcp_3d" not in outputs or "tcp_uv" not in outputs:
                    raise ValueError(
                        "DiTActionHead inference requires LatentAuxiliaryModel TCP outputs (tcp_*). "
                        "Enable use_latent_aux_model=True."
                    )
            if "tcp_uv" in outputs:
                pred_uv = self._tcp_stacked_per_cam_dict(outputs, "tcp_uv", tcp_camera_names)
                pred_3d = self._tcp_stacked_per_cam_dict(outputs, "tcp_3d", tcp_camera_names)
                pred_6d = self._tcp_stacked_per_cam_dict(outputs, "tcp_6d", tcp_camera_names)
                pred_valid = (
                    self._tcp_stacked_per_cam_dict(outputs, "tcp_valid", tcp_camera_names)
                    if "tcp_valid" in outputs else None
                )
            else:
                pred_uv, pred_3d, pred_6d, pred_valid = None, None, None, None

            if self.use_oracle_tcp and state is not None and _camera_extrinsics is not None:
                oracle_uv, oracle_3d, oracle_6d = self._compute_oracle_tcp(
                    state, _camera_extrinsics, intrinsics_inputs, tcp_camera_names, B_, To_,
                    image_hw=input_hw,
                )
                for cam in oracle_uv:
                    pred_uv[cam] = oracle_uv[cam]
                    pred_3d[cam] = oracle_3d[cam]
                    pred_6d[cam] = oracle_6d[cam]
                pred_valid = None

            outputs["actions"] = self.action_head.sample_trajectory(
                dit_visual_tokens,
                depths=dit_depths,
                camera_intrinsics=dit_K,
                pred_tcp_uv=pred_uv,
                pred_tcp_3d=pred_3d,
                pred_tcp_6d=pred_6d,
                pred_tcp_valid=pred_valid,
                state=state if self.use_state else None
            )

    def _forward_fm_action(
        self, outputs, visual_tokens, actions, state,
        intrinsics_inputs, camera_names,
        latent_aux_out,  # TODO: FM not used tcp prediction!
    ):
        """Forward path for TransformerFlowMatchingHead / TransformerDiffusionHead.

        Wraps single-layer ``visual_tokens`` as ``obs_features`` matching the FM
        head's expected ``{cam: [(patch[B,T,N,C], cls[B,T,C])]}`` format (one entry
        per camera). ``visual_tokens`` is set by the top-level router to either
        backbone last-layer output (``use_latent_aux_model=False``) or LatentAuxiliaryModel
        ``attended_patch`` (``use_latent_aux_model=True``).
        """

        obs_features = {}
        for cam_name, cam_tokens in visual_tokens.items():
            b, c, t, h, w = cam_tokens.shape
            patch_tokens = cam_tokens.permute(0, 2, 3, 4, 1).reshape(b, t, h * w, c)
            obs_features[cam_name] = [(patch_tokens, None)]
        patch_h = list(visual_tokens.values())[0].shape[3]
        patch_w = list(visual_tokens.values())[0].shape[4]

        fm_state = state if self.use_state else None

        if self.training and actions is not None:
            outputs["actions"] = self.action_head.forward(
                obs_features, patch_h, patch_w,
                actions=actions, state=fm_state,
                camera_intrinsics=intrinsics_inputs,
            )
        else:
            outputs["actions"] = self.action_head.forward(
                obs_features, patch_h, patch_w,
                actions=None, state=fm_state,
                camera_intrinsics=intrinsics_inputs,
            )

    def forward_test(self, batch, training=False, resize=True, run_action_head=True):
        # Extract data from observation
        obs = batch["observation"]
        rgb = obs["image"]
        depth = obs["depth"] if "depth" in obs.keys() else None
        camera_names = list(rgb.keys())

        state = obs.get("state", None)
        if self.normalizer is not None:
            batch["action"] = self.normalizer.normalize({"action": batch["action"]})["action"]
            if state is not None and "state" in self.normalizer.params_dict:
                state = self.normalizer.normalize({"state": state})["state"]

        outputs = self.forward(
            x=rgb,
            depths=depth,
            camera_intrinsics=obs.get("camera_intrinsics", None),
            state=state,
            camera_names=camera_names,
            actions=None if not ("action" in batch) else batch["action"],
            tcp_pixel_coords=obs.get("tcp_pixel_coords"),
            tcp_pos=obs.get("tcp_pos"),
            tcp_orn=obs.get("tcp_orn"),
            tcp_valid=obs.get("tcp_valid"),
            future_tcp_pixel_coords=obs.get("future_tcp_pixel_coords"),
            future_tcp_valid=obs.get("future_tcp_valid"),
            run_action_head=run_action_head,
        )

        return outputs

    def forward_train(self, batches):
        if not isinstance(batches, List):
            return self.forward_train_batch(batches)
        loss = 0.0
        outputs = []
        for batch in batches:
            outputs.append(self.forward_train_batch(batch))
            loss += outputs[-1]["loss"]
        return {"loss": loss}

    @staticmethod
    def _get_device_from_batch(batch):
        """Get the device of the first tensor found in batch (for safe fallback tensor creation)."""
        for v in batch.get("observation", {}).get("image", {}).values():
            if torch.is_tensor(v):
                return v.device
        return torch.device("cpu")

    @staticmethod
    @contextlib.contextmanager
    def _temporarily_freeze_module(module, eval_mode=False):
        """Disable parameter grads during forward while preserving input gradients."""
        if module is None:
            yield
            return
        params = list(module.parameters())
        requires_grad = [p.requires_grad for p in params]
        was_training = module.training
        try:
            for p in params:
                p.requires_grad_(False)
            if eval_mode:
                module.eval()
            yield
        finally:
            if eval_mode:
                module.train(was_training)
            for p, old_requires_grad in zip(params, requires_grad):
                p.requires_grad_(old_requires_grad)

    @staticmethod
    def _parse_action_head_train_output(actions_out):
        """Normalize action-head training output to a 6-tuple.

        Returns:
            predicted_flow, target_flow, a_t, action_loss, sampled_t, sampled_noise
        """
        if not isinstance(actions_out, (tuple, list)):
            raise ValueError("Action head training output must be tuple/list")

        if len(actions_out) == 4:
            predicted_flow, target_flow, a_t, action_loss = actions_out
            sampled_t = None
            sampled_noise = None
        elif len(actions_out) == 5:
            predicted_flow, target_flow, a_t, action_loss, sampled_t = actions_out
            sampled_noise = None
        elif len(actions_out) >= 6:
            predicted_flow, target_flow, a_t, action_loss, sampled_t, sampled_noise = actions_out[:6]
        else:
            raise ValueError(f"Unexpected action head output length: {len(actions_out)}")

        return predicted_flow, target_flow, a_t, action_loss, sampled_t, sampled_noise

    def _crop_rgb_with_offsets(self, rgb_bt, offsets_bt):
        """Apply stored crop offsets to ``[B*T, C, H, W]`` RGB."""
        from sapolicy.models.utils.crop_randomizer import crop_image_from_indices

        if self.randomizer is None:
            return rgb_bt
        crop_h = int(self.randomizer.crop_height)
        crop_w = int(self.randomizer.crop_width)
        return crop_image_from_indices(
            rgb_bt,
            offsets_bt.to(device=rgb_bt.device, dtype=torch.long),
            crop_h,
            crop_w,
        )

    def _encode_future_backbone_tokens(self, future_images, camera_name, crop_offsets, batch_size, t_obs):
        """Encode future RGB ``[B, K, C, H, W]`` with the same crop as current obs.

        Returns ``[B, K, C_vis, H_p, W_p]`` under ``torch.no_grad``.
        """
        fut = future_images[camera_name]
        if fut.ndim != 5:
            raise ValueError(f"future_image[{camera_name}] must be [B,K,C,H,W], got {tuple(fut.shape)}")
        b, k, c, h, w = fut.shape
        if b != batch_size:
            raise ValueError(f"future_image batch {b} != {batch_size}")
        rgb_bt = fut.reshape(b * k, c, h, w)
        if self.randomizer is not None:
            if not crop_offsets or camera_name not in crop_offsets:
                raise ValueError(f"Missing crop_offsets for future encode cam={camera_name}")
            offsets_bt, _, _ = crop_offsets[camera_name]
            offsets_b = offsets_bt.reshape(batch_size, t_obs, 2)[:, -1]  # last obs frame
            offsets_bk = offsets_b.unsqueeze(1).expand(-1, k, -1).reshape(b * k, 2)
            rgb_bt = self._crop_rgb_with_offsets(rgb_bt, offsets_bk)

        with torch.no_grad():
            if "dinov" in self.backbone_type:
                _cam_feat, visual_tokens, _, _ = self._obtain_dinov2_visual_tokens(
                    [rgb_bt], [camera_name], b, k
                )
            elif self.backbone_type == "da3":
                visual_tokens, _cls, _, _ = self._obtain_da3_visual_tokens(
                    [rgb_bt], [camera_name], b, k
                )
            elif self.backbone_type == "vggt":
                visual_tokens, _, _ = self._obtain_vggt_visual_tokens(
                    [rgb_bt], [camera_name], b, k
                )
            else:
                raise ValueError(f"Unsupported backbone for future video: {self.backbone_type}")
        # visual_tokens[cam]: [B, C_vis, K, H_p, W_p] -> [B, K, C_vis, H_p, W_p]
        tok = visual_tokens[camera_name].permute(0, 2, 1, 3, 4).contiguous()
        return tok.detach()

    @staticmethod
    def _dynamics_horizon_slot_count(obs, cam, key):
        data = obs.get(key, None)
        if not isinstance(data, dict) or cam not in data:
            return 0
        val = data[cam]
        if not torch.is_tensor(val):
            val = torch.as_tensor(val)
        if val.ndim >= 2 and val.shape[1] not in (2, 3, 6, 9):
            return int(val.shape[1])
        return 1

    def _resolve_dynamics_horizon_idx(self, obs, cam, prediction_horizon):
        """Return the sole future slot index; dataset must load ``horizons=[T]``."""
        counts = []
        for key in ("future_image", "future_tcp_pos", "future_tcp_pixel_coords"):
            n = self._dynamics_horizon_slot_count(obs, cam, key)
            if n > 0:
                counts.append(n)
        if not counts:
            return 0
        k = max(counts)
        if k != 1:
            raise ValueError(
                f"Dynamics requires a single future horizon slot at T={prediction_horizon}; "
                f"got K={k} for camera {cam}. Set future_image_horizons / "
                f"future_tcp_horizons to [action_sequence_length] (T={prediction_horizon})."
            )
        return 0

    def _resolve_dynamics_camera(self, camera_names, visual_tokens, attended):
        """Pick the third-view canonical camera for dynamics supervision."""
        cam = self._dynamics_camera or "agentview"
        forbidden = {"robot0_eye_in_hand", "left_camera", "right_camera"}
        if cam in forbidden or "eye_in_hand" in cam:
            raise ValueError(
                f"Dynamics camera {cam!r} must be the third-view canonical name "
                "(default agentview), not a wrist camera."
            )
        if cam not in visual_tokens:
            raise ValueError(
                f"Dynamics camera {cam!r} missing from batch visual tokens "
                f"(available: {sorted(visual_tokens.keys())})."
            )
        if not isinstance(attended, dict) or cam not in attended:
            raise ValueError(
                f"Dynamics requires LatentAuxiliaryModel attended_patch for camera {cam}; "
                "ensure the third view is not in aux_bypass_cameras."
            )
        return cam

    def _crop_adjusted_future_tcp_from_output(self, output, cam, horizon_idx):
        """Return crop-adjusted future TCP UV/valid for ``cam`` if present in forward outputs."""
        aux_camera_names = output.get("_aux_camera_names", None)
        if not aux_camera_names or cam not in aux_camera_names:
            return None, None
        cam_idx = aux_camera_names.index(cam)
        uv = output.get("future_tcp_uv_target", None)
        valid = output.get("future_tcp_valid_target", None)
        if uv is None:
            return None, None
        # Stacked as [B, num_cams, K, ...] — always select horizon slot (including K=1).
        uv_cam = uv[:, cam_idx]
        if uv_cam.ndim >= 3:
            uv_cam = uv_cam[:, horizon_idx]
        gt_uv = uv_cam[..., :2].clamp(0.0, 1.0)
        gt_valid = None
        if valid is not None:
            valid_cam = valid[:, cam_idx]
            if valid_cam.ndim >= 3:
                valid_cam = valid_cam[:, horizon_idx]
            gt_valid = valid_cam.squeeze(-1)
            if gt_valid.dtype != torch.bool:
                gt_valid = gt_valid > 0.5
        return gt_uv, gt_valid

    def _extract_future_tcp_gt_at_horizon(
        self,
        obs,
        cam,
        horizon_idx=0,
        *,
        crop_uv=None,
        crop_valid=None,
    ):
        """Slice batched multi-horizon future TCP tensors to one step ``[B, *]``."""

        def _take(key):
            data = obs.get(key, None)
            if not isinstance(data, dict) or cam not in data:
                return None
            val = data[cam]
            device = next(self.parameters()).device
            if not torch.is_tensor(val):
                val = torch.as_tensor(val, device=device)
            else:
                val = val.to(device=device)
            if val.ndim >= 2 and val.shape[1] > 1:
                val = val[:, horizon_idx : horizon_idx + 1]
            elif val.ndim == 2 and val.shape[1] in (2, 3, 6, 9):
                val = val.unsqueeze(1)
            return val

        gt_uv_raw = _take("future_tcp_pixel_coords")
        gt_pos = _take("future_tcp_pos")
        gt_orn = _take("future_tcp_orn")
        gt_valid = _take("future_tcp_valid")
        if gt_pos is None or gt_orn is None:
            return None

        gt_pos_dict, gt_6d_dict = self._extract_gt_tcp_camera_dicts(
            [cam],
            tcp_pos={cam: gt_pos},
            tcp_orn={cam: gt_orn},
            t_steps=1,
            dtype=gt_pos.dtype,
        )
        gt_tcp_3d = gt_pos_dict[cam][:, 0]
        gt_tcp_6d = gt_6d_dict[cam][:, 0]

        gt_tcp_uv = crop_uv
        if gt_tcp_uv is None and gt_uv_raw is not None:
            gt_tcp_uv = gt_uv_raw[:, 0, ..., :2].clamp(0.0, 1.0)

        gt_tcp_valid = crop_valid
        if gt_tcp_valid is None and gt_valid is not None:
            gt_tcp_valid = gt_valid[:, 0]
            if gt_tcp_valid.dtype != torch.bool:
                gt_tcp_valid = gt_tcp_valid > 0.5
        return gt_tcp_uv, gt_tcp_3d, gt_tcp_6d, gt_tcp_valid

    def _compute_dynamics_loss(self, output, batch, camera_names):
        """Run DynamicsHead (future TCP + semantic diffusion) at horizon T."""
        device = self._get_device_from_batch(batch)
        zero = torch.zeros((), device=device)
        stats = {
            "loss_future_dynamics": zero,
            "loss_future_tcp": zero,
            "loss_future_semantic": zero,
            "loss_future_video": zero,
        }
        if not self.use_dynamics_head or self.dynamics_head is None:
            return zero, stats

        obs = batch["observation"]
        visual_tokens = output["_visual_tokens"]
        attended = output.get("_attended_patch", None)
        cam = self._resolve_dynamics_camera(camera_names, visual_tokens, attended)
        horizon_idx = self._resolve_dynamics_horizon_idx(
            obs, cam, self.dynamics_head.prediction_horizon
        )

        feat_t = visual_tokens[cam][:, :, -1]
        q_tokens = attended[cam][:, :, -1]
        b = feat_t.shape[0]

        feat_future = None
        feat_future_valid = None
        if self.dynamics_head.predict_semantic:
            future_images = obs.get("future_image", None)
            if not isinstance(future_images, dict) or cam not in future_images:
                raise ValueError(
                    f"dynamics predict_semantic requires observation.future_image[{cam}] "
                    f"at horizon T={self.dynamics_head.prediction_horizon}"
                )
            t_obs = int(output.get("_time_steps", visual_tokens[cam].shape[2]))
            feat_future_all = self._encode_future_backbone_tokens(
                future_images,
                cam,
                output.get("_crop_offsets", {}),
                b,
                t_obs,
            )
            if feat_future_all.shape[1] <= horizon_idx:
                raise ValueError(
                    f"future_image[{cam}] has K={feat_future_all.shape[1]} slots; "
                    f"need horizon_idx={horizon_idx} for T={self.dynamics_head.prediction_horizon}"
                )
            feat_future = feat_future_all[:, horizon_idx]
            valid = obs.get("future_image_valid", None)
            if isinstance(valid, dict) and cam in valid:
                valid_t = valid[cam]
                if valid_t.ndim == 1:
                    valid_t = valid_t.unsqueeze(0).expand(b, -1)
                valid_t = valid_t.to(device=device)
                if valid_t.shape[1] > 1:
                    feat_future_valid = valid_t[:, horizon_idx]
                else:
                    feat_future_valid = valid_t
            else:
                feat_future_valid = torch.ones(b, device=device)

        gt_tcp_uv = gt_tcp_3d = gt_tcp_6d = gt_tcp_valid = None
        if self.dynamics_head.predict_tcp:
            crop_uv, crop_valid = self._crop_adjusted_future_tcp_from_output(
                output, cam, horizon_idx
            )
            future_tcp = self._extract_future_tcp_gt_at_horizon(
                obs,
                cam,
                horizon_idx=horizon_idx,
                crop_uv=crop_uv,
                crop_valid=crop_valid,
            )
            if future_tcp is None:
                raise ValueError(
                    f"dynamics predict_tcp requires future_tcp_* tensors for camera {cam}"
                )
            gt_tcp_uv, gt_tcp_3d, gt_tcp_6d, gt_tcp_valid = future_tcp

        if self.compile_dynamics_head and self.dynamics_head.future_semantic is not None \
                and not getattr(self.dynamics_head.future_semantic, '_velocity_compiled', False):
            # Compile the velocity net (the 4-layer wide DiT). Bound-method compile via the same cache the
            # backbone / action head use; EMA deepcopy gets its own compile, state_dict keys are unchanged.
            sem = self.dynamics_head.future_semantic
            sem._predict_velocity = self._compiled_fn('dyn_velocity', sem._predict_velocity)
            sem._velocity_compiled = True
        tcp_loss, semantic_loss, dyn_stats = self.dynamics_head.compute_losses(
            q_tokens,
            feat_t,
            feat_future=feat_future,
            feat_future_valid=feat_future_valid,
            gt_tcp_uv=gt_tcp_uv,
            gt_tcp_3d=gt_tcp_3d,
            gt_tcp_6d=gt_tcp_6d,
            gt_tcp_valid=gt_tcp_valid,
        )
        stats.update(dyn_stats)

        total = zero
        if self.dynamics_head.predict_tcp and self._dynamics_tcp_loss_weight > 0:
            total = total + self._dynamics_tcp_loss_weight * tcp_loss
        if self.dynamics_head.predict_semantic and self._dynamics_semantic_loss_weight > 0:
            total = total + self._dynamics_semantic_loss_weight * semantic_loss
        stats["loss_future_dynamics"] = total.detach()
        stats["loss_future_video"] = stats.get("loss_future_semantic", zero)
        return total, stats

    def forward_train_batch(self, batch, loss_mode="both"):
        if loss_mode not in ("both", "tcp", "action"):
            raise ValueError(f"Invalid loss_mode={loss_mode!r}; expected 'both', 'tcp', or 'action'.")
        compute_action_loss = loss_mode in ("both", "action")
        compute_tcp_loss = loss_mode in ("both", "tcp")
        camera_names = list(batch["observation"]["image"].keys())
        obs = batch["observation"]
        device = self._get_device_from_batch(batch)

        # loss_mode="action" only: freeze LatentAuxiliaryModel (grads still flow to backbone).
        # loss_mode="both" (default for joint action loader): latent-aux is trainable too.
        freeze_latent_aux = (
            loss_mode == "action"
            and self.use_latent_aux_model
            and self.latent_aux_model is not None
        )
        freeze_ctx = (
            self._temporarily_freeze_module(self.latent_aux_model, eval_mode=True)
            if freeze_latent_aux
            else contextlib.nullcontext()
        )
        with freeze_ctx:
            output = self.forward_test(
                batch,
                resize=False,
                training=True,
                run_action_head=compute_action_loss,
            )

        # Handle action sequence training
        action_loss = torch.zeros((), device=device)
        action_consistency_loss = torch.zeros((), device=device)
        action_consistency_pose_loss = torch.zeros((), device=device)
        action_consistency_uv_loss = torch.zeros((), device=device)
        if compute_action_loss and self.use_action_head and self.action_head is not None:
            if 'action' not in batch:
                raise ValueError("Ground truth actions not found in batch")

            actions_out = output['actions']
            predicted_flow, target_flow, a_t, action_loss, sampled_t, sampled_noise = self._parse_action_head_train_output(
                actions_out
            )
            if action_loss is None:
                action_loss = F.mse_loss(predicted_flow, target_flow)

            # ActionConsistencyLoss: MSE( compose(pred_tcp, pred_action), compose(gt_tcp, gt_action) )
            # — pred/gt actions are the same 9D (pos3+rot6d) slice as DiT flow-matching targets (batch['action']).
            # — gt_tcp from observation (current step, same camera as pred_tcp). Requires sampled_t for x0 estimate.
            if (
                self.action_consistency_weight > 0
                and "tcp_3d" in output
                and "tcp_6d" in output
                and sampled_t is not None
                and predicted_flow.shape[-1] >= 9
            ):
                aux_camera_names = output.get("_aux_camera_names", camera_names)
                if not aux_camera_names:
                    aux_camera_names = []
                ref_cam = aux_camera_names[0] if aux_camera_names else None
                tcp_pos_obs = obs.get("tcp_pos", None)
                tcp_orn_obs = obs.get("tcp_orn", None)
                if (
                    ref_cam is not None
                    and tcp_pos_obs is not None
                    and isinstance(tcp_pos_obs, dict)
                    and ref_cam in tcp_pos_obs
                    and tcp_orn_obs is not None
                    and ref_cam in tcp_orn_obs
                ):
                    pred_tcp_9d = torch.cat(
                        [
                            output["tcp_3d"][:, 0, -1, :],
                            output["tcp_6d"][:, 0, -1, :],
                        ],
                        dim=-1,
                    )
                    t_expand = sampled_t.view(-1, 1, 1)
                    pred_clean_action = a_t - t_expand * predicted_flow
                    pred_clean_action_raw = self.normalizer.unnormalize({"action": pred_clean_action})["action"]
                    pred_action_9d = pred_clean_action_raw[..., :9]

                    gt_action_full = batch["raw_relative_action"].to(device=device, dtype=pred_action_9d.dtype)
                    if gt_action_full.dim() == 2:
                        gt_action_full = gt_action_full.unsqueeze(1)
                    if gt_action_full.shape[-1] >= 9:
                        gt_action_9d = gt_action_full[..., :9]
                        ta = min(pred_action_9d.shape[1], gt_action_9d.shape[1])
                        pred_action_9d = pred_action_9d[:, :ta, :]
                        gt_action_9d = gt_action_9d[:, :ta, :]

                        gt_pos_dict, gt_6d_dict = self._extract_gt_tcp_camera_dicts(
                            aux_camera_names,
                            tcp_pos=tcp_pos_obs,
                            tcp_orn=tcp_orn_obs,
                        )
                        gt_tcp_pos = gt_pos_dict[ref_cam][:, -1, :].to(
                            device=device, dtype=pred_tcp_9d.dtype
                        )
                        gt_tcp_6d = gt_6d_dict[ref_cam][:, -1, :].to(
                            device=device, dtype=pred_tcp_9d.dtype
                        )
                        gt_tcp_9d = torch.cat([gt_tcp_pos, gt_tcp_6d], dim=-1)

                        intrinsics_inputs = output.get("_camera_intrinsics_inputs", {})
                        ref_cam_K = (
                            intrinsics_inputs.get(ref_cam, None)
                            if isinstance(intrinsics_inputs, dict)
                            else None
                        )
                        if ref_cam_K is not None:
                            ref_cam_K = ref_cam_K[:, -1, :].to(
                                device=device, dtype=pred_tcp_9d.dtype
                            )
                        consistency_out = self.action_consistency_loss_fn(
                            pred_tcp_9d,
                            pred_action_9d,
                            gt_tcp_9d,
                            gt_action_9d,
                            camera_intrinsics=ref_cam_K,
                            image_hw=output.get("_input_hw", None),
                            uv_weight=self.action_consistency_uv_weight,
                        )
                        action_consistency_loss = consistency_out["loss"]
                        action_consistency_pose_loss = consistency_out["loss_pose"]
                        action_consistency_uv_loss = consistency_out["loss_uv"]
                        action_loss = (
                            action_loss
                            + self.action_consistency_weight * action_consistency_loss
                        )

        # TCP auxiliary loss (3D position + 6D rotation; optional UV/projection terms)
        tcp_loss = {
            "total": torch.zeros((), device=device),
            "loss_uv": torch.zeros((), device=device),
            "loss_3d": torch.zeros((), device=device),
            "loss_6d": torch.zeros((), device=device),
            "loss_valid": torch.zeros((), device=device),
            "loss_proj_gt": torch.zeros((), device=device),
            "loss_proj_self": torch.zeros((), device=device),
        }
        if (compute_tcp_loss
                and self.use_latent_aux_model
                and hasattr(self, '_tcp_loss_weight')
                and 'tcp_3d' in output
                and 'tcp_6d' in output):
            aux_camera_names = output.get("_aux_camera_names", camera_names)
            tcp_pos_obs = obs.get("tcp_pos", None)
            tcp_orn_obs = obs.get("tcp_orn", None)
            if tcp_pos_obs is not None and aux_camera_names:
                gt_pos_dict, gt_6d_dict = self._extract_gt_tcp_camera_dicts(
                    aux_camera_names,
                    tcp_pos=tcp_pos_obs,
                    tcp_orn=tcp_orn_obs,
                )
                pred_3d = output['tcp_3d']  # [B, num_cam, T, 3]
                pred_6d = output['tcp_6d']  # [B, num_cam, T, 6]
                pred_uv = output.get('tcp_uv', None)  # [B, num_cam, T, 2] if present
                pred_valid_logit = output.get('tcp_valid_logit', None)  # [B, num_cam, T, 1] if present
                pred_valid_target = output.get('tcp_valid_target', None)  # [B, num_cam, T, 1] if present
                t_pred = pred_3d.shape[2]
                gt_pos_exp = torch.stack(
                    [gt_pos_dict[camera_name][:, -t_pred:] for camera_name in aux_camera_names],
                    dim=1,
                ).to(device=pred_3d.device, dtype=pred_3d.dtype)
                gt_6d_exp = torch.stack(
                    [gt_6d_dict[camera_name][:, -t_pred:] for camera_name in aux_camera_names],
                    dim=1,
                ).to(device=pred_6d.device, dtype=pred_6d.dtype)
                # Build GT uv from dataset if available: observation['tcp_pixel_coords'][camera_name] -> [B,T,2] (normalized)
                gt_uv_exp = None
                gt_valid_exp = None
                tcp_pixel_coords = obs.get("tcp_pixel_coords", None)
                if isinstance(tcp_pixel_coords, dict) and len(tcp_pixel_coords) > 0 and pred_uv is not None:
                    gt_uv_per_cam = []
                    gt_valid_per_cam = []
                    obs_tcp_valid = obs.get("tcp_valid", None)
                    for camera_name in aux_camera_names:
                        cam_uv = tcp_pixel_coords.get(camera_name, None)
                        if cam_uv is None:
                            gt_uv_per_cam = []
                            gt_valid_per_cam = []
                            break
                        if cam_uv.ndim == 2:
                            cam_uv = cam_uv.unsqueeze(1)  # [B,1,2]
                        # Slice time on dim 1 and the uv channels on the LAST dim, so a
                        # bimanual [B,T,num_tcp,3] tensor keeps its arm axis intact.
                        cam_uv = cam_uv[:, -t_pred:][..., :2]  # [B,T,(num_tcp,)2]
                        gt_uv_per_cam.append(cam_uv)
                        cam_valid = ((cam_uv >= 0) & (cam_uv <= 1)).all(dim=-1)
                        if isinstance(obs_tcp_valid, dict) and camera_name in obs_tcp_valid:
                            raw_valid = obs_tcp_valid[camera_name]
                            if raw_valid.ndim == 3 and raw_valid.shape[-1] == 1:
                                raw_valid = raw_valid[..., 0]
                            if raw_valid.ndim == 1:
                                raw_valid = raw_valid.unsqueeze(0)
                            raw_valid = raw_valid[:, -t_pred:].bool()
                            cam_valid = cam_valid & raw_valid
                        gt_valid_per_cam.append(cam_valid)
                    if len(gt_uv_per_cam) == len(aux_camera_names):
                        gt_uv_exp = torch.stack(gt_uv_per_cam, dim=1).to(pred_uv.dtype)  # [B,num_cam,T,2]
                        gt_valid_exp = torch.stack(gt_valid_per_cam, dim=1)  # [B,num_cam,T]
                if pred_valid_target is not None:
                    gt_valid_exp = pred_valid_target.squeeze(-1).bool()

                # Mask the regression terms by label validity. Episodes without a usable
                # TCP sidecar are emitted by the loader as zero position + identity
                # orientation with tcp_valid=0 (see AbcEpisodeDataset docstring); an
                # unmasked L1 pulls the head towards "TCP at the camera origin with
                # identity rotation" on those samples. On the ABC bottles real split
                # that placeholder covered 415/3621 episodes = 1,137,258 of 7,081,084
                # steps (16.1%). gt_valid_exp was previously computed but only ever
                # used as the BCE target of the validity head.
                def _masked_l1(pred, tgt, valid):
                    if valid is None:
                        return F.l1_loss(pred, tgt)
                    m = valid.to(pred.dtype)
                    while m.ndim < pred.ndim:
                        m = m.unsqueeze(-1)
                    m = m.expand_as(pred)
                    denom = m.sum()
                    # Fully-unlabelled batch: return a real zero that still carries a
                    # gradient path, so DDP sees every parameter and does not hang.
                    return torch.where(
                        denom > 0,
                        ((pred - tgt).abs() * m).sum() / denom.clamp(min=1.0),
                        (pred * 0.0).sum(),
                    )

                loss_3d = _masked_l1(pred_3d, gt_pos_exp, gt_valid_exp)
                loss_6d = _masked_l1(pred_6d, gt_6d_exp, gt_valid_exp)
                if pred_uv is not None and gt_uv_exp is not None:
                    loss_uv = _masked_l1(pred_uv, gt_uv_exp.clamp(0.0, 1.0), gt_valid_exp)
                else:
                    loss_uv = torch.zeros((), device=device)
                if pred_valid_logit is not None and gt_valid_exp is not None:
                    loss_valid = F.binary_cross_entropy_with_logits(
                        pred_valid_logit.squeeze(-1),
                        gt_valid_exp.to(dtype=pred_valid_logit.dtype),
                    )
                else:
                    loss_valid = torch.zeros((), device=device)
                total_tcp = (
                    self.tcp_aux_loss.pos_w * loss_3d
                    + self.tcp_aux_loss.rot_w * loss_6d
                    + self.tcp_aux_loss.uv_w * loss_uv
                    + self._tcp_valid_loss_weight * loss_valid
                )
                tcp_loss = {
                    "total": total_tcp,
                    "loss_uv": loss_uv,
                    "loss_3d": loss_3d,
                    "loss_6d": loss_6d,
                    "loss_valid": loss_valid,
                    # Projection terms remain disabled in this path for simplicity.
                    # GT and predictions are now both trained in camera frame.
                    "loss_proj_gt": torch.zeros((), device=device),
                    "loss_proj_self": torch.zeros((), device=device),
                }

        # ---- Aggregate total loss ----
        tcp_weight = getattr(self, '_tcp_loss_weight', 0.0)
        # Short-circuit when tcp_weight=0 to avoid 0*NaN=NaN poisoning total_loss (geo_only mode)
        if compute_tcp_loss and tcp_weight > 0:
            total_loss = action_loss + tcp_weight * tcp_loss["total"]
        else:
            total_loss = action_loss

        future_video_loss = torch.zeros((), device=device)
        future_video_stats = {}
        if compute_action_loss and self.use_dynamics_head and self.dynamics_head is not None:
            future_video_loss, future_video_stats = self._compute_dynamics_loss(
                output, batch, camera_names
            )
            total_loss = total_loss + future_video_loss

        ret_dict = {
            "loss": total_loss,
            "loss_action_flow": action_loss,
            "loss_tcp": tcp_loss["total"],
            "loss_tcp_uv": tcp_loss["loss_uv"],
            "loss_tcp_3d": tcp_loss["loss_3d"],
            "loss_tcp_6d": tcp_loss["loss_6d"],
            "loss_tcp_valid": tcp_loss["loss_valid"],
            "loss_tcp_proj_gt": tcp_loss["loss_proj_gt"],
            "loss_tcp_proj_self": tcp_loss["loss_proj_self"],
            "loss_action_consistency": action_consistency_loss,
            "loss_action_consistency_pose": action_consistency_pose_loss,
            "loss_action_consistency_uv": action_consistency_uv_loss,
            "loss_future_video": future_video_loss,
            "loss_future_semantic": future_video_stats.get(
                "loss_future_semantic", torch.zeros((), device=device)
            ),
            "loss_future_tcp": future_video_stats.get(
                "loss_future_tcp", torch.zeros((), device=device)
            ),
        }
        for k, v in future_video_stats.items():
            if k not in ret_dict:
                ret_dict[k] = v
        if "dit_tcp_kv_pred_prob" in output:
            ret_dict["dit_tcp_kv_pred_prob"] = output["dit_tcp_kv_pred_prob"]
        if "dit_tcp_kv_used_pred" in output:
            ret_dict["dit_tcp_kv_used_pred"] = output["dit_tcp_kv_used_pred"]

        # Async NaN/Inf skip: a non-finite loss becomes 0 with zero gradient through it (torch.where backward),
        # which is what the host-syncing guard below did by hand. The guard stays available for debugging.
        total_loss = torch.where(torch.isfinite(total_loss), total_loss, torch.zeros_like(total_loss))
        ret_dict["loss"] = total_loss
        if _DEBUG_NONFINITE and (total_loss.isnan().any() or total_loss.isinf().any()):
            nan_keys = [k for k, v in ret_dict.items()
                        if isinstance(v, torch.Tensor) and (v.isnan().any() or v.isinf().any())]
            Log.warn(f"[NaN guard] loss is nan/inf — affected keys: {nan_keys}. "
                     f"Replacing with zero to skip this step.")
            ret_dict["loss"] = torch.zeros((), device=total_loss.device, requires_grad=True)
        return ret_dict

    def reset(self):
        return

    @torch.no_grad()
    def infer(self, images, depths, camera_intrinsics=None, state=None, camera_names=None, input_size=256,
            num_action_samples=1, return_intermediate_actions=False, **kwargs):
        """
        Inference method for testing the model with action generation

        Args:
            raw_image: Input RGB image
            depth: Input depth image
            input_size: Size for image preprocessing
            camera_intrinsics: Camera intrinsics matrix (optional)
            num_action_samples: Number of action sequences to generate
            return_intermediate_actions: Whether to return intermediate generation steps

        Returns:
            Dictionary containing predictions and generated actions
        """
        camera_extrinsics = kwargs.get('camera_extrinsics', None)
        outputs = self.forward(
            images, depths, camera_intrinsics=camera_intrinsics,
            state=state, camera_names=camera_names,
            _camera_extrinsics=camera_extrinsics,
        )

        result = {
            "actions": outputs['actions'].float().cpu().numpy(),
        }
        # TCP probe: pass-through TCP head outputs (no behavior change)
        for src_key, dst_key in (('tcp_uv', 'tcp_pred_uv'), ('tcp_3d', 'tcp_pred_3d'),
                                  ('tcp_6d', 'tcp_pred_6d'), ('tcp_valid', 'tcp_valid_pred'),
                                  ('tcp_valid_target', 'tcp_valid_target')):
            if src_key in outputs:
                v = outputs[src_key]
                if isinstance(v, dict):
                    result[dst_key] = {k: vv.detach().float().cpu().numpy() if hasattr(vv, 'detach') else vv for k, vv in v.items()}
                elif hasattr(v, 'detach'):
                    result[dst_key] = v.detach().float().cpu().numpy()
                else:
                    result[dst_key] = v
        return result
