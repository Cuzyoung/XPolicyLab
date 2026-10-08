import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sapolicy.models.action_head.unet import FeaturePyramidModule, _make_noise_scheduler
from sapolicy.models.utils.pos_embed import (
    SinusoidalPositionEmbeddings,
    SinusoidalSequencePosEnc,
    add_pos_embed,
)


class TransformerHead(nn.Module):
    def __init__(
        self,
        obs_in_channels=256,
        obs_pyramid_channels=256,
        sequence_length=10,
        obs_hist_length=1,
        embed_dim=256,
        num_heads=4,
        num_layers=8,
        num_inference_steps=10,
        exploration_noise=False,
        action_orn_mode='6d', # '6d', 'quat', 'euler'
        num_cameras=2,
        use_state=False,
        state_dim=7,
        encoder_type='mlp',  # 'mlp' or 'transformer'
        fpn_last_scale_only=False,
        num_spatial_patches_per_cam=25,
        action_dim=None,  # explicit override; None -> derive from action_orn_mode (single arm)
    ):
        """
        Encoder-Decoder Transformer Flow Matching Head.

        encoder_type='mlp' (default, backward-compatible):
            - FPN output pooled to single vector per camera → MLP encoder → 2 memory tokens
            - Matches checkpoints trained with the original architecture

        encoder_type='transformer':
            - FPN output preserved as spatial tokens [B, H*W, embed_dim] per camera
            - TransformerEncoder refines memory via self-attention
            - Decoder cross-attends over ~512+ spatial memory tokens
        """
        super().__init__()
        # Rebind all Hydra/OmegaConf values to native Python types
        sequence_length = self.sequence_length = int(sequence_length)
        embed_dim = self.embed_dim = int(embed_dim)
        num_inference_steps = self.num_inference_steps = int(num_inference_steps)
        state_dim = self.state_dim = int(state_dim)
        self.exploration_noise = exploration_noise
        self.action_orn_mode = str(action_orn_mode)
        num_cameras = self.num_cameras = int(num_cameras)
        self.use_state = bool(use_state) if isinstance(use_state, bool) else bool(int(use_state))
        obs_hist_length = self.obs_hist_length = int(obs_hist_length)
        obs_in_channels = int(obs_in_channels)
        obs_pyramid_channels = int(obs_pyramid_channels)
        num_heads = self.num_heads = int(num_heads)
        num_layers = int(num_layers)
        self.encoder_type = str(encoder_type)
        self.fpn_last_scale_only = bool(fpn_last_scale_only)
        self.num_spatial_patches_per_cam = int(num_spatial_patches_per_cam)
        self._per_camera_spatial_tokens = {}  # store for cross-view contrastive

        # ===== Time embedding (continuous t) =====
        self.time_embd = SinusoidalPositionEmbeddings(embed_dim)

        # ===== Feature Pyramid Network for multi-scale features =====
        self.obs_fpn_modules = nn.ModuleList([
            FeaturePyramidModule(obs_in_channels, obs_pyramid_channels, spatial_attention=False) for _ in range(4)
        ])

        # The orn-mode table gives the SINGLE-arm width. Bimanual data (RoboTwin)
        # passes action_dim explicitly, e.g. 20 = 2 x (pos3 + rot6d + gripper1).
        if action_dim is None:
            action_dim = 10 if action_orn_mode == '6d' else 8 if action_orn_mode == 'quat' else 7 if action_orn_mode == 'euler' else 7 if action_orn_mode == 'rotvec' else None
        action_dim = int(action_dim)
        self.action_dim = action_dim

        self.action_embed = nn.Linear(action_dim, embed_dim)

        # ===== Action positional encoding =====
        self.pos_enc = nn.Parameter(torch.zeros(1, self.sequence_length, embed_dim))

        if self.encoder_type == 'mlp':
            # ----- Legacy bottleneck architecture -----
            self.cond_length = 1 + 1  # time token + pooled obs token

            # Global condition vector from pooled FPN + state features
            global_cond_dim = obs_pyramid_channels * num_cameras * obs_hist_length
            if self.use_state:
                global_cond_dim += state_dim * obs_hist_length

            self.obs_feature_flatten = nn.Sequential(
                nn.Flatten(),
            )
            self.obs_feature_proj2vec = nn.Sequential(
                nn.Linear(global_cond_dim, embed_dim),
            )

            # MLP encoder
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.Mish(),
                nn.Linear(4 * embed_dim, embed_dim),
            )


        elif self.encoder_type == 'transformer':
            # ----- New spatial cross-attention architecture -----
            # Spatial token projection (FPN channels -> embed_dim)
            self.spatial_proj = nn.Sequential(
                nn.Linear(obs_pyramid_channels, embed_dim),
            )

            # State projection (if used)
            if self.use_state:
                self.state_proj = nn.Sequential(
                    nn.Linear(state_dim * obs_hist_length, embed_dim),
                )

            # Spatial positional encoding (fixed learned, supports up to 32x32 patch grid)
            # max_spatial_tokens = 32 * 32
            # self.spatial_pos_enc = nn.Parameter(torch.zeros(1, max_spatial_tokens, embed_dim))

            # Time token positional encoding
            self.time_pos_enc = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Condition token positional encoding for memory tokens (spatial/state).
            cond_length = self.num_cameras * self.num_spatial_patches_per_cam
            if self.use_state:
                cond_length += 1
            self.cond_pos_enc = nn.Parameter(torch.zeros(1, cond_length, embed_dim))

            # Transformer Encoder (self-attention over memory tokens)
            enc_layers = max(1, num_layers // 4)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}. Choose 'mlp' or 'transformer'.")

        # ===== Transformer Decoder (cross-attention to memory) =====
        dec_layers = max(1, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.after_norm_decoder = nn.LayerNorm(embed_dim)

        # ===== Output projection to flow (velocity field) =====
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, action_dim),
        )

        # init
        self._init_weights()

    # ---------- Initialization ----------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                weight_names = [
                    'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
                for name in weight_names:
                    weight = getattr(m, name)
                    if weight is not None:
                        torch.nn.init.normal_(weight, mean=0.0, std=0.02)

                bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
                for name in bias_names:
                    bias = getattr(m, name)
                    if bias is not None:
                        torch.nn.init.zeros_(bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.zeros_(m.bias)
                torch.nn.init.ones_(m.weight)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _validate_condition_views(self, obs_features):
        if not isinstance(obs_features, dict):
            raise ValueError(f"obs_features must be a dict, got {type(obs_features)}")

        obs_views = len(obs_features)
        if obs_views != self.num_cameras:
            raise ValueError(
                f"Action head expected num_cameras={self.num_cameras}, "
                f"but got {obs_views} camera feature streams. "
                "Set action_cfg.num_cameras to match dataset camera_names."
            )

    # ---------- Forward ----------
    def forward(self, obs_features, patch_h, patch_w, actions=None, state=None, num_samples=None,
                camera_intrinsics=None, **kwargs):
        """
        obs_features: dict of {camera_name: list of (patch_tokens[B,T,N,C], cls_token) per layer}
        patch_h, patch_w: spatial dimensions of patch grid
        actions: [B, T, D] ground truth actions (training only)
        camera_intrinsics: Dict[str, Tensor] of [B, 3, 3] per camera (for geometric pos enc)
        """
        self._validate_condition_views(obs_features)

        if self.encoder_type == 'mlp':
            cond_tokens = self._build_memory_mlp(obs_features, patch_h, patch_w, state)
        else:
            cond_tokens = self._build_memory_transformer(obs_features, patch_h, patch_w, state,
                                                          camera_intrinsics=camera_intrinsics)

        if self.training and actions is not None:
            return self._forward_training(actions, cond_tokens)
        else:
            num_samples = int(num_samples or 1)
            return self._forward_inference(cond_tokens, cond_tokens.device, num_samples)

    def _build_memory_mlp(self, obs_features, patch_h, patch_w, state):
        """Legacy bottleneck: pool FPN → single vector per camera → MLP encoder."""
        cond_features = []
        for camera_name, camera_features in obs_features.items():
            B = camera_features[0][0].shape[0]
            fpn_features = []
            for i, feature_tuple in enumerate(camera_features):
                patch_features = feature_tuple[0]  # [B, T, N, C]
                B, T, N, C = patch_features.shape
                patch_features = patch_features.reshape(B * T, N, C)
                N, C = patch_features.shape[1], patch_features.shape[2]
                spatial_features = patch_features.permute(0, 2, 1).reshape(B * T, C, patch_h, patch_w)
                fpn_features.append(spatial_features)

            pyramid_outputs = []
            x = None
            for i in range(len(fpn_features) - 1, -1, -1):
                x = self.obs_fpn_modules[i](fpn_features[i], x)
                pyramid_outputs.append(x)

            feature_pool = F.adaptive_avg_pool2d(pyramid_outputs[-1], output_size=1)
            feature_pool = feature_pool.view(feature_pool.size(0), -1)
            feature_pool = feature_pool.reshape(B, T, -1)
            cond_features.append(self.obs_feature_flatten(feature_pool))

        if state is not None and self.use_state:
            cond_features.append(state.reshape(state.shape[0], -1))

        cond_features = torch.concat(cond_features, dim=-1)  # [B, cond_dim]
        cond_features = self.obs_feature_proj2vec(cond_features).unsqueeze(1)  # [B, 1, embed_dim]

        return cond_features

    def _build_memory_transformer(self, obs_features, patch_h, patch_w, state,
                                    camera_intrinsics=None):
        """New spatial architecture: keep spatial tokens → TransformerEncoder."""
        spatial_tokens_list = []
        self._per_camera_spatial_tokens = {}

        for cam_idx, (camera_name, camera_features) in enumerate(obs_features.items()):
            B = camera_features[0][0].shape[0]
            fpn_features = []
            for i, feature_tuple in enumerate(camera_features):
                patch_features = feature_tuple[0]  # [B, T, N, C]
                B, T, N, C = patch_features.shape
                patch_features = patch_features.reshape(B * T, N, C)
                N, C = patch_features.shape[1], patch_features.shape[2]
                spatial_features = patch_features.permute(0, 2, 1).reshape(B * T, C, patch_h, patch_w)
                fpn_features.append(spatial_features)

            num_scales = len(fpn_features)
            if num_scales == 1:
                # Single scale: project obs_in_channels → obs_pyramid_channels via FPN, then to embed_dim.
                fpn_out = self.obs_fpn_modules[-1](fpn_features[0], None)  # [B*T, obs_pyramid_channels, H, W]
                fpn_out = fpn_out.reshape(B, T, fpn_out.shape[1], patch_h, patch_w)
                fpn_out = fpn_out[:, -1]  # [B, C, H, W]
                spatial = fpn_out.flatten(2).permute(0, 2, 1)  # [B, H'*W', obs_pyramid_channels]
                spatial = self.spatial_proj(spatial)  # [B, H'*W', embed_dim]
            elif self.fpn_last_scale_only:
                # Only process the deepest backbone layer — no cascade from shallow scales
                fpn_out = self.obs_fpn_modules[num_scales - 1](fpn_features[-1], None)
                fpn_out = fpn_out.reshape(B, T, fpn_out.shape[1], patch_h, patch_w)
                fpn_out = fpn_out[:, -1]  # Take last frame: [B, C, H, W]
                spatial = fpn_out.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
                spatial = self.spatial_proj(spatial)  # [B, H*W, embed_dim]
            else:
                # Full top-down FPN cascade (coarsest → finest)
                pyramid_outputs = []
                x = None
                for i in range(num_scales - 1, -1, -1):
                    x = self.obs_fpn_modules[i](fpn_features[i], x)
                    pyramid_outputs.append(x)
                fpn_out = pyramid_outputs[-1]  # [B*T, C, patch_h, patch_w]
                fpn_out = fpn_out.reshape(B, T, fpn_out.shape[1], patch_h, patch_w)
                fpn_out = fpn_out[:, -1]  # Take last frame: [B, C, H, W]
                spatial = fpn_out.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
                spatial = self.spatial_proj(spatial)  # [B, H*W, embed_dim]

            HW = spatial.shape[1]
            spatial = add_pos_embed(spatial, patch_w, patch_h)

            self._per_camera_spatial_tokens[camera_name] = spatial
            spatial_tokens_list.append(spatial)

        cond_tokens = torch.cat(spatial_tokens_list, dim=1)

        if state is not None and self.use_state:
            state_flat = state.reshape(state.shape[0], -1)
            state_token = self.state_proj(state_flat)
            cond_tokens = torch.cat([cond_tokens, state_token.unsqueeze(1)], dim=1)

        return cond_tokens

    @torch.no_grad()
    def sample_trajectory(
        self,
        cond_features,
        device=None,
        num_samples=1,
        return_intermediate=False,
    ):
        raise NotImplementedError("This method is not implemented for TransformerHead")

    def prepare_actions(self, actions_9d):
        pos = actions_9d[..., :3]
        gripper = actions_9d[..., -1:]
        R_mat = actions_9d[..., 3:-1].reshape(*actions_9d.shape[:-1], 3, 3)

        rot6d = R_mat[..., :2].reshape(*actions_9d.shape[:-1], 6)
        actions_6d = torch.cat([pos, rot6d, gripper], dim=-1)
        return actions_6d

    # ============================================================
    # Training / Inference entry points (overridden by subclasses)
    # ============================================================
    def _forward_training(self, actions, cond_features):
        raise NotImplementedError("This method is not implemented for TransformerHead")

    def _forward_inference(self, cond_features, device, num_samples: int):
        return self.sample_trajectory(cond_features, device=device, num_samples=num_samples)

    # ---------- Core: Encoder-Decoder Transformer predicts flow ----------
    def _predict_transformer(self, actions, time_emb, cond_features):
        """
        actions:       [B, T, action_dim]
        time_emb:      [B, embed_dim]
        cond_features: [B, N, embed_dim]  (memory tokens from _build_memory_*)
        return:        [B, T, action_dim]
        """
        B, T, _ = actions.shape

        t_token = time_emb.unsqueeze(1)  # [B, 1, embed_dim]

        if self.encoder_type == 'mlp':
            # Legacy: cond_features is [B, 1, embed_dim], build [time, cond] → MLP
            cond_seq = torch.cat([t_token, cond_features], dim=1)  # [B, 2, embed_dim]
            pos_cond = self.pos_enc[:, :self.cond_length, :]
            cond_seq = cond_seq + pos_cond
            memory = self.encoder(cond_seq)  # [B, 2, embed_dim]
        else:
            # Transformer: cond_features is [B, N_spatial, embed_dim]
            t_token = t_token + self.time_pos_enc  # [B, 1, embed_dim]
            cond_len = cond_features.shape[1]
            if cond_len != self.cond_pos_enc.shape[1]:
                raise ValueError(
                    f"cond_features length {cond_len} != cond_pos_enc {self.cond_pos_enc.shape[1]}; "
                    f"check num_cameras={self.num_cameras}, "
                    f"num_spatial_patches_per_cam={self.num_spatial_patches_per_cam}, "
                    f"use_state={self.use_state}."
                )
            cond_features = cond_features + self.cond_pos_enc
            memory = torch.cat([t_token, cond_features], dim=1)  # [B, 1+N, embed_dim]
            memory = self.encoder(memory)  # [B, 1+N, embed_dim]

        # Action tokens with positional encoding
        x = self.action_embed(actions)  # [B, T, embed_dim]
        x = x + self.pos_enc[:, :T, :]  # [B, T, embed_dim]

        # Decoder: cross-attend to memory
        x = self.decoder(tgt=x, memory=memory)  # [B, T, embed_dim]

        # Output to flow
        x = self.after_norm_decoder(x)
        flow = self.output_proj(x)  # [B, T, action_dim]
        return flow


class TransformerFlowMatchingHead(TransformerHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_training(self, actions, cond_features, preset_t=None, preset_noise=None, **kwargs):
        B, T, D = actions.shape
        device = actions.device

        if self.action_orn_mode == '6d' and actions.shape[-1] == 13:
            actions = self.prepare_actions(actions)

        # R68: Check for preset t/noise from action consistency loss
        if preset_t is None:
            preset_t = getattr(self, '_preset_t', None)
        if preset_noise is None:
            preset_noise = getattr(self, '_preset_noise', None)

        # 1. t ~ Beta(1.5, 1.0) for legacy MLP, Uniform for transformer
        if preset_t is not None:
            t = preset_t
        elif self.encoder_type == 'mlp':
            t = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device) * 0.999 + 0.001
        else:
            t = torch.rand(B, device=device) * 0.998 + 0.001
        t_expand = t.view(B, 1, 1)                               # [B,1,1]

        # 2. Noise from standard Gaussian
        noise = preset_noise if preset_noise is not None else torch.randn_like(actions)  # [B,T,D]

        # 3. Interpolate actions and noise
        a_t = (1.0 - t_expand) * actions + t_expand * noise      # [B,T,D]

        # 4. Target flow (Pi0 objective)
        target_flow = noise - actions

        # 5. Predict flow field
        time_emb = self.time_embd(t)                              # [B, time_dim]
        pred_flow = self._predict_transformer(a_t, time_emb, cond_features)  # [B,T,D]

        # Return sampled t, noise for exact clean-action reconstruction and consistency loss.
        return pred_flow, target_flow, a_t, None, t, noise

    @torch.no_grad()
    def sample_trajectory(
        self,
        cond_features,           # [B, N, embed_dim]
        device=None,
        num_samples=1,
        return_intermediate=False,
    ):
        B = cond_features.shape[0]
        T = self.sequence_length
        D = self.action_dim
        S = num_samples
        debug_nonfinite = os.environ.get("SAPOLICY_DEBUG_FLOW_NONFINITE", "0") == "1"
        if debug_nonfinite:
            self._debug_infer_call = getattr(self, "_debug_infer_call", 0) + 1
            debug_call = self._debug_infer_call

        # 1. initial actions from Gaussian
        actions = torch.randn(B, S, T, D, device=device, dtype=cond_features.dtype)
        intermediates = []

        # Expand cond_features for num_samples > 1: [B, N, E] -> [B*S, N, E]
        if S > 1:
            cond_expanded = cond_features.unsqueeze(1).expand(
                -1, S, -1, -1
            ).reshape(B * S, *cond_features.shape[1:])
        else:
            cond_expanded = cond_features

        # 2. time from 1 -> 0
        dt = -1.0 / self.num_inference_steps
        t_curr = 1.0
        step_idx = 0
        while t_curr >= -dt / 2:
            a_flat = actions.view(B * S, T, D)

            t_vec = torch.full((B * S,), t_curr, device=device)
            time_emb = self.time_embd(t_vec)

            v_t = self._predict_transformer(a_flat, time_emb, cond_expanded)  # [B*S,T,D]
            if debug_nonfinite and not torch.isfinite(v_t).all():
                bad = int((~torch.isfinite(v_t)).sum().item())
                cond_finite = bool(torch.isfinite(cond_expanded).all())
                action_finite = bool(torch.isfinite(a_flat).all())
                cond_absmax = float(torch.nan_to_num(cond_expanded).abs().max().item())
                action_absmax = float(torch.nan_to_num(a_flat).abs().max().item())
                velocity_absmax = float(torch.nan_to_num(v_t).abs().max().item())
                print(
                    "[flow-debug] first non-finite velocity "
                    f"call={debug_call} step={step_idx}/{self.num_inference_steps} "
                    f"t={t_curr:.6f} bad={bad} "
                    f"cond_finite={cond_finite} cond_absmax={cond_absmax:.6g} "
                    f"action_finite={action_finite} action_absmax={action_absmax:.6g} "
                    f"velocity_absmax={velocity_absmax:.6g}",
                    flush=True,
                )
            v_t = v_t.view(B, S, T, D)

            # Euler update: a_{t-dt} = a_t - dt * v(a_t, t)
            actions = actions + dt * v_t
            if debug_nonfinite and not torch.isfinite(actions).all():
                bad = int((~torch.isfinite(actions)).sum().item())
                print(
                    "[flow-debug] non-finite Euler state "
                    f"call={debug_call} step={step_idx}/{self.num_inference_steps} "
                    f"t={t_curr:.6f} bad={bad}",
                    flush=True,
                )

            if return_intermediate and (t_curr < -dt / 2):
                intermediates.append(actions.clone())

            t_curr += dt
            step_idx += 1

        if num_samples == 1:
            actions = actions.squeeze(1)  # [B,T,D]

        if return_intermediate:
            return actions, intermediates
        return actions


class TransformerDiffusionHead(TransformerHead):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # NOTE: SinusoidalPositionEmbeddings works for both continuous [0,1] and discrete [0,1,2,...,T] timesteps
        # WARNING: Make sure noise scheduler parameters match your training setup
        self.noise_scheduler = _make_noise_scheduler(
            "DDIM",
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type='epsilon',
        )

    def _forward_training(self, actions, cond_features):
        # Forward diffusion.
        # Sample noise to add to the actions.
        B, T, D = actions.shape
        eps = torch.randn_like(actions)

        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,),
            device=actions.device,
        ).long()

        # Add noise to the clean actions according to the noise magnitude at each timestep.
        noisy_actions = self.noise_scheduler.add_noise(actions, eps, timesteps)

        time_emb = self.time_embd(timesteps)  # [B, embed_dim]

        # Run the denoising network (predict noise or denoised actions).
        pred = self._predict_transformer(noisy_actions, time_emb, cond_features)  # [B, T, D]

        target = eps

        return pred, target, noisy_actions, None

    @torch.no_grad()
    def sample_trajectory(
        self,
        cond_features,           # [B, N, embed_dim]
        device=None,
        num_samples=1,
        return_intermediate=False,
    ):
        device = device or next(iter(self.parameters())).device
        dtype = cond_features.dtype

        B = cond_features.shape[0]
        T = self.sequence_length
        D = self.action_dim
        S = num_samples

        # Sample prior noise
        actions = torch.randn(
            size=(B, S, T, D),
            dtype=dtype,
            device=device,
        )

        # Set timesteps for inference
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Expand cond_features for num_samples > 1: [B, N, E] -> [B*S, N, E]
        if S > 1:
            cond_expanded = cond_features.unsqueeze(1).expand(
                -1, S, -1, -1
            ).reshape(B * S, *cond_features.shape[1:])
        else:
            cond_expanded = cond_features

        intermediates = [] if return_intermediate else None

        for t in self.noise_scheduler.timesteps:
            a_flat = actions.view(B * S, T, D)
            t_vec = torch.full((B * S,), t.long(), device=device)
            time_emb = self.time_embd(t_vec)
            # Predict model output (noise or denoised actions)
            model_output = self._predict_transformer(a_flat, time_emb, cond_expanded)

            # Compute previous sample: x_t -> x_t-1
            a_flat = self.noise_scheduler.step(
                model_output, t, a_flat
            ).prev_sample
            actions = a_flat.view(B, S, T, D)

            if return_intermediate:
                intermediates.append(actions.clone())

        if num_samples == 1:
            actions = actions.squeeze(1)  # [B, T, D]

        if return_intermediate:
            return actions, intermediates
        return actions
