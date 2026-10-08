from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from sapolicy.models.utils.pos_embed import SinusoidalPositionEmbeddings


def _make_noise_scheduler(
    name: str, **kwargs: dict
) -> Union[DDPMScheduler, DDIMScheduler]:
    """Factory for noise scheduler instances. All kwargs are passed to the scheduler."""
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    if name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type {name}")


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention


class FeaturePyramidModule(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_attention=True):
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if spatial_attention:
            self.spatial_attention = SpatialAttention(out_channels)
        else:
            self.spatial_attention = None

    def forward(self, x, higher_res_feat=None):
        x = self.lateral_conv(x)
        if higher_res_feat is not None:
            x = F.interpolate(x, size=higher_res_feat.shape[2:], mode='bilinear', align_corners=False)
            x = x + higher_res_feat
        if self.spatial_attention is not None:
            x = self.spatial_attention(x)
        x = self.output_conv(x)
        return x


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
        

class UNetHead(nn.Module):
    def __init__(
        self,
        obs_in_channels=256,
        obs_pyramid_channels=256,
        enhanced_dim=256,
        sequence_length=10,
        obs_hist_length=1,
        embed_dim=256,
        down_dims=[512, 1024, 2048],
        kernel_size=5,
        n_groups=8,
        num_inference_steps=100,
        exploration_noise=False,
        action_orn_mode='6d',  # '6d', 'quat', 'euler'
        num_cameras=2,
        use_state=False,
        state_dim=7,
        cond_predict_scale=True,
        use_tcp_features=False,
        fpn_last_scale_only=False,
    ):
        """
        Base UNet Head with FiLM conditioning.
        - Uses ConditionalResidualBlock1D with FiLM-style modulation
        - Separates global conditioning (time + features) from action input
        - More sophisticated architecture with proper residual connections
        """
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.obs_hist_length = int(obs_hist_length)
        self.embed_dim = int(embed_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.exploration_noise = exploration_noise
        self.action_orn_mode = action_orn_mode
        self.num_cameras = int(num_cameras)
        self.use_state = use_state
        self.use_tcp_features = bool(use_tcp_features)
        self.state_dim = state_dim
        self.down_dims = down_dims
        self.fpn_last_scale_only = bool(fpn_last_scale_only)

        # ===== Time embedding (continuous t) - Simplified =====
        self.time_embd = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # ===== Condition features (from TCP decoder's enhanced features) =====
        if use_tcp_features:
            self.tcp_feature_proj2vec = nn.Sequential(
                nn.Conv2d(enhanced_dim, embed_dim, kernel_size=1),
                nn.Mish(),
                nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
                nn.Flatten(),             # [B, C]
                nn.Linear(embed_dim, embed_dim),
            )

        # Feature Pyramid Network for multi-scale features
        self.obs_fpn_modules = nn.ModuleList([
            FeaturePyramidModule(obs_in_channels, obs_pyramid_channels, spatial_attention=False) for _ in range(4)
        ])

        self.obs_feature_proj2vec = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Flatten(),             # [B, C]
        )
        obs_feature_dim = obs_pyramid_channels

        action_dim = 10 if action_orn_mode == '6d' else 8 if action_orn_mode == 'quat' else 7 if action_orn_mode == 'euler' else 7 if action_orn_mode == 'rotvec' else None
        self.action_dim = action_dim

        # ===== Calculate global condition dimension =====
        # Global condition: time_emb + pooled/softmax FPN features + optional TCP + optional state
        global_cond_dim = embed_dim + obs_feature_dim * self.num_cameras * self.obs_hist_length
        if use_tcp_features:
            global_cond_dim += embed_dim * self.num_cameras
        if use_state:
            global_cond_dim += (state_dim * self.obs_hist_length)

        # ===== UNet Architecture with Conditional Residual Blocks =====
        unet_input_dim = action_dim
        all_dims = [unet_input_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Bottleneck
        mid_dim = all_dims[-1]
        self.mid_dim = mid_dim
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=global_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=global_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        # Downsampling path
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=global_cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=global_cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # Upsampling path
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=global_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=global_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        # Final output convolution
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, action_dim, 1),
        )

        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _validate_condition_views(self, obs_features, enhanced_tcp_features):
        if not isinstance(obs_features, dict):
            raise ValueError(f"obs_features must be a dict, got {type(obs_features)}")

        obs_views = len(obs_features)
        if obs_views != self.num_cameras:
            raise ValueError(
                f"Action head expected num_cameras={self.num_cameras}, "
                f"but got {obs_views} camera feature streams. "
                "Set action_cfg.num_cameras to match dataset camera_names."
            )

        if self.use_tcp_features:
            tcp_views = len(enhanced_tcp_features) if isinstance(enhanced_tcp_features, dict) else 0
            if tcp_views != self.num_cameras:
                raise ValueError(
                    f"TCP conditioning expected {self.num_cameras} views, "
                    f"but got {tcp_views}. Ensure TCP outputs exist for all configured cameras."
                )

    def forward(self, obs_features, enhanced_tcp_features, patch_h, patch_w, actions=None, state=None, num_samples=None):
        """
        obs_features: dict of {camera_name: list of (patch_tokens[B,T,N,C], cls_token) per layer}
        enhanced_tcp_features: Dict of multi-scale feature lists from TCP head or None
        actions: [B, T, D]
        num_samples: int

        return:
        training: (predicted_flow, target_flow, a_t, action_loss_or_none, sampled_t)
        inference: generated action sequence
        """
        self._validate_condition_views(obs_features, enhanced_tcp_features)
        cond_features = []
        for camera_name, camera_features in obs_features.items():
            B = camera_features[0][0].shape[0]
            fpn_features = []
            for i, feature_tuple in enumerate(camera_features):
                patch_features = feature_tuple[0]  # [B, T, N, C]
                if patch_features.ndim == 4:
                    B, T_obs, N, C = patch_features.shape
                    patch_features = patch_features.reshape(B * T_obs, N, C)
                else:
                    N, C = patch_features.shape[1], patch_features.shape[2]

                spatial_features = patch_features.permute(0, 2, 1).reshape(-1, C, patch_h, patch_w)
                fpn_features.append(spatial_features)

            if self.fpn_last_scale_only:
                # Only process the deepest backbone layer (index 3)
                # through its FPN module — no cascade from shallow scales
                x = self.obs_fpn_modules[len(fpn_features) - 1](fpn_features[-1], None)
                pooled = self.obs_feature_proj2vec(x)
            else:
                # Original: cascade all scales (coarsest→finest)
                pyramid_outputs = []
                x = None
                for i in range(len(fpn_features) - 1, -1, -1):
                    x = self.obs_fpn_modules[i](fpn_features[i], x)
                    pyramid_outputs.append(x)
                pooled = self.obs_feature_proj2vec(pyramid_outputs[-1])  # [B*T, obs_pyramid_channels]
            pooled = pooled.reshape(B, -1)  # collapse T into feature dim
            cond_features.append(pooled)

        if enhanced_tcp_features is not None and self.use_tcp_features:
            for camera_name, camera_features in enhanced_tcp_features.items():
                cond_features.append(self.tcp_feature_proj2vec(camera_features))

        if state is not None:
            cond_features.append(state.reshape(state.shape[0], -1))

        cond_features = torch.concat(cond_features, dim=-1)  # [B, cond_dim]

        if self.training and actions is not None:
            return self._forward_training(actions, cond_features)
        else:
            num_samples = int(num_samples or 1)
            return self._forward_inference(cond_features, cond_features.device, num_samples)

    @torch.no_grad()
    def sample_trajectory(
        self,
        cond_features,
        device=None,
        num_samples=1,
        return_intermediate=False,
    ):
        raise NotImplementedError("This method is not implemented for UNetHead")

    def prepare_actions(self, actions_9d):
        """Convert 9D actions (pos + 3x3 rotation matrix + gripper) to appropriate format"""
        pos = actions_9d[..., :3]
        gripper = actions_9d[..., -1:]
        R_mat = actions_9d[..., 3:-1].reshape(*actions_9d.shape[:-1], 3, 3)

        rot6d = R_mat[..., :2].reshape(*actions_9d.shape[:-1], 6)
        actions_6d = torch.cat([pos, rot6d, gripper], dim=-1)
        return actions_6d

    def _forward_training(self, actions, cond_features):
        raise NotImplementedError("This method is not implemented for UNetHead")

    def _forward_inference(self, cond_features, device, num_samples: int):
        return self.sample_trajectory(cond_features, device=device, num_samples=num_samples)

    def _predict_flow_unet(self, actions, time_emb, cond_features, return_feat=False):
        """
        Core UNet flow prediction with FiLM-style conditioning.
        
        actions:       [B, T, action_dim]
        time_emb:      [B, embed_dim]
        cond_features: [B, cond_dim]
        return:        [B, T, action_dim]
        """
        B, T, _ = actions.shape

        # Flatten condition features [B, num_cond_tokens * embed_dim]
        cond_flat = cond_features.flatten(1)  # [B, num_cond_tokens * embed_dim]

        # Build global condition: time_emb + all visual/state features
        global_feature = torch.cat([time_emb, cond_flat], dim=-1)  # [B, cond_dim]

        # Rearrange actions for Conv1d: [B, action_dim, T]
        x = einops.rearrange(actions, 'b t d -> b d t')

        # Downsampling path with skip connections
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        feat = x.mean(dim=-1)

        # Upsampling path with skip connections
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # Final convolution
        x = self.final_conv(x)

        # Rearrange back: [B, T, action_dim]
        flow = einops.rearrange(x, 'b d t -> b t d')

        if return_feat: 
            return flow, feat
        return flow

    @torch.no_grad()
    def generate_actions(self, enhanced_tcp_features, num_samples=1, return_intermediate=False):
        self.eval()
        device = enhanced_tcp_features.device
        cond_features = self.tcp_feature_proj2vec(enhanced_tcp_features)
        return self.sample_trajectory(
            cond_features,
            device=device,
            num_samples=num_samples,
            return_intermediate=return_intermediate,
        )


class UNetFlowMatchingHead(UNetHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward_training(self, actions, cond_features):
        B, T, D = actions.shape
        device = actions.device

        if self.action_orn_mode == '6d' and actions.shape[-1] == 13:
            actions = self.prepare_actions(actions)

        # 1. t ~ Beta
        t = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(device) * 0.999 + 0.001  # [B]
        t_expand = t.view(B, 1, 1)

        # 2. Noise from standard Gaussian
        noise = torch.randn_like(actions)

        # 3. Interpolate actions and noise
        a_t = (1.0 - t_expand) * actions + t_expand * noise

        # 4. Target flow (Pi0 objective)
        target_flow = noise - actions

        # 5. Predict flow field using simplified time embedding
        time_emb = self.time_embd(t)  # [B, embed_dim]
        pred_flow = self._predict_flow_unet(a_t, time_emb, cond_features)

        # Return sampled t for exact clean-action reconstruction in downstream consistency loss.
        return pred_flow, target_flow, a_t, None, t

    @torch.no_grad()
    def sample_trajectory(
        self,
        cond_features,
        device=None,
        num_samples=1,
        return_intermediate=False,
    ):
        self.eval()
        B = cond_features.shape[0]
        T = self.sequence_length
        D = self.action_dim
        S = num_samples

        # 1. initial actions from Gaussian
        actions = torch.randn(B, S, T, D, device=device)
        intermediates = []

        # 2. time from 1 → 0
        dt = -1.0 / self.num_inference_steps
        t_curr = 1.0
        while t_curr >= -dt / 2:
            a_flat = actions.view(B * S, T, D)

            t_vec = torch.full((B * S,), t_curr, device=device)
            time_emb = self.time_embd(t_vec)  # [B*S, embed_dim]

            # Expand cond_features from [B, ...] to [B*S, ...] for num_samples > 1
            if S > 1:
                cond_expanded = cond_features.unsqueeze(1).expand(-1, S, *cond_features.shape[1:]).reshape(B * S, *cond_features.shape[1:])
            else:
                cond_expanded = cond_features
            v_t = self._predict_flow_unet(a_flat, time_emb, cond_expanded)  # [B*S,T,D]
            v_t = v_t.view(B, S, T, D)

            # Euler update: a_{t-dt} = a_t - dt * v(a_t, t)
            actions = actions + dt * v_t

            if return_intermediate and (t_curr < -dt / 2):
                intermediates.append(actions.clone())

            t_curr += dt

        if num_samples == 1:
            actions = actions.squeeze(1)  # [B,T,D]

        if return_intermediate:
            return actions, intermediates
        return actions


class UNetAdaFlowHead(UNetHead):
    def __init__(self, *args, sampling_method="euler", eta=0.1, pos_emb_scale=1000.0, freeze_rf=False, **kwargs):
        """
        AdaFlow UNet Head with variance estimation and adaptive sampling.
        
        Args:
            sampling_method: "euler" or "adaptive"
            eta: parameter for adaptive step size computation
            pos_emb_scale: scale factor for time positional embedding
            freeze_rf: whether to freeze the rectified flow during training
        """
        super().__init__(*args, **kwargs)
        
        self.sampling_method = sampling_method
        self.eta = eta
        self.pos_emb_scale = pos_emb_scale
        self.freeze_rf = freeze_rf
        
        # Override final conv to output both velocity and log variance
        start_dim = self.down_dims[0]
        kernel_size = 3
        n_groups = 8
        
        self.var_est = nn.Sequential(
            nn.Linear(self.mid_dim, self.embed_dim),
            nn.SiLU(), 
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(), 
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(), 
            nn.Linear(self.embed_dim, 1)
        )

    def _predict_velocity_and_variance(self, actions, time_emb, cond_features, freeze_rf=True):
        """
        Core UNet prediction with velocity and variance estimation.
        
        actions:       [B, T, action_dim]
        time_emb:      [B, embed_dim]
        cond_features: [B, num_cond_tokens, embed_dim]
        return:        velocity [B, T, action_dim], log_sqrt_var [B]
        """
        
        if freeze_rf: 
            with torch.no_grad(): 
                velocity, feat = super()._predict_flow_unet(actions, time_emb, cond_features, return_feat=True)
                velocity = velocity.detach()
                feat = feat.detach()
        else:
            velocity, feat = super()._predict_flow_unet(actions, time_emb, cond_features, return_feat=True)

        log_sqrt_var = self.var_est(feat)
        log_sqrt_var = log_sqrt_var.squeeze()

        return velocity, log_sqrt_var

    def _forward_training(self, actions, cond_features):
        B, T, D = actions.shape
        device = actions.device

        if self.action_orn_mode == '6d' and actions.shape[-1] == 13:
            actions = self.prepare_actions(actions)

        # Sample noise and time
        noise = torch.randn_like(actions)
        t = torch.rand(B, 1, 1).to(device)  # [B, 1, 1]
        
        # Interpolate: z_t = t * actions + (1-t) * noise
        z_t = t * actions + (1.0 - t) * noise
        
        # Target velocity: actions - noise
        target_velocity = actions - noise
        
        # Predict velocity and variance
        t_flat = t.squeeze()  # [B]
        time_emb = self.time_embd(t_flat * self.pos_emb_scale)
        
        velocity_pred, log_sqrt_var_pred = self._predict_velocity_and_variance(
            z_t, time_emb, cond_features, freeze_rf=self.freeze_rf
        )
        
        # Compute variance-weighted loss
        # loss = 1/(2*exp(2*log_sqrt_var)) * ||target - pred||^2 + log_sqrt_var
        error = (target_velocity - velocity_pred).pow(2).sum(dim=(-1, -2))  # [B]
        
        if self.freeze_rf:
            error = error.detach()
        
        loss_per_sample = 1.0 / (2.0 * torch.exp(2 * log_sqrt_var_pred)) * error + log_sqrt_var_pred
        
        # Return in format compatible with existing training loop
        # Store loss in a way that can be extracted
        return velocity_pred, target_velocity, z_t, loss_per_sample.mean()

    @torch.no_grad()
    def sample_trajectory(
        self,
        cond_features,
        device=None,
        num_samples=1,
        return_intermediate=False,
    ):
        """Sample trajectory using either Euler or adaptive sampling."""
        if self.sampling_method == "euler":
            return self._sample_trajectory_euler(
                cond_features, device, num_samples, return_intermediate
            )
        elif self.sampling_method == "adaptive":
            return self._sample_trajectory_adaptive(
                cond_features, device, num_samples, return_intermediate
            )
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    @torch.no_grad()
    def _sample_trajectory_euler(
        self,
        cond_features,
        device=None,
        num_samples=1,
        return_intermediate=False,
    ):
        """Euler sampling with fixed step size."""
        self.eval()
        B = cond_features.shape[0]
        T = self.sequence_length
        D = self.action_dim
        S = num_samples

        # Expand cond_features for num_samples > 1
        # [B, num_cond_tokens, embed_dim] -> [B*S, num_cond_tokens, embed_dim]
        if S > 1:
            cond_features_expanded = cond_features.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, -1, cond_features.shape[-1])
        else:
            cond_features_expanded = cond_features

        # Initial noise
        z = torch.randn(B, S, T, D, device=device)
        intermediates = []
        sqrt_var_traj = []

        # Euler integration from t=0 to t=1
        dt = 1.0 / self.num_inference_steps
        
        for i in range(self.num_inference_steps):
            t = i / self.num_inference_steps
            
            z_flat = z.view(B * S, T, D)
            t_vec = torch.full((B * S,), t, device=device)
            time_emb = self.time_embd(t_vec * self.pos_emb_scale)
            
            v_pred, log_sqrt_var_pred = self._predict_velocity_and_variance(
                z_flat, time_emb, cond_features_expanded
            )
            v_pred = v_pred.view(B, S, T, D)
            
            z = z.detach().clone() + v_pred * dt
            sqrt_var_traj.append(log_sqrt_var_pred.exp() ** 2)
            
            if return_intermediate:
                intermediates.append(z.clone())
        
        sqrt_var_traj = torch.stack(sqrt_var_traj, dim=-1)  # [B*S, num_steps]
        
        if num_samples == 1:
            z = z.squeeze(1)  # [B, T, D]
        
        if return_intermediate:
            return z, intermediates, sqrt_var_traj
        return z

    @torch.no_grad()
    def _sample_trajectory_adaptive(
        self,
        cond_features,
        device=None,
        num_samples=1,
        return_intermediate=False,
    ):
        """Adaptive sampling with variance-based step size."""
        self.eval()
        B = cond_features.shape[0]
        T = self.sequence_length
        D = self.action_dim
        S = num_samples

        # Expand cond_features for num_samples > 1
        # [B, num_cond_tokens, embed_dim] -> [B*S, num_cond_tokens, embed_dim]
        if S > 1:
            cond_features_expanded = cond_features.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, -1, cond_features.shape[-1])
        else:
            cond_features_expanded = cond_features

        # Initial noise
        z = torch.randn(B, S, T, D, device=device)
        
        # Track trajectories
        gen_traj = [z.clone()]
        valid_action = torch.zeros_like(z)
        valid_action_found = torch.zeros(B * S, dtype=torch.bool, device=device)
        num_steps_taken = torch.zeros(B * S, device=device)
        current_t = torch.zeros(B * S, device=device)
        step_traj = [current_t.clone()]
        var_traj = []
        
        for i in range(self.num_inference_steps):
            z_flat = z.view(B * S, T, D)
            t_vec = current_t
            time_emb = self.time_embd(t_vec * self.pos_emb_scale)
            
            v_pred, log_sqrt_var_pred = self._predict_velocity_and_variance(
                z_flat, time_emb, cond_features_expanded
            )
            
            # Compute variance and adaptive step size
            var_pred = log_sqrt_var_pred.exp() ** 2
            step_size = torch.max(
                self.eta / var_pred.sqrt(),
                torch.tensor(1.0 / self.num_inference_steps, device=device)
            )
            step_size = torch.min(step_size, 1.0 - current_t)
            
            # Update z
            v_pred = v_pred.view(B, S, T, D)
            step_size_expand = step_size.view(B, S, 1, 1)
            z = z.detach().clone() + v_pred * step_size_expand
            
            gen_traj.append(z.clone())
            current_t = current_t + step_size.view(-1)
            step_traj.append(current_t.clone())
            var_traj.append(var_pred)
            
            # Check if samples have reached t=1
            mask = (current_t >= 1.0).cpu() & (~valid_action_found)
            if mask.any():
                z_flat = z.view(B * S, T, D)
                valid_action.view(B * S, T, D)[mask] = z_flat[mask].detach().clone()
                valid_action_found[mask] = True
                num_steps_taken[mask] = i + 1
            
            if valid_action_found.all():
                break
        
        var_traj = torch.stack(var_traj, dim=-1)  # [B*S, num_steps]
        step_traj = torch.stack(step_traj, dim=-1)  # [B*S, num_steps+1]
        gen_traj = torch.stack(gen_traj, dim=1)  # [B, S, num_steps+1, T, D]
        
        if num_samples == 1:
            z = z.squeeze(1)  # [B, T, D]
            num_steps_taken = num_steps_taken.view(B, S).squeeze(1)
        
        result = {
            'actions': z,
            'nfe': num_steps_taken,
            'variance': var_traj,
            'step_traj': step_traj,
            'gen_traj': gen_traj if return_intermediate else None,
        }
        
        if return_intermediate:
            return result
        return z


class UNetDiffusionHead(UNetHead):
    def __init__(self, *args, cfg_dropout_prob=0.0, guidance_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.guidance_scale = float(guidance_scale)

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
        B, T, D = actions.shape
        eps = torch.randn_like(actions)

        # Classifier-Free Guidance: randomly drop conditioning during training
        if self.cfg_dropout_prob > 0.0:
            drop_mask = (torch.rand(B, 1, device=cond_features.device) < self.cfg_dropout_prob).float()
            cond_features = cond_features * (1.0 - drop_mask)

        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,),
            device=actions.device,
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(actions, eps, timesteps)
        time_emb = self.time_embd(timesteps)
        pred = self._predict_flow_unet(noisy_actions, time_emb, cond_features)
        target = eps

        return pred, target, noisy_actions, None

    @torch.no_grad()
    def sample_trajectory(
        self,
        cond_features,           # [B, cond_dim]
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
        use_cfg = (self.guidance_scale > 1.0) and (self.cfg_dropout_prob > 0.0)

        actions = torch.randn(
            size=(B * S, self.sequence_length, self.action_dim),
            dtype=dtype,
            device=device,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        intermediates = [] if return_intermediate else None

        # Precompute unconditional features (zeros) for CFG
        if use_cfg:
            uncond_features = torch.zeros_like(cond_features)

        for t in self.noise_scheduler.timesteps:
            t_vec = torch.full((B * S,), t.long(), device=device)
            time_emb = self.time_embd(t_vec)

            if use_cfg:
                # Dual forward: conditional + unconditional
                pred_cond = self._predict_flow_unet(actions, time_emb, cond_features)
                pred_uncond = self._predict_flow_unet(actions, time_emb, uncond_features)
                model_output = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
            else:
                model_output = self._predict_flow_unet(actions, time_emb, cond_features)

            actions = self.noise_scheduler.step(
                model_output, t, actions
            ).prev_sample

            if return_intermediate:
                intermediates.append(actions.clone())

        actions = actions.view(B, S, T, D)
        if num_samples == 1:
            actions = actions.squeeze(1)  # [B, T, D]

        if return_intermediate:
            return actions, intermediates
        return actions
