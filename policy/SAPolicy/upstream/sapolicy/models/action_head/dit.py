from typing import Dict, Optional, Tuple, Union
from contextlib import contextmanager

from distutils.util import strtobool
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sapolicy.models.latent_trunk import build_ray_maps
from sapolicy.models.transformer import DiTBlock, RMSNorm
from sapolicy.models.utils.pos_embed import SinusoidalPositionEmbeddings, add_pos_embed
from sapolicy.models.utils.rope import rope_params
from sapolicy.models.action_head.rtc import guided_velocity


_COMPILED_STEP_FNS = {}  # (id(head), key) -> compiled _forward_cond (kept off the module tree)


class DiTActionHead(nn.Module):
    """DiT-style action head with vision/TCP conditioning and split action tokens.

    **Data layout (per timestep)**  
    Flat action vector length ``action_dim`` is always ``[EE pose | gripper]`` in that order:
    ``[..., :d_pose]``, ``[..., d_pose:]`` with ``(d_pose, d_grip) = action_part_dims``.
    Default ``(9, 1)`` merges xyz + 6D rotation into pose, gripper stays separate.

    **Network structure (high level)**::

        patch_tokens [B,C,T,H,W]  ──► patch_embedding + optional geo (depth/K) ──► visual KV
                              (optional ``use_last_frame_visual``: only frame ``T-1``, no ``obs_time_embed`` on visual)
                              └──► else + ``obs_time_embed(τ)`` on all frames' patch tokens
        TCP dicts (uv/3d/6d)      ──► uv + merged pose (3d+6d) Linear ──► 2 KV tokens per (τ, TCP slot)
                              (τ = observation time: …, t−2, t−1, **t**; ``T`` matches ``patch_tokens``).
                              Bimanual: ``num_tcp`` slots per camera (``[B,T,num_tcp,C]``).
                              ``tcp_valid_embed`` bias on all TCP tokens when ``tcp_valid`` is given.
                              └──► + same ``obs_time_embed(τ)`` on uv and pose at frame τ
        cond_kv = concat(visual KV per camera, TCP KV per camera, state KV)

        **Time axes (contract for dataloading)**  
        - TCP sequence length ``T_tcp``: past + current, index ``T_tcp-1`` is **current** time ``t``.  
        - Action horizon ``T_a``: **future** commands ``t, t+1, …, t+T_a-1`` (first step aligns with TCP last frame).  
        - Only observation up to ``t`` is available; the same TCP KV conditions all future action steps.

        state [B,T_obs,A] ──► per-frame pose/grip Linear ──► [B,2·T_obs,D] state KV tokens
                              └──► + ``obs_time_embed(τ)`` shared by pose and grip at frame τ

        actions [B,T_a,A] ──► pose/grip Linear ──► interleave ──► [B,2·T_a,D]
                              └──► + TCP **pooled + horizon** additive bias (gated) + part_type_embed
                              └──► + learnable per-step PE (shared pose/grip) + RoPE in DiT blocks

        DiT: ``num_layers`` × (AdaLN self-attn + …) + from mid-layer: residual cross-attn to cond_kv
        decode: two Linear heads → concat → [B,T,A] (same order as input)

    **Flow matching** (``forward_training``): linear path ``x_t = (1-t)x_0 + t·ε``, target field
    ``ε - x_0``, MSE between predicted and target. With ``num_diffusion_draws = k > 1`` (training
    only) the conditioning KV is encoded once, then ``cond_kv`` / ``tcp_kv`` / ``x_0`` are
    ``repeat_interleave``'d ``k``× along batch and each of the ``k·B`` rows gets its own ``(t, ε)``;
    the MSE is averaged over ``k·B`` (ABC-style amortised draws). **Sampling** (``sample_trajectory``):
    Euler steps from ``t=1`` (noise) toward ``0`` with fixed ``num_inference_steps``.
    """

    def __init__(
        self,
        patch_in_dim: int,
        action_dim: int,
        embed_dim: int = 2048,
        num_heads: int = 8,
        num_layers: int = 8,
        patch_size: int = 1,
        use_depth: bool = True,
        use_camera_intrinsics: bool = True,
        geo_mode: str = "ray_depth",
        geo_embed_init_std: float = 0.0,
        dropout: float = 0.1,
        sequence_length: int = 16,
        num_inference_steps: int = 10,
        action_part_dims: Union[Tuple[int, int], list] = (9, 1),
        use_tcp_valid: bool = True,
        skip_patch_kv_proj: bool = False,
        disable_tcp_kv: bool = False,
        disable_tcp_additive_bias: bool = False,
        use_state: bool = False,
        state_dim: int = 7,
        obs_hist_length: int = 1,
        full_conditional: bool = False,
        num_cameras: int = 2,
        num_tcp_cameras: Optional[int] = None,
        num_tcp: int = 1,
        num_spatial_patches_per_cam: int = 25,
        use_last_frame_visual: bool = False,
        num_diffusion_draws: int = 1,
        use_rgb_kv_norm: bool = True,
    ):
        """
        Args:
            patch_in_dim: Channel dimension of ``patch_tokens`` (e.g. fused DA3 feature dim).
            action_dim: Total action size per step; must equal sum(action_part_dims).
            embed_dim: Transformer width ``D`` for action tokens, KV, and cross-attention.
            num_heads: Self-attention heads in DiT blocks and cross-attention.
            num_layers: Number of DiT blocks; second half also runs cross-attn to ``cond_kv``.
            skip_patch_kv_proj: If True, patch_tokens are already in embed_dim space (e.g. from TCP attended_patch); skip patch_embedding, geo injection, and the DiT RGB norms in _build_patch_kv.
            disable_tcp_kv: If True, skip TCP KV tokens and additive bias entirely — DiT cross-attends only to visual KV.
            disable_tcp_additive_bias: If True, keep TCP KV but skip the pooled-TCP additive bias on action tokens (ablation of the shortcut path).
            patch_size: Conv stride on ``(H,W)`` inside geo embedding (often ``1`` for 1×1 tokens).
            use_depth: If True, build geometry from depth (and intrinsics if both True).
            use_camera_intrinsics: If True with ``use_depth``, ``build_ray_maps`` uses ``(fx,fy,cx,cy)``.
            geo_mode: ``"ray_depth"`` or ``"xyz"`` passed to ``build_ray_maps``.
            geo_embed_init_std: Standard deviation for geometry projections. Keep the
                gate at zero while setting this nonzero to let the gate learn.
            dropout: Dropout inside DiT / cross-attention blocks.
            sequence_length: Action horizon ``T_a`` (number of action steps): sampling length, training ``actions.shape[1]``, and TCP horizon-embedding table size (indices ``0 … T_a-1``).
            num_inference_steps: Euler steps for ``sample_trajectory`` (``dt = -1 / num_inference_steps``).
            action_part_dims: ``(d_pose, d_grip)`` — merged EE pose (e.g. pos3+rot6d=9) and gripper.
            num_cameras: Number of cameras (must match ``len(patch_tokens)`` in forward).
            num_tcp_cameras: Number of cameras supplying TCP KV. Defaults to ``num_cameras``.
            num_tcp: Number of TCP points per camera (1=single-arm; >1=bimanual). Each TCP is
                one point-KV slot (uv+pose); must match LatentAuxiliaryModel / dataset ``num_tcp``.
            num_spatial_patches_per_cam: ``H*W`` patch tokens per camera per observation timestep
                (e.g. ``5*5=25`` for a ``70×70`` crop with ViT patch size ``14``). Point TCP uses ``HW=num_tcp``.
            use_last_frame_visual: If True, visual KV uses only the last observation frame (like FM);
                ``obs_time_embed`` is not added to visual tokens (TCP/state still use full ``obs_hist_length``).
            num_diffusion_draws: ``k`` independent ``(t, ε)`` draws per sample in ``forward_training``
                (train mode only). Conditioning is encoded once and repeated ``k``× along batch, so the
                DiT blocks run on ``k·B`` rows and the loss is averaged over them. ``1`` = unchanged.
            use_rgb_kv_norm: Apply ``norm_after_geo_1/2`` (GroupNorm + RMSNorm) to third-person RGB visual KV
                even without depth/geo input (main since 2026-09). ``False`` restores the pre-2026-09 graph
                (norms only exist when ``geo_in_channels > 0``) - required to load checkpoints trained before
                the norms were added (e.g. the YAM/ABC bottles runs).
        """
        super().__init__()
        self.num_diffusion_draws = int(num_diffusion_draws)
        # Set by SAPolicy(compile_action_head=True): run _forward_cond through torch.compile.
        self.compile_step = False
        self.compile_mode = "default"
        if self.num_diffusion_draws < 1:
            raise ValueError(f"num_diffusion_draws must be >= 1, got {num_diffusion_draws}")
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.geo_mode = geo_mode
        self.geo_embed_init_std = float(geo_embed_init_std)
        self.action_dim = action_dim
        self.sequence_length = int(sequence_length)
        self.num_inference_steps = int(num_inference_steps)
        self.use_tcp_valid = bool(use_tcp_valid)
        self.skip_patch_kv_proj = bool(skip_patch_kv_proj)
        self.disable_tcp_kv = bool(disable_tcp_kv)
        self.disable_tcp_additive_bias = bool(disable_tcp_additive_bias)
        if isinstance(use_state, bool):
            self.use_state = use_state
        else:
            self.use_state = bool(strtobool(str(use_state)))
        self.state_dim = int(state_dim) if self.use_state else 0
        self.obs_hist_length = int(obs_hist_length)
        self.num_cameras = int(num_cameras)
        self.num_tcp_cameras = (
            self.num_cameras if num_tcp_cameras is None else int(num_tcp_cameras)
        )
        self.num_tcp = int(num_tcp)
        if self.num_tcp < 1:
            raise ValueError(f"num_tcp must be >= 1, got {num_tcp}")
        self.num_spatial_patches_per_cam = int(num_spatial_patches_per_cam)
        self.use_last_frame_visual = bool(use_last_frame_visual)
        # Shared observation-time index τ for TCP / state (and multi-frame visual when enabled).
        self.obs_time_embed = nn.Embedding(self.obs_hist_length, embed_dim)

        part = tuple(int(x) for x in action_part_dims)
        if len(part) != 2:
            raise ValueError(f"action_part_dims must be length-2 (pose, gripper), got {part}")
        if sum(part) != action_dim:
            raise ValueError(f"action_part_dims {part} must sum to action_dim={action_dim}")
        self._d_pose, self._d_grip = part
        self.action_part_dims = part
        # Each physical timestep is expanded to one token per action part (order fixed).
        self._num_part_tokens = len(self.action_part_dims)

        # --- Visual branch: per-camera feature map -> token sequence for cross-attention ---
        self.patch_embedding = nn.Conv3d(patch_in_dim, embed_dim, kernel_size=1, stride=1)

        # geo_in_channels: 0=RGB-only; 1=depth only; 3=depth + ray map from intrinsics (see LatentTrunk.build_ray_maps)
        #
        # Keep GroupNorm + RMSNorm on the DiT visual KV path even when use_depth=false.
        # Closed-loop notcp (RoboSuite Nut t4, RoboTwin handover/lift) consistently
        # improves with these layers present; audited depth ckpts have geo_gate=0, so
        # the gain is trained RGB norms, not ray/depth tokens. Wrist cameras skip them
        # (same as the depth recipe). TCP/auxbypass third-person uses LatentTrunk
        # instead (`skip_patch_kv_proj=true`); do not add the same RGB-only norms there.
        self.geo_in_channels = 3 if (use_depth and use_camera_intrinsics) else (1 if use_depth else 0)
        self.use_rgb_kv_norm = bool(use_rgb_kv_norm)
        if self.use_rgb_kv_norm or self.geo_in_channels > 0:
            self.norm_after_geo_1 = nn.GroupNorm(1, embed_dim)
            self.norm_after_geo_2 = RMSNorm(embed_dim, eps=1e-6)

        if self.geo_in_channels > 0:
            hidden = embed_dim // 4
            self.geo_stem = nn.Sequential(
                nn.Conv3d(self.geo_in_channels, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
                nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
            self.geo_embedding_1 = nn.Conv3d(hidden, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
            self.geo_embedding_2 = nn.Conv3d(hidden, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
            nn.init.normal_(self.geo_embedding_1.weight, std=self.geo_embed_init_std)
            nn.init.constant_(self.geo_embedding_1.bias, 0.0)
            nn.init.normal_(self.geo_embedding_2.weight, std=self.geo_embed_init_std)
            nn.init.constant_(self.geo_embedding_2.bias, 0.0)
            self.geo_gate_1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.geo_gate_2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # `norm_after_geo_*` are built above (always when use_rgb_kv_norm, else only here).

        # --- Action branch: 2 encoders -> 2*T tokens; AdaLN uses continuous flow time t in [0,1] ---
        self.action_proj_pose = nn.Linear(self._d_pose, embed_dim)
        self.action_proj_grip = nn.Linear(self._d_grip, embed_dim)
        # Distinguishes pose vs grip slots beyond step position encoding.
        self.part_type_embed = nn.Parameter(torch.zeros(self._num_part_tokens, embed_dim))
        # Learnable per future-step embedding; pose and grip at the same step share one vector.
        self.action_step_pos = nn.Parameter(torch.zeros(1, self.sequence_length, embed_dim))
        self.timestep_embed = SinusoidalPositionEmbeddings(embed_dim)
        # Produces 9×D from scalar t: 6×D for self-attn + MLP AdaLN, 3×D for cross-attn branch (DiTBlock conditional).
        self.timestep_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim * 9))

        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout,
                qkv_bias=True,
                proj_bias=False,
                norm_eps=1e-6,
                conditional=True if full_conditional else (_>=num_layers//2),
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(embed_dim, eps=1e-6)
        self.out_pose = nn.Linear(embed_dim, self._d_pose)
        self.out_grip = nn.Linear(embed_dim, self._d_grip)

        # --- TCP → KV: uv + pose (3d+6d) tokens per (time, TCP slot); valid via tcp_valid_embed bias ---
        # Per-TCP pose is always pos3+rot6d=9, even when action_part_dims pose is 9*num_tcp (bimanual).
        self._tcp_pose_dim = 9
        if not self.disable_tcp_kv:
            self._num_tcp_kv_tokens = 2
            self.tcp_uv_embed = nn.Linear(2, embed_dim)
            self.tcp_pose_embed = nn.Linear(self._tcp_pose_dim, embed_dim)
            self.tcp_part_type_embed = nn.Parameter(torch.zeros(self._num_tcp_kv_tokens, embed_dim))
            self.tcp_valid_embed = nn.Embedding(2, embed_dim) if self.use_tcp_valid else None

            if not self.disable_tcp_additive_bias:
                # Pooled TCP KV + per-future-step embedding → additive bias on action tokens (gated).
                # One row per action step index; same ``T_a`` bound as ``sequence_length``.
                self.tcp_horizon_emb = nn.Embedding(self.sequence_length, embed_dim)
                self.tcp_bias_mlp = nn.Sequential(
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                # Sigmoid gate; init negative so bias starts small.
                self.tcp_bias_gate = nn.Parameter(torch.tensor(-4.0))

        # Cached RoPE frequencies for the action token stream (length up to 4096 entries).
        assert (embed_dim % num_heads) == 0 and (embed_dim // num_heads) % 2 == 0
        head_dim = embed_dim // num_heads
        self.action_freqs = rope_params(4096, head_dim)

        # ``cond_pos_enc`` length must match ``cond_kv`` in forward: visual + optional TCP KV + optional state.
        # Visual: ``T_vis * num_cameras * (H*W)`` with ``T_vis=1`` if ``use_last_frame_visual`` else ``T_obs``;
        # TCP (point): ``T_obs * num_tcp_cameras * num_tcp * 2`` (uv+pose per arm); state: ``T_obs * 2`` tokens.
        n_vis_frames = 1 if self.use_last_frame_visual else self.obs_hist_length
        n_vis = n_vis_frames * self.num_cameras * self.num_spatial_patches_per_cam
        n_tcp = (
            0
            if self.disable_tcp_kv
            else self.obs_hist_length
            * self.num_tcp_cameras
            * self.num_tcp
            * self._num_tcp_kv_tokens
        )
        n_state = self.obs_hist_length * self._num_part_tokens if self.use_state else 0
        cond_length = n_vis + n_tcp + n_state

        if self.use_state:
            # Per-frame state tokens aligned with action parts (pose + grip per obs timestep).
            if self.state_dim != self.action_dim:
                raise ValueError(
                    f"DiTActionHead expects state_dim == action_dim when use_state=True, "
                    f"got state_dim={self.state_dim}, action_dim={self.action_dim}"
                )
            self.state_proj_pose = nn.Linear(self._d_pose, embed_dim)
            self.state_proj_grip = nn.Linear(self._d_grip, embed_dim)
            self.state_part_type_embed = nn.Parameter(torch.zeros(self._num_part_tokens, embed_dim))

        self.cond_pos_enc = nn.Parameter(torch.zeros(1, cond_length, embed_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Light init: action stream learns task; geo branches start near identity (gates already 0)."""
        for m in (self.action_proj_pose, self.action_proj_grip):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.part_type_embed, std=0.02)
        nn.init.normal_(self.action_step_pos, std=0.02)
        nn.init.normal_(self.obs_time_embed.weight, std=0.02)
        for head in (self.out_pose, self.out_grip):
            nn.init.xavier_uniform_(head.weight, gain=1e-2)
            nn.init.zeros_(head.bias)
        if not self.disable_tcp_kv:
            nn.init.normal_(self.tcp_part_type_embed, std=0.02)
            for m in (self.tcp_uv_embed, self.tcp_pose_embed):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if self.tcp_valid_embed is not None:
                nn.init.normal_(self.tcp_valid_embed.weight, std=0.02)
            if not self.disable_tcp_additive_bias:
                nn.init.normal_(self.tcp_horizon_emb.weight, std=0.02)
                for layer in self.tcp_bias_mlp:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        if self.use_state:
            for m in (self.state_proj_pose, self.state_proj_grip):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            nn.init.normal_(self.state_part_type_embed, std=0.02)

    def _split_action(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slice flat actions into pose / grip along the last dim (order matches `action_part_dims`).

        Input ``a``: ``[B, T_a, action_dim]``.
        """
        dp = self._d_pose
        return a[..., :dp], a[..., dp:]

    def _merge_action(self, pose: torch.Tensor, grip: torch.Tensor) -> torch.Tensor:
        """Concatenate parts back to ``[B, T_a, action_dim]`` for loss / env."""
        return torch.cat([pose, grip], dim=-1)

    def _encode_action_tokens(self, a: torch.Tensor) -> torch.Tensor:
        """Project each part and interleave to ``[B, 2·T_a, D]``; add learnable type embedding per slot.

        Input ``a``: ``[B, T_a, action_dim]``; output: ``[B, 2·T_a, D]`` (``D = embed_dim``).
        """
        b, ta, _ = a.shape
        pose, grip = self._split_action(a)
        zp = self.action_proj_pose(pose)
        zg = self.action_proj_grip(grip)
        x = torch.stack([zp, zg], dim=2).reshape(b, self._num_part_tokens * ta, self.embed_dim)
        type_emb = self.part_type_embed.view(1, 1, self._num_part_tokens, -1).expand(
            b, ta, self._num_part_tokens, -1
        ).reshape(b, self._num_part_tokens * ta, self.embed_dim)
        return x + type_emb

    def _decode_action_tokens(self, x: torch.Tensor, ta: int) -> torch.Tensor:
        """Split ``2·T_a`` stream back to per-step heads and merge to flat actions.

        Input ``x``: ``[B, 2·T_a, D]``; ``ta``: action horizon ``T_a``. Output: ``[B, T_a, action_dim]``.
        """
        b, seq, d = x.shape
        expected = self._num_part_tokens * ta
        if seq != expected:
            raise ValueError(f"Expected seq_len {expected} (2*T), got {seq}")
        x = x.view(b, ta, self._num_part_tokens, d)
        pose = self.out_pose(x[:, :, 0])
        grip = self.out_grip(x[:, :, 1])
        return self._merge_action(pose, grip)

    def _action_step_pos_embed(self, ta: int, device: torch.device) -> torch.Tensor:
        """Per future-step embedding broadcast to pose/grip slots: ``[1, 2·T_a, D]``."""
        if ta > self.sequence_length:
            raise ValueError(
                f"Action horizon ta={ta} exceeds sequence_length={self.sequence_length}"
            )
        step_emb = self.action_step_pos[:, :ta, :].to(device)
        return step_emb.repeat_interleave(self._num_part_tokens, dim=1)

    def _encode_state_tokens(self, state: torch.Tensor) -> torch.Tensor:
        """Project each observation frame to interleaved pose/grip tokens ``[B, 2·T_obs, D]``."""
        b, t_obs, _ = state.shape
        if t_obs != self.obs_hist_length:
            raise ValueError(
                f"state time dim {t_obs} != obs_hist_length={self.obs_hist_length}"
            )
        s = state.to(dtype=self.state_proj_pose.weight.dtype)
        s_pose = s[..., : self._d_pose]
        s_grip = s[..., self._d_pose :]
        pose_tok = self.state_proj_pose(s_pose)
        grip_tok = self.state_proj_grip(s_grip)
        x = torch.stack([pose_tok, grip_tok], dim=2).reshape(
            b, t_obs * self._num_part_tokens, self.embed_dim
        )
        type_emb = self.state_part_type_embed.view(1, 1, self._num_part_tokens, -1).expand(
            b, t_obs, self._num_part_tokens, -1
        ).reshape(b, t_obs * self._num_part_tokens, self.embed_dim)
        time_emb = self._obs_time_embed_broadcast(
            t_obs, self._num_part_tokens, b, state.device, s.dtype
        )
        return x + type_emb + time_emb

    def _obs_time_embed_broadcast(
        self,
        t_steps: int,
        tokens_per_frame: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Broadcast ``obs_time_embed(τ)`` to all tokens at each obs frame.

        Returns ``[B, t_steps * tokens_per_frame, D]`` with the same τ vector repeated
        for every token belonging to that observation frame (e.g. all H·W patches, uv+pose, pose+grip).
        """
        if t_steps > self.obs_hist_length:
            raise ValueError(
                f"t_steps={t_steps} exceeds obs_hist_length={self.obs_hist_length}"
            )
        idx = torch.arange(t_steps, device=device, dtype=torch.long)
        emb = self.obs_time_embed(idx).to(dtype=dtype)
        emb = emb.repeat_interleave(tokens_per_frame, dim=0)
        return emb.unsqueeze(0).expand(batch_size, -1, -1)

    def _create_action_freqs(self, length: int, device: torch.device, start: int = 0) -> torch.Tensor:
        """RoPE table slice of shape ``[length, 1, head_dim]`` for DiTBlock self-attention."""
        if self.action_freqs.device != device:
            self.action_freqs = self.action_freqs.to(device)
        return self.action_freqs[start:start + length].view(length, 1, -1)

    @staticmethod
    def _tcp_to_bthwc(t: torch.Tensor, b: int, t_steps: int) -> torch.Tensor:
        """TCP layout ``[B, T, C]`` or ``[B, T, N, C]`` → ``[B, T, N, C]``.

        ``N`` is the per-camera TCP-slot axis: ``1`` for single-arm point TCP, or ``num_tcp``
        for bimanual (one slot per arm). Inserts ``N=1`` when the input is ``[B, T, C]``.

        Args:
            ``t``: ``[B, T, C]`` or ``[B, T, N, C]`` with ``T = t_steps``.
            ``b``, ``t_steps``: batch size ``B`` and observation length ``T`` (must match ``t.shape[:2]``).
        """
        if t.dim() == 4:
            if t.shape[0] != b or t.shape[1] != t_steps:
                raise ValueError(
                    f"Expected TCP [B, T, N, C] with B={b}, T={t_steps}; got {tuple(t.shape)}"
                )
            return t
        if t.dim() == 3:
            if t.shape[0] != b or t.shape[1] != t_steps:
                raise ValueError(
                    f"Expected TCP [B, T, C] with B={b}, T={t_steps}; got {tuple(t.shape)}"
                )
            return t.unsqueeze(2)
        raise ValueError(
            f"TCP tensor must be [B, T, C] or [B, T, N, C]; got dim={t.dim()}, shape={tuple(t.shape)}"
        )

    def _tcp_valid_ids(
        self,
        valid: torch.Tensor,
        b: int,
        t_steps: int,
        num_slots: int,
    ) -> torch.Tensor:
        """Map ``tcp_valid`` to integer ids ``{0, 1}`` of shape ``[B, T, N]``.

        Accepts ``[B, T]`` / ``[B, T, 1]`` (broadcast over all TCP slots) or
        ``[B, T, N]`` / ``[B, T, N, 1]`` (per-arm validity).
        """
        if valid.dim() >= 1 and valid.shape[-1] == 1:
            valid = valid.squeeze(-1)
        if valid.dim() == 2:
            if valid.shape != (b, t_steps):
                raise ValueError(
                    f"tcp_valid must be [B, T] with B={b}, T={t_steps}; got {tuple(valid.shape)}"
                )
            valid = valid.unsqueeze(-1).expand(b, t_steps, num_slots)
        elif valid.dim() == 3:
            if valid.shape == (b, t_steps, 1):
                valid = valid.expand(b, t_steps, num_slots)
            elif valid.shape != (b, t_steps, num_slots):
                raise ValueError(
                    f"tcp_valid must be [B, T, N] with B={b}, T={t_steps}, N={num_slots}; "
                    f"got {tuple(valid.shape)}"
                )
        else:
            raise ValueError(
                f"tcp_valid must be [B, T], [B, T, 1], [B, T, N], or [B, T, N, 1]; "
                f"got dim={valid.dim()}, shape={tuple(valid.shape)}"
            )
        if valid.dtype == torch.bool:
            valid_ids = valid.long()
        elif valid.is_floating_point():
            valid_ids = (valid > 0.5).long()
        else:
            valid_ids = valid.long()
        if os.environ.get("SA_DEBUG_NONFINITE", "0") == "1" and not torch.logical_or(valid_ids == 0, valid_ids == 1).all():  # host sync; debug only
            raise ValueError("tcp_valid must be boolean or in {0, 1} per entry")
        return valid_ids

    def _build_tcp_kv(
        self,
        b: int,
        t_steps: int,
        tcp_uv: Dict[str, torch.Tensor],
        tcp_3d: Dict[str, torch.Tensor],
        tcp_6d: Dict[str, torch.Tensor],
        tcp_valid: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """TCP keys per camera: interleave ``uv → pose(3d+6d)`` per (obs time τ, TCP slot).

        **Timeline:** ``t_steps`` should follow observation time …, t−2, t−1, **t** (last index = current).

        Each camera contributes ``T · num_tcp · 2`` tokens (uv + pose per arm). Bimanual inputs are
        ``[B, T, num_tcp, C]``; single-arm ``[B, T, C]`` is treated as ``num_tcp=1``.

        When ``tcp_valid`` is provided, ``tcp_valid_embed`` adds a visibility bias to **each** TCP
        token (uv and pose slots), with optional per-arm masks.

        Output: ``[B, L_tcp, D]`` with ``L_tcp = sum_cam (T·num_tcp·2)``.
        """

        tokens = []
        d = self.embed_dim
        if len(tcp_3d) != self.num_tcp_cameras:
            raise ValueError(
                f"Expected TCP for {self.num_tcp_cameras} cameras, got {len(tcp_3d)}."
            )

        for cam in sorted(tcp_3d.keys()):
            uv = self._tcp_to_bthwc(tcp_uv[cam], b, t_steps)
            p3 = self._tcp_to_bthwc(tcp_3d[cam], b, t_steps)
            r6 = self._tcp_to_bthwc(tcp_6d[cam], b, t_steps)
            if uv.shape[2] != self.num_tcp:
                raise ValueError(
                    f"tcp_uv[{cam!r}] TCP-slot dim must be num_tcp={self.num_tcp}; got {uv.shape[2]}"
                )
            if p3.shape[2] != self.num_tcp or r6.shape[2] != self.num_tcp:
                raise ValueError(
                    f"TCP pose slot dim for {cam!r} must be num_tcp={self.num_tcp}; "
                    f"got pos={p3.shape[2]}, rot={r6.shape[2]}"
                )
            if uv.shape[-1] != 2:
                raise ValueError(f"tcp_uv[{cam!r}] last dim must be 2; got {uv.shape[-1]}")
            if p3.shape[-1] != 3 or r6.shape[-1] != 6:
                raise ValueError(
                    f"TCP pose parts for {cam!r} must be 3D pos + 6D rot; got {p3.shape[-1]=}, {r6.shape[-1]=}"
                )
            pose_feat = torch.cat([p3, r6], dim=-1)
            if pose_feat.shape[-1] != self._tcp_pose_dim:
                raise ValueError(
                    f"Merged per-TCP pose dim {pose_feat.shape[-1]} != {self._tcp_pose_dim}"
                )

            zu = self.tcp_uv_embed(uv)
            zp = self.tcp_pose_embed(pose_feat)
            x = torch.stack([zu, zp], dim=3)
            _, tt, n_tcp, _, _ = x.shape
            x = x.reshape(b, tt * n_tcp * self._num_tcp_kv_tokens, d)

            te = self.tcp_part_type_embed.view(1, 1, 1, self._num_tcp_kv_tokens, d).expand(
                b, tt, n_tcp, self._num_tcp_kv_tokens, d
            ).reshape(b, tt * n_tcp * self._num_tcp_kv_tokens, d)
            cam_tokens = x + te

            if self.use_tcp_valid and self.tcp_valid_embed is not None and tcp_valid is not None and cam in tcp_valid:
                valid_ids = self._tcp_valid_ids(tcp_valid[cam], b, t_steps, n_tcp)
                valid_emb = self.tcp_valid_embed(valid_ids)  # [B, T, N, D]
                valid_emb = valid_emb.unsqueeze(3).expand(
                    b, tt, n_tcp, self._num_tcp_kv_tokens, d
                ).reshape(b, tt * n_tcp * self._num_tcp_kv_tokens, d)
                cam_tokens = cam_tokens + valid_emb

            time_emb = self._obs_time_embed_broadcast(
                tt, n_tcp * self._num_tcp_kv_tokens, b, cam_tokens.device, cam_tokens.dtype
            )
            cam_tokens = cam_tokens + time_emb

            tokens.append(cam_tokens)

        return torch.cat(tokens, dim=1)

    def _build_patch_kv(
        self,
        patch_tokens: Dict[str, torch.Tensor],  # cam -> [B, C_in, T, H, W] (SAPolicy / DA3)
        depths: Optional[Dict[str, torch.Tensor]],  # cam -> [B, 1, T, H_d, W_d]; required if geo enabled
        camera_intrinsics: Optional[Dict[str, torch.Tensor]],  # cam -> [B, T, 4] (fx,fy,cx,cy); with depth+K geo
    ) -> torch.Tensor:
        """Encode each camera's spatio-temporal grid to ``[B, T·H·W, D]`` per cam; concat cams on sequence dim.

        When ``use_last_frame_visual`` is True, only the last observation frame is kept (FM-aligned).

        Output: ``[B, L_vis, D]`` with ``L_vis = sum_cam (T_vis·H·W)``.
        """
        kv_list = []
        for cam in sorted(patch_tokens.keys()):
            feat = patch_tokens[cam]
            b, _, t_steps, h, w = feat.shape
            # Wrist RGB still conditions DiT; wrist depth/K are not used for ray_depth.
            skip_wrist_geo = "eye_in_hand" in cam
            depth_cam = (
                None
                if skip_wrist_geo
                else (depths[cam] if depths is not None and cam in depths else None)
            )
            k_cam = (
                None
                if skip_wrist_geo
                else (
                    camera_intrinsics[cam]
                    if camera_intrinsics is not None and cam in camera_intrinsics
                    else None
                )
            )

            if self.use_last_frame_visual:
                feat = feat[:, :, -1:, :, :]
                if depth_cam is not None:
                    depth_cam = depth_cam[:, :, -1:, :, :]
                if k_cam is not None:
                    k_cam = k_cam[:, -1:, :]
                t_steps = 1

            if self.skip_patch_kv_proj:
                # attended_patch already in embed_dim space; just flatten
                x = feat.permute(0, 2, 3, 4, 1).contiguous().view(b, t_steps * h * w, self.embed_dim)
            else:
                x = self.patch_embedding(feat)  # [B,D,T,H,W]

                geo_feat = None
                if self.geo_in_channels > 0 and not skip_wrist_geo:
                    if depth_cam is None:
                        raise ValueError(f"Missing depth for camera {cam}")
                    if self.geo_in_channels == 1:
                        geo = depth_cam
                    else:
                        if k_cam is None:
                            raise ValueError(f"Missing camera intrinsics for camera {cam}")
                        depth_bthw = depth_cam.permute(0, 2, 1, 3, 4).contiguous()
                        geo = build_ray_maps(depth_bthw, k_cam, self.geo_mode)
                    geo_feat = self.geo_stem(geo)
                    g1 = self.geo_embedding_1(geo_feat)
                    if g1.shape[2:] != x.shape[2:]:
                        g1 = F.interpolate(g1, size=x.shape[2:], mode="trilinear", align_corners=False)
                    x = self.norm_after_geo_1(x + self.geo_gate_1.view(1, -1, 1, 1, 1) * g1)
                elif self.use_rgb_kv_norm and not skip_wrist_geo:
                    # notcp nodepth / geo_gate=0 equivalent: third-person RGB through the two DiT norms.
                    x = self.norm_after_geo_1(x)

                x = x.permute(0, 2, 3, 4, 1).contiguous().view(b, t_steps * h * w, self.embed_dim)

                if geo_feat is not None:
                    g2 = self.geo_embedding_2(geo_feat)
                    if g2.shape[2:] != (t_steps, h, w):
                        g2 = F.interpolate(g2, size=(t_steps, h, w), mode="trilinear", align_corners=False)
                    g2 = g2.permute(0, 2, 3, 4, 1).contiguous().view(b, t_steps * h * w, self.embed_dim)
                    x = self.norm_after_geo_2(x + self.geo_gate_2 * g2)
                elif self.use_rgb_kv_norm and not skip_wrist_geo:
                    x = self.norm_after_geo_2(x)

            x = add_pos_embed(x, w, h)
            if not self.use_last_frame_visual:
                x = x + self._obs_time_embed_broadcast(t_steps, h * w, b, x.device, x.dtype)
            kv_list.append(x)
        return torch.cat(kv_list, dim=1)

    def _tcp_additive_gate_bias(self, tcp_kv: torch.Tensor, ta: int, b: int) -> torch.Tensor:
        """Bias each future action step using pooled TCP history + horizon index (gated).

        Inputs: ``tcp_kv`` ``[B, L_tcp, D]``, ``ta`` = action horizon ``T_a``, ``b`` = ``B``.
        Output: ``[B, 2·T_a, D]`` (broadcast to pose/grip token slots).

        Same TCP context (observation up to **t**) biases all steps ``t … t+ta-1``; ``a`` encodes
        offset within the predicted horizon. Broadcast to ``2·ta`` action tokens (pose/grip).
        """
        # tcp_kv: [B, L_tcp, D]; ta: T_a; b: B; D = embed_dim
        if ta > self.sequence_length:
            raise ValueError(
                f"Action horizon ta={ta} exceeds sequence_length={self.sequence_length} "
                "(configure sequence_length to match the action chunk length)."
            )
        device = tcp_kv.device  # same as tcp_kv
        dtype = tcp_kv.dtype  # for MLP matmul dtype
        pooled = tcp_kv.mean(dim=1)  # [B, D] — mean over L_tcp
        h_idx = torch.arange(ta, device=device)  # [ta]
        h_emb = self.tcp_horizon_emb(h_idx)  # [ta, D]
        inp = torch.cat(
            [
                pooled.unsqueeze(1).expand(-1, ta, -1),  # [B, ta, D]
                h_emb.unsqueeze(0).expand(b, -1, -1),  # [B, ta, D]
            ],
            dim=-1,  # [B, ta, 2D]
        )
        delta = self.tcp_bias_mlp(inp.to(dtype))  # [B, ta, D]
        delta = delta.repeat_interleave(self._num_part_tokens, dim=1)  # [B, 2*ta, D]
        return torch.sigmoid(self.tcp_bias_gate) * delta  # [] * [B, 2*ta, D] → [B, 2*ta, D]

    def forward(
        self,
        action_tokens: torch.Tensor,  # [B, T_a, action_dim] 噪声或当前样本；T_a = 动作预测步数
        timesteps: torch.Tensor,  # [B] Flow 时间 t∈(0,1)，每样本标量
        patch_tokens: Dict[str, torch.Tensor],  # cam -> [B, C, T_obs, H, W]；T_obs 与 TCP 的 T 一致
        depths: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, 1, T_obs, H_d, W_d]；无 geo 时可 None
        camera_intrinsics: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 4] (fx,fy,cx,cy)
        gt_tcp_uv: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 2]；use_gt_tcp=True 时必填
        gt_tcp_3d: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 3]
        gt_tcp_6d: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 6]
        gt_tcp_valid: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs] 或 [B, T_obs, 1]；可选
        pred_tcp_uv: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 2]；use_gt_tcp=False 时必填
        pred_tcp_3d: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 3]
        pred_tcp_6d: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 6]
        pred_tcp_valid: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs] 或 [B, T_obs, 1]；可选
        use_gt_tcp: bool = True,
        state: Optional[torch.Tensor] = None,  # [B, state_dim]
    ) -> torch.Tensor:
        """Predict flow / velocity field in action space for given noisy actions and time ``t``.

        Args:
            action_tokens: Noisy actions ``x_t`` of shape ``[B, T, action_dim]`` (training) or current sample (inference).
            timesteps: Scalar ``t`` per batch element ``[B]`` in ``(0, 1)`` for AdaLN and FM path.
            patch_tokens: Per-camera visual features ``[B, C, T, H, W]`` (e.g. DA3 patch grid).
            depths: Per-camera depth ``[B, C, T, H, W]`` (channel-minor) when geo is enabled.
            camera_intrinsics: Per-camera ``[B, T, 4]`` as ``(fx, fy, cx, cy)`` when using ray geometry.
            gt_tcp_* / pred_tcp_*: Per-camera ``[B, T, …]`` over observation time (``T`` must match ``patch_tokens``; action horizon ``T_a`` is separate).
            use_gt_tcp: If True, build TCP KV from ``gt_tcp_*``; else from ``pred_tcp_*``.

        **Alignment:** ``action_tokens`` are future commands at ``t, t+1, …``; TCP dicts end at current ``t``.
        Cross-attn sees full TCP+vision KV; additive bias injects pooled TCP + step index into each action slot.

        Returns:
            Tensor of shape ``[B, T, action_dim]`` (same layout as input: pose | grip).
        """
        cond_kv, tcp_kv = self._build_cond_kv(
            action_tokens.shape[0],
            patch_tokens,
            depths=depths,
            camera_intrinsics=camera_intrinsics,
            gt_tcp_uv=gt_tcp_uv,
            gt_tcp_3d=gt_tcp_3d,
            gt_tcp_6d=gt_tcp_6d,
            gt_tcp_valid=gt_tcp_valid,
            pred_tcp_uv=pred_tcp_uv,
            pred_tcp_3d=pred_tcp_3d,
            pred_tcp_6d=pred_tcp_6d,
            pred_tcp_valid=pred_tcp_valid,
            use_gt_tcp=use_gt_tcp,
            state=state,
        )
        return self._step_fn()(action_tokens, timesteps, cond_kv, tcp_kv)

    def _build_cond_kv(
        self,
        b: int,
        patch_tokens: Dict[str, torch.Tensor],
        depths: Optional[Dict[str, torch.Tensor]] = None,
        camera_intrinsics: Optional[Dict[str, torch.Tensor]] = None,
        gt_tcp_uv: Optional[Dict[str, torch.Tensor]] = None,
        gt_tcp_3d: Optional[Dict[str, torch.Tensor]] = None,
        gt_tcp_6d: Optional[Dict[str, torch.Tensor]] = None,
        gt_tcp_valid: Optional[Dict[str, torch.Tensor]] = None,
        pred_tcp_uv: Optional[Dict[str, torch.Tensor]] = None,
        pred_tcp_3d: Optional[Dict[str, torch.Tensor]] = None,
        pred_tcp_6d: Optional[Dict[str, torch.Tensor]] = None,
        pred_tcp_valid: Optional[Dict[str, torch.Tensor]] = None,
        use_gt_tcp: bool = True,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode every condition (visual / TCP / state) into cross-attn KV, independent of ``(x_t, t)``.

        Returns ``(cond_kv, tcp_kv)``: ``cond_kv`` ``[B, L_cond, D]`` with ``cond_pos_enc`` added;
        ``tcp_kv`` ``[B, L_tcp, D]`` (needed by the additive bias) or ``None`` when ``disable_tcp_kv``.
        """
        # Build visual condition tokens (KV base).
        cond_kv = self._build_patch_kv(patch_tokens, depths, camera_intrinsics)

        b_vis, _, t_steps, _, _ = next(iter(patch_tokens.values())).shape
        if b_vis != b:
            raise ValueError(f"patch_tokens batch {b_vis} != action_tokens batch {b}")

        # TCP condition tokens: train with GT, test with predicted TCP.
        tcp_kv = None
        if not self.disable_tcp_kv:
            if use_gt_tcp:
                if gt_tcp_uv is None or gt_tcp_3d is None or gt_tcp_6d is None:
                    raise ValueError("use_gt_tcp=True requires gt_tcp_uv/gt_tcp_3d/gt_tcp_6d")
                tcp_kv = self._build_tcp_kv(b, t_steps, gt_tcp_uv, gt_tcp_3d, gt_tcp_6d, gt_tcp_valid)
            else:
                if pred_tcp_uv is None or pred_tcp_3d is None or pred_tcp_6d is None:
                    raise ValueError("use_gt_tcp=False requires pred_tcp_uv/pred_tcp_3d/pred_tcp_6d")
                tcp_kv = self._build_tcp_kv(b, t_steps, pred_tcp_uv, pred_tcp_3d, pred_tcp_6d, pred_tcp_valid)

            cond_kv = torch.cat([cond_kv, tcp_kv], dim=1)  # [B, Lcond, D]

        # State conditioning — per obs frame: pose + grip tokens (2·T_obs total).
        if self.use_state and state is not None:
            if state.dim() == 2:
                expected_flat = self.obs_hist_length * self.state_dim
                if state.shape[1] != expected_flat:
                    raise ValueError(
                        f"state flat shape mismatch: expected [B, {expected_flat}], got {tuple(state.shape)}"
                    )
                state = state.view(state.shape[0], self.obs_hist_length, self.state_dim)
            elif state.dim() == 3:
                if state.shape[1] != self.obs_hist_length or state.shape[2] != self.state_dim:
                    raise ValueError(
                        f"state shape mismatch: expected [B, {self.obs_hist_length}, {self.state_dim}], "
                        f"got {tuple(state.shape)}"
                    )
            else:
                raise ValueError(f"state must be rank-2 or rank-3 tensor, got shape={tuple(state.shape)}")

            s_tokens = self._encode_state_tokens(state.to(dtype=cond_kv.dtype))
            cond_kv = torch.cat([cond_kv, s_tokens], dim=1)  # [B, Lcond+2*T_obs, D]

        if cond_kv.shape[1] != self.cond_pos_enc.shape[1]:
            raise ValueError(
                f"cond_kv length {cond_kv.shape[1]} != cond_pos_enc {self.cond_pos_enc.shape[1]}; "
                f"check num_cameras={self.num_cameras}, obs_hist_length={self.obs_hist_length}, "
                f"use_last_frame_visual={self.use_last_frame_visual}, "
                f"num_spatial_patches_per_cam={self.num_spatial_patches_per_cam}, disable_tcp_kv={self.disable_tcp_kv}, "
                f"use_state={self.use_state} (and patch token H*W vs config)."
            )
        cond_kv = cond_kv + self.cond_pos_enc
        return cond_kv, tcp_kv

    def _build_cross_kv_cache(
        self, cond_kv: torch.Tensor
    ) -> tuple[Optional[tuple[torch.Tensor, torch.Tensor]], ...]:
        """Project per-layer cross-attention K/V once; cond does not depend on t."""
        return tuple(blk.project_cross_kv(cond_kv) for blk in self.blocks)

    def _step_fn(self):
        """``_forward_cond`` (eager) or its torch.compile'd version, compiled lazily per instance."""
        if not self.compile_step:
            return self._forward_cond
        k = (id(self), "dit_forward_cond")
        if k not in _COMPILED_STEP_FNS:
            _COMPILED_STEP_FNS[k] = torch.compile(self._forward_cond, dynamic=False, mode=self.compile_mode)
        return _COMPILED_STEP_FNS[k]

    def _forward_cond(
        self,
        action_tokens: torch.Tensor,  # [B, T_a, action_dim]
        timesteps: torch.Tensor,  # [B]
        cond_kv: torch.Tensor,  # [B, L_cond, D] from ``_build_cond_kv``
        tcp_kv: Optional[torch.Tensor],  # [B, L_tcp, D] or None
        cross_kv_cache: Optional[tuple[Optional[tuple[torch.Tensor, torch.Tensor]], ...]] = None,
    ) -> torch.Tensor:
        """Run the action stream (AdaLN DiT blocks + cross-attn) on pre-built conditioning KV."""
        b, ta, adim = action_tokens.shape
        if adim != self.action_dim:
            raise ValueError(f"action_tokens last dim {adim} != action_dim {self.action_dim}")
        if cond_kv.shape[0] != b:
            raise ValueError(f"cond_kv batch {cond_kv.shape[0]} != action_tokens batch {b}")

        # Action stream: [B, T_a, A] -> [B, 2·T_a, D]; TCP bias then shared per-step position embed.
        x = self._encode_action_tokens(action_tokens)
        seq_len = self._num_part_tokens * ta
        if not self.disable_tcp_kv and not self.disable_tcp_additive_bias:
            x = x + self._tcp_additive_gate_bias(tcp_kv, ta, b)
        x = x + self._action_step_pos_embed(ta, x.device)

        # Flow time -> AdaLN modulation ``e``; RoPE length matches 2·T_a action tokens.
        e = self.timestep_mlp(self.timestep_embed(timesteps)).view(b, 1, 9, self.embed_dim)
        freqs = self._create_action_freqs(seq_len, x.device)

        if cross_kv_cache is None:
            for blk in self.blocks:
                x = blk(x, e=e, cond=cond_kv, freqs=freqs)
        else:
            if len(cross_kv_cache) != len(self.blocks):
                raise ValueError(
                    f"cross_kv_cache length {len(cross_kv_cache)} != num blocks {len(self.blocks)}"
                )
            for blk, block_kv in zip(self.blocks, cross_kv_cache):
                x = blk(x, e=e, cond=cond_kv, freqs=freqs, cross_kv=block_kv)

        x = self.final_norm(x)
        return self._decode_action_tokens(x, ta)

    def forward_training(
        self,
        actions: torch.Tensor,  # [B, T_a, action_dim] 干净动作轨迹
        patch_tokens: Dict[str, torch.Tensor],  # cam -> [B, C, T_obs, H, W]
        depths: Optional[Dict[str, torch.Tensor]],  # cam -> [B, 1, T_obs, H_d, W_d]
        camera_intrinsics: Optional[Dict[str, torch.Tensor]],  # cam -> [B, T_obs, 4]
        gt_tcp_uv: Dict[str, torch.Tensor],  # cam -> [B, T_obs, 2]（SAPolicy `_build_dit_gt_tcp_dicts`）
        gt_tcp_3d: Dict[str, torch.Tensor],  # cam -> [B, T_obs, 3]
        gt_tcp_6d: Dict[str, torch.Tensor],  # cam -> [B, T_obs, 6]
        gt_tcp_valid: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs] bool
        preset_t: Optional[torch.Tensor] = None,  # [B] 或标量广播；覆盖随机 t
        preset_noise: Optional[torch.Tensor] = None,  # [B, T_a, action_dim]；覆盖随机噪声
        state: Optional[torch.Tensor] = None,  # [B, state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flow-matching supervision: ``x_t = (1-t)·x_0 + t·ε``, target ``v = ε - x_0``.

        ``preset_t`` / ``preset_noise`` (e.g. from a second forward for consistency loss) override sampling.
        Returns tuple ``(pred_flow, target_flow, x_t, loss, t, noise)`` with ``loss = MSE(pred, target)``.

        With ``num_diffusion_draws = k > 1`` in train mode every returned tensor has batch ``k·B``
        (row ``i·k + j`` = draw ``j`` of sample ``i``); ``preset_*`` of batch ``B`` are repeated too.
        """
        b, _, _ = actions.shape
        device = actions.device
        dtype = actions.dtype

        # Conditioning (visual / TCP / state) does not depend on (t, ε): encode once at batch B.
        cond_kv, tcp_kv = self._build_cond_kv(
            b,
            patch_tokens,
            depths=depths,
            camera_intrinsics=camera_intrinsics,
            gt_tcp_uv=gt_tcp_uv,
            gt_tcp_3d=gt_tcp_3d,
            gt_tcp_6d=gt_tcp_6d,
            gt_tcp_valid=gt_tcp_valid,
            use_gt_tcp=True,
            state=state,
        )

        if preset_t is None:
            preset_t = getattr(self, "_preset_t", None)
        if preset_noise is None:
            preset_noise = getattr(self, "_preset_noise", None)

        k = self.num_diffusion_draws if self.training else 1
        if k > 1:
            actions = actions.repeat_interleave(k, dim=0)
            cond_kv = cond_kv.repeat_interleave(k, dim=0)
            if tcp_kv is not None:
                tcp_kv = tcp_kv.repeat_interleave(k, dim=0)
            if preset_t is not None and preset_t.dim() >= 1 and preset_t.shape[0] == b:
                preset_t = preset_t.repeat_interleave(k, dim=0)
            if preset_noise is not None and preset_noise.shape[0] == b:
                preset_noise = preset_noise.repeat_interleave(k, dim=0)
            b = b * k

        if preset_t is not None:
            t = preset_t.to(device=device, dtype=dtype)
        else:
            t = torch.rand(b, device=device, dtype=dtype) * 0.998 + 0.001

        noise = preset_noise if preset_noise is not None else torch.randn_like(actions)
        t_e = t.view(b, 1, 1)
        a_t = (1.0 - t_e) * actions + t_e * noise
        target_flow = noise - actions

        pred = self._step_fn()(a_t, t, cond_kv, tcp_kv)
        loss = F.mse_loss(pred, target_flow)
        return pred, target_flow, a_t, loss, t, noise

    @contextmanager
    def rtc_condition(self, condition, weights, beta):
        """Scope one normalized RTC condition to a serialized inference call."""
        if getattr(self, "_rtc_sampling", None) is not None:
            raise RuntimeError("An RTC condition is already active")
        if condition.shape != (1, self.sequence_length, self.action_dim):
            raise ValueError("RTC condition must match the DiT action horizon and dimension")
        if weights.shape != (1, self.sequence_length, 1):
            raise ValueError("RTC weights must contain one weight per action step")
        if not torch.isfinite(condition).all() or not torch.isfinite(weights).all():
            raise ValueError("RTC condition and weights must be finite")
        if not ((weights >= 0) & (weights <= 1)).all() or not 0 < beta < float("inf"):
            raise ValueError("RTC weights must be in [0,1] and beta must be positive and finite")
        # An empty mask uses exactly the ordinary sampler, including its RNG path.
        self._rtc_sampling = (condition, weights, beta) if torch.any(weights > 0) else None
        try:
            yield
        finally:
            self._rtc_sampling = None

    @torch.no_grad()
    def sample_trajectory(
        self,
        patch_tokens: Dict[str, torch.Tensor],  # cam -> [B, C, T_obs, H, W]
        depths: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, 1, T_obs, H_d, W_d]
        camera_intrinsics: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 4]
        pred_tcp_uv: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 2]（SAPolicy `_tcp_stacked_per_cam_dict`）
        pred_tcp_3d: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 3]
        pred_tcp_6d: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs, 6]
        pred_tcp_valid: Optional[Dict[str, torch.Tensor]] = None,  # cam -> [B, T_obs] 或 [B, T_obs, 1]；可选
        num_samples: int = 1,
        state: Optional[torch.Tensor] = None,  # [B, state_dim]
    ) -> torch.Tensor:
        """Generate actions by integrating the learned field from pure noise (``t≈1``) to data (``t≈0``).

        Uses fixed-step Euler: ``x ← x + Δt · v(x, t)`` with ``Δt = -1 / num_inference_steps``.
        TCP conditioning comes from TCP-head predictions (``pred_tcp_*``), not GT.

        Returns:
            ``[B, T_a, action_dim]`` with ``T_a = sequence_length``.
        """
        if not self.disable_tcp_kv:
            if pred_tcp_uv is None or pred_tcp_3d is None or pred_tcp_6d is None:
                raise ValueError("DiT sampling requires pred_tcp_uv / pred_tcp_3d / pred_tcp_6d (e.g. from TCP head).")

        device = next(iter(patch_tokens.values())).device
        dtype = next(iter(patch_tokens.values())).dtype
        b = next(iter(patch_tokens.values())).shape[0]
        ta = self.sequence_length
        d = self.action_dim

        if num_samples != 1:
            raise NotImplementedError("num_samples > 1 not implemented for DiTActionHead")

        cond_kv, tcp_kv = self._build_cond_kv(
            b,
            patch_tokens,
            depths=depths,
            camera_intrinsics=camera_intrinsics,
            pred_tcp_uv=pred_tcp_uv,
            pred_tcp_3d=pred_tcp_3d,
            pred_tcp_6d=pred_tcp_6d,
            pred_tcp_valid=pred_tcp_valid,
            use_gt_tcp=False,
            state=state,
        )
        cross_kv_cache = self._build_cross_kv_cache(cond_kv)
        rtc = getattr(self, "_rtc_sampling", None)
        # The default compiled forward is inference-only; RTC differentiates the
        # action input through the eager field while reusing constant visual KV.
        step = self._step_fn() if rtc is None else self._forward_cond
        actions = torch.randn(b, ta, d, device=device, dtype=dtype)
        dt = -1.0 / float(self.num_inference_steps)
        t_curr = 1.0
        while t_curr >= -dt / 2:
            t_vec = torch.full((b,), t_curr, device=device, dtype=dtype)
            if rtc is None:
                v = step(actions, t_vec, cond_kv, tcp_kv, cross_kv_cache)
            else:
                condition, weights, beta = rtc
                v = guided_velocity(
                    lambda sample: step(sample, t_vec, cond_kv, tcp_kv, cross_kv_cache),
                    actions, t_curr, condition.to(actions), weights.to(actions), beta,
                )
            actions = actions + dt * v
            t_curr += dt
        return actions
