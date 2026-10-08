"""
Flux2ActionTransformer2DModel: a three-stream (image + text + action) VLA head built on
top of the FLUX.2 hybrid MMDiT backbone, mirroring the ImageWAM-aligned design that the
Qwen version (transformer_qwen_image_action_v9.py) uses.

Why this file exists / key design decisions
--------------------------------------------
FLUX.2 is a HYBRID backbone: `num_layers` DOUBLE-stream blocks (image + text run with
their own params, joint attention) followed by `num_single_layers` SINGLE-stream blocks
(image and text are concatenated into one sequence and share one set of params). This is
different from Qwen-Image-Edit, which is double-stream all the way.

The action stream does NOT care whether the backbone layer is double or single. We treat
it as a THIRD, independent bottleneck expert that is present in EVERY layer:

  * it keeps its own (smaller) residual width `action_inner_dim`;
  * every layer up-projects action -> inner_dim to produce Q/K/V, joins the attention,
    then down-projects the attention output back to `action_inner_dim`;
  * the image/text schedule (double for the first `num_layers`, single afterwards) is left
    completely untouched, so the FLUX pretrained weights load as-is.

Because the double blocks build the joint sequence as [txt, img, action] and the single
blocks receive main=[txt, img] (already concatenated at the model level) and then append
action, the joint token order is IDENTICAL in both: [txt, target, context, action]. So a
SINGLE ImageWAM visibility mask is built once and reused for all blocks.

ImageWAM visibility mask (identical rules to the Qwen v9 model):

  query \\ key   | text+state | noisy target | clean context | action
  text+state    |     T      |      F       |       T       |   F
  noisy target  |     T      |      T       |       T       |   F
  clean context |     T      |      F       |       T       |   F
  action        |     T      |      F       |       T       |   T

1. NOTHING attends to action tokens (their noise level differs from the image sigma).
2. The clean prefix (text+state, context) never attends the noisy target -> its
   representations are noise-independent (enables ImageWAM KV-cache action-only decode).
3. Action attends only to the clean prefix (text+state, context) + itself.

zero_cond_t: the context (clean history) frames are modulated with a t=0 timestep while the
target uses the real timestep. FLUX modulation is produced globally (shared across layers),
so we compute both the real-t and zero-t modulations once and blend them per-token.
"""

from typing import Optional

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import logging

# Reuse the FLUX.2 primitives verbatim (identical class names -> pretrained weights load).
from diffusers.models.transformers.transformer_flux2 import (
    Flux2FeedForward,
    Flux2Modulation,
    Flux2PosEmbed,
    Flux2SwiGLU,
    Flux2TimestepGuidanceEmbeddings,
)

logger = logging.get_logger(__name__)


def _split3(mod):
    """Split a modulation tensor [..., 3C] -> (shift, scale, gate), each [..., C]."""
    return mod.chunk(3, dim=-1)


def _blend_pertoken(mod_real, mod_zero, zero_mask):
    """Per-token blend of two global modulations for zero_cond_t.

    mod_real / mod_zero: [B, C] modulation for real-t / t=0.
    zero_mask: [B, S] bool, True where the token should use the t=0 modulation
               (i.e. clean context frames).
    returns [B, S, C].
    """
    mr = mod_real.unsqueeze(1)
    mz = mod_zero.unsqueeze(1)
    return torch.where(zero_mask.unsqueeze(-1), mz, mr)


class Flux2ActionDoubleBlock(nn.Module):
    """FLUX.2 double-stream block (image + text, FLUX-compatible naming) + action expert.

    image/text sub-modules keep the exact FLUX names (norm1, norm1_context, attn.to_q/... ,
    ff, ff_context) so `transformer_blocks.{i}.*` pretrained weights load unchanged. The
    action expert (act_*) is new and bottlenecked at `action_inner_dim`.
    """

    def __init__(self, dim, num_heads, head_dim, action_inner_dim, mlp_ratio=3.0, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.action_inner_dim = action_inner_dim
        inner = num_heads * head_dim

        # ---- image stream (FLUX names) ----
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=False)

        # ---- text stream (FLUX names: add_*_proj / to_add_out / norm_added_*) ----
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.add_q_proj = nn.Linear(dim, inner, bias=False)
        self.add_k_proj = nn.Linear(dim, inner, bias=False)
        self.add_v_proj = nn.Linear(dim, inner, bias=False)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_add_out = nn.Linear(inner, dim, bias=False)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=False)

        # ---- action expert (NEW, bottleneck) ----
        self.act_norm1 = nn.LayerNorm(action_inner_dim, elementwise_affine=False, eps=eps)
        self.act_to_q = nn.Linear(action_inner_dim, inner, bias=False)
        self.act_to_k = nn.Linear(action_inner_dim, inner, bias=False)
        self.act_to_v = nn.Linear(action_inner_dim, inner, bias=False)
        self.act_norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.act_norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.act_to_out = nn.Linear(inner, action_inner_dim, bias=False)   # down-proj
        self.act_norm2 = nn.LayerNorm(action_inner_dim, elementwise_affine=False, eps=eps)
        self.act_ff = Flux2FeedForward(dim=action_inner_dim, dim_out=action_inner_dim, mult=mlp_ratio, bias=False)

    def _heads(self, x):
        return x.unflatten(-1, (self.num_heads, -1))

    def forward(self, img, txt, act, img_mod, txt_mod, act_mod,
                img_rotary, txt_rotary, attn_mask):
        # img_mod: [B, S_img, 6C] (per-token, zero_cond_t blended); txt_mod/act_mod: [B,1,6C]
        (i_sh1, i_sc1, i_g1), (i_sh2, i_sc2, i_g2) = _split_two_sets(img_mod)
        (t_sh1, t_sc1, t_g1), (t_sh2, t_sc2, t_g2) = _split_two_sets(txt_mod)
        (a_sh1, a_sc1, a_g1), (a_sh2, a_sc2, a_g2) = _split_two_sets(act_mod)

        # ---- pre-attention norm + modulate ----
        img_n = self.norm1(img) * (1 + i_sc1) + i_sh1
        txt_n = self.norm1_context(txt) * (1 + t_sc1) + t_sh1
        act_n = self.act_norm1(act) * (1 + a_sc1) + a_sh1

        # ---- projections ----
        iq, ik, iv = self._heads(self.to_q(img_n)), self._heads(self.to_k(img_n)), self._heads(self.to_v(img_n))
        tq, tk, tv = self._heads(self.add_q_proj(txt_n)), self._heads(self.add_k_proj(txt_n)), self._heads(self.add_v_proj(txt_n))
        aq, ak, av = self._heads(self.act_to_q(act_n)), self._heads(self.act_to_k(act_n)), self._heads(self.act_to_v(act_n))

        iq, ik = self.norm_q(iq), self.norm_k(ik)
        tq, tk = self.norm_added_q(tq), self.norm_added_k(tk)
        aq, ak = self.act_norm_q(aq), self.act_norm_k(ak)

        # RoPE on img/txt only; action carries no positional rotation.
        if img_rotary is not None:
            iq = apply_rotary_emb(iq, img_rotary, sequence_dim=1)
            ik = apply_rotary_emb(ik, img_rotary, sequence_dim=1)
        if txt_rotary is not None:
            tq = apply_rotary_emb(tq, txt_rotary, sequence_dim=1)
            tk = apply_rotary_emb(tk, txt_rotary, sequence_dim=1)

        # joint order [txt, img, action]
        q = torch.cat([tq, iq, aq], dim=1)
        k = torch.cat([tk, ik, ak], dim=1)
        v = torch.cat([tv, iv, av], dim=1)
        out = dispatch_attention_fn(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.flatten(2, 3).to(q.dtype)

        s_txt, s_img = txt.shape[1], img.shape[1]
        t_out = self.to_add_out(out[:, :s_txt])
        i_out = self.to_out(out[:, s_txt:s_txt + s_img])
        a_out = self.act_to_out(out[:, s_txt + s_img:])

        img = img + i_g1 * i_out
        txt = txt + t_g1 * t_out
        act = act + a_g1 * a_out

        # ---- FFN ----
        img = img + i_g2 * self.ff(self.norm2(img) * (1 + i_sc2) + i_sh2)
        txt = txt + t_g2 * self.ff_context(self.norm2_context(txt) * (1 + t_sc2) + t_sh2)
        act = act + a_g2 * self.act_ff(self.act_norm2(act) * (1 + a_sc2) + a_sh2)
        return img, txt, act


class Flux2ActionSingleBlock(nn.Module):
    """FLUX.2 single-stream parallel block (main = txt+img) + parallel action expert.

    The main stream keeps the exact FLUX single-block fused projection
    (`to_qkv_mlp_proj` / `to_out`) so `single_transformer_blocks.{i}.attn.*` weights load.
    Action gets its own parallel fused projection at `action_inner_dim`.
    """

    def __init__(self, dim, num_heads, head_dim, action_inner_dim, mlp_ratio=3.0, eps=1e-6, mlp_mult=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.action_inner_dim = action_inner_dim
        inner = num_heads * head_dim
        self.inner = inner
        self.mlp_hidden = int(dim * mlp_ratio)
        self.act_mlp_hidden = int(action_inner_dim * mlp_ratio)
        self.mlp_mult = mlp_mult

        # ---- main stream (FLUX names under .attn) ----
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = _FluxSingleAttnParams(dim, inner, head_dim, self.mlp_hidden, mlp_mult, eps)

        # ---- action parallel expert (NEW) ----
        self.act_norm = nn.LayerNorm(action_inner_dim, elementwise_affine=False, eps=eps)
        self.act_to_qkv_mlp_proj = nn.Linear(action_inner_dim, inner * 3 + self.act_mlp_hidden * mlp_mult, bias=False)
        self.act_norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.act_norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.act_mlp_act = Flux2SwiGLU()
        self.act_to_out = nn.Linear(inner + self.act_mlp_hidden, action_inner_dim, bias=False)

    def _heads(self, x):
        return x.unflatten(-1, (self.num_heads, -1))

    def forward(self, main, act, main_mod, act_mod, main_rotary, attn_mask):
        # main_mod: [B, S_main, 3C] per-token (zero_cond_t blended); act_mod: [B,1,3C]
        m_sh, m_sc, m_g = _split3(main_mod)
        a_sh, a_sc, a_g = _split3(act_mod)

        m_n = self.norm(main) * (1 + m_sc) + m_sh
        a_n = self.act_norm(act) * (1 + a_sc) + a_sh

        # main fused proj -> qkv + mlp
        m_proj = self.attn.to_qkv_mlp_proj(m_n)
        m_qkv, m_mlp = torch.split(m_proj, [3 * self.inner, self.mlp_hidden * self.mlp_mult], dim=-1)
        mq, mk, mv = m_qkv.chunk(3, dim=-1)
        mq, mk, mv = self._heads(mq), self._heads(mk), self._heads(mv)
        mq, mk = self.attn.norm_q(mq), self.attn.norm_k(mk)
        if main_rotary is not None:
            mq = apply_rotary_emb(mq, main_rotary, sequence_dim=1)
            mk = apply_rotary_emb(mk, main_rotary, sequence_dim=1)

        # action fused proj -> qkv + mlp
        a_proj = self.act_to_qkv_mlp_proj(a_n)
        a_qkv, a_mlp = torch.split(a_proj, [3 * self.inner, self.act_mlp_hidden * self.mlp_mult], dim=-1)
        aq, ak, av = a_qkv.chunk(3, dim=-1)
        aq, ak, av = self._heads(aq), self._heads(ak), self._heads(av)
        aq, ak = self.act_norm_q(aq), self.act_norm_k(ak)

        # joint order [main(=txt,img), action]
        q = torch.cat([mq, aq], dim=1)
        k = torch.cat([mk, ak], dim=1)
        v = torch.cat([mv, av], dim=1)
        out = dispatch_attention_fn(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.flatten(2, 3).to(q.dtype)

        s_main = main.shape[1]
        m_attn = out[:, :s_main]
        a_attn = out[:, s_main:]

        m_out = self.attn.to_out(torch.cat([m_attn, self.attn.mlp_act(m_mlp)], dim=-1))
        a_out = self.act_to_out(torch.cat([a_attn, self.act_mlp_act(a_mlp)], dim=-1))

        main = main + m_g * m_out
        act = act + a_g * a_out
        return main, act


class _FluxSingleAttnParams(nn.Module):
    """Holds the FLUX single-block fused params under `.attn` so that
    `single_transformer_blocks.{i}.attn.to_qkv_mlp_proj / to_out / norm_q / norm_k`
    pretrained weights map 1:1."""

    def __init__(self, dim, inner, head_dim, mlp_hidden, mlp_mult, eps):
        super().__init__()
        self.to_qkv_mlp_proj = nn.Linear(dim, inner * 3 + mlp_hidden * mlp_mult, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(inner + mlp_hidden, dim, bias=False)
        self.mlp_act = Flux2SwiGLU()


def _split_two_sets(mod):
    """Split a 2-set modulation [.., 6C] into ((sh1,sc1,g1),(sh2,sc2,g2))."""
    a, b = mod[..., : mod.shape[-1] // 2], mod[..., mod.shape[-1] // 2:]
    return _split3(a), _split3(b)


class Flux2ActionTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """Three-stream (image + text + action) VLA transformer on the FLUX.2 hybrid backbone."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux2ActionDoubleBlock", "Flux2ActionSingleBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 5,
        num_single_layers: int = 20,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 7680,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = False,
        action_dim: int = 7,
        action_horizon: int = 32,
        state_dim: int = 8,
        action_inner_dim: int = 3072,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.action_inner_dim = action_inner_dim
        self.zero_cond_t = True

        # ---- image/text backbone (FLUX names -> pretrained load) ----
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels, embedding_dim=self.inner_dim,
            bias=False, guidance_embeds=guidance_embeds,
        )
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList([
            Flux2ActionDoubleBlock(self.inner_dim, num_attention_heads, attention_head_dim,
                                   action_inner_dim, mlp_ratio, eps)
            for _ in range(num_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            Flux2ActionSingleBlock(self.inner_dim, num_attention_heads, attention_head_dim,
                                   action_inner_dim, mlp_ratio, eps)
            for _ in range(num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        # ---- action stream (NEW) ----
        # action modulation (global, mirrors FLUX's shared-modulation design)
        self.act_temb_proj = nn.Linear(self.inner_dim, action_inner_dim, bias=False)
        self.double_stream_modulation_act = Flux2Modulation(action_inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation_act = Flux2Modulation(action_inner_dim, mod_param_sets=1, bias=False)
        self.action_in = nn.Sequential(
            nn.Linear(action_dim, action_inner_dim), nn.SiLU(),
            nn.Linear(action_inner_dim, action_inner_dim),
        )
        self.action_pos_embed = nn.Parameter(torch.randn(1, action_horizon, action_inner_dim) * 0.02)
        self.action_out_norm = nn.LayerNorm(action_inner_dim, eps=eps)
        self.action_out = nn.Linear(action_inner_dim, action_dim)
        # state token injected into the text stream (ImageWAM/v9 style)
        self.state_in = nn.Sequential(
            nn.Linear(state_dim, self.inner_dim), nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        )
        # dual-camera context view embedding
        self.view_embed = nn.Embedding(2, self.inner_dim)
        nn.init.zeros_(self.view_embed.weight)

        self.gradient_checkpointing = False
        self.checkpoint_stride = 1

    @staticmethod
    def build_rope_ids(target_grid, ctx_grids, txt_len, device, scale: int = 10):
        """Construct FLUX.2 4-axis (T, H, W, L) RoPE ids for the VLA layout.

        FLUX.2 uses 4 coordinate axes: T = frame/time index, H/W = patch row/col, L = the
        text-sequence axis. Images live on (T,H,W) with L=0; text lives on L with T=H=W=0
        (matches the pipeline's `_prepare_latent_ids` / `_prepare_text_ids`). Different frames
        are separated on the T axis: target = 0, context frame i = scale*(i+1). Patch order is
        row-major (h outer, w inner) to match `_pack_latents`.

        Args:
            target_grid: (h, w) patchified grid of the target frame.
            ctx_grids: list of (h, w) for each context frame (view/history), or None.
            txt_len: number of text tokens INCLUDING the appended state token.
        Returns:
            img_ids [n_img_patches, 4], txt_ids [txt_len, 4]  (both long, on `device`).
        """
        def frame_ids(t, hw):
            h, w = hw
            return torch.cartesian_prod(
                torch.tensor([t]), torch.arange(h), torch.arange(w), torch.arange(1)
            )
        blocks = [frame_ids(0, target_grid)]
        for i, hw in enumerate(ctx_grids or []):
            blocks.append(frame_ids(scale * (i + 1), hw))
        img_ids = torch.cat(blocks, dim=0).to(device)
        txt_ids = torch.cartesian_prod(
            torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(txt_len)
        ).to(device)
        return img_ids, txt_ids

    # ---- ImageWAM visibility mask over [txt(+state), target, context, action] ----
    def _build_mask(self, seq_txt, target_len, ctx_len, seq_act, device, text_valid_mask=None):
        total = seq_txt + target_len + ctx_len + seq_act
        t0 = seq_txt
        t1 = seq_txt + target_len
        c1 = seq_txt + target_len + ctx_len
        m = torch.zeros(1, 1, total, total, dtype=torch.bool, device=device)
        # text+state rows: text+state, context
        m[:, :, :t0, :t0] = True
        m[:, :, :t0, t1:c1] = True
        # noisy target rows: text+state, target, context (all but action)
        m[:, :, t0:t1, :c1] = True
        # clean context rows: text+state, context
        m[:, :, t1:c1, :t0] = True
        m[:, :, t1:c1, t1:c1] = True
        # action rows: text+state, context, action
        m[:, :, c1:, :t0] = True
        m[:, :, c1:, t1:c1] = True
        m[:, :, c1:, c1:] = True
        if text_valid_mask is not None:
            # padded text tokens (reasoner left-padding): nothing attends them.
            # text occupies joint-sequence cols [0:seq_txt].
            tv = text_valid_mask.to(device=device, dtype=torch.bool)  # [B, seq_txt]
            Bt = tv.shape[0]
            m = m.expand(Bt, 1, total, total).clone()
            m[:, :, :, :seq_txt] = m[:, :, :, :seq_txt] & tv[:, None, None, :]
        return m

    def forward(
        self,
        hidden_states: torch.Tensor,          # noisy target [B, S_target, in_channels]
        encoder_hidden_states: torch.Tensor,  # text [B, S_txt, joint_attention_dim]
        timestep: torch.Tensor = None,        # [B]
        img_ids: torch.Tensor = None,         # [S_target+S_ctx, len(axes_dims_rope)]
        txt_ids: torch.Tensor = None,         # [S_txt(+1 state), len(axes_dims_rope)]
        noisy_actions: torch.Tensor = None,   # [B, action_horizon, action_dim]
        state_tokens: torch.Tensor = None,    # [B, state_dim]
        context_hidden_states: torch.Tensor = None,  # [B, S_ctx, in_channels]
        context_view_ids: torch.Tensor = None,       # [B, S_ctx]
        action_timestep: torch.Tensor = None,        # [B] independent action sigma
        guidance: torch.Tensor = None,
        text_valid_mask: torch.Tensor = None,        # [B, seq_txt] bool, False=padded text token
        return_dict: bool = True,
    ):
        dtype = self.x_embedder.weight.dtype
        hidden_states = hidden_states.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)
        noisy_actions = noisy_actions.to(dtype)
        state_tokens = state_tokens.to(dtype)
        if context_hidden_states is not None:
            context_hidden_states = context_hidden_states.to(dtype)

        B = hidden_states.shape[0]
        target_len = hidden_states.shape[1]
        ctx_len = context_hidden_states.shape[1] if context_hidden_states is not None else 0

        # ---- image stream: [target, context] ----
        if context_hidden_states is not None:
            img = torch.cat([hidden_states, context_hidden_states], dim=1)
        else:
            img = hidden_states
        img = self.x_embedder(img)
        if context_hidden_states is not None and context_view_ids is not None:
            img[:, target_len:] = img[:, target_len:] + self.view_embed(context_view_ids)
        seq_img = img.shape[1]

        # ---- text stream + state token ----
        txt = self.context_embedder(encoder_hidden_states)
        state_emb = self.state_in(state_tokens).unsqueeze(1)
        txt = torch.cat([txt, state_emb], dim=1)
        seq_txt = txt.shape[1]

        # ---- action stream ----
        act = self.action_in(noisy_actions)
        act = act + self.action_pos_embed[:, : act.shape[1]]
        seq_act = act.shape[1]

        # ---- timestep + modulation (global, FLUX style) ----
        if timestep is not None:
            timestep = timestep.to(dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        act_temb_src = self.time_guidance_embed(action_timestep.to(dtype) * 1000, guidance) if action_timestep is not None else temb
        act_temb = self.act_temb_proj(act_temb_src)

        # double-stream modulations
        dm_img_real = self.double_stream_modulation_img(temb)   # [B, 6C]
        dm_txt = self.double_stream_modulation_txt(temb)        # [B, 6C]
        dm_act = self.double_stream_modulation_act(act_temb)    # [B, 6Ca]
        sm_main_real = self.single_stream_modulation(temb)      # [B, 3C]
        sm_act = self.single_stream_modulation_act(act_temb)    # [B, 3Ca]

        # zero_cond_t: context frames use t=0 modulation, blended per-token.
        if self.zero_cond_t and ctx_len > 0:
            temb_zero = self.time_guidance_embed(torch.zeros_like(timestep), guidance)
            dm_img_zero = self.double_stream_modulation_img(temb_zero)
            sm_main_zero = self.single_stream_modulation(temb_zero)
            # image stream tokens: [target(real), context(zero)]
            img_zero_mask = torch.zeros(B, seq_img, dtype=torch.bool, device=img.device)
            img_zero_mask[:, target_len:] = True
            dm_img = _blend_pertoken(dm_img_real, dm_img_zero, img_zero_mask)   # [B, S_img, 6C]
            # main stream (single blocks) tokens: [txt(real), target(real), context(zero)]
            main_zero_mask = torch.zeros(B, seq_txt + seq_img, dtype=torch.bool, device=img.device)
            main_zero_mask[:, seq_txt + target_len:] = True
            sm_main = _blend_pertoken(sm_main_real, sm_main_zero, main_zero_mask)  # [B, S_main, 3C]
        else:
            dm_img = dm_img_real.unsqueeze(1)      # [B,1,6C] broadcast
            sm_main = sm_main_real.unsqueeze(1)    # [B,1,3C]
        dm_txt = dm_txt.unsqueeze(1)
        dm_act = dm_act.unsqueeze(1)
        sm_act = sm_act.unsqueeze(1)

        # ---- RoPE (img + txt only) ----
        img_rotary = self.pos_embed(img_ids) if img_ids is not None else None
        txt_rotary = self.pos_embed(txt_ids) if txt_ids is not None else None

        # ---- single ImageWAM mask, reused for all blocks (order [txt, target, ctx, action]) ----
        attn_mask = self._build_mask(seq_txt, target_len, ctx_len, seq_act, img.device, text_valid_mask=text_valid_mask)

        # ---- double-stream blocks ----
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and i % self.checkpoint_stride == 0:
                img, txt, act = torch.utils.checkpoint.checkpoint(
                    block, img, txt, act, dm_img, dm_txt, dm_act, img_rotary, txt_rotary, attn_mask,
                    use_reentrant=False,
                )
            else:
                img, txt, act = block(img, txt, act, dm_img, dm_txt, dm_act, img_rotary, txt_rotary, attn_mask)

        # ---- merge to single stream: main = [txt, img] ----
        main = torch.cat([txt, img], dim=1)
        # RoPE for main = [txt, img]
        if txt_rotary is not None and img_rotary is not None:
            main_rotary = (torch.cat([txt_rotary[0], img_rotary[0]], dim=0),
                           torch.cat([txt_rotary[1], img_rotary[1]], dim=0))
        else:
            main_rotary = None

        for i, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and i % self.checkpoint_stride == 0:
                main, act = torch.utils.checkpoint.checkpoint(
                    block, main, act, sm_main, sm_act, main_rotary, attn_mask, use_reentrant=False,
                )
            else:
                main, act = block(main, act, sm_main, sm_act, main_rotary, attn_mask)

        # ---- split main back, take target tokens ----
        img = main[:, seq_txt:]
        target = img[:, :target_len]

        target = self.norm_out(target, temb)
        image_output = self.proj_out(target)

        action_output = self.action_out(self.action_out_norm(act))

        if not return_dict:
            return (image_output, action_output)
        return Transformer2DModelOutput(sample=image_output)


def _remap_flux2_key(k: str) -> str:
    """Map an ORIGINAL FLUX.2 transformer key to this model's key.

    Only the DOUBLE blocks differ: FLUX nests the image/text attention under `.attn.` and
    stores the image output proj as a ModuleList (`to_out.0`). We keep those projections
    flat on the block and use a single `to_out` Linear. Single blocks and all model-level
    modules already share FLUX's naming, so they pass through unchanged.
    """
    if k.startswith("transformer_blocks."):
        parts = k.split(".")
        # parts: ["transformer_blocks", idx, ...]
        rest = ".".join(parts[2:])
        if rest.startswith("attn.to_out.0."):
            rest = "to_out." + rest[len("attn.to_out.0."):]
        elif rest.startswith("attn."):
            rest = rest[len("attn."):]
        return f"transformer_blocks.{parts[1]}.{rest}"
    return k


@torch.no_grad()
def init_action_from_flux(model: "Flux2ActionTransformer2DModel"):
    """Warm-start the action expert from the (already-loaded) FLUX image/main stream.

    Mirrors the Qwen v9 `init_action_from_img`: the action expert starts as a copy of the
    pretrained image-stream projections so it inherits a sane feature space instead of
    training from scratch. Per-tensor no-op when shapes differ (i.e. bottleneck
    `action_inner_dim != inner_dim`), leaving those tensors at their random init -- same
    policy as Qwen v9 (which skips copy-init unless action_inner_dim == inner_dim).

    Call AFTER `load_flux2_pretrained`. Returns the number of tensors copied.
    """
    copied = 0

    def cp(dst, src):
        nonlocal copied
        if dst.weight.shape == src.weight.shape:
            dst.weight.data.copy_(src.weight.data)
            if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
                dst.bias.data.copy_(src.bias.data)
            copied += 1

    # double-stream blocks: action expert <- image stream
    for blk in model.transformer_blocks:
        cp(blk.act_to_q, blk.to_q)
        cp(blk.act_to_k, blk.to_k)
        cp(blk.act_to_v, blk.to_v)
        cp(blk.act_to_out, blk.to_out)          # action down-proj <- image out-proj
        cp(blk.act_norm_q, blk.norm_q)
        cp(blk.act_norm_k, blk.norm_k)
        cp(blk.act_ff.linear_in, blk.ff.linear_in)
        cp(blk.act_ff.linear_out, blk.ff.linear_out)

    # single-stream blocks: action parallel expert <- main fused proj
    for blk in model.single_transformer_blocks:
        cp(blk.act_to_qkv_mlp_proj, blk.attn.to_qkv_mlp_proj)
        cp(blk.act_to_out, blk.attn.to_out)
        cp(blk.act_norm_q, blk.attn.norm_q)
        cp(blk.act_norm_k, blk.attn.norm_k)

    # global action modulation <- image / main modulation
    cp(model.double_stream_modulation_act.linear, model.double_stream_modulation_img.linear)
    cp(model.single_stream_modulation_act.linear, model.single_stream_modulation.linear)

    # act_temb_proj: identity when square (action_inner_dim == inner_dim)
    w = model.act_temb_proj.weight
    if w.shape[0] == w.shape[1]:
        w.data.copy_(torch.eye(w.shape[0], dtype=w.dtype, device=w.device))
        copied += 1

    return copied


def load_flux2_pretrained(model: Flux2ActionTransformer2DModel, flux_transformer_dir: str, device: str = "cpu"):
    """Load the FLUX.2 pretrained image/text backbone into a Flux2ActionTransformer2DModel.

    Loads every shard under `<flux_transformer_dir>/*.safetensors`, remaps double-block keys,
    then `load_state_dict(strict=False)`. Returns (missing, unexpected): `unexpected` MUST be
    empty (every FLUX weight maps to a slot); `missing` should be ONLY the new action-expert /
    state / view modules (fresh init).
    """
    import glob
    import os
    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(flux_transformer_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {flux_transformer_dir}")
    remapped = {}
    for shard in shards:
        for k, v in load_file(shard, device=device).items():
            remapped[_remap_flux2_key(k)] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    return missing, unexpected
