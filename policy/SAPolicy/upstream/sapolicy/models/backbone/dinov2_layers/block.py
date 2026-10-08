# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import Tensor, nn, zero_
import torch.nn.functional as F

from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import fmha, index_select_cat, scaled_index_add

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()

# AdaLN
# img: B x (H/p x W/p + 1) x DIM
# lowres_depth: 1 x 1 x H x W
    return module


# channelwise: conv -> adaptive pooling -> linear -> B x 1 X 6xDIM
# pixelwise: conv -> resize -> flatten and concat adaptive pooling -> 1x1 conv  ->  B DIMx6
class AdaLN(nn.Module):
    def __init__(self, dim, channelwise: bool = True, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.channelwise = channelwise
        if channelwise:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(1, dim // 2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.zero_output_layer = zero_module(nn.Linear(dim, dim * 6))
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(1, dim // 2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
            self.zero_output_layer = zero_module(
                nn.Conv2d(dim, dim * 6, kernel_size=1, stride=1, padding=0)
            )
            self.zero_output_layer_ada = zero_module(nn.Linear(dim, dim * 6))

    def forward(self, lowres_depth: Tensor, tokens: Tensor) -> Tensor:
        x = lowres_depth
        x = self.feat_extract(x)
        if self.channelwise:
            x = F.adaptive_max_pool2d(x, output_size=1)
            x = self.zero_output_layer(x.squeeze((2, 3)))
            return x.unsqueeze(1).chunk(6, dim=-1)
        else:
            h_lowres, w_lowres = lowres_depth.shape[-2:]
            token_size = tokens.shape[1] - 1
            img_ratio = w_lowres / h_lowres
            token_h = int((token_size / img_ratio) ** 0.5)
            token_w = int(token_h * img_ratio)
            assert token_h * token_w == token_size
            basic_x = F.adaptive_avg_pool2d(x, output_size=1)
            basic_x = self.zero_output_layer_ada(basic_x.squeeze((2, 3)))
            x = F.interpolate(
                x, size=(token_h, token_w), mode="bilinear", align_corners=True
            )
            x = self.zero_output_layer(x)
            x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            basic_x = basic_x.unsqueeze(1)
            output = torch.cat([basic_x, x], dim=1)
            return output.chunk(6, dim=-1)


class CrossPre(nn.Module):
    def __init__(self, dim, channelwise: bool = True, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.channelwise = channelwise
        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, lowres_depth: Tensor, tokens: Tensor) -> Tensor:
        x = lowres_depth
        x = self.feat_extract(x)
        if self.channelwise:
            x = F.adaptive_max_pool2d(x, output_size=1)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        return x


class CopyBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        adaLN: bool = False,
        vit_cross: bool = False,
        controlnet: bool = False,
        channelwise: bool = True,  # True: conv -> adaptive pooling -> linear, False: resize  (HxW/pxp)-> conv -> flatten
        **kwargs,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        assert controlnet == False
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

        self.adaLN = adaLN
        self.vit_cross = vit_cross
        self.controlnet = controlnet
        self.channelwise = channelwise
        if self.adaLN:
            self.adaLN_modulation = AdaLN(dim, channelwise=self.channelwise)
        if self.vit_cross:
            self.vit_cross_pre = CrossPre(dim, channelwise=self.channelwise)
            self.vit_cross_attn = nn.MultiheadAttention(
                dim, num_heads=4, batch_first=True
            )
            self.norm3 = norm_layer(dim)
            init_values = 0.0
            self.ls3 = (
                LayerScale(dim, init_values=init_values)
                if init_values
                else nn.Identity()
            )

    def forward(self, x: Tensor, lowres_depth: Optional[Tensor] = None) -> Tensor:
        if self.adaLN:
            ada_params = self.adaLN_modulation(lowres_depth, x)
        else:
            ada_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            raise NotImplementedError
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            raise NotImplementedError
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + (1 + ada_params[0]) * self.ls1(
                self.attn((1 + ada_params[1]) * self.norm1(x) + ada_params[2])
            )
            if self.vit_cross:
                depth_feat = self.vit_cross_pre(lowres_depth, x)
                x = x + self.ls3(
                    self.vit_cross_attn(self.norm3(x), depth_feat, depth_feat)[0]
                )
            x = x + (1 + ada_params[3]) * self.ls2(
                self.mlp((1 + ada_params[4]) * self.norm2(x) + ada_params[5])
            )
        return x


class ZeroInBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feat_extract = nn.Sequential(
            nn.Conv2d(1, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.zero_output_layer = zero_module(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        )
        self.zero_output_layer_ada = zero_module(nn.Linear(dim, dim))

    def forward(self, lowres_depth: Tensor, tokens: Tensor) -> Tensor:
        x = lowres_depth
        x = self.feat_extract(x)
        h_lowres, w_lowres = lowres_depth.shape[-2:]
        token_size = tokens.shape[1] - 1
        img_ratio = w_lowres / h_lowres
        token_h = int((token_size / img_ratio) ** 0.5)
        token_w = int(token_h * img_ratio)
        assert token_h * token_w == token_size
        basic_x = F.adaptive_avg_pool2d(x, output_size=1)
        basic_x = self.zero_output_layer_ada(basic_x.squeeze((2, 3)))
        x = F.interpolate(
            x, size=(token_h, token_w), mode="bilinear", align_corners=True
        )
        x = self.zero_output_layer(x)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        basic_x = basic_x.unsqueeze(1)
        output = torch.cat([basic_x, x], dim=1)
        return output


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        adaLN: bool = False,
        vit_cross: bool = False,
        controlnet: bool = False,
        channelwise: bool = True,  # True: conv -> adaptive pooling -> linear, False: resize  (HxW/pxp)-> conv -> flatten
        **kwargs,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

        self.adaLN = adaLN
        self.vit_cross = vit_cross
        self.controlnet = controlnet
        self.channelwise = channelwise
        # AdaLN
        # img: B x (H/p x W/p + 1) x DIM
        # lowres_depth: 1 x 1 x H x W

        # channelwise: conv -> adaptive pooling -> linear -> B x 1 X 6xDIM
        # pixelwise: conv -> flatten and concat adaptive pooling -> 1x1 conv  ->  B DIMx6
        if self.adaLN:
            self.adaLN_modulation = AdaLN(dim, channelwise=self.channelwise)

        if self.vit_cross:
            self.vit_cross_pre = CrossPre(dim, channelwise=self.channelwise)
            self.vit_cross_attn = nn.MultiheadAttention(
                dim, num_heads=4, batch_first=True
            )
            self.norm3 = norm_layer(dim)
            init_values = 0.0
            self.ls3 = (
                LayerScale(dim, init_values=init_values)
                if init_values
                else nn.Identity()
            )

        if self.controlnet:
            self.copy_block = CopyBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=drop,
                attn_drop=attn_drop,
                init_values=init_values,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                attn_class=attn_class,
                ffn_layer=ffn_layer,
                adaLN=adaLN,
                vit_cross=vit_cross,
                controlnet=False,
                channelwise=channelwise,
                **kwargs,
            )
            self.zero_in_block = ZeroInBlock(dim)
            self.zero_out_block = zero_module(nn.Linear(dim, dim))

    def forward(self, x: Tensor, lowres_depth: Optional[Tensor] = None) -> Tensor:
        if self.controlnet:
            depth_branch_input = self.zero_in_block(lowres_depth, x)
            depth_x = self.copy_block(x + depth_branch_input, lowres_depth=lowres_depth)
            depth_branch_output = self.zero_out_block(depth_x)
        if self.adaLN:
            ada_params = self.adaLN_modulation(lowres_depth, x)
        else:
            ada_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            raise NotImplementedError
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            raise NotImplementedError
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + (1 + ada_params[0]) * self.ls1(
                self.attn((1 + ada_params[1]) * self.norm1(x) + ada_params[2])
            )
            if self.vit_cross:
                depth_feat = self.vit_cross_pre(lowres_depth, x)
                x = x + self.ls3(
                    self.vit_cross_attn(self.norm3(x), depth_feat, depth_feat)[0]
                )
            x = x + (1 + ada_params[3]) * self.ls2(
                self.mlp((1 + ada_params[4]) * self.norm2(x) + ada_params[5])
            )
        if self.controlnet:
            x = x + depth_branch_output
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(
            x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
        )
    else:
        x_plus_residual = scaled_index_add(
            x,
            brange,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = (
        [b.shape[0] for b in branges]
        if branges is not None
        else [x.shape[0] for x in x_list]
    )
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(
            1, -1, x_list[0].shape[-1]
        )
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [
        get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list
    ]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(
        x_list, branges, residual_list, residual_scale_factors
    ):
        outputs.append(
            add_residual(
                x, brange, residual, residual_scale_factor, scaling_vector
            ).view_as(x)
        )
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(
                    self.ls1.gamma if isinstance(self.ls1, LayerScale) else None
                ),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(
                    self.ls2.gamma if isinstance(self.ls1, LayerScale) else None
                ),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, lowres_depth: Optional[Tensor] = None):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, lowres_depth=lowres_depth)
        elif isinstance(x_or_x_list, list):
            assert (
                XFORMERS_AVAILABLE
            ), "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list, lowres_depth=lowres_depth)
        else:
            raise AssertionError
