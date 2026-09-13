import os
import warnings
from typing import Optional

import torch

try:
    import flash_attn_interface

    def _is_hopper_gpu() -> bool:
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name

    FLASH_ATTN_3_AVAILABLE = _is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    import transformer_engine  # noqa: F401

    TRANSFORMER_ENGINE_AVAILABLE = True
except ModuleNotFoundError:
    TRANSFORMER_ENGINE_AVAILABLE = False


def _gpu_supports_flash_attention() -> bool:
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 8


def _sdpa_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if q_lens is not None or k_lens is not None:
        warnings.warn("q_lens/k_lens are ignored in SDPA fallback.")
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale
    if softmax_scale is not None:
        q = q * softmax_scale
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous()


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[tuple[int, int]] = None,
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version: Optional[int] = None,
) -> torch.Tensor:
    if window_size is None:
        window_size = (-1, -1)
    if version is None:
        version = 3

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    if not _gpu_supports_flash_attention():
        return _sdpa_attention_fallback(
            q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, dtype
        )

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def _half(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = _half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = _half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = _half(k.flatten(0, 1))
        v = _half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = _half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = _half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    zeros = torch.zeros([1], dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.cat([zeros, q_lens]).cumsum(0).to(torch.int32)
    cu_seqlens_k = torch.cat([zeros, k_lens]).cumsum(0).to(torch.int32)

    # FA3 (Hopper) has no dropout API. Use it only when dropout is off; otherwise
    # fall back to FA2 so training dropout is not silently ignored.
    use_fa3 = version == 3 and FLASH_ATTN_3_AVAILABLE and dropout_p == 0.0
    if use_fa3:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        raise ValueError(
            "No compatible flash-attn backend "
            f"(version={version}, dropout_p={dropout_p}, "
            f"FA3={FLASH_ATTN_3_AVAILABLE}, FA2={FLASH_ATTN_2_AVAILABLE}). "
            "FA3 does not support dropout; install flash-attn 2 or set dropout_p=0."
        )

    return x.type(out_dtype)


class AttentionModule(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        q_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[tuple[int, int]] = None,
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        backend: Optional[str] = None,
    ):
        super().__init__()
        _ = num_heads, head_dim

        if backend is None:
            backend = "FA2"
        if os.getenv("ATTENTION_BACKEND") is not None:
            backend = os.getenv("ATTENTION_BACKEND")
        if os.getenv("ENABLE_TENSORRT", "False").lower() == "true":
            backend = "torch"
        assert backend in ["torch", "FA2", "FA3", "torch_onnx"]

        self.backend = backend
        self.dropout_p = float(dropout_p)
        self.softmax_scale = softmax_scale
        self.q_scale = q_scale
        self.causal = causal
        self.window_size = window_size
        self.deterministic = deterministic
        self.dtype = dtype
        self._flash_version = 3 if backend == "FA3" else 2

    def _effective_dropout_p(self) -> float:
        return self.dropout_p if self.training else 0.0

    def _torch_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out_dtype = q.dtype
        q = q.transpose(1, 2).to(self.dtype)
        k = k.transpose(1, 2).to(self.dtype)
        v = v.transpose(1, 2).to(self.dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=self.causal,
            dropout_p=self._effective_dropout_p(),
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2).contiguous().to(out_dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.backend in ("torch", "torch_onnx"):
            if q_lens is not None or k_lens is not None:
                warnings.warn("q_lens/k_lens are ignored for torch backend.")
            return self._torch_attn(q, k, v)
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=self._effective_dropout_p(),
            softmax_scale=self.softmax_scale,
            q_scale=self.q_scale,
            causal=self.causal,
            window_size=self.window_size,
            deterministic=self.deterministic,
            dtype=self.dtype,
            version=self._flash_version,
        )
