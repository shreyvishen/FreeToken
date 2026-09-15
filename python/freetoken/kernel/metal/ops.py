"""The elementwise and gather half of the Metal path: torch ops on MPS with the same
signatures as the CUDA and Triton kernels they stand in for."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, head_size: int,
    cos_sin_cache: torch.Tensor, is_neox: bool = True,
) -> None:
    """Rotate the first ``rotary_dim`` dims of every head of ``query`` ``[nnz, HQ * head]``
    and ``key`` ``[nnz, HK * head]`` in place off a float32 ``[max_pos, rotary_dim]`` cos
    | sin cache, passing the rest through."""
    from .rope import apply_rope_metal, supports

    if supports(query, key, positions):
        apply_rope_metal(positions, query, key, head_size, cos_sin_cache, is_neox)
        return
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")
    nnz = query.shape[0]
    if nnz == 0:
        return
    rotary_dim = cos_sin_cache.shape[1]
    half = rotary_dim // 2
    cs = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:].unsqueeze(1)

    for t in (query, key):
        heads = t.shape[1] // head_size
        view = t.view(nnz, heads, head_size)
        if is_neox:  # halves: (0..half-1) pairs with (half..rotary_dim-1)
            x0, x1 = view[:, :, :half], view[:, :, half:rotary_dim]
        else:  # GPT-J interleaved: adjacent pairs
            pairs = view[:, :, :rotary_dim].view(nnz, heads, half, 2)
            x0, x1 = pairs[..., 0], pairs[..., 1]
        f0, f1 = x0.float(), x1.float()
        o0 = f0 * cos - f1 * sin
        o1 = f1 * cos + f0 * sin
        x0.copy_(o0.to(t.dtype))
        x1.copy_(o1.to(t.dtype))


def rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, out=None, enable_pdl=False
) -> torch.Tensor:
    """``x * rsqrt(mean(x^2) + eps) * w`` in fp32, cast back."""
    from .norm import rmsnorm_metal, supports

    if supports(input, weight):
        return rmsnorm_metal(input, weight, eps, out)
    return _rmsnorm_torch(input, weight, eps, out)


def _rmsnorm_torch(input: torch.Tensor, weight: torch.Tensor, eps: float, out=None):
    x = input.float()
    y = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(input.dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y


def fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    eps: float = 1e-6, enable_pdl=False,
) -> None:
    """``residual += input; input = rmsnorm(residual)``, both in place: the residual
    carries the pre-norm sum on to the next layer, so this cannot be a pure function."""
    from .norm import fused_add_rmsnorm_metal, supports

    if supports(input, weight) and residual.is_contiguous() and residual.dtype == input.dtype:
        fused_add_rmsnorm_metal(input, residual, weight, eps)
        return
    residual.add_(input)
    input.copy_(_rmsnorm_torch(residual, weight, eps))


def rms_norm_gated(
    x: torch.Tensor, weight: torch.Tensor, bias, z: torch.Tensor, eps: float,
    is_rms_norm: bool = True, norm_before_gate: bool = True, activation: str = "silu",
) -> torch.Tensor:
    """``norm(x) * silu(z)``, the GDN output gate."""
    if not is_rms_norm or not norm_before_gate or activation != "silu":
        raise NotImplementedError("metal rms_norm_gated: rms + norm_before_gate + silu only")
    from .norm import rms_norm_gated_metal, supports

    if (supports(x, weight) and z.is_contiguous() and z.dtype == x.dtype
            and (bias is None or (bias.dtype == x.dtype and bias.is_contiguous()))):
        return rms_norm_gated_metal(x, weight, bias, z, eps)
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    if bias is not None:
        y = y + bias.float()
    return (y * F.silu(z.float())).to(x.dtype)


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """SwiGLU over UNINTERLEAVED halves: ``silu(x[..., :d]) * x[..., d:]``."""
    from .elementwise import silu_and_mul_metal, supports_silu_and_mul

    if supports_silu_and_mul(x):
        return silu_and_mul_metal(x, out)
    gate, up = x.chunk(2, dim=-1)
    y = F.silu(gate.float()).to(x.dtype) * up
    if out is not None:
        out.copy_(y)
        return out
    return y


def indexing(
    weights: torch.Tensor, indices: torch.Tensor, *, output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """Embedding gather."""
    idx = indices.to(torch.long)
    if vocab_range is None:
        y = weights.index_select(0, idx)
    else:
        start, length = vocab_range
        local = idx - start
        keep = (local >= 0) & (local < length)
        y = weights.index_select(0, local.clamp_(0, max(length - 1, 0)))
        y = y * keep.unsqueeze(-1).to(y.dtype)
    if output is not None:
        output.copy_(y)
        return output
    return y


def store_cache(
    k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Write ``k``/``v`` into the ``[num_tokens, kv_heads, head_dim]`` caches at the slots
    ``indices`` names, with ``index_put_``: MPS costs ``index_copy_`` O(destination)."""
    idx = indices.to(torch.long)
    k_cache.index_put_((idx,), k.view(-1, *k_cache.shape[1:]).to(k_cache.dtype))
    v_cache.index_put_((idx,), v.view(-1, *v_cache.shape[1:]).to(v_cache.dtype))


def fused_topk_softmax(
    gating_output: torch.Tensor, topk: int, renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax over the ``[M, num_experts]`` logits, top-k, then renormalize."""
    from .router import fused_topk_softmax_metal, supports

    if gating_output.device.type == "mps" and supports(gating_output, topk):
        return fused_topk_softmax_metal(gating_output, topk, renormalize, num_token_non_padded)
    return _topk_softmax_torch(gating_output, topk, renormalize, num_token_non_padded)


def _topk_softmax_torch(
    gating_output: torch.Tensor, topk: int, renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(gating_output.float(), dim=-1)
    weights, indices = torch.topk(probs, topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    indices = indices.to(torch.int32)
    if num_token_non_padded is not None:
        rows = torch.arange(indices.shape[0], device=indices.device)
        indices[rows >= num_token_non_padded, :] = -1
    return weights.contiguous(), indices.contiguous()


def causal_conv1d_decode(
    x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Single-token depthwise causal conv with silu: shifts ``x`` ``[batch, conv_dim]`` into
    the history of its ``[num_slots, conv_dim, kernel-1]`` state slot, in place, and
    returns ``silu(conv)``."""
    from .conv import causal_conv1d_decode_metal, supports

    if supports(x, conv_state, weight):
        return causal_conv1d_decode_metal(x, conv_state, weight, conv_state_indices)
    idx = conv_state_indices.to(torch.long)
    hist = conv_state.index_select(0, idx).to(x.dtype)
    window = torch.cat([hist, x.unsqueeze(-1)], dim=-1)
    out = (window * weight.unsqueeze(0).to(x.dtype)).sum(-1)
    # index_put_, not index_copy_: MPS costs the latter O(destination), not O(indices).
    conv_state.index_put_((idx,), window[..., 1:].to(conv_state.dtype))
    return F.silu(out)


def causal_conv1d_varlen(
    x: torch.Tensor, weight: torch.Tensor, conv_states: torch.Tensor, cu_seqlens: torch.Tensor,
    cache_indices: torch.Tensor, has_initial_state: torch.Tensor,
    host: tuple[list, list, list] | None = None,
) -> torch.Tensor:
    """Varlen (prefill) depthwise causal conv with silu over ``x`` ``[conv_dim, total]``:
    writes ``silu(conv)`` into ``x`` in place and refreshes each request's conv state with
    its tail."""
    conv_dim, kernel = weight.shape
    # Reading the index tensors off the device instead flushes the pipeline three times
    # in every GDN layer.
    if host is None:
        host = (cu_seqlens.tolist(), cache_indices.to(torch.long).tolist(),
                has_initial_state.tolist())
    bounds, slots, init = host
    w = weight.to(x.dtype).unsqueeze(1)   # depthwise
    for i, slot in enumerate(slots):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi == lo:
            continue
        seg = x[:, lo:hi]
        left = (conv_states[slot].to(x.dtype) if init[i] else x.new_zeros(conv_dim, kernel - 1))
        padded = torch.cat([left, seg], dim=-1).unsqueeze(0)
        out = F.conv1d(padded, w, groups=conv_dim)[0]
        conv_states[slot].copy_(padded[0, :, -(kernel - 1):].to(conv_states.dtype))
        x[:, lo:hi] = F.silu(out).to(x.dtype)
    return x


__all__ = [
    "apply_rope_with_cos_sin_cache_inplace", "silu_and_mul", "causal_conv1d_decode",
    "causal_conv1d_varlen", "_topk_softmax_torch", "fused_add_rmsnorm", "fused_topk_softmax",
    "indexing", "rms_norm_gated", "rmsnorm", "store_cache",
]
