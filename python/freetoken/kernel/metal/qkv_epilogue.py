"""Everything between an attention layer's qkv projection and its attention kernel, as one
Metal launch: split the merged ``[q | gate]`` heads, RMSNorm q and k per head, rotate q
and k, copy the gate out contiguous, and write this token's k and v rows straight into the
paged KV cache."""

from __future__ import annotations

import functools

import torch

from .shaders import MAX_TG_FLOATS, MSL_INT, compile, is_available, msl_type

# SIMD-groups per threadgroup (head slots served concurrently per token).
_NSG = 8

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define NSG {nsg}
#define IS_NEOX {is_neox}

// Head slots, in order: HQ query heads (each 2*D wide in qkv: [q | gate]), HK key heads,
// HK value heads. One SIMD-group per slot, NSG slots per threadgroup.
kernel void qkv_epilogue(
    device const T*     qkv     [[buffer(0)]],   // [N, (2*HQ + 2*HK) * D]
    device       T*     q_out   [[buffer(1)]],   // [N, HQ, D] normed + rotated
    device       T*     g_out   [[buffer(2)]],   // [N, HQ * D] the gate half, contiguous
    device       T*     k_cache [[buffer(3)]],   // [slots, HK, D]
    device       T*     v_cache [[buffer(4)]],   // [slots, HK, D]
    device const T*     q_w     [[buffer(5)]],   // [D]
    device const T*     k_w     [[buffer(6)]],   // [D]
    device const float* cs      [[buffer(7)]],   // [max_pos, 2 * HALF] cos | sin
    device const POS_T* pos     [[buffer(8)]],   // [N]
    device const LOC_T* loc     [[buffer(9)]],   // [N] cache slot per token
    constant uint&  HQ          [[buffer(10)]],
    constant uint&  HK          [[buffer(11)]],
    constant uint&  HALF        [[buffer(12)]],
    constant float& eps         [[buffer(13)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  const uint t = tgid.y;
  const uint row = (2u * HQ + 2u * HK) * D;
  device const T* src_t = qkv + ulong(t) * row;
  const ulong slot = ulong(loc[t]);
  const uint p = uint(pos[t]) * (2u * HALF);

  threadgroup float tile[NSG * D];
  threadgroup float* xs = tile + sg * D;

  const uint h = tgid.x * NSG + sg;
  if (h < HQ + 2u * HK) {{
    const bool is_q = h < HQ;
    const bool is_k = !is_q && h < HQ + HK;
    device const T* src;
    device T* dst;
    device const T* w = q_w;
    if (is_q) {{
      src = src_t + h * 2u * D;
      dst = q_out + (ulong(t) * HQ + h) * D;
    }} else if (is_k) {{
      src = src_t + HQ * 2u * D + (h - HQ) * D;
      dst = k_cache + (slot * HK + (h - HQ)) * D;
      w = k_w;
    }} else {{
      src = src_t + HQ * 2u * D + HK * D + (h - HQ - HK) * D;
      dst = v_cache + (slot * HK + (h - HQ - HK)) * D;
    }}

    if (!is_q && !is_k) {{
      // v: a copy into its cache row.
      for (uint i = lane; i < D; i += 32u) {{ dst[i] = src[i]; }}
      return;
    }}
    if (is_q) {{
      // The gate is the second half of the [q | gate] head: copied out contiguous.
      device T* g = g_out + (ulong(t) * HQ + h) * D;
      for (uint i = lane; i < D; i += 32u) {{ g[i] = src[D + i]; }}
    }}

    // --- RMSNorm over the head, float32 ---
    float ss = 0.0f;
    for (uint i = lane; i < D; i += 32u) {{
      const float v = float(src[i]);
      ss += v * v;
    }}
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(D) + eps);
    for (uint i = lane; i < D; i += 32u) {{
      // Rounded to storage dtype here: the chain's rope read rmsnorm's stored output.
      xs[i] = float(T(float(src[i]) * scale * float(w[i])));
    }}
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // --- rope over the leading 2*HALF dims, the rest copied ---
    for (uint e = lane; e < D; e += 32u) {{
      float y;
#if IS_NEOX
      if (e < HALF) {{
        y = xs[e] * cs[p + e] - xs[e + HALF] * cs[p + HALF + e];
      }} else if (e < 2u * HALF) {{
        const uint i = e - HALF;
        y = xs[e] * cs[p + i] + xs[i] * cs[p + HALF + i];
      }} else {{
        y = xs[e];
      }}
#else
      if (e < 2u * HALF) {{
        const uint i = e >> 1;
        const float c = cs[p + i], s = cs[p + HALF + i];
        y = (e & 1u) ? (xs[e] * c + xs[e - 1u] * s) : (xs[e] * c - xs[e + 1u] * s);
      }} else {{
        y = xs[e];
      }}
#endif
      dst[e] = T(y);
    }}
  }}
}}
"""


@functools.lru_cache(maxsize=None)
def _library(dtype: torch.dtype, pos_dtype: torch.dtype, loc_dtype: torch.dtype,
             head_dim: int, is_neox: bool):
    return compile(
        f"#define T {msl_type(dtype)}\n#define POS_T {MSL_INT[pos_dtype]}\n"
        f"#define LOC_T {MSL_INT[loc_dtype]}\n"
        + _SOURCE.format(d=head_dim, nsg=_NSG, is_neox=int(is_neox))
    )


def supports(qkv, num_q, num_kv, head_dim, q_weight, k_weight, cos_sin_cache, positions,
             k_cache, v_cache, out_loc) -> bool:
    """The kernel's contract; anything outside it takes the torch chain."""
    return (
        is_available()
        and qkv.device.type == "mps"
        and qkv.dim() == 2
        and qkv.is_contiguous()
        and qkv.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and qkv.shape[1] == (2 * num_q + 2 * num_kv) * head_dim
        and q_weight.dtype == k_weight.dtype == qkv.dtype
        and q_weight.is_contiguous() and k_weight.is_contiguous()
        and q_weight.numel() == k_weight.numel() == head_dim
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.shape[1] <= head_dim
        and positions.dtype in MSL_INT and positions.is_contiguous()
        and out_loc.dtype in MSL_INT and out_loc.is_contiguous()
        and k_cache.dtype == v_cache.dtype == qkv.dtype
        and k_cache.dim() == v_cache.dim() == 3
        and k_cache.is_contiguous() and v_cache.is_contiguous()
        and tuple(k_cache.shape[1:]) == tuple(v_cache.shape[1:]) == (num_kv, head_dim)
        and _NSG * head_dim <= MAX_TG_FLOATS
    )


def qkv_epilogue_metal(
    qkv: torch.Tensor,            # [N, (2*num_q + 2*num_kv) * head_dim]
    num_q: int,
    num_kv: int,
    head_dim: int,
    q_weight: torch.Tensor,       # [head_dim]
    k_weight: torch.Tensor,       # [head_dim]
    eps: float,
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim] fp32, cos | sin
    positions: torch.Tensor,      # [N]
    is_neox: bool,
    k_cache: torch.Tensor,        # [slots, num_kv, head_dim], written in place
    v_cache: torch.Tensor,
    out_loc: torch.Tensor,        # [N] slot per token
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(q [N, num_q, head_dim], gate [N, num_q * head_dim])``; k and v land in
    the cache rows ``out_loc`` names. One launch."""
    n = qkv.shape[0]
    q = torch.empty((n, num_q, head_dim), dtype=qkv.dtype, device=qkv.device)
    gate = torch.empty((n, num_q * head_dim), dtype=qkv.dtype, device=qkv.device)
    if n == 0:
        return q, gate
    half = cos_sin_cache.shape[1] // 2
    _library(qkv.dtype, positions.dtype, out_loc.dtype, head_dim, is_neox).qkv_epilogue(
        qkv, q, gate, k_cache, v_cache, q_weight, k_weight, cos_sin_cache.contiguous(),
        positions, out_loc, num_q, num_kv, half, float(eps),
        threads=(32 * _NSG * -(-(num_q + 2 * num_kv) // _NSG), n), group_size=(32 * _NSG, 1),
    )
    return q, gate


__all__ = ["qkv_epilogue_metal", "supports"]
