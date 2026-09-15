"""Rotary position embedding as one Metal launch: one thread per rotary pair, with q and k in
the same dispatch because they share the cos/sin row of their token."""

from __future__ import annotations

import functools

import torch

from .shaders import MSL_INT, compile, is_available, msl_type

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define IS_NEOX {is_neox}

kernel void rope_inplace(
    device T* q               [[buffer(0)]],   // [nnz, HQ * D], in place
    device T* k               [[buffer(1)]],   // [nnz, HK * D], in place
    device const float* cs    [[buffer(2)]],   // [max_pos, 2 * HALF] cos | sin
    device const POS_T* pos   [[buffer(3)]],   // [nnz]
    constant uint& HQ         [[buffer(4)]],
    constant uint& HK         [[buffer(5)]],
    constant uint& D          [[buffer(6)]],
    constant uint& HALF       [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{{
  uint i = gid.x;              // which rotary pair
  uint h = gid.y;              // head, q heads first then k heads
  uint t = gid.z;              // token
  if (i >= HALF || h >= HQ + HK) {{ return; }}

  device T* row = (h < HQ) ? (q + (t * HQ + h) * D) : (k + (t * HK + (h - HQ)) * D);

  uint p = uint(pos[t]) * (2 * HALF);
  float c = cs[p + i];
  float s = cs[p + HALF + i];

#if IS_NEOX
  uint i0 = i, i1 = i + HALF;
#else
  uint i0 = 2 * i, i1 = 2 * i + 1;
#endif
  float x0 = float(row[i0]);
  float x1 = float(row[i1]);
  row[i0] = T(x0 * c - x1 * s);
  row[i1] = T(x1 * c + x0 * s);
}}
"""


@functools.lru_cache(maxsize=None)
def _library(dtype: torch.dtype, pos_dtype: torch.dtype, is_neox: bool):
    return compile(
        f"#define T {msl_type(dtype)}\n#define POS_T {MSL_INT[pos_dtype]}\n"
        + _SOURCE.format(is_neox=int(is_neox))
    )


def supports(query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor) -> bool:
    """The kernel's contract; anything outside it falls back to the torch chain."""
    return (
        is_available()
        and query.device.type == "mps"
        and query.dtype == key.dtype
        and query.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and query.is_contiguous()
        and key.is_contiguous()
        and query.dim() == key.dim() == 2
        and positions.dtype in MSL_INT
        and positions.is_contiguous()
    )


def apply_rope_metal(
    positions: torch.Tensor,
    query: torch.Tensor,       # [nnz, HQ * head_size]
    key: torch.Tensor,         # [nnz, HK * head_size]
    head_size: int,
    cos_sin_cache: torch.Tensor,   # [max_pos, rotary_dim] fp32, cos | sin
    is_neox: bool = True,
) -> None:
    """Rotate the leading ``rotary_dim`` dims of every head of ``query`` and ``key`` in
    place."""
    nnz = query.shape[0]
    if nnz == 0:
        return
    half = cos_sin_cache.shape[1] // 2
    hq = query.shape[1] // head_size
    hk = key.shape[1] // head_size

    # 256 threads per group, laid out along the pair axis first so neighbouring
    # threads touch neighbouring dims of the same head.
    tgx = min(half, 256)
    tgy = max(1, min(256 // tgx, hq + hk))
    _library(query.dtype, positions.dtype, is_neox).rope_inplace(
        query, key, cos_sin_cache.contiguous(), positions, hq, hk, head_size, half,
        threads=(half, hq + hk, nnz), group_size=(tgx, tgy, 1),
    )


__all__ = ["apply_rope_metal", "supports"]
