"""The decode step of the depthwise causal conv as one Metal launch: one thread per (request,
channel), reading the channel's history out of the state slot into registers, folding in
the new token, writing the shifted history back and returning ``silu(conv)``."""

from __future__ import annotations

import functools

import torch

from .shaders import compile, is_available, msl_type

_MAX_WIDTH = 16

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define K {width}

kernel void conv1d_decode(
    device T* out              [[buffer(0)]],   // [B, C] silu(conv)
    device ST_T* state         [[buffer(1)]],   // [num_slots, C, K-1], in place
    device const T* x          [[buffer(2)]],   // [B, C] one token per request
    device const T* w          [[buffer(3)]],   // [C, K]
    device const int* slots    [[buffer(4)]],   // [B]
    constant uint& C           [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{{
  uint c = gid.x;
  uint b = gid.y;
  if (c >= C) {{ return; }}

  device ST_T* st = state + (uint(slots[b]) * C + c) * (K - 1);
  device const T* wc = w + c * K;

  // Read the whole history first: the shift below overwrites it.
  float hist[K - 1];
  for (uint j = 0; j < K - 1; ++j) {{ hist[j] = float(st[j]); }}

  float acc = 0.0f;
  for (uint j = 0; j < K - 1; ++j) {{ acc += hist[j] * float(wc[j]); }}
  float xv = float(x[b * C + c]);
  acc += xv * float(wc[K - 1]);

  for (uint j = 0; j + 2 < K; ++j) {{ st[j] = ST_T(hist[j + 1]); }}
  st[K - 2] = ST_T(xv);

  out[b * C + c] = T(acc / (1.0f + exp(-acc)));
}}
"""


@functools.lru_cache(maxsize=None)
def _library(dtype: torch.dtype, state_dtype: torch.dtype, width: int):
    return compile(
        f"#define T {msl_type(dtype)}\n#define ST_T {msl_type(state_dtype)}\n"
        + _SOURCE.format(width=width)
    )


def supports(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor) -> bool:
    """The kernel's contract; anything outside it falls back to the torch chain."""
    if not (is_available() and x.device.type == "mps"):
        return False
    width = weight.shape[-1]
    return (
        x.dim() == 2
        and weight.dim() == 2
        and conv_state.dim() == 3
        and 2 <= width <= _MAX_WIDTH
        # The torch form concatenates one token onto the history and multiplies by
        # the whole filter, so the stored history is exactly ``width - 1`` long.
        and conv_state.shape[2] == width - 1
        and conv_state.shape[1] == x.shape[1] == weight.shape[0]
        and x.is_contiguous()
        and conv_state.is_contiguous()
        and weight.is_contiguous()
        and x.dtype == weight.dtype
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and conv_state.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def causal_conv1d_decode_metal(
    x: torch.Tensor,                   # [B, C]
    conv_state: torch.Tensor,          # [num_slots, C, K-1], in place
    weight: torch.Tensor,              # [C, K]
    conv_state_indices: torch.Tensor,  # [B]
) -> torch.Tensor:
    """Advance the conv state by one token and return ``silu(conv)``, one launch."""
    b, c = x.shape
    out = torch.empty_like(x)
    if b == 0:
        return out
    tgx = min(c, 256)
    _library(x.dtype, conv_state.dtype, weight.shape[-1]).conv1d_decode(
        out, conv_state, x, weight, conv_state_indices.to(torch.int32).contiguous(), c,
        threads=(c, b, 1), group_size=(tgx, 1, 1),
    )
    return out


__all__ = ["causal_conv1d_decode_metal", "supports"]
