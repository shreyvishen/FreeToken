"""The hyper-connection mix and combine bodies as single Metal launches, where the torch chain
is ~15 per mix, twice per layer. Same op boundaries as ``kernel/triton/hc.py``; fp32 statistics
and intermediates, rounded to the storage dtype only at the store."""

from __future__ import annotations

import functools

import torch

from .norm import tg_for
from .shaders import compile, is_available, msl_type

# Only the grouped norm reads TG; the other three are one thread per element, so they pin the
# library key at the width tg_for returns for every shipped hidden size and share its compile.
_TG = 256

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define TG {tg}
#define NSG (TG / 32)

// Per-stream (1+w) RMSNorm of a [M, HC*GD] residual: one threadgroup per (row, stream).
// gid.y is the row and gid.z the stream, so the slice starts at (row*HC + stream)*GD.
kernel void grouped_plus_one_rms_norm(
    device T* out          [[buffer(0)]],   // [M, HC * GD]
    device const T* inp    [[buffer(1)]],   // [M, HC * GD]
    device const T* weight [[buffer(2)]],   // [HC * GD]
    constant uint& GD      [[buffer(3)]],
    constant uint& HC      [[buffer(4)]],
    constant float& eps    [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
  uint stream = gid.z;
  uint base = (gid.y * HC + stream) * GD;
  uint wbase = stream * GD;
  device const T* x = inp + base;

  threadgroup float red[NSG];
  float acc = 0.0f;
  for (uint i = tid; i < GD; i += TG) {{
    float v = float(x[i]);
    acc += v * v;
  }}
  acc = simd_sum(acc);
  if (lane == 0) {{ red[sg] = acc; }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {{
    float t = red[0];
    for (uint i = 1; i < NSG; ++i) {{ t += red[i]; }}
    red[0] = t;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = rsqrt(red[0] / float(GD) + eps);

  for (uint i = tid; i < GD; i += TG) {{
    out[base + i] = T(float(x[i]) * scale * (1.0f + float(weight[wbase + i])));
  }}
}}

// silu(x / HC) over the low-rank GEMM output, the mix's activation. ``inp`` is a column
// slice of the merged down GEMM, so it carries its own row pitch.
kernel void hc_silu(
    device T* out       [[buffer(0)]],   // [M, D]
    device const T* inp [[buffer(1)]],   // [M, D] with stride in_stride
    constant uint& D    [[buffer(2)]],
    constant uint& in_stride [[buffer(3)]],
    constant float& inv_hc [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{{
  uint d = gid.x;
  if (d >= D) {{ return; }}
  uint row = gid.y;
  float v = float(inp[row * in_stride + d]) * inv_hc;
  out[row * D + d] = T(v / (1.0f + exp(-v)));
}}

// mean over streams of sigmoid(gate) * rn: [M, HC*GD] x [M, HC*GD] -> [M, GD].
kernel void hc_gate_mix(
    device T* out         [[buffer(0)]],   // [M, GD]
    device const T* rn    [[buffer(1)]],   // [M, HC * GD]
    device const T* gate  [[buffer(2)]],   // [M, HC * GD]
    constant uint& GD     [[buffer(3)]],
    constant uint& HC     [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{{
  uint d = gid.x;
  if (d >= GD) {{ return; }}
  uint row = gid.y;
  uint base = row * HC * GD + d;

  float acc = 0.0f;
  for (uint s = 0; s < HC; ++s) {{
    uint i = base + s * GD;
    float g = float(gate[i]);
    acc += float(rn[i]) / (1.0f + exp(-g));
  }}
  out[row * GD + d] = T(acc / float(HC));
}}

// R'_s = R_s + y * 2*sigmoid(logit_s / HC), every stream of one row in one thread. ``logits``
// is a column slice of the merged down GEMM, so it carries its own row pitch.
kernel void hc_combine(
    device T* out          [[buffer(0)]],   // [M, HC * GD]
    device const T* resid  [[buffer(1)]],   // [M, HC * GD]
    device const T* block  [[buffer(2)]],   // [M, GD]
    device const T* logits [[buffer(3)]],   // [M, HC] with stride logit_stride
    constant uint& GD      [[buffer(4)]],
    constant uint& HC      [[buffer(5)]],
    constant uint& logit_stride [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{{
  uint d = gid.x;
  if (d >= GD) {{ return; }}
  uint row = gid.y;
  uint base = row * HC * GD + d;
  float y = float(block[row * GD + d]);
  device const T* lg = logits + row * logit_stride;

  for (uint s = 0; s < HC; ++s) {{
    float v = float(lg[s]) / float(HC);
    float inject = 2.0f / (1.0f + exp(-v));
    uint i = base + s * GD;
    out[i] = T(float(resid[i]) + y * inject);
  }}
}}
"""


@functools.lru_cache(maxsize=None)
def _library(dtype: torch.dtype, tg: int):
    return compile(f"#define T {msl_type(dtype)}\n" + _SOURCE.format(tg=tg))


def _ok(*tensors: torch.Tensor) -> bool:
    first = tensors[0]
    return (
        is_available()
        and first.device.type == "mps"
        and first.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and all(t.dim() == 2 and t.dtype == first.dtype for t in tensors)
        and all(t.is_contiguous() for t in tensors)
    )


def supports_grouped_norm(x: torch.Tensor, weight: torch.Tensor, num_groups: int) -> bool:
    return (
        _ok(x)
        and weight.dim() == 1
        and weight.dtype == x.dtype
        and weight.is_contiguous()
        # the torch reference broadcasts a full-width weight only, so neither does this
        and weight.numel() == x.shape[1]
        and num_groups > 0
        and x.shape[1] % num_groups == 0
    )


def grouped_plus_one_rms_norm_metal(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    """RMSNorm each of ``num_groups`` slices of the row on its own fp32 statistic, scale by (1+w)."""
    m, dim = x.shape
    gd = dim // num_groups
    y = torch.empty_like(x)
    if m:
        tg = tg_for(gd)
        _library(x.dtype, tg).grouped_plus_one_rms_norm(
            y, x, weight, gd, num_groups, float(eps),
            threads=(tg, m, num_groups), group_size=(tg, 1, 1),
        )
    return y


def hc_silu_metal(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``silu(x / hc_count)`` in one launch, over a row-strided slice."""
    m, d = x.shape
    y = torch.empty((m, d), dtype=x.dtype, device=x.device)
    if m:
        _library(x.dtype, _TG).hc_silu(
            y, x, d, int(x.stride(0)), 1.0 / float(hc_count),
            threads=(d, m, 1), group_size=(min(d, _TG), 1, 1),
        )
    return y


def hc_gate_mix_metal(rn: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``mean_s(sigmoid(gate_s) * rn_s)`` over the ``hc_count`` streams of each row."""
    m, dim = rn.shape
    gd = dim // hc_count
    y = torch.empty((m, gd), dtype=rn.dtype, device=rn.device)
    if m:
        _library(rn.dtype, _TG).hc_gate_mix(
            y, rn, gate, gd, hc_count,
            threads=(gd, m, 1), group_size=(min(gd, _TG), 1, 1),
        )
    return y


def supports_combine(
    residual: torch.Tensor, block: torch.Tensor, logits: torch.Tensor, hc_count: int
) -> bool:
    return (
        _ok(residual, block)
        and logits.dim() == 2
        and logits.dtype == residual.dtype
        and logits.stride(1) == 1  # a column slice of the merged GEMM keeps unit inner stride
        and residual.shape[1] % hc_count == 0
        and block.shape == (residual.shape[0], residual.shape[1] // hc_count)
        and logits.shape == (residual.shape[0], hc_count)
    )


def hc_combine_metal(
    residual: torch.Tensor, block: torch.Tensor, logits: torch.Tensor, hc_count: int
) -> torch.Tensor:
    """``R_s + block * 2*sigmoid(logits_s / hc_count)`` for every stream, one launch."""
    m, dim = residual.shape
    gd = dim // hc_count
    y = torch.empty_like(residual)
    if m:
        _library(residual.dtype, _TG).hc_combine(
            y, residual, block, logits, gd, hc_count, int(logits.stride(0)),
            threads=(gd, m, 1), group_size=(min(gd, _TG), 1, 1),
        )
    return y


__all__ = [
    "grouped_plus_one_rms_norm_metal", "hc_combine_metal", "hc_gate_mix_metal", "hc_silu_metal",
    "supports_combine", "supports_grouped_norm",
]
