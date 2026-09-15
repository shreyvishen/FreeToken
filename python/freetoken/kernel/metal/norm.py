"""RMSNorm and its neighbours as single Metal launches, one threadgroup per row."""

from __future__ import annotations

import functools

import torch

from .shaders import compile, is_available, msl_type

_MAX_TG = 256

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define TG {tg}
#define NSG (TG / 32)
#define HAS_RESIDUAL {has_residual}
#define HAS_GATE {has_gate}
#define HAS_BIAS {has_bias}
#define PLUS_ONE {plus_one}
#define HAS_WEIGHT {has_weight}
#define IS_L2 {is_l2}

kernel void rms_norm(
{params}    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
  uint row = gid.y;
  uint base = row * N;
  device const IN_T* x = inp + base;

  threadgroup float red[NSG];

  // --- sum of squares, fp32 whatever the storage dtype ---
  float acc = 0.0f;
  for (uint i = tid; i < N; i += TG) {{
    float v = float(x[i]);
#if HAS_RESIDUAL
    // Round the new residual to storage dtype before normalizing it: the torch
    // chain rounds at ``residual.add_`` and reads the rounded value back.
    IN_T r = IN_T(v + float(resid[base + i]));
    resid[base + i] = r;
    v = float(r);
#endif
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
#if IS_L2
  // l2norm divides by the norm of the row, not its root-mean-square.
  float scale = rsqrt(red[0] + eps) * post_scale;
#else
  float scale = rsqrt(red[0] / float(N) + eps) * post_scale;
#endif

  // --- scale, weight, and whatever epilogue this variant has ---
  for (uint i = tid; i < N; i += TG) {{
#if HAS_RESIDUAL
    float v = float(resid[base + i]);
#else
    float v = float(x[i]);
#endif
#if HAS_WEIGHT
    float w = float(weight[i]);
#if PLUS_ONE
    w += 1.0f;
#endif
#endif
    float y = v * scale;
#if HAS_WEIGHT
    y *= w;
#endif
#if HAS_BIAS
    y += float(bias[i]);
#endif
#if HAS_GATE
    float g = float(gate[base + i]);
    y *= g / (1.0f + exp(-g));
#endif
    out[base + i] = IN_T(y);
  }}
}}
"""


@functools.lru_cache(maxsize=None)
def _library(dtype, tg, *, has_residual, has_gate, has_bias, plus_one,
             has_weight=True, is_l2=False):
    decls = ["device IN_T* out", "device const IN_T* inp"]
    if has_weight:
        decls.append("device const IN_T* weight")
    if has_residual:
        decls.insert(1, "device IN_T* resid")
    if has_gate:
        decls.append("device const IN_T* gate")
    if has_bias:
        decls.append("device const IN_T* bias")
    decls.append("constant uint& N")
    decls.append("constant float& eps")
    decls.append("constant float& post_scale")
    params = "".join(f"    {d} [[buffer({i})]],\n" for i, d in enumerate(decls))
    return compile(
        f"#define IN_T {msl_type(dtype)}\n"
        + _SOURCE.format(
            tg=tg,
            params=params,
            has_residual=int(has_residual),
            has_gate=int(has_gate),
            has_bias=int(has_bias),
            plus_one=int(plus_one),
            has_weight=int(has_weight),
            is_l2=int(is_l2),
        )
    )


def _tg_for(n: int) -> int:
    return max(32, min(_MAX_TG, ((n + 31) // 32) * 32))


def supports(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """The kernel's contract; anything outside it falls back to the torch chain."""
    return (
        is_available()
        and x.device.type == "mps"
        and x.dim() == 2
        and x.is_contiguous()
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and weight.is_contiguous()
        and weight.numel() == x.shape[-1]
    )


def rmsnorm_metal(x: torch.Tensor, weight: torch.Tensor, eps: float, out=None, plus_one=False):
    """``x * rsqrt(mean(x^2) + eps) * w`` in one launch."""
    m, n = x.shape
    y = torch.empty_like(x) if out is None else out
    if m:
        tg = _tg_for(n)
        lib = _library(x.dtype, tg, has_residual=False, has_gate=False, has_bias=False,
                       plus_one=plus_one)
        lib.rms_norm(y, x, weight, n, float(eps), 1.0, threads=(tg, m, 1), group_size=(tg, 1, 1))
    return y


def fused_add_rmsnorm_metal(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, plus_one=False
) -> None:
    """``residual += x; x = rmsnorm(residual)``, both in place, one launch."""
    m, n = x.shape
    if not m:
        return
    tg = _tg_for(n)
    lib = _library(x.dtype, tg, has_residual=True, has_gate=False, has_bias=False,
                   plus_one=plus_one)
    lib.rms_norm(x, residual, x, weight, n, float(eps), 1.0,
                 threads=(tg, m, 1), group_size=(tg, 1, 1))


def rms_norm_gated_metal(
    x: torch.Tensor, weight: torch.Tensor, bias, z: torch.Tensor, eps: float, plus_one=False
) -> torch.Tensor:
    """``(rmsnorm(x) + bias) * silu(z)`` in one launch -- the GDN output gate."""
    m, n = x.shape
    y = torch.empty_like(x)
    if not m:
        return y
    tg = _tg_for(n)
    lib = _library(x.dtype, tg, has_residual=False, has_gate=True, has_bias=bias is not None,
                   plus_one=plus_one)
    args = [y, x, weight, z.contiguous()]
    if bias is not None:
        args.append(bias.contiguous())
    lib.rms_norm(*args, n, float(eps), 1.0, threads=(tg, m, 1), group_size=(tg, 1, 1))
    return y


def l2norm_metal(x: torch.Tensor, eps: float = 1e-6, post_scale: float = 1.0) -> torch.Tensor:
    """``x * rsqrt(sum(x^2) + eps) * post_scale`` over the last axis, one launch."""
    rows = x.reshape(-1, x.shape[-1])
    m, n = rows.shape
    y = torch.empty_like(rows)
    if m:
        tg = _tg_for(n)
        lib = _library(x.dtype, tg, has_residual=False, has_gate=False, has_bias=False,
                       plus_one=False, has_weight=False, is_l2=True)
        lib.rms_norm(y, rows, n, float(eps), float(post_scale),
                     threads=(tg, m, 1), group_size=(tg, 1, 1))
    return y.view(x.shape)


__all__ = [
    "fused_add_rmsnorm_metal", "l2norm_metal", "rms_norm_gated_metal", "rmsnorm_metal", "supports",
]
