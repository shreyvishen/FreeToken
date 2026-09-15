"""Dense bf16 GEMV and a SwiGLU-fused gate/up GEMV, as single Metal launches."""

from __future__ import annotations

import functools

import torch

from .fp8 import MAX_GEMV_ROWS
from .shaders import MAX_TG_FLOATS, compile, msl_type, pick_tile

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define KDIM {k}
#define NDIM {n}
#define NSG {nsg}
#define NR0 {nr0}
#define TG (32 * NSG)
#define SWIGLU {swiglu}
#define VEC {vec}
#define PACK8 {pack8}
#define U {u}
// Eight T values out of one uint4. bf16 is the high half of a float, so a shift and a mask are
// the conversion; fp16 goes through half4.
#define UNPACK8_BF16(v, f) \
  f[0] = as_type<float>((v).x << 16); f[1] = as_type<float>((v).x & 0xffff0000u); \
  f[2] = as_type<float>((v).y << 16); f[3] = as_type<float>((v).y & 0xffff0000u); \
  f[4] = as_type<float>((v).z << 16); f[5] = as_type<float>((v).z & 0xffff0000u); \
  f[6] = as_type<float>((v).w << 16); f[7] = as_type<float>((v).w & 0xffff0000u)
#define UNPACK8_F16(v, f) \
  {{ const half4 lo = as_type<half4>((v).xy); const half4 hi = as_type<half4>((v).zw); \
    f[0] = float(lo.x); f[1] = float(lo.y); f[2] = float(lo.z); f[3] = float(lo.w); \
    f[4] = float(hi.x); f[5] = float(hi.y); f[6] = float(hi.z); f[7] = float(hi.w); }}

// out[m, n] = sum_k x[m, k] * w[n, k]
// SWIGLU: w is the column-merged [gate | up], so the up row of pair i is NDIM rows on.
kernel void dense_gemv(
    device       T*     out  [[buffer(0)]],
    device const T*     x    [[buffer(1)]],
    device const T*     w    [[buffer(2)]],
#if NORM
    device const T*     rin   [[buffer(3)]],   // [M, KDIM] the residual stream
    device       T*     rout  [[buffer(4)]],   // [M, KDIM] the new residual, x + rin
    device       T*     xnorm [[buffer(5)]],   // [M, KDIM] rmsnorm(x + rin) * nw, next launch
    device const T*     nw    [[buffer(6)]],   // [KDIM]
    constant     float& eps   [[buffer(7)]],
#endif
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  threadgroup float xs[KDIM];
  device const T* xrow = x + ulong(gid.y) * KDIM;
#if NORM
  // Prologue: residual add + RMSNorm into the staged row, so the norm ahead of this
  // projection is not a launch. Threadgroup 0 alone writes the residual and normed row.
  {{
    threadgroup float red[NSG];
    device const T* rrow = rin + ulong(gid.y) * KDIM;
    float ss = 0.0f;
    for (uint i = tid; i < KDIM; i += TG) {{
      const T r = T(float(xrow[i]) + float(rrow[i]));
      if (gid.x == 0) {{ rout[ulong(gid.y) * KDIM + i] = r; }}
      const float v = float(r);
      xs[i] = v;
      ss += v * v;
    }}
    ss = simd_sum(ss);
    if (lane == 0) {{ red[sg] = ss; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < NSG; ++i) {{ tot += red[i]; }}
    const float nscale = rsqrt(tot / float(KDIM) + eps);
    for (uint i = tid; i < KDIM; i += TG) {{
      const T y = T(xs[i] * nscale * float(nw[i]));
      if (gid.x == 0) {{ xnorm[ulong(gid.y) * KDIM + i] = y; }}
      xs[i] = float(y);
    }}
  }}
#else
  for (uint i = tid; i < KDIM; i += TG) {{ xs[i] = float(xrow[i]); }}
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint row0 = (gid.x * NSG + sg) * NR0;
  float acc[NR0];
  for (uint r = 0; r < NR0; ++r) {{ acc[r] = 0.0f; }}
#if SWIGLU
  float up[NR0];
  for (uint r = 0; r < NR0; ++r) {{ up[r] = 0.0f; }}
#endif

  for (uint r = 0; r < NR0; ++r) {{
    uint row = row0 + r;
    if (row >= NDIM) {{ break; }}
    device const T* wp = w + ulong(row) * KDIM;
#if SWIGLU
    device const T* wu = w + (ulong(row) + NDIM) * KDIM;
#endif
    float a = 0.0f;
#if SWIGLU
    float b = 0.0f;
#endif
#if PACK8
    // 16-byte loads: eight T values per lane per load, U loads issued before any is
    // used, so a SIMD-group has 32 x 16 x U bytes of one row in flight.
    device const uint4* wp8 = (device const uint4*)wp;
#if SWIGLU
    device const uint4* wu8 = (device const uint4*)wu;
#endif
    for (uint k0 = lane; k0 * 8 < KDIM; k0 += 32u * U) {{
      uint4 wv[U];
#if SWIGLU
      uint4 uv[U];
#endif
      for (uint u = 0; u < U; ++u) {{
        if ((k0 + 32u * u) * 8 < KDIM) {{
          wv[u] = wp8[k0 + 32u * u];
#if SWIGLU
          uv[u] = wu8[k0 + 32u * u];
#endif
        }}
      }}
      for (uint u = 0; u < U; ++u) {{
        const uint kb = (k0 + 32u * u) * 8;
        if (kb >= KDIM) {{ break; }}
        float wf[8];
        UNPACK8(wv[u], wf);
        for (uint j = 0; j < 8; ++j) {{ a += xs[kb + j] * wf[j]; }}
#if SWIGLU
        float uf[8];
        UNPACK8(uv[u], uf);
        for (uint j = 0; j < 8; ++j) {{ b += xs[kb + j] * uf[j]; }}
#endif
      }}
    }}
#else
    // Vector loads: each lane takes VEC consecutive elements. VEC is 1 unless it divides K,
    // which is also what keeps every row base VEC-aligned.
    for (uint k = lane * VEC; k < KDIM; k += 32u * VEC) {{
#if VEC > 1
      TVEC wv = *((device const TVEC*)(wp + k));
#if SWIGLU
      TVEC uv = *((device const TVEC*)(wu + k));
#endif
      for (uint j = 0; j < VEC; ++j) {{
        float xv = xs[k + j];
        a += xv * float(wv[j]);
#if SWIGLU
        b += xv * float(uv[j]);
#endif
      }}
#else
      float xv = xs[k];
      a += xv * float(wp[k]);
#if SWIGLU
      b += xv * float(wu[k]);
#endif
#endif
    }}
#endif
    acc[r] = simd_sum(a);
#if SWIGLU
    up[r] = simd_sum(b);
#endif
  }}

  if (lane != 0) {{ return; }}
  device T* orow = out + ulong(gid.y) * NDIM;
  for (uint r = 0; r < NR0; ++r) {{
    uint row = row0 + r;
    if (row >= NDIM) {{ return; }}
#if SWIGLU
    float g = acc[r];
    orow[row] = T(g / (1.0f + precise::exp(-g)) * up[r]);   // silu(gate) * up
#else
    orow[row] = T(acc[r]);
#endif
  }}
}}
"""

# (SIMD-groups per threadgroup, output rows per SIMD-group), the same table as
# kernel/metal/nvfp4.py:_TILES: the product amortizes the staged activation tile, and the
# first entry that divides the row count and still fills _MIN_TGS threadgroups wins.
_TILES = ((16, 2), (16, 1), (8, 1), (4, 1), (2, 1), (1, 1))
_MIN_TGS = 16

# MAX_GEMV_ROWS -- above this many activation rows the GEMV re-reads the whole weight per
# row block and F.linear reads it once -- comes from fp8.py; re-exported for moe.py.

# uint4 loads in flight per lane on the PACK8 path.
_U = 2


def _tile(n_rows: int) -> tuple[int, int]:
    return pick_tile(_TILES, n_rows, _MIN_TGS)


@functools.lru_cache(maxsize=None)
def _lib(k: int, n: int, nsg: int, nr0: int, dtype: torch.dtype, swiglu: bool, norm: bool = False):
    # VEC*sizeof(T) must be a legal MSL vector width and the row base must stay aligned
    # to it, which it is when VEC divides K.
    vec = 4 if k % 4 == 0 else 1
    # 16-bit storage with K a multiple of 8 reads the row as uint4s (PACK8); float32
    # keeps the float4 path.
    pack8 = dtype in (torch.float16, torch.bfloat16) and k % 8 == 0
    t = msl_type(dtype)
    unpack = "UNPACK8_BF16" if dtype == torch.bfloat16 else "UNPACK8_F16"
    src = _SOURCE.format(k=k, n=n, nsg=nsg, nr0=nr0, swiglu=int(swiglu), vec=vec,
                         pack8=int(pack8), u=_U)
    return compile(
        f"#define T {t}\n#define TVEC {t}{vec if vec > 1 else ''}\n#define NORM {int(norm)}\n"
        f"#define UNPACK8 {unpack}\n" + src
    )


def _gemv(x: torch.Tensor, w: torch.Tensor, n: int, swiglu: bool, norm=None):
    m, k = x.shape
    nsg, nr0 = _tile(n)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    rows = nsg * nr0
    args = [out, x, w]
    xnorm = None
    if norm is not None:
        # The call site gates on can_run_norm (models/qwen3_5_moe/moe.py:186).
        residual, residual_out, nw, eps = norm
        xnorm = torch.empty_like(x)
        args += [residual, residual_out, xnorm, nw, float(eps)]
    _lib(k, n, nsg, nr0, x.dtype, swiglu, norm is not None).dense_gemv(
        *args, threads=(32 * nsg * -(-n // rows), m), group_size=(32 * nsg, 1),
    )
    return out if norm is None else (out, xnorm)


def can_run_norm(x, residual, residual_out, weight) -> bool:
    """Two more contiguous ``[M, K]`` tensors in ``x``'s dtype (``residual_out`` not the
    same buffer as ``residual``) and a ``[K]`` weight."""
    return (
        all(t.shape == x.shape and t.dtype == x.dtype and t.is_contiguous()
            for t in (residual, residual_out))
        and residual_out.data_ptr() != residual.data_ptr()
        and weight.dtype == x.dtype and weight.numel() == x.shape[1] and weight.is_contiguous()
        and x.shape[1] <= MAX_TG_FLOATS   # the staged row, same bound as can_run
    )


def can_run(x: torch.Tensor, w: torch.Tensor) -> bool:
    """The activation tile is staged in threadgroup memory, so K is bounded; the rest is
    the usual contiguity and dtype agreement."""
    return (
        x.device.type == "mps"
        and x.dtype == w.dtype
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and x.dim() == 2
        and x.shape[1] <= MAX_TG_FLOATS
        and x.is_contiguous()
        and w.is_contiguous()
    )


def dense_gemv(x: torch.Tensor, w: torch.Tensor, *, norm=None):
    """``x [M, K] @ w [N, K].T -> [M, N]``, the drop-in for ``F.linear(x, w)``."""
    assert w.shape[1] == x.shape[1], (w.shape, x.shape)
    return _gemv(x, w, w.shape[0], swiglu=False, norm=norm)


def swiglu_gemv(x: torch.Tensor, gate_up: torch.Tensor) -> torch.Tensor:
    """``silu(x @ gate.T) * (x @ up.T) -> [M, I]`` off the column-merged ``[2I, K]``
    weight: the SwiGLU projection and its activation in one launch, with the
    ``[M, 2I]`` intermediate never written to device memory."""
    assert gate_up.shape[0] % 2 == 0, gate_up.shape
    assert gate_up.shape[1] == x.shape[1], (gate_up.shape, x.shape)
    return _gemv(x, gate_up, gate_up.shape[0] // 2, swiglu=True)


__all__ = ["MAX_GEMV_ROWS", "can_run", "can_run_norm", "dense_gemv", "swiglu_gemv"]
