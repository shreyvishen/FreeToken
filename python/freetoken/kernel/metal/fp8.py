"""Per-tensor FP8 (W8A16) GEMV and dequant for Apple GPUs, the Metal twin of
``kernel/triton/fp8_pertensor_linear``."""

from __future__ import annotations

import functools

import torch

from .shaders import E4M3_U8_TO_F32, compile, msl_type, pick_tile

_SOURCE = r"""
#define KDIM {k}
#define NDIM {n}
#define NSG {nsg}
#define NR0 {nr0}
#define VEC {vec}
#define NW (VEC / 4)

// out[m, n] = scale[n] * sum_k x[m, k] * e4m3(w[n, k]). The activation row is read from device
// memory, not staged: each lane reads its own slice, so staging would only cap threadgroups.
kernel void fp8_gemv(
    device       T*     out   [[buffer(0)]],
    device const T*     x     [[buffer(1)]],
    device const uchar* w     [[buffer(2)]],
    device const float* scale [[buffer(3)]],
#if NX > 0
    device const XW_T*  xw    [[buffer(4)]],   // [NX, KDIM] unquantized tail rows
    device       T*     xout  [[buffer(5)]],   // [M, NX]
#endif
#if CONV
    device const T*     cw    [[buffer(6)]],   // [NC, CK] depthwise causal conv taps
    device       ST_T*  cst   [[buffer(7)]],   // [slots, NC, CK-1] history, in place
    device const int*   cslot [[buffer(8)]],   // [M] state slot per token
#endif
#if NORM
    device const T*     hin   [[buffer(9)]],   // [M, KDIM] the sublayer input
    device const T*     rin   [[buffer(10)]],  // [M, KDIM] the residual stream
    device       T*     rout  [[buffer(11)]],  // [M, KDIM] the new residual, hin + rin
    device const T*     nw    [[buffer(12)]],  // [KDIM] RMSNorm weight
    constant     float& eps   [[buffer(13)]],
#endif
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  device const T* xrow = x + ulong(gid.y) * KDIM;
#if NORM
  // Prologue: residual add + RMSNorm of this token's row. Threadgroup 0 alone writes the new
  // residual, out of place, so no threadgroup reads a row another is writing.
  threadgroup float xs[KDIM];
  {{
    threadgroup float red[NSG];
    device const T* hrow = hin + ulong(gid.y) * KDIM;
    device const T* rrow = rin + ulong(gid.y) * KDIM;
    device       T* orow = rout + ulong(gid.y) * KDIM;
    const uint tid = sg * 32u + lane;
    float ss = 0.0f;
    for (uint k = tid; k < KDIM; k += NSG * 32u) {{
      const T r = T(float(hrow[k]) + float(rrow[k]));
      if (gid.x == 0) {{ orow[k] = r; }}
      const float v = float(r);
      xs[k] = v;
      ss += v * v;
    }}
    ss = simd_sum(ss);
    if (lane == 0) {{ red[sg] = ss; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < NSG; ++i) {{ tot += red[i]; }}
    const float nscale = rsqrt(tot / float(KDIM) + eps);
    for (uint k = tid; k < KDIM; k += NSG * 32u) {{
      xs[k] = float(T(xs[k] * nscale * float(nw[k])));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
#define XV(k) xs[k]
#else
#define XV(k) float(xrow[k])
#endif
#if NX > 0
  // Threadgroups past the FP8 grid serve the unquantized tail: NR0 rows per SIMD-group off a
  // plain [NX, KDIM] weight, same lanes-split-K shape, no scale.
  if (gid.x >= NTGQ) {{
    const uint xrow0 = ((gid.x - NTGQ) * NSG + sg) * NR0;
    float xacc[NR0];
    for (uint r = 0; r < NR0; ++r) {{
      const uint row = xrow0 + r;
      float a = 0.0f;
      if (row < NX) {{
        device const XW_T* wp = xw + ulong(row) * KDIM;
        for (uint k = lane * 4u; k < KDIM; k += 128u) {{
          const XW_T4 wv = *((device const XW_T4*)(wp + k));
          for (uint j = 0; j < 4; ++j) {{ a += XV(k + j) * float(wv[j]); }}
        }}
      }}
      xacc[r] = simd_sum(a);
    }}
    if (lane != 0) {{ return; }}
    for (uint r = 0; r < NR0; ++r) {{
      const uint row = xrow0 + r;
      if (row < NX) {{ xout[ulong(gid.y) * NX + row] = T(xacc[r]); }}
    }}
    return;
  }}
#endif
  const uint row0 = (gid.x * NSG + sg) * NR0;

  float acc[NR0];
  for (uint r = 0; r < NR0; ++r) {{ acc[r] = 0.0f; }}

  for (uint r = 0; r < NR0; ++r) {{
    uint row = row0 + r;
    if (row >= NDIM) {{ break; }}
    device const uchar* wp = w + ulong(row) * KDIM;
    float a = 0.0f;
    // Vector loads: each lane takes VEC consecutive bytes. VEC divides K, which keeps every row
    // base VEC-aligned; MSL has no uchar8/uchar16, so past 4 bytes a lane issues NW uchar4 loads.
    for (uint k = lane * VEC; k < KDIM; k += 32u * VEC) {{
#if VEC >= 4
      for (uint u = 0; u < NW; ++u) {{
        uchar4 wv = *((device const uchar4*)(wp + k + 4u * u));
        for (uint j = 0; j < 4; ++j) {{
          a += XV(k + 4u * u + j) * e4m3_u8_to_f32(wv[j]);
        }}
      }}
#elif VEC == 2
      uchar2 wv = *((device const uchar2*)(wp + k));
      for (uint j = 0; j < 2; ++j) {{ a += XV(k + j) * e4m3_u8_to_f32(wv[j]); }}
#else
      a += XV(k) * e4m3_u8_to_f32(wp[k]);
#endif
    }}
    acc[r] = simd_sum(a);
  }}

  if (lane != 0) {{ return; }}
  device T* orow = out + ulong(gid.y) * NDIM;
  for (uint r = 0; r < NR0; ++r) {{
    uint row = row0 + r;
    if (row >= NDIM) {{ return; }}
#if CONV
    if (row < NC) {{
      // The leading NC rows feed a depthwise causal conv (a GDN's q|k|v channels).
      device ST_T* st = cst + (ulong(cslot[gid.y]) * NC + row) * (CK - 1);
      device const T* wc = cw + row * CK;
      float hist[CK - 1];
      for (uint j = 0; j < CK - 1; ++j) {{ hist[j] = float(st[j]); }}
      float a = 0.0f;
      for (uint j = 0; j < CK - 1; ++j) {{ a += hist[j] * float(wc[j]); }}
      const float xv = float(T(acc[r] * scale[row]));
      a += xv * float(wc[CK - 1]);
      for (uint j = 0; j + 2 < CK; ++j) {{ st[j] = ST_T(hist[j + 1]); }}
      st[CK - 2] = ST_T(xv);
      orow[row] = T(a / (1.0f + exp(-a)));
      continue;
    }}
#endif
    orow[row] = T(acc[r] * scale[row]);
  }}
}}

// weight[n, k] = e4m3(w[n, k]) * scale[n].
kernel void dequant_fp8(
    device       T*     out   [[buffer(0)]],
    device const uchar* w     [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{{
  const uint k0 = (tid * VEC) % KDIM;
  const uint row = (tid * VEC) / KDIM;
  if (row >= NDIM) {{ return; }}
  const float s = scale[row];
  device const uchar* wp = w + ulong(row) * KDIM + k0;
  device T* op = out + ulong(row) * KDIM + k0;
#if VEC >= 4
  for (uint u = 0; u < NW; ++u) {{
    uchar4 wv = *((device const uchar4*)(wp + 4u * u));
    for (uint j = 0; j < 4; ++j) {{ op[4u * u + j] = T(e4m3_u8_to_f32(wv[j]) * s); }}
  }}
#elif VEC == 2
  uchar2 wv = *((device const uchar2*)wp);
  for (uint j = 0; j < 2; ++j) {{ op[j] = T(e4m3_u8_to_f32(wv[j]) * s); }}
#else
  op[0] = T(e4m3_u8_to_f32(wp[0]) * s);
#endif
}}
"""

# (SIMD-groups per threadgroup, output rows per SIMD-group), the first entry whose product
# divides the row count.
_TILES = ((16, 1), (8, 1), (4, 1), (2, 1), (1, 1))

# Above this many rows of activation the GEMV re-reads the whole packed weight per row block,
# where dequantizing once into a scratch for F.linear reads it once.
MAX_GEMV_ROWS = 8

# Bytes each lane loads per step, so a SIMD-group covers 32*_VEC contiguous weight bytes.
_VEC = 16

_E4M3_MAX = 448.0  # largest finite float8_e4m3fn


def _vec(k: int) -> int:
    # VEC*sizeof(uchar) must be a legal MSL vector width and the row base must stay
    # aligned to it, which it is when VEC divides K.
    for v in (_VEC, 8, 4, 2):
        if k % v == 0:
            return v
    return 1


@functools.lru_cache(maxsize=None)
def _lib(k: int, n: int, nsg: int, nr0: int, vec: int, dtype: torch.dtype,
         nx: int = 0, xw_dtype: torch.dtype | None = None,
         conv: tuple[int, int, torch.dtype] | None = None, norm: bool = False):
    t = msl_type(dtype)
    src = _SOURCE.format(k=k, n=n, nsg=nsg, nr0=nr0, vec=vec)
    xt = msl_type(xw_dtype) if nx else "float"
    nc, ck, st = conv if conv else (0, 2, torch.float32)
    head = (f"#include <metal_stdlib>\nusing namespace metal;\n{E4M3_U8_TO_F32}\n#define T {t}\n"
            f"#define NX {nx}\n#define NTGQ {-(-n // (nsg * nr0))}\n"
            f"#define XW_T {xt}\n#define XW_T4 {xt}4\n"
            f"#define CONV {int(conv is not None)}\n#define NC {nc}\n#define CK {ck}\n"
            f"#define ST_T {msl_type(st)}\n#define NORM {int(norm)}\n")
    return compile(head + src)


def fp8_gemv(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    *, extra_weight: torch.Tensor | None = None,
    conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float] | None = None,
    tile: tuple[int, int] | None = None, vec: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``x [M, K] @ (e4m3(weight [N, K]) * scale [N])^T -> [M, N]`` in ``x``'s dtype, with
    three optional fusions, each bit-identical to running its own kernel around this one:
    ``norm = (hidden, residual, residual_out, weight, eps)`` replaces ``x`` by
    ``rmsnorm(hidden + residual) * weight`` in every threadgroup's prologue and writes
    ``hidden + residual`` to ``residual_out``; ``extra_weight [NX, K]`` (unquantized,
    needs ``K % 4 == 0``) adds threadgroups computing ``x @ extra^T`` and makes the call
    return ``(out, extra_out [M, NX])``; ``conv = (taps [NC, CK], state [slots, NC, CK-1],
    slots [M])`` runs a depthwise causal conv's decode step over the leading ``NC`` rows
    in the epilogue, advancing the state in place."""
    m, k = x.shape
    n = weight.shape[0]
    # No _MIN_TGS bar here: this kernel stages nothing, so a dividing tile is the pick.
    nsg, nr0 = tile or pick_tile(_TILES, n)
    rows = nsg * nr0
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    args = [out, x, weight, scale]
    nx = 0
    if extra_weight is not None:
        nx = extra_weight.shape[0]
        extra_out = torch.empty((m, nx), dtype=x.dtype, device=x.device)
        args += [extra_weight, extra_out]
    conv_spec = None
    if conv is not None:
        taps, state, slots = conv
        if nx == 0:
            args += [out, out]   # buffers 4 and 5 are positional; the kernel binds neither
        args += [taps, state, slots.to(torch.int32).contiguous()]
        conv_spec = (taps.shape[0], taps.shape[1], state.dtype)
    if norm is not None:
        hidden, residual, residual_out, nw, eps = norm
        while len(args) < 9:
            args.append(out)   # buffers 4..8 are positional; the kernel binds none of them here
        args += [hidden, residual, residual_out, nw, float(eps)]
    _lib(k, n, nsg, nr0, vec or _vec(k), x.dtype, nx, extra_weight.dtype if nx else None,
         conv_spec, norm is not None).fp8_gemv(
        *args, threads=(32 * nsg * (-(-n // rows) + -(-nx // rows)), m),
        group_size=(32 * nsg, 1),
    )
    return (out, extra_out) if nx else out


def can_run_norm(x, hidden, residual, residual_out, weight) -> bool:
    """Three contiguous ``[M, K]`` tensors and a ``[K]`` weight, all in ``x``'s dtype, and
    a ``K`` that fits the staged row (``K * 4`` bytes of threadgroup memory)."""
    return (
        all(t.shape == x.shape and t.dtype == x.dtype and t.is_contiguous()
            for t in (hidden, residual, residual_out))
        and residual_out.data_ptr() != residual.data_ptr()
        and weight.dtype == x.dtype and weight.numel() == x.shape[1] and weight.is_contiguous()
        and x.shape[1] * 4 <= 32768
    )


def can_run_conv(x: torch.Tensor, weight: torch.Tensor, taps: torch.Tensor,
                 state: torch.Tensor) -> bool:
    """Taps ``[NC, CK]`` in the activation dtype over the leading ``NC <= N`` rows, a
    contiguous ``[slots, NC, CK-1]`` state of any float dtype, ``2 <= CK <= 16``."""
    return (
        taps.dim() == 2
        and taps.dtype == x.dtype
        and 2 <= taps.shape[1] <= 16
        and taps.shape[0] <= weight.shape[0]
        and taps.is_contiguous()
        and state.dim() == 3
        and tuple(state.shape[1:]) == (taps.shape[0], taps.shape[1] - 1)
        and state.is_contiguous()
        and state.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def can_run_extra(x: torch.Tensor, extra_weight: torch.Tensor) -> bool:
    """An unquantized ``[NX, K]`` weight in the activation's dtype, contiguous, with
    ``K`` a multiple of 4 for the tail's vector loads."""
    return (
        extra_weight.dim() == 2
        and extra_weight.dtype == x.dtype
        and extra_weight.shape[1] == x.shape[1]
        and extra_weight.is_contiguous()
        and x.shape[1] % 4 == 0
    )


def dequant_fp8(
    weight: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``e4m3(weight [N, K]) * scale [N] -> [N, K]`` of ``dtype``."""
    n, k = weight.shape
    vec = _vec(k)
    out = torch.empty((n, k), dtype=dtype, device=weight.device)
    _lib(k, n, 1, 1, vec, dtype).dequant_fp8(
        out, weight, scale, threads=(n * k // vec,), group_size=(256,),
    )
    return out


def fp8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
    bias: torch.Tensor | None = None, extra_weight: torch.Tensor | None = None,
    conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``F.linear`` over a per-tensor-FP8 weight, dispatching on the number of activation
    rows: the GEMV at decode M, dequantize-once + ``F.linear`` at prefill M."""
    *lead, k = x.shape
    x2 = x.reshape(-1, k)
    extra = None
    if x2.shape[0] <= MAX_GEMV_ROWS:
        out = fp8_gemv(x2, weight, scale, extra_weight=extra_weight, conv=conv, norm=norm)
        if extra_weight is not None:
            out, extra = out
    else:
        if norm is not None:
            from .norm import fused_add_rmsnorm_metal

            hidden, residual, residual_out, nw, eps = norm
            residual_out.copy_(residual)
            x2 = hidden.reshape(-1, k).clone()
            fused_add_rmsnorm_metal(x2, residual_out.reshape(-1, k), nw, eps)
        out = torch.nn.functional.linear(x2, dequant_fp8(weight, scale, dtype=x.dtype))
        if extra_weight is not None:
            extra = torch.nn.functional.linear(x2, extra_weight)
        if conv is not None:
            from .conv import causal_conv1d_decode_metal

            taps, state, slots = conv
            nc = taps.shape[0]
            out[:, :nc] = causal_conv1d_decode_metal(out[:, :nc].contiguous(), state, taps, slots)
    if bias is not None:
        out = out + bias.to(out.dtype)
    out = out.reshape(*lead, weight.shape[0])
    if extra_weight is None:
        return out
    return out, extra.reshape(*lead, extra_weight.shape[0])


def quantize_fp8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating ``[N, K]`` weight to this path's ``(fp8-e4m3 [N, K], scale float32
    [N])`` pair: ``scale[n] = amax(|w[n, :]|) / 448`` then ``w / scale`` cast to e4m3."""
    assert weight.dim() == 2, weight.shape
    src = weight.detach().to("cpu", torch.float32)
    scale = src.abs().amax(dim=1) / _E4M3_MAX
    # An all-zero row has amax 0: any positive scale reproduces it, and 0 would divide by 0.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    # The clamp only guards the rounding of the row's own maximum; nothing else can exceed 448.
    q = (src / scale[:, None]).clamp_(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return q.to(weight.device), scale.to(weight.device)


__all__ = ["quantize_fp8_per_row", "can_run_conv", "can_run_extra",
           "can_run_norm", "dequant_fp8", "fp8_gemv", "fp8_linear"]
