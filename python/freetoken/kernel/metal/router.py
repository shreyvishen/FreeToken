"""The MoE router as one Metal kernel: softmax over the expert logits, top-k and renormalize
in a single launch."""

from __future__ import annotations

import functools
from typing import Tuple

import torch

from .shaders import compile, msl_type

# Threads per row on the threadgroup-scratch path: 8 simdgroups, one element per thread
# at num_experts=256. A smaller expert count only leaves threads idle.
_TG = 256

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define E {num_experts}
#define K {topk}
#define TG {tg}
#define NSG (TG / 32)

kernel void router_topk(
    device float* out_w        [[buffer(0)]],   // [M, K] renormalized probabilities
    device int* out_i          [[buffer(1)]],   // [M, K] expert ids, -1 on padded rows
    device const IN_T* logits  [[buffer(2)]],   // [M, E]
{limit_arg}    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
  uint row = gid.y;
  device const IN_T* x = logits + row * E;

  threadgroup float p[E];
  threadgroup float red_v[NSG];
  threadgroup int red_i[NSG];

  // --- max, for the numerically stable exp ---
  float m = -INFINITY;
  for (uint i = tid; i < E; i += TG) {{
    m = max(m, static_cast<float>(x[i]));
  }}
  m = simd_max(m);
  if (lane == 0) {{ red_v[sg] = m; }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {{
    float t = red_v[0];
    for (uint i = 1; i < NSG; ++i) {{ t = max(t, red_v[i]); }}
    red_v[0] = t;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red_v[0];
  // red_v is reused by the next reduction, so every broadcast read of red_v[0] needs a
  // barrier after it or a fast simdgroup overwrites slot 0 while a slow one still reads.
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- exp and the normalizing sum ---
  float acc = 0.0f;
  for (uint i = tid; i < E; i += TG) {{
    float e = exp(static_cast<float>(x[i]) - m);
    p[i] = e;
    acc += e;
  }}
  acc = simd_sum(acc);
  if (lane == 0) {{ red_v[sg] = acc; }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {{
    float t = red_v[0];
    for (uint i = 1; i < NSG; ++i) {{ t += red_v[i]; }}
    red_v[0] = t;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float denom = red_v[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- k masked argmax passes over the row ---
  float sel_v[K];
  int sel_i[K];
  for (uint kk = 0; kk < K; ++kk) {{
    float bv = -1.0f;   // every p[i] is exp(...) >= 0; a taken slot is set to -2
    int bi = 0;
    for (uint i = tid; i < E; i += TG) {{
      float v = p[i];
      if (v > bv || (v == bv && int(i) < bi)) {{ bv = v; bi = int(i); }}
    }}
    for (uint off = 16; off > 0; off >>= 1) {{
      float ov = simd_shuffle_down(bv, off);
      int oi = simd_shuffle_down(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {{ bv = ov; bi = oi; }}
    }}
    if (lane == 0) {{ red_v[sg] = bv; red_i[sg] = bi; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
      float tv = red_v[0];
      int ti = red_i[0];
      for (uint i = 1; i < NSG; ++i) {{
        if (red_v[i] > tv || (red_v[i] == tv && red_i[i] < ti)) {{ tv = red_v[i]; ti = red_i[i]; }}
      }}
      red_v[0] = tv;
      red_i[0] = ti;
      p[ti] = -2.0f;   // masked out of the next pass
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sel_v[kk] = red_v[0];
    sel_i[kk] = red_i[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);   // before the next pass writes red_v/red_i
  }}

  if (tid != 0) {{ return; }}

  float scale = 1.0f / denom;
{renorm}
  // Padded rows (the CUDA-graph tail) route nowhere: expert id -1, weights left as computed.
{limit_decl}
  for (uint kk = 0; kk < K; ++kk) {{
    out_w[row * K + kk] = sel_v[kk] * scale;
    out_i[row * K + kk] = padded ? -1 : sel_i[kk];
  }}
}}
"""

# One simdgroup instead of eight when a row fits V = ceil(E / 32) registers per lane: the simd_*
# reductions need no barrier, so the K argmax passes never touch threadgroup memory.
_SIMD_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define E {num_experts}
#define K {topk}
#define V (((E) + 31) / 32)

kernel void router_topk(
    device float* out_w        [[buffer(0)]],   // [M, K] renormalized probabilities
    device int* out_i          [[buffer(1)]],   // [M, K] expert ids, -1 on padded rows
    device const IN_T* logits  [[buffer(2)]],   // [M, E]
{limit_arg}    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
  uint row = gid.y;
  device const IN_T* x = logits + row * E;

  // Lane `lane` owns experts lane, lane+32, ...; the tail past E reads -INFINITY, which
  // loses every comparison below and contributes nothing to the sum.
  float v[V];
  float m = -INFINITY;
  #pragma unroll
  for (uint j = 0; j < V; ++j) {{
    uint i = lane + 32u * j;
    v[j] = (i < uint(E)) ? float(x[i]) : -INFINITY;
    m = max(m, v[j]);
  }}
  m = simd_max(m);

  float acc = 0.0f;
  #pragma unroll
  for (uint j = 0; j < V; ++j) {{
    if (lane + 32u * j < uint(E)) {{ v[j] = exp(v[j] - m); acc += v[j]; }}
  }}
  float denom = simd_sum(acc);

  // A taken slot is set to -2, below every exp(...) >= 0 and above the -INFINITY tail,
  // so neither is picked while a real expert is left -- and K <= E guarantees one is.
  float sel_v[K];
  int sel_i[K];
  for (uint kk = 0; kk < K; ++kk) {{
    float bv = -INFINITY;
    // 0, not E: a row that is entirely -inf or NaN beats nothing, and an id of E would be
    // gathered out of bounds by the expert path. The threadgroup kernel above starts at 0 too.
    int bi = 0;
    #pragma unroll
    for (uint j = 0; j < V; ++j) {{
      int i = int(lane + 32u * j);
      if (v[j] > bv || (v[j] == bv && i < bi)) {{ bv = v[j]; bi = i; }}
    }}
    for (uint off = 16; off > 0; off >>= 1) {{
      float ov = simd_shuffle_down(bv, off);
      int oi = simd_shuffle_down(bi, off);
      if (ov > bv || (ov == bv && oi < bi)) {{ bv = ov; bi = oi; }}
    }}
    bv = simd_broadcast_first(bv);
    bi = simd_broadcast_first(bi);
    if (uint(bi) % 32u == lane) {{ v[uint(bi) / 32u] = -2.0f; }}
    sel_v[kk] = bv;
    sel_i[kk] = bi;
  }}

  if (lane != 0) {{ return; }}

  float scale = 1.0f / denom;
{renorm}
  // Padded rows (the CUDA-graph tail) route nowhere: expert id -1, weights left as computed.
{limit_decl}
  for (uint kk = 0; kk < K; ++kk) {{
    out_w[row * K + kk] = sel_v[kk] * scale;
    out_i[row * K + kk] = padded ? -1 : sel_i[kk];
  }}
}}
"""

# Values per lane the register path will hold.
_MAX_V = 16

_LIMIT_ARG = "    device const int* limit    [[buffer(3)]],\n"
_RENORM = r"""  float s = 0.0f;
  for (uint kk = 0; kk < K; ++kk) { s += sel_v[kk]; }
  scale = 1.0f / s;
"""


def _threads(num_experts: int) -> int:
    return 32 if -(-num_experts // 32) <= _MAX_V else _TG


@functools.lru_cache(maxsize=None)
def _library(in_dtype: torch.dtype, num_experts: int, topk: int, renormalize: bool, limited: bool):
    header = f"#define IN_T {msl_type(in_dtype)}\n"
    simd = _threads(num_experts) == 32
    return compile(
        header
        + (_SIMD_SOURCE if simd else _SOURCE).format(
            num_experts=num_experts,
            topk=topk,
            tg=_TG,
            limit_arg=_LIMIT_ARG if limited else "",
            limit_decl=("  bool padded = int(row) >= limit[0];" if limited
                        else "  const bool padded = false;"),
            renorm=_RENORM if renormalize else "",
        )
    )


def supports(gating_output: torch.Tensor, topk: int) -> bool:
    """The kernel's shape contract. Anything outside it falls back to torch."""
    num_experts = gating_output.shape[-1]
    return (
        gating_output.dim() == 2
        and 0 < topk <= num_experts
        and topk <= 32          # sel_v/sel_i are per-thread registers
        and num_experts <= 4096  # threadgroup scratch is one float per expert (16 KiB cap)
        and gating_output.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def fused_topk_softmax_metal(
    gating_output: torch.Tensor,  # [M, num_experts]
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax + top-k + renormalize in one launch."""
    m, num_experts = gating_output.shape
    x = gating_output.contiguous()
    weights = torch.empty((m, topk), dtype=torch.float32, device=x.device)
    ids = torch.empty((m, topk), dtype=torch.int32, device=x.device)
    if m == 0:
        return weights, ids

    lib = _library(x.dtype, num_experts, topk, renormalize, num_token_non_padded is not None)
    args = [weights, ids, x]
    if num_token_non_padded is not None:
        args.append(num_token_non_padded.to(torch.int32).contiguous())
    tg = _threads(num_experts)
    lib.router_topk(*args, threads=(tg, m, 1), group_size=(tg, 1, 1))
    return weights, ids


__all__ = ["fused_topk_softmax_metal", "supports"]
