"""The QSA indexer on Metal: compress, ring store, norm+rope, block scores, block expansion and
the sparse attend, each addressing the pool as ``indices[indptr[req] + logical]`` like
``attention.py``. Kernels rather than torch: every step is per-row ragged, and the decode tape
rejects a torch op that allocates without an ``out=`` form."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from .shaders import compile, ept, msl_type

_TG_BYTES = 32768  # Metal's threadgroup memory limit on Apple silicon


# D per row across 32 lanes as EPT elements each: one SIMD-group per row, RMSNorm as a simd_sum.
# The rope reads its NeoX partner from threadgroup memory, so rotary_dim need not align to 32.
_PRELUDE = r"""
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define HALF {half}
#define EPT {ept}

inline void qsa_norm_rope_store(
    thread float* acc, threadgroup float* y, device T* dst,
    device const float* cos_sin, device const float* weight, float eps, int position, uint lane)
{{
  float ss = 0.0f;
  for (uint i = 0; i < EPT; ++i) {{
    uint d = lane + i * 32u;
    if (d < D) {{ ss += acc[i] * acc[i]; }}
  }}
  const float r = rsqrt(simd_sum(ss) / float(D) + eps);
  for (uint i = 0; i < EPT; ++i) {{
    uint d = lane + i * 32u;
    if (d < D) {{ y[d] = acc[i] * r * (1.0f + weight[d]); }}
  }}
  simdgroup_barrier(mem_flags::mem_threadgroup);
  device const float* cs = cos_sin + ulong(uint(max(position, 0))) * ulong(2 * HALF);
  for (uint i = 0; i < EPT; ++i) {{
    uint d = lane + i * 32u;
    if (d >= D) {{ continue; }}
    float v = y[d];
    if (d < 2 * HALF) {{
      const uint pair = d % HALF;
      v = (d < HALF) ? (y[d] * cs[pair] - y[d + HALF] * cs[HALF + pair])
                     : (y[d] * cs[pair] + y[d - HALF] * cs[HALF + pair]);
    }}
    dst[d] = T(v);
  }}
}}
"""


# The head count arrives as a grid dimension rather than a compile-time constant, so one
# SIMD-group runs one (row, head) and the library is shared across head counts.
_NORM_ROPE_SRC = _PRELUDE + r"""
kernel void qsa_index_norm_rope(
    device       T*      out          [[buffer(0)]],   // [rows, HEADS, D]
    device const T*      x            [[buffer(1)]],   // [rows, HEADS, D] raw index queries
    device const int*    positions    [[buffer(2)]],   // [rows]
    device const float*  cos_sin      [[buffer(3)]],
    device const float*  weight       [[buffer(4)]],
    constant     float&  eps          [[buffer(5)]],
    constant     int&    heads        [[buffer(6)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  const uint head = gid.y;
  const uint row  = gid.z;
  const ulong base = (ulong(row) * uint(heads) + head) * D;
  float acc[EPT];
  for (uint i = 0; i < EPT; ++i) {{
    uint d = lane + i * 32u;
    acc[i] = (d < D) ? float(x[base + d]) : 0.0f;
  }}
  threadgroup float y[D];
  qsa_norm_rope_store(acc, y, out + base, cos_sin, weight, eps, positions[row], lane);
}}
"""


_COMPRESS_SRC = _PRELUDE + r"""
#define RATIO {ratio}
#define CAP {cap}

kernel void qsa_compress_groups(
    device       T*      cmp          [[buffer(0)]],   // [slab_rows, D]
    device const T*      raw          [[buffer(1)]],   // [rows, D] this forward's raw index keys
    device const T*      ring         [[buffer(2)]],   // one layer's [slots, CAP, D] ring view
    device const int*    ring_slots   [[buffer(3)]],   // [bs] request slot of the ring / scratch
    device const int*    q_indptr     [[buffer(4)]],   // [bs + 1]
    device const int*    token_to_req [[buffer(5)]],   // [rows]
    device const int*    positions    [[buffer(6)]],   // [rows] logical positions
    device const int*    out_loc      [[buffer(7)]],   // [rows] this token's KV slot
    device const float*  cos_sin      [[buffer(8)]],
    device const float*  weight       [[buffer(9)]],
    constant     float&  eps          [[buffer(10)]],
    constant     int&    scratch_base [[buffer(11)]],
    constant     int&    ring_stride  [[buffer(12)]],  // elements between ring slots
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  const uint row     = gid.y;
  const int  req     = token_to_req[row];
  const int  q_start = q_indptr[req];
  const int  pos     = positions[row];
  const int  chunk   = positions[q_start];      // first logical position of this forward
  const int  slot    = ring_slots[req];

  float acc[EPT];
  for (uint i = 0; i < EPT; ++i) {{ acc[i] = 0.0f; }}
  // A group spans the pending ring and this forward's raw rows. An incomplete group still
  // pools onto the scratch row below, never read, so its reads need only stay in bounds.
  for (int g = 0; g < RATIO; ++g) {{
    const int p = pos - (RATIO - 1 - g);
    device const T* src = (p >= chunk)
        ? raw + ulong(uint(q_start + p - chunk)) * D
        : ring + ulong(uint(slot)) * uint(ring_stride) + ulong((p % CAP + CAP) % CAP) * D;
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      if (d < D) {{ acc[i] += float(src[d]); }}
    }}
  }}
  for (uint i = 0; i < EPT; ++i) {{ acc[i] /= float(RATIO); }}

  // out_loc % page_size == position % page_size and RATIO divides page_size, so a group
  // closes exactly on out_loc % RATIO == RATIO - 1 and its slab row is out_loc / RATIO.
  const int kv_slot = out_loc[row];
  const int dest = (kv_slot % RATIO == RATIO - 1) ? (kv_slot / RATIO) : (scratch_base + slot);
  threadgroup float y[D];
  qsa_norm_rope_store(
      acc, y, cmp + ulong(uint(dest)) * D, cos_sin, weight, eps, pos - RATIO + 1, lane);
}}
"""


_RING_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define CAP {cap}
#define EPT {ept}

kernel void qsa_store_ring(
    device       T*      ring         [[buffer(0)]],   // one layer's [slots, CAP, D] view
    device const T*      raw          [[buffer(1)]],   // [rows, D]
    device const int*    ring_slots   [[buffer(2)]],
    device const int*    q_indptr     [[buffer(3)]],
    device const int*    positions    [[buffer(4)]],
    constant     int&    ring_stride  [[buffer(5)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  const uint req   = gid.y;
  const int  start = q_indptr[req];
  const int  end   = q_indptr[req + 1];
  if (start >= end) {{ return; }}
  // Only a request's last CAP rows survive the forward. A shorter chunk rewrites its first row
  // into that row's own residue slot, which is idempotent and needs no host-side count.
  const int src = max(end - CAP + int(gid.z), start);
  const int p   = positions[src];
  device       T* dst = ring + ulong(uint(ring_slots[req])) * uint(ring_stride)
                             + ulong(uint(p % CAP)) * D;
  device const T* s   = raw + ulong(uint(src)) * D;
  for (uint i = 0; i < EPT; ++i) {{
    uint d = lane + i * 32u;
    if (d < D) {{ dst[d] = s[d]; }}
  }}
}}
"""


# One threadgroup per query row; BN SIMD-groups stride the block columns and the 32 lanes split
# the index head dim, so a block's compressed key is read once and reduced with simd_sum.
_SCORE_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define H {h}
#define RATIO {ratio}
#define BN {bn}
#define EPT {ept}

kernel void qsa_block_scores(
    device       float*  logits       [[buffer(0)]],   // [rows, columns]
    device const T*      q            [[buffer(1)]],   // [rows, H, D] post norm+rope
    device const T*      cmp          [[buffer(2)]],   // [slab_rows, D]
    device const int*    kv_indptr    [[buffer(3)]],
    device const int*    kv_indices   [[buffer(4)]],   // physical slot per logical token
    device const int*    token_to_req [[buffer(5)]],
    device const int*    positions    [[buffer(6)]],
    constant     int&    columns      [[buffer(7)]],
    constant     float&  scale        [[buffer(8)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  const uint row    = gid.y;
  const int  req    = token_to_req[row];
  const int  kv_lo  = kv_indptr[req];
  const int  kv_len = kv_indptr[req + 1] - kv_lo;
  // Slab rows are never cleared, so a stale block must stay unreachable: bound by the blocks
  // this row may see AND by the ones this request has written.
  const int  visible = min((positions[row] + 1) / RATIO, kv_len / RATIO);

  device const T* qp = q + ulong(row) * H * D;
  float qreg[H][EPT];
  for (uint h = 0; h < H; ++h) {{
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      qreg[h][i] = (d < D) ? float(qp[h * D + d]) : 0.0f;
    }}
  }}

  device float* out = logits + ulong(row) * uint(columns);
  for (int c = int(sg); c < columns; c += BN) {{
    if (c >= visible) {{
      if (lane == 0) {{ out[c] = -INFINITY; }}
      continue;
    }}
    const ulong slab = ulong(uint(kv_indices[kv_lo + c * RATIO])) / RATIO;
    device const T* kp = cmp + slab * D;
    float kreg[EPT];
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      kreg[i] = (d < D) ? float(kp[d]) : 0.0f;
    }}
    float total = 0.0f;
    for (uint h = 0; h < H; ++h) {{
      float dot = 0.0f;
      for (uint i = 0; i < EPT; ++i) {{ dot += qreg[h][i] * kreg[i]; }}
      total += max(simd_sum(dot), 0.0f);
    }}
    if (lane == 0) {{ out[c] = total * scale; }}
  }}
}}
"""


# One thread per (row, selection column): top blocks expand to RATIO tokens each, then the row's
# open causal tail. Ranks past the visible-block count are never read, so -inf cannot win.
_EXPAND_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define RATIO {ratio}
#define TOPK {topk}

kernel void qsa_expand(
    device       int*    out          [[buffer(0)]],   // [rows, width]
    device const int*    blocks       [[buffer(1)]],   // [rows, TOPK]
    device const int*    positions    [[buffer(2)]],
    device const int*    kv_indptr    [[buffer(3)]],
    device const int*    token_to_req [[buffer(4)]],
    constant     int&    width        [[buffer(5)]],
    uint3 tp [[thread_position_in_grid]])
{{
  const int col = int(tp.x);
  if (col >= width) {{ return; }}
  const uint row    = tp.y;
  const int  req    = token_to_req[row];
  const int  kv_len = kv_indptr[req + 1] - kv_indptr[req];
  const int  seen   = positions[row] + 1;
  const int  expanded = min(min(seen / RATIO, kv_len / RATIO), TOPK) * RATIO;

  int token = -1;
  if (col < expanded) {{
    token = blocks[ulong(row) * TOPK + ulong(col / RATIO)] * RATIO + col % RATIO;
  }} else {{
    const int tail_start = (seen / RATIO) * RATIO;
    const int offset = col - expanded;
    if (offset < seen - tail_start) {{ token = tail_start + offset; }}
  }}
  out[ulong(row) * ulong(uint(width)) + ulong(uint(col))] = (token < kv_len) ? token : -1;
}}
"""


# A threadgroup per (query head, row); BN SIMD-groups stride the selected tokens with an online
# softmax each, merged in threadgroup memory: attention.py's decode kernel over a selection list.
_ATTEND_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define HQ {hq}
#define GROUP {group}
#define BN {bn}
#define EPT {ept}

kernel void qsa_index_attention(
    device       T*      out          [[buffer(0)]],
    device const T*      q            [[buffer(1)]],   // [rows, HQ, D]
    device const T*      k_cache      [[buffer(2)]],
    device const T*      v_cache      [[buffer(3)]],
    device const int*    sel          [[buffer(4)]],   // [rows, width], -1 padded
    device const int*    kv_indptr    [[buffer(5)]],
    device const int*    kv_indices   [[buffer(6)]],
    device const int*    token_to_req [[buffer(7)]],
    constant     int&    width        [[buffer(8)]],
    constant     float&  sm_scale     [[buffer(9)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  const uint qh    = gid.y;
  const uint row   = gid.z;
  const uint kvh   = qh / GROUP;
  const uint hkv   = HQ / GROUP;
  const int  kv_lo = kv_indptr[token_to_req[row]];

  device const T* qp = q + (ulong(row) * HQ + qh) * D;
  float qreg[EPT];
  for (uint i = 0; i < EPT; ++i) {{
    uint d = lane + i * 32u;
    qreg[i] = (d < D) ? float(qp[d]) : 0.0f;
  }}

  float m = -INFINITY, s = 0.0f, acc[EPT];
  for (uint i = 0; i < EPT; ++i) {{ acc[i] = 0.0f; }}

  device const int* srow = sel + ulong(row) * uint(width);
  for (int c = int(sg); c < width; c += BN) {{
    const int token = srow[c];
    if (token < 0) {{ continue; }}     // uniform across the SIMD-group: one row per threadgroup
    const ulong slot = ulong(uint(kv_indices[kv_lo + token]));
    device const T* kp = k_cache + (slot * hkv + kvh) * D;
    device const T* vp = v_cache + (slot * hkv + kvh) * D;
    float kreg[EPT], vreg[EPT];
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      kreg[i] = (d < D) ? float(kp[d]) : 0.0f;
      vreg[i] = (d < D) ? float(vp[d]) : 0.0f;
    }}
    float dot = 0.0f;
    for (uint i = 0; i < EPT; ++i) {{ dot += qreg[i] * kreg[i]; }}
    dot = simd_sum(dot) * sm_scale;

    float nm = max(m, dot);
    float f  = precise::exp(m - nm);     // 0 on the first token (m == -inf)
    float e  = precise::exp(dot - nm);
    for (uint i = 0; i < EPT; ++i) {{ acc[i] = acc[i] * f + e * vreg[i]; }}
    s = s * f + e;
    m = nm;
  }}

  threadgroup float tg_m[BN];
  threadgroup float tg_s[BN];
  threadgroup float tg_o[BN * D];
  if (lane == 0) {{ tg_m[sg] = m; tg_s[sg] = s; }}
  for (uint i = 0; i < EPT; ++i) {{
    uint d = lane + i * 32u;
    if (d < D) {{ tg_o[sg * D + d] = acc[i]; }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) {{ return; }}

  float gm = -INFINITY;
  for (uint j = 0; j < BN; ++j) {{ gm = max(gm, tg_m[j]); }}
  float w[BN];
  float gs = 0.0f;
  for (uint j = 0; j < BN; ++j) {{
    // A SIMD-group that saw no token kept m = -inf: weigh it 0, not exp(-inf - -inf) = NaN.
    w[j] = (tg_m[j] == -INFINITY) ? 0.0f : precise::exp(tg_m[j] - gm);
    gs += tg_s[j] * w[j];
  }}
  const float inv = (gs > 0.0f) ? 1.0f / gs : 0.0f;

  device T* op = out + (ulong(row) * HQ + qh) * D;
  for (uint i = 0; i < EPT; ++i) {{
    uint d = lane + i * 32u;
    if (d >= D) {{ continue; }}
    float o = 0.0f;
    for (uint j = 0; j < BN; ++j) {{ o += tg_o[j * D + d] * w[j]; }}
    op[d] = T(o * inv);
  }}
}}
"""


@lru_cache(maxsize=None)
def _norm_rope_lib(head_dim: int, half: int, dtype: torch.dtype):
    src = _NORM_ROPE_SRC.format(d=head_dim, half=half, ept=ept(head_dim))
    return compile(f"#define T {msl_type(dtype)}\n" + src)


@lru_cache(maxsize=None)
def _compress_lib(head_dim: int, half: int, ratio: int, cap: int, dtype: torch.dtype):
    src = _COMPRESS_SRC.format(
        d=head_dim, half=half, ept=ept(head_dim), ratio=ratio, cap=cap
    )
    return compile(f"#define T {msl_type(dtype)}\n" + src)


@lru_cache(maxsize=None)
def _ring_lib(head_dim: int, cap: int, dtype: torch.dtype):
    src = _RING_SRC.format(d=head_dim, cap=cap, ept=ept(head_dim))
    return compile(f"#define T {msl_type(dtype)}\n" + src)


@lru_cache(maxsize=None)
def _score_lib(head_dim: int, heads: int, ratio: int, dtype: torch.dtype, bn: int):
    src = _SCORE_SRC.format(d=head_dim, h=heads, ratio=ratio, bn=bn, ept=ept(head_dim))
    return compile(f"#define T {msl_type(dtype)}\n" + src)


@lru_cache(maxsize=None)
def _expand_lib(ratio: int, topk: int):
    return compile(_EXPAND_SRC.format(ratio=ratio, topk=topk))


@lru_cache(maxsize=None)
def _attend_lib(head_dim: int, q_heads: int, group: int, dtype: torch.dtype, bn: int):
    src = _ATTEND_SRC.format(d=head_dim, hq=q_heads, group=group, bn=bn, ept=ept(head_dim))
    return compile(f"#define T {msl_type(dtype)}\n" + src)


def qsa_index_norm_rope_mps(
    x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, weight: torch.Tensor,
    eps: float, out: torch.Tensor,
) -> torch.Tensor:
    """Zero-centered ``(1 + w)`` RMSNorm then partial NeoX rope on ``x`` [rows, heads, dim]."""
    rows, heads, head_dim = x.shape
    # One MSL scalar type serves x and out, and the rope table is read as float32.
    assert out.dtype == x.dtype, (out.dtype, x.dtype)
    assert cos_sin.dtype == weight.dtype == torch.float32, (cos_sin.dtype, weight.dtype)
    if not rows:
        return out
    lib = _norm_rope_lib(head_dim, cos_sin.shape[1] // 2, x.dtype)
    lib.qsa_index_norm_rope(
        out, x.contiguous(), positions, cos_sin, weight, float(eps), int(heads),
        threads=(32, heads, rows), group_size=(32, 1, 1),
    )
    return out


def qsa_compress_groups_mps(
    cmp_k: torch.Tensor, raw: torch.Tensor, ring: torch.Tensor, ring_slots: torch.Tensor,
    q_indptr: torch.Tensor, token_to_req: torch.Tensor, positions: torch.Tensor,
    out_loc: torch.Tensor, cos_sin: torch.Tensor, weight: torch.Tensor, eps: float,
    ratio: int, scratch_base: int,
) -> None:
    """Pool every row's closing group (ring + this forward's raw keys), norm and rope it at the
    group's first position, and scatter it into the slab; a row whose group does not close
    lands on its request's scratch row."""
    rows, head_dim = raw.shape
    # One MSL scalar type reads raw and ring and writes cmp, and the rope table is float32.
    assert cmp_k.dtype == ring.dtype == raw.dtype, (cmp_k.dtype, ring.dtype, raw.dtype)
    assert cos_sin.dtype == weight.dtype == torch.float32, (cos_sin.dtype, weight.dtype)
    if not rows:
        return
    lib = _compress_lib(head_dim, cos_sin.shape[1] // 2, ratio, ring.shape[1], raw.dtype)
    lib.qsa_compress_groups(
        cmp_k, raw, ring, ring_slots, q_indptr, token_to_req, positions, out_loc, cos_sin,
        weight, float(eps), int(scratch_base), int(ring.stride(0)),
        threads=(32, rows, 1), group_size=(32, 1, 1),
    )


def qsa_store_ring_mps(
    ring: torch.Tensor, raw: torch.Tensor, ring_slots: torch.Tensor, q_indptr: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Keep each request's last ``ring_capacity`` raw index keys, at ``position % capacity``."""
    rows, head_dim = raw.shape
    capacity = ring.shape[1]
    assert ring.dtype == raw.dtype, (ring.dtype, raw.dtype)
    if not rows:
        return
    lib = _ring_lib(head_dim, capacity, raw.dtype)
    lib.qsa_store_ring(
        ring, raw, ring_slots, q_indptr, positions, int(ring.stride(0)),
        threads=(32, q_indptr.numel() - 1, capacity), group_size=(32, 1, 1),
    )


def qsa_block_scores_mps(
    q_index: torch.Tensor, cmp_k: torch.Tensor, kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor, token_to_req: torch.Tensor, positions: torch.Tensor,
    ratio: int, logits: torch.Tensor,
) -> torch.Tensor:
    """``logits[r, b] = sum_h relu(<q_h[r], kbar_b>) / sqrt(index_head_dim)`` over the blocks
    row ``r`` may see, ``-inf`` past them."""
    rows, heads, head_dim = q_index.shape
    if heads > 8:
        raise NotImplementedError(f"qsa block scoring keeps the query heads in registers, got {heads}")
    assert logits.shape[0] == rows and logits.dtype == torch.float32
    if not rows:
        return logits
    bn = max(1, min(8, logits.shape[1]))
    lib = _score_lib(head_dim, heads, ratio, q_index.dtype, bn)
    lib.qsa_block_scores(
        logits, q_index.contiguous(), cmp_k, kv_indptr, kv_indices, token_to_req, positions,
        int(logits.shape[1]), float(1.0 / math.sqrt(head_dim)),
        threads=(32 * bn, rows, 1), group_size=(32 * bn, 1, 1),
    )
    return logits


def qsa_expand_mps(
    blocks: torch.Tensor, positions: torch.Tensor, kv_indptr: torch.Tensor,
    token_to_req: torch.Tensor, ratio: int, out: torch.Tensor,
) -> torch.Tensor:
    """Expand the top blocks to token ids and append the causal tail of the open group."""
    rows, topk = blocks.shape
    width = out.shape[1]
    if not rows:
        return out
    group = min(256, width)
    lib = _expand_lib(ratio, topk)
    lib.qsa_expand(
        out, blocks, positions, kv_indptr, token_to_req, int(width),
        threads=(-(-width // group) * group, rows, 1), group_size=(group, 1, 1),
    )
    return out


def qsa_index_attention_mps(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, sel: torch.Tensor,
    kv_indptr: torch.Tensor, kv_indices: torch.Tensor, token_to_req: torch.Tensor,
    sm_scale: float, out: torch.Tensor,
) -> torch.Tensor:
    """GQA over each query row's own token list: ``sel[r]`` holds request-logical token ids
    (-1 padding), resolved to pool slots through ``kv_indices[kv_indptr[req] + token]``."""
    rows, q_heads, head_dim = q.shape
    kv_heads = k_cache.shape[1]
    assert sel.shape[0] == rows and sel.dtype == torch.int32
    assert q.dtype == k_cache.dtype == v_cache.dtype
    assert q_heads % kv_heads == 0
    if not rows:
        return out
    # BN is capped by the merge buffers: one float32 output row plus a running max and sum per
    # SIMD-group must fit the threadgroup limit.
    bn = max(1, min(8, _TG_BYTES // (4 * head_dim + 8), sel.shape[1]))
    lib = _attend_lib(head_dim, q_heads, q_heads // kv_heads, q.dtype, bn)
    lib.qsa_index_attention(
        out, q.contiguous(), k_cache, v_cache, sel, kv_indptr, kv_indices, token_to_req,
        int(sel.shape[1]), float(sm_scale),
        threads=(32 * bn, q_heads, rows), group_size=(32 * bn, 1, 1),
    )
    return out


__all__ = [
    "qsa_block_scores_mps",
    "qsa_compress_groups_mps",
    "qsa_expand_mps",
    "qsa_index_attention_mps",
    "qsa_index_norm_rope_mps",
    "qsa_store_ring_mps",
]
