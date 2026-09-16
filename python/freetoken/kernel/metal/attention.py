"""Paged GQA attention on MPS: one Metal launch for a whole decode batch, one for a whole
prefill batch, and a gather into contiguous tensors plus ``scaled_dot_product_attention`` per
request for the shapes neither kernel is defined for."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from .shaders import compile, msl_type


# Flat 1-D scratch buffers keyed by (slot, dtype, device), capacity rounded up to a power of
# two.
_SCRATCH: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def _scratch(slot: int, numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (slot, dtype, device)
    buf = _SCRATCH.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(1 << max(numel - 1, 0).bit_length(), dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf[:numel]


def _gather_kv_rows(cache: torch.Tensor, slots: torch.Tensor, slot: int) -> torch.Tensor:
    kv_heads, head_dim = cache.shape[1], cache.shape[2]
    n = slots.numel()
    rows = _scratch(slot, n * kv_heads * head_dim, cache.dtype, cache.device)
    rows = rows.view(n, kv_heads, head_dim)
    torch.index_select(cache, 0, slots, out=rows)
    return rows


def _decode_attention(
    q: torch.Tensor, k_rows: torch.Tensor, v_rows: torch.Tensor, sm_scale: float,
) -> torch.Tensor:
    # Grouped-query without expanding K/V: the queries fold into [kv_heads, group, D] so each KV
    # head's bmm serves its whole group.
    kv_len, kv_heads, head_dim = k_rows.shape
    group = q.shape[0] // kv_heads
    n_qk = kv_heads * group * head_dim

    qf = _scratch(5, n_qk, torch.float32, q.device).view(kv_heads, group, head_dim)
    qf.copy_(q.view(kv_heads, group, head_dim))
    kf = _scratch(6, kv_len * kv_heads * head_dim, torch.float32, q.device)
    kf = kf.view(kv_len, kv_heads, head_dim)
    kf.copy_(k_rows)
    vf = _scratch(7, kv_len * kv_heads * head_dim, torch.float32, q.device)
    vf = vf.view(kv_len, kv_heads, head_dim)
    vf.copy_(v_rows)

    probs = _scratch(3, kv_heads * group * kv_len, torch.float32, q.device)
    probs = probs.view(kv_heads, group, kv_len)
    torch.bmm(qf, kf.permute(1, 2, 0), out=probs)
    probs.mul_(sm_scale)
    probs.sub_(probs.amax(-1, keepdim=True))
    probs.exp_()
    probs.div_(probs.sum(-1, keepdim=True))

    acc = _scratch(8, n_qk, torch.float32, q.device).view(kv_heads, group, head_dim)
    torch.bmm(probs, vf.permute(1, 0, 2), out=acc)
    out = _scratch(4, n_qk, q.dtype, q.device).view(kv_heads, group, head_dim)
    out.copy_(acc)
    return out.view(q.shape[0], head_dim)


# A whole decode batch in one launch, structured like MLX's sdpa_vector: a threadgroup per
# (query head, request), BN SIMD-groups striding the request's KV positions, the 32 lanes
# splitting the head dim.
_DECODE_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define D {d}
#define HQ {hq}
#define GROUP {group}
#define BN {bn}
#define EPT {ept}
#define QPT {qpt}

kernel void paged_decode_attention(
    device       T*      out      [[buffer(0)]],
    device const T*      q        [[buffer(1)]],
    device const T*      k_cache  [[buffer(2)]],
    device const T*      v_cache  [[buffer(3)]],
    device const int*    indptr   [[buffer(4)]],
    device const int*    indices  [[buffer(5)]],
    device const int*    q_indptr [[buffer(6)]],
    constant     float&  sm_scale [[buffer(7)]],
#if HAS_GATE
    device const T*      gate     [[buffer(8)]],   // [nnz, HQ * D] pre-sigmoid output gate
#endif
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{{
  const uint qh0 = gid.y * QPT;              // QPT query heads of one group per group
  const uint req = gid.z;
  const uint kvh = qh0 / GROUP;              // GQA: GROUP query heads share a KV head
  const uint hkv = HQ / GROUP;

  const int  kv_lo  = indptr[req];
  const int  kv_len = indptr[req + 1] - kv_lo;
  const uint qrow   = uint(q_indptr[req]);   // decode: one query token per request

  device const T* qp = q + (ulong(qrow) * HQ + qh0) * D;
  float qreg[QPT][EPT];
  for (uint u = 0; u < QPT; ++u) {{
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      qreg[u][i] = (d < D) ? float(qp[u * D + d]) : 0.0f;
    }}
  }}

  // Online softmax over this SIMD-group's slice of the KV positions, one running
  // (max, sum, output) per query head this threadgroup carries.
  float m[QPT], s[QPT], acc[QPT][EPT];
  for (uint u = 0; u < QPT; ++u) {{
    m[u] = -INFINITY;
    s[u] = 0.0f;
    for (uint i = 0; i < EPT; ++i) {{ acc[u][i] = 0.0f; }}
  }}

  for (int p = int(sg); p < kv_len; p += BN) {{
    ulong slot = ulong(uint(indices[kv_lo + p]));
    // The K row and the V row are read once and serve all QPT query heads.
    device const T* kp = k_cache + (slot * hkv + kvh) * D;
    float kreg[EPT];
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      kreg[i] = (d < D) ? float(kp[d]) : 0.0f;
    }}
    device const T* vp = v_cache + (slot * hkv + kvh) * D;
    float vreg[EPT];
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      vreg[i] = (d < D) ? float(vp[d]) : 0.0f;
    }}

    for (uint u = 0; u < QPT; ++u) {{
      float dot = 0.0f;
      for (uint i = 0; i < EPT; ++i) {{ dot += qreg[u][i] * kreg[i]; }}
      dot = simd_sum(dot) * sm_scale;

      float nm = max(m[u], dot);
      float f  = precise::exp(m[u] - nm);    // 0 on the first position (m == -inf)
      float e  = precise::exp(dot - nm);
      for (uint i = 0; i < EPT; ++i) {{ acc[u][i] = acc[u][i] * f + e * vreg[i]; }}
      s[u] = s[u] * f + e;
      m[u] = nm;
    }}
  }}

  // Merge the BN partial softmaxes, per query head.
  threadgroup float tg_m[BN * QPT];
  threadgroup float tg_s[BN * QPT];
  threadgroup float tg_o[BN * QPT * D];
  for (uint u = 0; u < QPT; ++u) {{
    if (lane == 0) {{ tg_m[sg * QPT + u] = m[u]; tg_s[sg * QPT + u] = s[u]; }}
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      if (d < D) {{ tg_o[(sg * QPT + u) * D + d] = acc[u][i]; }}
    }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) {{ return; }}

  for (uint u = 0; u < QPT; ++u) {{
    float gm = -INFINITY;
    for (uint j = 0; j < BN; ++j) {{ gm = max(gm, tg_m[j * QPT + u]); }}
    float w[BN];
    float gs = 0.0f;
    for (uint j = 0; j < BN; ++j) {{
      // A SIMD-group with no positions (kv_len < BN) kept m = -inf: contribute nothing
      // rather than exp(-inf - -inf) = NaN.
      w[j] = (tg_m[j * QPT + u] == -INFINITY) ? 0.0f : precise::exp(tg_m[j * QPT + u] - gm);
      gs += tg_s[j * QPT + u] * w[j];
    }}
    float inv = (gs > 0.0f) ? 1.0f / gs : 0.0f;

    device T* op = out + (ulong(qrow) * HQ + qh0 + u) * D;
    for (uint i = 0; i < EPT; ++i) {{
      uint d = lane + i * 32u;
      if (d >= D) {{ continue; }}
      float o = 0.0f;
      for (uint j = 0; j < BN; ++j) {{ o += tg_o[(j * QPT + u) * D + d] * w[j]; }}
#if HAS_GATE
      // Gated attention's output gate.
      const float g = float(gate[(ulong(qrow) * HQ + qh0 + u) * D + d]);
      op[d] = T(float(T(o * inv)) / (1.0f + exp(-g)));
#else
      op[d] = T(o * inv);
#endif
    }}
  }}
}}
"""

_TG_BYTES = 32768  # Metal's threadgroup memory limit on Apple silicon


@lru_cache(maxsize=None)
def _decode_lib(head_dim: int, q_heads: int, group: int, dtype: torch.dtype, bn: int,
                qpt: int, has_gate: bool = False):
    src = _DECODE_SRC.format(
        d=head_dim, hq=q_heads, group=group, bn=bn, ept=-(-head_dim // 32), qpt=qpt
    )
    return compile(f"#define T {msl_type(dtype)}\n#define HAS_GATE {int(has_gate)}\n" + src)


# Query heads per threadgroup: a threadgroup reads its KV head's K and V rows once and
# serves _QPT of them from registers, which divides the threadgroup count by as much.
_QPT = 1


def paged_decode_attention_mps(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indptr: torch.Tensor,
    indices: torch.Tensor, cu_seqlens_q: torch.Tensor, sm_scale: float,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Whole decode batch, one launch: ``q [bs, HQ, D]`` against the ``[slots, HKV, D]``
    caches, over the KV slots ``indices`` lists and ``indptr`` delimits."""
    bs = indptr.numel() - 1
    q_heads, head_dim = q.shape[1], q.shape[2]
    kv_heads = k_cache.shape[1]
    qpt = _QPT
    while qpt > 1 and (q_heads // kv_heads) % qpt:   # must divide the GQA group
        qpt //= 2
    # BN is capped by the merge buffers: one float32 output row plus a running max and sum
    # per (SIMD-group, query head) must fit the 32 KiB threadgroup limit.
    bn = max(1, min(8, _TG_BYTES // (qpt * (4 * head_dim + 8))))
    # No-ops in the engine (prepare_metadata already builds int32 and the caches are
    # contiguous views); they keep direct callers, i.e. the tests, on the same kernel.
    indptr = indptr.to(torch.int32).contiguous()
    indices = indices.to(torch.int32).contiguous()
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    q = q.contiguous()
    out = torch.empty_like(q)
    args = [out, q, k_cache, v_cache, indptr, indices, cu_seqlens_q, float(sm_scale)]
    if gate is not None:
        assert gate.dtype == q.dtype and gate.numel() == q.numel(), (gate.shape, gate.dtype,
                                                                     q.shape)
        args.append(gate.contiguous())
    lib = _decode_lib(head_dim, q_heads, q_heads // kv_heads, q.dtype, bn, qpt, gate is not None)
    lib.paged_decode_attention(
        *args, threads=(32 * bn, q_heads // qpt, bs), group_size=(32 * bn, 1, 1),
    )
    return out


# FlashAttention-2 tiling for prefill: a threadgroup per (query tile, query head) walks the
# request's KV in tiles of C with a running (max, sum, accumulator) per query row, so no score
# reaches device memory and the staged working set is O(C * head_dim). Paged K/V has to stage
# through threadgroup memory, which llama.cpp's fa.metal never does; splitting the score and
# then the output columns across the SIMD-groups keeps the register budget flat in head_dim.
_PREFILL_SRC = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define NW 32
#define DV (D / 4)              // 4-wide staging: one lane's load is 4 head-dim elements
#define RB (BQ / 8)             // query-row blocks per threadgroup
#define CB (C / 8)              // KV blocks per tile
#define NC (CB / NSG)           // score column blocks this SIMD-group owns
#define DB (D / (8 * NSG))      // output column blocks this SIMD-group owns
#define NQ (BQ / NSG)           // query rows this SIMD-group runs the softmax for
#define RPP (NW / C)            // of those, how many it does per pass (C lanes each)
#define NP (NQ / RPP)

kernel void paged_prefill_attention(
    device       T*      out      [[buffer(0)]],
    device const T*      q        [[buffer(1)]],
    device const T*      k_cache  [[buffer(2)]],
    device const T*      v_cache  [[buffer(3)]],
    device const int*    indptr   [[buffer(4)]],
    device const int*    indices  [[buffer(5)]],
    device const int*    q_indptr [[buffer(6)]],
    constant     int&    bs       [[buffer(7)]],
    constant     float&  sm_scale [[buffer(8)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
  // Query head on x: threadgroups dispatch x-fastest, so the GROUP heads sharing a KV head
  // run together and hit cache instead of pulling the same rows from DRAM GROUP times over.
  const uint h = gid.x;
  // Ragged batch: scan the device-side query indptr for the tile's request. A host-built
  // tile map instead costs a 0.17 ms upload, the kernel's whole margin at a short prompt.
  uint req = 0, r0 = 0;
  for (uint i = 0, t = gid.y; i < uint(bs); ++i) {
    const uint n = (uint(q_indptr[i + 1]) - uint(q_indptr[i]) + BQ - 1) / BQ;
    if (t < n) { req = i; r0 = t * BQ; break; }
    t -= n;
  }
  const uint hkv = HQ / GROUP;
  const uint kvh = h / GROUP;

  const uint q_lo   = uint(q_indptr[req]);
  const uint q_len  = uint(q_indptr[req + 1]) - q_lo;
  const uint kv_lo  = uint(indptr[req]);
  const uint kv_len = uint(indptr[req + 1]) - kv_lo;
  const uint rows   = min(uint(BQ), q_len - r0);
  const uint prefix = kv_len - q_len;
  // Causal: row j reaches KV position prefix + r0 + j, so every tile past the last row's
  // limit is entirely masked and is never read at all.
  const uint kv_end = min(kv_len, prefix + r0 + rows);

  threadgroup TV    skvv[C * DV];    // the K tile, then overwritten by the V tile
  threadgroup T     sp[BQ * C];
  threadgroup float ss[BQ * C];      // scores, and the epilogue's 8x8 staging area
  threadgroup float sf[RB * 64];     // diag(rescale) per row block, fed to the matrix unit
  threadgroup T* skv = (threadgroup T*) skvv;

  const simdgroup_float8x8 zero = simdgroup_float8x8(0.0f);
  simdgroup_float8x8 acc[RB][DB];
  for (uint rb = 0; rb < RB; ++rb) {
    for (uint cb = 0; cb < DB; ++cb) { acc[rb][cb] = zero; }
  }
  float m[NP], sum[NP];
  for (uint pp = 0; pp < NP; ++pp) { m[pp] = -INFINITY; sum[pp] = 0.0f; }
  const uint col = lane % C;

  for (uint p0 = 0; p0 < kv_end; p0 += C) {
    const uint kvn = min(uint(C), kv_end - p0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < C * DV; i += NSG * NW) {
      const uint j = i / DV;
      const uint d = (i - j * DV) * 4;
      const ulong slot = ulong(uint(indices[kv_lo + p0 + min(j, kvn - 1)]));
      skvv[i] = (j < kvn)
          ? *((device const TV*) (k_cache + (slot * hkv + kvh) * D + d)) : TV(0);
    }
    for (uint i = tid; i < RB * 64; i += NSG * NW) { sf[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Q is read from device memory: rows HQ * D apart, contiguous along the head dim, is
    // exactly a fragment. Staging it costs 40 %, competing with the K/V tile for both.
    for (uint rb = 0; rb < RB; ++rb) {
      if (rb * 8 >= rows) { continue; }     // a dead row block, and its load would be OOB
      for (uint cc = 0; cc < NC; ++cc) {
        const uint kb = cc * NSG + sg;
        simdgroup_float8x8 qk = zero;
        for (uint db = 0; db < D / 8; ++db) {
          simdgroup_matrix<T, 8, 8> mq, mk;
          simdgroup_load(mq, q + (ulong(q_lo + r0 + rb * 8) * HQ + h) * D + db * 8, HQ * D);
          simdgroup_load(mk, skv + ulong(kb * 8) * D + db * 8, D, ulong2(0, 0), true);
          simdgroup_multiply_accumulate(qk, mq, mk, qk);
        }
        simdgroup_store(qk, ss + ulong(rb * 8) * C + kb * 8, C);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pp = 0; pp < NP; ++pp) {
      const uint j = (pp * RPP + lane / C) * NSG + sg;
      const bool live = j < rows && col < kvn && p0 + col <= prefix + r0 + j;
      const float x = live ? ss[j * C + col] * sm_scale : -INFINITY;
      float mx = x;
      for (uint k = 1; k < C; k <<= 1) { mx = max(mx, simd_shuffle_xor(mx, ushort(k))); }
      const float nm = max(m[pp], mx);
      // exp(-inf - -inf) is a NaN, so a row that has still seen nothing carries no weight.
      const float ms = (m[pp] == -INFINITY) ? 0.0f : precise::exp(m[pp] - nm);
      // Round P before summing it: the matmul below reads the rounded copy, so the softmax
      // denominator has to be built from the same numbers.
      const float p = live ? float(T(precise::exp(x - nm))) : 0.0f;
      float tot = p;
      for (uint k = 1; k < C; k <<= 1) { tot += simd_shuffle_xor(tot, ushort(k)); }
      sp[j * C + col] = T(p);
      if (col == 0) { sf[(j / 8) * 64 + (j % 8) * 9] = ms; }
      m[pp] = nm;
      sum[pp] = sum[pp] * ms + tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < C * DV; i += NSG * NW) {
      const uint j = i / DV;
      const uint d = (i - j * DV) * 4;
      const ulong slot = ulong(uint(indices[kv_lo + p0 + min(j, kvn - 1)]));
      skvv[i] = (j < kvn)
          ? *((device const TV*) (v_cache + (slot * hkv + kvh) * D + d)) : TV(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // diag(rescale) through the matrix unit rather than thread_elements(): which fragment
    // element a lane holds is not part of the MSL contract.
    for (uint rb = 0; rb < RB; ++rb) {
      simdgroup_float8x8 fm;
      simdgroup_load(fm, sf + rb * 64, 8);
      for (uint cb = 0; cb < DB; ++cb) {
        simdgroup_multiply_accumulate(acc[rb][cb], fm, acc[rb][cb], zero);
      }
      for (uint t = 0; t < CB; ++t) {
        simdgroup_matrix<T, 8, 8> mp, mv;
        simdgroup_load(mp, sp + ulong(rb * 8) * C + t * 8, C);
        for (uint cb = 0; cb < DB; ++cb) {
          simdgroup_load(mv, skv + ulong(t * 8) * D + (sg * DB + cb) * 8, D);
          simdgroup_multiply_accumulate(acc[rb][cb], mp, mv, acc[rb][cb]);
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < RB * 64; i += NSG * NW) { sf[i] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint pp = 0; pp < NP; ++pp) {
    const uint j = (pp * RPP + lane / C) * NSG + sg;
    if (col == 0) { sf[(j / 8) * 64 + (j % 8) * 9] = (sum[pp] > 0.0f) ? 1.0f / sum[pp] : 0.0f; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint rb = 0; rb < RB; ++rb) {
    simdgroup_float8x8 fm;
    simdgroup_load(fm, sf + rb * 64, 8);
    for (uint cb = 0; cb < DB; ++cb) {
      simdgroup_multiply_accumulate(acc[rb][cb], fm, acc[rb][cb], zero);
      simdgroup_store(acc[rb][cb], ss + sg * 64, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = lane; i < 64; i += NW) {
        const uint r = rb * 8 + i / 8;
        if (r < rows) {
          out[(ulong(q_lo + r0 + r) * HQ + h) * D + (sg * DB + cb) * 8 + i % 8] =
              T(ss[sg * 64 + i]);
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
"""


# (SIMD-groups, KV tile, query tile), best measured first. C carries 8 score columns per
# SIMD-group and at most 32 so a softmax row fits the lanes, BQ/NSG divides into whole softmax
# passes, and BQ 32 fits at head_dim 256 but spills its 32 accumulator fragments.
_PREFILL_TILES = ((4, 32, 16), (4, 32, 8), (2, 16, 16), (2, 16, 8),
                  (1, 16, 8), (1, 8, 8))


def prefill_tile(head_dim: int, elem: int) -> tuple[int, int, int] | None:
    """The first ``(nsg, kv_tile, q_tile)`` whose staging fits the threadgroup limit, or
    None when even the smallest does not -- the caller keeps the SDPA path."""
    for nsg, c, bq in _PREFILL_TILES:
        if head_dim % (8 * nsg) or (bq // nsg) % (32 // c):
            continue
        used = c * head_dim * elem + bq * c * (4 + elem) + (bq // 8) * 256
        if used <= _TG_BYTES:
            return nsg, c, bq
    return None


@lru_cache(maxsize=None)
def _prefill_lib(head_dim: int, q_heads: int, group: int, dtype: torch.dtype, nsg: int,
                 c: int, bq: int):
    defs = {"T": msl_type(dtype), "TV": msl_type(dtype) + "4", "D": head_dim,
            "HQ": q_heads, "GROUP": group, "NSG": nsg, "C": c, "BQ": bq}
    return compile("".join(f"#define {k} {v}\n" for k, v in defs.items()) + _PREFILL_SRC)


def paged_prefill_attention_mps(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indptr: torch.Tensor,
    indices: torch.Tensor, cu_seqlens_q: torch.Tensor, sm_scale: float,
    q_ptr: list[int] | None = None,
) -> torch.Tensor:
    """Causal paged attention over a ragged prefill batch, one launch: query row ``j`` of
    request ``i`` attends KV positions ``0 .. prefix + j``, ``prefix = kv_len - q_len``."""
    q_heads, head_dim = q.shape[1], q.shape[2]
    nsg, c, bq = prefill_tile(head_dim, q.element_size())
    if q_ptr is None:
        q_ptr = cu_seqlens_q.tolist()
    # The host only counts the tiles, and how far past the last query row the kernel reads.
    n_tiles, last_row = 0, 0
    for i in range(len(q_ptr) - 1):
        lo, hi = int(q_ptr[i]), int(q_ptr[i + 1])
        for r0 in range(0, hi - lo, bq):
            n_tiles += 1
            last_row = max(last_row, lo + r0 + -(-min(bq, hi - lo - r0) // 8) * 8)
    indptr = indptr.to(torch.int32).contiguous()
    indices = indices.to(torch.int32).contiguous()
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    q = q.contiguous()
    out = torch.empty_like(q)
    if last_row > q.shape[0]:
        # The kernel reads Q eight rows at a time, so a last row block that is only partly
        # live still loads eight. Give it the rows rather than bounds-check every fragment.
        padded = q.new_zeros(last_row, *q.shape[1:])
        padded[: q.shape[0]] = q
        q = padded
    lib = _prefill_lib(head_dim, q_heads, q_heads // k_cache.shape[1], q.dtype, nsg, c, bq)
    lib.paged_prefill_attention(
        out, q, k_cache, v_cache, indptr, indices, cu_seqlens_q, len(q_ptr) - 1,
        float(sm_scale),
        threads=(32 * nsg * q_heads, n_tiles, 1), group_size=(32 * nsg, 1, 1),
    )
    return out


def paged_attention_mps(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indptr: torch.Tensor,
    indices: torch.Tensor, cu_seqlens_q: torch.Tensor, sm_scale: float,
    kv_ptr: list[int] | None = None, q_ptr: list[int] | None = None,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal paged attention over ``bs`` requests: query token ``j`` attends to KV positions
    ``0 .. prefix + j``, ``prefix = kv_len - q_len``, which covers decode, whole-prompt
    prefill and chunked-prefill continuation with one rule."""
    # Both indptrs drive a host loop, so reading them off the device flushes the Metal pipeline;
    # `prepare_metadata` hands down the lists it already holds.
    if kv_ptr is None:
        kv_ptr = indptr.tolist()
    if q_ptr is None:
        q_ptr = cu_seqlens_q.tolist()

    # Decode -- one query token per request, one dtype, shapes the kernel is defined for
    # -- is the whole batch in one launch, with nothing read back to the host.
    if (q_ptr is not None and len(q_ptr) - 1 == q.shape[0]
            and all(int(q_ptr[i + 1]) - int(q_ptr[i]) == 1 for i in range(len(q_ptr) - 1))
            and q.dtype == k_cache.dtype == v_cache.dtype
            and q.shape[1] % k_cache.shape[1] == 0
            and k_cache.is_contiguous() and v_cache.is_contiguous()):
        return paged_decode_attention_mps(
            q, k_cache, v_cache, indptr, indices, cu_seqlens_q, sm_scale, gate=gate
        )
    if gate is not None:
        # The torch path has no epilogue: the gate is the elementwise launch it always was.
        from .elementwise import sigmoid_gate_mul_metal

        out = paged_attention_mps(q, k_cache, v_cache, indptr, indices, cu_seqlens_q, sm_scale,
                                  kv_ptr, q_ptr)
        return sigmoid_gate_mul_metal(out.reshape(gate.shape), gate).reshape(out.shape)

    # Prefill is one launch too. What is left for SDPA below is an odd head_dim, a dtype the
    # matrix units have no fragment for, or a cache that is not a flat slab.
    if (q.dtype == k_cache.dtype == v_cache.dtype
            and q.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and q.shape[1] % k_cache.shape[1] == 0
            and k_cache.is_contiguous() and v_cache.is_contiguous()
            and prefill_tile(q.shape[2], q.element_size()) is not None):
        return paged_prefill_attention_mps(
            q, k_cache, v_cache, indptr, indices, cu_seqlens_q, sm_scale, q_ptr
        )

    idx = indices.to(torch.long)
    out = torch.empty_like(q)

    for i in range(len(q_ptr) - 1):
        q_lo, q_hi = int(q_ptr[i]), int(q_ptr[i + 1])
        kv_lo, kv_hi = int(kv_ptr[i]), int(kv_ptr[i + 1])
        q_len, kv_len = q_hi - q_lo, kv_hi - kv_lo
        if q_len == 0:
            continue
        slots = idx[kv_lo:kv_hi]
        k_rows = _gather_kv_rows(k_cache, slots, 0)
        v_rows = _gather_kv_rows(v_cache, slots, 1)

        # The one query token sees all of its KV, so the whole of SDPA is one softmax;
        # doing it here keeps K and V unexpanded and every buffer in scratch.
        if q_len == 1 and q.dtype == k_cache.dtype:
            out[q_lo] = _decode_attention(q[q_lo], k_rows, v_rows, sm_scale)
            continue

        k = k_rows.transpose(0, 1).unsqueeze(0)
        v = v_rows.transpose(0, 1).unsqueeze(0)
        qi = q[q_lo:q_hi].transpose(0, 1).unsqueeze(0)

        prefix = kv_len - q_len
        if q_len == 1:
            mask, causal = None, False  # the single query token sees all of its KV
        elif prefix == 0:
            mask, causal = None, True
        else:
            pos = torch.arange(q_len, device=q.device).unsqueeze(1) + prefix
            mask = pos >= torch.arange(kv_len, device=q.device).unsqueeze(0)
            mask, causal = mask.view(1, 1, q_len, kv_len), False

        o = F.scaled_dot_product_attention(
            qi, k, v, attn_mask=mask, is_causal=causal, scale=sm_scale, enable_gqa=True
        )
        out[q_lo:q_hi] = o[0].transpose(0, 1)
    return out


__all__ = ["paged_attention_mps", "paged_decode_attention_mps",
           "paged_prefill_attention_mps"]
