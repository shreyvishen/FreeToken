"""Paged GQA attention on MPS: one Metal launch for a whole decode batch, and a gather into
contiguous tensors plus ``scaled_dot_product_attention`` per request for prefill."""

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


__all__ = ["paged_attention_mps", "paged_decode_attention_mps"]
