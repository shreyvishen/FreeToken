"""GatedDeltaNet on Metal: the recurrence as one MSL kernel, ported from mlx_lm's
``gated_delta_step``, plus a fused kernel that runs a whole decode step of the layer."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from .shaders import compile, is_available, msl_type

# Threadgroup y so the simdgroups of a group sweep adjacent dv columns, which share the
# 128 B cache lines of the [Dk, Dv] state rows. mlx's y = 4 uses 1/8 of each line.
_TG_Y = 32

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define DK {dk}
#define DV {dv}
#define HK {hk}
#define HV {hv}
#define N_PER_T (DK / 32)

kernel void gdn_step(
    device IN_T* y             [[buffer(0)]],   // [B, T, HV, DV]
    device ST_T* state         [[buffer(1)]],   // [num_slots, HV, DK, DV], in place
    device const IN_T* q       [[buffer(2)]],   // [B, T, HK, DK], l2-normed and scaled
    device const IN_T* k       [[buffer(3)]],   // [B, T, HK, DK], l2-normed
    device const IN_T* v       [[buffer(4)]],   // [B, T, HV, DV]
    device const float* g      [[buffer(5)]],   // [B, T, HV] log-decay
    device const float* beta   [[buffer(6)]],   // [B, T, HV]
    device const int* slots    [[buffer(7)]],   // [B] state slot per request
    constant uint& T           [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
  uint n = gid.z;
  uint b_idx = n / HV;
  uint hv_idx = n % HV;
  uint hk_idx = hv_idx / (HV / HK);          // GQA: repeat_interleave of q/k heads
  uint dk_lane = tid.x;
  uint dv_idx = gid.y;

  auto q_ = q + b_idx * T * HK * DK + hk_idx * DK;
  auto k_ = k + b_idx * T * HK * DK + hk_idx * DK;
  auto v_ = v + b_idx * T * HV * DV + hv_idx * DV;
  auto y_ = y + b_idx * T * HV * DV + hv_idx * DV;
  auto g_ = g + b_idx * T * HV;
  auto beta_ = beta + b_idx * T * HV;

  // element (dk, dv) of this (request, head) lives at st[dk * DV].
  device ST_T* st = state + (uint(slots[b_idx]) * HV + hv_idx) * DK * DV + dv_idx;

  float s[N_PER_T];
  for (int i = 0; i < N_PER_T; ++i) {{
    s[i] = static_cast<float>(st[(N_PER_T * dk_lane + i) * DV]);
  }}

  for (uint t = 0; t < T; ++t) {{
    float decay = exp(g_[hv_idx]);
    float kv_mem = 0.0f;
    for (int i = 0; i < N_PER_T; ++i) {{
      uint s_idx = N_PER_T * dk_lane + i;
      s[i] *= decay;
      kv_mem += s[i] * static_cast<float>(k_[s_idx]);
    }}
    kv_mem = simd_sum(kv_mem);

    float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_[hv_idx];

    float out = 0.0f;
    for (int i = 0; i < N_PER_T; ++i) {{
      uint s_idx = N_PER_T * dk_lane + i;
      s[i] += static_cast<float>(k_[s_idx]) * delta;
      out += s[i] * static_cast<float>(q_[s_idx]);
    }}
    out = simd_sum(out);
    if (lane == 0) {{
      y_[dv_idx] = static_cast<IN_T>(out);
    }}

    q_ += HK * DK;
    k_ += HK * DK;
    v_ += HV * DV;
    y_ += HV * DV;
    g_ += HV;
    beta_ += HV;
  }}

  for (int i = 0; i < N_PER_T; ++i) {{
    st[(N_PER_T * dk_lane + i) * DV] = static_cast<ST_T>(s[i]);
  }}
}}
"""


@functools.lru_cache(maxsize=None)
def _lib(source: str, header: str, **fmt: int):
    # One cache for all three kernels here.
    return compile(header + (source.format(**fmt) if fmt else source))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Same normalisation the reference applies to q and k."""
    return x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps).to(x.dtype)


def _l2_supported(x: torch.Tensor) -> bool:
    return (
        is_available()
        and x.device.type == "mps"
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def gdn_recurrent_metal(
    q: torch.Tensor,        # [B, T, Hk, Dk]
    k: torch.Tensor,        # [B, T, Hk, Dk]
    v: torch.Tensor,        # [B, T, Hv, Dv]
    g: torch.Tensor,        # [B, T, Hv] log-decay, fp32
    beta: torch.Tensor,     # [B, T, Hv] fp32
    *,
    state_source: torch.Tensor,  # [num_slots, Hv, Dk, Dv], updated in place
    indices: torch.Tensor,       # [B] int32 slot per request
    scale: float,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Gated delta rule over T steps on the GPU, one kernel launch."""
    b, t_len, hk, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    if dk % 32 or dv % _TG_Y:
        raise ValueError(f"metal gdn needs Dk % 32 == 0 and Dv % {_TG_Y} == 0, got {dk}, {dv}")
    if hv % hk:
        raise ValueError(f"metal gdn needs Hv % Hk == 0, got {hv}, {hk}")
    if state_source.shape[1:] != (hv, dk, dv):
        raise ValueError(
            f"state {tuple(state_source.shape)} does not match (*, {hv}, {dk}, {dv})"
        )

    if use_qk_l2norm and _l2_supported(q) and _l2_supported(k):
        # One launch each with the scale folded in, against six per tensor in torch.
        from .norm import l2norm_metal

        q = l2norm_metal(q.contiguous(), post_scale=scale).to(v.dtype)
        k = l2norm_metal(k.contiguous()).to(v.dtype)
    else:
        if use_qk_l2norm:
            q = l2norm(q)
            k = l2norm(k)
        q = (q.float() * scale).to(v.dtype)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.float().contiguous()
    beta = beta.float().contiguous()
    slots = indices.to(torch.int32).contiguous()

    y = torch.empty((b, t_len, hv, dv), dtype=v.dtype, device=v.device)
    lib = _lib(
        _SOURCE,
        f"#define IN_T {msl_type(v.dtype)}\n#define ST_T {msl_type(state_source.dtype)}\n",
        dk=dk, dv=dv, hk=hk, hv=hv,
    )
    lib.gdn_step(
        y, state_source, q, k, v, g, beta, slots, t_len, threads=(32, dv, b * hv),
        group_size=(32, _TG_Y, 1),
    )
    return y


def gdn_recurrent_torch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, *,
    state_source: torch.Tensor, indices: torch.Tensor, scale: float, use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Pure-torch twin of :func:`gdn_recurrent_metal`, same signature and same in-place state
    update: ``gdn_reference.recurrent_gated_delta_rule``'s loop with the state read from
    and written back to the pool."""
    b, t_len, hk, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    if use_qk_l2norm:
        q = l2norm(q)
        k = l2norm(k)
    q, k, v_f = q.float() * scale, k.float(), v.float()
    if hv != hk:
        rep = hv // hk
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    g, beta = g.float(), beta.float()

    idx = indices.to(torch.long)
    state = state_source[idx].float()  # [B, Hv, Dk, Dv]
    out = torch.empty((b, t_len, hv, dv), dtype=torch.float32, device=v.device)
    for i in range(t_len):
        q_t, k_t, v_t = q[:, i], k[:, i], v_f[:, i]
        g_t = g[:, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    # index_put_, not index_copy_: see kernel/metal/ops.py:store_cache (MPS index_copy_
    # is O(destination), not O(indices)).
    state_source.index_put_((idx,), state.to(state_source.dtype))
    return out.to(v.dtype)


# --- the sigmoid/softplus gating ---------------------------------------------

_GATE_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void gdn_gate(
    device float* g            [[buffer(0)]],   // [B, H] log-decay
    device float* beta         [[buffer(1)]],   // [B, H]
    device const IN_T* a       [[buffer(2)]],   // [B, H] raw
    device const IN_T* b       [[buffer(3)]],   // [B, H] raw
    device const P_T* A_log    [[buffer(4)]],   // [H]
    device const P_T* dt_bias  [[buffer(5)]],   // [H]
    constant uint& H           [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint h = gid.x;
  if (h >= H) { return; }
  uint i = gid.y * H + h;

  // softplus at both tails: past +20 the exp overflows and log(1+exp(x)) == x under a
  // float ulp; below -20, 1 + exp(x) rounds to 1. MSL has no log1p, hence the low branch.
  float t = float(a[i]) + float(dt_bias[h]);
  float sp = (t > 20.0f) ? t : ((t < -20.0f) ? exp(t) : log(1.0f + exp(t)));
  g[i] = -exp(float(A_log[h])) * sp;

  float bv = float(b[i]);
  beta[i] = 1.0f / (1.0f + exp(-bv));
}
"""


def gate_params_metal(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(g, beta)`` in one launch."""
    rows = a.reshape(-1, a.shape[-1])
    bb = b.reshape(-1, b.shape[-1])
    m, h = rows.shape
    g = torch.empty((m, h), dtype=torch.float32, device=a.device)
    beta = torch.empty((m, h), dtype=torch.float32, device=a.device)
    if m:
        _lib(
            _GATE_SOURCE,
            f"#define IN_T {msl_type(rows.dtype)}\n#define P_T {msl_type(A_log.dtype)}\n",
        ).gdn_gate(
            g, beta, rows.contiguous(), bb.contiguous(),
            A_log.contiguous(), dt_bias.contiguous(), h,
            threads=(h, m, 1), group_size=(min(h, 256), 1, 1),
        )
    return g.view(a.shape), beta.view(b.shape)


def gate_params(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(g, beta)`` from the raw ``a``/``b`` projection slices, one Metal launch when the
    shapes fit."""
    if (
        is_available()
        and a.device.type == "mps"
        and a.dtype == b.dtype
        and a.shape == b.shape
        and A_log.dtype == dt_bias.dtype
        and a.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and A_log.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and A_log.shape == dt_bias.shape == a.shape[-1:]
    ):
        return gate_params_metal(a, b, A_log, dt_bias)
    beta = b.float().sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g, beta


def gdn_decode_metal(
    q: torch.Tensor,        # [1, B, Hk, Dk]
    k: torch.Tensor,        # [1, B, Hk, Dk]
    v: torch.Tensor,        # [1, B, Hv, Dv]
    a: torch.Tensor,        # [B, Hv] raw
    b: torch.Tensor,        # [B, Hv] raw
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_source: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    use_kernel: bool = True,
) -> torch.Tensor:
    """Single-token decode step, signature-compatible with ``gdn_decode_fla``."""
    g, beta = gate_params(a, b, A_log, dt_bias)
    bs = v.shape[1]
    run = gdn_recurrent_metal if use_kernel else gdn_recurrent_torch
    y = run(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), g.reshape(bs, 1, -1),
        beta.reshape(bs, 1, -1), state_source=state_source, indices=indices, scale=scale,
    )
    return y[:, 0]


def gdn_prefill_metal(
    q: torch.Tensor,        # [1, total, Hk, Dk]
    k: torch.Tensor,
    v: torch.Tensor,        # [1, total, Hv, Dv]
    g: torch.Tensor,        # [1, total, Hv] log-decay
    beta: torch.Tensor,     # [1, total, Hv]
    *,
    state_source: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    use_kernel: bool = True,
    cu_seqlens_host: list[int] | None = None,
) -> torch.Tensor:
    """Varlen prefill, signature-compatible with ``gdn_prefill_chunk_fla``."""
    run = gdn_recurrent_metal if use_kernel else gdn_recurrent_torch
    # FLAMetadata's host copy when the caller has it: one flush per forward, not per layer.
    bounds = cu_seqlens.tolist() if cu_seqlens_host is None else cu_seqlens_host
    outs = []
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi == lo:
            continue
        outs.append(
            run(
                q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi],
                state_source=state_source, indices=indices[i : i + 1], scale=scale,
            )[0]
        )
    return torch.cat(outs, dim=0)  # [total, Hv, Dv]


# --- the whole decode step of one GDN layer as one launch --------------------

# The threadgroup is one (request, v-head), because the output RMSNorm reduces over the whole Dv
# axis, which a threadgroup can do and a grid cannot.
_FUSED_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#define DK {dk}
#define DV {dv}
#define HK {hk}
#define HV {hv}
#define TGY {tgy}
#define NPT (DK / 32)      // dk elements per lane, one contiguous block
#define DVPT (DV / TGY)    // dv columns per simdgroup
#define NREP_K (DK / 32)   // l2norm_metal's simdgroup partials over Dk
#define NREP_V (DV / 32)   // rms_norm_gated_metal's partials over Dv

kernel void gdn_decode_fused(
    device IN_T* out           [[buffer(0)]],   // [B, HV * DV]
    device ST_T* state         [[buffer(1)]],   // [num_slots, HV, DK, DV], in place
    device const IN_T* mixed   [[buffer(2)]],   // [B, mixed_stride] q | k | v after the conv
    device const IN_T* zin     [[buffer(3)]],   // [B, z_stride] output-gate rows
    device const IN_T* araw    [[buffer(4)]],   // [B, ab_stride]
    device const IN_T* braw    [[buffer(5)]],   // [B, ab_stride]
    device const P_T* A_log    [[buffer(6)]],   // [HV]
    device const P_T* dt_bias  [[buffer(7)]],   // [HV]
    device const IN_T* nw      [[buffer(8)]],   // [DV] output-norm weight
    device const int* slots    [[buffer(9)]],   // [B]
    constant uint& mixed_stride  [[buffer(10)]],
    constant uint& z_stride      [[buffer(11)]],
    constant uint& ab_stride     [[buffer(12)]],
    constant float& qk_scale     [[buffer(13)]],
    constant float& l2_eps       [[buffer(14)]],
    constant float& norm_eps     [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
  uint n = gid.z;
  uint b_idx = n / HV;
  uint hv_idx = n % HV;
  uint hk_idx = hv_idx / (HV / HK);          // GQA: repeat_interleave of q/k heads
  uint sg = tid.y;

  device const IN_T* row = mixed + b_idx * mixed_stride;
  device const IN_T* q_ = row + hk_idx * DK;
  device const IN_T* k_ = row + HK * DK + hk_idx * DK;
  device const IN_T* v_ = row + 2 * HK * DK + hv_idx * DV;

  // --- the gating, summand for summand from gdn_gate ---
  uint ab = b_idx * ab_stride + hv_idx;
  float t = float(araw[ab]) + float(dt_bias[hv_idx]);
  float sp = (t > 20.0f) ? t : ((t < -20.0f) ? exp(t) : log(1.0f + exp(t)));
  float decay = exp(-exp(float(A_log[hv_idx])) * sp);
  float bv = float(braw[ab]);
  float beta = 1.0f / (1.0f + exp(-bv));

  // --- the q/k l2 norms, replaying l2norm_metal's partials in its order ---
  float sq = 0.0f;
  float sk = 0.0f;
  for (uint r = 0; r < NREP_K; ++r) {{
    float qv = float(q_[r * 32 + lane]);
    float kv = float(k_[r * 32 + lane]);
    sq += simd_sum(qv * qv);
    sk += simd_sum(kv * kv);
  }}
  float q_scale = rsqrt(sq + l2_eps) * qk_scale;
  float k_scale = rsqrt(sk + l2_eps);

  // Rounded to IN_T exactly where l2norm_metal stored its output tensor.
  float qn[NPT];
  float kn[NPT];
  for (uint i = 0; i < NPT; ++i) {{
    uint s_idx = NPT * lane + i;
    qn[i] = float(IN_T(float(q_[s_idx]) * q_scale));
    kn[i] = float(IN_T(float(k_[s_idx]) * k_scale));
  }}

  // --- the recurrence, one timestep ---
  device ST_T* st = state + (uint(slots[b_idx]) * HV + hv_idx) * DK * DV;
  float s[DVPT][NPT];
  threadgroup float co[DV];

  for (uint j = 0; j < DVPT; ++j) {{
    uint dv_idx = j * TGY + sg;
    float kv_mem = 0.0f;
    for (uint i = 0; i < NPT; ++i) {{
      float sv = float(st[(NPT * lane + i) * DV + dv_idx]) * decay;
      s[j][i] = sv;
      kv_mem += sv * kn[i];
    }}
    kv_mem = simd_sum(kv_mem);

    float delta = (float(v_[dv_idx]) - kv_mem) * beta;

    float o = 0.0f;
    for (uint i = 0; i < NPT; ++i) {{
      s[j][i] += kn[i] * delta;
      o += s[j][i] * qn[i];
      st[(NPT * lane + i) * DV + dv_idx] = ST_T(s[j][i]);
    }}
    o = simd_sum(o);
    // Rounded to IN_T exactly where gdn_step stored its y tensor.
    if (lane == 0) {{ co[dv_idx] = float(IN_T(o)); }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- the gated output norm, replaying rms_norm_gated_metal's partials ---
  threadgroup float nscale;
  if (sg == 0) {{
    float acc = 0.0f;
    for (uint r = 0; r < NREP_V; ++r) {{
      float cv = co[r * 32 + lane];
      acc += simd_sum(cv * cv);
    }}
    if (lane == 0) {{ nscale = rsqrt(acc / float(DV) + norm_eps); }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device IN_T* o_ = out + (b_idx * HV + hv_idx) * DV;
  device const IN_T* zr = zin + b_idx * z_stride + hv_idx * DV;
  for (uint i = sg * 32 + lane; i < DV; i += TGY * 32) {{
    float y = co[i] * nscale * float(nw[i]);
    float gz = float(zr[i]);
    y *= gz / (1.0f + exp(-gz));
    o_[i] = IN_T(y);
  }}
}}
"""

_FUSED_MAX_THREADS = 1024


def _row_stride(x: torch.Tensor) -> int | None:
    # A torch.split of the fused input projection has its column offset already folded
    # into the tensor's data pointer, so the kernel needs only the row pitch.
    if x.dim() != 2 or x.stride(1) != 1:
        return None
    return int(x.stride(0))


def fused_decode_supports(
    mixed, z, a, b, A_log, dt_bias, norm_weight, state_source, hk, hv, dk, dv
) -> bool:
    """The fused kernel's contract; anything outside it falls back to the chain."""
    if not (is_available() and mixed.device.type == "mps"):
        return False
    if dk % 32 or dk > 256 or dv % 32 or dv > 256 or hv % hk:
        # > 256 would need l2norm_metal's strided accumulation to be replayed too,
        # and no GDN config has a head that wide.
        return False
    tgy = min(dv, _FUSED_MAX_THREADS // 32)
    if dv % tgy:
        return False
    if any(_row_stride(t) is None for t in (mixed, z, a, b)):
        return False
    if _row_stride(a) != _row_stride(b):
        # One pitch is passed for both: a and b are always two slices of the
        # same projection row (in_proj, or in_proj_ba under fp8).
        return False
    if mixed.shape[1] < 2 * hk * dk + hv * dv or z.shape[1] < hv * dv:
        return False
    if a.shape[1] < hv or b.shape[1] < hv or a.shape[0] != mixed.shape[0]:
        return False
    if state_source.shape[1:] != (hv, dk, dv):
        return False
    if mixed.dtype != z.dtype or a.dtype != mixed.dtype or b.dtype != mixed.dtype:
        return False
    if norm_weight.dtype != mixed.dtype or norm_weight.numel() != dv:
        return False
    if A_log.dtype != dt_bias.dtype or A_log.shape != dt_bias.shape != (hv,):
        return False
    return (
        mixed.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and A_log.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and state_source.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and norm_weight.is_contiguous()
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
        and state_source.is_contiguous()
    )


def gdn_decode_fused_metal(
    mixed: torch.Tensor,    # [B, conv_dim] q | k | v, the conv output
    z: torch.Tensor,        # [B, Hv*Dv] output-gate rows
    a: torch.Tensor,        # [B, Hv] raw
    b: torch.Tensor,        # [B, Hv] raw
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    state_source: torch.Tensor,  # [num_slots, Hv, Dk, Dv], updated in place
    indices: torch.Tensor,       # [B] slot per request
    scale: float,
    num_k_heads: int,
    head_k_dim: int,
    l2_eps: float = 1e-6,
) -> torch.Tensor:
    """One launch from the conv output to the gated-normed GDN output, ``[B, Hv*Dv]``."""
    hv, dk, dv = state_source.shape[1], head_k_dim, state_source.shape[3]
    bs = mixed.shape[0]
    out = torch.empty((bs, hv * dv), dtype=mixed.dtype, device=mixed.device)
    if bs == 0:
        return out
    tgy = min(dv, _FUSED_MAX_THREADS // 32)
    lib = _lib(
        _FUSED_SOURCE,
        f"#define IN_T {msl_type(mixed.dtype)}\n"
        f"#define ST_T {msl_type(state_source.dtype)}\n"
        f"#define P_T {msl_type(A_log.dtype)}\n",
        dk=dk, dv=dv, hk=num_k_heads, hv=hv, tgy=tgy,
    )
    lib.gdn_decode_fused(
        out, state_source, mixed, z, a, b, A_log, dt_bias, norm_weight,
        indices.to(torch.int32).contiguous(), _row_stride(mixed), _row_stride(z), _row_stride(a),
        float(scale), float(l2_eps), float(norm_eps), threads=(32, tgy, bs * hv),
        group_size=(32, tgy, 1),
    )
    return out


__all__ = [
    "fused_decode_supports", "gate_params", "gdn_decode_fused_metal", "gdn_decode_metal",
    "gdn_prefill_metal", "gdn_recurrent_metal", "gdn_recurrent_torch", "is_available", "l2norm",
]
