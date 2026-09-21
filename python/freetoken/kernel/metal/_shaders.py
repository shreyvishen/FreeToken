"""MSL sources for the Metal NVFP4 kernels."""

from __future__ import annotations

import functools

from .shaders import E4M3_U8_TO_F32

# Preamble shared by every kernel below: the e4m3 decode and the E2M1 value table.
_PRELUDE = r"""
#include <metal_stdlib>
using namespace metal;

constant float E2M1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// The low 16 bits of a packed word: elements 4n..4n+3, each in bits [4j, 4j+3].
inline float4 e2m1x4(uint p) {
    return float4(E2M1[p & 0xFu], E2M1[(p >> 4) & 0xFu],
                  E2M1[(p >> 8) & 0xFu], E2M1[(p >> 12) & 0xFu]);
}
""" + E4M3_U8_TO_F32

# Dequant, the port of the Triton _dequant_nvfp4_kernel. OUT_T comes from the caller.
DEQUANT_SRC = _PRELUDE + r"""
kernel void dequant_nvfp4(
    device       OUT_T * out          [[buffer(0)]],
    device const uchar * packed       [[buffer(1)]],
    device const uchar * scale        [[buffer(2)]],
    device const half  * glob         [[buffer(3)]],
    device const int   * slots        [[buffer(4)]],
    constant     int   & OUT          [[buffer(5)]],
    constant     int   & IN_PACKED    [[buffer(6)]],
    constant     int   & NUM_BLOCKS   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    // gid.x = byte index inside the row, gid.y = flattened (expert, output row).
    int byte_off = int(gid.x);
    int pid_row  = int(gid.y);
    if (byte_off >= IN_PACKED) { return; }

    int n       = pid_row / OUT;
    int out_idx = pid_row % OUT;

    // 64-bit row arithmetic: slot * OUT * IN overflows int32 for a full cache
    // (nvfp4_dequant.py:44-46).
    long slot = long(slots[n]);
    long row  = slot * long(OUT) + long(out_idx);

    float g = float(glob[row]);

    uchar b  = packed[row * long(IN_PACKED) + long(byte_off)];
    int   lo = b & 0xF;
    int   hi = (b >> 4) & 0xF;

    // Both nibbles of a byte (elements 2b, 2b+1) share one 16-wide block.
    int   blk = byte_off / 8;
    float s   = e4m3_u8_to_f32(scale[row * long(NUM_BLOCKS) + long(blk)]) * g;

    long base = long(pid_row) * long(IN_PACKED) * 2;
    out[base + 2 * byte_off + 0] = OUT_T(E2M1[lo] * s);
    out[base + 2 * byte_off + 1] = OUT_T(E2M1[hi] * s);
}
"""

# Decode gemm1: gate/up GEMV with dequant in the load path and SiLU-mul fused.
GATE_UP_SRC = _PRELUDE + r"""
kernel void nvfp4_moe_gate_up_silu(
    device       float * out        [[buffer(0)]],   // [routes, INTER] fp32
    device const X_T   * x          [[buffer(1)]],   // [M, KDIM] activation dtype
    device const uint  * packed     [[buffer(2)]],   // [E, 2I, KDIM/8] (uint8 viewed as uint)
    device const uchar * scale      [[buffer(3)]],   // [E, 2I, KDIM/16]
    device const half  * glob       [[buffer(4)]],   // [E, 2I]
    device const int   * topk_ids   [[buffer(5)]],   // [M, TOP_K] int32
    constant     int   & TOP_K      [[buffer(6)]],
#if SHARED
    // The shared expert as route TOP_K of every token: a one-expert bank of the routed
    // [2I, KDIM] shape, plus a gate row whose pre-sigmoid dot is written for the down kernel.
    device const uint  * s_packed   [[buffer(7)]],   // [2I, KDIM/8]
    device const uchar * s_scale    [[buffer(8)]],   // [2I, KDIM/16]
    device const half  * s_glob     [[buffer(9)]],   // [2I]
    device const X_T   * gw         [[buffer(10)]],  // [KDIM] the shared expert's gate row
    device       X_T   * gout       [[buffer(11)]],  // [M] pre-sigmoid gate per token
#endif
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  tiitg [[thread_index_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const uint W  = KDIM / 8;    // 32-bit words per weight row
    const uint NB = KDIM / 16;   // e4m3 block scales per weight row

    const uint route = tgid.y;   // token-major: routes per token are TOP_K (+1 shared)
#if SHARED
    const uint RPT   = uint(TOP_K) + 1u;
    const uint m     = route / RPT;
    const bool shared = (route % RPT) == uint(TOP_K);
    if (shared && tgid.x == 0) {
        // The scalar gate: every thread a K slice, one simd_sum per SIMD-group, one
        // pass over the partials. Its consumer is the down launch.
        threadgroup float gpart[NSG];
        float gp = 0.0f;
        for (uint t = tiitg; t < KDIM; t += NSG * 32u) {
            gp += float(x[ulong(m) * KDIM + t]) * float(gw[t]);
        }
        gp = simd_sum(gp);
        if (tiisg == 0) { gpart[sgitg] = gp; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg == 0) {
            float gsum = 0.0f;
            for (uint sgi = 0; sgi < NSG; ++sgi) { gsum += gpart[sgi]; }
            gout[m] = X_T(gsum);
        }
    }
    device const uint  * pk = shared ? s_packed : packed;
    device const uchar * ps = shared ? s_scale  : scale;
    device const half  * pg = shared ? s_glob   : glob;
#else
    const uint m     = route / uint(TOP_K);
    device const uint  * pk = packed;
    device const uchar * ps = scale;
    device const half  * pg = glob;
#endif

    // The token's activations, as float4 so the inner loop reads 16 bytes at a time.
    device const X_T4 * xs4 = (device const X_T4 *)(x + ulong(m) * KDIM);

    // NR0 consecutive intermediate indices per SIMD-group (ggml's structure): each lane reuses
    // its activation words across NR0 rows, so a larger NR0 cuts the activation reads.
    const uint i0 = (tgid.x * NSG + sgitg) * NR0;

#if SHARED
    const long e = shared ? 0L : long(topk_ids[m * uint(TOP_K) + route % RPT]);
#else
    const long e = long(topk_ids[route]);
#endif
    long grow[NR0];
    #pragma unroll
    for (uint j = 0; j < NR0; ++j) {
        // Clamp rather than branch: a tail threadgroup re-reads the last row and
        // drops the store, so the loop below stays uniform.
        const uint i = min(i0 + j, uint(INTER) - 1u);
        grow[j] = e * long(2 * INTER) + long(i);
    }

    float ag[NR0];
    float au[NR0];
    #pragma unroll
    for (uint j = 0; j < NR0; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

    // Unrolled 2x so two iterations' loads are in flight per lane; 8x spills.
    #pragma unroll 2
    for (uint w = tiisg; w < W; w += 32u) {
        // One word is 8 e2m1 codes == elements 8w..8w+7 == two float4 of activations.
        const float4 a0 = float4(xs4[2u * w]);
        const float4 a1 = float4(xs4[2u * w + 1u]);
        const uint blk = w >> 1;   // 16 k-values per block == 2 words
        #pragma unroll
        for (uint j = 0; j < NR0; ++j) {
            const long gr = grow[j];
            const long ur = gr + long(INTER);
            const uint wg = pk[gr * long(W) + long(w)];
            const uint wu = pk[ur * long(W) + long(w)];
            const float sg = e4m3_u8_to_f32(ps[gr * long(NB) + long(blk)]);
            const float su = e4m3_u8_to_f32(ps[ur * long(NB) + long(blk)]);
            ag[j] += sg * (dot(e2m1x4(wg), a0) + dot(e2m1x4(wg >> 16), a1));
            au[j] += su * (dot(e2m1x4(wu), a0) + dot(e2m1x4(wu >> 16), a1));
        }
    }

    // Deferred reduction: one simd_sum per row, not one per K iteration.
    #pragma unroll
    for (uint j = 0; j < NR0; ++j) {
        const float g = simd_sum(ag[j]) * float(pg[grow[j]]);
        const float u = simd_sum(au[j]) * float(pg[grow[j] + long(INTER)]);
        if (tiisg == 0 && i0 + j < uint(INTER)) {
            // silu(gate) * up, matching kernel/triton/activation.py:6 with d = INTER.
            out[ulong(route) * INTER + i0 + j] = (g / (1.0f + exp(-g))) * u;
        }
    }
}
"""

# Decode gemm2: down GEMV, weighted by the router probabilities and summed over the TOP_K
# routes inside the kernel, so no [M, TOP_K, H] scratch and no separate reduce.
DOWN_SRC = _PRELUDE + r"""
kernel void nvfp4_moe_down(
    device       OUT_T * out        [[buffer(0)]],   // [M, HDIM] activation dtype
    device const float * inter      [[buffer(1)]],   // [M, TOPK, KDIM] fp32
    device const uint  * packed     [[buffer(2)]],   // [E, HDIM, KDIM/8]
    device const uchar * scale      [[buffer(3)]],   // [E, HDIM, KDIM/16]
    device const half  * glob       [[buffer(4)]],   // [E, HDIM]
    device const int   * topk_ids   [[buffer(5)]],   // [M, TOPK] int32
    device const float * topk_w     [[buffer(6)]],   // [M, TOPK] fp32
#if HAS_BASE
    device const OUT_T * base       [[buffer(7)]],   // [M, HDIM] activation dtype
#endif
#if SHARED
    // The shared expert as route TOPK of every token: its own [HDIM, KDIM] bank, weighted by
    // sigmoid(gate[m]) and summed into the routed registers. Same threadgroup count.
    device const uint  * s_packed   [[buffer(8)]],   // [HDIM, KDIM/8]
    device const uchar * s_scale    [[buffer(9)]],   // [HDIM, KDIM/16]
    device const half  * s_glob     [[buffer(10)]],  // [HDIM]
    device const OUT_T * sgate      [[buffer(11)]],  // [M] pre-sigmoid, from the gate/up launch
#define RPT (TOPK + 1)
#else
#define RPT TOPK
#endif
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  tiitg [[thread_index_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const uint W  = KDIM / 8;
    const uint NB = KDIM / 16;

    const uint m = tgid.y;

    // This token's TOPK intermediates, read straight from device memory -- see the
    // gate/up kernel. Staging them would be TOPK times larger and cap the occupancy.
    device const float4 * xs4 = (device const float4 *)(inter + ulong(m) * (RPT * KDIM));

    // NR0 consecutive output rows per SIMD-group: each lane reuses its activation words
    // across NR0 rows, so a larger NR0 cuts the activation reads by NR0.
    const uint h0 = (tgid.x * NSG + sgitg) * NR0;

    // One lane-private accumulator per row across all TOPK routes: the route's global scale
    // and router weight are lane-invariant, so one simd_sum per row finishes the sum.
    float total[NR0];
    #pragma unroll
    for (uint j = 0; j < NR0; ++j) { total[j] = 0.0f; }

    for (uint r = 0; r < RPT; ++r) {
#if SHARED
        const bool sh = (r == TOPK);
        const long e  = sh ? 0L : long(topk_ids[m * TOPK + r]);
        const float rw = sh ? 1.0f / (1.0f + exp(-float(sgate[m]))) : topk_w[m * TOPK + r];
        device const uint  * pk = sh ? s_packed : packed;
        device const uchar * ps = sh ? s_scale  : scale;
        device const half  * pg = sh ? s_glob   : glob;
#else
        const long e  = long(topk_ids[m * TOPK + r]);
        const float rw = topk_w[m * TOPK + r];
        device const uint  * pk = packed;
        device const uchar * ps = scale;
        device const half  * pg = glob;
#endif

        float acc[NR0];
        #pragma unroll
        for (uint j = 0; j < NR0; ++j) { acc[j] = 0.0f; }

        // Unrolled 2x so both loads of a row and route issue together.
        #pragma unroll 2
        for (uint w = tiisg; w < W; w += 32u) {
            // `off`, not `base`: the HAS_BASE epilogue below binds a buffer of that name.
            const uint off = (r * KDIM) / 4u + 2u * w;
            const float4 a0 = xs4[off];
            const float4 a1 = xs4[off + 1u];
            const uint blk = w >> 1;
            #pragma unroll
            for (uint j = 0; j < NR0; ++j) {
                // Clamp rather than branch: a tail threadgroup re-reads the last row
                // and drops the store, so the loop stays uniform.
                const long row = e * long(HDIM) + long(min(h0 + j, uint(HDIM) - 1u));
                const uint p = pk[row * long(W) + long(w)];
                const float s = e4m3_u8_to_f32(ps[row * long(NB) + long(blk)]);
                acc[j] += s * (dot(e2m1x4(p), a0) + dot(e2m1x4(p >> 16), a1));
            }
        }
        #pragma unroll
        for (uint j = 0; j < NR0; ++j) {
            const long row = e * long(HDIM) + long(min(h0 + j, uint(HDIM) - 1u));
            total[j] += acc[j] * float(pg[row]) * rw;
        }
    }

    #pragma unroll
    for (uint j = 0; j < NR0; ++j) {
        float t = simd_sum(total[j]);
        if (tiisg == 0 && h0 + j < uint(HDIM)) {
#if HAS_BASE
            // out = base + sum_k w_k * down_k(h_k) in one launch.
            t += float(base[ulong(m) * HDIM + h0 + j]);
#endif
            out[ulong(m) * HDIM + h0 + j] = OUT_T(t);
        }
    }
}
"""


@functools.lru_cache(maxsize=None)
def specialize(source: str, **defines: object) -> str:
    """Prepend ``#define`` lines."""
    header = "".join(f"#define {k} {v}\n" for k, v in sorted(defines.items()))
    return header + source


# Dense NVFP4 GEMV: out[m, n] = sum_k x[m, k] * weight[n, k], no routing indirection.
GEMV_SRC = _PRELUDE + r"""
kernel void nvfp4_gemv(
    device       OUT_T * out    [[buffer(0)]],
    device const X_T   * x      [[buffer(1)]],
    device const uint  * packed [[buffer(2)]],
    device const uchar * scale  [[buffer(3)]],
    device const half  * glob   [[buffer(4)]],
#if HAS_GATE
    device const X_T   * gate   [[buffer(5)]],   // [M] pre-sigmoid, one per token
#endif
#if SIDE_GATE
    device const GW_T  * gw     [[buffer(6)]],   // [KDIM] one extra unquantized weight row
    device       X_T   * gout   [[buffer(7)]],   // [M] its dot with the activation
#endif
    uint2 tgid   [[threadgroup_position_in_grid]],
    uint  tiitg  [[thread_index_in_threadgroup]],
    uint  sgitg  [[simdgroup_index_in_threadgroup]],
    uint  tiisg  [[thread_index_in_simdgroup]])
{
    const uint W  = KDIM / 8u;    // 32-bit words per row: 8 e2m1 codes each
    const uint NB = KDIM / 16u;   // e4m3 block scales per row
    const uint m  = tgid.y;       // one token per threadgroup row

    // The token's activations, staged once per threadgroup and shared by its NSG SIMD-groups
    // and the NR0 rows each of them owns.
    threadgroup float4 xs4[KDIM / 4];
    threadgroup float * xs = (threadgroup float *)xs4;
    for (uint t = tiitg; t < KDIM; t += NSG * 32u) {
        // Widened here, so a bf16 residual stream needs no cast launch before the call.
        xs[t] = float(x[ulong(m) * KDIM + t]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#if SIDE_GATE
    // One more output, off an unquantized [KDIM] row, computed by threadgroup 0 alone: every
    // thread takes a slice of K, one simd_sum per SIMD-group, one pass over the NSG partials.
    if (tgid.x == 0) {
        threadgroup float gpart[NSG];
        float gp = 0.0f;
        for (uint t = tiitg; t < KDIM; t += NSG * 32u) { gp += xs[t] * float(gw[t]); }
        gp = simd_sum(gp);
        if (tiisg == 0) { gpart[sgitg] = gp; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg == 0) {
            float g = 0.0f;
            for (uint s = 0; s < NSG; ++s) { g += gpart[s]; }
            gout[m] = X_T(g);
        }
    }
#endif

    // n0 indexes OUTPUT columns; with SWIGLU column i owns weight rows i (gate) and ODIM+i (up).
    const uint n0 = (tgid.x * NSG + sgitg) * NR0;

    long row[NR0];
    float acc[NR0];
#if SWIGLU
    float up[NR0];
#endif
    #pragma unroll
    for (uint j = 0; j < NR0; ++j) {
        // Clamp rather than branch: a tail threadgroup re-reads the last row and
        // drops the store, so the K loop and the simd_sum below stay uniform.
        row[j] = long(min(n0 + j, uint(ODIM) - 1u));
        acc[j] = 0.0f;
#if SWIGLU
        up[j] = 0.0f;
#endif
    }

    for (uint w = tiisg; w < W; w += 32u) {
        // One word is 8 e2m1 codes == elements 8w..8w+7 == two float4 of activations.
        const float4 a0 = xs4[2u * w];
        const float4 a1 = xs4[2u * w + 1u];
        const uint blk = w >> 1;   // 16 k-values per block == 2 words
        #pragma unroll
        for (uint j = 0; j < NR0; ++j) {
            const uint p = packed[row[j] * long(W) + long(w)];
            const float s = e4m3_u8_to_f32(scale[row[j] * long(NB) + long(blk)]);
            acc[j] += s * (dot(e2m1x4(p), a0) + dot(e2m1x4(p >> 16), a1));
#if SWIGLU
            const long ur = row[j] + long(ODIM);
            const uint pu = packed[ur * long(W) + long(w)];
            const float su = e4m3_u8_to_f32(scale[ur * long(NB) + long(blk)]);
            up[j] += su * (dot(e2m1x4(pu), a0) + dot(e2m1x4(pu >> 16), a1));
#endif
        }
    }

#if HAS_GATE
    const float gm = 1.0f / (1.0f + exp(-float(gate[m])));
#endif

    // Deferred reduction: one simd_sum per row, not one per K iteration.
    #pragma unroll
    for (uint j = 0; j < NR0; ++j) {
        float v = simd_sum(acc[j]) * float(glob[row[j]]);
#if SWIGLU
        // silu(gate) * up, matching kernel/triton/activation.py:6.
        const float u = simd_sum(up[j]) * float(glob[row[j] + long(ODIM)]);
        v = (v / (1.0f + exp(-v))) * u;
#endif
#if HAS_GATE
        v *= gm;
#endif
        if (tiisg == 0 && n0 + j < uint(ODIM)) {
            out[ulong(m) * ODIM + n0 + j] = OUT_T(v);
        }
    }
}
"""

# Prefill: the same two expert GEMMs, grouped by expert.
GATE_UP_GROUPED_SRC = _PRELUDE + r"""
kernel void nvfp4_moe_gate_up_silu_grouped(
    device       float * out        [[buffer(0)]],   // [R, INTER] fp32, natural route order
    device const float * x          [[buffer(1)]],   // [M, KDIM] fp32
    device const uint  * packed     [[buffer(2)]],   // [E, 2I, KDIM/8]
    device const uchar * scale      [[buffer(3)]],   // [E, 2I, KDIM/16]
    device const half  * glob       [[buffer(4)]],   // [E, 2I]
    device const int   * route_row  [[buffer(5)]],   // [R] expert order -> token row
    device const int   * route_id   [[buffer(6)]],   // [R] expert order -> natural route
    device const int   * offsets    [[buffer(7)]],   // [E+1]
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const uint W  = KDIM / 8u;
    const uint NB = KDIM / 16u;

    const uint e  = tgid.y;
    const int  lo = offsets[e];
    const int  hi = offsets[e + 1];
    if (lo >= hi) { return; }          // expert not routed to: no work at all

    const uint i0 = (tgid.x * NSG + sgitg) * NR0;

    // The 2*NR0 weight rows this SIMD-group owns: gate row e*2I+i, up row e*2I+I+i.
    long grow[NR0];
    #pragma unroll
    for (uint j = 0; j < NR0; ++j) {
        grow[j] = long(e) * long(2 * INTER) + long(min(i0 + j, uint(INTER) - 1u));
    }

    for (int c = lo; c < hi; c += NM) {
        // NM of this expert's routes; clamping onto the run's last route keeps the loop uniform.
        device const float4 * xr[NM];
        #pragma unroll
        for (uint r = 0; r < NM; ++r) {
            const int j = min(c + int(r), hi - 1);
            xr[r] = (device const float4 *)(x + ulong(route_row[j]) * KDIM);
        }

        float ag[NM][NR0];
        float au[NM][NR0];
        #pragma unroll
        for (uint r = 0; r < NM; ++r) {
            #pragma unroll
            for (uint j = 0; j < NR0; ++j) { ag[r][j] = 0.0f; au[r][j] = 0.0f; }
        }

        for (uint w = tiisg; w < W; w += 32u) {
            const uint blk = w >> 1;
            // The grouped part: one load of the weight word and its block scale,
            // then NM multiply-accumulates against it.
            float4 cg[NR0][2];
            float4 cu[NR0][2];
            float  sg[NR0];
            float  su[NR0];
            #pragma unroll
            for (uint j = 0; j < NR0; ++j) {
                const long gr = grow[j];
                const long ur = gr + long(INTER);
                const uint pg = packed[gr * long(W) + long(w)];
                const uint pu = packed[ur * long(W) + long(w)];
                cg[j][0] = e2m1x4(pg);      cg[j][1] = e2m1x4(pg >> 16);
                cu[j][0] = e2m1x4(pu);      cu[j][1] = e2m1x4(pu >> 16);
                sg[j] = e4m3_u8_to_f32(scale[gr * long(NB) + long(blk)]);
                su[j] = e4m3_u8_to_f32(scale[ur * long(NB) + long(blk)]);
            }
            #pragma unroll
            for (uint r = 0; r < NM; ++r) {
                const float4 a0 = xr[r][2u * w];
                const float4 a1 = xr[r][2u * w + 1u];
                #pragma unroll
                for (uint j = 0; j < NR0; ++j) {
                    ag[r][j] += sg[j] * (dot(cg[j][0], a0) + dot(cg[j][1], a1));
                    au[r][j] += su[j] * (dot(cu[j][0], a0) + dot(cu[j][1], a1));
                }
            }
        }

        #pragma unroll
        for (uint r = 0; r < NM; ++r) {
            if (c + int(r) >= hi) { break; }   // uniform: c, r, hi are lane-invariant
            const uint rid = uint(route_id[c + int(r)]);
            #pragma unroll
            for (uint j = 0; j < NR0; ++j) {
                const float g = simd_sum(ag[r][j]) * float(glob[grow[j]]);
                const float u = simd_sum(au[r][j]) * float(glob[grow[j] + long(INTER)]);
                if (tiisg == 0 && i0 + j < uint(INTER)) {
                    out[ulong(rid) * INTER + i0 + j] = (g / (1.0f + exp(-g))) * u;
                }
            }
        }
    }
}
"""

DOWN_GROUPED_SRC = _PRELUDE + r"""
kernel void nvfp4_moe_down_grouped(
    device       float * out        [[buffer(0)]],   // [R, HDIM] fp32, natural route order
    device const float * inter      [[buffer(1)]],   // [R, KDIM] fp32, natural route order
    device const uint  * packed     [[buffer(2)]],   // [E, HDIM, KDIM/8]
    device const uchar * scale      [[buffer(3)]],   // [E, HDIM, KDIM/16]
    device const half  * glob       [[buffer(4)]],   // [E, HDIM]
    device const int   * route_id   [[buffer(5)]],   // [R] expert order -> natural route
    device const float * route_w    [[buffer(6)]],   // [R] router weight, natural order
    device const int   * offsets    [[buffer(7)]],   // [E+1]
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const uint W  = KDIM / 8u;
    const uint NB = KDIM / 16u;

    const uint e  = tgid.y;
    const int  lo = offsets[e];
    const int  hi = offsets[e + 1];
    if (lo >= hi) { return; }

    const uint h0 = (tgid.x * NSG + sgitg) * NR0;

    long row[NR0];
    #pragma unroll
    for (uint j = 0; j < NR0; ++j) {
        row[j] = long(e) * long(HDIM) + long(min(h0 + j, uint(HDIM) - 1u));
    }

    for (int c = lo; c < hi; c += NM) {
        uint rid[NM];
        device const float4 * ir[NM];
        #pragma unroll
        for (uint r = 0; r < NM; ++r) {
            const int j = min(c + int(r), hi - 1);
            rid[r] = uint(route_id[j]);
            ir[r]  = (device const float4 *)(inter + ulong(rid[r]) * KDIM);
        }

        float acc[NM][NR0];
        #pragma unroll
        for (uint r = 0; r < NM; ++r) {
            #pragma unroll
            for (uint j = 0; j < NR0; ++j) { acc[r][j] = 0.0f; }
        }

        for (uint w = tiisg; w < W; w += 32u) {
            const uint blk = w >> 1;
            float4 cw[NR0][2];
            float  s[NR0];
            #pragma unroll
            for (uint j = 0; j < NR0; ++j) {
                const uint p = packed[row[j] * long(W) + long(w)];
                cw[j][0] = e2m1x4(p);   cw[j][1] = e2m1x4(p >> 16);
                s[j] = e4m3_u8_to_f32(scale[row[j] * long(NB) + long(blk)]);
            }
            #pragma unroll
            for (uint r = 0; r < NM; ++r) {
                const float4 a0 = ir[r][2u * w];
                const float4 a1 = ir[r][2u * w + 1u];
                #pragma unroll
                for (uint j = 0; j < NR0; ++j) {
                    acc[r][j] += s[j] * (dot(cw[j][0], a0) + dot(cw[j][1], a1));
                }
            }
        }

        #pragma unroll
        for (uint r = 0; r < NM; ++r) {
            if (c + int(r) >= hi) { break; }
            const float rw = route_w[rid[r]];
            #pragma unroll
            for (uint j = 0; j < NR0; ++j) {
                const float v = simd_sum(acc[r][j]) * float(glob[row[j]]) * rw;
                if (tiisg == 0 && h0 + j < uint(HDIM)) {
                    out[ulong(rid[r]) * HDIM + h0 + j] = v;
                }
            }
        }
    }
}
"""

# Prefill, large M: a tiled grouped GEMM per expert, after mul_mm.metal. A threadgroup owns NR0
# columns of one expert and walks its routes NR1 at a time, so weights are read routes/NR1 times.
MOE_TILED_SRC = _PRELUDE + r"""
#include <metal_simdgroup_matrix>

#define NR0 64                   // weight rows (output columns) per threadgroup
#define NR1 32                   // routes per chunk
#define NK  32                   // K elements staged per step
#define NPL (GATE_UP + 1)        // weight planes: gate and up together, or down alone
#define WSTRIDE (NDIM * NPL)     // weight rows per expert
#define WORDS (KDIM / 8)
#define BLOCKS (KDIM / 16)
#define NOUT (NR1 * NR0 / 128)   // output elements each thread carries in the epilogue

// Staged activation scalar: bfloat for gate/up, 2.8x faster than float and exact on the widened
// bf16 residual; the down GEMM has half the accumulators and a real float32 input, so float.
#if GATE_UP
#define AT  bfloat
#define AT4 bfloat4
#else
#define AT  float
#define AT4 float4
#endif

kernel void nvfp4_moe_tiled(
    device       float * out       [[buffer(0)]],   // [R, NDIM] fp32, natural route order
    device const float * act       [[buffer(1)]],   // [., KDIM] fp32
    device const uint  * packed    [[buffer(2)]],   // [E, NPL*NDIM, KDIM/8]
    device const uchar * scale     [[buffer(3)]],   // [E, NPL*NDIM, KDIM/16]
    device const half  * glob      [[buffer(4)]],   // [E, NPL*NDIM]
    device const int   * a_row     [[buffer(5)]],   // [R] expert order -> activation row
    device const int   * o_row     [[buffer(6)]],   // [R] expert order -> natural route
    device const int   * offsets   [[buffer(7)]],   // [E+1]
    device const float * route_w   [[buffer(8)]],   // [R] router weight, natural order
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  tiitg [[thread_index_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint e  = tgid.y;
    const int  lo = offsets[e];
    const int  hi = offsets[e + 1];
    if (lo >= hi) { return; }          // expert not routed to: no work at all

    const uint n0 = tgid.x * NR0;

    // Both tiles sit in 8x8 blocks that simdgroup_load reads at stride 8, each declared at its
    // widest reader's type so the alias below is aligned; sc is exactly the epilogue's NR1 x NR0.
    threadgroup float sc[NR1 * NR0];
    threadgroup AT4   sa4[NR1 * NK / 4];
    threadgroup half * sw = (threadgroup half *) sc;
    threadgroup AT   * sa = (threadgroup AT *) sa4;

    // Four words of eight e2m1 codes cover one weight row's NK slice, so 128 threads stage 32
    // rows per pass; the activation tile is one thread per (route, k-block).
    const uint kb = tiitg % (NK / 8u);
    const uint rw = tiitg / (NK / 8u);

    for (int c = lo; c < hi; c += NR1) {
        simdgroup_float8x8 mc[NPL][8];
        for (uint p = 0; p < NPL; ++p) {
            for (uint i = 0; i < 8; ++i) { mc[p][i] = simdgroup_float8x8(0.0f); }
        }

        for (uint k0 = 0; k0 < uint(KDIM); k0 += NK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint w = k0 / 8u + kb;
            for (uint p = 0; p < NPL; ++p) {
                for (uint nl = rw; nl < NR0; nl += 128u / (NK / 8u)) {
                    // Clamp rather than branch: a tail tile re-reads the last row and drops
                    // the store, so every lane stays on one path through the K loop.
                    const long row = long(e) * WSTRIDE + long(p) * NDIM
                                   + long(min(n0 + nl, uint(NDIM) - 1u));
                    const uint pk = (w < WORDS) ? packed[row * WORDS + long(w)] : 0u;
                    // An e2m1 code times an e4m3 block scale needs five mantissa bits, so
                    // half holds it exactly; the per-row global scale stays in the epilogue.
                    const float s = (w < WORDS)
                        ? e4m3_u8_to_f32(scale[row * BLOCKS + long(w >> 1)]) : 0.0f;
                    const float4 c0 = e2m1x4(pk) * s;
                    const float4 c1 = e2m1x4(pk >> 16) * s;
                    threadgroup half * d = sw + p * (NR0 * NK)
                                         + 64u * (8u * kb + (nl >> 3)) + (nl & 7u);
                    d[0]  = half(c0.x);  d[8]  = half(c0.y);
                    d[16] = half(c0.z);  d[24] = half(c0.w);
                    d[32] = half(c1.x);  d[40] = half(c1.y);
                    d[48] = half(c1.z);  d[56] = half(c1.w);
                }
            }

            const uint ak = k0 + 8u * kb;
            threadgroup AT4 * ad = (threadgroup AT4 *)
                (sa + 64u * (4u * kb + (rw >> 3)) + 8u * (rw & 7u));
            if (ak < uint(KDIM)) {
                device const float4 * as4 = (device const float4 *)
                    (act + ulong(a_row[min(c + int(rw), hi - 1)]) * KDIM + ak);
                ad[0] = AT4(as4[0]);  ad[1] = AT4(as4[1]);
            } else {
                ad[0] = AT4(0);  ad[1] = AT4(0);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint ik = 0; ik < NK / 8u; ++ik) {
                simdgroup_matrix<AT, 8, 8> mb[2];
                threadgroup const AT * lb = sa + 128u * (sgitg >> 1) + ik * 256u;
                simdgroup_barrier(mem_flags::mem_none);
                for (uint i = 0; i < 2; ++i) { simdgroup_load(mb[i], lb + 64u * i, 8); }
                for (uint p = 0; p < NPL; ++p) {
                    simdgroup_matrix<half, 8, 8> ma[4];
                    threadgroup const half * la = sw + p * (NR0 * NK)
                                                + 256u * (sgitg & 1u) + ik * 512u;
                    simdgroup_barrier(mem_flags::mem_none);
                    for (uint i = 0; i < 4; ++i) { simdgroup_load(ma[i], la + 64u * i, 8); }
                    simdgroup_barrier(mem_flags::mem_none);
                    for (uint i = 0; i < 8; ++i) {
                        simdgroup_multiply_accumulate(mc[p][i], mb[i / 4], ma[i % 4], mc[p][i]);
                    }
                }
            }
        }

        // The accumulators land in the weight tile's bytes as [NR1][NR0] floats; the store is
        // a scatter (expert order -> natural route), so plain threads finish it.
        threadgroup float * ts = sc + 32u * (sgitg & 1u) + 1024u * (sgitg >> 1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; ++i) {
            simdgroup_store(mc[0][i], ts + 8u * (i % 4) + 8u * NR0 * (i / 4), NR0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if GATE_UP
        // Gate and up share the one tile, so the gate half moves to registers first.
        float gv[NOUT];
        #pragma unroll
        for (uint j = 0; j < NOUT; ++j) { gv[j] = sc[tiitg + j * 128u]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; ++i) {
            simdgroup_store(mc[1][i], ts + 8u * (i % 4) + 8u * NR0 * (i / 4), NR0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
        #pragma unroll
        for (uint j = 0; j < NOUT; ++j) {
            const uint i  = tiitg + j * 128u;
            const uint ml = i / NR0;
            const uint nl = i % NR0;
            if (c + int(ml) >= hi || n0 + nl >= uint(NDIM)) { continue; }
            const uint rid = uint(o_row[c + int(ml)]);
            const long grow = long(e) * WSTRIDE + long(n0 + nl);
#if GATE_UP
            const float g = gv[j] * float(glob[grow]);
            const float u = sc[i] * float(glob[grow + NDIM]);
            // silu(gate) * up, matching kernel/triton/activation.py:6 with d = NDIM.
            out[ulong(rid) * NDIM + n0 + nl] = (g / (1.0f + exp(-g))) * u;
#else
            out[ulong(rid) * NDIM + n0 + nl] = sc[i] * float(glob[grow]) * route_w[rid];
#endif
        }
    }
}
"""
