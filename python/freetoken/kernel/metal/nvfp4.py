"""NVFP4 kernels for Apple GPUs, the Metal twin of the Triton NVFP4 path."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel.metal._shaders import (
    DEQUANT_SRC,
    DOWN_GROUPED_SRC,
    DOWN_SRC,
    GATE_UP_GROUPED_SRC,
    GATE_UP_SRC,
    GEMV_SRC,
    specialize,
)

from .shaders import MAX_TG_FLOATS, compile, msl_type, pick_tile

# E2M1 value table by 4-bit code. The GEMVs carry their own MSL copy; dequant takes this.
E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def as_e4m3_bytes(scale: torch.Tensor) -> torch.Tensor:
    """The scale banks are ``float8_e4m3fn`` on CUDA and MPS has no fp8 dtype, so the kernels
    take the raw bytes. Free: a reinterpret, not a copy."""
    return scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale


# Threadgroup tiles: (SIMD-groups per threadgroup, output rows per SIMD-group).
_TILES = ((16, 2), (16, 1), (8, 1), (4, 1), (2, 1), (1, 1))
_MIN_TGS = 32


def _tile(n_rows: int, groups: int = 1) -> tuple[int, int]:
    return pick_tile(_TILES, n_rows, _MIN_TGS, groups)


def dequant_nvfp4(
    packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor, slots: torch.Tensor, *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``weight[n, o, i] = E2M1[code] * e4m3(block_scale) * glob`` for the cache slots
    ``slots`` names: packed ``[S, OUT, IN//2]`` uint8, scale ``[S, OUT, IN//16]``, glob
    ``[S, OUT]`` fp16 -> ``[N, OUT, IN]``. Bit-exact against the float32 CPU reference."""
    scale = as_e4m3_bytes(scale)
    _, out_rows, in_packed = packed.shape
    num_blocks = scale.shape[2]
    n = slots.shape[0]
    out = torch.empty((n, out_rows, in_packed * 2), dtype=dtype, device=packed.device)

    lib = compile(specialize(DEQUANT_SRC, OUT_T=msl_type(dtype)))
    lib.dequant_nvfp4(
        out, packed, scale, glob, slots.to(torch.int32),
        torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device), out_rows, in_packed,
        num_blocks, threads=(in_packed, n * out_rows), group_size=(64, 1),
    )
    return out


def dequant_nvfp4_dense(
    packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor, *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a dense NVFP4 matrix ``[OUT, IN//2]`` to ``[OUT, IN]``: the shared expert."""
    slots = torch.zeros(1, dtype=torch.int32, device=packed.device)
    return dequant_nvfp4(
        packed.unsqueeze(0), scale.unsqueeze(0), glob.unsqueeze(0), slots, dtype=dtype
    )[0]


def nvfp4_gemv(
    x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor, *,
    swiglu: bool = False, gate: torch.Tensor | None = None,
    side_gate_weight: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
    tile: tuple[int, int] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``out[m, n] = sum_k x[m, k] * weight[n, k]`` straight off one NVFP4 bank: x ``[M,
    K]``, packed ``[N, K//2]`` uint8, scale ``[N, K//16]``, glob ``[N]`` fp16 -> ``[M,
    N]``, the activation read in its own dtype, accumulation float32, the epilogue writing
    ``out_dtype``."""
    x = x.contiguous()
    out_dtype = out_dtype or x.dtype
    m, k = x.shape
    n = packed.shape[0]
    if swiglu and n % 2:
        raise ValueError(f"swiglu needs an even row count ([gate | up]), got {n}")
    o = n // 2 if swiglu else n
    if side_gate_weight is not None and side_gate_weight.numel() != k:
        raise ValueError(
            f"side_gate_weight has {side_gate_weight.numel()} elements, expected K={k}"
        )

    # 8 for the word-wide loads, 16 because the bank carries one e4m3 scale per 16-element
    # block: a shorter last block has no scale and the kernel would read the next row's.
    if k % 16:
        raise ValueError(f"K must be a multiple of 16 (word loads and block scales), got {k}")
    if packed.shape[1] * 2 != k:
        raise ValueError(f"packed is {packed.shape[1] * 2} wide, x is {k}")
    if k > MAX_TG_FLOATS:
        raise ValueError(f"K={k} activation tile exceeds the 32 KiB threadgroup limit")

    if gate is not None and gate.numel() != m:
        raise ValueError(f"gate has {gate.numel()} elements, expected one per row ({m})")

    nsg, nr0 = tile or _tile(o, m)
    rows = nsg * nr0
    out = torch.empty((m, o), dtype=out_dtype, device=x.device)
    args = [out, x, packed.view(torch.int32), as_e4m3_bytes(scale), glob]
    # Buffer slots are fixed by position: 5 is the gate, 6 and 7 the side row and its
    # output, so a side row without a gate needs a placeholder at 5.
    if gate is not None:
        args.append(gate.reshape(m).to(x.dtype).contiguous())
    side = None
    if side_gate_weight is not None:
        if gate is None:
            args.append(out)
        side = torch.empty((m,), dtype=x.dtype, device=x.device)
        args += [side_gate_weight.reshape(k).contiguous(), side]
    compile(specialize(GEMV_SRC, KDIM=k, NDIM=n, ODIM=o, NSG=nsg, NR0=nr0,
                       X_T=msl_type(x.dtype), OUT_T=msl_type(out_dtype),
                       SWIGLU=int(swiglu), HAS_GATE=int(gate is not None),
                       SIDE_GATE=int(side_gate_weight is not None),
                       GW_T=msl_type(side_gate_weight.dtype) if side_gate_weight is not None
                       else "float")).nvfp4_gemv(
        *args,
        threads=(32 * nsg * -(-o // rows), m),
        group_size=(32 * nsg, 1),
    )
    return (out, side) if side_gate_weight is not None else out


def moe_decode_nvfp4(
    x: torch.Tensor, gate_up_packed: torch.Tensor, gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor, down_packed: torch.Tensor, down_scale: torch.Tensor,
    down_global: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, *,
    base: torch.Tensor | None = None, shared: tuple[torch.Tensor, ...] | None = None,
    out_dtype: torch.dtype | None = None, tile_gate_up: tuple[int, int] | None = None,
    tile_down: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Routed-expert MoE for decode: x ``[M, H]``, topk_weights/topk_ids ``[M, TOP_K]`` ->
    ``[M, H]`` in two kernels, a gate/up GEMV with SiLU-mul fused and a down GEMV that
    applies the router weights and sums the TOP_K routes."""
    x = x.contiguous()
    out_dtype = out_dtype or x.dtype
    m, h = x.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev = x.device

    if h % 16 or inter % 16:
        raise ValueError(
            f"H and I must be multiples of 16 (word loads and block scales), got {h}, {inter}"
        )

    ids = topk_ids.to(torch.int32).contiguous()
    weights = topk_weights.float().contiguous()
    # uint8 [.., K//2] as 32-bit words: one load per 8 e2m1 codes, adjacent lanes adjacent.
    gu_words = gate_up_packed.view(torch.int32)
    dn_words = down_packed.view(torch.int32)

    rpt = top_k + (1 if shared is not None else 0)
    gu_args = []
    dn_args = []
    if shared is not None:
        s_gu_p, s_gu_s, s_gu_g, s_dn_p, s_dn_s, s_dn_g, gate_w = shared
        if tuple(s_gu_p.shape) != (two_i, h // 2) or tuple(s_dn_p.shape) != (h, inter // 2):
            raise ValueError(
                f"shared expert banks are {tuple(s_gu_p.shape)} / {tuple(s_dn_p.shape)}, "
                f"expected {(two_i, h // 2)} / {(h, inter // 2)} (the routed shape)"
            )
        if gate_w.numel() != h or gate_w.dtype != x.dtype:
            raise ValueError(
                f"gate_weight must be [{h}] of {x.dtype}, got {tuple(gate_w.shape)} {gate_w.dtype}"
            )
        sgate = torch.empty((m,), dtype=x.dtype, device=dev)
        gu_args = [s_gu_p.view(torch.int32), as_e4m3_bytes(s_gu_s), s_gu_g,
                   gate_w.reshape(h).contiguous(), sgate]
        dn_args = [s_dn_p.view(torch.int32), as_e4m3_bytes(s_dn_s), s_dn_g, sgate.to(out_dtype)]

    inter_buf = torch.empty((m, rpt, inter), dtype=torch.float32, device=dev)
    nsg_i, nr0_i = tile_gate_up or _tile(inter, m * rpt)
    rows_i = nsg_i * nr0_i
    compile(specialize(GATE_UP_SRC, KDIM=h, INTER=inter, NSG=nsg_i, NR0=nr0_i,
                       X_T=msl_type(x.dtype), X_T4=msl_type(x.dtype) + "4",
                       SHARED=int(shared is not None))).nvfp4_moe_gate_up_silu(
        inter_buf, x, gu_words, as_e4m3_bytes(gate_up_scale), gate_up_global, ids,
        top_k, *gu_args,
        threads=(32 * nsg_i * -(-inter // rows_i), m * rpt),
        group_size=(32 * nsg_i, 1),
    )

    out = torch.empty((m, h), dtype=out_dtype, device=dev)
    nsg_h, nr0_h = tile_down or _tile(h, m)
    rows_h = nsg_h * nr0_h
    args = [out, inter_buf, dn_words, as_e4m3_bytes(down_scale), down_global, ids, weights]
    if base is not None:
        if base.shape != (m, h):
            raise ValueError(f"base is {tuple(base.shape)}, expected {(m, h)}")
        args.append(base.to(out_dtype).contiguous())
    elif shared is not None:
        args.append(out)   # positional; the kernel binds buffer 7 only under HAS_BASE
    compile(specialize(DOWN_SRC, KDIM=inter, HDIM=h, TOPK=top_k, NSG=nsg_h, NR0=nr0_h,
                       HAS_BASE=int(base is not None), SHARED=int(shared is not None),
                       OUT_T=msl_type(out_dtype))).nvfp4_moe_down(
        *args, *dn_args,
        threads=(32 * nsg_h * -(-h // rows_h), m),
        group_size=(32 * nsg_h, 1),
    )
    return out


def moe_prefill_nvfp4(
    x: torch.Tensor, gate_up_packed: torch.Tensor, gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor, down_packed: torch.Tensor, down_scale: torch.Tensor,
    down_global: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, *,
    dtype: torch.dtype = torch.bfloat16, expert_chunk: int = 16,
) -> torch.Tensor:
    """Prefill MoE: dequantize the active experts, ``expert_chunk`` at a time, and matmul
    each expert's token group."""
    m, h = x.shape
    top_k = topk_ids.shape[1]
    inter = gate_up_packed.shape[1] // 2
    out = torch.zeros((m, h), dtype=torch.float32, device=x.device)

    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    flat_w = topk_weights.reshape(-1).float()
    order = torch.argsort(flat_ids)
    sorted_ids = flat_ids[order]
    active, counts = torch.unique(sorted_ids, return_counts=True)
    # Each expert's routes are one contiguous run of `order` delimited by the cumulative
    # counts, so slicing by them syncs once here rather than once per expert.
    bounds = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).tolist()
    rows_all = order // top_k              # route -> token row
    xc = x.to(dtype)

    for start in range(0, active.numel(), expert_chunk):
        slots = active[start:start + expert_chunk].to(torch.int32)
        gu = dequant_nvfp4(gate_up_packed, gate_up_scale, gate_up_global, slots, dtype=dtype)
        dn = dequant_nvfp4(down_packed, down_scale, down_global, slots, dtype=dtype)
        for j in range(slots.numel()):
            lo, hi = bounds[start + j], bounds[start + j + 1]
            routes = order[lo:hi]
            rows = rows_all[lo:hi]
            gate_up = F.linear(xc.index_select(0, rows), gu[j])
            act = F.silu(gate_up[:, :inter]) * gate_up[:, inter:]
            y = F.linear(act, dn[j]).float() * flat_w[routes].unsqueeze(1)
            out.index_add_(0, rows, y)
    return out


# Rows per chunk for the grouped prefill kernels: NM rows need NM x NR0 accumulator pairs
# and NM activation pointers live across the whole K loop, and past 2 that block spills.
_MAX_GROUP_ROWS = 2


def moe_prefill_nvfp4_grouped(
    x: torch.Tensor, gate_up_packed: torch.Tensor, gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor, down_packed: torch.Tensor, down_scale: torch.Tensor,
    down_global: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, *,
    group_rows: int | None = None, tile_gate_up: tuple[int, int] | None = None,
    tile_down: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Grouped-by-expert prefill MoE: same contract and same bytes as
    :func:`moe_prefill_nvfp4`, in two custom launches and with no bf16 copy of any expert."""
    x = x.contiguous().float()
    m, h = x.shape
    top_k = topk_ids.shape[1]
    experts = gate_up_packed.shape[0]
    inter = gate_up_packed.shape[1] // 2
    routes = m * top_k
    dev = x.device

    if h % 16 or inter % 16:
        raise ValueError(
            f"H and I must be multiples of 16 (word loads and block scales), got {h}, {inter}"
        )

    # Routes in expert order: the lower bound of an expert id is its run start, and
    # offsets[E] is the route count.
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_ids)
    offsets = torch.searchsorted(
        flat_ids[order], torch.arange(experts + 1, device=dev)
    ).to(torch.int32)
    route_id = order.to(torch.int32)
    route_row = torch.div(order, top_k, rounding_mode="floor").to(torch.int32)
    route_w = topk_weights.reshape(-1).float().contiguous()

    # A chunk of NM routes loads each weight word once for NM accumulators, but a shorter chunk
    # pays for the rows it clamps away, so the mean run length is the break-even.
    avg = max(1, routes // max(experts, 1))
    nm = group_rows or min(_MAX_GROUP_ROWS, 1 << (avg.bit_length() - 1))

    inter_buf = torch.empty((routes, inter), dtype=torch.float32, device=dev)
    nsg_i, nr0_i = tile_gate_up or _tile(inter, experts)
    rows_i = nsg_i * nr0_i
    compile(specialize(GATE_UP_GROUPED_SRC, KDIM=h, INTER=inter, NSG=nsg_i, NR0=nr0_i,
                       NM=nm)).nvfp4_moe_gate_up_silu_grouped(
        inter_buf, x, gate_up_packed.view(torch.int32), as_e4m3_bytes(gate_up_scale),
        gate_up_global, route_row, route_id, offsets,
        threads=(32 * nsg_i * -(-inter // rows_i), experts),
        group_size=(32 * nsg_i, 1),
    )

    # [R, H] in route order, so the top_k reduction is a plain sum over a view --
    # deterministic, unlike the index_add_ the dequantize path needs.
    parts = torch.empty((routes, h), dtype=torch.float32, device=dev)
    nsg_h, nr0_h = tile_down or _tile(h, experts)
    rows_h = nsg_h * nr0_h
    compile(specialize(DOWN_GROUPED_SRC, KDIM=inter, HDIM=h, NSG=nsg_h, NR0=nr0_h,
                       NM=nm)).nvfp4_moe_down_grouped(
        parts, inter_buf, down_packed.view(torch.int32), as_e4m3_bytes(down_scale),
        down_global, route_id, route_w, offsets,
        threads=(32 * nsg_h * -(-h // rows_h), experts),
        group_size=(32 * nsg_h, 1),
    )
    return parts.view(m, top_k, h).sum(1)
