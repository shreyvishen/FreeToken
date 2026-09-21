"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    return sum(t[0][0].numel() * t[0].element_size() for t in sources.values())


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
    kv_cap_pages: int | None = None,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = remaining // cache_per_page
    if kv_cap_pages is not None:
        num_pages = min(num_pages, kv_cap_pages)
    num_pages = max(num_pages, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int | None = None,
    kv_cap_pages: int | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
        kv_cap_pages=kv_cap_pages,
    )


# Buffers the hyper-connection chain holds at once, each one residual-wide: R itself, the
# normed copy and the gate inside ``GatedResidual.mix``, and ``combine``'s output.
_HC_LIVE_COPIES = 4


def extend_activation_bytes(config) -> int:
    """A FLOOR under the device bytes one Metal extend allocates outside every pool. Per row:
    the residual (times the hyper-connection live copies), the qkv buffer, the fp32 MoE staging
    and routing tables of ``kernel/metal/nvfp4.py``, and QSA selection; per forward: the GDN
    chunk snapshots of a snapshotting cache and the QSA score workspace. Smaller transients are
    left to the ``(1 - memory_ratio)`` that ``_startup_kv_budget`` withholds."""
    import torch

    from freetoken.attention import AttnType
    from freetoken.attention.metal import _LOGITS_WORKSPACE_BYTES
    from freetoken.kernel.fla.const import CHUNK_SIZE
    from freetoken.kvcache.linear_state_pool import ssm_state_dtype

    mc = config.model_config
    item = config.dtype.itemsize
    i32, i64, f32 = torch.int32.itemsize, torch.int64.itemsize, torch.float32.itemsize
    hidden = mc.hidden_size
    streams = mc.qwen4_args.hc_count * _HC_LIVE_COPIES if mc.qwen4_args is not None else 1
    per_row = (streams * hidden + (mc.num_qo_heads + 2 * mc.num_kv_heads) * mc.head_dim) * item
    if mc.is_moe:
        # flat_ids + order (int64) and route_id, route_row, route_w (4 B each), per route.
        route = 2 * i64 + 3 * f32
        per_row += mc.num_experts_per_tok * (
            (mc.moe_intermediate_size + hidden) * f32 + route
        )
    else:
        per_row += 2 * mc.intermediate_size * item
    total = config.max_forward_len * per_row

    group = mc.linear_attention_group()
    if group is not None and getattr(config, "cache_type", "radix") == "hybrid_radix":
        state = group.num_value_heads * group.key_head_dim * group.value_head_dim
        total += div_ceil(config.max_forward_len, CHUNK_SIZE) * state * ssm_state_dtype().itemsize

    qsa = next((s for s in mc.kv_cache_group_specs() if s.attn_type is AttnType.QSA), None)
    if qsa is not None and mc.qwen4_args is not None:
        args = mc.qwen4_args
        # topk_values f32 + topk_blocks i64 + blocks i32, the expanded ids, the roped queries.
        selection = (args.index_topk_blocks * (f32 + i64 + i32)
                     + (args.index_budget + qsa.index_ratio - 1) * i32
                     + args.index_n_heads * args.index_head_dim * item)
        total += config.max_forward_len * selection + _LOGITS_WORKSPACE_BYTES
    return total


# Unified memory only: the most host memory the disk tier leaves unspent.
MPS_MIN_FREE_BYTES = 3 * 1024**3 // 2


def mps_net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int,
    min_free_bytes: int = MPS_MIN_FREE_BYTES,
) -> int:
    """:func:`net_cache_budget_bytes` with the headroom capped at ``min_free_bytes``."""
    # Not int((1 - ratio) * baseline_free): that is a byte off the reserve
    # net_cache_budget_bytes keeps, and the small-box case must equal it exactly.
    headroom = min(baseline_free - int(memory_ratio * baseline_free), min_free_bytes)
    return baseline_free - headroom - weights_bytes - fixed_cache_size


def mps_disk_tier_plan(
    *, baseline_free: int, weights_bytes: int, memory_ratio: float, reserved_bytes: int,
    activation_bytes: int, arena_bytes: int, per_expert_bytes: int, row_bytes: "list[int]",
    cache_per_page: int, num_experts: int, total_experts: int, max_slots: int,
    kv_floor_pages: int, kv_cap_pages: int | None, state_sizes: "list[int]",
) -> tuple[int, int, int, int]:
    """Settle the disk tier's ``(slots, kv_pages, budget, heap_overhead)``: the activations and
    reserved pools leave the budget before the split, then the MPS heap gap is re-priced."""
    fixed = reserved_bytes + arena_bytes + activation_bytes
    overhead = 0
    for _ in range(4):
        budget = mps_net_cache_budget_bytes(
            memory_ratio, baseline_free, weights_bytes, fixed + overhead
        )
        size, pages, _overlap = plan_cache_budget(
            budget_bytes=budget, per_expert_bytes=per_expert_bytes,
            cache_per_page=cache_per_page, num_experts=num_experts,
            total_experts=total_experts, prefill_overlap=False,
            kv_reserve_pages=kv_floor_pages, max_slots=max_slots, kv_cap_pages=kv_cap_pages,
        )
        # Every device allocation the tier and its sibling pools make, in order: the slot
        # banks, the staging arena, the KV slab, the GDN state.
        sizes = [size * rb for rb in row_bytes]
        sizes += [arena_bytes, pages * cache_per_page, *state_sizes]
        settled = max(overhead, mps_driver_bytes(sizes) - sum(sizes))
        if settled == overhead:
            break
        overhead = settled
    return size, pages, budget, overhead


def resolve_kv_cap_pages(
    *, max_running_req: int, max_seq_len: int, kv_cap_tokens: int, page_size: int,
) -> int:
    """KV pages worth capping the plan at, in pages of ``page_size`` tokens: the smaller of
    what the request limits address (``max_running_req x max_seq_len``) and ``kv_cap_tokens``
    (0 or less -> off)."""
    tokens = max_running_req * max_seq_len
    if kv_cap_tokens > 0:
        tokens = min(tokens, kv_cap_tokens)
    return max(div_ceil(tokens, page_size), 1)


def resolve_kv_floor_pages(
    *, max_running_req: int, max_seq_len: int, max_seq_len_override: int | None,
    kv_reserve_tokens: int, page_size: int, kv_cap_pages: int,
) -> int:
    """KV pages the plan must keep before experts take the rest (unified memory only)."""
    reserve = div_ceil(kv_reserve_tokens, page_size)
    if max_seq_len_override is None:
        return reserve
    addressable = div_ceil(max_running_req * max_seq_len, page_size)
    return max(reserve, min(addressable, kv_cap_pages))


# Unified memory only: torch's MPS allocator suballocates from MTLHeaps created in four size
# classes (aten/src/ATen/mps/MPSAllocator.h), so the driver charges more than the byte count.
MPS_MAX_SMALL_ALLOC = 1 << 20
MPS_MIN_LARGE_ALLOC = 10 << 20
MPS_SMALL_HEAP = 8 << 20
MPS_LARGE_HEAP = 32 << 20
MPS_XLARGE_HEAP = 1 << 30
MPS_ROUND_LARGE = 2 << 20


def mps_driver_bytes(sizes: "list[int]") -> int:
    """Driver bytes a sequence of MPS allocations costs, heaps and all."""
    heaps: list[list[int]] = []  # [total, free] per heap, in creation order
    for n in sizes:
        for heap in heaps:
            if heap[1] >= n:
                heap[1] -= n
                break
        else:
            if n <= MPS_MAX_SMALL_ALLOC:
                total = MPS_SMALL_HEAP
            elif n < MPS_MIN_LARGE_ALLOC:
                total = MPS_LARGE_HEAP
            elif n < MPS_XLARGE_HEAP // 2:
                total = MPS_XLARGE_HEAP
            else:
                total = div_ceil(n, MPS_ROUND_LARGE) * MPS_ROUND_LARGE
            heaps.append([total, total - n])
    return sum(heap[0] for heap in heaps)
