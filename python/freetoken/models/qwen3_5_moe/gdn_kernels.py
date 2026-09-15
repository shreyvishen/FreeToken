from __future__ import annotations

import torch


def gdn_prefill_chunk_fla(
    q: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, total, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, total, num_v_heads, head_v_dim] bf16
    g: torch.Tensor,        # [1, total, num_v_heads] log-decay (<=0), fp32
    beta: torch.Tensor,     # [1, total, num_v_heads] fp32
    *,
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,       # [num_seqs] slot id per sequence
    cu_seqlens: torch.Tensor,    # [num_seqs+1] int64
    scale: float,
    return_h: bool = False,
    cu_seqlens_host: list | None = None,
) -> torch.Tensor:
    """Chunked gated-delta-rule prefill via the vendored fla kernel."""
    if q.device.type == "mps":
        from freetoken.kernel.metal.gdn import gdn_prefill_metal

        if return_h:
            raise NotImplementedError(
                "metal GDN prefill has no per-chunk h yet; the hybrid-radix track "
                "checkpoint (--cache-type hybrid_radix) is CUDA-only on this path"
            )
        return gdn_prefill_metal(
            q, k, v, g, beta, state_source=state_source, indices=indices, cu_seqlens=cu_seqlens,
            scale=scale, cu_seqlens_host=cu_seqlens_host,
        )
    from freetoken.kernel.fla import chunk_gated_delta_rule

    o, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=state_source, initial_state_indices=indices.to(torch.int32),
        cu_seqlens=cu_seqlens.to(torch.int64), head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    if return_h:
        return o[0], h  # h: [1, NT_total, num_v_heads, head_v_dim, head_k_dim]
    return o[0]  # [total, num_v_heads, head_v_dim]


def gdn_decode_fla(
    q: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16 (NOT GQA-expanded)
    k: torch.Tensor,        # [1, B, num_k_heads, head_k_dim] bf16
    v: torch.Tensor,        # [1, B, num_v_heads, head_v_dim] bf16
    a: torch.Tensor,        # [B, num_v_heads] raw
    b: torch.Tensor,        # [B, num_v_heads] raw
    *,
    A_log: torch.Tensor,        # [num_v_heads]
    dt_bias: torch.Tensor,      # [num_v_heads]
    state_source: torch.Tensor,  # [num_slots, num_v_heads, head_k_dim, head_v_dim] fp32 (in place)
    indices: torch.Tensor,      # [B] int32 slot id per request
    cu_seqlens: torch.Tensor,   # [B+1] query indptr (arange) from FLAMetadata
    scale: float,
) -> torch.Tensor:
    """Fused sigmoid-gating gated-delta-rule decode: gating, in-kernel l2norm, recurrent
    update and state read/write-by-index (state_source[indices]) in one kernel."""
    if q.device.type == "mps":
        from freetoken.kernel.metal.gdn import gdn_decode_metal

        return gdn_decode_metal(
            q, k, v, a, b, A_log=A_log, dt_bias=dt_bias,
            state_source=state_source, indices=indices, scale=scale,
        )
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias,  # already fp32 (stored fp32)
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,  # already int32 (built int32 in the scheduler)
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
    )
    # kernel returns o = [NK, *v.shape] then squeeze(NK) -> [1, B, num_v, V].
    # o[0] -> [B, num_v, V] (all B decode tokens; o[0,0] would drop B>1).
    return o[0]


def gdn_decode_fused(
    mixed: torch.Tensor, z: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, A_log: torch.Tensor,
    dt_bias: torch.Tensor, norm_weight: torch.Tensor, norm_eps: float, state_source: torch.Tensor,
    indices: torch.Tensor, scale: float, num_k_heads: int, head_k_dim: int,
):
    """The whole GDN decode step after the conv (mixed: [B, conv_dim] q|k|v), in one Metal
    launch: q/k l2 norms, gating, recurrent update, gated output RMSNorm."""
    if mixed.device.type != "mps":
        return None
    from freetoken.kernel.metal.gdn import fused_decode_supports, gdn_decode_fused_metal

    hv, dv = state_source.shape[1], state_source.shape[3]
    if not fused_decode_supports(
        mixed, z, a, b, A_log, dt_bias, norm_weight, state_source,
        num_k_heads, hv, head_k_dim, dv,
    ):
        return None
    return gdn_decode_fused_metal(
        mixed, z, a, b, A_log=A_log, dt_bias=dt_bias, norm_weight=norm_weight,
        norm_eps=norm_eps, state_source=state_source, indices=indices,
        scale=scale, num_k_heads=num_k_heads, head_k_dim=head_k_dim,
    )


__all__ = ["gdn_prefill_chunk_fla", "gdn_decode_fla", "gdn_decode_fused"]
