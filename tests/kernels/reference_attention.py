"""Triton-free causal paged GQA attention oracle, shared by the CUDA/Triton and the Metal attention tests -- one reference, not a fresh reimplementation per backend."""

from __future__ import annotations

import torch


def reference_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    q_to_req: torch.Tensor,
    q_positions: torch.Tensor,
    sm_scale: float,
    sliding_window: int | None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    outs = []
    group = q.shape[1] // k_cache.shape[1]
    for tok in range(q.shape[0]):
        req = int(q_to_req[tok].item())
        start = int(indptr[req].item())
        end = int(indptr[req + 1].item())
        slots = indices[start:end].to(torch.long)
        k = k_cache[slots].repeat_interleave(group, dim=1).transpose(0, 1)
        v = v_cache[slots].repeat_interleave(group, dim=1).transpose(0, 1)
        scores = torch.einsum("hd,hkd->hk", q[tok].float(), k.float()) * sm_scale
        key_pos = torch.arange(end - start, device=q.device)
        mask = key_pos <= q_positions[tok]
        if sliding_window is not None:
            mask = mask & (key_pos + sliding_window > q_positions[tok])
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        if sinks is None:
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,hkd->hd", probs, v.float())
        else:
            sink_logits = sinks.to(device=q.device, dtype=torch.float32)
            m = torch.maximum(scores.max(dim=-1).values, sink_logits)
            probs = torch.exp(scores - m[:, None])
            denom = probs.sum(dim=-1) + torch.exp(sink_logits - m)
            out = torch.einsum("hk,hkd->hd", probs, v.float()) / denom[:, None]
        outs.append(out.to(q.dtype))
    return torch.stack(outs, dim=0)


__all__ = ["reference_paged_attention"]
