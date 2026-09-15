"""Shared float32 CPU oracles and the tolerance helper for the Metal kernel tests."""

from __future__ import annotations

import torch

from freetoken.kernel.metal import E2M1_VALUES

# (H, I) pairs. The second is not a multiple of the 32-lane SIMD width times the
# words-per-row stride, so the K-loop remainder and a partial lane count both run.
SHAPES = [(256, 64), (80, 48)]


def _close(a, b, atol, rtol):
    torch.testing.assert_close(a.cpu().float(), b.cpu().float(), atol=atol, rtol=rtol)


def banks(experts: int, out_rows: int, in_dim: int, seed: int):
    """One NVFP4 bank triple in the checkpoint-native layout (models/nvfp4_banks.py)."""
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (experts, out_rows, in_dim // 2), dtype=torch.uint8, generator=g)
    scale = torch.randn(experts, out_rows, in_dim // 16, generator=g) + 0.01
    scale = scale.to(torch.float8_e4m3fn).view(torch.uint8)
    glob = (torch.rand(experts, out_rows, generator=g) * 0.5 + 0.1).to(torch.float16)
    return packed, scale, glob


def dequant_ref(packed, scale, glob, slots):
    """float32 CPU oracle for dequant, sharing no arithmetic with the kernel's bit trick."""
    slots = slots.cpu().long()
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    rows = packed.cpu()[slots].long()
    codes = torch.stack([lut[rows & 0xF], lut[(rows >> 4) & 0xF]], dim=-1).flatten(-2)
    blocks = scale.cpu()[slots].view(torch.float8_e4m3fn).float()
    scales = blocks.repeat_interleave(16, dim=-1)
    return codes * scales * glob.cpu().float()[slots].unsqueeze(-1)


def moe_ref(x, gu, dn, topk_weights, topk_ids):
    """float32 CPU oracle for the whole MoE block: dequant, dense matmul, silu-mul, weighted sum."""
    m, h = x.shape
    inter = gu[0].shape[1] // 2
    out = torch.zeros(m, h, dtype=torch.float32)
    for t in range(m):
        ids = topk_ids[t].cpu().int()
        w_gu = dequant_ref(*gu, ids)
        w_dn = dequant_ref(*dn, ids)
        gate_up = w_gu @ x[t].cpu().float()
        act = torch.nn.functional.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        y = (w_dn @ act.unsqueeze(-1)).squeeze(-1)
        out[t] = (y * topk_weights[t].cpu().float().unsqueeze(1)).sum(0)
    return out


def routing(m: int, experts: int, top_k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=g)[:top_k] for _ in range(m)]).int()
    w = torch.rand(m, top_k, generator=g)
    return ids, w / w.sum(-1, keepdim=True)


def rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """Elementwise relative error floored by the output's own scale."""
    denom = want.abs().clamp_min(want.abs().max() * 1e-3)
    return ((got.cpu().float() - want).abs() / denom).max().item()


def rel_norm(got: torch.Tensor, want: torch.Tensor) -> float:
    """Error normalized by the output's largest entry (right metric for a bf16 path)."""
    return ((got.cpu().float() - want).abs().max() / want.abs().max()).item()
