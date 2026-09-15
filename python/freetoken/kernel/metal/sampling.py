"""Sampling on MPS: the entry points ``engine/sample.py`` calls, in torch ops, with the same
surface as ``freetoken.kernel.triton.sampling`` and flashinfer's."""

from __future__ import annotations

import torch


def softmax(logits: torch.Tensor, temperature=None, enable_pdl=None) -> torch.Tensor:
    """Temperature-scaled softmax. ``temperature`` is ``[B]`` or a scalar."""
    x = logits.float()
    if temperature is not None:
        t = temperature if torch.is_tensor(temperature) else torch.tensor(temperature)
        x = x / t.to(x.device, x.dtype).reshape(-1, 1)
    return torch.softmax(x, dim=-1)


def _as_col(t, b: int, device, dtype):
    if torch.is_tensor(t):
        return t.to(device=device, dtype=dtype).reshape(-1, 1).expand(b, 1)
    return torch.full((b, 1), t, device=device, dtype=dtype)


def _draw(probs: torch.Tensor, generator=None) -> torch.Tensor:
    """Inverse-CDF draw."""
    cdf = probs.cumsum(-1)
    u = torch.rand(probs.shape[0], 1, device=probs.device, generator=generator)
    u = u * cdf[:, -1:].clamp_min(1e-20)
    return (cdf < u).sum(-1).clamp_max_(probs.shape[-1] - 1).to(torch.int32)


def _renorm(probs: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    kept = probs * keep.to(probs.dtype)
    return kept / kept.sum(-1, keepdim=True).clamp_min(1e-20)


def top_k_renorm_probs(probs: torch.Tensor, top_k) -> torch.Tensor:
    b, v = probs.shape
    k = _as_col(top_k, b, probs.device, torch.long).clamp(1, v)
    order = probs.argsort(dim=-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(v, device=probs.device).expand(b, v))
    return _renorm(probs, rank < k)


def top_p_renorm_probs(probs: torch.Tensor, top_p) -> torch.Tensor:
    b, v = probs.shape
    p = _as_col(top_p, b, probs.device, probs.dtype)
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    cdf = sorted_probs.cumsum(-1)
    # keep every token up to and including the one that first reaches p
    keep_sorted = (cdf - sorted_probs) < p
    keep = torch.zeros_like(keep_sorted).scatter_(-1, order, keep_sorted)
    return _renorm(probs, keep)


def sampling_from_probs(probs, indices=None, deterministic=True, generator=None,
                        return_valid=False):
    out = _draw(probs, generator)
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_sampling_from_probs(probs, top_k, indices=None, deterministic=True,
                              generator=None, return_valid=False):
    out = _draw(top_k_renorm_probs(probs, top_k), generator)
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_p_sampling_from_probs(probs, top_p, indices=None, deterministic=True,
                              generator=None, return_valid=False):
    out = _draw(top_p_renorm_probs(probs, top_p), generator)
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_top_p_sampling_from_probs(probs, top_k, top_p, indices=None, deterministic=True,
                                    generator=None, return_valid=False):
    out = _draw(top_p_renorm_probs(top_k_renorm_probs(probs, top_k), top_p), generator)
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def top_k_top_p_sample_from_logits(
    logits: torch.Tensor, temperature, top_k: torch.Tensor, top_p, k_max: int,
) -> torch.Tensor:
    """Draw with top-k, and optionally top-p, without ever sorting the full row: the chain
    the probs entry points implement only compares the survivors against each other, and
    the renormalization after top-k cancels the full-vocabulary softmax denominator, so
    the same distribution comes out of ``topk(k_max)`` and the same steps on ``k_max``
    values."""
    b, v = logits.shape
    x = logits.float()
    if temperature is not None:
        t = temperature if torch.is_tensor(temperature) else torch.tensor(temperature)
        x = x / t.to(x.device, x.dtype).reshape(-1, 1)
    values, index = torch.topk(x, min(k_max, v), dim=-1)  # descending
    probs = torch.softmax(values, dim=-1)
    k = _as_col(top_k, b, probs.device, torch.long).clamp(1, values.shape[-1])
    rank = torch.arange(values.shape[-1], device=probs.device).expand(b, -1)
    probs = _renorm(probs, rank < k)
    if top_p is not None:
        p = _as_col(top_p, b, probs.device, probs.dtype)
        cdf = probs.cumsum(-1)
        probs = probs * ((cdf - probs) < p).to(probs.dtype)
    drawn = _draw(probs).to(torch.int64).unsqueeze(-1)
    return index.gather(-1, drawn).squeeze(-1).to(torch.int32)


__all__ = [
    "sampling_from_probs", "softmax", "top_k_renorm_probs", "top_k_sampling_from_probs",
    "top_k_top_p_sample_from_logits", "top_k_top_p_sampling_from_probs", "top_p_renorm_probs",
    "top_p_sampling_from_probs",
]
