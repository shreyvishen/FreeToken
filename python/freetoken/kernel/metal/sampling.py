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


def _draw(probs: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draw."""
    cdf = probs.cumsum(-1)
    # u in (0, total], so the draw never lands on a zero-probability token at the row's start
    u = (1 - torch.rand(probs.shape[0], 1, device=probs.device)) * cdf[:, -1:].clamp_min(1e-20)
    return (cdf < u).sum(-1).clamp_max_(probs.shape[-1] - 1).to(torch.int32)


def _renorm(probs: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    kept = probs * keep.to(probs.dtype)
    return kept / kept.sum(-1, keepdim=True).clamp_min(1e-20)


def top_k_renorm_probs(probs: torch.Tensor, top_k) -> torch.Tensor:
    b, v = probs.shape
    k = _as_col(top_k, b, probs.device, torch.long).clamp(1, v)
    # every token tied with the k-th largest stays, as upstream's kernel and flashinfer keep it
    kth = probs.sort(dim=-1, descending=True).values.gather(-1, k - 1)
    return _renorm(probs, probs >= kth)


def top_p_renorm_probs(probs: torch.Tensor, top_p) -> torch.Tensor:
    b, v = probs.shape
    p = _as_col(top_p, b, probs.device, probs.dtype)
    s = probs.sort(dim=-1, descending=True).values
    # the threshold is the prob that brings the descending mass to p; its ties stay too
    last = ((s.cumsum(-1) - s) < p).sum(-1, keepdim=True) - 1
    return _renorm(probs, probs >= s.gather(-1, last))


def sampling_from_probs(probs):
    return _draw(probs)


def top_k_sampling_from_probs(probs, top_k):
    return _draw(top_k_renorm_probs(probs, top_k))


def top_p_sampling_from_probs(probs, top_p):
    return _draw(top_p_renorm_probs(probs, top_p))


def top_k_top_p_sampling_from_probs(probs, top_k, top_p):
    return _draw(top_p_renorm_probs(top_k_renorm_probs(probs, top_k), top_p))


def top_k_top_p_sample_from_logits(
    logits: torch.Tensor, temperature, top_k: torch.Tensor, top_p, k_max: int,
) -> torch.Tensor:
    """Draw with top-k, and optionally top-p, without sorting the full row: ``topk(k_max)``
    finds each row's thresholds, then the draw runs over the row with every token at or above
    them, so ties past the ``k_max`` window stay as they do in the probs entry points."""
    b, v = logits.shape
    x = logits.float()
    if temperature is not None:
        t = temperature if torch.is_tensor(temperature) else torch.tensor(temperature)
        x = x / t.to(x.device, x.dtype).reshape(-1, 1)
    values = torch.topk(x, min(k_max, v), dim=-1).values  # descending
    k = _as_col(top_k, b, x.device, torch.long).clamp(1, values.shape[-1])
    cut = values.gather(-1, k - 1)
    if top_p is not None:
        p = _as_col(top_p, b, x.device, x.dtype)
        pv = (values - torch.logsumexp(x.masked_fill(x < cut, float("-inf")), -1, keepdim=True)).exp()
        pv = pv * (values >= cut)
        last = ((pv.cumsum(-1) - pv) < p).sum(-1, keepdim=True) - 1
        cut = torch.maximum(cut, values.gather(-1, last))
    return _draw(torch.softmax(x.masked_fill(x < cut, float("-inf")), -1))


__all__ = [
    "sampling_from_probs", "softmax", "top_k_renorm_probs", "top_k_sampling_from_probs",
    "top_k_top_p_sample_from_logits", "top_k_top_p_sampling_from_probs", "top_p_renorm_probs",
    "top_p_sampling_from_probs",
]
