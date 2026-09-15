from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.kernel import backend
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    # The batch's largest per-row top_k, kept from the host list ``prepare`` already has.
    top_k_max: int | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # The host tensor is unreferenced the moment this returns, so the copy has to have
    # landed by then -- or the source has to be held until it has (backend.stage_h2d).
    host = torch.tensor(data, dtype=dtype, pin_memory=backend.PIN_MEMORY)
    return host.to(device, non_blocking=backend.stage_h2d(host))


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
    top_k_max: int | None = None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if logits.device.type == "mps":
        # Apple GPU: neither flashinfer nor triton runs here.
        import freetoken.kernel.metal.sampling as sampling

        # Selecting k candidates beats ranking the row once k is a small fraction of the
        # vocabulary: the probs path sorts V twice, this path runs one topk and then works
        # on k values.
        if top_k is not None and top_k_max is not None and top_k_max * 8 <= logits.shape[-1]:
            return sampling.top_k_top_p_sample_from_logits(
                logits, temperatures, top_k, top_p, top_k_max
            )
    elif is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, top_k_max=max(top_ks))

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with backend.nvtx_range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(
                logits.float(), args.temperatures, args.top_k, args.top_p, args.top_k_max
            )
