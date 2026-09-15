"""Attention backend for Apple GPUs (MPS)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import Batch

from .base import AttentionSpec
from .triton import TritonAttentionBackend

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


class MetalAttentionBackend(TritonAttentionBackend):
    # forward takes gate= and applies the output gate inside the decode kernel's epilogue.
    fuses_output_gate = True

    # paged_attention_mps loops over requests on the host and needs the indptrs as
    # Python lists; the parent publishes them once per batch instead of per layer.
    needs_host_indptr = True

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from freetoken.kernel.metal import is_available

        if not is_available():
            raise RuntimeError("the metal attention backend needs a torch build with MPS")

    def prepare_for_capture(self, batch: Batch) -> None:
        super().prepare_for_capture(batch)
        # Host lists so a recording forward (mps_tape.py) never reads them off device.
        bs = batch.size
        batch.attn_metadata.cu_seqlens_q_host = list(range(bs + 1))
        batch.attn_metadata.indptr_host = list(range(bs + 1))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch,
        attn_spec: AttentionSpec | None = None, gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """gate [nnz, num_q_heads*head_dim] (pre-sigmoid) scales the output, in-kernel
        or via sigmoid_gate_mul -- same result either way."""
        from freetoken.kernel.metal.attention import paged_attention_mps

        if attn_spec is not None and (
            attn_spec.sliding_window is not None or attn_spec.sinks is not None
        ):
            raise NotImplementedError(
                "metal attention backend: no sliding window and no attention sinks"
            )

        metadata = batch.attn_metadata
        # k/v are None when the caller already wrote this batch's rows (fused qkv epilogue).
        if k is not None:
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads, head_dim = k_raw.shape[-2], k_raw.shape[-1]

        scale = head_dim ** -0.5
        if attn_spec is not None and attn_spec.sm_scale is not None:
            scale = attn_spec.sm_scale

        return paged_attention_mps(
            q=q, k_cache=k_raw.view(-1, kv_heads, head_dim),
            v_cache=v_raw.view(-1, kv_heads, head_dim), indptr=metadata.indptr,
            indices=metadata.indices, cu_seqlens_q=metadata.cu_seqlens_q_gpu, sm_scale=scale,
            kv_ptr=metadata.indptr_host, q_ptr=metadata.cu_seqlens_q_host, gate=gate,
        )


__all__ = ["MetalAttentionBackend"]
