"""Attention backend for Apple GPUs (MPS): dense causal GQA, plus QSA sparse attention whenever
the pool is a QSAKVCache. Decode runs ONE path at every context length, because the decode tape
freezes whichever branch ran at capture; prefill is eager and may route."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec
from .triton import TritonAttentionBackend, _int32_on

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

# qsa_sparse's cap on the fp32 block-score tile, so a long prefill scores in row chunks.
_LOGITS_WORKSPACE_BYTES = 128 << 20


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

        from freetoken.kvcache.qsa_pool import QSAKVCache

        self._qsa_slot: dict[int, int] | None = None
        if not isinstance(self.kvcache, QSAKVCache):
            return
        from freetoken.layers.rotary import get_rope

        from .qsa_sparse import QSASparseAttnBackend

        args = config.qwen4_args
        assert args is not None, "the metal QSA path needs ModelConfig.qwen4_args"
        group = QSASparseAttnBackend._qsa_group(config)
        rotary = group.rotary_config
        if rotary.mrope_section is not None:
            raise NotImplementedError(
                "metal QSA ropes the indexer at one scalar position; mrope is not implemented"
            )
        page_size = get_global_ctx().page_size
        self._qsa_ratio = self.kvcache.index_ratio
        assert page_size % self._qsa_ratio == 0, (
            f"QSA needs page_size ({page_size}) divisible by index_ratio ({self._qsa_ratio})"
        )
        self._qsa_slot = {lid: i for i, lid in enumerate(group.layer_ids)}
        self._qsa_topk = args.index_topk_blocks
        self._qsa_width = args.index_budget + self._qsa_ratio - 1
        self._qsa_columns = 0  # pinned by init_capture_graph, cleared by reset_capture
        # Its own get_rope instance: the indexer ropes at head_size 128, and the kernels read
        # the table directly because a compressed key ropes at its group's first position.
        with torch.device(self.device):
            rope = get_rope(
                head_dim=self.kvcache.index_head_dim,
                rotary_dim=rotary.rotary_dim,
                max_position=rotary.max_position,
                base=rotary.base,
                rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
            )
        self._qsa_cos_sin = rope._cos_sin_cache.to(self.device)
        self._qsa_buffers: dict[tuple, torch.Tensor] = {}
        self._qsa_norm_weights: dict[tuple, torch.Tensor] = {}
        # Sized once: _qsa_stage writes it OUTSIDE the tape, so a realloc strands the taped view.
        self._qsa_ring_slots = torch.zeros(
            self.kvcache.num_req_slots, dtype=torch.int32, device=self.device
        )

    def prepare_for_capture(self, batch: Batch) -> None:
        super().prepare_for_capture(batch)
        # Host lists so a recording forward (mps_tape.py) never reads them off device.
        bs = batch.size
        batch.attn_metadata.cu_seqlens_q_host = list(range(bs + 1))
        batch.attn_metadata.indptr_host = list(range(bs + 1))
        if self._qsa_slot is not None:
            self._qsa_stage(batch)

    def prepare_metadata(self, batch: Batch) -> None:
        super().prepare_metadata(batch)
        if self._qsa_slot is not None:
            self._qsa_stage(batch)

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

    # ----- QSA ------------------------------------------------------------------------------
    @staticmethod
    def _qsa_round(columns: int) -> int:
        """Up to a power of two. Exact: the kernel -infs every column past a row's visible
        count, which is what bounds the top-k and the expansion."""
        return 1 << max(columns - 1, 0).bit_length()

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        super().init_capture_graph(max_seq_len, bs_list)
        if self._qsa_slot is not None:
            self._qsa_columns = self._qsa_round(
                max(self._qsa_topk, max_seq_len // self._qsa_ratio)
            )

    def reset_capture(self) -> None:
        super().reset_capture()
        # A stale capacity would leave the newest blocks unscored, and unselectable, in silence.
        self._qsa_columns = 0

    def _qsa_buf(self, name: str, rows: int, *shape: int, dtype: torch.dtype) -> torch.Tensor:
        """A per-forward transient, keyed by trailing shape so the key space stays O(1). A
        growing prefill DOES replace the entry a decode tape recorded; that is safe only
        because ``mps_tape._launch`` keeps a strong ref to the tensor it taped, so the replay
        stays self-consistent on the old one (which then stays resident, ~200 KB at bs 8)."""
        key = (name, shape, dtype)
        buf = self._qsa_buffers.get(key)
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty((rows, *shape), dtype=dtype, device=self.device)
            self._qsa_buffers[key] = buf
        return buf[:rows]

    def _qsa_stage(self, batch: Batch) -> None:
        """Each request's ring / scratch slot, staged so the recorded forward does no host work."""
        reqs = batch.padded_reqs
        slots = self._qsa_ring_slots[: len(reqs)]
        live = getattr(batch, "active_table_idx", None)  # decode only, and absent at capture
        if live is not None and live.numel() == len(reqs):
            slots.copy_(live)  # decode: the scheduler already put these on the device
        else:
            slots.copy_(_int32_on([r.table_idx for r in reqs], self.device))

    def qsa_forward(
        self,
        q: torch.Tensor,  # [T, HQ, D]
        k: torch.Tensor,  # [T, KVH * D]
        v: torch.Tensor,  # [T, KVH * D]
        index,  # models.qwen4_exp.attention.QSAIndexerInputs
        layer_id: int,
        batch: Batch,
    ) -> torch.Tensor:
        from freetoken.kernel.metal.attention import paged_attention_mps
        from freetoken.kernel.metal.qsa import (
            qsa_compress_groups_mps,
            qsa_index_attention_mps,
            qsa_store_ring_mps,
        )

        assert self._qsa_slot is not None, "qsa_forward needs a QSA pool"
        md = batch.attn_metadata
        assert md.q_positions.dtype == torch.int32, "the QSA kernels read int32 positions"
        slot = self._qsa_slot[layer_id]
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads, head_dim = k_raw.shape[-2], k_raw.shape[-1]
        k_cache = k_raw.view(-1, kv_heads, head_dim)
        v_cache = v_raw.view(-1, kv_heads, head_dim)
        scale = head_dim**-0.5

        # The slab and the ring feed LATER tokens, so both update whichever way we attend. Compress
        # before the ring refresh: the refresh overwrites exactly the rows a straddling group reads.
        ring = self.kvcache.pending_ring(slot)
        ring_slots = self._qsa_ring_slots[: md.indptr.numel() - 1]
        qsa_compress_groups_mps(
            self.kvcache.cmp_k_cache(slot), index.k, ring, ring_slots, md.cu_seqlens_q_gpu,
            md.q_to_req, md.q_positions, batch.out_loc, self._qsa_cos_sin,
            self._qsa_fp32_weight((slot, "k"), index.k_norm_weight), index.eps,
            self._qsa_ratio, self.kvcache.cmp_scratch_base,
        )
        qsa_store_ring_mps(ring, index.k, ring_slots, md.cu_seqlens_q_gpu, md.q_positions)
        if not md.is_decode and self._qsa_is_dense(md):
            # Under index_budget + index_ratio - 1 visible tokens the selection IS the causal
            # prefix, so the tiled kernel is exact and computing it first would be waste.
            return paged_attention_mps(
                q=q, k_cache=k_cache, v_cache=v_cache, indptr=md.indptr, indices=md.indices,
                cu_seqlens_q=md.cu_seqlens_q_gpu, sm_scale=scale, kv_ptr=md.indptr_host,
                q_ptr=md.cu_seqlens_q_host,
            )
        return qsa_index_attention_mps(
            q, k_cache, v_cache, self._select(index, md, slot), md.indptr, md.indices,
            md.q_to_req, scale, torch.empty_like(q),
        )

    def _qsa_is_dense(self, md) -> bool:
        host = md.indptr_host
        return all(
            int(host[i + 1]) - int(host[i]) <= self._qsa_width for i in range(len(host) - 1)
        )

    def _qsa_fp32_weight(self, key: tuple, weight: torch.Tensor) -> torch.Tensor:
        """Read as float32 though they ride the model dtype, and cached per (layer, role):
        they are constants, so recopying them would tape two device copies per layer per step."""
        buf = self._qsa_norm_weights.get(key)
        if buf is None:
            buf = torch.empty(weight.numel(), dtype=torch.float32, device=self.device)
            buf.copy_(weight)
            self._qsa_norm_weights[key] = buf
        return buf

    def _select(self, index, md, slot: int) -> torch.Tensor:
        """Score, top-k and expand: [T, index_budget + index_ratio - 1] int32 token ids."""
        from freetoken.kernel.metal.qsa import (
            qsa_block_scores_mps,
            qsa_expand_mps,
            qsa_index_norm_rope_mps,
        )

        rows, heads, dim = index.q.shape
        q_index = self._qsa_buf("q_index", rows, heads, dim, dtype=index.q.dtype)
        qsa_index_norm_rope_mps(
            index.q, md.q_positions, self._qsa_cos_sin,
            self._qsa_fp32_weight((slot, "q"), index.q_norm_weight), index.eps, q_index,
        )
        cmp_k = self.kvcache.cmp_k_cache(slot)
        host = md.indptr_host
        longest = max(int(host[i + 1]) - int(host[i]) for i in range(len(host) - 1))
        needed = -(-longest // self._qsa_ratio)
        if md.is_decode and self._qsa_columns:
            # One tape serves every kv_len, so it walks the pinned capacity and masks per row;
            # a short kv_len still pays for scoring the whole page table.
            columns = self._qsa_columns
            assert needed <= columns, (
                f"QSA decode scores {columns} blocks but kv_len {longest} needs {needed}; "
                "init_capture_graph sized the capacity for a smaller page table"
            )
        else:
            columns = self._qsa_round(max(self._qsa_topk, needed))
        sel = self._qsa_buf("sel", rows, self._qsa_width, dtype=torch.int32)
        per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // (columns * 4))
        for lo in range(0, rows, per_chunk):
            hi = min(lo + per_chunk, rows)
            # Flat, then viewed: a [rows, columns] key mints a buffer per column count.
            logits = self._qsa_buf(
                "logits", (hi - lo) * columns, dtype=torch.float32
            ).view(hi - lo, columns)
            qsa_block_scores_mps(
                q_index[lo:hi], cmp_k, md.indptr, md.indices, md.q_to_req[lo:hi],
                md.q_positions[lo:hi], self._qsa_ratio, logits,
            )
            values = self._qsa_buf("topk_values", hi - lo, self._qsa_topk, dtype=torch.float32)
            ranked = self._qsa_buf("topk_blocks", hi - lo, self._qsa_topk, dtype=torch.int64)
            torch.topk(logits, self._qsa_topk, dim=-1, out=(values, ranked))
            blocks = self._qsa_buf("blocks", hi - lo, self._qsa_topk, dtype=torch.int32)
            blocks.copy_(ranked)
            # A -inf winner can only land at a rank past the row's visible-block count, and
            # the expansion reads no rank past it, so masked columns need no second pass.
            qsa_expand_mps(
                blocks, md.q_positions[lo:hi], md.indptr, md.q_to_req[lo:hi], self._qsa_ratio,
                sel[lo:hi],
            )
        return sel


__all__ = ["MetalAttentionBackend"]
