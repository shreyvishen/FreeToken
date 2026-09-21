"""The ``metal`` attention backend: a real forward through a real ``MHAKVCache`` and page table against the float32 CPU oracle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttentionSpec, create_attention_backend
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.metal import is_available

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)

Q_HEADS, KV_HEADS, HEAD_DIM, DEV = 16, 2, 256, "mps"
CONFIG = SimpleNamespace(num_qo_heads=Q_HEADS, head_dim=HEAD_DIM, kv_cache_group_specs=lambda: ())


@pytest.mark.skipif(not is_available(), reason="needs a torch build with MPS")
def test_forward_matches_the_cpu_oracle():
    from freetoken import core
    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.kvcache.mha_pool import MHAKVCache
    from tests.kernels.test_triton_attention import _reference_paged_attention

    ctx = core.Context(page_size=1)
    ctx.kv_cache = pool = MHAKVCache(num_kv_heads=KV_HEADS, num_layers=1, head_dim=HEAD_DIM,
                                     num_pages=128, page_size=1, dtype=torch.float32,
                                     device=torch.device(DEV))
    ctx.page_table = torch.zeros((8, 256), dtype=torch.int32, device=DEV)
    saved, core._GLOBAL_CTX = core._GLOBAL_CTX, ctx
    try:
        lens, _ = [7, 3], torch.manual_seed(0)
        # each request's tokens occupy scattered KV slots, as the allocator gives them
        slots = {0: torch.arange(20, 27), 1: torch.arange(5, 8)}
        reqs = [Req(input_ids=torch.zeros(n, dtype=torch.int64), table_idx=i, cached_len=0,
                    output_len=8, uid=i, sampling_params=SamplingParams(), cache_handle=None)
                for i, n in enumerate(lens)]
        for r in reqs:
            ctx.page_table[r.table_idx, : r.device_len] = slots[r.table_idx].to(DEV, torch.int32)
        batch = Batch(reqs=reqs, phase="prefill")
        batch.padded_reqs = reqs
        batch.positions = torch.cat([torch.arange(r.device_len) for r in reqs]).to(DEV)
        batch.out_loc = torch.cat([slots[0], slots[1]]).to(DEV, torch.int32)
        q = torch.randn(sum(lens), Q_HEADS, HEAD_DIM, device=DEV)
        k, v = torch.randn(2, sum(lens), KV_HEADS, HEAD_DIM, device=DEV)
        backend = create_attention_backend("metal", config=CONFIG)
        backend.prepare_metadata(batch)
        out = backend.forward(q, k, v, 0, batch)
        # the backend must have written this step's K/V into the pool on the way through
        cached_k = pool.k_cache(0).view(-1, KV_HEADS, HEAD_DIM)
        torch.testing.assert_close(cached_k[batch.out_loc.long()], k)  # measured bit-exact
        md = batch.attn_metadata
        want = _reference_paged_attention(
            q.float().cpu(), cached_k.float().cpu(),
            pool.v_cache(0).view(-1, KV_HEADS, HEAD_DIM).float().cpu(), md.indptr.cpu(),
            md.indices.cpu(), torch.tensor([0] * lens[0] + [1] * lens[1], dtype=torch.int32),
            torch.cat([torch.arange(n) for n in lens]), HEAD_DIM**-0.5, None)
        # measured max abs 1.4e-06, max rel 1.7e-02
        torch.testing.assert_close(out.cpu(), want, rtol=2e-5, atol=2e-5)
        with pytest.raises(NotImplementedError, match="sliding window"):
            backend.forward(q, k, v, 0, batch, attn_spec=AttentionSpec(sliding_window=2))
    finally:
        core._GLOBAL_CTX = saved
