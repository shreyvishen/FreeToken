"""Paged GQA attention on MPS against the repo's shared float32 causal-GQA oracle, at Qwen3.6's shape: 16 query heads, 2 K/V heads, head dim 256."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.metal import is_available
from freetoken.kernel.metal.attention import (
    paged_attention_mps,
    paged_decode_attention_mps,
    prefill_tile,
)
from tests.kernels.test_triton_attention import _reference_paged_attention

pytestmark = pytest.mark.skipif(not is_available(), reason="needs a torch build with MPS")

Q_HEADS, KV_HEADS, HEAD_DIM = 16, 2, 256
SCALE = HEAD_DIM**-0.5


def _oracle_args(indptr, cu_q):
    q_to_req, q_positions = [], []
    for i in range(len(cu_q) - 1):
        q_len, kv_len = int(cu_q[i + 1] - cu_q[i]), int(indptr[i + 1] - indptr[i])
        q_to_req += [i] * q_len
        q_positions += list(range(kv_len - q_len, kv_len))
    return torch.tensor(q_to_req, dtype=torch.int32), torch.tensor(q_positions)


def _run(kernel, q, k, v, indptr, indices, cu_q, scale=SCALE, **kw):
    got = kernel(*[t.to("mps") for t in (q, k, v, indptr, indices, cu_q)], scale, **kw).cpu()
    q_to_req, q_positions = _oracle_args(indptr, cu_q)
    want = _reference_paged_attention(q.float(), k.float(), v.float(), indptr, indices,
                                      q_to_req, q_positions, scale, None)
    return got, want


def _cache(slots, ql, hq=Q_HEADS, hk=KV_HEADS, d=HEAD_DIM, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    return torch.randn(ql, hq, d, dtype=dtype), *torch.randn(2, slots, hk, d, dtype=dtype)


def test_paged_decode():
    # kv_len 3 leaves five of the kernel's 8 SIMD-groups with no position (they must add 0, not
    # NaN); 1000 is not a multiple of 8; head_dim 96 is not one of 32 lanes, so the tail guard runs.
    qwen = (Q_HEADS, KV_HEADS, HEAD_DIM)
    for kv_lens, (hq, hk, d) in (([3, 17], qwen), ([1000], qwen), ([20, 7], (4, 4, 96))):
        for dtype, tol in ((torch.float32, 2e-5), (torch.bfloat16, 8e-3)):
            total, reqs = sum(kv_lens), len(kv_lens)
            slots = total + 5
            q, k, v = _cache(slots, reqs, hq, hk, d, dtype=dtype, seed=total)
            # scattered slots, so a kernel that ignored the page table would fail
            indices = torch.randperm(slots)[:total].to(torch.int32)
            indptr = torch.tensor([0] + torch.tensor(kv_lens).cumsum(0).tolist(),
                                  dtype=torch.int32)
            cu_q = torch.arange(reqs + 1, dtype=torch.int32)
            got, want = _run(paged_decode_attention_mps, q, k, v, indptr, indices, cu_q, d**-0.5)
            assert got.dtype == dtype
            # measured max abs 7.4e-03
            torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)
            # paged_attention_mps must route a one-token-per-request batch to this kernel
            routed, _ = _run(paged_attention_mps, q, k, v, indptr, indices, cu_q, d**-0.5)
            assert torch.equal(routed, got)

    # the fused output gate must equal the separate elementwise launch it folds in
    from freetoken.kernel.metal.elementwise import sigmoid_gate_mul_metal

    bs, hq, hk, d, slots, lens = 3, 8, 2, 64, 40, [5, 1, 12]
    q, k, v = [t.to("mps") for t in _cache(slots, bs, hq, hk, d, torch.bfloat16, seed=3)]
    md = [torch.tensor([0, 5, 6, 18], dtype=torch.int32).to("mps"),
          torch.randperm(slots)[: sum(lens)].to(torch.int32).to("mps"),
          torch.arange(bs + 1, dtype=torch.int32, device="mps")]
    gate = torch.randn(bs, hq * d, dtype=torch.bfloat16, device="mps")
    plain = paged_attention_mps(q, k, v, *md, d**-0.5)
    want = sigmoid_gate_mul_metal(plain.reshape(bs, hq * d), gate).reshape(bs, hq, d)
    assert torch.equal(paged_attention_mps(q, k, v, *md, d**-0.5, gate=gate).cpu(), want.cpu())


# (q lengths, kv lengths) per request. 37 is not a multiple of any query tile, so its last
# row block is only partly live; 100 queries after a 900-token prefix is the chunked-prefill
# case, with enough KV tiles for the running softmax to be rescaled many times over.
_PREFILL_BATCHES = (([12], [12]), ([5], [14]), ([37], [64]), ([100], [1000]),
                    ([6, 3, 1, 40], [6, 11, 40, 40]))


@pytest.mark.parametrize("dtype,tol", ((torch.float32, 2e-5), (torch.bfloat16, 8e-3)))
def test_tiled_prefill_matches_the_oracle(dtype, tol):
    assert prefill_tile(HEAD_DIM, torch.empty((), dtype=dtype).element_size()) is not None, (
        "the tiled kernel must own Qwen3.6's shape, or this file is testing the SDPA fallback"
    )
    for q_lens, kv_lens in _PREFILL_BATCHES:
        total, slots = sum(kv_lens), sum(kv_lens) + 9
        q, k, v = _cache(slots, sum(q_lens), dtype=dtype, seed=total)
        # scattered slots: a kernel that walked the pool in slot order rather than page order
        # would pass every contiguous case and fail here
        indices = torch.randperm(slots)[:total].to(torch.int32)
        cum = lambda lens: torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(),
                                        dtype=torch.int32)
        got, want = _run(paged_attention_mps, q, k, v, cum(kv_lens), indices, cum(q_lens))
        assert got.dtype == dtype
        # measured max abs 8.2e-03
        torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)


def test_prefill_falls_back_when_the_kernel_has_no_shape_for_it():
    md = (torch.tensor([0, 12, 30], dtype=torch.int32),
          torch.randperm(40)[:30].to(torch.int32),
          torch.tensor([0, 4, 9], dtype=torch.int32))
    # head_dim 20 leaves a tail no 8x8 fragment covers
    assert prefill_tile(20, 2) is None
    q, k, v = _cache(40, 9, hq=4, hk=2, d=20, seed=2)
    got, want = _run(paged_attention_mps, q, k, v, *md, scale=20**-0.5)
    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)  # measured max abs 3.6e-07

    # and a K/V cache sliced out of a wider tensor is not the flat slab the kernel indexes
    q, wide_k, wide_v = _cache(40, 9, hk=2 * KV_HEADS, seed=3)
    q, wide_k, wide_v = (t.to("mps") for t in (q, wide_k, wide_v))
    k, v = wide_k[:, :KV_HEADS], wide_v[:, :KV_HEADS]
    assert not k.is_contiguous()
    got = paged_attention_mps(q, k, v, *[t.to("mps") for t in md], SCALE).cpu()
    q_to_req, q_positions = _oracle_args(md[0], md[2])
    want = _reference_paged_attention(q.cpu().float(), k.cpu().float(), v.cpu().float(),
                                      md[0], md[1], q_to_req, q_positions, SCALE, None)
    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)  # measured max abs 7.7e-07


def test_a_growing_decode_does_not_grow_the_mps_allocator():
    _, k, v = [t.to("mps") for t in _cache(4096, 1, dtype=torch.bfloat16, seed=8)]
    q = torch.randn(1, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="mps")
    cu_q = torch.tensor([0, 1], device="mps")
    slots = torch.arange(4096, dtype=torch.int32, device="mps")
    step = lambda n: paged_attention_mps(q, k, v, torch.tensor([0, n], device="mps"),
                                         slots[:n], cu_q, SCALE)
    # The growth needs kv_len to rise monotonically: a free buffer is reused for any
    # smaller request, so only a new high-water size ever mints another one.
    step(1024)
    torch.mps.synchronize()
    before = torch.mps.driver_allocated_memory()
    for kv_len in range(1025, 1537):
        step(kv_len)
    torch.mps.synchronize()
    grew = torch.mps.driver_allocated_memory() - before
    # Measured over these 512 steps: 11592.7 MiB before the scratch pool, 0.4 MiB after.
    assert grew < 64 << 20, f"driver pool grew {grew / (1 << 20):.1f} MiB over 512 decode steps"


def test_qsa_index_attention():
    """The QSA sparse attend against an fp32 gather-and-softmax over the same token lists:
    ragged widths, a row whose list is entirely padding, and scattered pool slots."""
    from freetoken.kernel.metal.qsa import qsa_index_attention_mps

    torch.manual_seed(5)
    kv_lens, width = [70, 9], 24
    total, slots = sum(kv_lens), sum(kv_lens) + 6
    token_to_req = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    rows = token_to_req.numel()
    indptr = torch.tensor([0] + torch.tensor(kv_lens).cumsum(0).tolist(), dtype=torch.int32)
    indices = torch.randperm(slots)[:total].to(torch.int32)
    q, k, v = _cache(slots, rows, dtype=torch.bfloat16, seed=5)

    sel = torch.full((rows, width), -1, dtype=torch.int32)
    for row in range(rows):
        kv_len = kv_lens[int(token_to_req[row])]
        n = min(row * 6, kv_len, width)  # row 0 selects nothing at all
        sel[row, :n] = torch.randperm(kv_len)[:n].to(torch.int32)

    out = torch.empty(rows, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="mps")
    got = qsa_index_attention_mps(
        *[t.to("mps") for t in (q, k, v, sel, indptr, indices, token_to_req)], SCALE, out
    ).cpu()

    rep = Q_HEADS // KV_HEADS
    want = torch.zeros(rows, Q_HEADS, HEAD_DIM)
    for row in range(rows):
        tokens = sel[row][sel[row] >= 0].long()
        if not tokens.numel():
            continue  # an all-padding row must come back zero, not NaN
        pool = indices[int(indptr[token_to_req[row]]) + tokens].long()
        keys = k[pool].float().repeat_interleave(rep, dim=1)
        values = v[pool].float().repeat_interleave(rep, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys) * SCALE
        want[row] = torch.einsum("hk,khd->hd", scores.softmax(-1), values)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want, rtol=8e-3, atol=8e-3)  # measured max abs 3.9e-03
