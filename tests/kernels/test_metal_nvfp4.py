"""Metal NVFP4 kernels against float32 CPU oracles (tests/kernels/_refs.py): bit-exact for dequant, a tolerance for the GEMVs."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.metal import is_available
from freetoken.kernel.metal.nvfp4 import (
    dequant_nvfp4,
    dequant_nvfp4_dense,
    moe_decode_nvfp4,
    moe_prefill_nvfp4_grouped,
    nvfp4_gemv,
)
from tests.kernels._refs import SHAPES, banks, dequant_ref, moe_ref, rel, rel_norm, routing

pytestmark = pytest.mark.skipif(not is_available(), reason="needs a torch build with MPS")

SLOT0 = torch.tensor([0], dtype=torch.int32)


def _mps(*tensors):
    return [t.to("mps") for t in tensors]


def _moe_args(experts, top_k, m, h, inter, seed):
    gu = banks(experts, 2 * inter, h, seed=seed)
    dn = banks(experts, h, inter, seed=seed + 1)
    x = torch.randn(m, h, generator=torch.Generator().manual_seed(seed + 2))
    ids, weights = routing(m, experts, top_k, seed=seed + 3)
    args = [x.to("mps"), *_mps(*gu), *_mps(*dn), weights.to("mps"), ids.to("mps")]
    return x, gu, dn, ids, weights, args


def test_dequant():
    # no accumulation, so demand bit-exactness of both the routed and the dense form
    for out_rows, in_dim in SHAPES:
        packed, scale, glob = banks(8, out_rows, in_dim, seed=1)
        slots = torch.tensor([3, 0, 7, 3], dtype=torch.int32)
        got = dequant_nvfp4(*_mps(packed, scale, glob, slots))
        want = dequant_ref(packed, scale, glob, slots)
        assert torch.equal(got.cpu(), want), (got.cpu() - want).abs().max()
        dense = dequant_nvfp4_dense(*_mps(packed[0], scale[0], glob[0]), dtype=torch.float32)
        assert torch.equal(dense.cpu(), dequant_ref(packed, scale, glob, SLOT0)[0])


def test_gemv():
    # the dense lm_head path; out dtype follows the activation unless out_dtype overrides it
    for n, k in SHAPES:
        for dtype, metric in ((torch.float32, rel), (torch.bfloat16, rel_norm),
                              (torch.float16, rel_norm)):
            packed, scale, glob = banks(1, n, k, seed=19)
            x = torch.randn(3, k, generator=torch.Generator().manual_seed(20)).to(dtype)
            args = _mps(packed[0], scale[0], glob[0])
            got = nvfp4_gemv(x.to("mps"), *args)
            want = x.float() @ dequant_ref(packed, scale, glob, SLOT0)[0].T
            assert got.dtype == dtype and metric(got, want) < 1e-2
            assert nvfp4_gemv(x.to("mps"), *args, out_dtype=torch.float32).dtype == torch.float32
            # and against the unfused dequant-then-matmul through the same banks
            w = dequant_nvfp4_dense(*args, dtype=torch.float32)
            assert rel(nvfp4_gemv(x.float().to("mps"), *args), (x.float().to("mps") @ w.T).cpu()) < 1e-2


def test_gemv_epilogues():
    # swiglu, the sigmoid gate, and the side gate row, each against the unfused pair
    for n, k in ((256, 64), (400, 80)):
        m = 3
        packed, scale, glob = banks(1, n, k, seed=61)
        g = torch.Generator().manual_seed(62)
        x = torch.randn(m, k, generator=g)
        gate = torch.randn(m, generator=g)
        gw = torch.randn(1, k, generator=g)
        args = [x.to("mps"), *_mps(packed[0], scale[0], glob[0])]

        plain = nvfp4_gemv(*args)
        sw = nvfp4_gemv(*args, swiglu=True)
        assert sw.shape == (m, n // 2)
        assert rel(sw, (torch.nn.functional.silu(plain[:, : n // 2]) * plain[:, n // 2:]).cpu()) < 1e-5
        gated = nvfp4_gemv(*args, gate=gate.to("mps").unsqueeze(1))
        assert rel(gated, (plain * torch.sigmoid(gate.to("mps")).unsqueeze(1)).cpu()) < 1e-5
        both = nvfp4_gemv(*args, swiglu=True, gate=gate.to("mps"))
        assert rel(both, (sw * torch.sigmoid(gate.to("mps")).unsqueeze(1)).cpu()) < 1e-5
        out, side = nvfp4_gemv(*args, swiglu=True, side_gate_weight=gw.to("mps"))
        assert side.shape == (m,) and torch.equal(out.cpu(), sw.cpu())
        assert rel(side.float(), torch.nn.functional.linear(x, gw).reshape(m)) < 1e-5


def test_decode():
    # TOP_K compiles a different kernel per value (the route loop is unrolled), so both
    # ends of the shipped range run; m is a batch size, not a code path.
    for h, inter in SHAPES:
        for top_k in (1, 8):
            x, gu, dn, ids, weights, args = _moe_args(8, top_k, 3, h, inter, seed=3)
            got = moe_decode_nvfp4(*args)
            assert got.shape == (3, h)
            err = rel(got, moe_ref(x, gu, dn, weights, ids))
            assert err < 1e-2, f"h={h} inter={inter} top_k={top_k}: rel {err:.3e}"
            # the fused base epilogue is a plain add on top, so demand bit-exactness
            base = torch.randn(3, h, generator=torch.Generator().manual_seed(44))
            fused = moe_decode_nvfp4(*args, base=base.to("mps"))
            assert torch.equal(fused.cpu(), got.cpu() + base)


def test_decode_shared_expert():
    # the shared expert rides as an extra route: sigmoid(x @ gate_w) * down(silu(gate) * up)
    for h, inter in SHAPES:
        _, _, _, _, _, args = _moe_args(8, 4, 3, h, inter, seed=91)
        sgu = _mps(*[t[0] for t in banks(1, 2 * inter, h, seed=93)])
        sdn = _mps(*[t[0] for t in banks(1, h, inter, seed=94)])
        gw = torch.randn(h, generator=torch.Generator().manual_seed(96)).to("mps")
        x = args[0]
        routed = moe_decode_nvfp4(*args)
        gate = torch.nn.functional.linear(x, gw.unsqueeze(0)).reshape(x.shape[0])
        want = routed + nvfp4_gemv(nvfp4_gemv(x, *sgu, swiglu=True), *sdn, gate=gate)
        got = moe_decode_nvfp4(*args, shared=(*sgu, *sdn, gw))
        assert got.shape == (x.shape[0], h) and rel(got, want.cpu()) < 1e-4


def test_prefill_grouped():
    # one threadgroup row per expert, walking group_rows routes at a time; 1 and 8 bound
    # the ragged-run and the idle-expert clamp
    for h, inter in SHAPES:
        for group_rows in (1, 8):
            x, gu, dn, ids, weights, args = _moe_args(8, 4, 12, h, inter, seed=7)
            got = moe_prefill_nvfp4_grouped(*args, group_rows=group_rows)
            err = rel(got, moe_ref(x, gu, dn, weights, ids))
            assert err < 1e-2, f"h={h} group_rows={group_rows}: rel {err:.3e}"
        assert rel(moe_prefill_nvfp4_grouped(*args), moe_decode_nvfp4(*args).cpu()) < 1e-2
    # half the experts are never routed to, so their threadgroup rows must clamp to nothing
    gu = banks(16, 2 * 48, 80, seed=31)
    dn = banks(16, 80, 48, seed=32)
    x = torch.randn(6, 80, generator=torch.Generator().manual_seed(33))
    ids, weights = routing(6, 8, 2, seed=34)
    idle = moe_prefill_nvfp4_grouped(x.to("mps"), *_mps(*gu), *_mps(*dn),
                                     weights.to("mps"), ids.to("mps"))
    assert rel(idle, moe_ref(x, gu, dn, weights, ids)) < 1e-2


def test_prefill_grouped_ignores_which_slot_an_expert_sits_in():
    # the SSD tier hands these kernels slot ids, and slots follow the cache's history: the same
    # experts under another numbering must give the same bits, or greedy ids follow the cache
    h, inter = SHAPES[0]
    for m in (12, 300):  # the row-grouped kernel, then the tiled one
        x, gu, dn, ids, weights, args = _moe_args(16, 4, m, h, inter, seed=41)
        perm = torch.randperm(16, generator=torch.Generator().manual_seed(42))
        held = torch.argsort(perm)  # slot s holds expert held[s]
        moved = [x.to("mps"), *_mps(*(b[held] for b in gu)), *_mps(*(b[held] for b in dn)),
                 weights.to("mps"), perm[ids.long()].to(ids.dtype).to("mps")]
        assert torch.equal(moe_prefill_nvfp4_grouped(*args), moe_prefill_nvfp4_grouped(*moved)), m


def _ragged(m, experts, top_k, seed):
    """Routing whose run lengths span the tile: expert 0 takes every token, expert 1 exactly
    one route, and the last expert none. The rest come from experts 2..E-2, distinct per row."""
    g = torch.Generator().manual_seed(seed)
    pool = torch.arange(2, experts - 1)
    rest = [pool[torch.randperm(pool.numel(), generator=g)[: top_k - 1]] for _ in range(m)]
    ids = torch.cat([torch.zeros(m, 1, dtype=torch.long), torch.stack(rest).view(m, -1)], 1)
    ids[0, 1:2] = 1
    w = torch.rand(m, top_k, generator=g)
    return ids.int(), w / w.sum(-1, keepdim=True)


@pytest.mark.parametrize("m", (128, 512))
def test_prefill_tiled(m):
    # m straddles _TILED_MIN_M: 128 keeps the row-grouped kernel, 512 takes the tiled GEMM.
    # H 80 / I 48 also puts a K tail and a partial output tile in the tiled path, since
    # neither divides its NK 32 / NR0 64.
    for h, inter in SHAPES:
        x, gu, dn, ids, weights, args = _moe_args(8, 4, m, h, inter, seed=13)
        got = moe_prefill_nvfp4_grouped(*args)
        assert got.shape == (m, h)
        # The tiled gate/up GEMM stages activations as bfloat16, so a float32 x costs that
        # much; the engine's x is a widened bfloat16 residual and is exact (see below).
        err = rel_norm(got, moe_ref(x, gu, dn, weights, ids))
        assert err < 1e-2, f"m={m} h={h} inter={inter}: rel_norm {err:.3e}"
        # Naming group_rows always lands on the row-grouped kernel, so below the threshold
        # the two agree bit for bit and above it they cannot: that proves the predicate routed.
        rows = moe_prefill_nvfp4_grouped(*args, group_rows=2)
        assert rel_norm(got, rows.cpu()) < 1e-2
        assert torch.equal(got.cpu(), rows.cpu()) == (m < 256)
        # The MoE layer widens a bfloat16 residual to float32, so on that input the tiled
        # GEMM rounds nothing and only the K order differs (measured 1.3e-7 to 2.5e-7).
        args[0] = args[0].to(torch.bfloat16).float()
        err = rel_norm(moe_prefill_nvfp4_grouped(*args),
                       moe_prefill_nvfp4_grouped(*args, group_rows=2).cpu())
        assert err < 1e-6, f"m={m} h={h} inter={inter}: rel_norm {err:.3e}"


def test_prefill_tiled_ragged_routing():
    # One expert with every token, one with a single route, one with none, and a top_k
    # reduction over all of them: the run bounds are what the route tile clamps against.
    for h, inter in SHAPES:
        for top_k in (1, 4):
            experts = 8
            gu = banks(experts, 2 * inter, h, seed=51)
            dn = banks(experts, h, inter, seed=52)
            x = torch.randn(512, h, generator=torch.Generator().manual_seed(53))
            ids, weights = _ragged(512, experts, top_k, seed=54)
            counts = torch.bincount(ids.reshape(-1), minlength=experts)
            assert counts[0] == 512 and counts[experts - 1] == 0
            assert top_k == 1 or counts[1] == 1
            got = moe_prefill_nvfp4_grouped(x.to("mps"), *_mps(*gu), *_mps(*dn),
                                            weights.to("mps"), ids.to("mps"))
            err = rel_norm(got, moe_ref(x, gu, dn, weights, ids))
            assert err < 1e-2, f"h={h} inter={inter} top_k={top_k}: rel_norm {err:.3e}"


def test_shape_contract_is_refused_not_silently_wrong():
    packed, scale, glob = banks(1, 33, 32, seed=69)
    dense = _mps(packed[0], scale[0], glob[0])
    x = torch.zeros(2, 32, device="mps")
    with pytest.raises(ValueError, match="swiglu needs an even row count"):
        nvfp4_gemv(x, *dense, swiglu=True)
    with pytest.raises(ValueError, match="gate has"):
        nvfp4_gemv(x, *dense, gate=torch.zeros(3, device="mps"))
    with pytest.raises(TypeError, match="no Metal scalar type"):
        nvfp4_gemv(torch.zeros(1, 32, dtype=torch.int32, device="mps"), *dense)

    _, _, _, _, _, args = _moe_args(4, 2, 2, 32, 16, seed=46)
    with pytest.raises(ValueError, match="base is"):
        moe_decode_nvfp4(*args, base=torch.zeros(3, 32, device="mps"))
    small_gu = _mps(*[t[0] for t in banks(1, 2 * 8, 32, seed=101)])   # I = 8, not 16
    small_dn = _mps(*[t[0] for t in banks(1, 32, 8, seed=102)])
    with pytest.raises(ValueError, match="shared expert banks"):
        moe_decode_nvfp4(*args, shared=(*small_gu, *small_dn, torch.randn(32).to("mps")))
