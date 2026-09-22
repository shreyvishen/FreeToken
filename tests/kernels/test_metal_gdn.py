"""GatedDeltaNet on Metal against the repo's CPU float32 oracle (``recurrent_gated_delta_rule``, loaded by file path to avoid the CUDA-only import chain)."""

from __future__ import annotations

import importlib.util
import itertools
import pathlib

import pytest
import torch

from freetoken.kernel.fla.const import CHUNK_SIZE
from freetoken.kernel.metal import is_available
from freetoken.kernel.metal.gdn import (
    fused_decode_supports, gate_params_metal, gdn_decode_fused_metal, gdn_decode_metal,
    gdn_prefill_metal, gdn_recurrent_metal,
)
from freetoken.kernel.metal.ops import rms_norm_gated

pytestmark = pytest.mark.skipif(not is_available(), reason="needs a torch build with MPS")

DK = DV = 128  # Qwen3.5/3.6 GDN head dims; the kernel needs Dk % 32 == 0
# a single decode step with no GQA sharing, then Qwen3.6's own geometry with deep sharing
SHAPES = ((1, 1, 2, 2), (1, 32, 16, 32))


def _oracle():
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "python/freetoken/models/qwen3_5_moe/gdn_reference.py")
    spec = importlib.util.spec_from_file_location("gdn_reference_oracle", path)
    spec.loader.exec_module(module := importlib.util.module_from_spec(spec))
    return module


def test_recurrent():
    for b, t, hk, hv in SHAPES:
        torch.manual_seed(0)
        inp = dict(q=torch.randn(b, t, hk, DK), k=torch.randn(b, t, hk, DK),
                   v=torch.randn(b, t, hv, DV), g=-torch.rand(b, t, hv),  # log-decay is <= 0
                   beta=torch.rand(b, t, hv))
        up = lambda n: inp[n].float().repeat_interleave(hv // hk, dim=2)
        ref_out, ref_state = _oracle().recurrent_gated_delta_rule(
            up("q"), up("k"), inp["v"].float(), inp["g"], inp["beta"])
        slots = torch.arange(1, b + 1, dtype=torch.int32).to("mps")
        state = torch.randn(b + 2, hv, DK, DV, device="mps")
        before, dev = state.clone(), {n: x.to("mps") for n, x in inp.items()}
        state[slots.long()] = 0
        # split the sequence so the second call has to continue from a live state
        spans = ((0, t),) if t == 1 else ((0, t // 2), (t // 2, t))
        out = torch.cat(
            [gdn_recurrent_metal(**{n: x[:, lo:hi] for n, x in dev.items()},
                                 state_source=state, indices=slots, scale=DK**-0.5)
             for lo, hi in spans], dim=1).cpu()
        assert (out - ref_out).abs().max() < 1e-5
        # the state the pool keeps is the whole memory of the sequence, so check it too
        assert (state.cpu()[slots.cpu().long()] - ref_state).abs().max() < 1e-5
        idle = [i for i in range(b + 2) if i not in slots.cpu().tolist()]
        assert torch.equal(state[idle], before[idle])

    z = torch.zeros   # Dk = 48 is not a whole number of 32-lane tiles
    with pytest.raises(ValueError, match="Dk % 32"):
        gdn_recurrent_metal(z(1, 1, 1, 48, device="mps"), z(1, 1, 1, 48, device="mps"),
                            z(1, 1, 1, 128, device="mps"), z(1, 1, 1, device="mps"),
                            z(1, 1, 1, device="mps"), state_source=z(1, 1, 48, 128, device="mps"),
                            indices=z(1, dtype=torch.int32, device="mps"), scale=1.0)


def test_gate_params():
    sp = torch.nn.functional.softplus
    for bs, heads in ((1, 32), (3, 100)):   # 100 heads is not a multiple of the SIMD width
        for dtype in (torch.bfloat16, torch.float32):
            torch.manual_seed(6)
            a, b = (torch.randn(bs, heads, dtype=dtype) for _ in range(2))
            A_log, dt_bias = torch.randn(heads), torch.randn(heads)
            g, beta = gate_params_metal(*[t.to("mps") for t in (a, b, A_log, dt_bias)])
            want = -A_log.double().exp() * sp(a.double() + dt_bias.double())
            # measured max abs 1.0e-06 (g), 7.2e-08 (beta)
            torch.testing.assert_close(g.cpu().double(), want, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(beta.cpu().double(), b.double().sigmoid(), atol=1e-6,
                                       rtol=1e-6)
    # softplus must not overflow or underflow at either tail
    a = torch.tensor([[30.0, -30.0, 0.0, -25.0, 21.0]])
    z = torch.zeros(a.shape[1])
    g, _ = gate_params_metal(a.to("mps"), torch.zeros_like(a).to("mps"), z.to("mps"), z.to("mps"))
    # measured max abs 8.7e-19
    torch.testing.assert_close(g.cpu(), -z.exp() * sp(a), rtol=1e-6, atol=1e-15)


def _fused_case(b, hk, hv, dk, dv, dtype, seed):
    torch.manual_seed(seed)
    key_dim, val_dim = hk * dk, hv * dv
    z, b_raw, a_raw = torch.split(
        torch.randn(b, val_dim + 2 * hv, device="mps", dtype=dtype), [val_dim, hv, hv], dim=-1)
    return dict(mixed=torch.randn(b, 2 * key_dim + val_dim, device="mps", dtype=dtype),
                z=z, a=a_raw, b=b_raw, A_log=torch.randn(hv, device="mps"),
                dt_bias=torch.randn(hv, device="mps"),
                norm_weight=torch.randn(dv, device="mps", dtype=dtype),
                state=torch.randn(b + 2, hv, dk, dv, device="mps"),
                slots=torch.arange(b, device="mps", dtype=torch.int32))


def test_fused_decode():
    # Qwen3.6-35B's GDN shape, then Dv=96 (three simdgroup sweeps) with deep GQA sharing
    for b, hk, hv, dk, dv in ((1, 16, 32, 128, 128), (3, 2, 6, 64, 96)):
        for dtype in (torch.bfloat16, torch.float32):
            # sigmoid is Qwen3.8-Flash-Next's output gate; both variants must reproduce the
            # unfused chain bit for bit, not just to the end-to-end tolerance
            for act in ("silu", "sigmoid"):
                c = _fused_case(b, hk, hv, dk, dv, dtype, seed=b * 31 + dk)
                args = (c["A_log"], c["dt_bias"], c["norm_weight"], c["state"])
                assert fused_decode_supports(
                    c["mixed"], c["z"], c["a"], c["b"], *args, hk, hv, dk, dv, act)
                chain_state, fused_state = c["state"].clone(), c["state"].clone()
                qf, kf, vf = torch.split(c["mixed"], [hk * dk, hk * dk, hv * dv], dim=-1)
                core = gdn_decode_metal(
                    qf.reshape(1, b, hk, dk), kf.reshape(1, b, hk, dk), vf.reshape(1, b, hv, dv),
                    c["a"], c["b"], A_log=c["A_log"], dt_bias=c["dt_bias"],
                    state_source=chain_state, indices=c["slots"], scale=dk**-0.5)
                want = rms_norm_gated(
                    x=core.reshape(-1, dv), weight=c["norm_weight"], bias=None,
                    z=c["z"].reshape(-1, dv), eps=1e-6, activation=act).reshape(b, -1)
                got = gdn_decode_fused_metal(
                    c["mixed"], c["z"], c["a"], c["b"], A_log=c["A_log"], dt_bias=c["dt_bias"],
                    norm_weight=c["norm_weight"], norm_eps=1e-6, state_source=fused_state,
                    indices=c["slots"], scale=dk**-0.5, num_k_heads=hk, head_k_dim=dk,
                    activation=act)
                torch.mps.synchronize()
                assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()
                assert torch.equal(fused_state, chain_state)

    # outside its contract the fused path must decline rather than compute something else
    c = _fused_case(2, 4, 8, 128, 128, torch.bfloat16, seed=11)
    args = (c["A_log"], c["dt_bias"], c["norm_weight"], c["state"])
    quad = (c["mixed"], c["z"], c["a"], c["b"])
    assert not fused_decode_supports(*quad, *args, 4, 8, 288, 128)   # Dk over the tile
    assert not fused_decode_supports(*quad, *args, 4, 8, 100, 128)   # Dk % 32 != 0
    assert not fused_decode_supports(*quad, *args, 3, 8, 128, 128)   # mixed width mismatch
    other = torch.randn(2, 8, device="mps", dtype=torch.bfloat16)
    assert not fused_decode_supports(c["mixed"], c["z"], c["a"], other, *args, 4, 8, 128, 128)


def test_prefill_matches_decode():
    # gdn_prefill_metal splits a packed batch at cu_seqlens; each slice must reproduce the
    # solo recurrent run, and gdn_decode_metal must be gdn_recurrent_metal over one step
    hk, hv, total = 2, 4, 8   # two packed requests of 5 and 3 tokens
    torch.manual_seed(4)
    q, k = (torch.randn(1, total, hk, DK, device="mps") for _ in range(2))
    v = torch.randn(1, total, hv, DV, device="mps")
    g, beta = -torch.rand(1, total, hv, device="mps"), torch.rand(1, total, hv, device="mps")
    slots = torch.tensor([2, 4], dtype=torch.int32, device="mps")
    state = torch.zeros(6, hv, DK, DV, device="mps")
    out = gdn_prefill_metal(q, k, v, g, beta, state_source=state, indices=slots,
                            cu_seqlens=torch.tensor([0, 5, 8]), scale=DK**-0.5)
    assert out.shape == (total, hv, DV)
    for i, (lo, hi) in enumerate(((0, 5), (5, 8))):
        solo = gdn_recurrent_metal(
            q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi],
            state_source=torch.zeros(6, hv, DK, DV, device="mps"),
            indices=slots[i:i + 1], scale=DK**-0.5)[0]
        assert torch.equal(out[lo:hi], solo)


def test_prefill_per_chunk_state():
    """``return_h`` feeds the hybrid-radix track checkpoint: row ``boh[i] + c`` for a request's
    last chunk c is its state after c whole CHUNK_SIZE slices, which is what
    ``_write_track_snapshot`` copies into the pool. Turning it on must not move the output or
    the state the pool keeps."""
    hk, hv = 2, 4
    lens = [70, 64, 200, 1]        # ragged: past a chunk, exactly a chunk, deep, single token
    cu = list(itertools.accumulate(lens, initial=0))
    total = cu[-1]
    torch.manual_seed(7)
    q, k = (torch.randn(1, total, hk, DK, device="mps") for _ in range(2))
    v = torch.randn(1, total, hv, DV, device="mps")
    g, beta = -torch.rand(1, total, hv, device="mps"), torch.rand(1, total, hv, device="mps")
    slots = torch.tensor([1, 3, 5, 2], dtype=torch.int32, device="mps")
    # non-zero start: a continuation resumes from whatever the COW-restored slot holds
    start = torch.randn(8, hv, DK, DV, device="mps")
    args = dict(indices=slots, cu_seqlens=torch.tensor(cu), scale=DK**-0.5)

    off_state = start.clone()
    off = gdn_prefill_metal(q, k, v, g, beta, state_source=off_state, **args)
    on_state = start.clone()
    out, h = gdn_prefill_metal(q, k, v, g, beta, state_source=on_state, return_h=True, **args)
    assert torch.equal(out, off) and torch.equal(on_state, off_state)

    nchunks = [-(-n // CHUNK_SIZE) for n in lens]
    boh = list(itertools.accumulate(nchunks, initial=0))   # prepare_chunk_offsets, triton-free
    assert h.shape == (1, boh[-1], hv, DK, DV)
    for i, n in enumerate(lens):
        # the deepest row is the only one _build_track_metadata ever asks for
        c = (n - 1) // CHUNK_SIZE
        if not c:
            continue
        lo, hi = cu[i], cu[i] + c * CHUNK_SIZE
        up = lambda x: x[:, lo:hi].float().cpu().repeat_interleave(hv // hk, dim=2)  # noqa: E731
        _, ref = _oracle().recurrent_gated_delta_rule(
            up(q), up(k), v[:, lo:hi].float().cpu(), g[:, lo:hi].cpu(), beta[:, lo:hi].cpu(),
            initial_state=start[int(slots[i])][None].cpu())
        err = (h[0, boh[i] + c].cpu() - ref[0]).abs().max().item()
        assert err < 1e-5, f"request {i} chunk {c}: {err}"
