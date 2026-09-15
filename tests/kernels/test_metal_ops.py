"""Metal kernel set (rope, norms, gathers, router, sampling, fp8, conv, elementwise, mlp, qkv epilogue) against independent CPU/torch references."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel.metal import is_available
from freetoken.kernel.metal import conv, elementwise, norm, ops, rope, sampling
from freetoken.kernel.metal.fp8 import can_run as fp8_can_run
from freetoken.kernel.metal.fp8 import dequant_fp8, fp8_gemv, fp8_linear
from freetoken.kernel.metal.mlp import can_run as mlp_can_run
from freetoken.kernel.metal.mlp import dense_gemv, swiglu_gemv
from freetoken.kernel.metal.qkv_epilogue import qkv_epilogue_metal, supports as qkv_supports
from tests.kernels._refs import _close

pytestmark = pytest.mark.skipif(not is_available(), reason="needs a torch build with MPS")

# The first of each pair divides every shipped tile and the 32-lane SIMD width; the second
# divides none of them, so the K-loop remainder and the tail threadgroup both run.
ROPE_SHAPES = ((5, 16, 2, 256, 64), (7, 2, 1, 128, 20))
NORM_SHAPES = ((1, 2048), (7, 300))
GEMV_SHAPES = ((1, 2048, 512), (3, 126, 200))
CONV_SHAPES = ((1, 1536, 4), (2, 300, 2))


def _cos_sin(max_pos, rotary_dim, base=1e7):
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    freqs = torch.einsum("i,j->ij", torch.arange(max_pos, dtype=torch.float), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _rope_ref(src, nnz, heads, head_size, rot, cos, sin, is_neox):
    half = rot // 2
    want = src.float().view(nnz, heads, head_size).clone()
    i0 = torch.arange(half) * (1 if is_neox else 2)
    i1 = i0 + (half if is_neox else 1)
    x0, x1 = want[:, :, i0], want[:, :, i1]
    want[:, :, i0], want[:, :, i1] = x0 * cos - x1 * sin, x1 * cos + x0 * sin
    return want


def test_rope():
    for nnz, hq, hk, head_size, rot in ROPE_SHAPES:
        for is_neox in (True, False):
            cache, half = _cos_sin(64, rot), rot // 2
            torch.manual_seed(0)
            q, k = (torch.randn(nnz, h * head_size).bfloat16() for h in (hq, hk))
            pos = torch.randint(0, 64, (nnz,), dtype=torch.int32)
            qm, km = q.to("mps"), k.to("mps")
            rope.apply_rope_metal(pos.to("mps"), qm, km, head_size, cache.to("mps"), is_neox)
            cs = cache[pos.long()][:, None]
            cos, sin = cs[..., :half], cs[..., half:]
            for got, src, heads in ((qm, q, hq), (km, k, hk)):
                view = got.cpu().view(nnz, heads, head_size)
                want = _rope_ref(src, nnz, heads, head_size, rot, cos, sin, is_neox)
                _close(view, want.bfloat16(), 8e-3, 8e-3)
                assert torch.equal(view[:, :, rot:],
                                   src.view(nnz, heads, head_size)[:, :, rot:])
            qc, kc = q.clone(), k.clone()
            ops.apply_rope_with_cos_sin_cache_inplace(pos, qc, kc, head_size, cache, is_neox)
            assert torch.equal(qm.cpu(), qc) and torch.equal(km.cpu(), kc)


def _rmsnorm_f32(x, w, eps, plus_one=False):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (w.float() + plus_one)


def test_norm():
    for m, n in NORM_SHAPES:
        for plus_one in (False, True):
            torch.manual_seed(0)
            x, z = torch.randn(2, m, n, dtype=torch.bfloat16)
            w = torch.randn(n, dtype=torch.bfloat16)
            b = torch.randn(n, dtype=torch.bfloat16) if plus_one else None
            _close(norm.rmsnorm_metal(x.to("mps"), w.to("mps"), 1e-6, plus_one=plus_one),
                   _rmsnorm_f32(x, w, 1e-6, plus_one), 1e-3, 4e-3)
            xm, rm = x.to("mps"), z.to("mps").clone()
            norm.fused_add_rmsnorm_metal(xm, rm, w.to("mps"), 1e-6)
            resid = (x.float() + z.float()).bfloat16()
            assert torch.equal(rm.cpu(), resid)
            _close(xm, _rmsnorm_f32(resid, w, 1e-6), 1e-3, 4e-3)
            gated = norm.rms_norm_gated_metal(
                x.to("mps"), w.to("mps"), None if b is None else b.to("mps"), z.to("mps"), 1e-6
            )
            want = _rmsnorm_f32(x, w, 1e-6) + (0.0 if b is None else b.float())
            _close(gated, want * F.silu(z.float()), 2e-3, 8e-3)
            sq = x.float().pow(2).sum(-1, keepdim=True)
            _close(norm.l2norm_metal(x.to("mps"), 1e-6, n**-0.5),
                   x.float() * torch.rsqrt(sq + 1e-6) * n**-0.5, 8e-3, 8e-3)
    empty, ones = torch.empty(0, 128, device="mps").bfloat16(), torch.ones(128, device="mps")
    assert norm.rmsnorm_metal(empty, ones.bfloat16(), 1e-6).shape == (0, 128)   # zero-row return


def test_elementwise():
    torch.manual_seed(1)
    for m, n in NORM_SHAPES:
        for broadcast in (False, True):
            x = torch.randn(m, 2 * n, dtype=torch.bfloat16).to("mps")
            gate, up = x.cpu().float().chunk(2, dim=-1)
            _close(elementwise.silu_and_mul_metal(x), F.silu(gate) * up, 8e-3, 8e-3)
            y, g = x[:, :n].contiguous(), torch.randn(
                m, 1 if broadcast else n, dtype=torch.bfloat16).to("mps")
            add = torch.randn(m, n, dtype=torch.bfloat16).to("mps") if broadcast else None
            got = elementwise.sigmoid_gate_mul_metal(y, g, add)
            want = y.cpu().float() * torch.sigmoid(g.cpu().float())
            _close(got, want if add is None else want + add.cpu().float(), 8e-3, 8e-3)
    empty = torch.empty(0, 64, dtype=torch.bfloat16, device="mps")   # zero-row early return
    assert (elementwise.silu_and_mul_metal(empty).shape,
            elementwise.sigmoid_gate_mul_metal(empty, empty).shape) == ((0, 32), (0, 64))


def test_conv():
    for b, c, width in CONV_SHAPES:
        torch.manual_seed(0)
        slots = b + 4
        x, weight = torch.randn(b, c).bfloat16(), torch.randn(c, width).bfloat16()
        state = torch.randn(slots, c, width - 1, dtype=torch.bfloat16)
        idx, state_m = torch.randperm(slots)[:b].to(torch.int32), state.to("mps")
        got = conv.causal_conv1d_decode_metal(x.to("mps"), state_m, weight.to("mps"), idx.to("mps"))
        window = torch.cat([state[idx.long()].double(), x.double().unsqueeze(-1)], dim=-1)
        want = F.silu((window * weight.double().unsqueeze(0)).sum(-1))
        torch.testing.assert_close(got.cpu().double(), want, rtol=8e-3, atol=2e-2)
        assert torch.equal(state_m.cpu()[idx.long()].double(), window[..., 1:])
        idle = [i for i in range(slots) if i not in idx.tolist()]
        assert torch.equal(state_m.cpu()[idle], state[idle])   # unaddressed slots stay put
        # the kernel sums in fp32 and rounds once; ops.causal_conv1d_decode reduces in bf16
        loose = ops.causal_conv1d_decode(x, state.clone(), weight, idx).double()
        assert (got.cpu().double() - want).abs().max() <= (loose - want).abs().max()
    # varlen (prefill): one pass over 9 tokens must equal two chunks carrying state across
    torch.manual_seed(6)
    x, w = torch.randn(8, 9, device="mps"), torch.randn(8, 4, device="mps")
    one, split = torch.zeros(2, 2, 8, 3, device="mps")
    varlen, T = ops.causal_conv1d_varlen, torch.tensor
    kw = dict(cache_indices=T([1], dtype=torch.int32))   # one request, slot 1
    whole = varlen(x.clone(), w, one, T([0, 9]), has_initial_state=T([False]), **kw)
    a = varlen(x[:, :4].clone(), w, split, T([0, 4]), has_initial_state=T([False]), **kw)
    b = varlen(x[:, 4:].clone(), w, split, T([0, 5]), has_initial_state=T([True]), **kw)
    torch.testing.assert_close(whole, torch.cat([a, b], dim=-1))
    torch.testing.assert_close(one, split)   # and the state each carried away


def test_mlp():
    for m, k, n in GEMV_SHAPES:
        for dtype in (torch.float32, torch.bfloat16):
            g, tol = torch.Generator().manual_seed(41), 1e-4 if dtype is torch.float32 else 3e-2
            x, w = torch.randn(m, k, generator=g), torch.randn(2 * n, k, generator=g) * 0.05
            _close(dense_gemv(x.to(dtype).to("mps"), w[:n].to(dtype).to("mps")), F.linear(x, w[:n]),
                   tol, tol)
            _close(swiglu_gemv(x.to("mps"), w.to("mps")),
                   F.silu(F.linear(x, w[:n])) * F.linear(x, w[n:]), 1e-4, 1e-4)
    # the fused prologue must leave the same residual and normed x as the standalone launch
    g = torch.Generator().manual_seed(81)
    x, r = (torch.randn(3, 256, generator=g).to("mps") for _ in range(2))
    nw = (torch.rand(256, generator=g) + 0.5).to("mps")
    w = (torch.randn(64, 256, generator=g) * 0.05).to("mps")
    x_ref, r_ref, r_out = x.clone(), r.clone(), torch.empty_like(r)
    norm.fused_add_rmsnorm_metal(x_ref, r_ref, nw, 1e-6)
    got, xn = dense_gemv(x, w, norm=(r, r_out, nw, 1e-6))
    assert torch.equal(r_out.cpu(), r_ref.cpu())
    _close(xn, x_ref, 1e-5, 1e-5)
    _close(got, dense_gemv(x_ref, w), 1e-5, 1e-5)
    # K past the 32 KiB threadgroup tile: the caller falls back to F.linear
    z = torch.zeros
    assert not mlp_can_run(z(1, 16384, device="mps"), z(4, 16384, device="mps"))
    assert mlp_can_run(z(1, 64, device="mps"), z(4, 64, device="mps"))
    assert not mlp_can_run(z(1, 64), z(4, 64))


def _fp8_weight(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    w8 = (torch.randn(n, k, generator=g) * 0.3).to(torch.float8_e4m3fn)
    return w8, w8.view(torch.uint8)


def test_fp8_gemv():
    for m, k, n in GEMV_SHAPES:
        for dtype in (torch.float32, torch.bfloat16):
            g, tol = torch.Generator().manual_seed(7), 1e-4 if dtype is torch.float32 else 3e-2
            x, (w8, wu8) = torch.randn(m, k, generator=g), _fp8_weight(n, k, seed=41)
            scale = (torch.rand(n, generator=g) * 0.05 + 0.01).float()
            want = F.linear(x.float(), w8.float() * scale[:, None])
            got = fp8_gemv(x.to(dtype).to("mps"), wu8.to("mps"), scale.to("mps"))
            _close(got, want, tol * max(1.0, want.abs().max().item()), tol)
            assert torch.equal(dequant_fp8(wu8.to("mps"), scale.to("mps"), dtype=torch.float32).cpu(),
                               w8.float() * scale[:, None])
    g = torch.Generator().manual_seed(31)   # fp8_linear keeps the leading dims, adds the bias
    x, (w8, wu8) = torch.randn(2, 3, 64, generator=g), _fp8_weight(32, 64, seed=37)
    scale, bias = torch.full((32,), 0.037), torch.randn(32, generator=g)
    got = fp8_linear(x.to("mps"), wu8.to("mps"), scale.to("mps"), bias.to("mps"))
    want = F.linear(x.reshape(-1, 64).float(), w8.float() * scale[:, None]).reshape(2, 3, 32) + bias
    _close(got, want, 1e-4, 1e-4)
    w, s = torch.zeros(8, 16, dtype=torch.uint8, device="mps"), torch.ones(8, device="mps")
    assert fp8_can_run(torch.zeros(1, 16, device="mps"), w, s)
    assert not fp8_can_run(torch.zeros(1, 8, device="mps"), w, s)   # K mismatch
    assert not fp8_can_run(torch.zeros(1, 16, device="mps"), w.float(), s)   # not packed u8

    # the three optional epilogues, each against the unfused launch it folds in
    m, k, n, nc, ck = 2, 256, 96, 40, 4
    g = torch.Generator().manual_seed(78)
    x = torch.randn(m, k, generator=g).to(torch.bfloat16).to("mps")
    w = torch.randint(0, 120, (n, k), dtype=torch.uint8, generator=g).to("mps")
    s = (torch.rand(n, generator=g) * 0.01 + 0.001).to("mps")
    # nx=40 leaves a partial tail threadgroup on the unquantized extra-weight block
    xw = (torch.randn(40, k, generator=g) * 0.02).to(torch.bfloat16).to("mps")
    taps = torch.randn(nc, ck, generator=g).to(torch.bfloat16).to("mps")
    state, plain = torch.randn(6, nc, ck - 1, generator=g).to("mps"), fp8_gemv(x, w, s)
    slots = torch.tensor([4, 1], dtype=torch.int32, device="mps")
    out, tail = fp8_gemv(x, w, s, extra_weight=xw)
    want_tail = F.linear(x.float().cpu(), xw.float().cpu())
    assert torch.equal(out.cpu(), plain.cpu()) and tail.shape == (m, 40)
    assert ((tail.float().cpu() - want_tail).abs().max() / want_tail.abs().max()).item() < 1e-2
    st_ref, st, cdm = state.clone(), state.clone(), conv.causal_conv1d_decode_metal
    want_conv, fused = cdm(plain[:, :nc].contiguous(), st_ref, taps, slots), fp8_gemv(
        x, w, s, conv=(taps, st, slots))
    assert torch.equal(fused[:, :nc].cpu(), want_conv.cpu()) and torch.equal(st.cpu(), st_ref.cpu())
    assert torch.equal(fused[:, nc:].cpu(), plain[:, nc:].cpu())
    hidden, residual = (torch.randn(m, k, generator=g).bfloat16().to("mps") for _ in range(2))
    nw = (torch.rand(k, generator=g) + 0.5).to(torch.bfloat16).to("mps")
    x_ref, r_ref, r_out = hidden.clone(), residual.clone(), torch.empty_like(residual)
    norm.fused_add_rmsnorm_metal(x_ref, r_ref, nw, 1e-6)
    got, want = fp8_gemv(hidden, w, s, norm=(hidden, residual, r_out, nw, 1e-6)), fp8_gemv(x_ref, w, s)
    assert torch.equal(r_out.cpu(), r_ref.cpu())
    _close(got, want, 1e-2 * want.float().abs().max().item(), 1e-2)


def _qkv_chain(qkv, num_q, num_kv, d, qw, kw, eps, cs, pos, kc, vc, loc):
    qg, k, v = torch.split(qkv, [num_q * 2 * d, num_kv * d, num_kv * d], dim=-1)
    gate = (qg := qg.view(-1, num_q, 2 * d))[..., d:].reshape(-1, num_q * d)
    q = ops.rmsnorm(qg[..., :d].contiguous().reshape(-1, d), qw, eps).reshape(-1, num_q * d)
    k = ops.rmsnorm(k.reshape(-1, d), kw, eps).reshape(-1, num_kv * d)
    ops.apply_rope_with_cos_sin_cache_inplace(pos, q, k, d, cs, True)
    ops.store_cache(kc, vc, loc, k, v.contiguous())
    return q.view(-1, num_q, d), gate


def test_qkv_epilogue():
    # Qwen3.6's geometry, then a head size not 32-aligned with 3 kv heads
    for n, num_q, num_kv, d, rot in ((1, 16, 2, 256, 64), (3, 6, 3, 96, 16)):
        for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
            g = torch.Generator().manual_seed(n * 7 + d)
            qkv = torch.randn(n, (2 * num_q + 2 * num_kv) * d, generator=g).to(dtype).to("mps")
            qw, kw = ((torch.rand(d, generator=g) + 0.5).to(dtype).to("mps") for _ in "01")
            cs = _cos_sin(64, rot, base=10000).to("mps")
            pos = torch.randint(0, 64, (n,), generator=g).to(torch.int32).to("mps")
            loc = torch.randperm(40, generator=g)[:n].to(torch.int32).to("mps")
            kc, vc, kc2, vc2 = torch.zeros(4, 40, num_kv, d, dtype=dtype, device="mps")
            # the exact keys qwen3_5_moe/attention.py::_project_fused hands both entry points
            sig = dict(qkv=qkv, num_q=num_q, num_kv=num_kv, head_dim=d, q_weight=qw, k_weight=kw,
                       cos_sin_cache=cs, positions=pos, k_cache=kc, v_cache=vc, out_loc=loc)
            assert qkv_supports(**sig)
            q, gate = qkv_epilogue_metal(**sig, eps=1e-6, is_neox=True)
            want_q, want_gate = _qkv_chain(qkv, num_q, num_kv, d, qw, kw, 1e-6, cs, pos,
                                           kc2, vc2, loc)
            assert torch.equal(gate.cpu(), want_gate.cpu())
            assert torch.equal(vc.cpu(), vc2.cpu())
            ref = torch.cat([want_q.float().flatten(), kc2.float().flatten()])
            err = torch.cat([q.float().flatten(), kc.float().flatten()]) - ref
            assert (err.abs().max() / ref.abs().max()).item() < tol
            mask = torch.ones(40, dtype=torch.bool).index_fill_(0, loc.cpu().long(), False)
            assert kc.cpu()[mask].abs().sum() == 0 and vc.cpu()[mask].abs().sum() == 0
    assert not qkv_supports(**{**sig, "k_cache": kc.float()})
    assert not qkv_supports(**{**sig, "k_cache": kc[:, :1]})


def test_router():
    from freetoken.kernel.metal.router import _threads, fused_topk_softmax_metal
    from freetoken.moe.fused import _torch_fused_topk

    # E=256 takes the simdgroup path, E=1024 the threadgroup-scratch one; 129 rows is odd
    for experts, rows in ((256, 129), (1024, 6)):
        for renormalize in (True, False):
            torch.manual_seed(11)
            logits = torch.randn(rows, experts)
            limit = torch.tensor(rows - 2, dtype=torch.int32)
            want_w, want_i = _torch_fused_topk(logits.float(), 8, renormalize, limit)
            got_w, got_i = fused_topk_softmax_metal(
                logits.to("mps"), 8, renormalize, limit.to("mps")
            )
            assert torch.equal(got_i.cpu(), want_i) and (got_i.cpu()[rows - 2:] == -1).all()
            assert (got_w.cpu() - want_w).abs().max() < 1e-5
    assert (_threads(256), _threads(1024)) == (32, 256)
    tied = torch.tensor([[5.0, 3.0, 5.0, 1.0, 0.0, -1.0, -2.0, -3.0]], dtype=torch.bfloat16)
    _, got_i = fused_topk_softmax_metal(tied.to("mps"), 3, True, None)
    # ties break toward the lower expert index, like torch.topk
    assert torch.equal(got_i.cpu()[0], torch.topk(tied[0], 3).indices.to(torch.int32))
    # was flaky ~9/100: a broadcast read of red_v[0] raced a slower simdgroup's write
    torch.manual_seed(11)
    _, first = fused_topk_softmax_metal(x := torch.randn(129, 256).to("mps"), 8, True, None)
    for _ in range(100):
        _, again = fused_topk_softmax_metal(x, 8, True, None)
        assert torch.equal(again.cpu(), first.cpu())
    logits = torch.randn(2, 64, device="mps")   # topk over the per-lane budget: torch answers
    _, ids = ops.fused_topk_softmax(logits, topk=40, renormalize=True)
    want = torch.topk(torch.softmax(logits.float(), -1), 40, dim=-1).indices
    assert torch.equal(ids.cpu(), want.cpu().to(torch.int32))


def test_sampling():
    V, K_MAX = 1499, 20   # V is a multiple of no tile and of no k
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 200), -1).to("mps")
    top_k = torch.tensor([1, 3, 5, 2], dtype=torch.int32, device="mps")
    allowed = [set(torch.topk(probs[i], int(top_k[i])).indices.tolist()) for i in range(4)]
    for _ in range(20):
        drawn = sampling.top_k_sampling_from_probs(probs, top_k)
        assert all(int(drawn[i]) in allowed[i] for i in range(4))
    ones = torch.ones(4, dtype=torch.int32, device="mps")   # k=1 draws the argmax every time
    assert torch.equal(sampling.top_k_sampling_from_probs(probs, ones).long(), probs.argmax(-1))
    p4 = torch.tensor([[0.5, 0.3, 0.15, 0.05]], device="mps")   # top_p keeps the token crossing p
    _close(sampling.top_p_renorm_probs(p4, torch.tensor([0.7], device="mps")),
           torch.tensor([[0.625, 0.375, 0.0, 0.0]]), 1e-6, 1e-6)
    temps, logits2 = torch.tensor([0.5, 2.0], device="mps"), torch.randn(2, 16, device="mps")
    _close(sampling.softmax(logits2, temps),
           torch.softmax(logits2.float().cpu() / temps.cpu().reshape(-1, 1), -1), 1e-6, 1e-6)
    # the candidate-selection fast path must draw from the same support as the probs path
    torch.manual_seed(0)
    logits, ks = torch.randn(3, V) * 2, torch.tensor([1, 5, 20], dtype=torch.int32)
    keep = [set(torch.topk(logits[i], int(ks[i])).indices.tolist()) for i in range(3)]
    seen, _ = [set(), set(), set()], torch.manual_seed(3)
    for _ in range(2000):
        out = sampling.top_k_top_p_sample_from_logits(logits, torch.ones(3), ks, None, K_MAX)
        for i in range(3):
            seen[i].add(int(out[i]))
    assert seen[0] == {int(logits.argmax(-1)[0])}
    assert all(seen[i] <= keep[i] for i in range(3))
    # engine/sample.py must reach this module without a flashinfer import
    from freetoken.engine.sample import BatchSamplingArgs, Sampler

    dev = torch.randn(3, 128, device="mps")
    sampler = Sampler(device=torch.device("mps"), vocab_size=128)
    assert torch.equal(sampler.sample(dev, BatchSamplingArgs(temperatures=None)), dev.argmax(-1))


def test_gather_and_store():
    from freetoken.kernel.store import store_cache

    weights = torch.arange(40, dtype=torch.float32).reshape(10, 4).to("mps")
    ids = torch.tensor([0, 3, 11, 7], device="mps")   # 11 is outside this rank's shard
    assert torch.equal(ops.indexing(weights, ids[:2]), weights[[0, 3]])
    got = ops.indexing(weights, ids, vocab_range=(2, 10)).cpu()
    want = torch.stack([torch.zeros(4), weights[1].cpu(), weights[9].cpu(), weights[5].cpu()])
    assert torch.equal(got, want)
    k_cache, v_cache = torch.zeros(2, 10, 2, 8, device="mps")
    k, v = torch.randn(2, 3, 2, 8, generator=torch.Generator("mps").manual_seed(4), device="mps")
    store_cache(k_cache, v_cache, torch.tensor([7, 1, 4], dtype=torch.int32, device="mps"), k, v)
    assert torch.equal(k_cache[[7, 1, 4]], k) and torch.equal(v_cache[[7, 1, 4]], v)
    assert k_cache[[0, 2, 3, 5, 6, 8, 9]].abs().sum() == 0
