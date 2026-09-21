"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points equals one-shot prefill (the dual-source compress);
(c) a captured decode replay equals the eager decode step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import DEVICE, Fixture, requires_cuda, requires_gpu, requires_mps, parsed_config, selection_spy

QSA_LAYER = 3


def _inputs(fixture: Fixture, lengths, extra: int = 0, seed: int = 11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(
            n + extra, fixture.config.hidden_size, device=fixture.device,
            dtype=fixture.dtype, generator=generator,
        )
        * 0.5
        for n in lengths
    ]


def _assert_selection_is_causal_prefix(indices: torch.Tensor, positions: torch.Tensor) -> None:
    for row, position in enumerate(positions.tolist()):
        selected = indices[row][indices[row] >= 0]
        assert torch.equal(
            selected.sort().values,
            torch.arange(position + 1, dtype=selected.dtype, device=selected.device),
        ), f"row {row} (position {position}) did not select its whole causal prefix"


@requires_gpu
def test_prefill_is_dense_below_the_budget(monkeypatch):
    """bs=3 ragged prefill, longest request exactly at budget + ratio - 1."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths = [2051, 1000, 137]
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    batch = fixture.batch(reqs, "prefill")
    got = attn.forward(x, batch)  # whatever route the backend picks for this batch

    # Metal skips the selection on a dense batch, so force the sparse route (it rewrites the same K/V rows)
    seen = selection_spy(monkeypatch, fixture.backend)
    if hasattr(fixture.backend, "_qsa_is_dense"):
        monkeypatch.setattr(type(fixture.backend), "_qsa_is_dense", lambda self, md: False)
    sparse = attn.forward(x, batch)
    _assert_selection_is_causal_prefix(seen["indices"], batch.positions)

    fixture.ctx.attn_backend = _dense_oracle(fixture)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sparse.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config,
        num_slots=fixture.num_req_slots,
        max_len=4096,
        device=fixture.device,
        dtype=fixture.dtype,
    )


@requires_gpu
def test_decode_is_dense_below_the_budget(monkeypatch):
    """Prefill then five decode steps, sparse path vs the fp32 dense oracle."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 5
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)

    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)

    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [
        torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)
    ]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        seen.pop("indices", None)
        got = attn.forward(x, batch)
        # Metal skips the selection on a dense prefill; test_prefill_is_dense_below_the_budget pins it
        if step or not hasattr(fixture.backend, "_qsa_is_dense"):
            _assert_selection_is_causal_prefix(seen["indices"], batch.positions)
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_gpu
def test_flashinfer_dense_matches_the_sparse_path():
    """The engine's dense FULL backend over the same pool, as an independent oracle."""
    pytest.importorskip("flashinfer")
    from freetoken.attention.fi import FlashInferBackend

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 500
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)
    got = attn.forward(x, fixture.batch([req], "prefill"))

    dense = FlashInferBackend(config)
    fixture.ctx.attn_backend = SimpleNamespace(
        qsa_forward=lambda q, k, v, index, layer_id, batch: dense.forward(
            q, k, v, layer_id, batch
        )
    )
    batch = fixture.batch([req], "prefill")
    dense.prepare_metadata(batch)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_gpu
@pytest.mark.parametrize("cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"])
def test_chunked_prefill_matches_one_shot(cut: int):
    """Cut points that are not multiples of index_ratio exercise the dual-source compress."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]

    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    assert torch.equal(got, one_shot[cut:])


@requires_cuda
def test_decode_graph_replay_matches_eager():
    """CUDA graphs only; test_decode_tape_replay_matches_eager_across_the_budget is the MPS twin."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=256)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs = len(lengths)
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(
        torch.cat([row[:n] for row, n in zip(inputs, lengths)]),
        fixture.batch(reqs, "prefill"),
    )

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs])
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=2, extend_len=1
    )
    static = {
        "x": torch.zeros(bs, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(bs, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase="decode", size=bs, padded_size=bs,
        is_prefill=False, is_decode=True, positions=static["positions"],
        get_attn_positions=lambda: static["positions"],
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()

    for step in range(steps):
        for req in reqs:
            fixture.step(req)
        x = torch.stack([row[n + step] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, "decode")
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        # replay must stage into the captured buffers, never reallocate them
        md = batch.attn_metadata
        assert md.block_table.data_ptr() == fixture.backend._graph["block_table"].data_ptr()
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, "decode"))
        assert torch.equal(replayed, eager), f"graph replay diverged at decode step {step}"


@requires_gpu
def test_row_chunked_scoring_matches_one_chunk(monkeypatch):
    """The scoring workspace bound splits long prefills into row chunks."""
    import freetoken.attention.metal as metal
    import freetoken.attention.qsa_sparse as qsa_sparse

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 600
    x = _inputs(fixture, [length])[0]
    whole = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))

    columns = fixture.page_table.shape[1] // config.qwen4_args.index_ratio
    module = metal if DEVICE == "mps" else qsa_sparse
    monkeypatch.setattr(module, "_LOGITS_WORKSPACE_BYTES", 64 * columns * 4)
    chunked = attn.forward(x, fixture.batch([fixture.req(1, 0, length)], "prefill"))
    assert torch.equal(chunked, whole)


@requires_gpu
def test_two_qsa_layers_keep_separate_slab_slots(monkeypatch):
    """Both QSA layers of one forward must hit their own slab slot and ring slice."""
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    layers = [fixture.layer(layer_id, seed=layer_id) for layer_id in (3, 7)]
    oracle = _dense_oracle(fixture)
    lengths, steps = [200, 71], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    xs = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    xs += [torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)]
    for step, x in enumerate(xs):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        for attn in layers:
            fixture.ctx.attn_backend = fixture.backend
            got = attn.forward(x, batch)
            fixture.ctx.attn_backend = oracle
            reference = attn.forward(x, batch)
            torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)

    slab = fixture.pool.cmp_k_cache
    assert not torch.equal(slab(0), slab(1))


def _indexer_inputs(config, rows: int, device, dtype, generator):
    from freetoken.models.qwen4_exp.attention import QSAIndexerInputs

    args = config.qwen4_args
    shape = (rows, args.index_n_heads, args.index_head_dim)
    return QSAIndexerInputs(
        q=torch.randn(shape, device=device, dtype=dtype, generator=generator),
        k=torch.randn(rows, args.index_head_dim, device=device, dtype=dtype,
                      generator=generator),
        q_norm_weight=torch.randn(args.index_head_dim, device=device, dtype=dtype,
                                  generator=generator) * 0.1,
        k_norm_weight=torch.randn(args.index_head_dim, device=device, dtype=dtype,
                                  generator=generator) * 0.1,
        eps=config.rms_norm_eps,
    )


@requires_mps
@torch.inference_mode()  # DecodeTape.record allocates inference tensors, as the engine does
def test_decode_tape_replay_matches_eager_across_the_budget():
    """Replay one recorded decode step from kv_len 2040 past 2051: a tape that read kv_len on the host stays dense."""
    from freetoken.engine.mps_tape import DecodeTape
    from freetoken.models.qwen4_exp.attention import QSAIndexerInputs

    config = parsed_config()
    args = config.qwen4_args
    fixture, bs, steps = Fixture(config, num_pages=160), 1, 32
    backend, device, dtype = fixture.backend, fixture.device, fixture.dtype
    heads, dim, kv_dim = config.num_qo_heads, config.head_dim, config.num_kv_heads * config.head_dim
    generator = torch.Generator(device=device).manual_seed(19)

    def step_inputs(rows: int):
        q = torch.randn(rows, heads, dim, device=device, dtype=dtype, generator=generator)
        k = torch.randn(rows, kv_dim, device=device, dtype=dtype, generator=generator)
        v = torch.randn(rows, kv_dim, device=device, dtype=dtype, generator=generator)
        return q, k, v

    # Prefill to just under the dense threshold, so the sweep below crosses it.
    prefill_len = 2040
    req = fixture.req(0, 0, prefill_len)
    index = _indexer_inputs(config, prefill_len, device, dtype, generator)
    backend.qsa_forward(*step_inputs(prefill_len), index, QSA_LAYER,
                        fixture.batch([req], "prefill"))

    backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs])
    dummy_slot = fixture.num_req_slots - 1
    static = {
        "q": torch.zeros(bs, heads, dim, device=device, dtype=dtype),
        "k": torch.zeros(bs, kv_dim, device=device, dtype=dtype),
        "v": torch.zeros(bs, kv_dim, device=device, dtype=dtype),
        "iq": torch.zeros(bs, args.index_n_heads, args.index_head_dim, device=device,
                          dtype=dtype),
        "ik": torch.zeros(bs, args.index_head_dim, device=device, dtype=dtype),
        "out_loc": torch.full((bs,), int(fixture.page_table[dummy_slot, 0]),
                              dtype=torch.int32, device=device),
    }
    static_index = QSAIndexerInputs(
        q=static["iq"], k=static["ik"], q_norm_weight=index.q_norm_weight,
        k_norm_weight=index.k_norm_weight, eps=index.eps,
    )
    dummy = SimpleNamespace(table_idx=dummy_slot, cached_len=1, device_len=2, extend_len=1)
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase="decode", size=bs, padded_size=bs,
        is_prefill=False, is_decode=True, out_loc=static["out_loc"], attn_metadata=None,
    )
    backend.prepare_for_capture(capture_batch)
    captured = backend.qsa_forward(
        static["q"], static["k"], static["v"], static_index, QSA_LAYER, capture_batch
    ).clone()  # warm: shaders compiled, scratch sized

    def one_step():
        captured.copy_(
            backend.qsa_forward(
                static["q"], static["k"], static["v"], static_index, QSA_LAYER, capture_batch
            )
        )

    tape = DecodeTape.record(one_step)
    assert len(tape), "the recorded decode forward issued no launches"

    # The replayed selection, read off the scratch the expansion kernel writes. If
    # torch.topk(..., out=) were NOT recorded, every replay would expand the block set the
    # capture batch chose and this would be one constant.
    sel_key = ("sel", (backend._qsa_width,), torch.int32)
    picked = []
    for _ in range(steps):
        fixture.step(req)
        q, k, v = step_inputs(bs)
        one = _indexer_inputs(config, bs, device, dtype, generator)
        live = QSAIndexerInputs(
            q=one.q, k=one.k, q_norm_weight=index.q_norm_weight,
            k_norm_weight=index.k_norm_weight, eps=index.eps,
        )
        batch = fixture.batch([req], "decode")
        backend.prepare_for_replay(batch)
        for name, value in (("q", q), ("k", k), ("v", v), ("iq", one.q), ("ik", one.k)):
            static[name].copy_(value)
        static["out_loc"].copy_(batch.out_loc)
        tape.replay()
        replayed = captured.clone()
        # the expanded blocks only, before the open group's causal tail
        picked.append(backend._qsa_buffers[sel_key][0, : args.index_budget].clone())
        eager = backend.qsa_forward(q, k, v, live, QSA_LAYER, batch)
        assert torch.equal(replayed, eager), f"replay diverged at kv_len {req.device_len}"
    assert req.device_len > args.index_budget + args.index_ratio - 1, "sweep never crossed"
    assert any(not torch.equal(picked[0], row) for row in picked[1:]), (
        "every replay expanded the same blocks: torch.topk(out=) is not being re-run"
    )


@requires_mps
def test_a_chunked_prefill_does_not_grow_the_qsa_scratch():
    """Chunks that keep raising the block count must not mint a QSA workspace per column count."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    backend = fixture.backend
    chunk, chunks = 1024, 24  # block counts 256 .. 6144, crossing six powers of two
    generator = torch.Generator(device=fixture.device).manual_seed(17)
    x = torch.randn(
        chunk * chunks, config.hidden_size, device=fixture.device, dtype=fixture.dtype,
        generator=generator,
    ) * 0.5

    attn.forward(x[:chunk], fixture.batch([fixture.req(0, 0, chunk)], "prefill"))
    torch.mps.synchronize()
    before = torch.mps.driver_allocated_memory()
    for step in range(1, chunks):
        lo, hi = step * chunk, (step + 1) * chunk
        attn.forward(x[lo:hi], fixture.batch([fixture.req(0, lo, hi)], "prefill"))
    torch.mps.synchronize()
    grew = torch.mps.driver_allocated_memory() - before

    # The K/V the chunks themselves wrote dominates; the scratch must not add hundreds of
    # MiB on top, which is what one logits workspace per distinct column count would do.
    assert grew < 512 << 20, f"driver pool grew {grew / (1 << 20):.1f} MiB over {chunks} chunks"
