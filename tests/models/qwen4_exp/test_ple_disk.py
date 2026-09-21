"""Disk PLE backend, module level: store byte fidelity, on-disk layouts and errors, table vs GPU oracle, and the CUDA-graph sync protocol."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.ple import GpuResidentTable, NGramEmbedding

from .common import DEVICE, EOS, hash_constants, requires_cuda, requires_gpu, requires_mps, toy_hf_config
from .test_ple import _meta

_KEY_PREFIX = "model.layers.1.ple.ple_embedding.ngram_embedding"
# MPS has no fp8 cast, so the oracle reads the table where the cast is defined
_ORACLE_DEVICE = DEVICE if DEVICE == "cuda" else "cpu"


def _store_ext():
    """The C++ row store: CUDA/Linux only, so the Metal backend's tests must not import it."""
    return pytest.importorskip("freetoken.kernel._ple_store")


def _embedding() -> NGramEmbedding:
    args = parse_config(toy_hf_config()).qwen4_args
    emb = NGramEmbedding(args)
    multipliers, sizes, offsets = hash_constants(args)
    emb.layer_multipliers.copy_(multipliers)
    emb.ngram_heads_vocab_sizes.copy_(sizes)
    emb.ngram_heads_offsets.copy_(offsets)
    return emb


def _bitwise_equal(got: torch.Tensor, want: torch.Tensor) -> bool:
    # random table bytes include fp8 NaN encodings, and NaN != NaN under torch.equal
    as_int = {2: torch.int16, 4: torch.int32}[got.dtype.itemsize]
    return torch.equal(got.view(as_int), want.view(as_int))


def _make_store(tmp_path, *, write=True, use_io_uring=True):
    _ple_store = _store_ext()
    args = parse_config(toy_hf_config()).qwen4_args
    multipliers, sizes, offsets = hash_constants(args)
    total_rows = int(offsets[-1] + sizes[-1])
    cols = args.ngram_head_dim
    gen = torch.Generator().manual_seed(5)
    table = torch.randint(0, 256, (total_rows, cols), dtype=torch.uint8, generator=gen)
    path = tmp_path / "ple-table.bin"
    if write:
        path.write_bytes(table.numpy().tobytes())
    store = _ple_store.PleStore(
        paths=[str(path)],
        extent_file=[0],
        extent_base=[0],
        rows_per_extent=total_rows,
        row_bytes=cols,
        row_stride=cols,
        multipliers=multipliers.tolist(),
        head_vocab_sizes=sizes.tolist(),
        head_offsets=offsets.tolist(),
        eos_token_id=EOS,
        use_io_uring=use_io_uring,
    )
    return store, table, args


def _fill(store, args, window, tokens):
    ctx = torch.tensor([window[0], window[1], *tokens], dtype=torch.int64)
    staging = torch.empty(len(tokens) * args.num_ngram_heads * args.ngram_head_dim, dtype=torch.uint8)
    store.stage(ctx.data_ptr(), len(tokens), staging.data_ptr())
    store.flush(0)
    return staging


def _write_checkpoint(tmp_path, table, n_shards, scale=0.03125):
    from safetensors.torch import save_file

    per = table.shape[0] // n_shards
    tensors = {
        f"{_KEY_PREFIX}.shard_{i}.weight": table[i * per : (i + 1) * per].view(torch.float8_e4m3fn)
        for i in range(n_shards)
    }
    tensors[f"{_KEY_PREFIX}.weight_scale"] = torch.tensor(scale, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "model.safetensors"))


def _make_table(tmp_path, *, max_byte: int = 256, scale: float = 0.03125):
    """``max_byte=127`` drops the fp8 NaN encodings, for tests that assert on layer outputs."""
    from freetoken.models.qwen4_exp.ple_disk import DiskRowTable, source_from_safetensors

    args = parse_config(toy_hf_config()).qwen4_args
    multipliers, sizes, offsets = hash_constants(args)
    total_rows = int(offsets[-1] + sizes[-1])
    gen = torch.Generator().manual_seed(9)
    table = torch.randint(0, max_byte, (total_rows, args.ngram_head_dim), dtype=torch.uint8, generator=gen)
    n_shards = next(k for k in (4, 2, 1) if total_rows % k == 0)
    _write_checkpoint(tmp_path, table, n_shards, scale)
    constants = {
        "num_ngram_heads": args.num_ngram_heads,
        "layer_multipliers": multipliers.tolist(),
        "per_head_vocab_sizes": sizes.tolist(),
        "per_head_offsets": offsets.tolist(),
        "eos_token_id": EOS,
    }
    disk = DiskRowTable(source_from_safetensors(str(tmp_path)), constants)
    oracle = GpuResidentTable(table.to(_ORACLE_DEVICE).view(torch.float8_e4m3fn),
                              scale=float(torch.tensor(scale, dtype=torch.bfloat16)))
    return disk, oracle, args, constants


def _oracle_rows(oracle, ids: torch.Tensor) -> torch.Tensor:
    return oracle.lookup(ids.to(_ORACLE_DEVICE)).cpu()


def _decode_batch(history, token):
    req = SimpleNamespace(
        input_ids=torch.tensor(history, dtype=torch.int32),
        device_len=len(history) + 1,
        cached_len=len(history),
    )
    return SimpleNamespace(
        is_decode=True,
        input_ids=torch.tensor([token], dtype=torch.int32, device=DEVICE),
        reqs=[req],
        size=1,
        padded_size=1,
    )


def test_store_stages_bitwise_rows(tmp_path):
    store, table, args = _make_store(tmp_path)
    emb = _embedding()
    row = args.num_ngram_heads * args.ngram_head_dim

    # full run with mid-sequence eos vs production row ids
    seq = [3, 4, EOS, 5, EOS, EOS, 8, 9, 3, 4]
    whole = _fill(store, args, (EOS, EOS), seq)
    ids = emb.row_ids(_meta([seq], [[EOS, EOS]]))
    assert torch.equal(whole, table[ids.reshape(-1)].reshape(-1)), "prefill vs oracle"

    # decode = many 1-token stages; must reproduce the same bytes
    parts, window = [], (EOS, EOS)
    for t in seq:
        parts.append(_fill(store, args, window, [t]))
        window = (window[1], t)
    assert torch.equal(torch.cat(parts), whole), "split stages vs one stage"

    # several lanes merged into one flush stay independent
    contexts = [(3, 4), (EOS, EOS), (7, EOS)]
    ctx = torch.tensor([[o, nw, 9] for o, nw in contexts], dtype=torch.int64)
    staging = torch.empty(3 * row, dtype=torch.uint8)
    for i in range(len(contexts)):
        store.stage(ctx.data_ptr() + 24 * i, 1, staging.data_ptr() + i * row)
    store.flush(0)
    for i, context in enumerate(contexts):
        assert torch.equal(staging[i * row : (i + 1) * row], _fill(store, args, context, [9])), f"lane {i}"

    # hundreds of deduped reads through the 64-deep pipeline
    gen = torch.Generator().manual_seed(23)
    big = torch.randint(0, EOS, (150,), generator=gen, dtype=torch.int64)
    got = _fill(store, args, (EOS, EOS), big.tolist())
    ids = emb.row_ids(_meta([big.tolist()], [[EOS, EOS]]))
    assert torch.equal(got, table[ids.reshape(-1)].reshape(-1)), "pipeline vs oracle"

    # flush signals the flag, even when nothing was staged
    flag = torch.zeros(1, dtype=torch.int64)
    store.flush(flag.data_ptr())
    assert int(flag[0]) == 1, "empty flush must still signal"


def test_layouts_readers_and_errors(tmp_path):
    _ple_store = _store_ext()
    # 4 extents in 2 files, out of order, unaligned junk between; the last extent ends at EOF
    sizes, offsets = [500, 400, 300, 800], [0, 500, 900, 1200]
    total, per, cols, eos = 2000, 500, 24, 90
    gen = torch.Generator().manual_seed(11)
    table = torch.randint(0, 256, (total, cols), dtype=torch.uint8, generator=gen)
    shard = lambda i: table[i * per : (i + 1) * per].numpy().tobytes()  # noqa: E731
    nb = per * cols
    flat = tmp_path / "flat.bin"
    flat.write_bytes(table.numpy().tobytes())
    fa, fb = tmp_path / "a.bin", tmp_path / "b.bin"
    fa.write_bytes(b"j" * 1231 + shard(0) + b"k" * 77 + shard(2))
    fb.write_bytes(shard(1) + b"m" * 4095 + shard(3))
    kwargs = dict(
        rows_per_extent=per, row_bytes=cols, row_stride=cols,
        multipliers=[3, 5, 7], head_vocab_sizes=sizes, head_offsets=offsets,
        eos_token_id=eos,
    )
    ref = _ple_store.PleStore(
        paths=[str(flat)], extent_file=[0, 0, 0, 0], extent_base=[0, nb, 2 * nb, 3 * nb], **kwargs
    )
    multi = _ple_store.PleStore(
        paths=[str(fa), str(fb)], extent_file=[0, 1, 0, 1],
        extent_base=[1231, 0, 1231 + nb + 77, nb + 4095], **kwargs,
    )
    tokens = torch.randint(0, eos, (40,), generator=gen, dtype=torch.int64)
    ctx = torch.cat((torch.tensor([eos, eos], dtype=torch.int64), tokens))

    def run(store):
        staging = torch.empty(40 * 4 * cols, dtype=torch.uint8)
        store.stage(ctx.data_ptr(), 40, staging.data_ptr())
        store.flush(0)
        return staging

    assert torch.equal(run(multi), run(ref)), "multi-extent vs flat"

    # thread-pool fallback must produce the same bytes as io_uring
    ring, _, args = _make_store(tmp_path)
    pool, _, _ = _make_store(tmp_path, write=False, use_io_uring=False)
    gen2 = torch.Generator().manual_seed(31)
    seq = torch.randint(0, EOS, (150,), generator=gen2, dtype=torch.int64).tolist()
    assert torch.equal(_fill(pool, args, (EOS, EOS), seq), _fill(ring, args, (EOS, EOS), seq)), "pool vs ring"

    # geometry that exceeds the file is rejected at construction
    del ring, pool
    with open(tmp_path / "ple-table.bin", "r+b") as fh:
        fh.truncate(1000)
    with pytest.raises(Exception, match="extent needs"):
        _make_store(tmp_path, write=False)

    # checkpoint scan guards
    from safetensors.torch import save_file

    from freetoken.models.qwen4_exp.ple_disk import source_from_safetensors

    save_file(
        {f"{_KEY_PREFIX}.shard_0.weight": torch.zeros(8, 4, dtype=torch.uint8),
         f"{_KEY_PREFIX}.weight_scale": torch.tensor(1.0, dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(ValueError, match="dtype"):
        source_from_safetensors(str(tmp_path))
    save_file(
        {f"{_KEY_PREFIX}.shard_1.weight": torch.zeros(8, 4, dtype=torch.float8_e4m3fn),
         f"{_KEY_PREFIX}.weight_scale": torch.tensor(1.0, dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(ValueError, match="contiguous"):
        source_from_safetensors(str(tmp_path))
    save_file(
        {f"{_KEY_PREFIX}.shard_0.weight": torch.zeros(8, 4, dtype=torch.float8_e4m3fn),
         f"{_KEY_PREFIX}.weight_scale": torch.tensor(1.0, dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    save_file(
        {f"{_KEY_PREFIX}.shard_0.weight": torch.zeros(8, 4, dtype=torch.float8_e4m3fn)},
        str(tmp_path / "model-2.safetensors"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        source_from_safetensors(str(tmp_path))
    (tmp_path / "model-2.safetensors").unlink()

    # truncated checkpoint: a contiguous shard prefix passes the scan, init rejects the row count
    from freetoken.models.qwen4_exp.ple_disk import DiskRowTable

    args = parse_config(toy_hf_config()).qwen4_args
    multipliers, vocab, offs = hash_constants(args)
    rows = int(offs[-1] + vocab[-1])
    _write_checkpoint(tmp_path, torch.zeros(rows // 2, args.ngram_head_dim, dtype=torch.uint8), 1)
    constants = {
        "num_ngram_heads": args.num_ngram_heads, "layer_multipliers": multipliers.tolist(),
        "per_head_vocab_sizes": vocab.tolist(), "per_head_offsets": offs.tolist(), "eos_token_id": EOS,
    }
    with pytest.raises(ValueError, match="hash addresses"):
        DiskRowTable(source_from_safetensors(str(tmp_path)), constants)


@requires_gpu
def test_disk_table_matches_oracle(tmp_path):
    disk, oracle, args, _constants = _make_table(tmp_path)
    emb = _embedding()

    # prefill: two segments, one fresh and one mid-sequence window
    seqs = [[3, 4, EOS, 5, 6, 8], [2, EOS, 11, 12, 13, 14]]
    disk.fill([torch.tensor([EOS, EOS, *seqs[0]]), torch.tensor([21, 22, *seqs[1]])], graph=False)
    row_ids = emb.row_ids(_meta(seqs, [[EOS, EOS], [21, 22]])).to(DEVICE)
    assert _bitwise_equal(disk.lookup(row_ids).cpu(), _oracle_rows(oracle, row_ids)), "prefill"

    # decode steps with a rolling window, plus the out= contract
    older, newer = 41, EOS
    for token in (7, 9, 13):
        disk.fill([torch.tensor([older, newer, token])], graph=False)
        ids = emb.row_ids(_meta([[token]], [[older, newer]], decode=True)).to(DEVICE)
        out = torch.empty((1, ids.shape[-1] * disk.head_dim), dtype=disk.dtype, device=DEVICE)
        assert disk.lookup(ids, out) is out and _bitwise_equal(out.cpu(), _oracle_rows(oracle, ids)), f"token {token}"
        older, newer = newer, token

    # the engine hook end to end: eager decode, then fresh + continuation prefill
    disk.host_fill_batch(_decode_batch([3, 4, EOS, 5], 9), use_graph=False)
    ids = emb.row_ids(_meta([[9]], [[EOS, 5]], decode=True)).to(DEVICE)
    assert _bitwise_equal(disk.lookup(ids).cpu(), _oracle_rows(oracle, ids)), "hook decode"

    prompt = [3, 4, EOS, 5, 6, 8]
    fresh = SimpleNamespace(input_ids=torch.tensor(prompt[:4], dtype=torch.int32), device_len=4, cached_len=0)
    cont = SimpleNamespace(input_ids=torch.tensor(prompt, dtype=torch.int32), device_len=6, cached_len=4)
    disk.host_fill_batch(SimpleNamespace(is_decode=False, padded_reqs=[fresh, cont]), use_graph=False)
    ids = emb.row_ids(_meta([prompt[:4], prompt[4:]], [[EOS, EOS], [prompt[2], prompt[3]]])).to(DEVICE)
    assert _bitwise_equal(disk.lookup(ids).cpu(), _oracle_rows(oracle, ids)), "hook prefill"


@requires_cuda
def test_graph_sync_protocol(tmp_path, monkeypatch):
    _store_ext()
    disk, oracle, args, _constants = _make_table(tmp_path)
    emb = _embedding()
    rows = 1
    row_ids = torch.zeros((rows, args.num_ngram_heads), dtype=torch.int64, device="cuda")
    out = torch.empty((rows, args.num_ngram_heads * disk.head_dim), dtype=disk.dtype, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            disk.lookup(row_ids, out)
    torch.cuda.current_stream().wait_stream(stream)

    # plain fill-then-replay
    disk.fill([torch.tensor([3, 4, 7])], graph=True)
    graph.replay()
    torch.cuda.synchronize()
    ids = emb.row_ids(_meta([[7]], [[3, 4]], decode=True)).cuda()
    assert _bitwise_equal(out, oracle.lookup(ids)), "capture+replay"

    if disk._wait_sync:
        # launch first, fill after: an early WAIT pass would surface the previous step's bytes
        older, newer = 3, 4
        for token in (7, 9, 13):
            graph.replay()
            disk.fill([torch.tensor([older, newer, token])], graph=True)
            torch.cuda.synchronize()
            ids = emb.row_ids(_meta([[token]], [[older, newer]], decode=True)).cuda()
            assert _bitwise_equal(out, oracle.lookup(ids)), f"wait-sync token {token}"
            older, newer = newer, token

        # the engine-shaped seam: replay inside the context, deferred fill on exit
        with disk.forward_host_ctx(_decode_batch([3, 4], 7), use_graph=True):
            graph.replay()
        torch.cuda.synchronize()
        ids = emb.row_ids(_meta([[7]], [[3, 4]], decode=True)).cuda()
        assert _bitwise_equal(out, oracle.lookup(ids)), "forward_host_ctx deferred"

    # gate mode: the hook fills inline and returns no deferred
    monkeypatch.setenv("FREETOKEN_PLE_SYNC", "gate")
    gated, _, _, _ = _make_table(tmp_path)
    assert not gated._wait_sync
    assert gated.host_fill_batch(_decode_batch([3, 4], 5), use_graph=True) is None


@requires_mps
def test_host_hash_matches_row_ids(tmp_path):
    import numpy as np

    disk = _make_table(tmp_path)[0]
    emb = _embedding()
    # eos padding at the start, a mid-sequence eos, and an eos as the current token
    seq = [3, 4, EOS, 5, EOS, EOS, 8, 9, 3, 4]
    got = disk.hash_rows(np.array([EOS, EOS, *seq], dtype=np.int64))
    assert np.array_equal(got, emb.row_ids(_meta([seq], [[EOS, EOS]])).numpy())


@requires_mps
def test_decode_tape_replay_matches_eager(tmp_path):
    """A recorded decode PLE forward replays to the eager result, and new tokens flow through it."""
    from freetoken.engine.mps_tape import DecodeTape
    from freetoken.models.qwen4_exp.ple import build_ple_metadata

    from .test_ple import _make_layer

    # a NaN row would make every assertion below vacuous: NaN != NaN, and it poisons its whole token
    disk, _oracle, args, _constants = _make_table(tmp_path, max_byte=127)
    config = parse_config(toy_hf_config())
    layer = _make_layer(config, device=DEVICE, dtype=torch.float32, table=disk)

    bs = 2
    histories = [[5, 6, 11, 12], [9, EOS, 3, 4]]
    reqs = [SimpleNamespace(input_ids=torch.tensor(h, dtype=torch.int32), device_len=len(h) + 1,
                            cached_len=len(h)) for h in histories]
    ctxp = torch.tensor([[EOS, EOS], *[[h[-2], h[-1]] for h in histories]], dtype=torch.int32, device=DEVICE)
    batch = SimpleNamespace(
        is_decode=True, is_prefill=False, reqs=reqs, padded_reqs=reqs, size=bs, padded_size=bs,
        input_ids=torch.tensor([13, 21], dtype=torch.int32, device=DEVICE),
        linear_table_idx=torch.arange(1, bs + 1, dtype=torch.int32, device=DEVICE),
        # the scheduler always hands a decode batch this, on a persistent buffer (graph.py:64-66);
        # without it build_ple_metadata allocates an indptr the tape cannot replay
        fla_metadata=SimpleNamespace(
            cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=DEVICE),
            track_boundary_row=None,
        ),
    )
    R = torch.randn(bs, args.ple_state_width, device=DEVICE)
    states0 = torch.randn(bs + 1, args.ple_state_width, args.ple_conv_state_len, device=DEVICE) * 0.1
    states = states0.clone()

    def step():
        meta = build_ple_metadata(batch, args, R.device, context_pool=ctxp)
        return layer.forward(R, batch, meta=meta, conv_states=states)

    states.copy_(states0)
    with disk.forward_host_ctx(batch, use_graph=True):
        eager = step().clone()
    eager_states = states.clone()

    states.copy_(states0)
    out = torch.empty_like(eager)
    with disk.forward_host_ctx(batch, use_graph=True):
        tape = DecodeTape.record(lambda: out.copy_(step()))
    assert len(tape) > 0, "nothing recorded"

    # the tape's buffers are inference tensors; the engine replays inside inference mode too
    states.copy_(states0)
    with torch.inference_mode(), disk.forward_host_ctx(batch, use_graph=True):
        tape.replay()
    torch.mps.synchronize()
    assert torch.equal(out, eager), f"replay vs eager ({len(tape)} launches)"
    assert torch.equal(states, eager_states), "conv state after replay"

    # a new token in the same buffer must reach the replay through the staged rows
    batch.input_ids.copy_(torch.tensor([31, 7], dtype=torch.int32, device=DEVICE))
    states.copy_(states0)
    with torch.inference_mode(), disk.forward_host_ctx(batch, use_graph=True):
        tape.replay()
    torch.mps.synchronize()
    replayed, replayed_states = out.clone(), states.clone()
    assert not torch.equal(replayed, eager), "the tape froze the first step's rows"

    states.copy_(states0)
    with disk.forward_host_ctx(batch, use_graph=True):
        assert torch.equal(step(), replayed), "replay vs eager on the new token"
    assert torch.equal(states, replayed_states), "conv state on the new token"


@requires_mps
def test_lookup_with_nothing_staged_serves_zeros(tmp_path):
    """Graph capture and the prefill warmup stage nothing; raising there kills the engine before its first request."""
    from freetoken.engine.mps_tape import DecodeTape

    disk, _oracle, args, _constants = _make_table(tmp_path)
    ids = torch.zeros((2, args.num_ngram_heads), dtype=torch.int64, device=DEVICE)

    unstaged = disk.lookup(ids)
    assert unstaged.shape == (2, args.num_ngram_heads * disk.head_dim)
    assert not unstaged.any(), "an unstaged lookup must read the zeroed arena"

    # the same thing inside a recording: it must record, and replay must still read zeros
    out = torch.empty_like(unstaged)
    tape = DecodeTape.record(lambda: out.copy_(disk.lookup(ids)))
    assert len(tape) > 0, "nothing recorded"
    with torch.inference_mode():
        tape.replay()
    torch.mps.synchronize()
    assert not out.any(), "replay of an unstaged lookup must still read zeros"


@requires_mps
def test_duplicate_rows_fan_out_at_a_non_power_of_two_scale(tmp_path):
    """A repeating prompt repeats its n-gram rows, so the dedup fan-out runs; a non-power-of-two
    scale folded into the LUT on mps agrees with the oracle's cast-then-multiply only to rounding."""
    disk, oracle, args, _constants = _make_table(tmp_path, scale=0.0123456789)
    emb = _embedding()

    prompt = [11, 12] * 6  # period 2, so every bigram and trigram window recurs
    req = SimpleNamespace(input_ids=torch.tensor(prompt, dtype=torch.int32),
                          device_len=len(prompt), cached_len=0)
    ids = emb.row_ids(_meta([prompt], [[EOS, EOS]]))
    flat = ids.reshape(-1)
    assert flat.unique().numel() < flat.numel(), "this batch was meant to repeat rows"

    disk.host_fill_batch(SimpleNamespace(is_decode=False, padded_reqs=[req]), use_graph=False)
    got = disk.lookup(ids.to(DEVICE)).cpu()
    # every destination, the fanned-out duplicates included, must hold its own row's bytes
    assert torch.allclose(got.float(), _oracle_rows(oracle, ids).float(), rtol=1e-2, atol=1e-3)


@requires_mps
def test_short_shard_is_rejected_at_construction(tmp_path):
    """A truncated or still-downloading shard must fail here, not as a short read mid-forward."""
    from freetoken.models.qwen4_exp.ple_disk import DiskRowTable, source_from_safetensors

    _disk, _oracle, _args, constants = _make_table(tmp_path)
    source = source_from_safetensors(str(tmp_path))
    with open(tmp_path / "model.safetensors", "r+b") as fh:
        fh.truncate(os.path.getsize(tmp_path / "model.safetensors") - 4096)
    with pytest.raises(ValueError, match="extent needs"):
        DiskRowTable(source, constants)
