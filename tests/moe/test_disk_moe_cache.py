"""DiskMoeCache reading real bytes through both of ExpertReader's backends."""

from __future__ import annotations

import fcntl
import json
import os
import struct
import sys

import pytest
import safetensors.torch
import torch

from freetoken.kernel.metal import is_available
from freetoken.moe.disk_cache import PIN_FRAC, PIN_RANK_TOKENS, DiskMoeCache
from freetoken.moe.expert_reader import _F_RDADVISE, PIECE_ORDER, ExpertReader, Nvfp4DiskIndex

GROUP, FP8 = 16, torch.float8_e4m3fn
NUM_EXPERTS, HIDDEN, INTER, LAYER = 2, 32, 16, 0
PIECES = {"gate_up_packed", "gate_up_scale", "gate_up_global",
          "down_packed", "down_scale", "down_global"}


def prefix(expert: int, layer: int = LAYER) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}"


def write_tiny_checkpoint(tmp_path, num_experts=NUM_EXPERTS, layers=(LAYER,)):
    os.makedirs(tmp_path, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    for layer in layers:
        for e in range(num_experts):
            for proj, out_dim, in_dim in ((f"{prefix(e, layer)}.gate_proj", INTER, HIDDEN),
                                          (f"{prefix(e, layer)}.up_proj", INTER, HIDDEN),
                                          (f"{prefix(e, layer)}.down_proj", HIDDEN, INTER)):
                tensors[f"{proj}.weight"] = torch.randint(
                    0, 256, (out_dim, in_dim // 2), dtype=torch.uint8, generator=g)
                tensors[f"{proj}.weight_scale"] = (
                    torch.rand(out_dim, in_dim // GROUP, generator=g) * 4).to(FP8)
                tensors[f"{proj}.weight_scale_2"] = (torch.rand(1, generator=g) * 0.1)[0]
    safetensors.torch.save_file(tensors, str(tmp_path / "model.safetensors"))
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": {n: "model.safetensors" for n in tensors}}, f)
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"model_type": "qwen3_5_moe"}, f)
    return str(tmp_path), tensors


def test_index_reader_bytes_match_the_checkpoint(tmp_path):
    model_path, tensors = write_tiny_checkpoint(tmp_path)
    expert = 1
    assert set(Nvfp4DiskIndex(model_path).get(LAYER, expert).pieces()) == PIECES

    cache = DiskMoeCache(num_layers=1, num_experts=NUM_EXPERTS, cache_size=2,
                         device=torch.device("cpu"), quant_format="nvfp4")
    cache.set_disk_source(ExpertReader(model_path), HIDDEN, INTER)
    ids = torch.tensor([[expert]], dtype=torch.int32)
    cache.ensure_experts(LAYER, ids)
    cache.copy_missing()
    slot = int(ids[0, 0])

    raw = lambda proj, kind: tensors[f"{prefix(expert)}.{proj}_proj.weight{kind}"]
    for bank, kind in (("gate_up_packed", ""), ("gate_up_scale", "_scale")):
        got = cache.bank_caches[bank][slot]
        assert torch.equal(got[:INTER], raw("gate", kind)) and torch.equal(
            got[INTER:], raw("up", kind))
    assert torch.equal(cache.bank_caches["down_packed"][slot], raw("down", ""))
    assert torch.equal(cache.bank_caches["down_scale"][slot], raw("down", "_scale"))
    # the globals are per-tensor f32 on disk and per-output-row fp16 in the banks
    for bank, proj, rows in (("gate_up_global", "gate", slice(0, INTER)),
                             ("gate_up_global", "up", slice(INTER, 2 * INTER)),
                             ("down_global", "down", slice(0, HIDDEN))):
        want = raw(proj, "_scale_2").to(torch.float16).expand(rows.stop - rows.start)
        assert torch.equal(cache.bank_caches[bank][slot][rows], want)
    cache.close()


def test_repacked_reader_matches_the_index_reader(tmp_path):
    from freetoken.checkpoint.repack_nvfp4_experts import build as repack_build

    model_path, _ = write_tiny_checkpoint(tmp_path / "ckpt")
    repacked_dir = str(tmp_path / "repacked")
    repack_build(model_path, repacked_dir, layers=[LAYER])
    index_reader = ExpertReader(model_path)
    repacked_reader = ExpertReader(model_path, repacked_dir)
    assert repacked_reader.reads_per_expert == 1 and index_reader.reads_per_expert == 9

    nbytes = [t.nbytes for t in Nvfp4DiskIndex(model_path).get(LAYER, 1).all_locs()]
    a, b = ([bytearray(n) for n in nbytes] for _ in range(2))
    index_reader.read_into(LAYER, 1, [memoryview(x) for x in a])
    repacked_reader.read_into(LAYER, 1, [memoryview(x) for x in b])
    for name, xa, xb in zip(PIECE_ORDER, a, b):
        assert bytes(xa) == bytes(xb), name
    index_reader.close()
    repacked_reader.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="F_RDADVISE is a Darwin fcntl")
def test_the_readahead_hint_names_the_bytes_the_read_returns(tmp_path, monkeypatch):
    model_path, _ = write_tiny_checkpoint(tmp_path)
    locs = Nvfp4DiskIndex(model_path).get(LAYER, 1).all_locs()
    hinted, real = [], fcntl.fcntl

    def record(fd, cmd, arg=0):
        if cmd == _F_RDADVISE:
            hinted.append(struct.unpack("qi", arg))
        return real(fd, cmd, arg)

    monkeypatch.setattr(fcntl, "fcntl", record)
    reader = ExpertReader(model_path)
    dests = [bytearray(loc.nbytes) for loc in locs]
    reader.read_into(LAYER, 1, [memoryview(d) for d in dests])
    assert hinted == [(loc.offset, loc.nbytes) for loc in locs]
    for loc, got in zip(locs, dests):
        with open(loc.shard_path, "rb") as f:
            f.seek(loc.offset)
            assert bytes(got) == f.read(loc.nbytes)
    reader.close()

    # The repack path hints the one record it is about to read.
    from freetoken.checkpoint.repack_nvfp4_experts import build as repack_build

    repacked_dir = str(tmp_path / "repacked")
    repack_build(model_path, repacked_dir, layers=[LAYER])
    reader = ExpertReader(model_path, repacked_dir)
    hinted.clear()
    reader.read_into(LAYER, 1, [memoryview(bytearray(loc.nbytes)) for loc in locs])
    assert hinted == [reader._entries[(LAYER, 1)]]
    assert hinted[0][1] >= sum(loc.nbytes for loc in locs)  # the record holds all nine
    reader.close()


def test_the_pin_set_holds_the_routing_head_against_eviction(tmp_path):
    model_path, _ = write_tiny_checkpoint(tmp_path, num_experts=4, layers=(0, 1))
    cache = DiskMoeCache(num_layers=2, num_experts=4, cache_size=4,
                         device=torch.device("cpu"), quant_format="nvfp4")
    cache.set_disk_source(ExpertReader(model_path), HIDDEN, INTER)
    head = 3
    for step in range(PIN_RANK_TOKENS):
        for layer in (0, 1):
            cache.ensure_experts(layer, torch.tensor([[head, step % 4]], dtype=torch.int32))
            cache.copy_missing()
    assert cache._pin_expert.tolist() == [[i == head for i in range(4)]] * 2
    for expert in [e for e in range(4) if e != head] * 4:
        for layer in (0, 1):
            cache.ensure_experts(layer, torch.tensor([[expert]], dtype=torch.int32))
            cache.copy_missing()
    assert int(cache._slot_of[0, head]) >= 0 and int(cache._slot_of[1, head]) >= 0
    cache.close()


def test_the_pin_set_never_outgrows_the_slot_cache(tmp_path):
    """PIN_FRAC of a 512-expert layer over 48 layers names more slots than a small tier holds,
    and an all-pinned tier re-ranks unbiased on every miss. Here 8 layers x PIN_FRAC of 32
    experts wants 24 of 32 slots; the cap holds it to half, leaving the LRU something to evict."""
    experts, layers = 32, 8
    model_path, _ = write_tiny_checkpoint(tmp_path, num_experts=experts,
                                          layers=tuple(range(layers)))
    for cache_size in (experts, 2 * experts):
        cache = DiskMoeCache(num_layers=layers, num_experts=experts, cache_size=cache_size,
                             device=torch.device("cpu"), quant_format="nvfp4")
        cache.set_disk_source(ExpertReader(model_path), HIDDEN, INTER)
        cache._rank_pins()
        pinned = int(cache._pin_expert.sum())
        assert pinned == layers * min(int(PIN_FRAC * experts), cache_size // (2 * layers))
        assert pinned * 2 <= cache_size, cache_size
        cache.close()


@pytest.mark.skipif(not is_available(), reason="the slot scatter is a Metal kernel")
def test_the_metal_slot_scatter_writes_what_the_row_copies_write(tmp_path):
    """Slots differ from staged rows (expert 0 warms slot 0 first), and the prefetch path's
    ``rows=`` form moves staged rows 2 and 0 into slots 3 and 0."""
    model_path, _ = write_tiny_checkpoint(tmp_path, num_experts=4)
    banks = []
    for device in (torch.device("cpu"), torch.device("mps")):
        cache = DiskMoeCache(num_layers=1, num_experts=4, cache_size=4, device=device,
                             quant_format="nvfp4")
        cache.set_disk_source(ExpertReader(model_path), HIDDEN, INTER)
        assert cache._scatter == (device.type == "mps")
        for experts in ([[0]], [[3, 1, 2]]):
            cache.ensure_experts(LAYER, torch.tensor(experts, dtype=torch.int32, device=device))
            cache.copy_missing()
        assert cache._slot_of[LAYER].tolist() == [0, 1, 2, 3]   # experts 1-3 from staged rows 0-2
        torch.mps.synchronize()   # the engine's next routing read drains the queue before a reuse
        cache._flush_rows([3, 0], 3, rows=[2, 0])
        torch.mps.synchronize()
        banks.append({n: b.cpu().view(torch.uint8) for n, b in cache.bank_caches.items()})
        cache.close()
    for name, bank in banks[0].items():
        assert torch.equal(bank, banks[1][name]), name
