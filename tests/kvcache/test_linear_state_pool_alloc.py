"""LinearStatePool unit: the free-list allocator and the declared slot-state siblings.
CPU-only, fast — pure slot bookkeeping + state copy/zero, no kernels."""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.linear_state_pool import (
    LinearStatePool,
    linear_state_bytes_per_req,
    state_pool_bytes,
)
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec


def _group():
    return LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1),
        num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )


def _pool(num_slots=8, device="cpu", slot_states=()):
    return LinearStatePool(group=_group(), num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device(device), tp_size=1, slot_states=slot_states)


def test_alloc_free_roundtrip():
    pool = _pool(num_slots=8)
    assert pool.num_free_slots == 7          # slots 1..7 (slot 0 = padding)
    a = pool.alloc(3)
    assert len(set(a)) == 3 and all(1 <= s <= 7 for s in a)
    assert pool.padding_slot not in a        # slot 0 never allocated
    assert pool.num_free_slots == 4
    pool.free(a)
    assert pool.num_free_slots == 7
    # int and tensor free forms
    s = pool.alloc(1)[0]
    pool.free(s)
    s2 = pool.alloc(2)
    pool.free(torch.tensor(s2, dtype=torch.long))
    assert pool.num_free_slots == 7


def test_alloc_exhaustion_raises():
    pool = _pool(num_slots=4)                # 3 allocatable
    pool.alloc(3)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.alloc(1)


def test_clear_slots_zeros_all_layers():
    pool = _pool(num_slots=6)
    s = pool.alloc(1)[0]
    pool.conv_states[:, s] = 1.5
    pool.recurrent_states[:, s] = 2.0
    pool.clear_slots([s])
    assert pool.conv_states[:, s].abs().sum() == 0
    assert pool.recurrent_states[:, s].abs().sum() == 0


def test_copy_from_snapshot():
    pool = _pool(num_slots=6)
    src, dst = pool.alloc(2)
    torch.manual_seed(0)
    pool.conv_states[:, src] = torch.randn_like(pool.conv_states[:, src])
    pool.recurrent_states[:, src] = torch.randn_like(pool.recurrent_states[:, src])
    pool.copy_from(src, dst)
    assert torch.equal(pool.conv_states[:, dst], pool.conv_states[:, src])
    assert torch.equal(pool.recurrent_states[:, dst], pool.recurrent_states[:, src])


def _tracked_req(n_tokens, table_idx, pool, first_id):
    """A prefill Req with the live + ping-pong slots the hybrid path allocates at admission."""
    from freetoken.core import Req, SamplingParams

    ids = torch.arange(first_id, first_id + n_tokens, dtype=torch.int32)
    req = Req(input_ids=ids, table_idx=table_idx, cached_len=0, output_len=1, uid=table_idx,
              sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    return req


def test_snapshot_round_trip_through_the_radix_tree(monkeypatch):
    """Turn 2 on CPU, no kernel: snapshot row pick, GDN state write, radix reuse cut back to the chunk boundary, restore."""
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.kernel.fla.const import CHUNK_SIZE
    from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet

    pool = _pool(num_slots=12)
    km1, conv_dim = pool.conv_states.shape[-1], pool.conv_states.shape[-2]
    # two ragged requests, so the second one's row depends on the first one's chunk count
    reqs = [_tracked_req(CHUNK_SIZE + 6, 0, pool, 0),
            _tracked_req(2 * CHUNK_SIZE + 3, 1, pool, 500)]
    monkeypatch.setattr("freetoken.core.get_global_ctx",
                        lambda: SimpleNamespace(linear_state_pool=pool))
    batch = SimpleNamespace(padded_reqs=reqs, is_decode=False)
    fla = build_fla_metadata(batch, torch.device("cpu"))
    # the tracking path must stay importable without triton, which Metal cannot install; a fresh
    # interpreter, since this one has triton loaded wherever it is installed
    probe = "import sys, freetoken.attention.linear; assert 'triton' not in sys.modules"
    subprocess.run([sys.executable, "-c", probe], check=True)

    nchunks = [-(-r.extend_len // CHUNK_SIZE) for r in reqs]
    assert nchunks == [2, 3]
    assert fla.track_h_row.tolist() == [1, 2 + 2]          # boh = [0, 2, 5], c = [1, 2]
    assert fla.track_dst.tolist() == [r.mamba_ping_pong[0] for r in reqs]
    assert [r.mamba_last_track_seqlen for r in reqs] == [CHUNK_SIZE, 2 * CHUNK_SIZE]
    assert [r.mamba_next_track_idx for r in reqs] == [1, 1]

    total = sum(r.extend_len for r in reqs)
    torch.manual_seed(3)
    conv_in = torch.randn(total, conv_dim)
    h = torch.randn(1, sum(nchunks), *pool.recurrent_states.shape[2:])
    # __new__, not __init__: the snapshot reads only its arguments, and the real constructor
    # would allocate the layer's projections and need TP info for nothing.
    op = Qwen3_5GatedDeltaNet.__new__(Qwen3_5GatedDeltaNet)
    op._write_track_snapshot(pool, pool.local_index(0), conv_in, h, fla)

    req, frozen = reqs[1], reqs[1].mamba_ping_pong[0]
    assert torch.equal(pool.recurrent_states[0, frozen],
                       h[0, 4].to(pool.recurrent_states.dtype))
    want_conv = conv_in[fla.track_conv_src[1]].t().to(pool.conv_states.dtype)
    assert torch.equal(pool.conv_states[0, frozen], want_conv)

    # the scheduler donates the frozen slot at the tracked boundary, the live one at finish
    boundary = req.mamba_last_track_seqlen
    prompt = req.input_ids.to(torch.int64)
    kv = torch.arange(100, 100 + len(prompt), dtype=torch.int32)
    tree = HybridRadixCache(torch.device("cpu"), page_size=1)
    assert tree.insert(prompt[:boundary], kv[:boundary], frozen) == (0, False)
    assert tree.insert(prompt, kv, req.linear_slot_idx) == (boundary, False)

    # turn 2 shares the prefix then diverges INSIDE the deeper node (which spans 128..131), so
    # the split leaves its root-side half snapshot-less -> reuse falls back to the x64 boundary
    turn2 = torch.cat([prompt[:boundary + 2], torch.arange(900, 940, dtype=torch.int64)])
    m = tree.match_prefix(turn2)
    assert m.cached_len == boundary and m.mamba_value == frozen
    assert torch.equal(m.kv_indices, kv[:boundary])

    restore = pool.alloc(1)[0]
    pool.copy_from(m.mamba_value, restore)   # scheduler._restore_linear_states
    assert torch.equal(pool.recurrent_states[:, restore], pool.recurrent_states[:, frozen])
    assert torch.equal(pool.conv_states[:, restore], pool.conv_states[:, frozen])
    # the un-diverged turn still resumes at the deeper, newer snapshot
    assert tree.match_prefix(prompt).mamba_value == req.linear_slot_idx


if __name__ == "__main__":
    test_alloc_free_roundtrip()
    test_alloc_exhaustion_raises()
    test_clear_slots_zeros_all_layers()
    test_copy_from_snapshot()
    print("LinearStatePool allocator unit: PASS")


# qwen4_exp PLE conv-history shape at toy size: one PLE layer, 32 channels, 9 taps
_SPECS = (SlotStateSpec(name="ple_conv", shape=(32, 9), layer_ids=(1,)),)


def test_no_slot_states_by_default():
    pool = _pool()
    assert pool.slot_states == {} and not pool.has_slot_state("ple_conv")
    base = pool.bytes_per_slot()
    pool.clear_slots([1, 2])
    pool.copy_from(1, 2)
    pool.reset(3)
    pool.rebuild(4)
    assert pool.slot_states == {} and pool.bytes_per_slot() == base


def test_slot_state_geometry_and_accessor():
    pool = _pool(num_slots=6, slot_states=_SPECS)
    slab = pool.slot_states["ple_conv"]
    assert slab.shape == (1, 6, 32, 9) and slab.dtype is torch.bfloat16
    assert pool.slot_state("ple_conv", 1).shape == (6, 32, 9)
    with pytest.raises(KeyError):
        pool.slot_state("ple_conv", 0)  # not a declared layer
    with pytest.raises(KeyError):
        pool.slot_state("other")
    with pytest.raises(AssertionError):
        pool.slot_state("ple_conv")  # per-layer state needs layer_id


def test_layerless_spec_dtype_and_fill_value():
    spec = SlotStateSpec(name="ngram", shape=(2,), dtype=torch.int32, fill_value=7.0)
    pool = _pool(num_slots=6, slot_states=(spec,))
    t = pool.slot_state("ngram")
    assert t.shape == (6, 2) and t.dtype is torch.int32 and bool((t == 7).all())
    t[3] = 1
    pool.clear_slots([3])
    assert bool((pool.slot_state("ngram")[3] == 7).all())
    t[2] = 1
    pool.reset(2)
    assert bool((pool.slot_state("ngram")[2] == 7).all())
    pool.rebuild(5)
    assert bool((pool.slot_state("ngram") == 7).all())


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _pool(slot_states=_SPECS + _SPECS)


def test_slot_state_follows_every_slot_operation():
    pool = _pool(slot_states=_SPECS)
    pool.slot_states["ple_conv"].fill_(1.0)
    pool.conv_states.fill_(1.0)

    pool.clear_slots([2])
    slab = pool.slot_state("ple_conv", 1)
    assert slab[2].abs().sum().item() == 0.0
    assert slab[3].abs().sum().item() > 0.0

    pool.copy_from(3, 2)
    assert torch.equal(slab[2], slab[3])

    pool.reset(3)
    assert slab[3].abs().sum().item() == 0.0

    pool.rebuild(9)
    slab = pool.slot_state("ple_conv", 1)  # rebuild replaces the tensor
    assert pool.slot_states["ple_conv"].shape == (1, 9, 32, 9)
    assert slab.abs().sum().item() == 0.0

    # the ops write the rebuilt tensor, not a stale alias
    pool.slot_states["ple_conv"].fill_(1.0)
    pool.clear_slots([5])
    assert pool.slot_state("ple_conv", 1)[5].abs().sum().item() == 0.0
    pool.copy_from(1, 5)
    assert torch.equal(pool.slot_state("ple_conv", 1)[5], pool.slot_state("ple_conv", 1)[1])


def test_slot_state_in_the_byte_account():
    group = _group()
    gdn_only = linear_state_bytes_per_req(group, 1, torch.bfloat16)
    with_state = linear_state_bytes_per_req(group, 1, torch.bfloat16, _SPECS)
    assert with_state - gdn_only == 32 * 9 * 2
    assert _pool(slot_states=_SPECS).bytes_per_slot() == with_state

    mc = SimpleNamespace(slot_states=_SPECS)
    mc.linear_attention_group = lambda: group
    config = SimpleNamespace(
        model_config=mc, dtype=torch.bfloat16, tp_info=SimpleNamespace(size=1),
        cache_type="naive", max_running_req=3, linear_state_cache_ratio=0.5,
    )
    assert state_pool_bytes(config, num_slots=4) == with_state * 4

    mc.slot_states = ()
    assert state_pool_bytes(config, num_slots=4) == gdn_only * 4

    mc.slot_states = _SPECS
    mc.linear_attention_group = lambda: None
    with pytest.raises(ValueError, match="slot_states"):
        state_pool_bytes(config, num_slots=4)


def test_slot_state_bytes_for_the_real_geometry():
    # qwen4_exp PLE conv history: 4 streams x 2560 channels x 9 taps bf16 = 180 KiB per slot
    spec = SlotStateSpec(name="ple_conv", shape=(4 * 2560, 9), layer_ids=(1,))
    group = _group()
    delta = linear_state_bytes_per_req(group, 1, torch.bfloat16, (spec,)) - \
        linear_state_bytes_per_req(group, 1, torch.bfloat16)
    assert delta == 4 * 2560 * 9 * 2 == 180 * 1024
