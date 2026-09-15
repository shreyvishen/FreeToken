"""The NVFP4 disk offset index: every expert's nine tensor locations, their byte sizes, and reads that match what ``safe_open`` returns for the same names."""

from __future__ import annotations

import numpy as np
import safetensors
import torch

from freetoken.moe.expert_reader import _SAFETENSORS_DTYPE_TO_NUMPY, Nvfp4DiskIndex
from tests.moe.test_disk_moe_cache import LAYER, NUM_EXPERTS, PIECES, prefix, write_tiny_checkpoint

FIELDS = {f"{p}_{k}": (f"{p}_proj", f"weight{s}")
          for p in ("gate", "up", "down")
          for k, s in (("weight", ""), ("scale", "_scale"), ("global", "_scale_2"))}


def test_every_expert_has_six_pieces_over_nine_tensors(tmp_path):
    model_path, _ = write_tiny_checkpoint(tmp_path)
    index = Nvfp4DiskIndex(model_path)
    assert index.layers() == [LAYER]
    for expert in index.experts_per_layer(LAYER):
        loc = index.get(LAYER, expert)
        assert set(loc.pieces()) == PIECES and len(loc.all_locs()) == 9
        # the recorded byte span must be exactly what the shape and dtype imply
        for t in loc.all_locs():
            itemsize = np.dtype(_SAFETENSORS_DTYPE_TO_NUMPY[t.dtype]).itemsize
            assert t.nbytes == int(np.prod(t.shape)) * itemsize
    assert len(index.experts_per_layer(LAYER)) == NUM_EXPERTS


def test_reads_match_safe_open(tmp_path):
    model_path, _ = write_tiny_checkpoint(tmp_path)
    index = Nvfp4DiskIndex(model_path)
    # torch framework: numpy has no float8 or bfloat16, which several scale tensors use
    shards: dict[str, safetensors.safe_open] = {}
    for expert in index.experts_per_layer(LAYER):
        loc = index.get(LAYER, expert)
        for field, (proj, kind) in FIELDS.items():
            t = getattr(loc, field)
            if t.shard_path not in shards:
                shards[t.shard_path] = safetensors.safe_open(t.shard_path, framework="pt",
                                                             device="cpu")
            want = shards[t.shard_path].get_tensor(f"{prefix(expert)}.{proj}.{kind}")
            got = t.read()
            assert tuple(got.shape) == tuple(want.shape), field
            assert got.reshape(-1).view(np.uint8).tobytes() == want.reshape(-1).view(
                torch.uint8).numpy().tobytes(), field
