"""Disk-backed PLE table (--ple-backend disk): the C++ store hashes n-gram windows and batch-reads rows from the checkpoint's fp8 shard tensors into pinned staging; the captured ``lookup`` is a fixed-shape H2D copy + dequant.

Hash windows are pure functions of ``req.input_ids`` + ``device_len`` (prefix hits, restores and COW forks need no bookkeeping); the decode input token lives device-side under overlap scheduling and is read back here.

``MetalDiskRowTable`` is the Apple twin of the same data path: numpy does the hash the C++ store does,
``os.preadv`` does the batched read, and the dequant is a 256-entry LUT because MPS has no fp8 cast.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import safetensors
import torch

from freetoken.mm import MM_PAD_SHIFT_VALUE, restore_placeholder

from freetoken.core import Batch
from freetoken.kernel import backend as device_backend
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

from .weight import (
    _PLE_SCALE_SUFFIX,
    _PLE_SHARD_RE,
    _PLE_ST_DTYPE,
    _ple_table_files,
    _safetensors_header,
)

_IO_URING_ENV = "FREETOKEN_PLE_IO_URING"
_SYNC_ENV = "FREETOKEN_PLE_SYNC"  # auto | wait | gate
_READ_THREADS = 8  # same cap as moe/disk_cache.py: pread saturates this drive before more help

logger = init_logger(__name__)


def _context(ids: torch.Tensor, position: int, eos: int) -> list[int]:
    """The two token ids before ``position``; eos pads past the start."""
    return [int(ids[position - 2]) if position >= 2 else eos,
            int(ids[position - 1]) if position >= 1 else eos]


@dataclass(frozen=True)
class PleRowSource:
    """On-disk row layout: equal extents, row i of an extent at ``base + i * row_stride`` (a repacked flat file is one extent with its own stride)."""

    paths: list[str]
    extent_file: list[int]
    extent_base: list[int]
    rows_per_extent: int
    row_bytes: int
    row_stride: int
    scale: float

    @property
    def total_rows(self) -> int:
        return len(self.extent_base) * self.rows_per_extent


def source_from_safetensors(folder: str) -> PleRowSource:
    """Map the checkpoint's ``ngram_embedding.shard_<i>`` tensors in place: one extent per shard, no copy."""
    rows = cols = 0
    scale: torch.Tensor | None = None
    paths: list[str] = []
    path_idx: dict[str, int] = {}
    shards: dict[int, tuple[int, int]] = {}
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE shard {key} has dtype {meta['dtype']}, expected {_PLE_ST_DTYPE}")
            if rows and tuple(meta["shape"]) != (rows, cols):
                raise ValueError(f"PLE shard {key} is {meta['shape']}, expected {[rows, cols]}")
            rows, cols = meta["shape"]
            if path not in path_idx:
                path_idx[path] = len(paths)
                paths.append(path)
            idx = int(match.group("shard"))
            if idx in shards:
                raise ValueError(f"duplicate PLE shard {idx} in {path}")
            shards[idx] = (path_idx[path], base + meta["data_offsets"][0])
    if sorted(shards) != list(range(len(shards))) or not shards:
        raise ValueError(f"PLE shard indices are not contiguous 0..N-1: {sorted(shards)[:8]}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")
    order = [shards[i] for i in range(len(shards))]
    return PleRowSource(paths, [f for f, _ in order], [b for _, b in order], rows, cols, cols, float(scale))


def resolve_row_source(folder: str) -> PleRowSource:
    """Pick the row source for a checkpoint; the seam where a repacked format would plug in."""
    return source_from_safetensors(folder)


class _PleRuns:
    """The per-request token runs both disk backends hash: two context ids, then this forward's tokens."""

    def _ple_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.image_token_id is None:
            return input_ids
        return restore_placeholder(input_ids, self.image_token_id)

    def _ple_context(self, ids: torch.Tensor, position: int) -> list[int]:
        if self.image_token_id is None:
            return _context(ids, position, self.eos_token_id)
        return [self.image_token_id if t >= MM_PAD_SHIFT_VALUE else t for t in _context(ids, position, self.eos_token_id)]

    def _decode_runs(self, reqs, tokens: Sequence[int]) -> list[torch.Tensor]:
        return [torch.tensor([*self._ple_context(r.input_ids, r.device_len - 1), t], dtype=torch.int64)
                for r, t in zip(reqs, tokens)]

    def _prefill_runs(self, reqs) -> list[torch.Tensor]:
        return [
            torch.cat((
                torch.tensor(self._ple_context(req.input_ids, req.cached_len), dtype=torch.int64),
                self._ple_ids(req.input_ids[req.cached_len : req.device_len]).to(torch.int64),
            ))
            for req in reqs
        ]


def DiskRowTable(source: PleRowSource, hash_constants: dict, **kwargs):  # noqa: N802 - a constructor seam, not a class
    """The disk backend this accelerator can run: the C++ store on CUDA, pread + a LUT dequant on Metal."""
    from freetoken.kernel.backend import is_mps

    return (MetalDiskRowTable if is_mps() else CudaDiskRowTable)(source, hash_constants, **kwargs)


class CudaDiskRowTable(_PleRuns):
    """``PLETableBackend`` whose rows are read from disk per fill (--ple-backend disk)."""

    def __init__(
        self,
        source: PleRowSource,
        hash_constants: dict,
        *,
        max_graph_rows: int = 256,
        max_extend_tokens: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from freetoken.kernel import _ple_store

        self.num_rows = source.total_rows
        self.head_dim = source.row_bytes  # fp8: one byte per element
        self.dtype = dtype
        self.heads = int(hash_constants["num_ngram_heads"])
        self.scale = source.scale
        self.eos_token_id = int(hash_constants["eos_token_id"])
        self.image_token_id = hash_constants.get("image_token_id")
        sizes = [int(x) for x in hash_constants["per_head_vocab_sizes"]]
        offsets = [int(x) for x in hash_constants["per_head_offsets"]]
        need = max(o + s for o, s in zip(offsets, sizes))
        if need > source.total_rows:
            raise ValueError(
                f"PLE row source holds {source.total_rows} rows but the hash addresses {need}; incomplete checkpoint?"
            )
        self._store = _ple_store.PleStore(
            paths=list(source.paths),
            extent_file=list(source.extent_file),
            extent_base=list(source.extent_base),
            rows_per_extent=source.rows_per_extent,
            row_bytes=source.row_bytes,
            row_stride=source.row_stride,
            multipliers=[int(x) for x in hash_constants["layer_multipliers"]],
            head_vocab_sizes=sizes,
            head_offsets=offsets,
            eos_token_id=self.eos_token_id,
            use_io_uring=os.getenv(_IO_URING_ENV, "1") != "0",
        )
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._token_bytes = self.heads * self.head_dim
        # allocated up front: pinned alloc inside stream capture is illegal; one replay consumes it at a time
        self._graph_pinned = alloc_pinned_tensor(max_graph_rows * self._token_bytes, dtype=torch.uint8)
        self._graph_pinned.zero_()  # padded decode lanes read whatever sits here
        # outlives any one graph: a cache rebuild recaptures against the same pointer
        self._graph_dev = torch.empty(
            max_graph_rows * self._token_bytes, dtype=torch.uint8, device=self._device
        )
        eager_bytes = max_extend_tokens * self._token_bytes
        self._eager_pinned = alloc_pinned_tensor(eager_bytes, dtype=torch.uint8)
        self._eager_pinned.zero_()  # the warmup prefill stages nothing and reads whatever sits here
        self._eager_dev = torch.empty(eager_bytes, dtype=torch.uint8, device=self._device)
        # probe picks flag-sync (graph WAITs at the consume, host fills then signals) or launch-gating
        self._wait_sync = self._probe_wait_sync(os.getenv(_SYNC_ENV, "auto"))
        # one flag for all graphs: the readback event orders a fill after the previous graph, so signals never overlap
        self._flag = alloc_pinned_tensor(1, dtype=torch.int64)
        self._flag.zero_()
        self._token_readback = alloc_pinned_tensor(max_graph_rows, dtype=torch.int32)
        self._readback_event = torch.cuda.Event()
        sync = "wait-sync" if self._wait_sync else "launch-gating"
        logger.info_rank0(f"PLE disk backend: {self._store.io_backend()}, {sync}")

    def _probe_wait_sync(self, mode: str) -> bool:
        from freetoken.kernel import _ple_store

        if mode == "gate":
            return False
        scratch = alloc_pinned_tensor(1, dtype=torch.int64)
        scratch.zero_()
        stream = torch.cuda.current_stream(self._device)
        ok = (
            _ple_store.memop_write(stream.cuda_stream, scratch.data_ptr(), 7) == 0
            and _ple_store.memop_wait_geq(stream.cuda_stream, scratch.data_ptr(), 7) == 0
        )
        if ok:
            stream.synchronize()
            ok = int(scratch[0]) == 7
        if mode == "wait" and not ok:
            raise RuntimeError("FREETOKEN_PLE_SYNC=wait but stream memops are unavailable")
        return ok

    # ---------------- host side (engine thread, before the forward launches) ----------------

    def fill(self, runs: Sequence[torch.Tensor], *, graph: bool) -> None:
        """Stage per-request token runs (two context ids, then the new tokens) in batch order."""
        pinned = self._graph_pinned if graph else self._eager_pinned
        offset = 0
        for run in runs:
            self._store.stage(run.data_ptr(), run.numel() - 2, pinned.data_ptr() + offset * self._token_bytes)
            offset += run.numel() - 2
        self._store.flush(self._flag.data_ptr() if graph and self._wait_sync else 0)

    def host_fill_batch(self, batch: Batch, use_graph: bool):
        """Stage this batch's rows; returns the post-dispatch fill callable under flag-sync, else None."""
        if batch.is_decode:
            reqs = list(batch.reqs)
            if use_graph and self._wait_sync:
                bs = batch.padded_size
                self._token_readback[:bs].copy_(batch.input_ids, non_blocking=True)
                self._readback_event.record(torch.cuda.current_stream(self._device))

                def _complete() -> None:
                    try:
                        self._readback_event.synchronize()
                        tokens = self._token_readback[:bs].to(torch.int64).tolist()
                        self.fill(self._decode_runs(reqs, tokens), graph=True)
                    except BaseException:
                        from freetoken.kernel import _ple_store

                        # unblock the stream before surfacing; the step's output is discarded
                        _ple_store.signal_flag(self._flag.data_ptr())
                        raise

                return _complete
            # launch-gating: this D2H is the step's readback and orders the fill after sampling
            tokens = batch.input_ids.to("cpu").to(torch.int64).tolist()
            self.fill(self._decode_runs(reqs, tokens), graph=use_graph)
            return None
        self.fill(self._prefill_runs(batch.padded_reqs), graph=False)
        return None

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        """Around one dispatch: stage on enter, run the deferred fill+signal on exit."""
        deferred = self.host_fill_batch(batch, use_graph)
        yield
        # no try/finally: a failed launch leaves no WAIT pending, so the fill must not run
        if deferred is not None:
            deferred()

    # ---------------- device side (PLETableBackend protocol) ----------------

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = row_ids.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self._wait_sync:
            from freetoken.kernel import _ple_store

            _ple_store.memop_wait_reset(
                torch.cuda.current_stream(self._device).cuda_stream, self._flag.data_ptr()
            )
        pinned, dev = (
            (self._graph_pinned, self._graph_dev) if capturing else (self._eager_pinned, self._eager_dev)
        )
        nbytes = rows * self._token_bytes
        dev[:nbytes].copy_(pinned[:nbytes], non_blocking=True)
        values = dev[:nbytes].view(torch.float8_e4m3fn).to(self.dtype)
        if self.scale != 1.0:
            values = values * self.scale
        values = values.view(*row_ids.shape[:-1], -1)
        if out is None:
            return values
        out.copy_(values)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None


class MetalDiskRowTable(_PleRuns):
    """``PLETableBackend`` for Apple Metal: the same data path as ``CudaDiskRowTable`` with no C++.

    The n-gram hash is the uint64 numpy twin of ``PleStore::hash_rows`` (and of
    ``NGramEmbedding.row_ids``); its rows are deduplicated, sorted by file offset and read with
    buffered ``os.preadv`` on a small thread pool straight into an MPS-visible arena. ``lookup`` is
    then a fixed-shape copy plus a LUT gather, which is all the decode tape has to replay -- every
    host-side step happens in ``forward_host_ctx``, before the dispatch.
    ``benchmarks/bench_ple_disk.py`` weighs that read against ``F_NOCACHE``, ``F_RDADVISE`` and ``mmap``.
    """

    # rows are staged in batch order, so lookup reads only the shape of its row ids
    hashes_on_host = True

    def __init__(
        self,
        source: PleRowSource,
        hash_constants: dict,
        *,
        max_graph_rows: int = 256,
        max_extend_tokens: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_rows = source.total_rows
        self.head_dim = source.row_bytes  # fp8: one byte per element
        self.dtype = dtype
        self.heads = int(hash_constants["num_ngram_heads"])
        self.scale = source.scale
        self.eos_token_id = int(hash_constants["eos_token_id"])
        self.image_token_id = hash_constants.get("image_token_id")
        mult = [int(x) for x in hash_constants["layer_multipliers"]]
        sizes = [int(x) for x in hash_constants["per_head_vocab_sizes"]]
        offsets = [int(x) for x in hash_constants["per_head_offsets"]]
        if len(mult) != 3 or self.heads % 2 or len(sizes) != self.heads or len(offsets) != self.heads:
            raise ValueError(
                f"PLE hash geometry: want 3 multipliers and {self.heads} (even) per-head sizes and offsets, "
                f"got {len(mult)}, {len(sizes)}, {len(offsets)}"
            )
        need = max(o + s for o, s in zip(offsets, sizes))
        if need > source.total_rows:
            raise ValueError(
                f"PLE row source holds {source.total_rows} rows but the hash addresses {need}; incomplete checkpoint?"
            )
        self._mult = np.array(mult, dtype=np.int64).view(np.uint64)
        self._sizes = np.array(sizes, dtype=np.int64)
        self._offsets = np.array(offsets, dtype=np.int64)
        # first half of the heads hash the 2-gram, second half the 3-gram (ple_store_ext.cpp:489)
        self._bigram_head = np.arange(self.heads) < self.heads // 2

        self._rows_per_extent = source.rows_per_extent
        self._row_stride = source.row_stride
        self._extent_fd = np.array(source.extent_file, dtype=np.int64)
        self._extent_base = np.array(source.extent_base, dtype=np.int64)
        # reject a truncated or still-downloading shard here, not as a short read mid-forward
        extent_bytes = (source.rows_per_extent - 1) * source.row_stride + source.row_bytes
        sizes_on_disk = [os.path.getsize(p) for p in source.paths]
        for file_idx, base in zip(source.extent_file, source.extent_base):
            if base + extent_bytes > sizes_on_disk[file_idx]:
                raise ValueError(
                    f"{source.paths[file_idx]}: extent needs {base + extent_bytes} bytes, "
                    f"file has {sizes_on_disk[file_idx]}"
                )
        self._fds = [os.open(p, os.O_RDONLY) for p in source.paths]
        self._pool = ThreadPoolExecutor(max_workers=_READ_THREADS, thread_name_prefix="ft-ple-disk")

        self._device = torch.device("mps")
        self._token_bytes = self.heads * self.head_dim
        max_tokens = max(max_graph_rows, max_extend_tokens)
        self._pinned, arena = device_backend.shared_host_arena(max_tokens * self._token_bytes)
        # zeroed through the numpy alias: .zero_() on a shared-heap tensor dispatches to mps, which
        # has no kernel for it. The warmup and graph capture stage nothing and read these zeros.
        arena[:] = 0
        self._arena = arena.reshape(max_tokens * self.heads, self.head_dim)
        self._dev = torch.empty(self._pinned.numel(), dtype=torch.uint8, device=self._device)
        # every fp8-e4m3 byte dequantized once, indexed by the raw byte (NaN encodings survive):
        # a gather replaces the cast MPS lacks; the oracle's scale op keeps it bit-identical.
        lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(dtype).to(self._device)
        self._lut = lut * self.scale if self.scale != 1.0 else lut
        self._slices: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._live: tuple[torch.Tensor, torch.Tensor] | None = None
        logger.info_rank0(f"PLE disk backend: preadv x{_READ_THREADS}, 256-entry fp8 LUT")

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for fd in self._fds:
            os.close(fd)
        self._fds = []

    # ---------------- host side (engine thread, before the forward launches) ----------------

    def hash_rows(self, run: np.ndarray) -> np.ndarray:
        """Global row ids ``[n, heads]`` for ``n = len(run) - 2`` tokens preceded by two context ids:
        the uint64 twin of ``PleStore::hash_rows``, eos barrier included."""
        cur, prev1, prev0 = run[2:], run[1:-1], run[:-2]
        prev2 = np.where(prev1 == self.eos_token_id, self.eos_token_id, prev0)
        bigram = (cur.view(np.uint64) * self._mult[0]) ^ (prev1.view(np.uint64) * self._mult[1])
        trigram = bigram ^ (prev2.view(np.uint64) * self._mult[2])
        mixed = np.where(self._bigram_head, bigram[:, None], trigram[:, None])
        return np.remainder(mixed.view(np.int64), self._sizes) + self._offsets

    def _read_rows(self, rows: np.ndarray) -> None:
        """Read every distinct row of ``rows`` into its first slot of the arena, then fan out the duplicates."""
        uniq, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
        extent = uniq // self._rows_per_extent
        offset = self._extent_base[extent] + (uniq % self._rows_per_extent) * self._row_stride
        fd_of = self._extent_fd[extent]
        # one monotonic seek chain per file: the drive reorders far worse than we can
        order = np.lexsort((offset, fd_of))
        spans = [s for s in np.array_split(np.arange(order.size), _READ_THREADS) if s.size]
        jobs = [(first[order[s]], fd_of[order[s]], offset[order[s]]) for s in spans]
        if len(jobs) > 1:
            list(self._pool.map(self._read_span, jobs))
        elif jobs:
            self._read_span(jobs[0])
        # every other copy of a row comes from the slot the read already filled
        src = first[inverse]
        dup = np.flatnonzero(src != np.arange(src.size))
        if dup.size:
            self._arena[dup] = self._arena[src[dup]]

    def _read_span(self, job) -> None:
        dst, file_of, offsets = job
        arena, row_bytes, fds = self._arena, self.head_dim, self._fds
        for i in range(dst.size):
            got = os.preadv(fds[file_of[i]], [memoryview(arena[dst[i]])], int(offsets[i]))
            if got != row_bytes:
                raise OSError(f"short PLE row read at offset {offsets[i]}: {got} of {row_bytes} bytes")

    def fill(self, runs: Sequence[torch.Tensor], *, graph: bool = False, tokens: int | None = None) -> None:
        """Stage per-request token runs (two context ids, then the new tokens) in batch order.

        ``tokens`` is the dispatch's row count when it exceeds what the runs stage; the padded decode
        lanes past those are zeroed. ``graph`` only keeps the two backends' signatures the same."""
        ids = [self.hash_rows(np.ascontiguousarray(run.numpy(), dtype=np.int64)) for run in runs]
        flat = (np.concatenate([a.reshape(-1) for a in ids]) if len(ids) > 1
                else ids[0].reshape(-1) if ids else np.empty(0, dtype=np.int64))
        staged = flat.size // self.heads
        tokens = staged if tokens is None else tokens
        if tokens * self._token_bytes > self._pinned.numel():
            raise ValueError(
                f"PLE staging holds {self._pinned.numel() // self._token_bytes} tokens, this forward wants {tokens}"
            )
        if flat.size:
            self._read_rows(flat)
        if staged < tokens:
            self._arena[staged * self.heads : tokens * self.heads] = 0  # padded decode lanes
        self._live = self._staging(tokens * self._token_bytes)

    def _staging(self, nbytes: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The one (host, device) slice pair a dispatch of this size always uses. Memoized because
        slicing the pinned CPU tensor inside a recorded decode makes a cpu tensor the tape cannot
        replay; the entries are views, so the dict holds no bytes."""
        pair = self._slices.get(nbytes)
        if pair is None:
            pair = (self._pinned[:nbytes], self._dev[:nbytes])
            self._slices[nbytes] = pair
        return pair

    def host_fill_batch(self, batch: Batch, use_graph: bool):
        """Stage this batch's rows. Metal has no captured WAIT to gate, so the fill is always inline."""
        # One rule for both paths: lookup's copy off the arena is lazy (moe/disk_cache.py:236) and
        # overlap scheduling drains nothing, so the preadv below could overwrite rows the previous
        # dispatch is still reading. Decode would also be ordered by its readback D2H, but resting
        # on that is an implicit contract; here the drain is free, the thread waits either way.
        if self._live is not None:
            device_backend.synchronize()
        if batch.is_decode:
            # padded lanes have no request, so the row count comes from input_ids, which is what
            # lookup is handed one id row of
            tokens = batch.input_ids.to("cpu").to(torch.int64).tolist()
            self.fill(self._decode_runs(batch.reqs, tokens), tokens=len(tokens))
            return None
        self.fill(self._prefill_runs(batch.padded_reqs))
        return None

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        """Around one dispatch: stage on enter; the tape's replay then reads a buffer already full."""
        self.host_fill_batch(batch, use_graph)
        yield

    # ---------------- device side (PLETableBackend protocol) ----------------

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        want = row_ids.shape[0] * self._token_bytes
        if self._live is None or self._live[0].numel() != want:
            # Graph capture and the prefill warmup dispatch without forward_host_ctx, so they stage
            # nothing; serve the arena the way the CUDA class serves its zeroed pinned buffer.
            self._live = self._staging(want)
        pinned, dev = self._live
        dev.copy_(pinned)
        values = torch.index_select(self._lut, 0, dev.to(torch.int32))
        values = values.view(*row_ids.shape[:-1], row_ids.shape[-1] * self.head_dim)
        if out is None:
            return values
        out.copy_(values)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None
