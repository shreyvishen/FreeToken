"""PLE row-read bench: which way to pull the n-gram rows off the SSD.

Four candidates over the same deduplicated, offset-sorted row list -- threaded pread
through the page cache, the same on ``F_NOCACHE`` descriptors, either plus an
``F_RDADVISE`` hint, and ``mmap`` + a numpy fancy index. Decode asks for
``heads`` rows per token; prefill asks for ``heads`` x chunk before dedup, over a Zipf-ish
row distribution, which is what a real prompt's n-grams look like.

Synthetic by default (a file it writes and deletes). ``--folder`` points it at a real
checkpoint and takes the row width and head count off it, which is the run that decides the
constant for the shipping backend.

**Read the residency line before the numbers.** macOS has no per-file page-cache drop --
``F_NOCACHE`` is a per-descriptor read policy, not a purge -- so a table that fits in RAM is
measured warm however it is opened, and the four strategies run in sequence over one row list.
Each token size therefore prints what ``mincore(2)`` says about those exact rows first: at 100%
resident, the columns are per-row overhead (syscall and thread cost for pread, fault and memcpy
for mmap) and say nothing about the drive. A cold number needs a table larger than RAM.

Run: PYTHONPATH=python python benchmarks/bench_ple_disk.py
     PYTHONPATH=python python benchmarks/bench_ple_disk.py --folder /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import mmap
import os
import shutil
import statistics
import struct
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

F_NOCACHE = 48
F_RDADVISE = 44
THREADS = 8
# Qwen3.8's geometry, used only to size the SYNTHETIC table; --folder takes both off the
# checkpoint, because reading the wrong length would bench the wrong thing and can run past an extent.
SYNTHETIC_ROW_BYTES = 160
SYNTHETIC_HEADS = 16


def open_fds(paths, *, nocache: bool):
    fds = [os.open(p, os.O_RDONLY) for p in paths]
    if nocache:
        for fd in fds:
            fcntl.fcntl(fd, F_NOCACHE, 1)
    return fds


def read_threaded(fds, arena, dst, fd_of, offset, row_bytes, *, hint: bool) -> None:
    if hint:
        for i in range(dst.size):
            fcntl.fcntl(fds[fd_of[i]], F_RDADVISE, struct.pack("qi", int(offset[i]), row_bytes))

    def span(sel):
        for i in sel:
            os.preadv(fds[fd_of[i]], [memoryview(arena[dst[i]])], int(offset[i]))

    chunks = [s for s in np.array_split(np.arange(dst.size), THREADS) if s.size]
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            list(pool.map(span, chunks))
    elif chunks:
        span(chunks[0])


def read_mmap(maps, arena, dst, fd_of, offset, row_bytes) -> None:
    for f in np.unique(fd_of):
        sel = np.flatnonzero(fd_of == f)
        rows = np.frombuffer(maps[f], dtype=np.uint8)
        idx = offset[sel][:, None] + np.arange(row_bytes)
        arena[dst[sel]] = rows[idx]


def resident_share(maps, fd_of, offset) -> float | None:
    """Share of the sampled rows whose page is already in core, from ``mincore(2)``; None if it fails.

    This is the honest answer to "was that first pass cold?" -- one syscall per mapped file, read
    off the mappings the mmap strategy already holds."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        page, hit, total = mmap.PAGESIZE, 0, 0
        for f in np.unique(fd_of):
            whole = np.frombuffer(maps[f], dtype=np.uint8)
            npages = (whole.size + page - 1) // page
            vec = ctypes.create_string_buffer(npages)
            if libc.mincore(ctypes.c_void_p(whole.ctypes.data), ctypes.c_size_t(whole.size), vec):
                return None
            flags = np.frombuffer(vec.raw, dtype=np.uint8, count=npages)
            sel = np.flatnonzero(fd_of == f)
            hit += int((flags[offset[sel] // page] & 1).sum())
            total += sel.size
        return hit / total if total else None
    except Exception:
        return None


def sample_rows(rng, total_rows: int, tokens: int, zipf: float, heads: int) -> np.ndarray:
    """A token's heads land in disjoint per-head bands; inside one, hot n-grams repeat."""
    band = total_rows // heads
    draw = (rng.zipf(zipf, size=(tokens, heads)) - 1) % band
    return (draw + np.arange(heads) * band).reshape(-1)


def plan(rows: np.ndarray, rows_per_extent: int, extent_fd, extent_base, row_stride: int):
    uniq, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    extent = uniq // rows_per_extent
    offset = extent_base[extent] + (uniq % rows_per_extent) * row_stride
    fd_of = extent_fd[extent]
    order = np.lexsort((offset, fd_of))
    return first[order], fd_of[order], offset[order], uniq.size


def bench(name, run, reps: int) -> tuple[str, float, float]:
    """First pass and the median of ``reps`` more. Neither is cold -- see the module docstring."""
    first = time.perf_counter()
    run()
    first = (time.perf_counter() - first) * 1e3
    rest = []
    for _ in range(reps):
        t = time.perf_counter()
        run()
        rest.append((time.perf_counter() - t) * 1e3)
    return name, first, statistics.median(rest)


def synthetic_source(tmpdir: str, mib: int):
    """(paths, extent_file, extent_base, rows_per_extent, row_stride, row_bytes, heads)."""
    path = os.path.join(tmpdir, "ple-rows.bin")
    row_bytes, heads = SYNTHETIC_ROW_BYTES, SYNTHETIC_HEADS
    rows = (mib * 2**20) // row_bytes
    rows -= rows % heads
    with open(path, "wb") as fh:
        block = np.random.default_rng(0).integers(0, 256, size=(1 << 20,), dtype=np.uint8).tobytes()
        written = 0
        while written < rows * row_bytes:
            fh.write(block[: min(len(block), rows * row_bytes - written)])
            written += len(block)
    zero = np.zeros(1, dtype=np.int64)
    return [path], zero, zero, rows, row_bytes, row_bytes, heads


def checkpoint_source(folder: str):
    """The same tuple, with the row width and head count taken off the checkpoint, not assumed."""
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.ple_disk import resolve_row_source
    from freetoken.utils import cached_load_hf_config

    src = resolve_row_source(folder)
    qwen4 = parse_config(cached_load_hf_config(folder)).qwen4_args
    if src.row_bytes != qwen4.ngram_head_dim:
        raise ValueError(
            f"{folder}: the table's rows are {src.row_bytes} bytes but the config says "
            f"ngram_head_dim={qwen4.ngram_head_dim}"
        )
    return (src.paths, np.array(src.extent_file, dtype=np.int64),
            np.array(src.extent_base, dtype=np.int64), src.rows_per_extent, src.row_stride,
            src.row_bytes, qwen4.num_ngram_heads)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", help="checkpoint dir; default is a synthetic file in a temp dir")
    ap.add_argument("--mib", type=int, default=256, help="synthetic table size (MiB)")
    ap.add_argument("--tokens", type=int, default=[1, 16, 256, 2048], nargs="+")
    ap.add_argument("--zipf", type=float, default=1.3)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    tmpdir = None if args.folder else tempfile.mkdtemp(prefix="ple-bench-")
    try:
        source = checkpoint_source(args.folder) if args.folder else synthetic_source(tmpdir, args.mib)
        paths, extent_fd, extent_base, rows_per_extent, row_stride, row_bytes, heads = source
        total_rows = rows_per_extent * len(extent_base)
        size_gib = sum(os.path.getsize(p) for p in paths) / 2**30
        print(f"table: {len(paths)} file(s), {total_rows} rows of {row_bytes} B, {heads} heads, "
              f"{size_gib:.2f} GiB, {THREADS} reader threads, zipf {args.zipf}")
        print("NOTE: other processes share this SSD and its page cache, so every number is a floor.")
        print(f"{'tokens':>7} {'rows':>8} {'uniq':>8}  {'strategy':<22} {'first ms':>9} {'med ms':>9} {'us/row':>8}")

        rng = np.random.default_rng(7)
        cached = open_fds(paths, nocache=False)
        nocache = open_fds(paths, nocache=True)
        map_fds = open_fds(paths, nocache=False)
        maps = [mmap.mmap(fd, 0, prot=mmap.PROT_READ) for fd in map_fds]
        for fd in map_fds:
            os.close(fd)  # the mapping keeps its own reference
        try:
            for tokens in args.tokens:
                rows = sample_rows(rng, total_rows, tokens, args.zipf, heads)
                dst, fd_of, offset, uniq = plan(rows, rows_per_extent, extent_fd, extent_base, row_stride)
                share = resident_share(maps, fd_of, offset)
                print(f"  tokens={tokens}: " + ("mincore unavailable, so the first pass is the"
                      " closest thing to cold this machine allows"
                      if share is None else f"{share:.0%} of these rows are already in core"
                      " before the first pass"))
                arena = np.zeros((rows.size, row_bytes), dtype=np.uint8)
                cases = [
                    ("pread + page cache",
                     lambda: read_threaded(cached, arena, dst, fd_of, offset, row_bytes, hint=False)),
                    ("pread F_NOCACHE",
                     lambda: read_threaded(nocache, arena, dst, fd_of, offset, row_bytes, hint=False)),
                    ("pread + F_RDADVISE",
                     lambda: read_threaded(cached, arena, dst, fd_of, offset, row_bytes, hint=True)),
                    ("mmap + fancy index", lambda: read_mmap(maps, arena, dst, fd_of, offset, row_bytes)),
                ]
                for name, run in cases:
                    label, first, med = bench(name, run, args.reps)
                    print(f"{tokens:>7} {rows.size:>8} {uniq:>8}  {label:<22} "
                          f"{first:>9.2f} {med:>9.2f} {med * 1e3 / uniq:>8.2f}")
        finally:
            for m in maps:
                m.close()
            for fd in cached + nocache:
                os.close(fd)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
