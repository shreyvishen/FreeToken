"""Expert-major repack of an NVFP4 checkpoint: each expert's nine raw tensors written
contiguous and 4096-aligned, so a runtime fetch of one expert is one aligned ``pread``
instead of nine scattered ones."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from freetoken.moe.expert_reader import (
    PIECE_ORDER,
    ExpertLocation,
    Nvfp4DiskIndex,
    config_sha,
)

MAX_THREADS = 8  # machine safety: I/O thread pools capped at 8
ALIGN = 4096


def build(model_path: str, out_dir: str, layers: list[int], threads: int = MAX_THREADS) -> dict:
    model_path = os.path.expanduser(model_path)
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    index = Nvfp4DiskIndex(model_path)
    threads = min(threads, MAX_THREADS)

    fds: dict[str, int] = {}

    def _read(loc) -> bytes:
        fd = fds.get(loc.shard_path)
        if fd is None:
            # setdefault would evaluate os.open on every call and leak one fd per read
            fd = fds[loc.shard_path] = os.open(loc.shard_path, os.O_RDONLY)
        return os.pread(fd, loc.nbytes, loc.offset)

    experts_meta: dict[str, dict] = {}
    bin_path = os.path.join(out_dir, "experts.bin")
    bin_tmp = bin_path + ".tmp"
    pos = 0
    try:
        with open(bin_tmp, "wb") as out, ThreadPoolExecutor(max_workers=threads) as pool:
            for layer in layers:
                for expert in index.experts_per_layer(layer):
                    loc: ExpertLocation = index.get(layer, expert)
                    pad = (-pos) % ALIGN
                    if pad:
                        out.write(b"\0" * pad)
                        pos += pad
                    start = pos
                    piece_spans = {}
                    pieces = list(
                        zip(PIECE_ORDER, pool.map(_read, (getattr(loc, n) for n in PIECE_ORDER)))
                    )
                    for name, data in pieces:
                        piece_spans[name] = [pos - start, len(data)]
                        out.write(data)
                        pos += len(data)
                    experts_meta[f"{layer}:{expert}"] = {
                        "offset": start,
                        "size": pos - start,
                        "pieces": piece_spans,
                    }
    finally:
        for fd in fds.values():
            os.close(fd)

    sizes = {e["size"] for e in experts_meta.values()}
    if len(sizes) != 1:
        raise ValueError(f"experts vary in size ({sorted(sizes)}), no fixed layout")
    expert_nbytes = sizes.pop()

    meta = {
        "order": list(PIECE_ORDER),
        "align": ALIGN,
        "source_model": model_path,
        "source_config_sha256": config_sha(model_path),
        "expert_nbytes": expert_nbytes,
        "experts": experts_meta,
    }
    index_path = os.path.join(out_dir, "experts.index.json")
    index_tmp = index_path + ".tmp"
    with open(index_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Publish atomically, bin before index: the index is the completion marker a reader
    # trusts, so it must never point at a bin file that isn't fully in place yet.
    os.replace(bin_tmp, bin_path)
    os.replace(index_tmp, index_path)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all", help="comma-separated layer indices, or 'all'")
    ap.add_argument("--threads", type=int, default=MAX_THREADS, help=f"capped at {MAX_THREADS}")
    args = ap.parse_args(argv)

    if args.layers == "all":
        layers = Nvfp4DiskIndex(os.path.expanduser(args.model)).layers()
    else:
        layers = [int(x) for x in args.layers.split(",")]

    meta = build(args.model, args.out, layers, threads=args.threads)
    n_experts = len(meta["experts"])
    total_bytes = sum(e["size"] for e in meta["experts"].values())
    print(f"wrote {args.out}: {n_experts} experts across layers {layers}, "
          f"{total_bytes / (1 << 20):.1f} MiB, expert_nbytes={meta['expert_nbytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
