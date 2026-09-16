"""Prefill attention on MPS: the tiled Metal kernel, and with ``--mlx`` MLX's own fused
attention (mlx.core.fast.scaled_dot_product_attention) at the same shape and dtype.

Synthetic, no checkpoint. Qwen3.6-35B's attention shape by default (16 query heads, 2 K/V
heads, head_dim 256, bf16), one request per batch, KV slots scattered as a paged pool hands
them out. Each (path, shape) runs in its own subprocess: ``driver_allocated_memory`` reads a
pool that never shrinks, so one process per row is what makes the peak mean anything.

The comparison favours MLX: its K/V are contiguous while ours are gathered through a page
table, about a third of our kernel's time. Read the MLX column as the ceiling a paged kernel is
chasing, not as a like-for-like loss.

    PYTHONPATH=python:. python benchmarks/bench_prefill_attention.py --mlx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import torch

SHAPES = ((512, 512), (2048, 2048), (8192, 8192), (8192, 16384), (8192, 32768))
MIB = 1 << 20


def _run(path: str, q_len: int, kv_len: int, heads: int, kv_heads: int, head_dim: int,
         iters: int) -> dict:
    # A small shape is launch-bound, so five passes of it is noise, not a measurement.
    iters = max(iters, min(50, int(2e8 / (q_len * kv_len))))
    if path == "mlx":
        # MLX's own peak counter, not torch's: the two frameworks allocate from different pools.
        import mlx.core as mx

        q = mx.random.normal((1, heads, q_len, head_dim)).astype(mx.bfloat16)
        k = mx.random.normal((1, kv_heads, kv_len, head_dim)).astype(mx.bfloat16)
        v = mx.random.normal((1, kv_heads, kv_len, head_dim)).astype(mx.bfloat16)
        mx.eval(q, k, v)

        def once():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5,
                                                        mask="causal")

        mx.eval(once())
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(once())
        dt = (time.perf_counter() - t0) / iters
        return {"ms": dt * 1e3, "iters": iters, "peak_mib": mx.get_peak_memory() / MIB}
    from freetoken.kernel.metal import attention

    dev = torch.device("mps")
    dtype = torch.bfloat16
    q = torch.randn(q_len, heads, head_dim, dtype=dtype, device=dev)
    k = torch.randn(kv_len, kv_heads, head_dim, dtype=dtype, device=dev)
    v = torch.randn(kv_len, kv_heads, head_dim, dtype=dtype, device=dev)
    indices = torch.randperm(kv_len, device=dev).to(torch.int32)
    indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=dev)
    args = (q, k, v, indptr, indices, cu_q, head_dim**-0.5, [0, kv_len], [0, q_len])
    for _ in range(2):
        attention.paged_attention_mps(*args)
    torch.mps.synchronize()
    start = torch.mps.Event(enable_timing=True)
    end = torch.mps.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        attention.paged_attention_mps(*args)
    end.record()
    torch.mps.synchronize()
    # Absolute, not a delta: torch suballocates from MTLHeaps, so an output that fits slack
    # an input's heap already holds costs zero extra driver bytes and reads as 0 MiB.
    return {"ms": start.elapsed_time(end) / iters, "iters": iters,
            "peak_mib": torch.mps.driver_allocated_memory() / MIB}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--mlx", action="store_true", help="also time MLX's fused attention")
    ap.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        path, q_len, kv_len = args.child[0], int(args.child[1]), int(args.child[2])
        print(json.dumps(_run(path, q_len, kv_len, args.heads, args.kv_heads, args.head_dim,
                              args.iters)))
        return

    print(f"HQ={args.heads} HKV={args.kv_heads} D={args.head_dim} bf16 bs=1. The MiB columns "
          f"are the process's total MPS driver bytes.\n")
    paths = ("kernel", "mlx") if args.mlx else ("kernel",)
    head = "".join(f" | {p + ' ms':>9} {'MiB':>8}" for p in paths)
    print(f"{'q_len':>7} {'kv_len':>7}{head}")
    print("-" * (15 + 21 * len(paths)))
    for q_len, kv_len in SHAPES:
        cells = []
        for path in paths:
            out = subprocess.run(
                [sys.executable, __file__, "--child", path, str(q_len), str(kv_len),
                 "--heads", str(args.heads), "--kv-heads", str(args.kv_heads),
                 "--head-dim", str(args.head_dim), "--iters", str(args.iters)],
                capture_output=True, text=True)
            if out.returncode:
                print(f"  {path} {q_len}x{kv_len} did not run:\n{out.stderr[-400:]}",
                      file=sys.stderr, flush=True)
                cells.append(f"{'-':>9} {'-':>8}")
            else:
                r = json.loads(out.stdout)
                cells.append(f"{r['ms']:>9.2f} {r['peak_mib']:>8.0f}")
        print(f"{q_len:>7} {kv_len:>7} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
