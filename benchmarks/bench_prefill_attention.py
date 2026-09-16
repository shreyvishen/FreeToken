"""Prefill attention on MPS: the tiled Metal kernel against the SDPA fallback it replaces,
and against MLX's own fused attention at the same shape.

Synthetic, no checkpoint. Qwen3.6-35B's attention shape by default (16 query heads, 2 K/V
heads, head_dim 256, bf16), one request per batch, KV slots scattered as a paged pool hands
them out. Each (path, shape) runs in its own subprocess: ``driver_allocated_memory`` reads a
pool that never shrinks, so one process per row is what makes the peak mean anything.

``--mlx`` adds mlx.core.fast.scaled_dot_product_attention, which is MLX's fused flash
attention, at the same shapes and dtype. The comparison is not quite even and it favours MLX:
its K/V are contiguous while ours are gathered through a page table, which is about a third of
our kernel's time. Read the MLX column as the ceiling a paged kernel is chasing, not as a
like-for-like loss.

    PYTHONPATH=python:. python benchmarks/bench_prefill_attention.py --mlx
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

import torch

SHAPES = ((512, 512), (2048, 2048), (8192, 8192), (8192, 16384), (8192, 32768))
MIB = 1 << 20
GIB = 1 << 30


def host_mem() -> tuple[float, float]:
    """(free GiB, swap used GiB). The SDPA baseline's scores are what push a 36 GiB box into
    swap, so the table has to show that and not only the time."""
    pages = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    size = int(re.search(r"page size of (\d+)", pages).group(1))
    free = sum(int(re.search(rf"Pages {n}:\s+(\d+)", pages).group(1))
               for n in ("free", "speculative"))
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
    used = float(re.search(r"used = ([\d.]+)M", swap.stdout).group(1))
    return free * size / GIB, used / 1024


def _run_mlx(q_len: int, kv_len: int, heads: int, kv_heads: int, head_dim: int,
             iters: int) -> dict:
    """MLX's fused attention at the same shape. Its own peak counter, not torch's: the two
    frameworks allocate from different pools, which is also why this runs in its own process."""
    import mlx.core as mx

    scale = head_dim ** -0.5
    q = mx.random.normal((1, heads, q_len, head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, kv_heads, kv_len, head_dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, kv_heads, kv_len, head_dim)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    iters = max(iters, min(50, int(2e8 / (q_len * kv_len))))

    def once():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

    mx.eval(once())
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(once())
    dt = (time.perf_counter() - t0) / iters
    return {"ms": dt * 1e3, "iters": iters, "peak_mib": mx.get_peak_memory() / MIB}


def _run(path: str, q_len: int, kv_len: int, heads: int, kv_heads: int, head_dim: int,
         iters: int) -> dict:
    if path == "mlx":
        return _run_mlx(q_len, kv_len, heads, kv_heads, head_dim, iters)
    from freetoken.kernel.metal import attention

    if path == "sdpa":
        # Route through the module's own fallback loop rather than a copy of it, so the
        # baseline is the code the kernel displaces and not a re-derivation of it.
        attention.prefill_tile = lambda *_args: None

    dev = torch.device("mps")
    dtype = torch.bfloat16
    q = torch.randn(q_len, heads, head_dim, dtype=dtype, device=dev)
    k = torch.randn(kv_len, kv_heads, head_dim, dtype=dtype, device=dev)
    v = torch.randn(kv_len, kv_heads, head_dim, dtype=dtype, device=dev)
    indices = torch.randperm(kv_len, device=dev).to(torch.int32)
    indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=dev)
    args = (q, k, v, indptr, indices, cu_q, head_dim**-0.5, [0, kv_len], [0, q_len])

    # A small shape is launch-bound, so five passes of it is noise, not a measurement.
    iters = max(iters, min(50, int(2e8 / (q_len * kv_len))))
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
    # The SDPA path holds heads x q_len x kv_len fp32 scores at once. Past this it swaps a
    # 36 GiB box rather than reporting a time, so the row says what it would have needed.
    ap.add_argument("--sdpa-score-cap-gib", type=float, default=10.0)
    ap.add_argument("--mlx", action="store_true", help="also time MLX's fused attention")
    ap.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        path, q_len, kv_len = args.child[0], int(args.child[1]), int(args.child[2])
        print(json.dumps(_run(path, q_len, kv_len, args.heads, args.kv_heads, args.head_dim,
                              args.iters)))
        return

    free0, swap0 = host_mem()
    print(f"HQ={args.heads} HKV={args.kv_heads} D={args.head_dim} bf16 bs=1; host free "
          f"{free0:.1f} GiB, swap used {swap0:.1f} GiB. The MiB columns are the process's "
          f"total MPS driver bytes.\n")
    mlx_head = f" {'mlx ms':>9} {'MiB':>8} |" if args.mlx else ""
    print(f"{'q_len':>7} {'kv_len':>7} | {'kernel ms':>9} {'MiB':>8} |{mlx_head} "
          f"{'sdpa ms':>9} {'MiB':>8} | {'speedup':>7} | {'free GiB':>8} {'swap GiB':>8}")
    print("-" * (95 + (20 if args.mlx else 0)))
    for q_len, kv_len in SHAPES:
        row = {}
        scores = args.heads * q_len * kv_len * 4 / GIB
        for path in (("tiled", "mlx", "sdpa") if args.mlx else ("tiled", "sdpa")):
            free, _ = host_mem()
            # The SDPA scores are the whole reason for this commit; measuring them past the
            # point where they swap costs minutes and risks the box, so the row says what it
            # would have needed instead. The measured peak runs about 2.4x this estimate.
            if path == "sdpa" and (scores > args.sdpa_score_cap_gib or free < 2.4 * scores):
                continue
            out = subprocess.run(
                [sys.executable, __file__, "--child", path, str(q_len), str(kv_len),
                 "--heads", str(args.heads), "--kv-heads", str(args.kv_heads),
                 "--head-dim", str(args.head_dim), "--iters", str(args.iters)],
                capture_output=True, text=True)
            row[path] = json.loads(out.stdout) if out.returncode == 0 else None
            if row[path] is None:
                print(f"  {path} {q_len}x{kv_len} did not run:\n{out.stderr[-400:]}",
                      file=sys.stderr, flush=True)
        t, s, m = row.get("tiled"), row.get("sdpa"), row.get("mlx")
        free, swap = host_mem()
        left = (f"{t['ms']:>9.2f} {t['peak_mib']:>8.0f}" if t else f"{'-':>9} {'-':>8}")
        if args.mlx:
            left += " |" + (f" {m['ms']:>9.2f} {m['peak_mib']:>8.0f}" if m
                            else f" {'-':>9} {'-':>8}")
        if s:
            right = f"{s['ms']:>9.2f} {s['peak_mib']:>8.0f}"
            speed = f"{s['ms'] / t['ms']:>6.2f}x" if t else f"{'-':>7}"
            note = ""
        else:
            right = f"{'did not fit':>18}"
            speed = f"{'-':>7}"
            note = f"   (sdpa needs ~{2.4 * scores:.0f} GiB for {scores:.1f} GiB of scores)"
        print(f"{q_len:>7} {kv_len:>7} | {left} | {right} | {speed} | {free:>8.1f} "
              f"{swap:>8.1f}{note}", flush=True)


if __name__ == "__main__":
    main()
