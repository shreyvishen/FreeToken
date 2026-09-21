"""NVFP4 MoE prefill on MPS: the tiled grouped GEMM against the row-grouped kernel it replaces.

Synthetic, no checkpoint. Qwen3.6-35B's MoE shape by default (hidden 2048, moe_intermediate
512, 256 experts, top-8), one layer per timed call, routing drawn uniformly so every expert
gets a run of about ``M * top_k / E`` routes.

FLOPs count the two expert GEMMs only, ``routes * (2*2*INTER*H + 2*H*INTER)``, so the
TFLOP/s column is comparable across both paths.

    PYTHONPATH=python:. python benchmarks/bench_moe_prefill.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import torch

from freetoken.kernel.metal.nvfp4 import moe_prefill_nvfp4_grouped


def make_args(m: int, h: int, inter: int, experts: int, top_k: int, seed: int) -> list:
    """One layer's worth of NVFP4 banks, activations and routing, all on the GPU."""
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("mps")
    out = [torch.randn(m, h, generator=g).to(dev)]
    for rows, cols in ((2 * inter, h), (h, inter)):
        out.append(torch.randint(0, 256, (experts, rows, cols // 2), dtype=torch.uint8,
                                 generator=g).to(dev))
        scale = (torch.randn(experts, rows, cols // 16, generator=g) * 0.2 + 1.0)
        out.append(scale.to(torch.float8_e4m3fn).view(torch.uint8).to(dev))
        out.append((torch.rand(experts, rows, generator=g) * 0.5 + 0.1).to(torch.float16).to(dev))
    ids = torch.stack([torch.randperm(experts, generator=g)[:top_k] for _ in range(m)])
    w = torch.rand(m, top_k, generator=g)
    return out + [(w / w.sum(-1, keepdim=True)).to(dev), ids.int().to(dev)]


def time_ms(fn, iters: int) -> float:
    for _ in range(2):
        fn()
    torch.mps.synchronize()
    start, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.mps.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--m", default="512,2048,8192", help="comma list of token counts")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--child", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not torch.backends.mps.is_available():
        sys.exit("needs an Apple GPU")
    if args.child:
        # One M per process: sustained load clocks this machine down, and in sequence the
        # row-grouped baseline drifts from 383 ms to 468 ms by the third row.
        a = make_args(args.child, args.hidden, args.inter, args.experts, args.top_k, args.child)
        a[0] = a[0].to(torch.bfloat16).float()   # the engine's input: a widened bf16 residual
        print(json.dumps({
            "tiled": time_ms(lambda: moe_prefill_nvfp4_grouped(*a), args.iters),
            "grouped": time_ms(lambda: moe_prefill_nvfp4_grouped(*a, group_rows=2), args.iters),
        }))
        return

    print(f"H={args.hidden} INTER={args.inter} E={args.experts} top_k={args.top_k}, "
          f"one layer, {args.iters} timed iterations after 2 warm-ups, one process per row\n")
    print(f"{'M':>6} {'routes':>8} {'GFLOP':>8} | {'tiled ms':>9} {'TFLOP/s':>8} | "
          f"{'grouped ms':>11} {'TFLOP/s':>8} | {'speedup':>8}")
    print("-" * 82)

    for m in [int(x) for x in args.m.split(",") if x.strip()]:
        out = subprocess.run(
            [sys.executable, __file__, "--child", str(m), "--hidden", str(args.hidden),
             "--inter", str(args.inter), "--experts", str(args.experts),
             "--top-k", str(args.top_k), "--iters", str(args.iters)],
            capture_output=True, text=True)
        if out.returncode != 0:
            print(f"  M={m} did not run:\n{out.stderr[-500:]}", file=sys.stderr, flush=True)
            continue
        r = json.loads(out.stdout)
        flop = m * args.top_k * 6 * args.inter * args.hidden
        print(f"{m:>6} {m * args.top_k:>8} {flop / 1e9:>8.1f} | {r['tiled']:>9.2f} "
              f"{flop / r['tiled'] / 1e9:>8.2f} | {r['grouped']:>11.2f} "
              f"{flop / r['grouped'] / 1e9:>8.2f} | {r['grouped'] / r['tiled']:>7.2f}x",
              flush=True)


if __name__ == "__main__":
    main()
