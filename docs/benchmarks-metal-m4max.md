# Metal campaign -- M4 Max, 35B/122B MoE decode

Median of 3 runs, AC power, idle machine. `bench_decode_moe.py` only; run from the repo
root with `PYTHONPATH=python`.

| Engine | Model + quantization | Backend / cache | Decode tok/s (median of 3) | TTFT | Command |
|---|---|---|---|---|---|
| FreeToken | Qwen3.6-35B-A3B, NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4) | metal (fused, resident) | 110.13 (runs: 110.27, 103.89, 110.13) | 326.9 (runs: 244.8, 326.9, 345.1) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --backend metal` |
| FreeToken | Qwen3.6-35B-A3B, NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4) | offload, 25% cache from SSD | 12.97 (runs: 12.97, 12.55, 13.24) | 2737.0 (runs: 2737.0, 2815.6, 2705.3) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --backend offload --cache 2560` |
| FreeToken | Qwen3.5-122B-A10B, NVFP4 (nvidia/Qwen3.5-122B-A10B-NVFP4) | offload, from SSD | 4.12 (runs: 4.15, 4.12, 3.94) | 7763.8 (runs: 7763.8, 7766.6, 7075.1) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.5-122b-a10b-nvfp4 --backend offload` |
| FreeToken | Qwen3.5-122B-A10B, NVFP4 (nvidia/Qwen3.5-122B-A10B-NVFP4) | offload, from SSD, `--dense-quant-override fp8` | 4.96 (runs: 4.79, 5.09, 4.96) | 6607.6 (runs: 6607.6, 6489.5, 6653.6) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.5-122b-a10b-nvfp4 --backend offload --dense-quant-override fp8` |
| FreeToken | Qwen3.6-35B-A3B, NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4) | offload, 25% cache from SSD, readahead hint | 23.78 (runs: 23.78, 24.10, 22.98) | 2115.3 (runs: 2393.4, 1935.2, 2115.3) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --backend offload --cache 2560` |
| FreeToken | Qwen3.5-122B-A10B, NVFP4 (nvidia/Qwen3.5-122B-A10B-NVFP4) | offload, from SSD, readahead hint | 5.04 (runs: 5.04, 5.21, 4.85) | 8386.3 (runs: 6804.8, 8995.6, 8386.3) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.5-122b-a10b-nvfp4 --backend offload` |

The readahead rows (11 Sep 2026) are the same commands on the tree with `F_RDADVISE` issued
before each expert read. The hint fills the unified buffer cache whatever `F_NOCACHE` says, so
the 35B row's expert set (15 GiB) ends up served from memory on this 36 GiB box; the 122B's
(61 GiB) does not fit and gains only the read/compute overlap.

## Prefill -- M4 Max, 35B NVFP4 resident (16 Sep 2026)

Prefill tok/s is `(prompt_tokens(N) - prompt_tokens(base)) / (median t(N) - median t(base))`, every
request `max_tokens=1` and `temperature=0`, five repetitions with a fresh salt each so nothing is
served from a cache, one server process per column, AC power and an idle machine. Token counts are
chosen to match `llama-bench`'s `pN` rows, which time prompt processing alone.

| Test | Tokens | Before (SDPA prefill, grouped MoE) | After | Speedup |
|---|---|---|---|---|
| pp512 | 516 | 694.1 (669.9-696.4) | **906.3** (871.0-912.6) | 1.31x |
| pp2048 | 2,053 | 558.3 (556.5-564.7) | **870.0** (864.7-898.4) | 1.56x |
| pp8192 | 8,192 | 153.3 (142.5-171.1) | **897.6** (849.7-899.6) | **5.86x** |

Greedy ids are identical to the reference on both trees. The shape matters as much as the ratio:
before, prefill fell 78 % from 512 to 8,192 tokens (694 -> 153); after, it is flat within 1 %
(906 -> 898). The long-prompt collapse is gone, and with it the swap thrash -- an 8,192-token chunk
used to allocate 4.3 GiB of fp32 attention scores per full-attention layer.

Against `llama-bench` on the same machine (build 89fe242, Qwen3.5-9B Q4_K_M, 5.28 GiB, 3 reps):

| Test | FreeToken, 35B-A3B NVFP4 | llama.cpp, 9B Q4_K_M | ratio |
|---|---|---|---|
| pp512 | **906.3** | 663.31 +- 0.79 | 1.37x |
| pp2048 | **870.0** | 658.10 +- 3.24 | 1.32x |
| pp8192 | **897.6** | 612.01 +- 5.47 | 1.47x |

Different models -- a 35B-A3B MoE with 3B active against a dense 9B -- so this is not a
like-for-like head-to-head, and llama.cpp's peak RSS is 5.72 GiB against our 26.3 GiB. It is the
closest comparison the checkpoints on this disk allow.

### Against MLX's own attention, at identical shapes

The tiled kernel against `mlx.core.fast.scaled_dot_product_attention` at the 35B's attention shape
(16 query heads, 2 K/V heads, head_dim 256, bf16, one request), each row in its own process:

```
PYTHONPATH=python:. python benchmarks/bench_prefill_attention.py --mlx
```

| q_len x kv_len | FreeToken kernel | MLX |
|---|---|---|
| 512 x 512 | **0.86 ms**, 59 MiB | 0.92 ms, 21 MiB |
| 2048 x 2048 | 8.12 ms, 1,051 MiB | **7.62 ms**, 184 MiB |
| 8192 x 8192 | **121.74 ms**, 1,051 MiB | 124.65 ms, 2,320 MiB |
| 8192 x 16384 | 378.51 ms, **1,051 MiB** | **248.29 ms**, 4,448 MiB |
| 8192 x 32768 | 926.06 ms, **1,049 MiB** | **503.31 ms**, 8,704 MiB |

Within 2 % of MLX through 8192 x 8192; 1.52x and 1.84x behind at 16k and 32k. Memory goes the other
way and the gap widens with context: flat at ~1,050 MiB against MLX's 8,704 MiB at 32k, 8.3x less.
MLX's peak tracks `heads x q_len x kv_len x 2 B` (8,448 MiB measured against 8,192 predicted, and
`mask=None` reads the same), so at these shapes MLX materialises the bf16 score matrix rather than
running a memory-flat path -- it buys its speed with memory we do not spend.

Two caveats that favour MLX in the time column: its K and V are contiguous where ours are gathered
through a page table, about 37 % of our kernel's time by ablation; and the two memory columns are
different counters (torch driver bytes against `mx.get_peak_memory`), so compare how they scale
rather than their floors.

## Prefix cache -- second-turn TTFT on a conversation (16 Sep 2026)

`hybrid_radix` is the default for GDN models on Metal from the commit that emits the per-chunk
GDN state. Turn 2 of a conversation restores the newest state snapshot at or before the shared
prefix and prefills only the tail, instead of recomputing the whole prompt.

SCREEN, 2 repetitions, AC, idle machine, 35B NVFP4 resident:

| Conversation | Cache type | Turn 1 TTFT (s) | Turn 2 TTFT (s) | Turn 2 tokens prefilled |
|---|---|---|---|---|
| 4,565 tokens | `naive` | 9.436 | 10.730 | 4,565 of 4,565 |
| 4,565 tokens | `hybrid_radix` | 10.363 | **0.631** | **12** of 4,565 |
| 18,065 tokens | `naive` | 59.124 | 62.831 | 9,873 of 18,065 (chunked) |
| 18,065 tokens | `hybrid_radix` | 54.066 | **0.867** | **12** of 18,065 |

Second-turn TTFT is 17.0x lower on the short conversation and 72.4x lower on the long one, and the
generated text is byte-identical between the two cache types at both sizes. The reuse is what
scales: turn 2 prefills 12 tokens whatever the history length.

Turn 1 pays for the snapshots. On the 4,565-token conversation that is 0.93 s, about 10 %; measured
directly against prefill, the per-chunk state copy-out costs 27.8 % of a 722-token prefill, 10.0 %
of a 2,873-token one and 8.1 % of an 11,474-token one, because the extra launches are a fixed count
per 64-token chunk. At 18k it is inside the run-to-run spread. `--cache-type naive` opts out.

```
REPS=2 N=3000 bash .notes/briefs/screen_prefix.sh 30421 p6-prefix-hybrid
REPS=2 N=3000 bash .notes/briefs/screen_prefix.sh 30422 p6-prefix-naive --cache-type naive
```
