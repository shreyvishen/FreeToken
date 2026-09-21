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

## Same model, three engines -- M4 Max, Qwen3.6-35B-A3B (21 Sep 2026)

One base model in each engine's own 4-bit format, resident on the GPU, same machine, AC power, one
engine at a time with the GPU to itself. Five repetitions per row: the figure is the median, the
range is min-max. `pp` is prompt processing alone and `tg` is generation, both in tokens per
second, with the names and lengths `llama-bench` uses.

| Test | FreeToken, NVFP4 | llama.cpp, UD-Q4_K_M | mlx_lm, 4-bit |
|---|---|---|---|
| pp512 | 940.9 (903.8-942.8) | 1238.6 (1232.6-1250.0) | **1248.1** (1229.5-1260.9) |
| pp2048 | 882.8 (863.1-906.9) | 1218.1 (1137.9-1222.0) | **1481.7** (1475.2-1487.6) |
| pp4096 | 916.5 (908.1-923.0) | 1185.6 (1163.2-1187.6) | **1461.9** (1455.6-1469.9) |
| pp8192 | 878.5 (873.9-887.9) | 1068.5 (1060.0-1076.4) | **1397.9** (1338.8-1402.5) |
| tg128 | 108.3 (107.5-108.9) | 67.5 (67.0-68.2) | **113.7** (113.1-113.9) |
| tg512 | 106.9 (103.8-107.3) | 66.6 (66.4-66.8) | **112.5** (112.2-112.7) |
| peak memory | 25.5 GiB (server) | 20.7 GiB (RSS) | 20.5 GiB (MLX peak, 21.99 GB) |

MLX leads every row, though at pp512 it and llama.cpp are within 1 %. Over FreeToken it leads
prefill by 1.33x to 1.68x and decode by 1.05x. llama.cpp leads FreeToken at prefill by 1.22x to
1.38x, and FreeToken leads llama.cpp at decode by 1.6x. This replaces the earlier comparison
against a dense 9B, which was the only GGUF on disk at the time and flattered our prefill.

Checkpoints: `nvidia/Qwen3.6-35B-A3B-NVFP4`; `unsloth/Qwen3.6-35B-A3B-GGUF` `UD-Q4_K_M` (an
Unsloth Dynamic quant, so precision varies by layer); `mlx-community/Qwen3.6-35B-A3B-4bit`.
llama.cpp build 89fe242, Metal + BLAS, all layers on the GPU, flash attention `auto`, f16 KV.
mlx_lm 0.31.1. The three peak-memory figures are three different counters; compare them loosely.

All three columns come from one 66-minute window with nothing else on the GPU. An earlier pass
of the FreeToken and llama.cpp columns, taken while other processes held memory, read within 1.5 %
of these on every row but two: FreeToken pp8192 read 797.5 (739.9-881.3) and llama.cpp pp4096
read 1129.9 (1106.6-1182.9). Two rows here spread more than 5 % of their median and were run
again: llama.cpp pp2048, whose earlier pass read 1219.4 (1218.5-1223.2), and mlx_lm pp8192,
whose first run read 1338.9 (1330.4-1399.1) and whose second is in the table.

```
# FreeToken prefill, by the method of the Prefill section below (server-counted tokens 512 / 2048 / 4096 / 8192)
PYTHONPATH=python python -m freetoken.cli serve --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --moe-backend fused --num-tokens 32768
# FreeToken decode
PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --backend metal --decode 512 --decode-reps 5 --prefill ''
# llama.cpp
llama-bench -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p 512,2048,4096,8192 -n 128,512 -r 5
# mlx_lm, one process per row
python -m mlx_lm benchmark --model ~/assets/models/qwen3.6-35b-a3b-mlx -p 4096 -g 1 -n 5
python -m mlx_lm benchmark --model ~/assets/models/qwen3.6-35b-a3b-mlx -p 4 -g 512 -n 5
```

The FreeToken prefill rows size the KV pool to the workload (`--num-tokens 32768`), as
`llama-bench` and `mlx_lm` do. On this tree the default plan takes 183,614 KV tokens and leaves
2.8 GiB for the forward; a 4,096-token prefill then reads 195 tok/s (137-256) and an 8,192-token
one 340, because the machine swaps instead of failing an allocation.

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
