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

Prefill tok/s is `(prompt_tokens(N) - prompt_tokens(base)) / (median t(N) - median t(base))` with
every request `max_tokens=1` and `temperature=0`, three repetitions, the median taken, and a fresh
salt per repetition so nothing is served from a cache. Subtracting a 14-token baseline removes the
HTTP hop, the template and the one decode step. This is the analogue of `llama-bench`'s `pN` row,
which also times prompt processing alone. Median of 3, AC power, idle machine.

| Prompt tokens | SDPA per request (before) | Tiled kernel | Tiled kernel + prefix cache |
|---|---|---|---|
| 722 | 627.7 | **691.2** | 499.2 |
| 2,873 | 353.9 | **546.0** | 491.2 |
| 11,474 | 96.2 | **244.7** | 225.0 |

```
LENGTHS="512 2048 8192" bash .notes/briefs/screen_prefill.sh . 30423 prefill --cache-type naive
```

The middle column is `--cache-type naive`; the right-hand column is the default, which also
snapshots GDN state for cross-request reuse (see the prefix-cache section). Greedy ids are
identical to the SDPA path on every row, at a 6-token prompt and at a 5,613-token one.

The kernel wins at every length and the gain grows with the prompt, because attention is under 1 %
of a 722-token prefill on this model (10 full-attention layers of 16 query heads at head_dim 256,
against ~3B active MoE parameters) and about a third of a 32k one. Short-prompt prefill is bound by
the MoE and dense GEMMs, not by attention.

Memory is the larger result. Against the SDPA path at the same shapes
(`benchmarks/bench_prefill_attention.py`, total MPS driver bytes):

| q_len x kv_len | Kernel | SDPA |
|---|---|---|
| 2048 x 2048 | 8.20 ms, 1,051 MiB | 14.20 ms, 1,567 MiB |
| 8192 x 8192 | 122.22 ms, 1,051 MiB | 226.30 ms, 11,419 MiB |
| 8192 x 16384 | 371.85 ms, 1,051 MiB | 467.74 ms, 21,539 MiB |
| 8192 x 32768 | 918.91 ms, 1,049 MiB | did not fit (~38 GiB) |

The kernel's footprint is flat in kv_len and is almost entirely its inputs; the score matrix never
exists. On the 11,474-token end-to-end row, swap peaks at 6.9 GiB against 15.3 GiB before.

### Against MLX and llama.cpp on the same machine

The kernel against MLX's own fused attention at identical shapes (HQ=16, HKV=2, head_dim 256,
bf16, one request), each row in its own process:

```
PYTHONPATH=python:. python benchmarks/bench_prefill_attention.py --mlx
```

| q_len x kv_len | FreeToken kernel | MLX `fast.scaled_dot_product_attention` |
|---|---|---|
| 512 x 512 | **0.86 ms**, 59 MiB | 0.92 ms, 21 MiB |
| 2048 x 2048 | 8.12 ms, 1,051 MiB | **7.62 ms**, 184 MiB |
| 8192 x 8192 | **121.74 ms**, 1,051 MiB | 124.65 ms, 2,320 MiB |
| 8192 x 16384 | 378.51 ms, **1,051 MiB** | **248.29 ms**, 4,448 MiB |
| 8192 x 32768 | 926.06 ms, **1,049 MiB** | **503.31 ms**, 8,704 MiB |

Within 2 % of MLX through 8192 x 8192; 1.52x and 1.84x behind at 16k and 32k. Memory is the other
way round and it widens with context: flat at ~1,050 MiB against MLX's 8,704 MiB at 32k, 8.3x less.
MLX's peak tracks `heads x q_len x kv_len x 2 B` (8,448 MiB measured against 8,192 MiB predicted,
and `mask=None` gives the same), so at these shapes MLX is materialising the bf16 score matrix
rather than running a memory-flat path.

Two things make the time column favour MLX: its K and V are contiguous while ours are gathered
through a page table, which is about 37 % of our kernel's time by ablation, and the two memory
columns are different counters (torch driver bytes against `mx.get_peak_memory`), so compare how
they scale rather than their floors.

llama.cpp on the same box, `llama-bench` build 89fe242, the only Qwen GGUF still on disk:

| test | llama.cpp, Qwen3.5-9B Q4_K_M (5.28 GiB, 8.95 B) |
|---|---|
| pp512 | 663.31 +- 0.79 |
| pp2048 | 658.10 +- 3.24 |
| pp8192 | 612.01 +- 5.47 |

Peak RSS 5.72 GiB, so about 0.44 GiB above the weights. **This is a different model and is not a
head-to-head with the rows above.** What it does show is the shape of the curve: llama.cpp loses
8 % from pp512 to pp8192 where FreeToken loses 65 %. Since the attention kernel is within 2 % of
MLX to 8192 x 8192, that remaining slope is the MoE and dense GEMM path at long extend lengths, not
attention. A same-model comparison needs the 35B GGUF and MLX checkpoints, which are not on this
disk.
