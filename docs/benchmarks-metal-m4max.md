# FreeToken on Metal: M4 Max numbers

Apple M4 Max (14-core CPU, 32-core GPU), 36 GiB, macOS 26.6.2, torch 2.10.0, `nvidia/*-NVFP4`
checkpoints. AC power, one server at a time, nothing else on the GPU. Commands run from the repo
root with `PYTHONPATH=python`; tok/s unless a table says otherwise.

## Four serving configs, cold and hot (22 Sep 2026)

`pp4096` is prompt processing alone for a prompt the server counts as 4,092 to 4,094 tokens; `tg512` is 512
generated tokens with `ignore_eos`, timed from the first to the last token on the stream.
**Cold** is the first request a fresh server measures (after one 1-token request that compiles the
kernels), so an SSD tier starts with nearly empty expert slots; the OS page cache is not purged, so
it is cold slots, not a cold drive. **Hot** repeats the request three times in that server, each
prefill with a fresh prefix so no cache helps. Every cell is three fresh servers: cold is the median
of their first requests, hot the median of all nine repeats, min-max in brackets.

| Config | pp4096 hot | tg512 hot |
|---|---|---|
| 35B resident | 1206.3 (1186.0-1220.1) | 109.43 (109.29-109.57) |

| Config | pp4096 cold | pp4096 hot | tg512 cold | tg512 hot |
|---|---|---|---|---|
| 35B, 25 % of experts in memory | 851.6 (805.5-855.1) | 882.7 (783.9-990.3) | 35.98 (35.57-36.43) | 37.78 (37.03-41.83) |
| 122B, experts on the SSD tier | 258.6 (258.2-263.2) | 224.7 (195.2-248.8) | 4.84 (4.81-4.87) | 5.00 (4.89-5.11) |
| Qwen3.8-Flash-Next, experts and PLE table on the SSD | 276.1 (274.0-278.6) | 271.3 (246.1-285.2) | 9.16 (9.03-9.19) | 10.05 (9.93-10.19) |

```
python benchmarks/bench_hotcold.py --reps 3 --model /path/to/Qwen3.6-35B-A3B-NVFP4 -- --moe-backend fused
python benchmarks/bench_hotcold.py --reps 3 --model /path/to/Qwen3.6-35B-A3B-NVFP4 -- --moe-backend offload --moe-cache-size 2560 --max-running-requests 1 --max-seq-len-override 8704
python benchmarks/bench_hotcold.py --reps 3 --model /path/to/Qwen3.5-122B-A10B-NVFP4 -- --moe-backend offload --max-running-requests 1 --max-seq-len-override 8704
python benchmarks/bench_hotcold.py --reps 3 --model /path/to/Qwen3.8-Flash-Next-NVFP4 -- --moe-backend offload --max-running-requests 1 --max-seq-len-override 8704
```

The campaign ran each of these with `--reps 1`, three times, interleaved with the same run on
`5e72e7e` for the pairs below; `--reps 3` takes a cell's three servers in a row.

The SSD rows read experts from the expert-major repack beside each checkpoint
(`python -m freetoken.checkpoint.repack_nvfp4_experts`). 2,560 slots hold 25 % of the 35B's experts;
the 122B and Flash Next rows take the slot count the planner picks.

Against the port before this round's changes (`5e72e7e`), in the same campaign: for every config
and rep, one server on each tree back to back, the order flipped each rep:

| Config | Cell | Before | After | Change |
|---|---|---|---|---|
| 35B resident | pp4096 hot | 923.7 | 1206.3 | +31 % |
| 35B resident | tg512 hot | 108.97 | 109.43 | 0 % |
| 35B, 25 % of experts in memory | pp4096 cold | 730.5 | 851.6 | +17 % |
| 35B, 25 % of experts in memory | pp4096 hot | 726.5 | 882.7 | +22 % |
| 35B, 25 % of experts in memory | tg512 cold | 28.67 | 35.98 | +25 % |
| 35B, 25 % of experts in memory | tg512 hot | 30.06 | 37.78 | +26 % |
| 122B, experts on the SSD tier | pp4096 cold | 179.7 | 258.6 | +44 % |
| 122B, experts on the SSD tier | pp4096 hot | 159.0 | 224.7 | +41 % |
| 122B, experts on the SSD tier | tg512 cold | 4.53 | 4.84 | +7 % |
| 122B, experts on the SSD tier | tg512 hot | 4.95 | 5.00 | +1 % |
| Qwen3.8-Flash-Next, experts and PLE table on the SSD | pp4096 cold | 167.5 | 276.1 | +65 % |
| Qwen3.8-Flash-Next, experts and PLE table on the SSD | pp4096 hot | 164.5 | 271.3 | +65 % |
| Qwen3.8-Flash-Next, experts and PLE table on the SSD | tg512 cold | 8.75 | 9.16 | +5 % |
| Qwen3.8-Flash-Next, experts and PLE table on the SSD | tg512 hot | 9.47 | 10.05 | +6 % |

The 122B's tg512 hot pair is noise (-0.2, -1.4 and +2.7 % rep by rep); its cold pair gains 6 to 8 % on
every rep. The SSD rows' prefill leans on run order: the second server of a pair finds a warmer page cache, and
three reps put this branch second twice. Compared at the same position, the 35B at 25 % gains 10 to
14 % on pp4096 (cold and hot), and the 122B 27 to 46 % on pp4096 hot.

## Same model, three engines: Qwen3.6-35B-A3B resident

One base model in each engine's own 4-bit format, resident on the GPU. Five repetitions per row:
median, min-max in brackets. `pp` is prompt processing alone and `tg` generation, with the names and
lengths `llama-bench` uses. The llama.cpp and mlx_lm columns are from 21 Sep 2026 on this machine;
the FreeToken column is from 22 Sep 2026, on this branch.

| Test | FreeToken, NVFP4 | llama.cpp, UD-Q4_K_M | mlx_lm, 4-bit |
|---|---|---|---|
| pp512 | 1167.6 (1106.9-1177.0) | 1238.6 (1232.6-1250.0) | **1248.1** (1229.5-1260.9) |
| pp2048 | 1243.6 (1230.2-1250.3) | 1218.1 (1137.9-1222.0) | **1481.7** (1475.2-1487.6) |
| pp4096 | 1215.4 (1208.3-1223.7) | 1185.6 (1163.2-1187.6) | **1461.9** (1455.6-1469.9) |
| pp8192 | 1108.9 (1107.8-1119.7) | 1068.5 (1060.0-1076.4) | **1397.9** (1338.8-1402.5) |
| tg128 | 107.6 (107.5-107.9) | 67.5 (67.0-68.2) | **113.7** (113.1-113.9) |
| tg512 | 105.8 (105.6-106.0) | 66.6 (66.4-66.8) | **112.5** (112.2-112.7) |

Checkpoints: `nvidia/Qwen3.6-35B-A3B-NVFP4`; `unsloth/Qwen3.6-35B-A3B-GGUF` `UD-Q4_K_M` (an Unsloth
Dynamic quant, so precision varies by layer); `mlx-community/Qwen3.6-35B-A3B-4bit`. llama.cpp build
89fe242, Metal + BLAS, all layers on the GPU, flash attention `auto`, f16 KV. mlx_lm 0.31.1.

```
# FreeToken prefill: a server sized like llama-bench's; prompts of 512 / 2048 / 4096 / 8192 server-counted tokens,
# max_tokens=1, five fresh prefixes each: (prompt_tokens - base_tokens) / (median t - median t_base), base an 8-word prompt
python -m freetoken.cli serve --model /path/to/Qwen3.6-35B-A3B-NVFP4 --moe-backend fused --num-tokens 32768
# FreeToken decode
python benchmarks/bench_decode_moe.py --model /path/to/Qwen3.6-35B-A3B-NVFP4 --backend metal --decode 128 --decode-reps 5
python benchmarks/bench_decode_moe.py --model /path/to/Qwen3.6-35B-A3B-NVFP4 --backend metal --decode 512 --decode-reps 5
# llama.cpp
llama-bench -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p 512,2048,4096,8192 -n 128,512 -r 5
# mlx_lm, one process per row
python -m mlx_lm benchmark --model /path/to/Qwen3.6-35B-A3B-4bit -p 4096 -g 1 -n 5
python -m mlx_lm benchmark --model /path/to/Qwen3.6-35B-A3B-4bit -p 4 -g 512 -n 5
```

### Prefill attention against MLX's kernel

The tiled kernel against `mlx.core.fast.scaled_dot_product_attention` at the 35B's attention shape
(16 query heads, 2 K/V heads, head_dim 256, bf16, one request), each row in its own process
(16 Sep 2026):

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

Ahead of MLX at 512 and 8192 x 8192, 6.6 % behind at 2048, and 1.52x and 1.84x behind at 16k and
32k. Memory goes the other way: flat at about 1,050 MiB against MLX's 8,704 MiB at 32k, whose peak
tracks `heads x q_len x kv_len x 2 B`, the bf16 score matrix. Two caveats favour MLX in the time
column: its K and V are contiguous where ours are gathered through a page table (about 37 % of our
kernel's time by ablation), and the memory columns are different counters (torch driver bytes
against `mx.get_peak_memory`).

## Prefix cache: the second turn of a conversation (22 Sep 2026)

`hybrid_radix` is the default prefix cache for GDN models on Metal. Turn 2 of a conversation
restores the newest GDN state snapshot at or before the shared prefix and prefills only the tail.
Turn 1 is a 3,000-word prompt (4,522 tokens); turn 2 sends turn 1's prompt, its answer and a
one-line instruction (4,565 tokens). Each turn is one `/v1/completions` request that generates 32
tokens (`ignore_eos`), timed whole. Two repetitions, a fresh conversation each, 35B resident (`auto`
resolves to `fused`), with and without `--cache-type naive`. Turn-2 tokens prefilled is the server
log's `#new-token` for that turn.

| Cache type | Turn 1 (s) | Turn 2 (s) | Turn 2 tokens prefilled |
|---|---|---|---|
| `naive` | 4.176 | 4.202 | 4,565 of 4,565 |
| `hybrid_radix` | 4.238 | 0.469 | 12 of 4,565 |

Turn 1 under `hybrid_radix` takes 1.5 % longer than under `naive` here (4.238 s against
4.176). On `5e72e7e`, whose GDN prefill ran chunk by chunk to keep its snapshots, it took 21 %
longer (5.287 s against 4.369).

The two cache types give the same turn-2 text in the first repetition and part at its second token
in the second, on `5e72e7e` exactly as on this branch. The cached conversation keeps its answer's KV
and GDN state as decode computed them; `naive` recomputes them in prefill.

```
python -m freetoken.cli serve --model /path/to/Qwen3.6-35B-A3B-NVFP4
python -m freetoken.cli serve --model /path/to/Qwen3.6-35B-A3B-NVFP4 --cache-type naive
```

## Flash Next: the slot count trades prefill for decode (21 Sep 2026)

The auto plan sizes the expert slots for decode: a decode token misses about 5 MiB of experts per
layer, and more slots mean fewer misses. Prefill wants the opposite, since a chunk streams the
experts it routes to through the buffer cache, and every slot is memory that cache does not get.
On 21 Sep, before the changes in the matrix above, `--moe-cache-size 3000` against the auto plan's
4,739 slots took pp4096 from 187.0 to 254.3 tok/s and decode from 9.20 to 7.39 tok/s
(`bench_decode_moe.py --backend offload --cache 3000 --decode 256 --decode-reps 3 --prefill 362,1459,2922`).
