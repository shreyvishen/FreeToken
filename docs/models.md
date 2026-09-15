# Supported models

FreeToken loads HF safetensors checkpoints directly. The checkpoints below are known-good — the prebuilt kernels are tuned
for them; other checkpoints of the same architectures work too.

| Model | HF checkpoints |
|---|---|
| DeepSeek-V4 | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| GLM-5.3-Flash | [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) |
| GLM-5.2 | [nvidia/GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8), [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4), [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| Qwen3.6 / Qwen3.5 MoE | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)), [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)) |
| Qwen3.8 / Qwen3.6 dense | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)), [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) ([-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)), [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Qwen3-MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Qwen3-VL | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) |
| gpt-oss | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Gemma-4 | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4), [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) .. |
| MiniMax-M2.5 | [nvidia/MiniMax-M2.5-NVFP4](https://huggingface.co/nvidia/MiniMax-M2.5-NVFP4) |
| MiniMax-M3 | [nvidia/MiniMax-M3-NVFP4](https://huggingface.co/nvidia/MiniMax-M3-NVFP4) |
| Muse-Glimmer | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) |

### Image input

These families accept image input by default; pass `--text-model-only` to skip the vision encoder. The flags are described in the
[CLI reference](cli.md#image-input); each family reads them in its own units.

| Family | Image tokens | `--image-min-tokens` / `--image-max-tokens` | `--mm-processor-kwargs` example |
| --- | --- | --- | --- |
| Qwen3.6 (both variants, every listed weight format), Qwen3.8-Flash-Next, Qwen3-VL | one token per 32x32 pixels of the resized image, dynamic resolution | pixel areas in `size.shortest_edge` / `longest_edge`; checkpoint defaults 64 to 16384 tokens | `{"size": {"longest_edge": 1048576}}` |
| Gemma-4 26B-A4B, 31B (`gemma4`: ViT tower, streamed under `--mm-encoder-weights host`) | one of the soft-token budgets 70 / 140 / 280 / 560 / 1120, every image scaled to its budget as far as the aspect ratio allows | the maximum picks the largest budget within it, below 70 is refused at start-up; the minimum has no effect | `{"max_soft_tokens": 1120}` |
| Gemma-4 12B (`gemma4_unified`: linear patch embedder, resident under either placement) | same budgets, one 48x48 super-patch per soft token | same as the tower releases | same |
| GLM-5.3-Flash (`glm5_next`: ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, dynamic resolution on a canvas zero-padded to a 28-multiple | token counts, passed through as the processor's `min_image_tokens` / `max_image_tokens`; checkpoint defaults 16 to 8000 tokens | `{"max_image_tokens": 2048}` |
| Muse-Glimmer-30B (windowed ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, aspect ratio kept under a token cap; checkpoint default 4096 tokens | the maximum is the cap (`max_image_tokens`); the minimum has no effect | `{"max_image_tokens": 1024}` |
| MiniMax-M3 (`minimax_m3`: CLIP-style ViT tower, streamed under `--mm-encoder-weights host`) | one token per 28x28 pixels of the resized image, dynamic resolution | pixel areas in `size.shortest_edge` / `longest_edge`; checkpoint defaults 4 to 576 tokens | `{"size": {"longest_edge": 1048576}}` |

## MoE strategies

`ft serve --moe-strategy {auto,fused,offload,cpu,hybrid}` (`--moe-backend` is the deprecated old spelling):

- **fused** — experts resident on GPU (needs the VRAM); never auto-selected.
- **offload** — experts live in host RAM, an LRU cache of expert slots on GPU;
  misses stream over PCIe.
- **cpu** — misses are computed on the CPU instead of fetched.
- **hybrid** — per step, fetches some misses over PCIe and computes the rest on
  CPU, overlapped. Run `ft bench bw` once per machine to calibrate the split.
- **auto** — dense models always resolve to `fused`; MoE models resolve to
  `offload`, upgraded to `hybrid` when a cached `ft bench bw` profile
  recommends it.

## Notes

- Qwen3.6-35B-A3B-NVFP4 is tested on macOS (Apple Silicon, MPS) as well as CUDA. First-token
  and short-decode verified on an M4 Max, torch 2.10.0:
  `ft serve --model nvidia/Qwen3.6-35B-A3B-NVFP4 --moe-backend fused`
  (or omit `--moe-backend`; `auto` resolves to `fused` on Metal when the resident expert set
  fits). No chunked GDN prefill kernel yet, so `hybrid_radix` is unavailable on mps (falls back
  to `naive`); prefill attention is torch SDPA per request, decode attention is a paged Metal
  kernel, and both are bounded by `max_running_req`.
- `ft checkpoint` conversion is optional — it pre-converts a checkpoint into
  FreeToken's fast-load format, and `ft serve --model` auto-detects the result.
- FTW files converted by builds before the quantization refactor may fail to load;
  see [ftw-hotfix.md](ftw-hotfix.md) for the affected checkpoints and the repair tool.
- An FTW converted before its family served images holds no vision encoder: `ft serve`
  refuses it unless started with `--text-model-only` (or `--mm-disable vision`); reconvert it
  with `ft checkpoint`, or add the encoder in place with [scripts/ftw_hotfix.py](ftw-hotfix.md).
- DeepSeek-V4 checkpoints must keep the `inference/config.json` subdir — the
  authoritative model args are read from there.
- Qwen3.8-Flash-Next keeps a 47.7 GiB PLE n-gram table pinned in host RAM.

## Runtime flags on Metal (macOS)

Image input is not served on Metal: the encoder towers stream their weights over CUDA
streams, so a Metal process builds none and accepts text only.

These environment variables tune the Metal (MPS) serving path. All are optional;
each one has a default that needs no attention.

| Flag | Values | Default | What it does |
|---|---|---|---|
| `FREETOKEN_LOG_TOKEN_IDS` | `0` or `1` | `0` (off) | Prints every sampled token id as the scheduler appends it. |
| `FREETOKEN_LOG_MPS_MEM` | integer N | `0` (off) | Logs MPS driver pool, torch live tensors, process RSS, and swap every N decode steps. |
| `FREETOKEN_MPS_OS_RESERVE_GB` | integer GiB | `8` | Host RAM left to macOS and every other process. Depends on the machine's RAM. |
| `FREETOKEN_DENSE_QUANT_OVERRIDE` | `none`, `fp8` | `none` | Carries the `--dense-quant-override` choice (below) to child processes; set directly to skip the flag. |

The disk tier prefetches experts from a router run one layer ahead, decided per server
from its own miss counters (`prefetch_pays`): on when the misses per layer cost more than
4 MiB per decode token, off otherwise.

The disk tier (Metal SSD tier) takes these `ft serve` flags:

| Flag | Values | Default | What it does |
|---|---|---|---|
| `--moe-backend` | `auto`, `fused`, `offload`, `cpu`, `hybrid` | `auto` | The MoE backend. `auto` resolves a MoE model to `offload` (or `hybrid` when a `ft bench bw` profile recommends it); `fused` must be requested explicitly. |
| `--moe-cache-size` | integer | `0` | Number of unified MoE expert slots on GPU. |
| `--dense-quant-override` | `none`, `fp8` | `none` | Metal only: quantizes the checkpoint's unquantized dense weights (attention q/k/v/o, GatedDeltaNet projections, dense MLP) to fp8-e4m3 at load, W8A16 at serve. Halves their bytes, freeing memory for MoE cache slots. Lossy; opt-in. |

On unified memory the KV pool is capped at the smaller of 65,536 tokens and
`max_running_requests x max_seq_len`, so the rest of the budget stays free host memory;
`--num-pages` still wins. The disk tier pins the top tenth of each layer's experts by
routing count against eviction: routing is skewed and the skew belongs to the weights, so
pinning the head experts beats letting an LRU re-decide them every step.
`--dense-quant-override fp8` is lossy and off by default; use it only when you need the
freed memory for more expert slots and can accept the weight error.
