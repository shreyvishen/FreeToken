# Metal campaign -- M4 Max, 35B/122B MoE decode

Median of 3 runs, AC power, idle machine. `bench_decode_moe.py` only; run from the repo
root with `PYTHONPATH=python`.

| Engine | Model + quantization | Backend / cache | Decode tok/s (median of 3) | TTFT | Command |
|---|---|---|---|---|---|
| FreeToken | Qwen3.6-35B-A3B, NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4) | metal (fused, resident) | 110.13 (runs: 110.27, 103.89, 110.13) | 326.9 (runs: 244.8, 326.9, 345.1) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --backend metal` |
| FreeToken | Qwen3.6-35B-A3B, NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4) | offload, 25% cache from SSD | 12.97 (runs: 12.97, 12.55, 13.24) | 2737.0 (runs: 2737.0, 2815.6, 2705.3) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --backend offload --cache 2560` |
| FreeToken | Qwen3.5-122B-A10B, NVFP4 (nvidia/Qwen3.5-122B-A10B-NVFP4) | offload, from SSD | 4.12 (runs: 4.15, 4.12, 3.94) | 7763.8 (runs: 7763.8, 7766.6, 7075.1) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.5-122b-a10b-nvfp4 --backend offload` |
| FreeToken | Qwen3.5-122B-A10B, NVFP4 (nvidia/Qwen3.5-122B-A10B-NVFP4) | offload, from SSD, `--dense-quant-override fp8` | 4.96 (runs: 4.79, 5.09, 4.96) | 6607.6 (runs: 6607.6, 6489.5, 6653.6) | `PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.5-122b-a10b-nvfp4 --backend offload --dense-quant-override fp8` |
