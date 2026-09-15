# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

`--backend metal` is Darwin-only: resident experts via `--moe-backend fused`. The three
Metal rows in `docs/benchmarks-metal-m4max.md`:

```bash
PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --backend metal
PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.6-35b-a3b-nvfp4 --backend offload --cache-rate 0.25
PYTHONPATH=python python benchmarks/bench_decode_moe.py --model ~/assets/models/qwen3.5-122b-a10b-nvfp4 --backend offload
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
