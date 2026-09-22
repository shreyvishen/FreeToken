"""Prefill and decode tok/s of one served config, COLD and HOT, the way docs/benchmarks-metal-m4max.md reports them.

pp4096: a prompt the server counts as about 4,096 tokens (2,922 words), prompt processing alone:
``(prompt_tokens - base_tokens) / (t - base_t)``, ``max_tokens=1``, the base a one-word prompt.
tg512: 512 generated tokens with ``ignore_eos``, tok/s from the first to the last token on the SSE stream.

COLD is the first measured request of a fresh server (one 1-token request first, so the kernels are
compiled): an SSD tier's expert slots are nearly empty. The OS page cache is not purged, so this is
cold slots, not a cold drive. HOT is the same request repeated three times in that server, each
prefill with a fresh salt so no prefix cache helps. Prefill and decode each get their own fresh
server, so both colds are first requests. ``--reps`` servers per phase; the JSON keeps every run.

    PYTHONPATH=python python benchmarks/bench_hotcold.py --model <dir> --reps 3 -- --moe-backend fused
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
ESSAY = "Write a detailed essay about the history and engineering of Roman aqueducts."


def post(port: int, path: str, body: dict):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def prefill(port: int, words: int, salt: str) -> tuple[float, int]:
    prompt = f"{salt} " + " ".join(WORDS[i % 10] for i in range(words))
    t0 = time.perf_counter()
    out = json.load(post(port, "/v1/completions",
                         {"model": "x", "prompt": prompt, "max_tokens": 1, "temperature": 0}))
    return time.perf_counter() - t0, out["usage"]["prompt_tokens"]


def decode(port: int, n: int) -> float:
    stamps = []
    body = {"model": "x", "stream": True, "temperature": 0, "max_tokens": n, "ignore_eos": True,
            "messages": [{"role": "user", "content": ESSAY}]}
    with post(port, "/v1/chat/completions", body) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                d = json.loads(line[6:])
                if d.get("choices") and (d["choices"][0].get("delta") or {}):
                    stamps.append(time.perf_counter())
    return (len(stamps) - 1) / (stamps[-1] - stamps[0])


def serve(args, log) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "freetoken.cli", "serve", "--model", args.model, "--host", "127.0.0.1",
         "--port", str(args.port), *args.serve_args],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"server exited {proc.returncode}; see {log.name}")
        try:
            json.load(post(args.port, "/v1/completions",
                           {"model": "x", "prompt": "hi", "max_tokens": 1, "temperature": 0}))
            return proc
        except OSError:
            time.sleep(2)
    stop(proc)
    sys.exit(f"server not ready in 900 s; see {log.name}")


def stop(proc: subprocess.Popen) -> None:
    # its own session: the frontend and the scheduler process, nothing else
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(6)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--reps", type=int, default=1, help="fresh servers per phase")
    p.add_argument("--port", type=int, default=8951)
    p.add_argument("--out", default="bench_hotcold.json")
    p.add_argument("serve_args", nargs=argparse.REMAINDER, help="after --: ft serve flags")
    args = p.parse_args()
    args.serve_args = [a for a in args.serve_args if a != "--"]
    res = {"model": args.model, "serve_args": args.serve_args, "pp4096": [], "tg512": []}
    with open(args.out + ".serve.log", "ab") as log:
        for rep in range(args.reps):
            proc = serve(args, log)
            cold = prefill(args.port, 2922, f"cold{rep}")
            hot = [prefill(args.port, 2922, f"hot{rep}-{i}") for i in range(3)]
            base = sorted(prefill(args.port, 1, f"b{rep}-{i}") for i in range(3))[1]
            tps = lambda t, n: (n - base[1]) / (t - base[0])  # noqa: E731
            res["pp4096"].append({"prompt_tokens": cold[1], "cold": tps(*cold),
                                  "hot": [tps(*h) for h in hot]})
            stop(proc)
            proc = serve(args, log)
            res["tg512"].append({"cold": decode(args.port, 512),
                                 "hot": [decode(args.port, 512) for _ in range(3)]})
            stop(proc)
            print(f"rep {rep}: pp4096 cold {res['pp4096'][-1]['cold']:.1f}, tg512 cold "
                  f"{res['tg512'][-1]['cold']:.2f}", flush=True)
    for phase in ("pp4096", "tg512"):
        cold = [r["cold"] for r in res[phase]]
        hot = [h for r in res[phase] for h in r["hot"]]
        print(f"{phase}: cold {statistics.median(cold):.2f} ({min(cold):.2f}-{max(cold):.2f}), "
              f"hot {statistics.median(hot):.2f} ({min(hot):.2f}-{max(hot):.2f})")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
