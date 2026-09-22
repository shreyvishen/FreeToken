"""The elementwise epilogues that sit between the GEMMs, one launch each: ``silu_and_mul``,
SwiGLU over uninterleaved halves, and ``sigmoid_gate_mul``, ``x * sigmoid(gate) (+ add)``
with ``gate`` either as wide as ``x`` (the attention output gate) or one column that
broadcasts (the MoE epilogue). Plus ``scatter_rows``, the SSD tier's staging-to-slot copy."""

from __future__ import annotations

import functools

import torch

from .shaders import compile, is_available, msl_type

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void silu_and_mul(
    device T* out         [[buffer(0)]],   // [M, D]
    device const T* x     [[buffer(1)]],   // [M, 2 * D], gate | up
    constant uint& D      [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint d = gid.x;
  if (d >= D) { return; }
  uint m = gid.y;
  float g = float(x[m * 2 * D + d]);
  out[m * D + d] = T((g / (1.0f + exp(-g))) * float(x[m * 2 * D + D + d]));
}

kernel void sigmoid_gate_mul(
    device T* out         [[buffer(0)]],   // [M, N]
    device const T* x     [[buffer(1)]],   // [M, N]
    device const T* gate  [[buffer(2)]],   // [M, N] or [M, 1]
    device const T* add   [[buffer(3)]],   // [M, N]; ignored when HAS_ADD is 0
    constant uint& N      [[buffer(4)]],
    constant uint& GN     [[buffer(5)]],   // gate width: N, or 1 to broadcast
    constant uint& HAS_ADD [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint n = gid.x;
  if (n >= N) { return; }
  uint m = gid.y;
  uint i = m * N + n;
  float g = float(gate[m * GN + (GN == 1u ? 0u : n)]);
  float y = float(x[i]) / (1.0f + exp(-g));
  if (HAS_ADD != 0u) { y += float(add[i]); }
  out[i] = T(y);
}
"""

_SCATTER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void scatter_rows(
    device uint4* dst             [[buffer(0)]],   // [S, row_words] bank
    device const uint4* src       [[buffer(1)]],   // staged records
    device const int* idx         [[buffer(2)]],   // [2, n]: destination slot, staged row
    constant uint& n              [[buffer(3)]],
    constant uint& row_words      [[buffer(4)]],
    constant uint& record_words   [[buffer(5)]],
    constant uint& offset_words   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= row_words) { return; }
  dst[uint(idx[gid.y]) * row_words + gid.x] =
      src[uint(idx[n + gid.y]) * record_words + offset_words + gid.x];
}
"""


def scatter_rows(bank: torch.Tensor, records: torch.Tensor, idx: torch.Tensor, n: int,
                 record_bytes: int, offset: int) -> None:
    """``bank[idx[0, i]] = records[idx[1, i], offset : offset + row bytes]`` for ``i < n``, one
    launch; every size and offset a multiple of 16 bytes."""
    words = bank[0].numel() * bank.element_size() // 16
    compile(_SCATTER_SOURCE).scatter_rows(
        bank, records, idx, n, words, record_bytes // 16, offset // 16,
        threads=(words, n, 1), group_size=(min(words, 256), 1, 1),
    )


@functools.lru_cache(maxsize=None)
def _library(dtype: torch.dtype):
    return compile(f"#define T {msl_type(dtype)}\n" + _SOURCE)


def _ok(*tensors: torch.Tensor) -> bool:
    first = tensors[0]
    return (
        is_available()
        and first.device.type == "mps"
        and first.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and all(t.dim() == 2 and t.is_contiguous() and t.dtype == first.dtype for t in tensors)
    )


def supports_silu_and_mul(x: torch.Tensor) -> bool:
    return _ok(x) and x.shape[1] % 2 == 0


def silu_and_mul_metal(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``silu(x[..., :D]) * x[..., D:]`` in one launch."""
    m, two_d = x.shape
    d = two_d // 2
    y = torch.empty((m, d), dtype=x.dtype, device=x.device) if out is None else out
    if m:
        _library(x.dtype).silu_and_mul(y, x, d, threads=(d, m, 1), group_size=(min(d, 256), 1, 1))
    return y


def supports_sigmoid_gate_mul(x, gate, add=None) -> bool:
    tensors = (x, gate) if add is None else (x, gate, add)
    return (
        _ok(*tensors)
        and gate.shape[0] == x.shape[0]
        and gate.shape[1] in (1, x.shape[1])
        and (add is None or add.shape == x.shape)
    )


def sigmoid_gate_mul_metal(
    x: torch.Tensor, gate: torch.Tensor, add: torch.Tensor | None = None
) -> torch.Tensor:
    """``x * sigmoid(gate) + add`` in one launch."""
    m, n = x.shape
    y = torch.empty_like(x)
    if m:
        _library(x.dtype).sigmoid_gate_mul(
            y, x, gate, add if add is not None else x, n, gate.shape[1], int(add is not None),
            threads=(n, m, 1), group_size=(min(n, 256), 1, 1),
        )
    return y


__all__ = [
    "scatter_rows", "sigmoid_gate_mul_metal", "silu_and_mul_metal", "supports_sigmoid_gate_mul",
    "supports_silu_and_mul",
]
