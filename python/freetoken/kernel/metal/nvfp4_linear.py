"""W4A16 NVFP4 dense projections for Metal."""

from __future__ import annotations

import torch

from freetoken.kernel.metal.fp8 import MAX_GEMV_ROWS
from freetoken.kernel.metal.nvfp4 import dequant_nvfp4_dense, nvfp4_gemv


def nvfp4_dense_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, weight_global: torch.Tensor,
    bias: torch.Tensor | None = None, *, gate: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``F.linear`` over an NVFP4 weight: the GEMV at decode M, dequantize-once +
    ``F.linear`` at prefill M."""
    *lead, k = x.shape
    x2 = x.reshape(-1, k)
    out_features = weight.shape[0]
    if x2.shape[0] <= MAX_GEMV_ROWS:
        out = nvfp4_gemv(x2, weight, weight_scale, weight_global,
                         gate=None if gate is None else gate.reshape(-1),
                         out_dtype=out_dtype)
    else:
        w = dequant_nvfp4_dense(weight, weight_scale, weight_global, dtype=x.dtype)
        out = torch.nn.functional.linear(x2, w)
        if gate is not None:
            out = out * torch.sigmoid(gate.reshape(-1, 1).to(out.dtype))
        if out_dtype is not None:
            out = out.to(out_dtype)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.reshape(*lead, out_features)


def nvfp4_dense_swiglu(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, weight_global: torch.Tensor,
    *, side_gate_weight: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """``silu(x @ gate.T) * (x @ up.T)`` for a column-merged ``[gate | up]`` weight of equal
    parts, with the ``[M, 2I]`` product never written to device memory."""
    assert weight.shape[0] % 2 == 0, weight.shape
    *lead, k = x.shape
    x2 = x.reshape(-1, k)
    inter = weight.shape[0] // 2
    side = None
    if x2.shape[0] <= MAX_GEMV_ROWS:
        out = nvfp4_gemv(x2, weight, weight_scale, weight_global,
                         swiglu=True, side_gate_weight=side_gate_weight)
        if side_gate_weight is not None:
            out, side = out
    else:
        from freetoken.layers import silu_and_mul

        w = dequant_nvfp4_dense(weight, weight_scale, weight_global, dtype=x.dtype)
        out = silu_and_mul(torch.nn.functional.linear(x2, w))
        if side_gate_weight is not None:
            side = torch.nn.functional.linear(x2, side_gate_weight.reshape(1, k)).reshape(-1)
    out = out.reshape(*lead, inter)
    return out if side_gate_weight is None else (out, side.reshape(*lead))


__all__ = ["nvfp4_dense_linear", "nvfp4_dense_swiglu"]
