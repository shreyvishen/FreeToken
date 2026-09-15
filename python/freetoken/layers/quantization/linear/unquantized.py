"""bf16 Linear: one kernel (torch), no scheme."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from freetoken.kernel import backend

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod


class MetalLinearKernel(LinearKernel):
    """bf16 dense GEMV on Metal: one launch for the projection, with the residual-add RMSNorm
    or a SwiGLU merge folded in where asked."""

    name = "metal"
    mps_only = True

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        return None if backend.is_mps() else "the Metal dense GEMV needs torch mps"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.metal import mlp

        w = layer.weight
        # Above MAX_GEMV_ROWS a matmul reads the weight once vs the GEMV's per-row-block reread.
        if (layer.bias is None and x.dim() == 2 and x.shape[0] <= mlp.MAX_GEMV_ROWS
                and mlp.can_run(x, w)):
            return mlp.dense_gemv(x, w)
        return TorchLinearKernel.apply(self, layer, x)

    def can_fuse_norm(self, layer: Any, x, residual, residual_out, weight) -> bool:
        """True when apply_fused(..., norm=) would run, before the caller arranges buffers."""
        from freetoken.kernel.metal import mlp

        return (
            layer.bias is None and x.dim() == 2 and x.shape[0] <= mlp.MAX_GEMV_ROWS
            and mlp.can_run(x, layer.weight)
            and mlp.can_run_norm(x, residual, residual_out, weight)
        )

    def apply_fused(self, layer: Any, x: torch.Tensor, *, norm=None, swiglu: bool = False):
        """norm=(residual, residual_out, weight, eps) runs rmsnorm(x+residual)*weight in the
        projection's prologue."""
        from freetoken.kernel.metal import mlp

        w = layer.weight
        if layer.bias is not None or x.dim() != 2 or not mlp.can_run(x, w):
            return None
        if swiglu:
            return mlp.swiglu_gemv(x, w)
        if norm is not None and not mlp.can_run_norm(x, norm[0], norm[1], norm[2]):
            return None
        return mlp.dense_gemv(x, w, norm=norm)


class TorchLinearKernel(LinearKernel):
    name = "torch"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # an fp32 activation stream (DeepSeek-V4's compressors) upcasts the bf16 weight on the fly, as the reference does
        w, b = layer.weight, layer.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
            b = b.to(x.dtype) if b is not None else None
        return F.linear(x, w, b)


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (MetalLinearKernel, TorchLinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)
