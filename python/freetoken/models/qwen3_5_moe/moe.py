from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    sigmoid_gate_mul,
    silu_and_mul,
)
from freetoken.layers.quantization.linear.nvfp4 import MetalNvfp4LinearKernel
from freetoken.layers.quantization.linear.unquantized import MetalLinearKernel

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def _metal_kernel(layer, cls):
    """The layer's selected kernel when it is ``cls``, else None -- the seam the fused Metal
    forms hang off. Off Metal no table ever selects one, so every check below is False."""
    kernel = getattr(layer.quant_method, "kernel", None)
    return kernel if isinstance(kernel, cls) else None


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(
        self, x: torch.Tensor, gate: torch.Tensor | None = None,
        gate_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``gate`` is one pre-sigmoid scalar per token; the result is scaled by
        ``sigmoid(gate)``."""
        # Metal, NVFP4, small M: gate|up fuses SiLU-mul (and the scalar gate) and down fuses the
        # sigmoid gate, so the sub-block is two launches and writes no [M, 2I] product.
        gu_nvfp4 = _metal_kernel(self.gate_up_proj, MetalNvfp4LinearKernel)
        dn_nvfp4 = _metal_kernel(self.down_proj, MetalNvfp4LinearKernel)
        if gu_nvfp4 is not None and dn_nvfp4 is not None:
            h = gu_nvfp4.apply_fused(
                self.gate_up_proj, x, swiglu=True, side_gate_weight=gate_weight
            )
            if gate_weight is not None:
                h, gate = h
            return dn_nvfp4.apply_fused(self.down_proj, h, gate=gate)
        if gate_weight is not None:
            gate = torch.nn.functional.linear(x, gate_weight)
        # Metal, bf16, small M: one launch for gate|up + SiLU-mul and one for down, so no
        # [M, 2I] intermediate is written to device memory.
        h = out = None
        gu_bf16 = _metal_kernel(self.gate_up_proj, MetalLinearKernel)
        dn_bf16 = _metal_kernel(self.down_proj, MetalLinearKernel)
        # TP>1 all-reduces the down output, which only LinearRowParallel.forward does
        if gu_bf16 is not None and dn_bf16 is not None and self.down_proj._tp_size == 1:
            h = gu_bf16.apply_fused(self.gate_up_proj, x, swiglu=True)
            if h is not None:
                out = dn_bf16.apply_fused(self.down_proj, h)
        if out is None:
            out = self.down_proj.forward(
                silu_and_mul(self.gate_up_proj.forward(x)) if h is None else h
            )
        return out if gate is None else sigmoid_gate_mul(out, gate)


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def _router_kernel(self):
        """The router GEMV, when the bf16 table selected the Metal one (Metal, no bias)."""
        return _metal_kernel(self.gate, MetalLinearKernel)

    def fuses_input_norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """True when ``forward(..., norm=)`` runs the residual-add RMSNorm ahead of the block
        inside the router GEMV's launch (Metal, bf16 router, decode M)."""
        kernel = self._router_kernel()
        return kernel is not None and kernel.can_fuse_norm(self.gate, x, residual, x, weight)

    def forward(self, hidden_states: torch.Tensor, *, norm=None) -> torch.Tensor:
        """``norm = (residual, residual_out, weight, eps)``: ``hidden_states`` is then the raw
        sublayer input; the router GEMV's prologue adds the residual, norms, writes the new
        residual to ``residual_out`` and the normalized row for the expert launches. Only
        valid when ``fuses_input_norm`` said so."""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        if norm is not None:
            # Metal, bf16 router, decode M: one custom launch instead of an MPSGraph F.linear,
            # with the residual-add RMSNorm ahead of the block in its prologue.
            router_logits, hidden_states = self._router_kernel().apply_fused(
                self.gate, hidden_states, norm=norm
            )
        else:
            router_logits = self.gate.forward(hidden_states)
        shared_banks = self._shared_as_route()
        if shared_banks is not None:
            # Metal, NVFP4, decode: the shared expert rides as route top_k of the routed kernels,
            # so the block is the router plus the routed pair, four launches in all.
            return self.experts.forward(
                hidden_states=hidden_states, router_logits=router_logits,
                shared_nvfp4=shared_banks,
            ).view(num_tokens, hidden_dim)
        # The shared expert projects the scalar gate (in its gate|up kernel where it can) and
        # applies it in its down epilogue, so neither is a launch of its own.
        shared = self.shared_expert.forward(
            hidden_states, gate_weight=self.shared_expert_gate.weight
        )
        # ...and the sum lands in the routed down kernel's epilogue (``add_to``), so the block
        # is the router, the shared pair and the routed pair, nothing else.
        return self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits, add_to=shared
        ).view(num_tokens, hidden_dim)

    def _shared_as_route(self) -> tuple[torch.Tensor, ...] | None:
        """The shared expert's six NVFP4 banks and its gate row, when the routed layer
        can run it as an extra route: Metal NVFP4 shared expert, resident NVFP4 routed
        experts at decode, the same intermediate size, no biases."""
        se = self.shared_expert
        if not (
            _metal_kernel(se.gate_up_proj, MetalNvfp4LinearKernel) is not None
            and _metal_kernel(se.down_proj, MetalNvfp4LinearKernel) is not None
            and se.gate_up_proj.bias is None
            and se.down_proj.bias is None
            and self.shared_expert_gate.bias is None
            and getattr(self.experts, "accepts_shared_nvfp4", None) is not None
            and self.experts.accepts_shared_nvfp4()
        ):
            return None
        gu, dn = se.gate_up_proj, se.down_proj
        inter, hidden = self.experts.intermediate_size, self.experts.hidden_size
        if (tuple(gu.weight.shape) != (2 * inter, hidden // 2)
                or tuple(dn.weight.shape) != (hidden, inter // 2)):
            return None  # a shared expert of another width runs as its own two launches
        return (
            gu.weight, gu.weight_scale, gu.weight_global, dn.weight, dn.weight_scale,
            dn.weight_global, self.shared_expert_gate.weight.reshape(-1),
        )


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
