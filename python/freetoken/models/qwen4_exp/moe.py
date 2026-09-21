from __future__ import annotations

import torch
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def forward(self, hidden_states: torch.Tensor, *, norm=None) -> torch.Tensor:
        if not hidden_states.is_cuda:
            return super().forward(hidden_states, norm=norm)
        assert norm is None, "the triton shared-gate path cannot fuse the input norm"
        from freetoken.kernel.triton.moe_shared_gate import (
            shared_gate_mul_add,
            shared_gate_sigmoid,
        )

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
