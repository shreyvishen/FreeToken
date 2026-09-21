from __future__ import annotations

from freetoken.layers import GatedRMSNorm
from freetoken.layers.quantization import QuantConfig
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet


class Qwen4ExpGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Qwen3.5's GatedDeltaNet with the output gate taken from the config.

    Same recurrence, state pool, conv and kernels on both backends; Qwen3.8-Flash-Next only
    differs in the gate activation ("sigmoid" rather than silu) and in running 48 value heads
    over 16 key heads, which the GQA index arithmetic already handles for any whole ratio.
    ``output_gate`` is the activation name from ``LinearGatedDeltaGroupConfig``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, output_gate: str = "sigmoid",
        *, quant_config: QuantConfig | None = None, prefix: str = "",
    ):
        super().__init__(
            hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size,
            rms_norm_eps, layer_id, quant_config=quant_config, prefix=prefix,
        )
        # rebound, not a constructor argument: ``norm`` keeps its position in __dict__, so the
        # state-dict key order the weight loader walks is the parent's.
        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps, activation=output_gate)


__all__ = ["Qwen4ExpGatedDeltaNet"]
