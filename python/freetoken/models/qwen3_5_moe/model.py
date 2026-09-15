from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from freetoken.kernel import backend as device_backend
from freetoken.core import get_global_ctx
from freetoken.kernel.backend import is_mps
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


# Predict the next layer's routing one mixer early, only while the disk tier's counters say
# the step is stalled on demand reads (DiskMoeCache.prefetch_pays).
_MOE_PREFETCH = True


# The gate a tier that does not measure itself gets: never speculate.
_PF_NEVER = SimpleNamespace(prefetch_pays=False)


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Only the offload MoE layer can read ahead; False elsewhere keeps every other backend
        # byte-for-byte unchanged.
        self._moe_prefetch = _MOE_PREFETCH and hasattr(
            getattr(self.mlp, "experts", None), "predict_from_router"
        )
        # Resolved lazily: the disk tier attaches to the MoE layer after __init__ runs.
        self._pf_gate = None
        # Bound callables, not modules, so iter_offload_moe_layers doesn't yield them twice.
        self._next_router = None
        self._next_predict = None

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Fuse each residual-add into the next RMSNorm so add+norm is one kernel.
        mixer = self.linear_attn if self._is_linear else self.self_attn
        norm = None
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        elif mixer.fuses_input_norm(hidden, residual, self.input_layernorm.weight):
            # The residual add and the norm run in the mixer's input projection
            # (its GEMV's prologue); the new residual lands out of place.
            new_residual = torch.empty_like(residual)
            norm = (residual, new_residual, self.input_layernorm.weight, self.input_layernorm.eps)
            residual = new_residual
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        do_prefetch = self._moe_prefetch and hidden.shape[0] == 1
        if do_prefetch and self._pf_gate is None:
            gate = getattr(getattr(self.mlp, "experts", None), "offload_cache", None)
            self._pf_gate = gate if hasattr(gate, "prefetch_pays") else _PF_NEVER
        if do_prefetch and self._pf_gate.prefetch_pays:
            # On the fused-norm path `residual` is the mixer's out-of-place output and is
            # still empty here; the residual the router should see is norm[0] + hidden.
            res_pre = residual if norm is None else norm[0] + hidden
            # Reads the previous layer predicted (its async D2H already landed); then
            # predict for the next layer, again asynchronously. Decode only.
            self.mlp.experts.issue_from_prediction()
            if self._next_predict is not None:
                self._next_predict(self._next_router(res_pre))
        hidden = mixer.forward(hidden, norm=norm)
        # Submit the mixer's launches now: the GPU runs them while the host encodes the
        # norm and the router, and the disk tier's routing readback then waits on less.
        device_backend.flush()
        fuse2 = getattr(self.mlp, "fuses_input_norm", None)
        if fuse2 is not None and fuse2(hidden, residual, self.post_attention_layernorm.weight):
            # Same again for norm2: it runs in the router GEMV's prologue, which also
            # writes the normalized row the expert launches read.
            new_residual = torch.empty_like(residual)
            norm = (residual, new_residual, self.post_attention_layernorm.weight,
                    self.post_attention_layernorm.eps)
            residual = new_residual
        else:
            norm = None
            hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = (
            self.mlp.forward(hidden, norm=norm) if norm is not None
            else self.mlp.forward(hidden)
        )
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if _MOE_PREFETCH:
            # Layer L predicts layer L+1's routing, so it needs L+1's router and norm.
            layers = self.layers.op_list
            for layer, nxt in zip(layers, layers[1:]):
                experts = getattr(nxt.mlp, "experts", None)
                if layer._moe_prefetch and hasattr(experts, "predict_from_router"):
                    layer._next_predict = experts.predict_from_router
                    layer._next_router = (
                        lambda x, g=nxt.mlp.gate, n=nxt.post_attention_layernorm:
                        g.forward(n.forward(x))
                    )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
            # Submit this layer's launches now (no wait) so the GPU runs them while the
            # host encodes the next layer. No-op on CUDA; see backend.flush.
            device_backend.flush()
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def load_resident_experts(self, config) -> None:
        """Fill the resident NVFP4 expert banks in place, straight onto the device."""
        from freetoken.layers.quantization import QuantKind
        from freetoken.layers.quantization.moe.nvfp4 import BANK_ROLES
        from freetoken.moe.expert_banks import build_expert_banks
        from freetoken.moe.expert_pieces import iter_expert_pieces

        model_config = config.model_config
        layers = [getattr(layer.mlp, "experts", None) for layer in self.model.layers.op_list]
        methods = [getattr(e, "quant_method", None) for e in layers]
        if not layers or any(m is None or m.kind is not QuantKind.NVFP4 for m in methods):
            return
        if not all(hasattr(e, "_nvfp4_banks") for e in layers):
            return
        device = torch.device(device_backend.device_type())
        pieces = None if config.use_dummy_weight else iter_expert_pieces(
            config.model_path, model_config, QuantKind.NVFP4
        )
        banks = build_expert_banks(
            methods[0], len(layers), pieces,
            device=device, bank_device=device, dummy=config.use_dummy_weight,
        )
        for layer_id, experts in enumerate(layers):
            experts._nvfp4_banks = tuple(banks.sources[role][layer_id] for role in BANK_ROLES)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """The MoE releases share the dense code path: the decoder picks the routed or dense MLP from config.num_experts."""


class Qwen3_5ForConditionalGeneration(QwenVLVisionMixin, Qwen3_5ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.5 consumes no DeepStack features"
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The MoE releases with the vision tower; see Qwen3_5MoeForCausalLM."""


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]
