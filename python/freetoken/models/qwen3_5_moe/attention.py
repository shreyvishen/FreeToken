from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.kernel.backend import is_mps
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearColParallelMerged, LinearReplicated, sigmoid_gate_mul
from freetoken.layers.quantization.linear.fp8_tensor import MetalFp8TensorLinearKernel
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5Attention(BaseOP):
    """Gated full attention: per-head output gate, q/k RMSNorm, partial NeoX rope.

        query, gate = chunk(q_proj(x).view(.., num_q, head_dim*2), 2, -1)
        q = qnorm(query); k = knorm(k_proj(x)); v = v_proj(x)
        q, k = rope(q, k)                       # first rotary_dim dims
        attn = paged_attention(q, k, v)
        out = o_proj(attn * sigmoid(gate))

    TP note: uses replicated linears (tp=1 correctness milestone); swap to
    column/row-parallel for tensor parallelism later.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim

        # Fused q/k/v projection (one GEMM instead of three); q half is 2x for the
        # output gate. Split sizes: [num_q*head_dim*2, num_kv*head_dim, num_kv*head_dim].
        self._qkv_split = [self.num_q * head_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.qkv_proj",
        )
        # Qwen3.5 uses Gemma-style (1+weight) RMSNorm; the weight loader bakes the +1
        # into the stored weight (GemmaRMSNorm scales by the raw weight).
        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
            mrope_section=(
                tuple(config.rotary_config.mrope_section)
                if config.rotary_config.mrope_section is not None
                else None
            ),
            mrope_layout=config.rotary_config.mrope_layout,
        )
        self.o_proj = LinearReplicated(
            self.qo_attn_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )

    def fuses_input_norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """True when forward(..., norm=) runs the residual-add RMSNorm inside the qkv
        projection's launch (Metal fp8 GEMV at decode M)."""
        if self._metal_fp8_qkv() is None:
            return False
        from freetoken.kernel.metal import fp8

        return x.shape[0] <= fp8.MAX_GEMV_ROWS and fp8.can_run_norm(x, x, residual, x, weight)

    def _metal_fp8_qkv(self):
        """``qkv_proj``'s kernel when the fp8 table selected the Metal one, else None."""
        kernel = getattr(self.qkv_proj.quant_method, "kernel", None)
        return kernel if isinstance(kernel, MetalFp8TensorLinearKernel) else None

    def _project_fused(self, qkv: torch.Tensor, ctx):
        """Metal: the split, both norms, the rope, the gate copy and this token's KV write as
        one launch, k and v landing in the cache directly."""
        if not is_mps():
            return None
        from freetoken.kernel.metal import qkv_epilogue as ke

        batch = ctx.batch
        positions = batch.get_attn_positions()
        if positions.dim() != 1:
            return None  # mrope feeds [3, N] rows; the kernel takes one position per token
        kc = ctx.kv_cache.k_cache(self.layer_id).view(-1, self.num_kv, self.head_dim)
        vc = ctx.kv_cache.v_cache(self.layer_id).view(-1, self.num_kv, self.head_dim)
        # One arg list, passed by name, so supports() and the kernel cannot drift apart.
        args = dict(
            qkv=qkv, num_q=self.num_q, num_kv=self.num_kv, head_dim=self.head_dim,
            q_weight=self.q_norm.weight, k_weight=self.k_norm.weight,
            cos_sin_cache=self.rotary._cos_sin_cache,
            positions=positions, k_cache=kc, v_cache=vc, out_loc=batch.out_loc,
        )
        if not ke.supports(**args):
            return None
        return ke.qkv_epilogue_metal(**args, eps=self.q_norm.eps, is_neox=self.rotary.is_neox)

    def _project(self, qkv: torch.Tensor):
        """Returns (q, k, v, gate) post qk-norm+rope, the chain _project_fused fuses."""
        positions = get_global_ctx().batch.get_attn_positions()
        qg, k, v = torch.split(qkv, self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()  # [N, num_q, head_dim]
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.view(-1, self.num_kv, self.head_dim).contiguous()
        v = v.contiguous()  # split view has the qkv row stride; the KV store needs contiguous
        q = self.q_norm.forward(q).reshape(-1, self.qo_attn_dim)
        k = self.k_norm.forward(k).reshape(-1, self.kv_attn_dim)
        q, k = self.rotary.forward(positions, q, k)
        return q.view(-1, self.num_q, self.head_dim), k, v, gate

    def _combine(self, attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gated = sigmoid_gate_mul(attn_out.reshape(-1, self.qo_attn_dim), gate)
        return self.o_proj.forward(gated)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor, *, norm=None) -> torch.Tensor:
        """norm=(residual, residual_out, weight, eps): the residual-add RMSNorm runs in
        the qkv projection's prologue. Only valid when fuses_input_norm said so."""
        ctx = get_global_ctx()
        if norm is not None:
            r, r_out, nw, eps = norm
            qkv = self._metal_fp8_qkv().apply_fused(self.qkv_proj, x, norm=(x, r, r_out, nw, eps))
        else:
            qkv = self.qkv_proj.forward(x)
        fused = self._project_fused(qkv, ctx)
        if fused is None:
            q, k, v, gate = self._project(qkv)
        else:
            # k and v are already in the cache; the backend skips its own store.
            q, gate = fused
            k = v = None
        backend = ctx.attn_backend
        if fused is not None and getattr(backend, "fuses_output_gate", False):
            # The output gate rides in the attention kernel's epilogue.
            o = backend.forward(q, k, v, self.layer_id, ctx.batch, gate=gate)
            return self.o_proj.forward(o.reshape(-1, self.qo_attn_dim))
        o = backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self._combine(o, gate)


__all__ = ["Qwen3_5Attention"]
