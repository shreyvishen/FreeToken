from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, GatedRMSNorm, LinearColParallelMerged, LinearReplicated
from freetoken.layers.quantization import QuantConfig
from freetoken.layers.quantization.linear.fp8_tensor import MetalFp8TensorLinearKernel

from .gdn_kernels import gdn_decode_fla, gdn_decode_fused, gdn_prefill_chunk_fla


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class Qwen3_5GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, *, quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6/3.8 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # quantized checkpoints quantize qkv|z but not b|a, so the fusion splits into a qkvz GEMM and a ba GEMM with their own schemes (matches sglang / vLLM)
        self._split_in_proj = (
            quant_config is not None and quant_config.scheme_for(f"{prefix}.in_proj_qkvz") is not None
        )

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._split_in_proj:
            self.in_proj_qkvz = LinearColParallelMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_qkvz",
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_ba",
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(
                hidden_size, self._in_proj_split, has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj",
            )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps)
        self.out_proj = LinearReplicated(
            self.value_dim, hidden_size, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.out_proj",
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state,
                      host=None) -> torch.Tensor:
        """Varlen causal conv with silu; reads/updates each request's conv state in
        place by cache_indices slot."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state, host)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (fla's h is [V,K] and
        the Metal h is the pool's own [K,V]; both coincide with the pool because GDN requires
        head_k_dim == head_v_dim). Conv: the last (kernel-1) raw conv-input timesteps ending at
        the boundary. index_put_, not index_copy_: MPS costs the latter O(destination), 131x
        here on a 133-slot pool (kernel/metal/ops.py:store_cache)."""
        rec = pool.recurrent_states[li]
        dst = (fla.track_dst,)
        rec.index_put_(dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_put_(dst, conv_win.to(cv.dtype))

    def _metal_fp8_in_proj(self):
        """in_proj_qkvz's kernel when the fp8 table selected the Metal one, else None
        (a bf16 checkpoint builds in_proj instead: nothing to fuse into)."""
        proj = getattr(self, "in_proj_qkvz", None)
        kernel = None if proj is None else getattr(proj.quant_method, "kernel", None)
        return kernel if isinstance(kernel, MetalFp8TensorLinearKernel) else None

    def fuses_input_norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """True when forward(..., norm=) runs the residual-add RMSNorm inside the input
        projection's launch (Metal fp8 GEMV at decode M)."""
        if self._metal_fp8_in_proj() is None:
            return False
        from freetoken.kernel.metal import fp8

        return x.shape[0] <= fp8.MAX_GEMV_ROWS and fp8.can_run_norm(x, x, residual, x, weight)

    def forward(self, hidden_states: torch.Tensor, *, norm=None) -> torch.Tensor:
        """norm=(residual, residual_out, weight, eps): the residual-add RMSNorm runs in
        the input projection's prologue. Only valid when fuses_input_norm said so."""
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype
        if norm is not None:
            r, r_out, nw, eps = norm
            norm = (hidden_states, r, r_out, nw, eps)

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        li = pool.local_index(self.layer_id)
        conv_done = False  # the decode conv ran inside the projection's epilogue
        if self._split_in_proj:
            metal_fp8 = self._metal_fp8_in_proj()
            if metal_fp8 is not None:
                from freetoken.kernel.metal import fp8
            if (
                metal_fp8 is not None
                and self.in_proj_ba.bias is None
                and fp8.can_run_extra(hidden_states, self.in_proj_ba.weight)
            ):
                # bf16 b|a rides on the fp8 qkv|z launch; at decode so does the conv
                # step over q|k|v (state advanced in place).
                conv = None
                if batch.is_decode:
                    cw, cs = self._conv_weight(), pool.conv_states[li]
                    if fp8.can_run_conv(hidden_states, self.in_proj_qkvz.weight, cw, cs):
                        conv = (cw, cs, fla.cache_indices)
                qkvz, ba = metal_fp8.apply_fused(
                    self.in_proj_qkvz, hidden_states,
                    extra_weight=self.in_proj_ba.weight, conv=conv, norm=norm,
                )
                conv_done = conv is not None
            else:
                qkvz = self.in_proj_qkvz.forward(hidden_states)
                ba = self.in_proj_ba.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)

        if batch.is_decode:
            mixed = (conv_in if conv_done else self._conv_decode(conv_in, fla.cache_indices, pool))
            # Metal fused decode: l2 norms, gating, recurrence and gated output norm in one
            # launch (kernel/metal/gdn.py). Bit-identical to the chain fallback below.
            fused = gdn_decode_fused(
                mixed, z, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                norm_weight=self.norm.weight, norm_eps=self.norm.eps,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                scale=self.head_k_dim ** -0.5, num_k_heads=self.num_k_heads,
                head_k_dim=self.head_k_dim, activation=self.norm.activation,
            )
            if fused is not None:
                return self.out_proj.forward(fused)
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            core_out = gdn_decode_fla(
                q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
            )
        else:
            host = (
                None if fla.cu_seqlens_host is None
                else (fla.cu_seqlens_host, fla.cache_indices_host, fla.has_initial_state_host)
            )
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state, host)
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            result = gdn_prefill_chunk_fla(
                q, k, v, g, beta,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                return_h=track, cu_seqlens_host=fla.cu_seqlens_host,
            )
            if track:
                core_out, h = result
                self._write_track_snapshot(pool, li, conv_in, h, fla)
            else:
                core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen3_5GatedDeltaNet"]
