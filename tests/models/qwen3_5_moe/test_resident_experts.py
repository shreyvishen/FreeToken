"""Resident NVFP4 experts on Metal: MoELayer's bank wiring, a layer-0 block against the CPU float32 oracle, and the ``--dense-quant-override fp8`` loader half."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from freetoken.distributed import set_tp_info
from freetoken.kernel.backend import is_mps
from freetoken.layers.moe import MoELayer
from freetoken.layers.quantization import NoQuantConfig
from freetoken.layers.quantization.scheme import nvfp4_scheme
from freetoken.models.config import set_dense_quant_override
from freetoken.models.qwen3_5_moe.config import parse_config
from tests.kernels._refs import banks, moe_ref, routing

pytestmark = pytest.mark.skipif(not is_mps(), reason="needs an Apple GPU")

E, H, I, TOP_K = 8, 64, 32, 2


# TP info and the global context are process-wide singletons; hand each test its own
@contextlib.contextmanager
def tp1():
    from freetoken import core
    from freetoken.distributed import info

    saved = (info._TP_INFO, core._GLOBAL_CTX)
    info._TP_INFO, core._GLOBAL_CTX = None, None
    set_tp_info(rank=0, size=1)
    try:
        yield
    finally:
        info._TP_INFO, core._GLOBAL_CTX = saved


# a QuantConfig reporting one scheme for every module: the shortest way to build a layer of a
# given kind without a checkpoint
class _KindQuant(NoQuantConfig):
    def __init__(self, scheme):
        super().__init__()
        self._scheme = scheme

    def scheme_for_name(self, name):
        return self._scheme


def _layer(e=E, h=H, i=I, top_k=TOP_K):
    return MoELayer(num_experts=e, top_k=top_k, hidden_size=h, intermediate_size=i,
                    quant_config=_KindQuant(nvfp4_scheme(input_scale=False)), prefix="experts")


@torch.inference_mode()
def test_bank_wiring():
    from freetoken.core import Batch, Context, set_global_ctx
    from freetoken.kernel import backend as backend_mod
    from freetoken.kernel.metal.nvfp4 import moe_decode_nvfp4
    from freetoken.layers.quantization import KernelSelectionError

    with tp1():
        layer = _layer()
        assert layer._nvfp4_banks is None and layer.state_dict() == {}
        gu, dn = banks(E, 2 * I, H, seed=7), banks(E, H, I, seed=8)
        layer._nvfp4_banks = bank = tuple(t.to("mps") for t in (*gu, *dn))
        x = torch.randn(2, H, dtype=torch.bfloat16, device="mps")
        weights = torch.tensor([[0.6, 0.4], [0.3, 0.7]], device="mps")
        ids = torch.tensor([[0, 3], [5, 1]], dtype=torch.int32, device="mps")

        set_global_ctx(ctx := Context(1))
        with ctx.forward_batch(Batch(reqs=[], phase="decode")):
            got = layer._resident_gemm(x, weights, ids)
        want = moe_decode_nvfp4(x.float(), *bank, weights, ids).to(x.dtype)
        assert torch.equal(got, want) and got.shape == x.shape and got.dtype == x.dtype
        # the six banks are gate_up then down, and the order is not arbitrary
        swapped = moe_decode_nvfp4(x.float(), *bank[3:], *bank[:3], weights, ids)
        assert not torch.equal(want, swapped.to(x.dtype))

        real, backend_mod.is_mps = backend_mod.is_mps, lambda: False
        try:   # off Metal the resident path has no kernel and must say so, not fall through
            with pytest.raises(KernelSelectionError, match="offload"):
                _layer()
        finally:
            backend_mod.is_mps = real


@torch.inference_mode()
def test_layer0_parity():
    from freetoken.core import Batch, Context, set_global_ctx

    e, h, i, top_k, m = 8, 80, 48, 4, 4
    gu, dn = banks(e, 2 * i, h, seed=1), banks(e, h, i, seed=2)
    ids, weights = routing(m, e, top_k, seed=3)
    x = torch.randn(m, h, generator=torch.Generator().manual_seed(4)).bfloat16()
    want = moe_ref(x, gu, dn, weights, ids)
    with tp1():
        layer = _layer(e, h, i, top_k)
        layer._nvfp4_banks = tuple(t.to("mps") for t in (*gu, *dn))
        set_global_ctx(ctx := Context(1))
        # decode and prefill take different kernels inside the same layer
        for phase, n in (("decode", 1), ("prefill", m)):
            with ctx.forward_batch(Batch(reqs=[], phase=phase)):
                got = layer._resident_gemm(x[:n].to("mps"), weights[:n].to("mps"),
                                           ids[:n].to("mps")).float().cpu()
            cos = F.cosine_similarity(got, want[:n], dim=-1).min().item()
            assert cos > 0.999, f"{phase}: min cosine {cos:.6f}"


# Qwen3.5-122B: routed experts NVFP4, everything dense left bf16 by the ignore list
_NVFP4_BF16_DENSE = {"quant_algo": "NVFP4", "ignore": [
    "lm_head", "model.language_model.layers.0.linear_attn*",
    "model.language_model.layers.0.mlp.shared_expert*"]}
_HF_TEXT = dict(
    num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2, head_dim=64,
    hidden_size=256, vocab_size=1024, hidden_act="silu", rms_norm_eps=1e-6, num_experts=8,
    max_position_embeddings=4096, rope_parameters={"rope_theta": 10000.0},
    num_experts_per_tok=2, moe_intermediate_size=128, shared_expert_intermediate_size=128,
    full_attention_interval=4, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
    tie_word_embeddings=False)


def test_dense_quant_override_builds_w8a16_linears():
    from dataclasses import replace

    from freetoken.layers.quantization import dense_fp8_override
    from freetoken.layers.quantization.linear.fp8_tensor import MetalFp8TensorLinearKernel
    from freetoken.layers.quantization.linear.unquantized import MetalLinearKernel
    from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
    from freetoken.models.qwen3_5_moe.moe import _SharedExpert
    from freetoken.models.qwen3_5_moe.weight import _DenseReader
    from freetoken.models.register import get_model_spec

    hf = SimpleNamespace(text_config=SimpleNamespace(**_HF_TEXT), model_type="qwen3_5_moe",
                         quantization_config=_NVFP4_BF16_DENSE,
                         architectures=["Qwen3_5MoeForConditionalGeneration"])
    set_dense_quant_override("fp8")
    with mock.patch("freetoken.models.qwen3_5_moe.config.is_mps", return_value=True):
        cfg = parse_config(hf)
    set_dense_quant_override(None)   # a process-wide global; parse_config has consumed it
    with tp1():   # the bf16 in_proj_ba reads the TP info; TP=1 is this port's mode
        # what EngineConfig.model_config does: the checkpoint's config, wrapped by the override
        cfg = replace(cfg, quant=dense_fp8_override(NoQuantConfig(), cfg))
        kernel_of = lambda layer: type(layer.quant_method.kernel)
        attn = Qwen3_5Attention(cfg, layer_id=0, prefix="model.layers.0.self_attn")
        gdn = Qwen3_5GatedDeltaNet(cfg.hidden_size, 2, 4, 32, 32, 4, cfg.rms_norm_eps,
                                   layer_id=1, quant_config=cfg.quant,
                                   prefix="model.layers.1.linear_attn")
        shared = _SharedExpert(cfg, cfg.hidden_size, cfg.shared_expert_intermediate_size,
                               prefix="model.layers.0.mlp.shared_expert")
        for layer in (attn.qkv_proj, attn.o_proj, gdn.in_proj_qkvz, gdn.out_proj,
                      shared.gate_up_proj, shared.down_proj):
            assert kernel_of(layer) is MetalFp8TensorLinearKernel
        assert kernel_of(gdn.in_proj_ba) is MetalLinearKernel   # b|a stay bf16
        # and the keys the model asks for are exactly the pair _emit_dense yields, which must
        # reproduce the bf16 weight to within e4m3's own resolution
        name, w = "model.layers.0.mlp.shared_expert.gate_up_proj.weight", torch.randn(
            8, 16, dtype=torch.bfloat16)
        spec, base = get_model_spec(hf.architectures[0]), name[: -len(".weight")]
        assert _DenseReader(NoQuantConfig(), spec)._emit(base, [{"weight": w}], None) == [(name, w)]
        out = dict(_DenseReader(cfg.quant, spec)._emit(base, [{"weight": w}], None))  # override on
        reader = _DenseReader(cfg.quant, spec)  # the loader's own path: the checkpoint's two parts
        assert reader.add(base.replace("gate_up_proj", "gate_proj") + ".weight", w[:4]) == []
        got = dict(reader.add(base.replace("gate_up_proj", "up_proj") + ".weight", w[4:]))
        assert got.keys() == out.keys() and all(torch.equal(got[k], out[k]) for k in out)
        assert set(out) == {name, name + "_scale"} and out[name].dtype == torch.float8_e4m3fn
        err = (out[name].float() * out[name + "_scale"][:, None] - w.float()).norm()
        assert err / w.float().norm() < 0.05
        assert set(shared.state_dict()) == {"gate_up_proj.weight", "gate_up_proj.weight_scale",
                                            "down_proj.weight", "down_proj.weight_scale"}
