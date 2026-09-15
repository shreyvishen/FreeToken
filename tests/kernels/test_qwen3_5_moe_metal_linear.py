"""The Metal NVFP4 and per-tensor-FP8 dense linears, the W4A16 LM head and the ``--dense-quant-override fp8`` loader half against the bf16 dequant-then-``F.linear`` path they replace (the GEMVs themselves are in test_metal_nvfp4.py and test_metal_ops.py)."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.metal import is_available
from freetoken.kernel.metal.nvfp4 import dequant_nvfp4_dense
from freetoken.layers.quantization import LayerKind, NoQuantConfig, QuantKind, method_class
from freetoken.layers.quantization.linear.base import LinearConfig
from freetoken.layers.quantization.linear.fp8_tensor import MetalFp8TensorLinearKernel
from freetoken.layers.quantization.linear.nvfp4 import MetalNvfp4LinearKernel
from freetoken.layers.quantization.scheme import fp8_tensor_scheme, nvfp4_scheme
from freetoken.models.qwen3_5_moe.config import parse_config
from freetoken.models.qwen3_5_moe.weight import _dequant

pytestmark = pytest.mark.skipif(not is_available(), reason="needs a torch build with MPS")

K, N = 512, 250   # N divides no shipped tile, so the tail threadgroup runs
NVFP4, FP8 = MetalNvfp4LinearKernel(), MetalFp8TensorLinearKernel()


def _nvfp4_layer(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g).to("mps")
    # e4m3 bytes in a range that decodes to a sane block scale (exponent around the bias)
    scale = torch.randint(0x38, 0x44, (n, k // 16), dtype=torch.uint8, generator=g).to("mps")
    glob = (torch.rand(n, generator=g) * 0.1 + 0.01).to(torch.float16).to("mps")
    return types.SimpleNamespace(weight=packed, weight_scale=scale, weight_global=glob,
                                 bias=None, in_features=k, out_features=n), (packed, scale, glob)


@torch.inference_mode()
def test_nvfp4_linear_and_lm_head():
    layer, banks = _nvfp4_layer(N, K, seed=5)
    x = torch.randn(1, K, generator=torch.Generator().manual_seed(7)).bfloat16().to("mps")
    got = NVFP4.apply(layer, x)
    want = F.linear(x, dequant_nvfp4_dense(*banks, dtype=torch.bfloat16))
    assert got.dtype == x.dtype   # a plain projection follows the residual stream
    # measured max abs 1.2e-01
    torch.testing.assert_close(got.float().cpu(), want.float().cpu(), rtol=3e-2, atol=3e-2)
    assert NVFP4.apply(layer, x.reshape(1, 1, K)).shape == (1, 1, N)   # leading dims survive
    method = method_class(QuantKind.NVFP4, LayerKind.LINEAR)(
        LinearConfig(K, N, scheme=nvfp4_scheme(input_scale=False)), "auto")
    assert isinstance(method.kernel, MetalNvfp4LinearKernel)

    layer, banks = _nvfp4_layer(N, 128, seed=31)
    layer.num_embeddings = N   # what marks this projection as the LM head
    before = tuple(t.clone() for t in banks)
    NVFP4.finalize(layer)   # the Metal kernel keeps the layout; this must not move a byte
    for got, want in zip(banks, before):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    x = torch.randn(1, 128, generator=torch.Generator().manual_seed(32)).to("mps")
    got = NVFP4.apply(layer, x)
    want = x @ dequant_nvfp4_dense(*banks, dtype=torch.float32).T
    assert got.shape == (1, N)
    assert ((got - want).abs().max() / want.abs().max()).item() < 1e-2
    # logits stay float32 whatever the residual stream is
    assert NVFP4.apply(layer, x.bfloat16()).dtype is torch.float32


def test_fp8_linear():
    scalar = 0.041
    w8 = (torch.randn(N, K, generator=torch.Generator().manual_seed(3)) * 0.3).to(
        torch.float8_e4m3fn)
    layer = types.SimpleNamespace(weight=w8.to("mps"), weight_scale=torch.full((N,), scalar).to(
        "mps"), bias=None, in_features=K, out_features=N)
    FP8.finalize(layer)
    x = torch.randn(1, K, generator=torch.Generator().manual_seed(9)).bfloat16().to("mps")
    want = F.linear(x, _dequant(fp8_tensor_scheme("weight_scale", per_row=True), {
        "weight": w8.to("mps"), "weight_scale": torch.full((w8.shape[0],), scalar, device="mps")}))
    # measured max abs 3.9e-03
    torch.testing.assert_close(FP8.apply(layer, x).float().cpu(), want.float().cpu(),
                               rtol=3e-2, atol=3e-2)
    method = method_class(QuantKind.FP8_TENSOR, LayerKind.LINEAR)(
        LinearConfig(K, N, scheme=fp8_tensor_scheme("weight_scale", per_row=True)), "auto")
    assert isinstance(method.kernel, MetalFp8TensorLinearKernel)


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
    from freetoken.layers.quantization.linear.unquantized import MetalLinearKernel
    from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
    from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
    from freetoken.models.qwen3_5_moe.moe import _SharedExpert
    from freetoken.models.qwen3_5_moe.weight import _DenseReader
    from freetoken.models.register import get_model_spec

    hf = types.SimpleNamespace(text_config=types.SimpleNamespace(**_HF_TEXT), model_type="qwen3_5_moe",
                               quantization_config=_NVFP4_BF16_DENSE,
                               architectures=["Qwen3_5MoeForConditionalGeneration"])
    cfg = parse_config(hf)
    if try_get_tp_info() is None:   # the bf16 in_proj_ba reads the TP info; the test runs at TP=1
        set_tp_info(rank=0, size=1)
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
    # and the keys the model asks for are exactly the pair _DenseReader._emit yields, which must
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
