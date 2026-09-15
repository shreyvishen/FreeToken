"""The Metal NVFP4 and per-tensor-FP8 dense linears and the W4A16 LM head against the bf16 dequant-then-``F.linear`` path they replace (the GEMVs themselves are checked in tests/kernels)."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel.metal import is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="needs a torch build with MPS")

from freetoken.kernel.metal.nvfp4 import dequant_nvfp4_dense  # noqa: E402  (after the guard)
from freetoken.layers.quantization import LayerKind, QuantKind, method_class  # noqa: E402
from freetoken.layers.quantization.linear.base import LinearConfig  # noqa: E402
from freetoken.layers.quantization.linear.fp8_tensor import MetalFp8TensorLinearKernel  # noqa: E402
from freetoken.layers.quantization.linear.nvfp4 import MetalNvfp4LinearKernel  # noqa: E402
from freetoken.layers.quantization.scheme import fp8_tensor_scheme, nvfp4_scheme  # noqa: E402
from freetoken.layers.quantization.scheme import fp8_tensor_scheme  # noqa: E402
from freetoken.models.qwen3_5_moe.weight import _dequant  # noqa: E402

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


def test_nvfp4_linear():
    layer, banks = _nvfp4_layer(N, K, seed=5)
    x = torch.randn(1, K, generator=torch.Generator().manual_seed(7)).bfloat16().to("mps")
    got = NVFP4.apply(layer, x)
    want = F.linear(x, dequant_nvfp4_dense(*banks, dtype=torch.bfloat16))
    assert got.dtype == x.dtype
    torch.testing.assert_close(got.float().cpu(), want.float().cpu(), rtol=3e-2, atol=3e-2)
    assert NVFP4.apply(layer, x.reshape(1, 1, K)).shape == (1, 1, N)   # leading dims survive
    method = method_class(QuantKind.NVFP4, LayerKind.LINEAR)(
        LinearConfig(K, N, scheme=nvfp4_scheme(input_scale=False)), "auto")
    assert isinstance(method.kernel, MetalNvfp4LinearKernel)


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
    torch.testing.assert_close(FP8.apply(layer, x).float().cpu(), want.float().cpu(),
                               rtol=3e-2, atol=3e-2)
    method = method_class(QuantKind.FP8_TENSOR, LayerKind.LINEAR)(
        LinearConfig(K, N, scheme=fp8_tensor_scheme("weight_scale", per_row=True)), "auto")
    assert isinstance(method.kernel, MetalFp8TensorLinearKernel)


@torch.inference_mode()
def test_lm_head():
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
    # logits stay float32 whatever the residual stream is; a plain projection follows it
    assert NVFP4.apply(layer, x.bfloat16()).dtype is torch.float32
    assert NVFP4.apply(_nvfp4_layer(N, 128, seed=31)[0], x.bfloat16()).dtype is torch.bfloat16
