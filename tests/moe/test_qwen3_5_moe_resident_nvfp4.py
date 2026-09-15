"""Resident NVFP4 experts on Metal: MoELayer's bank wiring and a layer-0 block against the CPU float32 oracle."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.metal import is_available
from freetoken.layers.moe import MoELayer
from freetoken.layers.quantization import NoQuantConfig
from freetoken.layers.quantization.scheme import nvfp4_scheme
from tests.kernels._refs import banks, moe_ref, routing

pytestmark = pytest.mark.skipif(not is_available(), reason="needs a torch build with MPS")


# a QuantConfig reporting one scheme for every module: the shortest way to build a layer of a
# given kind without a checkpoint
class _KindQuant(NoQuantConfig):
    def __init__(self, scheme):
        super().__init__()
        self._scheme = scheme

    def scheme_for_name(self, name):
        return self._scheme


@torch.inference_mode()
def test_layer0_parity(monkeypatch):
    from freetoken import core
    from freetoken.core import Batch, Context

    e, h, i, top_k, m = 8, 80, 48, 4, 4
    gu, dn = banks(e, 2 * i, h, seed=1), banks(e, h, i, seed=2)
    ids, weights = routing(m, e, top_k, seed=3)
    x = torch.randn(m, h, generator=torch.Generator().manual_seed(4)).bfloat16()
    want = moe_ref(x, gu, dn, weights, ids)
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer = MoELayer(num_experts=e, top_k=top_k, hidden_size=h, intermediate_size=i,
                     quant_config=_KindQuant(nvfp4_scheme(input_scale=False)), prefix="experts")
    assert layer._nvfp4_banks is None and layer.state_dict() == {}
    layer._nvfp4_banks = tuple(t.to("mps") for t in (*gu, *dn))
    # monkeypatch hands the next test its own global context back
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx := Context(1))
    # decode and prefill take different kernels inside the same layer
    for phase, n in (("decode", 2), ("prefill", m)):
        xs, ws, ix = (t[:n].to("mps") for t in (x, weights, ids))
        with ctx.forward_batch(Batch(reqs=[], phase=phase)):
            got = layer._resident_gemm(xs, ws, ix)
        assert got.shape == xs.shape and got.dtype == xs.dtype
        cos = F.cosine_similarity(got.float().cpu(), want[:n], dim=-1).min().item()
        assert cos > 0.999, f"{phase}: min cosine {cos:.6f}"
