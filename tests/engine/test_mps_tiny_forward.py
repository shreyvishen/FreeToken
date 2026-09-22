"""Tiny Qwen3.5-MoE engine on the Apple GPU: prefill, decode, and decode replayed off the tape."""

import contextlib
import copy
import json
import pathlib
import tempfile

import pytest
import torch

from freetoken.kernel.backend import is_mps

mps = pytest.mark.skipif(not is_mps(), reason="the engine picks Metal only on an Apple GPU without CUDA")

# head_dim 64: layers/rotary.py asserts head_size in (64, 128, 256, 512); GDN key and value head dims must match
_TEXT = {
    "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2, "vocab_size": 512,
    "layer_types": ["linear_attention", "full_attention"], "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 64, "partial_rotary_factor": 0.125,
    "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "hidden_act": "silu", "tie_word_embeddings": True, "torch_dtype": "bfloat16",
    "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64, "norm_topk_prob": True, "linear_num_key_heads": 2,
    "linear_num_value_heads": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
}

# parse_config reads ``text_config``: flat top-level fields would fall back to the 40-layer default
TINY_MOE = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
    "text_config": dict(_TEXT, model_type="qwen3_5_moe"),
}

_TAPE_LAYERS = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]

# the tiny model quantized the way the tape ships it: NVFP4 experts/lm_head, FP8 dense
def _tape_config() -> dict:
    fp4, fp8 = {"quant_algo": "W4A16_NVFP4", "group_size": 16}, {"quant_algo": "FP8"}
    dense = {"linear_attention": ("linear_attn", ("in_proj_qkv", "in_proj_z", "out_proj")),
             "full_attention": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj"))}
    layers, shared = {"lm_head": fp4}, ("gate", "up", "down")   # NVFP4 everywhere but dense
    for i, kind in enumerate(_TAPE_LAYERS):
        pre, (mod, projs) = f"model.language_model.layers.{i}", dense[kind]
        layers.update({f"{pre}.{mod}.{n}": fp8 for n in projs})
        layers.update({f"{pre}.mlp.{m}": fp4 for m in
                       ("experts", *(f"shared_expert.{n}_proj" for n in shared))})
    c = copy.deepcopy(TINY_MOE)
    c["text_config"].update(num_hidden_layers=len(_TAPE_LAYERS), layer_types=_TAPE_LAYERS,
                            num_experts=16, num_experts_per_tok=4, max_position_embeddings=512,
                            tie_word_embeddings=False)
    c["quantization_config"] = dict(quant_algo="MIXED_PRECISION", quant_method="modelopt",
                                    quantized_layers=layers)
    return c


# A live Engine over random weights; the TP info and context are per-process, so the next test gets them back
@contextlib.contextmanager
def engine_on_mps(config: dict, plan: bool = False):
    import torch.distributed as dist

    from freetoken import core
    from freetoken.distributed import DistributedInfo, info
    from freetoken.engine import Engine
    from freetoken.engine.config import EngineConfig

    if dist.is_initialized():
        dist.destroy_process_group()
    saved, (info._TP_INFO, core._GLOBAL_CTX) = (info._TP_INFO, core._GLOBAL_CTX), (None, None)
    with tempfile.TemporaryDirectory(prefix="ft-tiny-moe-") as d:
        pathlib.Path(d, "config.json").write_text(json.dumps(config))
        engine = Engine(EngineConfig(
            model_path=d, tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16,
            max_running_req=1, use_dummy_weight=True, use_pynccl=False, page_size=1,
            cuda_graph_bs=[1] if plan else [], max_seq_len_override=64,
            num_page_override=64, moe_backend="fused"))
        try:
            with torch.inference_mode():
                yield engine
        finally:
            engine.shutdown()
            info._TP_INFO, core._GLOBAL_CTX = saved


# one forward through the engine's own dummy request row, eager or replayed off the tape
def forward(engine, phase, *, length=16, position=0, plan=False, replay=False, token=0):
    from freetoken.core import Batch, Req

    dev, row = engine.device, engine.page_table[engine.dummy_req.table_idx]
    n, cached = (length, 0) if (prefill := phase == "prefill") else (1, position)
    span = slice(0, length) if prefill else slice(position, position + 1)
    # the request's own tokens sit in KV slots numbered like their positions
    row[span] = ids = torch.arange(cached, cached + n, dtype=torch.int32, device=dev)
    req = Req(input_ids=torch.zeros(cached + n, dtype=torch.int32),
              table_idx=engine.dummy_req.table_idx, cached_len=cached, output_len=1, uid=-1,
              sampling_params=None, cache_handle=None)  # type: ignore[arg-type]
    req.linear_slot_idx = engine.dummy_req.linear_slot_idx
    batch = Batch(reqs=[req], phase=phase)
    batch.padded_reqs, batch.out_loc, batch.positions = batch.reqs, row[span], ids
    batch.input_ids = torch.full((n,), token, dtype=torch.int32, device=dev)
    batch.linear_table_idx = torch.tensor([req.linear_slot_idx or 0], dtype=torch.int32, device=dev)
    if plan:
        engine.graph_runner.pad_batch(batch)
    engine.attn_backend.prepare_metadata(batch)
    if replay:
        assert engine.graph_runner.can_use_cuda_graph(batch)
        return engine.graph_runner.replay(batch).clone()
    with engine.ctx.forward_batch(batch):
        return engine.model.forward().clone()


@mps
def test_tiny_moe_decode_on_mps():
    with engine_on_mps(TINY_MOE) as engine:
        assert engine.device.type == "mps"
        prefill = forward(engine, "prefill", length=8)
        logits = forward(engine, "decode", position=8)
    assert prefill.device.type == "mps" and prefill.ndim == 2 and prefill.shape[-1] == _TEXT["vocab_size"]
    assert torch.isfinite(prefill.float()).all()
    assert logits.shape == (1, _TEXT["vocab_size"]) and torch.isfinite(logits.float()).all()


@mps
def test_plan_replay_matches_eager():
    def tape(engine, replay):
        forward(engine, "prefill", plan=True)
        return [forward(engine, "decode", position=16 + i, plan=True, replay=replay,
                        token=(16 + i) * 7 % 512) for i in range(8)]

    with engine_on_mps(_tape_config(), plan=True) as engine:
        mc = engine.config.model_config
        assert mc.expert_quant == "nvfp4"  # the dense and lm_head schemes come from the QuantConfig
        assert engine.graph_runner.max_graph_bs == 1, "the plan was not recorded"
        # measured 39 fused ops (embed, GDN 8/7/7, attention 8, flushes, lm_head); under 30 a layer is missing
        assert len(engine.graph_runner.graph_map[1]) >= 30
        eager, plan, again = (tape(engine, r) for r in (False, True, False))
    for i, (a, b, c) in enumerate(zip(eager, plan, again)):
        assert torch.isfinite(a).all() and torch.equal(a, c), f"eager is nondeterministic at {i}"
        assert torch.equal(a, b), f"plan differs from eager at step {i}: {(a - b).abs().max()}"


@mps
def test_the_tape_replays_a_host_step_on_fresh_data():
    """A host step (the SSD tier's routing read) is kept whole on the tape: each replay runs it
    again between the launches around it, on what those launches just wrote."""
    from freetoken.engine.mps_tape import DecodeTape
    from freetoken.kernel.backend import host_step

    seen = []

    class Tier:
        @host_step
        def read(self, t):
            seen.append(t.item())
            t.add_(10)   # untraced: the replay reruns the step, not this launch

    tier, x, out = Tier(), torch.zeros(1, device="mps"), torch.zeros(1, device="mps")
    tape = DecodeTape.record(lambda: (x.add_(1), tier.read(x), out.copy_(x * 2)))
    with torch.inference_mode():   # the engine replays inside it, as the tape recorded
        for _ in range(2):
            tape.replay()
    assert seen == [1.0, 12.0, 23.0] and out.item() == 66.0
