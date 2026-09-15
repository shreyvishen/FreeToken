"""Bank and format names on disk (FTW), in the CPU executor and in the GGUF provider; the kernels use the canonical roles."""

from __future__ import annotations

from freetoken.layers.quantization import QuantKind

CANONICAL_ROLE = {
    "gate_up_packed": "gate_up",
    "gate_up_blocks": "gate_up",
    "gate_up_scales": "gate_up_scale",
    "down_packed": "down",
    "down_blocks": "down",
    "down_scales": "down_scale",
}


def canonical_role(bank_name: str) -> str:
    return CANONICAL_ROLE.get(bank_name, bank_name)


def legacy_bank_names(quant_format: str) -> dict[str, str]:
    """Canonical role -> the bank name FTW files store for ``quant_format``."""
    from freetoken.moe.offload_cache import _BANK_SCHEMAS

    return {canonical_role(name): name for name in _BANK_SCHEMAS[quant_format]}


def nvfp4_bank_specs(num_experts: int, hidden: int, intermediate: int) -> dict:
    """The six native NVFP4 banks under the names the cache and FTW use: one ``(shape,
    dtype)`` per bank, with the expert dim prepended."""
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    cfg = MoEConfig(num_experts=num_experts, hidden=hidden, intermediate=intermediate, top_k=1)
    names = legacy_bank_names("nvfp4")
    return {
        names[role]: ((num_experts, *spec.shape), spec.dtype)
        for role, spec in TritonNvfp4MoEKernel().layout(cfg).items()
    }


# (kind, kernel) <-> the quant_format tag
LEGACY_FORMAT = {
    (QuantKind.NONE, "fused"): "bf16",
    (QuantKind.FP8_BLOCK, "triton"): "fp8_block",
    # metal first: its banks are the triton layout byte for byte, and the inverse map
    # (built below, last entry wins) must name triton for a CUDA host reading an FTW file
    (QuantKind.NVFP4, "metal"): "nvfp4",
    (QuantKind.NVFP4, "triton"): "nvfp4",
    (QuantKind.NVFP4, "marlin"): "nvfp4_marlin",
    (QuantKind.NVFP4, "b12x"): "nvfp4_b12x",
    (QuantKind.MXFP4, "triton_gptoss"): "mxfp4_triton",
    (QuantKind.MXFP4, "triton"): "ds_fp4",
}
_KIND_KERNEL = {fmt: kk for kk, fmt in LEGACY_FORMAT.items()}


def legacy_format_for(kind: QuantKind, kernel: str) -> str:
    return LEGACY_FORMAT[(kind, kernel)]


def kind_kernel_for(legacy_format: str) -> tuple[QuantKind, str]:
    return _KIND_KERNEL[legacy_format]
