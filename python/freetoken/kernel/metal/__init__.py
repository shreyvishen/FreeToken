"""Metal / MPS kernels for Apple silicon, compiled at runtime."""

from __future__ import annotations

from .nvfp4 import (
    E2M1_VALUES,
    as_e4m3_bytes,
    dequant_nvfp4,
    dequant_nvfp4_dense,
    moe_decode_nvfp4,
    moe_prefill_nvfp4,
    moe_prefill_nvfp4_grouped,
    nvfp4_gemv,
)

from .shaders import is_available

__all__ = [
    "E2M1_VALUES", "as_e4m3_bytes", "dequant_nvfp4", "dequant_nvfp4_dense", "is_available",
    "moe_decode_nvfp4", "moe_prefill_nvfp4", "moe_prefill_nvfp4_grouped", "nvfp4_gemv",
]
