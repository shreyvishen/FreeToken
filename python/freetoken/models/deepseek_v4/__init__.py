"""DeepSeek-V4-Flash support for FreeToken.

This package ports the official ``inference/model.py`` reference (MLA + CSA/HCA
compressors + Lightning Indexer + manifold-constrained Hyper-Connections) onto
FreeToken's primitives. The exotic ops are reimplemented as Triton kernels (see
``freetoken/kernel/triton/dsv4_*.py``); the routed FP4 experts are served from
FreeToken's :class:`~freetoken.moe.offload_cache.OffloadMoeCache` so only a subset
of experts is resident on the GPU (the framework's core acceleration).

DeepSeek-V4-Flash is a first-class registered model on the shared paged-KV engine:
its window / compressed-attention / compressed-index KV live in DSV4-owned pools
addressed by page tables, and sparse attention is a physical-slot gather (see
:mod:`freetoken.attention.dsv4_sparse` and :mod:`freetoken.kvcache.dsv4_paged_pool`).
"""

import importlib

from .args import DeepseekV4Args, load_args
from .config import parse_config
from .weight import iter_expert_pieces, iter_weights

# The model module reaches the DSV4 Triton kernels at import time, and triton is Linux-only
# (pyproject).
_LAZY = {"DeepseekV4ForCausalLM": ".model"}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "DeepseekV4Args",
    "load_args",
    "parse_config",
    "DeepseekV4ForCausalLM",
    "iter_weights",
    "iter_expert_pieces",
]
