from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .backend import is_cuda
from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    if not is_cuda():
        # radix.cpp holds no device code, but building it needs nvcc, so off CUDA the
        # same std::mismatch offset is computed in torch.
        import torch

        n = min(x.numel(), y.numel())
        diff = torch.nonzero(x[:n] != y[:n])
        return n if diff.numel() == 0 else int(diff[0])
    return _load_radix_module().fast_compare_key(x, y)
