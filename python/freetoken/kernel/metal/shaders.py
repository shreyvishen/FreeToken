"""The substrate every kernel module here compiles through: MSL source text in, a launch-
recording library out, one library per exact source text."""

from __future__ import annotations

from functools import lru_cache

import torch

# MSL scalar name per torch dtype. bfloat needs Metal 3.1 (macOS 14+).
_MSL_TYPE = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}


MSL_INT = {torch.int32: "int", torch.int64: "long", torch.uint32: "uint"}

MAX_TG_FLOATS = 32768 // 4   # the threadgroup memory limit, in float32 slots

# The e4m3 bits placed in the fp16 field are exact, subnormals included: bitcast, rescale.
E4M3_U8_TO_F32 = r"""
inline float e4m3_u8_to_f32(uchar v) {
    ushort h = (ushort(v & 0x80) << 8) | (ushort(v & 0x7F) << 7);
    return float(as_type<half>(h)) * 256.0f;
}
"""


def ept(head_dim: int) -> int:
    """Elements per thread when a 32-lane SIMD-group splits one ``head_dim``-wide row."""
    return -(-head_dim // 32)


def msl_type(dtype: torch.dtype) -> str:
    try:
        return _MSL_TYPE[dtype]
    except KeyError:
        raise TypeError(f"no Metal scalar type for {dtype}") from None


def pick_tile(tiles, n_rows: int, min_tgs: int = 0, groups: int = 1) -> tuple[int, int]:
    """The first ``(nsg, nr0)`` of ``tiles`` whose ``nsg * nr0`` divides ``n_rows`` and whose
    ``ceil(n_rows / (nsg * nr0)) * groups`` threadgroups reach ``min_tgs``; the first dividing
    tile if none does, ``(1, 1)`` if none divides. Division keeps every threadgroup full."""
    fits = [(nsg, nr0) for nsg, nr0 in tiles if n_rows % (nsg * nr0) == 0]
    for nsg, nr0 in fits:
        if -(-n_rows // (nsg * nr0)) * groups >= min_tgs:
            return nsg, nr0
    return fits[0] if fits else (1, 1)


def is_available() -> bool:
    return (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and hasattr(torch.mps, "compile_shader")
    )


# The library comes back behind a proxy so engine/mps_tape.py can record every launch.
_recorder = None  # callable(fn, args, kwargs) while a tape records, else None

def set_recorder(callback) -> None:
    global _recorder
    _recorder = callback


class TapedLibrary:
    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name):
        fn = getattr(self._lib, name)

        def kernel(*args, **kwargs):
            if _recorder is not None:
                _recorder(fn, args, kwargs)
            return fn(*args, **kwargs)

        setattr(self, name, kernel)
        return kernel


@lru_cache(maxsize=None)
def compile(source: str) -> TapedLibrary:
    return TapedLibrary(torch.mps.compile_shader(source))
