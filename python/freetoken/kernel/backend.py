"""Availability probes for the optional native kernel packages.

When flashinfer / sgl_kernel are installed the call-sites use their fused CUDA
ops; otherwise they fall back to the pure-Triton kernels in
``freetoken.kernel.triton``. ``find_spec`` only checks that the package is
importable (no import side effects), and the result is cached.
"""
from __future__ import annotations

import atexit
import contextlib
import functools
import importlib.util
import os
from types import SimpleNamespace

import torch


def _importable(name: str) -> bool:
    # find_spec normally returns None when a package is absent, but it can raise
    # (broken parent package, or a meta_path finder that blocks the name); treat
    # any failure as "not available" so callers cleanly fall back to triton.
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@functools.cache
def is_flashinfer_installed() -> bool:
    return _importable("flashinfer")


@functools.cache
def is_sgl_kernel_installed() -> bool:
    return _importable("sgl_kernel")


@functools.cache
def is_vllm_installed() -> bool:
    return _importable("vllm")


@functools.cache
def device_capability() -> tuple[int, int]:
    """Compute capability of the current device as (major, minor); (0, 0) without CUDA."""
    if not torch.cuda.is_available():
        return (0, 0)
    major, minor = torch.cuda.get_device_capability()
    return (int(major), int(minor))


@functools.cache
def driver_cuda_version() -> int | None:
    """Max CUDA version the installed NVIDIA driver supports (``13000`` == CUDA 13.0),
    or None if undetermined. Driver-JIT kernels (PTX compiled at runtime, e.g.
    flashinfer's CuTe-DSL paths) are gated by this, not by any package's build-time
    toolkit version. Resolved through the ``_pinned_tensor`` extension's link-time
    cudart, so it works wherever the extension builds (including Windows) -- no dlopen
    by soname."""
    try:
        from freetoken.kernel.pinned import _load_pinned_extension

        version = int(_load_pinned_extension().driver_cuda_version())
    except Exception:
        return None
    return version or None  # 0 == no driver installed


@functools.cache
def is_cuda() -> bool:
    """The accelerator is an NVIDIA (or HIP) GPU reachable through ``torch.cuda``."""
    return torch.cuda.is_available()


@functools.cache
def is_mps() -> bool:
    """Apple Metal, and no CUDA: every Metal branch in the engine hangs off this probe."""
    return not is_cuda() and torch.backends.mps.is_available()


def triton_unusable_reason() -> str | None:
    """Why a Triton kernel cannot run here, or None."""
    return "triton is not importable on mps" if is_mps() else None


# MPS submits to one queue in issue order, so a cross-stream wait is already satisfied.
_NO_STREAM = SimpleNamespace(wait_stream=lambda *_args: None)


class _MpsEvent(torch.mps.Event):
    """``torch.mps.Event`` under ``torch.cuda.Event``'s signature: one queue, so the stream argument is ignored."""

    def record(self, stream=None) -> None:
        super().record()


# Pinned staging exists to overlap a PCIe copy: unified memory has none, and an unpinned
# non-blocking H2D returns garbage once its source is freed -- see ``stage_h2d``.
PIN_MEMORY = not is_mps()
NON_BLOCKING = not is_mps()


def shared_host_arena(nbytes: int):
    """An MPS-visible byte buffer and the numpy alias to fill it through, as ``(tensor, array)``.

    ``pin_memory`` on MPS allocates from the shared heap, so ``data_ptr()`` is the id<MTLBuffer>
    and ``[contents]`` its host address: a ``pread`` into the array lands where a device copy off
    the tensor picks it up. Callers still own the copy; this only hands out the two aliases."""
    import ctypes
    import ctypes.util

    import numpy as np

    # off Metal data_ptr() is a plain host pointer and objc_msgSend on it segfaults
    assert is_mps(), "shared_host_arena is Metal-only; elsewhere use a host tensor plus an H2D copy"

    buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    objc = ctypes.CDLL(ctypes.util.find_library("objc"))
    objc.objc_msgSend.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    ptr = objc.objc_msgSend(buf.data_ptr(), objc.sel_registerName(b"contents"))
    length = objc.objc_msgSend(buf.data_ptr(), objc.sel_registerName(b"length")) or 0
    if not ptr or length < nbytes:
        raise RuntimeError(f"pinned MPS tensor is not host-addressable (contents {ptr!r}, length {length})")
    array = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), shape=(nbytes,))
    return buf, array


# Queued-but-unproven H2D sources, in per-iteration generations, with a backstop cap.
_h2d_open: list[torch.Tensor] = []
_h2d_closed: list[list[torch.Tensor]] = []
_H2D_STAGING_CAP = 256


def stage_h2d(host: torch.Tensor) -> bool:
    """Register ``host`` as the source of a copy issued right after this call and return the
    ``non_blocking`` flag it should use; on Metal the source is held until the device
    passes it, because an unpinned async copy is correct only while its source lives."""
    if not is_mps():
        return NON_BLOCKING
    if len(_h2d_open) >= _H2D_STAGING_CAP or len(_h2d_closed) >= 4:
        torch.mps.synchronize()
        _h2d_open.clear()
        _h2d_closed.clear()
    _h2d_open.append(host)
    return True


def mark_h2d_generation() -> None:
    """Close the generation queued so far; call at the top of a scheduler iteration."""
    global _h2d_open
    if _h2d_open:
        _h2d_closed.append(_h2d_open)
        _h2d_open = []


def release_h2d_staging() -> None:
    """Drop a closed generation once a synchronized event proves the device passed it."""
    if len(_h2d_closed) >= 2:
        _h2d_closed.pop(0).clear()


def device_type() -> str:
    return "mps" if is_mps() else "cuda"


def Stream(device=None, **kwargs):  # noqa: N802 - mirrors torch.cuda.Stream
    return _NO_STREAM if is_mps() else torch.cuda.Stream(device=device, **kwargs)


def Event(*args, **kwargs):  # noqa: N802 - mirrors torch.cuda.Event
    return _MpsEvent(*args, **kwargs) if is_mps() else torch.cuda.Event(*args, **kwargs)


def current_stream(device=None):
    return _NO_STREAM if is_mps() else torch.cuda.current_stream(device)


def set_stream(stream) -> None:
    if not is_mps():
        torch.cuda.set_stream(stream)


def synchronize(device=None) -> None:
    if is_mps():
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize(device)


# In a list so atexit can drop it while torch is importable: Event.__del__ at teardown
# finds ``torch`` already None and prints a traceback.
_flush_event: list = []


def flush() -> None:
    """Hand the queued launches to the GPU without waiting."""
    if not is_mps():
        return
    if not _flush_event:
        _flush_event.append(torch.mps.Event())
        atexit.register(_flush_event.clear)
    _flush_event[0].record()


def empty_cache() -> None:
    (torch.mps if is_mps() else torch.cuda).empty_cache()


def reset_peak_memory_stats(device=None) -> None:
    if not is_mps():  # torch.mps has no peak-stat counter to reset
        torch.cuda.reset_peak_memory_stats(device)


# Host RAM left to macOS and every other process; Metal's recommended working set is 89% of
# the machine, so a ratio of it alone lands the pools in swap. A property of the machine.
_MPS_OS_RESERVE = int(float(os.environ.get("FREETOKEN_MPS_OS_RESERVE_GB", "8")) * 2**30)


def free_memory(device=None) -> int:
    """Bytes still available to this process."""
    if not is_mps():
        return torch.cuda.mem_get_info(device)[0]
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    working_set = max(min(torch.mps.recommended_max_memory(), physical - _MPS_OS_RESERVE), 0)
    return max(working_set - torch.mps.driver_allocated_memory(), 0)


@functools.cache
def max_single_alloc_bytes() -> int | None:
    """Largest single tensor the accelerator accepts (Metal's maxBufferLength, which torch
    exposes no binding for), or None when it caps only the total."""
    if not is_mps():
        return None

    def rejected(nbytes: int) -> bool:
        # Cheap: the size check runs before memory is committed, so an out-of-memory here
        # means the size was legal. Only "Invalid buffer size" is a rejection.
        try:
            probe = torch.empty(nbytes, dtype=torch.uint8, device="mps")
            del probe
        except RuntimeError as exc:
            return "Invalid buffer size" in str(exc)
        return False

    lo = 1 << 30  # 1 GiB: Metal never caps a buffer below this
    if rejected(lo):
        torch.mps.empty_cache()
        return lo
    hi = lo
    while not rejected(hi) and hi < (1 << 40):
        lo, hi = hi, hi * 2
    while hi - lo > (1 << 24):  # settle to 16 MiB
        mid = (lo + hi) // 2
        if rejected(mid):
            hi = mid
        else:
            lo = mid
    torch.mps.empty_cache()
    return lo


def nvtx_range(name: str):
    """``torch.cuda.nvtx.range``, which raises on a non-CUDA build."""
    return contextlib.nullcontext() if is_mps() else torch.cuda.nvtx.range(name)
