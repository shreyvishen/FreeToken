"""The decode plan for MPS: one decode forward recorded as a flat tape of launches, replayed
with only the static buffers refreshed -- the Metal stand-in for a CUDA graph
(engine/graph.py owns the static buffers, the capture hooks and the batch-size ladder)."""

from __future__ import annotations

import functools

import torch
from torch.overrides import TorchFunctionMode

from freetoken.kernel import backend as device_backend
from freetoken.kernel.metal import shaders


class TapeUnsupported(RuntimeError):
    """The eager forward did something a tape cannot replay."""


def _tensors(x) -> list[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        found = []
        for e in x:
            found.extend(_tensors(e))
        return found
    return []


def _storage(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


# Ops whose result is a plain copy of args[0] (a cast, a contiguous copy, a reshape that had
# to copy, a clone): ``out.copy_(args[0])`` is the kernel they ran.
_COPIES = frozenset({
    "to", "float", "half", "bfloat16", "double", "int", "long", "contiguous", "reshape",
    "clone", "type", "flatten",
})
# Device data read on the host: nothing to replay.
_HOST_READS = frozenset({
    "item", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__", "cpu",
})
_INPLACE_DUNDERS = frozenset({"__iadd__", "__isub__", "__imul__", "__itruediv__", "__setitem__"})


def _copy(args, kwargs, out):
    src = args[0]
    # A reshape that copied changes the shape; the bytes are src's in order, so copy into
    # the output seen in src's shape.
    dst = out if out.shape == src.shape else out.view(src.shape)
    return functools.partial(dst.copy_, src)


def _linear(args, kwargs, out):
    x, w = args[0], args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    if bias is not None or x.dim() != 2:
        raise TapeUnsupported("linear with a bias or a non-2D input")
    return functools.partial(torch.mm, x, w.t(), out=out)


def _binary(fn):
    def form(args, kwargs, out):
        if kwargs:
            raise TapeUnsupported(f"{fn.__name__} with keyword arguments")
        return functools.partial(fn, args[0], args[1], out=out)

    return form


_OUT_FORMS = {
    "linear": _linear,
    "add": _binary(torch.add), "__add__": _binary(torch.add), "__radd__": _binary(torch.add),
    "mul": _binary(torch.mul), "__mul__": _binary(torch.mul), "__rmul__": _binary(torch.mul),
    "sub": _binary(torch.sub), "__sub__": _binary(torch.sub),
    "index_select": lambda args, kwargs, out: functools.partial(
        torch.index_select, args[0], args[1], args[2], out=out),
    "silu": lambda args, kwargs, out: functools.partial(torch.ops.aten.silu.out, args[0], out=out),
}
_OUT_FORMS.update({name: _copy for name in _COPIES})


class _Recorder(TorchFunctionMode):
    def __init__(self, tape: "DecodeTape"):
        super().__init__()
        self._tape = tape

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if shaders.recording():  # not inside a host step, which the tape keeps whole
            self._tape._note(func, args, kwargs, out)
        return out


class DecodeTape:
    """A recorded decode forward. ``replay()`` re-issues every launch on the same buffers."""

    def __init__(self) -> None:
        self._ops: list = []

    def __len__(self) -> int:
        return len(self._ops)

    @classmethod
    def record(cls, fn) -> "DecodeTape":
        tape = cls()
        real_flush = device_backend.flush

        def flush() -> None:
            if shaders.recording():
                tape._ops.append(real_flush)
            real_flush()

        device_backend.flush = flush
        shaders.set_recorder(tape._launch)
        try:
            with torch.inference_mode(), _Recorder(tape):
                fn()
        finally:
            shaders.set_recorder(None)
            device_backend.flush = real_flush
        return tape

    def replay(self) -> None:
        for op in self._ops:
            op()

    def _launch(self, fn, args, kwargs) -> None:
        self._ops.append(functools.partial(fn, *args, **kwargs))

    def _note(self, func, args, kwargs, out) -> None:
        name = getattr(func, "__name__", "")
        if name in _INPLACE_DUNDERS or "out" in kwargs:
            self._ops.append(functools.partial(func, *args, **kwargs))
            return
        outs = _tensors(out)
        if not outs:
            if name in _HOST_READS:
                raise TapeUnsupported(f"{name} reads device data on the host")
            return
        for o in outs:
            if o.device.type != "mps":
                raise TapeUnsupported(f"{name} produced a {o.device.type} tensor")
        if name in ("empty", "empty_like", "empty_strided"):
            return  # the buffer persists on the tape; nothing to redo
        if name in ("zeros", "zeros_like"):
            self._ops.append(functools.partial(outs[0].zero_))
            return
        if name in ("ones", "ones_like"):
            self._ops.append(functools.partial(outs[0].fill_, 1))
            return
        if name.endswith("_") and not name.endswith("__"):
            self._ops.append(functools.partial(func, *args, **kwargs))
            return
        seen = {_storage(t) for t in _tensors(args) + _tensors(list(kwargs.values()))}
        if all(_storage(o) in seen for o in outs):
            return  # a view
        form = _OUT_FORMS.get(name)
        if form is None:
            raise TapeUnsupported(f"{name} allocates and has no replay form")
        self._ops.append(form(args, kwargs, outs[0]))


__all__ = ["DecodeTape", "TapeUnsupported"]
