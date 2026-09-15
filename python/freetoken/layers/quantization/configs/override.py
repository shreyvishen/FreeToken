"""Operator overrides layered on top of a checkpoint's QuantConfig."""

from __future__ import annotations

from typing import Any

from ..scheme import QuantScheme, fp8_tensor_scheme
from .base import QuantConfig


class DenseFp8OverrideConfig(QuantConfig):
    """Serve the dense projections named in modules as per-row fp8 (W8A16) when the
    checkpoint left them bf16. Lossy, so the checkpoint's own format always wins."""

    dialect = "dense-fp8-override"

    def __init__(self, inner: QuantConfig, modules: tuple[str, ...]):
        super().__init__(inner.name_map, ())
        self._inner = inner
        self._modules = modules

    @property
    def STORAGE(self):  # the wrapper stores nothing of its own; the checkpoint's dialect does
        return self._inner.STORAGE

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return False

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        return self._inner.scheme_for_name(name)

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        scheme = self._inner.scheme_for(prefix)
        if scheme is None and prefix.endswith(self._modules):
            return fp8_tensor_scheme("weight_scale", per_row=True)
        return scheme


def dense_fp8_override(quant: QuantConfig | None, model_config: Any) -> QuantConfig | None:
    """Wrap quant for --dense-quant-override fp8; a no-op when the family named no modules."""
    modules = tuple(getattr(model_config, "dense_fp8_modules", ()) or ())
    if not modules or quant is None:
        return quant
    return DenseFp8OverrideConfig(quant, modules)


__all__ = ["DenseFp8OverrideConfig", "dense_fp8_override"]
