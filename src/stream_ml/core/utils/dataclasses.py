"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import Field, fields
from typing import Any, ClassVar

from stream_ml.core._api import SupportsXP
from stream_ml.core.typing import Array


class ArrayNamespaceReprMixin(SupportsXP[Array]):
    """Mixin for array namespace repr."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __repr__(self) -> str:
        """Repr."""
        fs = (
            f"{f.name}={getattr(self, f.name)!r}"
            if f.name != "array_namespace"
            else f"{f.name}={self.array_namespace.__name__!r}"
            for f in fields(self)
        )
        return f"{self.__class__.__name__}({', '.join(fs)})"
