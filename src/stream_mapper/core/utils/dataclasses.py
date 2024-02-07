"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import Field, fields
from typing import Any, ClassVar

from stream_mapper.core._api import SupportsXP
from stream_mapper.core.typing import Array


class ArrayNamespaceReprMixin(SupportsXP[Array]):
    """Mixin for array namespace repr."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __repr__(self) -> str:
        """Repr."""
        xpname = self.xp if isinstance(self.xp, str) else self.xp.__name__
        fs = (
            (
                f"{f.name}={getattr(self, f.name)!r}"
                if f.name != "array_namespace"
                else f"{f.name}={xpname!r}"
            )
            for f in fields(self)
        )
        return f"{self.__class__.__name__}({', '.join(fs)})"
