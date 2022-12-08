"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Generic

# LOCAL
from stream_ml.core._typing import Array

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.params.core import Params

__all__: list[str] = []


@dataclass(frozen=True)
class PriorBase(Generic[Array], metaclass=ABCMeta):
    """Prior."""

    _: KW_ONLY
    inplace: bool = False  # whether to modify the params inplace
    name: str | None = None  # the name of the prior

    @abstractmethod
    def logpdf(self, lp: Params[Array], current_lnpdf: Array | None = None, /) -> Array:
        """Evaluate the logpdf."""
        ...

    @abstractmethod
    def __call__(
        self, x: Array, param_names: tuple[tuple[str] | tuple[str, str], ...], /
    ) -> Array:
        """Evaluate the forward step in the prior."""
        ...
