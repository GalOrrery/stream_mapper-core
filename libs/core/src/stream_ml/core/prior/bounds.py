"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, TypeVar

# LOCAL
from stream_ml.core._typing import Array, BoundsT
from stream_ml.core.params.names import FlatParamName, FlatParamNames
from stream_ml.core.prior.base import PriorBase

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.params.core import Params

__all__: list[str] = []


Self = TypeVar("Self", bound="PriorBounds")  # type: ignore[type-arg]


@dataclass(frozen=True)
class PriorBounds(PriorBase[Array]):
    """Base class for prior bounds."""

    lower: float
    upper: float
    _: KW_ONLY
    param_name: FlatParamName | None = None

    @abstractmethod
    def logpdf(self, lp: Params[Array], current_lnpdf: Array | None = None, /) -> Array:
        """Evaluate the logpdf."""
        ...

    @abstractmethod
    def __call__(self, x: Array, param_names: FlatParamNames, /) -> Array:
        """Evaluate the forward step in the prior."""
        ...

    # =========================================================================

    @classmethod
    def from_tuple(
        cls: type[Self],
        t: BoundsT,
        /,
        param_name: FlatParamName | None = None,
    ) -> Self:
        """Create from tuple."""
        return cls(*t, param_name=param_name)

    def as_tuple(self) -> BoundsT:
        """Get as tuple."""
        return self.lower, self.upper

    @property
    def bounds(self) -> BoundsT:
        """Get the bounds."""
        return self.as_tuple()

    # =========================================================================

    def __iter__(self) -> Iterator[float]:
        """Iterate over the bounds."""
        yield from self.bounds
