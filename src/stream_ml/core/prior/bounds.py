"""Core feature."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass
from math import inf
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from stream_ml.core.data import Data
from stream_ml.core.params.names import FlatParamName
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.typing import Array, ArrayNamespace, BoundsT
from stream_ml.core.utils.compat import array_at
from stream_ml.core.utils.funcs import within_bounds

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.api import Model
    from stream_ml.core.params.core import Params

    Self = TypeVar("Self", bound="PriorBounds")  # type: ignore[type-arg]


@dataclass(frozen=True)
class PriorBounds(PriorBase[Array]):
    """Base class for prior bounds."""

    lower: float
    upper: float
    _: KW_ONLY
    param_name: FlatParamName | None = None

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
        *,
        xp: ArrayNamespace[Array],
    ) -> Array:
        """Evaluate the logpdf."""
        if self.param_name is None:
            msg = "need to set param_name"
            raise ValueError(msg)

        bp = xp.zeros_like(mpars[self.param_name])
        return array_at(
            bp, ~within_bounds(mpars[self.param_name], self.lower, self.upper)
        ).set(-xp.inf)

    @abstractmethod
    def __call__(self, pred: Array, data: Data[Array], model: Model[Array], /) -> Array:
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


################################################################################


@dataclass(frozen=True)
class NoBounds(PriorBounds[Any]):
    """No bounds."""

    lower: float = -inf
    upper: float = inf

    def __post_init__(self) -> None:
        """Post-init."""
        if self.lower != -inf or self.upper != inf:
            msg = "lower and upper must be -inf and inf"
            raise ValueError(msg)

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
        *,
        xp: ArrayNamespace[Array],
    ) -> Array | Literal[0]:
        """Evaluate the logpdf."""
        if self.param_name is None:
            msg = "need to set param_name"
            raise ValueError(msg)
        return 0

    def __call__(self, pred: Array, data: Data[Array], model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior."""
        return pred
