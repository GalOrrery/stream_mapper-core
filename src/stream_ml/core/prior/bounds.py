"""Core feature."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass
from math import inf
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from stream_ml.core.params.names._core import FlatParamName  # noqa: TCH001
from stream_ml.core.prior._base import PriorBase
from stream_ml.core.typing import Array, ArrayNamespace
from stream_ml.core.utils.compat import array_at
from stream_ml.core.utils.funcs import within_bounds

__all__ = ["PriorBounds", "ClippedBounds", "NoBounds"]


if TYPE_CHECKING:
    from collections.abc import Iterator

    from stream_ml.core._api import Model
    from stream_ml.core.data import Data
    from stream_ml.core.params._core import Params
    from stream_ml.core.params.scales import ParamScaler
    from stream_ml.core.typing import NNModel

    Self = TypeVar("Self", bound="PriorBounds")  # type: ignore[type-arg]


@dataclass(frozen=True)
class PriorBounds(PriorBase[Array]):
    """Base class for prior bounds."""

    lower: Array | float
    upper: Array | float
    _: KW_ONLY
    param_name: FlatParamName | None = None
    scaler: ParamScaler[Array] | None = None

    def __post_init__(self) -> None:
        """Post-init."""
        self._scaled_bounds: tuple[Array, Array]
        if self.scaler is not None:
            object.__setattr__(
                self,
                "_scaled_bounds",
                (
                    self.scaler.transform(self.lower),
                    self.scaler.transform(self.upper),
                ),
            )

    # =========================================================================

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array, NNModel],
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
    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior."""
        ...

    # =========================================================================

    @property
    def bounds(self) -> tuple[Array | float, Array | float]:
        """Get the bounds."""
        return (self.lower, self.upper)

    @property
    def scaled_bounds(self) -> tuple[Array | float, Array | float]:
        """Get the scaled bounds."""
        if not hasattr(self, "_scaled_bounds"):
            msg = "need to pass scaler to prior bounds"
            raise ValueError(msg)
        return self._scaled_bounds

    # =========================================================================

    def __iter__(self) -> Iterator[Array | float]:
        """Iterate over the bounds."""
        yield from self.bounds


################################################################################


@dataclass(frozen=True)
class ClippedBounds(PriorBounds[Any]):
    """Clipped bounds."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Post-init."""
        if self.lower >= self.upper:
            msg = "lower must be less than upper"
            raise ValueError(msg)

        super().__post_init__()

    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior."""
        return model.xp.clip(pred, *self.scaled_bounds)


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

        super().__post_init__()

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array, NNModel],
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

    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior."""
        return pred
