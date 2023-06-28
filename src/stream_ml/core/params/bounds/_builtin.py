"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING, TypeVar

from stream_ml.core.params.bounds._base import ParameterBounds
from stream_ml.core.typing import Array
from stream_ml.core.utils.compat import array_at, copy

if TYPE_CHECKING:
    from stream_ml.core._core.api import Model
    from stream_ml.core.data import Data
    from stream_ml.core.params._values import Params
    from stream_ml.core.params.scaler import ParamScaler
    from stream_ml.core.typing import NNModel

    Self = TypeVar("Self", bound="ParameterBounds")  # type: ignore[type-arg]


@dataclass(frozen=True)
class NoBounds(ParameterBounds[Array]):
    """No bounds."""

    lower: float = -inf
    upper: float = inf

    def __post_init__(self, scaler: ParamScaler[Array] | None) -> None:
        """Post-init."""
        if self.lower != -inf or self.upper != inf:
            msg = "lower and upper must be -inf and inf"
            raise ValueError(msg)

        super().__post_init__(scaler)

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array, NNModel],
        current_lnpdf: Array | None = None,
        /,
    ) -> Array:
        """Evaluate the logpdf."""
        if self.param_name is None:
            msg = "need to set param_name"
            raise ValueError(msg)
        return self.xp.asarray(0)

    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior."""
        return pred


@dataclass(frozen=True)
class ClippedBounds(ParameterBounds[Array]):
    """Clipped bounds."""

    lower: Array | float
    upper: Array | float

    def __post_init__(self, scaler: ParamScaler[Array] | None) -> None:
        """Post-init."""
        if self.lower >= self.upper:
            msg = "lower must be less than upper"
            raise ValueError(msg)
        super().__post_init__(scaler)

    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior."""
        pred = copy(pred)
        col = model.params.flatskeys().index(self.param_name)
        return array_at(pred, (..., col)).set(
            model.xp.clip(pred[:, col], *self.scaled_bounds)
        )
