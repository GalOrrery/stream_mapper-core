"""Core feature."""

from __future__ import annotations

__all__ = ["NoBounds", "ClippedBounds"]

from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from stream_ml.core.params.bounds._base import PriorBounds
from stream_ml.core.typing import Array

if TYPE_CHECKING:
    from stream_ml.core._core.api import Model
    from stream_ml.core.data import Data
    from stream_ml.core.params._values import Params
    from stream_ml.core.typing import ArrayNamespace, NNModel

    Self = TypeVar("Self", bound="PriorBounds")  # type: ignore[type-arg]


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


@dataclass(frozen=True)
class ClippedBounds(PriorBounds[Array]):
    """Clipped bounds."""

    lower: Array | float
    upper: Array | float

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