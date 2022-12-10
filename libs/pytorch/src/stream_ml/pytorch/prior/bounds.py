"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.params.names import FlatParamNames
from stream_ml.core.prior.bounds import PriorBounds as CorePriorBounds
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.utils.misc import within_bounds
from stream_ml.pytorch.utils.sigmoid import scaled_sigmoid

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.params.core import Params

__all__: list[str] = []


@dataclass(frozen=True)
class PriorBounds(CorePriorBounds[Array]):
    """Base class for prior bounds."""

    def logpdf(self, lp: Params[Array], current_lnpdf: Array | None = None, /) -> Array:
        """Evaluate the logpdf."""
        if self.param_name is None:
            raise ValueError("need to set param_name")

        bp = xp.zeros_like(lp[self.param_name])
        bp[~within_bounds(lp[self.param_name], self.lower, self.upper)] = -xp.inf
        return bp

    @abstractmethod
    def __call__(self, x: Array, param_names: FlatParamNames, /) -> Array:
        """Evaluate the forward step in the prior."""
        ...


@dataclass(frozen=True)
class NoBounds(PriorBounds):
    """No bounds."""

    lower: float = -float("inf")
    upper: float = float("inf")

    def __post_init__(self, /) -> None:
        """Post-init."""
        if self.lower != -float("inf") or self.upper != float("inf"):
            raise ValueError("lower and upper must be -inf and inf")

    def logpdf(self, lp: Params[Array], current_lnpdf: Array | None = None, /) -> Array:
        """Evaluate the logpdf."""
        if self.param_name is None:
            raise ValueError("need to set param_name")
        return xp.zeros_like(lp[self.param_name])

    def __call__(self, x: Array, param_names: FlatParamNames, /) -> Array:
        """Evaluate the forward step in the prior."""
        return x


@dataclass(frozen=True)
class SigmoidBounds(PriorBounds):
    """Base class for prior bounds."""

    def __call__(self, x: Array, param_names: FlatParamNames, /) -> Array:
        """Evaluate the forward step in the prior."""
        # if not self.inplace:
        x = x.clone()

        col = param_names.index(self.param_name)
        x[:, col] = scaled_sigmoid(
            x[:, col], lower=xp.asarray(self.lower), upper=xp.asarray(self.upper)
        )
        return x
