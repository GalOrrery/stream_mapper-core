"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import jax.numpy as xp

# LOCAL
from stream_ml.core.params.names import FlatParamNames
from stream_ml.core.prior.bounds import PriorBounds as CorePriorBounds
from stream_ml.jax._typing import Array
from stream_ml.jax.utils.misc import within_bounds
from stream_ml.jax.utils.sigmoid import scaled_sigmoid

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
            raise ValueError("locator is None")

        bp = xp.zeros_like(lp[self.param_name])
        bp[~within_bounds(lp[self.param_name], self.lower, self.upper)] = -xp.inf
        return bp

    @abstractmethod
    def __call__(self, x: Array, param_names: FlatParamNames, /) -> Array:
        """Evaluate the forward step in the prior."""
        ...


@dataclass(frozen=True)
class SigmoidBounds(PriorBounds):
    """Base class for prior bounds."""

    def __call__(self, x: Array, param_names: FlatParamNames, /) -> Array:
        """Evaluate the forward step in the prior."""
        col = param_names.index(self.param_name)
        x = x.at[:, col].set(
            scaled_sigmoid(x[:, col], lower=self.lower, upper=self.upper)
        )
        return x
