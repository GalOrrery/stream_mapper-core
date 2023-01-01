"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import jax.numpy as xp

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.prior.bounds import PriorBounds as CorePriorBounds
from stream_ml.jax._typing import Array
from stream_ml.jax.utils.misc import within_bounds
from stream_ml.jax.utils.sigmoid import scaled_sigmoid

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.base import Model
    from stream_ml.core.params.core import Params

__all__: list[str] = []


@dataclass(frozen=True)
class PriorBounds(CorePriorBounds[Array]):
    """Base class for prior bounds."""

    def logpdf(
        self,
        pars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
    ) -> Array:
        """Evaluate the logpdf."""
        if self.param_name is None:
            msg = "locator is None"
            raise ValueError(msg)

        bp = xp.zeros_like(pars[self.param_name])
        bp[~within_bounds(pars[self.param_name], self.lower, self.upper)] = -xp.inf
        return bp

    @abstractmethod
    def __call__(self, nn: Array, data: Array, model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior."""
        ...


@dataclass(frozen=True)
class SigmoidBounds(PriorBounds):
    """Base class for prior bounds."""

    def __call__(self, nn: Array, data: Array, model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior."""
        col = model.param_names.flats.index(self.param_name)
        return nn.at[:, col].set(
            scaled_sigmoid(nn[:, col], lower=self.lower, upper=self.upper)
        )
