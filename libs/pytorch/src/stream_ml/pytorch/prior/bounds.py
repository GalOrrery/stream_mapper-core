"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.prior.bounds import NoBounds  # noqa: F401
from stream_ml.core.prior.bounds import PriorBounds as CorePriorBounds
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.utils.misc import within_bounds
from stream_ml.pytorch.utils.sigmoid import scaled_sigmoid

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
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
    ) -> Array | float:
        """Evaluate the logpdf."""
        if self.param_name is None:
            msg = "need to set param_name"
            raise ValueError(msg)

        bp = xp.zeros_like(pars[self.param_name])
        bp[~within_bounds(pars[self.param_name], self.lower, self.upper)] = -xp.inf
        return bp

    @abstractmethod
    def __call__(self, x: Array, model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior."""
        ...


@dataclass(frozen=True)
class SigmoidBounds(PriorBounds):
    """Base class for prior bounds."""

    def __call__(self, x: Array, model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior."""
        # if not self.inplace:
        x = x.clone()

        col = model.param_names.flats.index(self.param_name)
        x[:, col] = scaled_sigmoid(
            x[:, col], lower=xp.asarray([self.lower]), upper=xp.asarray([self.upper])
        )
        return x
