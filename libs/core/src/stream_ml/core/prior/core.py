"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

# LOCAL
from stream_ml.core._typing import Array
from stream_ml.core.params.names import FlatParamNames
from stream_ml.core.prior.base import PriorBase

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.params.core import Params

__all__: list[str] = []


@dataclass(frozen=True)
class Prior(PriorBase[Array]):
    """Prior."""

    logpdf_hook: Callable[[Params[Array], Array | None], Array]
    forward_hook: Callable[[Array, FlatParamNames], Array]

    def logpdf(self, lp: Params[Array], current_lnpdf: Array | None = None, /) -> Array:
        """Evaluate the logpdf."""
        return self.logpdf_hook(lp, current_lnpdf)

    def __call__(self, p: Array, param_names: FlatParamNames, /) -> Array:
        """Evaluate the forward step in the prior."""
        return self.forward_hook(p, param_names)
