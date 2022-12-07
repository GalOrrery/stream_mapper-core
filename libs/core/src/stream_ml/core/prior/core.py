"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

# LOCAL
from stream_ml.core._typing import Array
from stream_ml.core.prior.base import PriorBase

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.params.core import Params

__all__: list[str] = []


@dataclass(frozen=True)
class Prior(PriorBase[Array]):
    """Prior."""

    logpdf_hook: Callable[[Params[Array], Array | None], Array]
    forward_hook: Callable[[Array], Array]

    def logpdf(self, p: Params[Array], current_pdf: Array | None = None) -> Array:
        """Evaluate the logpdf."""
        return self.logpdf_hook(p, current_pdf)

    def __call__(self, p: Array, param_names: tuple[str, ...]) -> Array:
        """Evaluate the forward step in the prior."""
        return self.forward_hook(p)
