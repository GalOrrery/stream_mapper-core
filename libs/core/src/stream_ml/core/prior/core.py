"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.typing import Array

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.api import Model
    from stream_ml.core.params.core import Params


__all__: list[str] = []


@dataclass(frozen=True)
class Prior(PriorBase[Array]):
    """Prior."""

    logpdf_hook: Callable[
        [Params[Array], Data[Array], Model[Array], Array | None], Array
    ]
    forward_hook: Callable[[Array, Model[Array]], Array]

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
    ) -> Array:
        """Evaluate the logpdf.

        This log-pdf is added to the current logpdf. So if you want to set the
        logpdf to a specific value, you can uses the `current_lnpdf` to set the
        output value such that ``current_lnpdf + logpdf = <want>``.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], position-only
            The data for which evaluate the prior.
        model : Model[Array], position-only
            The model for which evaluate the prior.
        current_lnpdf : Array | None, optional position-only
            The current logpdf, by default `None`. This is useful for setting
            the additive log-pdf to a specific value.

        Returns
        -------
        Array
            The logpdf.
        """
        return self.logpdf_hook(mpars, data, model, current_lnpdf)

    def __call__(self, nn: Array, data: Data[Array], model: Model[Array], /) -> Array:
        """Evaluate the forward step in the prior.

        Parameters
        ----------
        nn : Array
            The input to evaluate the prior at.
        data : Data[Array]
            The data to evaluate the prior at.
        model : Model[Array]
            The model to evaluate the prior at.

        Returns
        -------
        Array
        """
        return self.forward_hook(nn, model)
