"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from stream_mapper.core.prior._base import Prior
from stream_mapper.core.typing import Array

if TYPE_CHECKING:
    from stream_mapper.core import Data, ModelAPI as Model, Params
    from stream_mapper.core.typing import NNModel


class LogPDFHook(Protocol[Array]):
    """LogPDF hook."""

    def __call__(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array, NNModel],
        current_lnpdf: Array | None,
        /,
    ) -> Array:
        """Evaluate the logpdf."""


class ForwardHook(Protocol[Array]):
    """Forward hook."""

    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior."""


@dataclass(frozen=True, repr=False)
class FunctionPrior(Prior[Array]):
    """Prior with custom function hooks."""

    logpdf_hook: LogPDFHook[Array]
    forward_hook: ForwardHook[Array]

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array, NNModel],
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
        model : Model[Array, NNModel], position-only
            The model for which evaluate the prior.
        current_lnpdf : Array | None, optional position-only
            The current logpdf, by default `None`. This is useful for setting
            the additive log-pdf to a specific value.

        xp : ArrayNamespace[Array], keyword-only
            The array namespace.

        Returns
        -------
        Array
            The logpdf.
        """
        return self.logpdf_hook(mpars, data, model, current_lnpdf)

    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior.

        Parameters
        ----------
        pred : Array
            The input to evaluate the prior at.
        data : Data[Array]
            The data to evaluate the prior at.
        model : Model[Array, NNModel]
            The model to evaluate the prior at.

        xp : ArrayNamespace[Array], keyword-only
            The array namespace.

        Returns
        -------
        Array
        """
        return self.forward_hook(pred, data, model)
