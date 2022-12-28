"""Base Stream Model class."""

from __future__ import annotations

# STDLIB
import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

# LOCAL
from stream_ml.core.params import Params
from stream_ml.core.stream.base import StreamModel as CoreStreamModel
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.core import ModelBase

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import DataT


__all__: list[str] = []


@dataclass(unsafe_hash=True)
class StreamModel(ModelBase, CoreStreamModel[Array]):
    """Stream Model."""

    # ========================================================================
    # Statistics

    @abc.abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT, **kwargs: Array
    ) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ln_prior_arr(self, pars: Params[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params
            Parameters.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    # ========================================================================
    # ML

    # TODO: keep moving up the hierarchy!
    def _forward_prior(self, out: Array) -> Array:
        """Forward pass.

        Parameters
        ----------
        out : Array
            Input.

        Returns
        -------
        Array
            Same as input.
        """
        for bnd in self.param_bounds.flatvalues():
            out = bnd(out, self)
        return out

    @abc.abstractmethod
    def forward(self, *args: Array) -> Array:
        """Forward pass.

        Parameters
        ----------
        args : Array
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        raise NotImplementedError
