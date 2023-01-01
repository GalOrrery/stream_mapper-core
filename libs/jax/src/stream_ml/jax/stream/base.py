"""Base Stream Model class."""

from __future__ import annotations

# STDLIB
import abc
from dataclasses import dataclass

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.core.stream.base import StreamModel as CoreStreamModel
from stream_ml.jax._typing import Array
from stream_ml.jax.core import ModelBase

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class StreamModel(ModelBase, CoreStreamModel[Array]):
    """Stream Model."""

    # ========================================================================
    # Statistics

    @abc.abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    # ========================================================================
    # ML

    @abc.abstractmethod
    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        raise NotImplementedError
