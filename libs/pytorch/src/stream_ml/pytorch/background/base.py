"""Base background model."""

from __future__ import annotations

# STDLIB
import abc
from typing import TYPE_CHECKING

# LOCAL
from stream_ml.pytorch.base import Model

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array, DataT, ParsT

__all__: list[str] = []


class BackgroundModel(Model):
    """Background Model."""

    # ========================================================================
    # Statistics

    @abc.abstractmethod
    def ln_likelihood(self, pars: ParsT, data: DataT) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : DataT
            Data (phi1).

        Returns
        -------
        Array
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ln_prior(self, pars: ParsT) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    # ========================================================================
    # ML

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
