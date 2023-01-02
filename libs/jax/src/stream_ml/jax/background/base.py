"""Base background model."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass

# LOCAL
from stream_ml.core.background.base import BackgroundModel as CoreBackgroundModel
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.jax._typing import Array
from stream_ml.jax.core import ModelBase

__all__: list[str] = []


@dataclass()
class BackgroundModel(ModelBase, CoreBackgroundModel[Array]):
    """Background Model."""

    # ========================================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the background.

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

    @abstractmethod
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

    @abstractmethod
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
