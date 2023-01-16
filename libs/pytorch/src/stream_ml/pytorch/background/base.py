"""Base background model."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass

# LOCAL
from stream_ml.core.background.base import BackgroundModel as CoreBackgroundModel
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class BackgroundModel(ModelBase, CoreBackgroundModel[Array]):
    """Background Model."""

    # ========================================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
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
    def ln_prior_arr(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.s

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
