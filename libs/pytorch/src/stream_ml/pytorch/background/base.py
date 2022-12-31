"""Base background model."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

# LOCAL
from stream_ml.core.background.base import BackgroundModel as CoreBackgroundModel
from stream_ml.core.params import Params
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.core import ModelBase

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import DataT

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class BackgroundModel(ModelBase, CoreBackgroundModel[Array]):
    """Background Model."""

    # ========================================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT, **kwargs: Array
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : DataT
            Data (phi1).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    @abstractmethod
    def ln_prior_arr(self, pars: Params[Array], data: DataT) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : DataT
            Data (phi1).s

        Returns
        -------
        Array
        """
        raise NotImplementedError

    # ========================================================================
    # ML

    @abstractmethod
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
