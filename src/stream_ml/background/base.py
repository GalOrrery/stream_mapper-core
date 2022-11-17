"""Base background model."""

from __future__ import annotations

# STDLIB
import abc
from typing import TYPE_CHECKING

# LOCAL
from stream_ml.base import Model

if TYPE_CHECKING:
    # THIRD-PARTY
    from torch import Tensor

    # LOCAL
    from stream_ml._typing import DataT, ParsT

__all__: list[str] = []


class BackgroundModel(Model):
    """Stream Model."""

    @abc.abstractmethod
    def ln_likelihood(self, pars: ParsT, data: DataT) -> Tensor:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : Tensor
            Data (phi1).

        Returns
        -------
        Tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ln_prior(self, pars: ParsT) -> Tensor:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Tensor
        """
        raise NotImplementedError
