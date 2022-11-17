"""Core feature."""

from __future__ import annotations

# STDLIB
import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # THIRD-PARTY
    from torch import Tensor

    # LOCAL
    from stream_ml._typing import DataT, ParsT

__all__: list[str] = []


class Model(metaclass=abc.ABCMeta):
    """Model base class."""

    @abc.abstractmethod
    def ln_likelihood(self, pars: ParsT, data: DataT) -> Tensor:
        """Log-likelihood of the model.

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

    # def ln_posterior(self, pars: ParsT, data: Tensor, *args: Tensor) -> Tensor:
    #     """Log posterior.

    #     Parameters
    #     ----------
    #     pars : ParsT
    #         Parameters.
    #     data : Tensor
    #         Data.
    #     args : Tensor
    #         Arguments.

    #     Returns
    #     -------
    #     Tensor
    #     """
    #     return self.ln_likelihood(pars, data, *args) + self.ln_prior(pars)

    def neg_ln_likelihood(self, pars: ParsT, data: DataT, scalar: bool = True) -> Tensor:
        """Negative log-likelihood.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : Tensor
            Data (phi1).
        scalar : bool, optional
            Sum over the batch dimension, by default `True`.

        Returns
        -------
        Tensor
        """
        if scalar:
            return -self.ln_likelihood(pars, data).sum()
        return -self.ln_likelihood(pars, data)
