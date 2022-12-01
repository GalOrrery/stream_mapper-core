"""Core feature."""

from __future__ import annotations

# STDLIB
import abc
from typing import TYPE_CHECKING, Generic

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import ArrayT, DataT, ParsT

__all__: list[str] = []


class ModelBase(Generic["ArrayT"], metaclass=abc.ABCMeta):
    """Model base class."""

    @property
    @abc.abstractmethod
    def param_names(self) -> dict[str, int]:
        """Parameter names."""
        raise NotImplementedError

    @abc.abstractmethod
    def unpack_pars(self, p_arr: ArrayT) -> ParsT[ArrayT]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        p_arr : Array
            Parameter array.

        Returns
        -------
        ParsT
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pack_pars(self, p_dict: ParsT[ArrayT]) -> ArrayT:
        """Pack parameters into an array.

        Parameters
        ----------
        p_dict : ParsT
            Parameter dictionary.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    # ========================================================================
    # Statistics

    @abc.abstractmethod
    def ln_likelihood(self, pars: ParsT[ArrayT], data: DataT[ArrayT]) -> ArrayT:
        """Log-likelihood of the model.

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
    def ln_prior(self, pars: ParsT[ArrayT]) -> ArrayT:
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

    def ln_posterior(
        self, pars: ParsT[ArrayT], data: DataT[ArrayT], *args: ArrayT
    ) -> ArrayT:
        """Log posterior.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : DataT
            Data.
        args : Array
            Arguments.

        Returns
        -------
        Array
        """
        ln_post: ArrayT = self.ln_likelihood(pars, data, *args) + self.ln_prior(pars)
        return ln_post
