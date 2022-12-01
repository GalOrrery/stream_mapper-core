"""Core feature."""

from __future__ import annotations

# STDLIB
import abc
from typing import TYPE_CHECKING, ClassVar

# THIRD-PARTY
import torch as xp
import torch.nn as nn

# LOCAL
from stream_ml.core.base import ModelBase

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array, DataT, ParsT

__all__: list[str] = []


class Model(nn.Module, ModelBase[Array]):  # type: ignore[misc]
    """Model base class."""

    _param_names: ClassVar[dict[str, int]]

    @property
    def param_names(self) -> dict[str, int]:
        """Parameter names."""
        return self._param_names

    def unpack_pars(self, p_arr: Array) -> ParsT:
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
        p_dict = {}
        j: int = 0
        for i, name in enumerate(self.param_names):
            p_dict[name] = p_arr[:, j + i].view(-1, 1)
        return p_dict

    def pack_pars(self, p_dict: ParsT) -> Array:
        """Pack parameters into an array.

        Parameters
        ----------
        p_dict : ParsT
            Parameter dictionary.

        Returns
        -------
        Array
        """
        p_arrs = []
        for name in self.param_names:
            p_arrs.append(xp.atleast_1d(p_dict[name]))
        return xp.concatenate(p_arrs)

    # ========================================================================
    # Statistics

    @abc.abstractmethod
    def ln_likelihood(self, pars: ParsT, data: DataT) -> Array:
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

    def ln_posterior(self, pars: ParsT, data: DataT, *args: Array) -> Array:
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
        return self.ln_likelihood(pars, data, *args) + self.ln_prior(pars)

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
