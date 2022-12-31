"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
from torch import nn

# LOCAL
from stream_ml.core.core import ModelBase as CoreModelBase
from stream_ml.core.params import MutableParams, Params
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.base import Model

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import DataT

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class ModelBase(nn.Module, CoreModelBase[Array], Model):  # type: ignore[misc]
    """Model base class."""

    # ========================================================================

    def unpack_params_from_arr(self, p_arr: Array) -> Params[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        p_arr : Array
            Parameter array.

        Returns
        -------
        Params
        """
        pars = MutableParams[Array]()
        for i, k in enumerate(self.param_names.flats):
            pars[k] = p_arr[:, i].view(-1, 1)
        return Params(pars)

    def pack_params_to_arr(self, pars: Params[Array]) -> Array:
        """Pack parameters into an array.

        Parameters
        ----------
        pars : Params[Array]
            Parameter dictionary.

        Returns
        -------
        Array
        """
        return Model.pack_params_to_arr(self, pars)

    # ========================================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT, **kwargs: Array
    ) -> Array:
        """Log-likelihood of the model.

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
            Data.

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
