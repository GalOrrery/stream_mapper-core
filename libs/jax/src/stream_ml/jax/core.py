"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass

# THIRD-PARTY
import flax.linen as nn

# LOCAL
from stream_ml.core.core import ModelBase as CoreModelBase
from stream_ml.core.data import Data
from stream_ml.core.params import MutableParams, Params
from stream_ml.flax.base import Model
from stream_ml.jax._typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class ModelBase(nn.Module, CoreModelBase[Array], Model):  # type: ignore[misc]
    """Model base class."""

    @abstractmethod
    def setup(self) -> None:
        """Setup."""

    def unpack_pars_to_arr(self, p_arr: Array) -> Params[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        p_arr : Array
            Parameter array.

        Returns
        -------
        Params[Array]
        """
        pars = MutableParams[Array]()
        for i, k in enumerate(self.param_names.flats):
            pars[k] = p_arr[:, i]
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
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the model.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data (phi1).
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
            Data (phi1).

        Returns
        -------
        Array
        """
        raise NotImplementedError

    # ========================================================================
    # ML

    @abstractmethod
    def __call__(self, *args: Array) -> Array:
        """Forward pass.

        Parameters
        ----------
        *args : Array
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        raise NotImplementedError
