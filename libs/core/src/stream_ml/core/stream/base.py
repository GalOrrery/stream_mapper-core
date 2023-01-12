"""Base background model."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass

# LOCAL
from stream_ml.core.core import ModelBase
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.core.typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class StreamModel(ModelBase[Array]):
    """Stream Model."""

    # ========================================================================

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def pack_params_to_arr(self, mpars: Params[Array], /) -> Array:
        """Pack parameters into an array.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.

        Returns
        -------
        Array
        """
        raise NotImplementedError

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
        **kwargs: Array
            Additional arguments.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    @abstractmethod
    def _ln_prior_coord_bnds(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Elementwise log prior for coordinate bounds.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.

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
            Data.

        Returns
        -------
        Array
        """
        raise NotImplementedError
