"""Base Stream Model class."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.core.stream.base import StreamModel as CoreStreamModel
from stream_ml.jax.core import ModelBase
from stream_ml.jax.typing import Array

__all__: list[str] = []


@dataclass()
class StreamModel(ModelBase, CoreStreamModel[Array]):
    """Stream Model."""

    # ========================================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the stream.

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
            Data.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    # ========================================================================
    # ML

    # TODO: keep moving up the hierarchy!
    def _forward_prior(self, out: Array, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        out : Array
            Input.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
            Same as input.
        """
        for bnd in self.param_bounds.flatvalues():
            out = bnd(out, data, self)
        return out

    @abstractmethod
    def __call__(self, *args: Array, **kwargs: Any) -> Array:
        """Forward pass.

        Parameters
        ----------
        *args : Array
            Input.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        raise NotImplementedError
