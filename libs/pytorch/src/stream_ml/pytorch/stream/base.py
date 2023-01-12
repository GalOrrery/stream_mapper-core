"""Base Stream Model class."""

from __future__ import annotations

# STDLIB
import abc
from dataclasses import KW_ONLY, dataclass

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.core.stream.base import StreamModel as CoreStreamModel
from stream_ml.pytorch.core import ModelBase
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class StreamModel(ModelBase, CoreStreamModel[Array]):
    """Stream Model."""

    _: KW_ONLY
    indep_coord_name: str = "phi1"  # TODO: move up class hierarchy

    # ========================================================================
    # Statistics

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        raise NotImplementedError
