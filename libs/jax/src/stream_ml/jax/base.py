"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from math import inf
from typing import Any, ClassVar, Protocol

# THIRD-PARTY
import jax.numpy as xp

# LOCAL
from stream_ml.core.base import Model as CoreModel
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.jax.prior.bounds import PriorBounds, SigmoidBounds
from stream_ml.jax.typing import Array

__all__: list[str] = []


class Model(CoreModel[Array], Protocol):
    """Pytorch model base class.

    Parameters
    ----------
    n_features : int
        The number off features used by the NN.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModelBase`).
    """

    DEFAULT_BOUNDS: ClassVar[PriorBounds] = SigmoidBounds(-inf, inf)

    @abstractmethod
    def setup(self) -> None:
        """Setup."""

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
        Params[Array]
        """
        raise NotImplementedError

    def pack_params_to_arr(self, pars: Params[Array]) -> Array:
        """Pack parameters into an array.

        Parameters
        ----------
        pars : Params[Arr]
            Parameter dictionary.

        Returns
        -------
        Array
        """
        # TODO: check that structure of pars matches self.param_names
        # ie, that if elt is a string, then pars[elt] is a 1D array
        # and if elt is a tuple, then pars[elt] is a dict.
        return xp.concatenate(
            [xp.atleast_1d(pars[elt]) for elt in self.param_names.flats]
        )

    # ========================================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Elementwise log-likelihood of the model.

        Parameters
        ----------
        pars : Params[Array]]
            Parameters.
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
    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Elementwise log prior.

        Parameters
        ----------
        pars : Params[Array]]
            Parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    # ========================================================================
    # ML

    def __call__(self, *args: Array, **kwds: Any) -> Array:
        """Pytoch call method."""
        ...
