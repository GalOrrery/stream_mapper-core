"""Core feature."""

from __future__ import annotations

# STDLIB
from math import inf
from typing import Any, ClassVar, Protocol

# THIRD-PARTY
import jax.numpy as xp

# LOCAL
from stream_ml.core.api import Model as CoreModel
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
        a mixture model (see :class:`~stream_ml.core.core.MixtureModel`).
    """

    DEFAULT_BOUNDS: ClassVar[PriorBounds] = SigmoidBounds(-inf, inf)

    # ========================================================================

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
        # TODO: check that structure of pars matches self.param_names
        # ie, that if elt is a string, then pars[elt] is a 1D array
        # and if elt is a tuple, then pars[elt] is a dict.
        return xp.concatenate(
            [xp.atleast_1d(mpars[elt]) for elt in self.param_names.flats]
        )

    # ========================================================================
    # ML

    def __call__(self, *args: Array, **kwds: Any) -> Array:
        """Pytoch call method."""
        ...
