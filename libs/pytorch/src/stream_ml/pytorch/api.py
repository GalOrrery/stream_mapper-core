"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from math import inf
from typing import Any, ClassVar, Protocol

# THIRD-PARTY
import torch as xp
from torch import nn

# LOCAL
from stream_ml.core.api import Model as CoreModel
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.pytorch.prior.bounds import PriorBounds, SigmoidBounds
from stream_ml.pytorch.typing import Array

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

    def __post_init__(self) -> None:
        nn.Module.__init__(self)  # Needed for PyTorch
        super().__post_init__()

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
        return xp.concatenate(
            [xp.atleast_1d(mpars[elt]) for elt in self.param_names.flats]
        )

    # ========================================================================
    # ML

    @abstractmethod
    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        """Pytoch call method."""
        ...
