"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from math import inf
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

# THIRD-PARTY
import torch as xp
from torch import nn

# LOCAL
from stream_ml.core.base import Model as CoreModel
from stream_ml.core.params import Params
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.prior.bounds import PriorBounds, SigmoidBounds

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import DataT, FlatParsT

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

    def __post_init__(self) -> None:
        nn.Module.__init__(self)  # Needed for PyTorch
        super().__post_init__()

    # ========================================================================

    @abstractmethod
    def unpack_params(self, packed_pars: FlatParsT) -> Params[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        packed_pars : Array
            Flat dictionary of parameters.

        Returns
        -------
        Params
            Nested dictionary of parameters wth parameters grouped by coordinate
            name.
        """
        raise NotImplementedError

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
        pars : Params[Array]
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
        self, pars: Params[Array], data: DataT, **kwargs: Array
    ) -> Array:
        """Elementwise log-likelihood of the model.

        Parameters
        ----------
        pars : Params
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
        """Elementwise log prior.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data (phi1).

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

    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        """Pytoch call method."""
        ...
