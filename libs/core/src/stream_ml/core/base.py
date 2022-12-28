"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol

# LOCAL
from stream_ml.core._typing import Array
from stream_ml.core.params.bounds import ParamBounds, ParamBoundsField
from stream_ml.core.params.core import Params
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.utils.hashdict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import BoundsT, DataT, FlatParsT

__all__: list[str] = []


class Model(Protocol[Array]):
    """Model base class."""

    name: str | None

    # Name of the coordinates and parameters.
    coord_names: tuple[str, ...]
    param_names: ParamNamesField = ParamNamesField()

    # Bounds on the coordinates and parameters.
    coord_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](ParamBounds())

    DEFAULT_BOUNDS: ClassVar  # TODO: PriorBounds[Any]

    def __post_init__(self) -> None:
        # Make sure the pieces of param_bounds are hashable. Static type
        # checkers will complain about mutable input, but this allows for the
        # run-time behavior to work regardless.
        self.param_bounds._freeze()
        return None

    # ========================================================================

    @abstractmethod
    def unpack_params(self, packed_pars: FlatParsT[Array]) -> Params[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        packed_pars : Array
            Flat dictionary of parameters.

        Returns
        -------
        Params[Array]
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

    @abstractmethod
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
        raise NotImplementedError

    # ========================================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT[Array], **kwargs: Array
    ) -> Array:
        """Elementwise log-likelihood of the model.

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
    def ln_prior_arr(self, pars: Params[Array]) -> Array:
        """Elementwise log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    def ln_posterior_arr(
        self, pars: Params[Array], data: DataT[Array], **kwargs: Array
    ) -> Array:
        """Elementwise log posterior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : DataT
            Data.
        **kwargs : Array
            Arguments.

        Returns
        -------
        Array
        """
        # TODO! move to ModelBase
        # fmt: off
        post_arr: Array = (
            self.ln_likelihood_arr(pars, data, **kwargs)
            + self.ln_prior_arr(pars)
        )
        # fmt: on
        return post_arr

    # ------------------------------------------------------------------------

    def ln_likelihood(
        self, pars: Params[Array], data: DataT[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the model.

        This is evaluated over the entire data set.

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
        # TODO! move to ModelBase
        return self.ln_likelihood_arr(pars, data, **kwargs).sum()

    def ln_prior(self, pars: Params[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.

        Returns
        -------
        Array
        """
        # TODO! move to ModelBase
        return self.ln_prior_arr(pars).sum()

    def ln_posterior(
        self, pars: Params[Array], data: DataT[Array], **kwargs: Array
    ) -> Array:
        """Log posterior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : DataT
            Data.
        **kwargs : Array
            Keyword arguments. These are passed to the likelihood function.

        Returns
        -------
        Array
        """
        # TODO! move to ModelBase
        ln_post: Array = self.ln_likelihood(pars, data, **kwargs) + self.ln_prior(pars)
        return ln_post
