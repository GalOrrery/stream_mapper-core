"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params.bounds import ParamBounds, ParamBoundsField
from stream_ml.core.params.core import Params, freeze_params, set_param
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.typing import Array
from stream_ml.core.utils.frozendict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.typing import BoundsT, FlatParsT

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
        return

    # ========================================================================

    def unpack_params(self, packed_pars: FlatParsT[Array], /) -> Params[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        packed_pars : Array, positional-only
            Flat dictionary of parameters.

        Returns
        -------
        Params[Array]
            Nested dictionary of parameters wth parameters grouped by coordinate
            name.
        """
        pars: dict[str, Array | dict[str, Array]] = {}

        for k in packed_pars:
            # Find the non-coordinate-specific parameters.
            if k in self.param_bounds:
                pars[k] = packed_pars[k]
                continue

            # separate the coordinate and parameter names.
            coord_name, par_name = k.split("_", maxsplit=1)
            # Add the parameter to the coordinate-specific dict.
            set_param(pars, (coord_name, par_name), packed_pars[k])

        return freeze_params(pars)

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

    # ------------------------------------------------------------------------
    # Elementwise versions

    @abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Elementwise log-likelihood of the model.

        Parameters
        ----------
        pars : Params[Array]
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
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    def ln_posterior_arr(
        self, pars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        """Elementwise log posterior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data
            Data.
        **kw : Array
            Arguments.

        Returns
        -------
        Array
        """
        return self.ln_likelihood_arr(pars, data, **kw) + self.ln_prior_arr(pars, data)

    # ------------------------------------------------------------------------
    # Scalar versions

    def ln_likelihood(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the model.

        This is evaluated over the entire data set.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        return self.ln_likelihood_arr(pars, data, **kwargs).sum()

    def ln_prior(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data
            Data.

        Returns
        -------
        Array
        """
        return self.ln_prior_arr(pars, data).sum()

    def ln_posterior(
        self, pars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        """Log posterior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data.
        **kw : Array
            Keyword arguments. These are passed to the likelihood function.

        Returns
        -------
        Array
        """
        return self.ln_likelihood(pars, data, **kw) + self.ln_prior(pars, data)
