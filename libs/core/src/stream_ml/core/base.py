"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

# LOCAL
from stream_ml.core._typing import Array
from stream_ml.core.utils.hashdict import HashableMap
from stream_ml.core.utils.params import ParamBoundsField, ParamNamesField, Params

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import DataT, FlatParsT

__all__: list[str] = []


class Model(Protocol[Array]):
    """Model base class."""

    name: str | None

    # Name of the coordinates and parameters.
    coord_names: tuple[str, ...]
    param_names: ParamNamesField = ParamNamesField()

    # Bounds on the coordinates and parameters.
    # name: (lower, upper)
    coord_bounds: HashableMap[str, tuple[float, float]]
    param_bounds: ParamBoundsField = ParamBoundsField()
    # defaults to (-inf, inf)

    def __post_init__(self) -> None:

        # Make sure the pieces of param_bounds are hashable.
        # Static type checkers will complain about mutable input, but this allows
        # for the run-time behavior to work regardless.
        for name, bounds in self.param_bounds.items():
            if isinstance(bounds, Mapping) and not isinstance(bounds, HashableMap):
                self.param_bounds._mapping[name] = HashableMap(bounds)  # type: ignore[unreachable] # noqa: E501

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
        ParsT
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
        self, pars: Params[Array], data: DataT[Array], *args: Array
    ) -> Array:
        """Elementwise log-likelihood of the model.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : DataT
            Data (phi1).
        *args : Array
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
        self, pars: Params[Array], data: DataT[Array], *args: Array
    ) -> Array:
        """Elementwise log posterior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : DataT
            Data.
        args : Array
            Arguments.

        Returns
        -------
        Array
        """
        # fmt: off
        post_arr: Array = (
            self.ln_likelihood_arr(pars, data, *args)
            + self.ln_prior_arr(pars)
        )
        # fmt: on
        return post_arr

    # ------------------------------------------------------------------------
    # Statistics: summmed

    def ln_likelihood(
        self, pars: Params[Array], data: DataT[Array], *args: Array
    ) -> Array:
        """Log-likelihood of the model.

        This is evaluated over the entire data set.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : DataT
            Data (phi1).
        *args : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        return self.ln_likelihood_arr(pars, data, *args).sum()

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
        return self.ln_prior_arr(pars).sum()  # FIXME?

    def ln_posterior(
        self, pars: Params[Array], data: DataT[Array], *args: Array
    ) -> Array:
        """Log posterior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : DataT
            Data.
        args : Array
            Arguments.

        Returns
        -------
        Array
        """
        ln_post: Array = self.ln_likelihood(pars, data, *args) + self.ln_prior(pars)
        return ln_post
