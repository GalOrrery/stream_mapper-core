"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params.bounds import ParamBounds, ParamBoundsField
from stream_ml.core.params.core import Params, freeze_params, set_param
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.typing import Array
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.typing import BoundsT, FlatParsT

__all__: list[str] = []


WEIGHT_NAME = "weight"


class Model(Protocol[Array]):
    """Model base class."""

    name: str | None

    # Name of the coordinates and parameters.
    coord_names: tuple[str, ...]
    param_names: ParamNamesField = ParamNamesField()

    # Bounds on the coordinates and parameters.
    coord_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](ParamBounds())

    # Priors on the parameters.
    priors: tuple[PriorBase[Array], ...] = ()

    DEFAULT_BOUNDS: ClassVar[Any]  # TODO: PriorBounds[Any]

    def __post_init__(self) -> None:
        # TODO: have ``xp`` be a property of the model.
        pass

    # ========================================================================

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.coord_names)

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
    def pack_params_to_arr(self, mpars: Params[Array], /) -> Array:
        """Pack model parameters into an array.

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

    # ------------------------------------------------------------------------
    # Elementwise versions

    @abstractmethod
    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Elementwise log-likelihood of the model.

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

    def ln_prior_arr(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Elementwise log prior.

        .. todo::

            Add a private argument ``lnp`` so that we can pass the current
            value of the log prior to the attached prior objects and
            accumulate the log prior using `super`.

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
        # Instead of raising an error, we return the sum of the priors held on
        # this model.
        #
        # TODO: this is a bit of a hack to start with 0. We should use a
        # ``get_namespace`` method to get ``xp.zeros``.
        lnp: Array = 0  # type: ignore[assignment]
        for prior in self.priors:
            lnp = lnp + prior.logpdf(mpars, data, self, lnp)

        return lnp

    def ln_posterior_arr(
        self, mpars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        """Elementwise log posterior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data
            Data.
        **kw : Array
            Arguments.

        Returns
        -------
        Array
        """
        return self.ln_likelihood_arr(mpars, data, **kw) + self.ln_prior_arr(
            mpars, data
        )

    # ------------------------------------------------------------------------
    # Scalar versions

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the model.

        This is evaluated over the entire data set.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        return self.ln_likelihood_arr(mpars, data, **kwargs).sum()

    def ln_prior(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data
            Data.

        Returns
        -------
        Array
        """
        return self.ln_prior_arr(mpars, data).sum()

    def ln_posterior(
        self, mpars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        """Log posterior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kw : Array
            Keyword arguments. These are passed to the likelihood function.

        Returns
        -------
        Array
        """
        return self.ln_likelihood(mpars, data, **kw) + self.ln_prior(mpars, data)

    # ------------------------------------------------------------------------
    # Misc

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call the model."""
        ...
