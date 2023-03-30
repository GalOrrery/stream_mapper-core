"""Core feature."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from stream_ml.core.params.bounds import ParamBounds, ParamBoundsField
from stream_ml.core.params.core import Params, freeze_params, set_param
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.typing import Array, ArrayNamespace, NNModel
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stream_ml.core.data import Data
    from stream_ml.core.prior.base import PriorBase
    from stream_ml.core.typing import BoundsT, NNNamespace

__all__: list[str] = []


class SupportsXP(Protocol[Array]):
    """Protocol for objects that support array namespaces."""

    array_namespace: ArrayNamespace[Array]

    @property
    def xp(self) -> ArrayNamespace[Array]:
        """Array namespace."""
        return self.array_namespace


class SupportsXPNN(SupportsXP[Array], Protocol[Array, NNModel]):
    """Protocol for objects that support array and NN namespaces."""

    _nn_namespace_: NNNamespace[NNModel, Array]

    @property
    def xpnn(self) -> NNNamespace[NNModel, Array]:
        """NN namespace."""
        return self._nn_namespace_


class Model(SupportsXPNN[Array, NNModel], Protocol[Array, NNModel]):
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

    DEFAULT_BOUNDS: ClassVar  # TODO: PriorBounds[Any]

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    # ========================================================================

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.coord_names)

    def unpack_params(self, packed_pars: Mapping[str, Array], /) -> Params[Array]:
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
        return self.xp.concatenate(
            tuple(self.xp.atleast_1d(mpars[elt]) for elt in self.param_names.flats)
        )

    # ========================================================================
    # Statistics

    # ------------------------------------------------------------------------
    # Elementwise versions

    @abstractmethod
    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Any
    ) -> Array:
        """Elementwise log-likelihood of the model.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    def ln_prior(
        self, mpars: Params[Array], data: Data[Array], current_lnp: Array | None = None
    ) -> Array:
        """Elementwise log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        current_lnp : Array | None, optional
            Current value of the log prior, by default `None`.

        Returns
        -------
        Array
        """
        ...

    def ln_posterior(
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
        return self.ln_likelihood(mpars, data, **kw) + self.ln_prior(mpars, data)

    # ------------------------------------------------------------------------
    # Scalar versions

    def ln_likelihood_tot(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Total log-likelihood of the model.

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
        return self.ln_likelihood(mpars, data, **kwargs).sum()

    def ln_prior_tot(self, mpars: Params[Array], data: Data[Array]) -> Array:
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
        return self.ln_prior(mpars, data).sum()

    def ln_posterior_tot(
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
        return self.ln_likelihood_tot(mpars, data, **kw) + self.ln_prior_tot(
            mpars, data
        )

    # ------------------------------------------------------------------------
    # Non-logarithmic elementwise versions

    def likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Elementwise likelihood of the model.

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
        return self.xp.exp(self.ln_likelihood(mpars, data, **kwargs))

    def prior(
        self, mpars: Params[Array], data: Data[Array], current_lnp: Array | None = None
    ) -> Array:
        """Elementwise prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        current_lnp : Array | None, optional
            Current value of the log prior, by default `None`.

        Returns
        -------
        Array
        """
        return self.xp.exp(self.ln_prior(mpars, data, current_lnp))

    def posterior(self, mpars: Params[Array], data: Data[Array], **kw: Array) -> Array:
        """Elementwise posterior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kw : Array
            Arguments.

        Returns
        -------
        Array
        """
        return self.xp.exp(self.ln_posterior(mpars, data, **kw))

    # ------------------------------------------------------------------------
    # Non-logarithmic scalar versions

    def likelihood_tot(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Total likelihood of the model.

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
        return self.xp.exp(self.ln_likelihood_tot(mpars, data, **kwargs))

    def prior_tot(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Total prior.

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
        return self.xp.exp(self.ln_prior_tot(mpars, data))

    def posterior_tot(
        self, mpars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        """Total posterior.

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
        return self.xp.exp(self.ln_posterior_tot(mpars, data, **kw))

    # ========================================================================
    # ML

    def __call__(self, *args: Any, **kwds: Any) -> Array:
        """Call the model."""
        ...
