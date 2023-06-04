"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from stream_ml.core._api import (
    HasName,
    LnProbabilities,
    Probabilities,
    SupportsXP,
    SupportsXPNN,
    TotalLnProbabilities,
    TotalProbabilities,
)
from stream_ml.core.params._field import ModelParametersField
from stream_ml.core.params._values import Params, freeze_params, set_param
from stream_ml.core.setup_package import PACK_PARAM_JOIN
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.prior._base import PriorBase
    from stream_ml.core.typing import BoundsT, ParamNameAllOpts, ParamsLikeDict


class Model(
    TotalProbabilities[Array],
    Probabilities[Array],
    TotalLnProbabilities[Array],
    LnProbabilities[Array],
    SupportsXPNN[Array, NNModel],
    SupportsXP[Array],
    HasName,
    Protocol[Array, NNModel],
):
    """Model base class."""

    # Coordinates of the model.
    coord_names: tuple[str, ...]
    coord_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())

    # Parameters of the model.
    params: ModelParametersField[Array] = ModelParametersField[Array]()

    # Priors on the parameters.
    priors: tuple[PriorBase[Array], ...] = ()

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    # ========================================================================

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.coord_names)

    @overload
    def _unpack_params_from_map(
        self,
        packed: Mapping[str, Array],
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[False],
    ) -> ParamsLikeDict[Array]:
        ...

    @overload
    def _unpack_params_from_map(
        self,
        packed: Mapping[str, Array],
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[True],
    ) -> Params[Array]:
        ...

    @overload
    def _unpack_params_from_map(
        self,
        packed: Mapping[str, Array],
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        ...

    def _unpack_params_from_map(
        self,
        packed: Mapping[str, Array],
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        packed : Mapping[str, Array], positional-only
            Flat dictionary of parameters.
        extras : dict[str, Array] | None, optional
            Additional parameters to add.
        freeze : bool, optional keyword-only
            Whether to freeze the parameters. Default is `True`.

        Returns
        -------
        Params[Array]
            Nested dictionary of parameters wth parameters grouped by coordinate
            name.
        """
        pars: ParamsLikeDict[Array] = {}
        for k in packed:
            # Find the non-coordinate-specific parameters.
            if k in self.params:
                pars[k] = packed[k]
                continue

            # separate the coordinate and parameter names.
            coord_name, par_name = k.split(PACK_PARAM_JOIN, maxsplit=1)
            # Add the parameter to the coordinate-specific dict.
            set_param(pars, (coord_name, par_name), packed[k])

        for ke, v in (extras or {}).items():  # update from extras
            set_param(pars, ke, v)

        return freeze_params(pars) if freeze else pars

    @overload
    @abstractmethod
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[False],
    ) -> ParamsLikeDict[Array]:
        ...

    @overload
    @abstractmethod
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[True],
    ) -> Params[Array]:
        ...

    @overload
    @abstractmethod
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        ...

    @abstractmethod
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        arr : Array, positional-only
            Parameter array.
        extras : dict[str, Array] | None, optional
            Additional parameters to add.
        freeze : bool, optional keyword-only
            Whether to freeze the parameters. Default is `True`.

        Returns
        -------
        Params[Array]
        """
        raise NotImplementedError

    @overload
    def unpack_params(
        self,
        inp: Array | Mapping[str, Array],
        /,
        extras: dict[ParamNameAllOpts, Array] | None = None,
        *,
        freeze: Literal[False],
    ) -> ParamsLikeDict[Array]:
        ...

    @overload
    def unpack_params(
        self,
        inp: Array | Mapping[str, Array],
        /,
        extras: dict[ParamNameAllOpts, Array] | None = None,
        *,
        freeze: Literal[True],
    ) -> Params[Array]:
        ...

    @overload
    def unpack_params(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None = None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        ...

    def unpack_params(
        self,
        inp: Array | Mapping[str, Array],
        /,
        extras: dict[ParamNameAllOpts, Array] | None = None,
        *,
        freeze: bool = True,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter dict or array and unpacks it into a
        dictionary with the parameter names as keys.

        Parameters
        ----------
        inp : Array or Mapping, positional-only
            Parameter array or dictionary.
        extras : dict[str, Array] | None, optional
            Additional parameters to add.
        freeze : bool, optional keyword-only
            Whether to freeze the parameters. Default is `True`.

        Returns
        -------
        Params[Array]
        """
        if isinstance(inp, Mapping):
            return self._unpack_params_from_map(inp, extras=extras, freeze=freeze)
        return self._unpack_params_from_arr(inp, extras=extras, freeze=freeze)

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
            tuple(self.xp.atleast_1d(mpars[elt]) for elt in self.params.flatskeys())
        )

    # ========================================================================
    # Statistics

    def ln_likelihood_tot(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        return self.xp.sum(self.ln_likelihood(mpars, data, **kwargs))

    def ln_prior_tot(self, mpars: Params[Array], data: Data[Array]) -> Array:
        return self.xp.sum(self.ln_prior(mpars, data))

    def ln_posterior_tot(
        self, mpars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        return self.xp.sum(self.ln_posterior(mpars, data, **kw))

    # ------------------------------------------------------------------------
    # Non-logarithmic elementwise versions

    def likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        return self.xp.exp(self.ln_likelihood(mpars, data, **kwargs))

    def prior(
        self, mpars: Params[Array], data: Data[Array], current_lnp: Array | None = None
    ) -> Array:
        return self.xp.exp(self.ln_prior(mpars, data, current_lnp))

    def posterior(self, mpars: Params[Array], data: Data[Array], **kw: Array) -> Array:
        return self.xp.exp(self.ln_posterior(mpars, data, **kw))

    # ------------------------------------------------------------------------
    # Non-logarithmic scalar versions

    def likelihood_tot(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        return self.xp.exp(self.ln_likelihood_tot(mpars, data, **kwargs))

    def prior_tot(self, mpars: Params[Array], data: Data[Array]) -> Array:
        return self.xp.exp(self.ln_prior_tot(mpars, data))

    def posterior_tot(
        self, mpars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        return self.xp.exp(self.ln_posterior_tot(mpars, data, **kw))

    # ========================================================================
    # ML

    def __call__(self, *args: Any, **kwds: Any) -> Array:
        """Call the model."""
        ...
