"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING, ClassVar

# LOCAL
from stream_ml.core._typing import Array, BoundsT
from stream_ml.core.base import Model
from stream_ml.core.data import Data
from stream_ml.core.params import MutableParams, ParamBounds, ParamNamesField, Params
from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import FlatParamName
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.prior.bounds import NoBounds, PriorBounds
from stream_ml.core.utils.hashdict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import FlatParsT

__all__: list[str] = []

inf = float("inf")


@dataclass(unsafe_hash=True)
class ModelBase(Model[Array], metaclass=ABCMeta):
    """Single-model base class.

    Parameters
    ----------
    n_features : int
        The number off features used by the NN.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModelBase`).

    coord_names : tuple[str, ...], keyword-only
        The names of the coordinates, not including the 'independent' variable.
        E.g. for independent variable 'phi1' this might be ('phi2', 'prlx',
        ...).
    param_names : `~stream_ml.core.params.ParamNames`, keyword-only
        The names of the parameters. Parameters dependent on the coordinates are
        grouped by the coordinate name.
        E.g. ('weight', ('phi1', ('mu', 'sigma'))).

    coord_bounds : Mapping[str, tuple[float, float]], keyword-only
        The bounds on the coordinates. If not provided, the bounds are
        (-inf, inf) for all coordinates.

    param_bounds : `~stream_ml.core.params.ParamBounds`, keyword-only
        The bounds on the parameters.
    """

    n_features: int

    _: KW_ONLY
    name: str | None = None  # the name of the model

    coord_names: tuple[str, ...]
    param_names: ParamNamesField = ParamNamesField()

    # Bounds on the coordinates and parameters.
    coord_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](ParamBounds())

    priors: tuple[PriorBase[Array], ...] = ()

    DEFAULT_BOUNDS: ClassVar  # TODO: [PriorBounds[Any]]

    def __post_init__(self) -> None:
        """Post-init validation."""
        super().__post_init__()

        # Validate the param_names
        if not self.param_names:
            msg = "param_names must be specified"
            raise ValueError(msg)

        # Make coord bounds if not provided
        crnt_cbs = self.coord_bounds._mapping
        cbs = {n: crnt_cbs.pop(n, (-inf, inf)) for n in self.coord_names}
        if crnt_cbs:  # Error if there are extra keys
            msg = f"coord_bounds contains invalid keys {crnt_cbs.keys()}."
            raise ValueError(msg)
        self.coord_bounds = FrozenDict(cbs)

        # Make parameter bounds
        # 1) Make the default bounds for all parameters.
        # 2) Update from the user-specified bounds.
        # 3) Fix up the names so each bound references its parameter.
        param_bounds: ParamBounds[Array] = (
            ParamBounds.from_names(self.param_names, default=self.DEFAULT_BOUNDS)
            | self.param_bounds
        )
        param_bounds._fixup_param_names()  # TODO: better method name
        self.param_bounds = param_bounds

    # ========================================================================

    def unpack_params(self, packed_pars: FlatParsT[Array]) -> Params[Array]:
        """Unpack parameters into a dictionary.

        Unpack a flat dictionary of parameters -- where keys have coordinate name,
        parameter name, and model component name -- into a nested dictionary with
        parameters grouped by coordinate name.

        Parameters
        ----------
        packed_pars : Array
            Parameter array.

        Returns
        -------
        Params[Array]
        """
        pars = MutableParams[Array]()

        for k in packed_pars:
            # Find the non-coordinate-specific parameters.
            if k in self.param_bounds:
                pars[k] = packed_pars[k]
                continue

            # separate the coordinate and parameter names.
            coord_name, par_name = k.split("_", maxsplit=1)
            # Add the parameter to the coordinate-specific dict.
            pars[coord_name, par_name] = packed_pars[k]

        return pars

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
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Elementwise log-likelihood of the model.

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
        raise NotImplementedError

    @abstractmethod
    def _ln_prior_coord_bnds(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Elementwise log prior for coordinate bounds.

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

    # ========================================================================
    # Misc

    @classmethod
    def _make_bounds(
        cls, bounds: PriorBounds[Array] | BoundsT | None, param_name: FlatParamName
    ) -> PriorBounds[Array]:
        """Make bounds."""
        if isinstance(bounds, PriorBounds):
            return bounds
        elif bounds is None:
            return NoBounds()
        else:
            return replace(
                cls.DEFAULT_BOUNDS,
                lower=bounds[0],
                upper=bounds[1],
                param_name=param_name,
            )
