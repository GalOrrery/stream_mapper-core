"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

# LOCAL
from stream_ml.core._typing import Array
from stream_ml.core.base import Model
from stream_ml.core.params import MutableParams, ParamBounds, ParamNamesField, Params
from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.utils.hashdict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import DataT, FlatParsT

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
    """

    n_features: int

    _: KW_ONLY
    name: str | None = None  # the name of the model

    coord_names: tuple[str, ...]
    param_names: ParamNamesField = ParamNamesField()

    # Bounds on the coordinates and parameters.
    # name: (lower, upper)
    coord_bounds: FrozenDictField[str, tuple[float, float]] = FrozenDictField(
        FrozenDict()
    )
    param_bounds: ParamBoundsField = ParamBoundsField(ParamBounds())

    def __post_init__(self) -> None:
        """Post-init validation."""
        super().__post_init__()

        # Shapes attribute
        self.shapes: dict[str, int | dict[str, int]]
        shapes: dict[str, int | dict[str, int]] = {}
        for pn in self.param_names:
            if isinstance(pn, str):  # e.g. "mixparam"
                shapes[pn] = self.n_features
            else:  # e.g. ("phi2", ("mu", "sigma"))
                shapes[pn[0]] = {p: self.n_features for p in pn[1]}
        object.__setattr__(self, "shapes", shapes)

        # Validate the param_names
        if not self.param_names:
            raise ValueError("param_names must be specified.")

        # Make coord bounds if not provided
        crnt_cbs = self.coord_bounds._mapping
        cbs = {n: crnt_cbs.pop(n, (-inf, inf)) for n in self.coord_names}
        if crnt_cbs:
            raise ValueError(f"coord_bounds contains invalid keys {crnt_cbs.keys()}.")
        self.coord_bounds = FrozenDict(cbs)

        # Make param bounds
        self.param_bounds = ParamBounds.from_names(self.param_names) | self.param_bounds

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

        for k in packed_pars.keys():
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
