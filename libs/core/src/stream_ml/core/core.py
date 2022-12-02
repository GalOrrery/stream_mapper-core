"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

# LOCAL
from stream_ml.core._typing import Array
from stream_ml.core.base import Model
from stream_ml.core.utils.hashdict import HashableMapField
from stream_ml.core.utils.params import (
    MutableParams,
    ParamBounds,
    ParamBoundsField,
    ParamNamesField,
    Params,
)

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import DataT, FlatParsT

__all__: list[str] = []


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
    coord_bounds: HashableMapField[str, tuple[float, float]] = HashableMapField()  # type: ignore[assignment] # noqa: E501
    param_bounds: ParamBoundsField = ParamBoundsField()

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
        for c in self.coord_names:
            if c not in self.coord_bounds:
                raise ValueError(f"coord_bounds must be provided for {c}.")

        # TODO: fill in -inf, inf bounds for missing parameters
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
            # mixparam is a special case.
            # TODO: make this more general by using the params_bounds dict
            if k == "mixparam":
                pars["mixparam"] = packed_pars["mixparam"]
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
