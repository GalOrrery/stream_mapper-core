"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar

# LOCAL
from stream_ml.core._typing import Array, BoundsT
from stream_ml.core.base import Model
from stream_ml.core.params import MutableParams, ParamBounds, ParamNames, Params
from stream_ml.core.utils.hashdict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import DataT, FlatParsT
    from stream_ml.core.prior.base import PriorBase

__all__: list[str] = []


V = TypeVar("V")


@dataclass
class MixtureModelBase(Model[Array], Mapping[str, Model[Array]], metaclass=ABCMeta):
    """Full Model.

    Parameters
    ----------
    components : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModelBase`).

    tied_params : Mapping[str, Callable[[Params], Array]], optional keyword-only
        Mapping of parameter names to functions that take the parameters of the
        model and return the value of the tied parameter. This is useful for
        tying parameters across models, e.g. the background and stream models
        in a mixture model.

    priors : tuple[PriorBase, ...], optional keyword-only
        Mapping of parameter names to priors. This is useful for setting priors
        on parameters across models, e.g. the background and stream models in a
        mixture model.
    """

    components: FrozenDictField[str, Model[Array]] = FrozenDictField()

    _: KW_ONLY
    name: str | None = None  # the name of the model
    tied_params: FrozenDictField[
        str, Callable[[Params[Array]], Array]
    ] = FrozenDictField({})
    priors: tuple[PriorBase[Array], ...] = ()

    DEFAULT_BOUNDS: ClassVar = None

    def __post_init__(self) -> None:
        # Add the coord_names
        cns: list[str] = []
        for m in self.components.values():
            cns.extend(c for c in m.coord_names if c not in cns)
        self._coord_names: tuple[str, ...] = tuple(cns)

        # Add the param_names  # TODO: make sure no duplicates
        self._param_names: ParamNames = ParamNames(
            (f"{n}_{p[0]}", p[1]) if isinstance(p, tuple) else f"{n}_{p}"
            for n, m in self.components.items()
            for p in m.param_names
        )

        # Add the coord_bounds
        # TODO: make sure duplicates have the same bounds
        cbs: FrozenDict[str, BoundsT] = FrozenDict()
        for m in self.components.values():
            cbs._mapping.update(m.coord_bounds)
        self._coord_bounds = cbs

        # Add the param_bounds
        cps: ParamBounds[Array] = ParamBounds()
        for n, m in self.components.items():
            cps._mapping.update({f"{n}_{k}": v for k, v in m.param_bounds.items()})
        self._param_bounds = cps

        super().__post_init__()

    @property
    def coord_names(self) -> tuple[str, ...]:
        """Coordinate names."""
        return self._coord_names

    @coord_names.setter  # hack to match the Protocol
    def coord_names(self, value: Any) -> None:
        """Set the coordinate names."""
        msg = "cannot set coord_names"
        raise AttributeError(msg)

    @property  # type: ignore[override]
    def param_names(self) -> ParamNames:
        """Parameter names."""
        return self._param_names

    @param_names.setter  # hack to match the Protocol
    def param_names(self, value: Any) -> None:
        """Set the parameter names."""
        msg = "cannot set param_names"
        raise AttributeError(msg)

    @property  # type: ignore[override]
    def coord_bounds(self) -> FrozenDict[str, BoundsT]:
        """Coordinate names."""
        return self._coord_bounds

    @coord_bounds.setter  # hack to match the Protocol
    def coord_bounds(self, value: Any) -> None:
        """Set the coordinate bounds."""
        msg = "cannot set coord_bounds"
        raise AttributeError(msg)

    @property  # type: ignore[override]
    def param_bounds(self) -> ParamBounds[Array]:
        """Coordinate names."""
        return self._param_bounds

    @param_bounds.setter  # hack to match the Protocol
    def param_bounds(self, value: Any) -> None:
        """Set the parameter bounds."""
        msg = "cannot set param_bounds"
        raise AttributeError(msg)

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str) -> Model[Array]:
        c: Model[Array] = self.components[key]
        return c

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __hash__(self) -> int:
        return hash(tuple(self.keys()))

    # ===============================================================

    def unpack_params(self, packed_pars: FlatParsT[Array]) -> Params[Array]:
        """Unpack parameters into a dictionary.

        Unpack a flat dictionary of parameters -- where keys have coordinate name,
        parameter name, and model component name -- into a nested dictionary with
        parameters grouped by coordinate name.

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
        # FIXME! this doesn't work with the model components.
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

        return Params(pars)

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
        # Unpack the parameters
        pars = Params[Array]()
        j = 0
        for n, m in self.components.items():  # iter thru models
            # Get relevant parameters by index
            mp_arr = p_arr[:, slice(j, j + len(m.param_names.flat))]

            # TODO: what to do about tied parameters?

            # Skip empty (and incrementing the index)
            if mp_arr.shape[1] == 0:
                continue

            # Add the component's parameters, prefixed with the component name
            pars._mapping.update(
                m.unpack_params_from_arr(mp_arr).add_prefix(n + "_", inplace=True)
            )

            # Increment the index
            j += len(m.param_names.flat)

        # Add the dependent parameters
        for name, tie in self.tied_params.items():
            pars._mapping[name] = tie(pars)

        return pars

    @abstractmethod
    def pack_params_to_arr(self, pars: Params[Array]) -> Array:
        """Pack parameters into an array.

        Parameters
        ----------
        pars : Params
            Parameter dictionary.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    @staticmethod
    def _get_prefixed_kwargs(prefix: str, kwargs: dict[str, V]) -> dict[str, V]:
        """Get the kwargs with a given prefix.

        Parameters
        ----------
        prefix : str
            Prefix.
        kwargs : dict[str, V]
            Keyword arguments.

        Returns
        -------
        dict[str, V]
        """
        prefix = prefix + "_" if not prefix.endswith("_") else prefix
        lp = len(prefix)
        return {k[lp:]: v for k, v in kwargs.items() if k.startswith(prefix)}

    # ===============================================================
    # Statistics

    @abstractmethod
    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT[Array], **kwargs: Array
    ) -> Array:
        """Log likelihood.

        Just the log-sum-exp of the individual log-likelihoods.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        raise NotImplementedError

    @abstractmethod
    def ln_prior_arr(self, pars: Params[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params
            Parameters.

        Returns
        -------
        Array
        """
        raise NotImplementedError
