"""Base for multi-component models."""

from __future__ import annotations

# STDLIB
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

# LOCAL
from stream_ml.core.api import Model
from stream_ml.core.params import ParamBounds, ParamNames, Params
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.typing import Array, BoundsT
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.data import Data
    from stream_ml.core.typing import FlatParsT

    V = TypeVar("V")

__all__: list[str] = []


@dataclass
class ModelsBase(Model[Array], Mapping[str, Model[Array]], metaclass=ABCMeta):
    """Multi-model base class."""

    components: FrozenDictField[str, Model[Array]] = FrozenDictField()

    _: KW_ONLY
    name: str | None = None  # the name of the model
    priors: tuple[PriorBase[Array], ...] = ()

    DEFAULT_BOUNDS: ClassVar = None

    def __post_init__(self) -> None:
        """Post-init validation."""
        # Check that there is at least one component
        if not self.components:
            msg = "must have at least one component."
            raise ValueError(msg)

        # Add the coord_names
        cns: list[str] = []
        for m in self.components.values():
            cns.extend(c for c in m.coord_names if c not in cns)
        self._coord_names: tuple[str, ...] = tuple(cns)

        # Add the coord_bounds
        # TODO: make sure duplicates have the same bounds
        cbs: FrozenDict[str, BoundsT] = FrozenDict()
        for m in self.components.values():
            cbs._dict.update(m.coord_bounds)
        self._coord_bounds = cbs

        # Hint the param_names
        self._param_names: ParamNames

        # Add the param_bounds
        cps: ParamBounds[Array] = ParamBounds()
        for n, m in self.components.items():
            cps._dict.update({f"{n}_{k}": v for k, v in m.param_bounds.items()})
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
        return self.components[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __hash__(self) -> int:
        return hash(tuple(self.keys()))

    # ===============================================================

    def unpack_params(self, packed_pars: FlatParsT[Array], /) -> Params[Array]:
        """Unpack parameters into a dictionary.

        Unpack a flat dictionary of parameters -- where keys have coordinate name,
        parameter name, and model component name -- into a nested dictionary with
        parameters grouped by coordinate name.

        Parameters
        ----------
        packed_pars : Array, positional-only
            Flat dictionary of parameters.

        Returns
        -------
        Params
            Nested dictionary of parameters wth parameters grouped by coordinate
            name.
        """
        return super().unpack_params(packed_pars)

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
        """Pack parameters into an array.

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
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log likelihood."""
        raise NotImplementedError

    def ln_prior_arr(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

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
        # Loop over the components
        # TODO: this is a bit of a hack to start with 0. We should use a
        #       ``get_namespace`` method to get ``xp.zeros``.
        lnp: Array = 0  # type: ignore[assignment]
        for name, m in self.components.items():
            # Get the relevant parameters
            mpars_ = mpars.get_prefixed(name + ".")

            # Compute the log-prior
            mlnp = m.ln_prior_arr(mpars_, data)

            # Add to the total
            lnp = lnp + mlnp

        # Plugin for priors
        for prior in self.priors:
            lnp = lnp + prior.logpdf(mpars, data, self, lnp)

        return lnp
