"""Base for multi-component models."""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from dataclasses import KW_ONLY, dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from stream_ml.core._api import Model
from stream_ml.core._base import NN_NAMESPACE
from stream_ml.core.params import ParamNames
from stream_ml.core.setup_package import CompiledShim
from stream_ml.core.typing import Array, ArrayNamespace, BoundsT, NNModel
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.core.prior import PriorBase

__all__: list[str] = []


def _get_namespace(
    components: FrozenDict[str, Model[Array, NNModel]]
) -> ArrayNamespace[Array]:
    """Get the array namespace."""
    ns = {v.array_namespace for v in components.values()}
    if len(ns) != 1:
        msg = "all components must use the same array namespace."
        raise ValueError(msg)
    return ns.pop()


class UnpackParamsCallable(Protocol):
    """Protocol for unpacking parameters."""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Callable."""
        ...


@dataclass
class ModelsBase(
    Model[Array, NNModel],
    Mapping[str, Model[Array, NNModel]],
    CompiledShim,
    metaclass=ABCMeta,
):
    """Multi-model base class."""

    components: FrozenDictField[str, Model[Array, NNModel]] = FrozenDictField()

    _: KW_ONLY
    name: str | None = None  # the name of the model
    priors: tuple[PriorBase[Array], ...] = ()
    unpack_params_hooks: tuple[UnpackParamsCallable, ...] = ()

    DEFAULT_PARAM_BOUNDS: ClassVar[Any] = None  # TODO: ClassVar[PriorBase[Array]]

    def __post_init__(self) -> None:
        """Post-init validation."""
        self._mypyc_init_descriptor()  # TODO: Remove this when mypyc is fixed.

        self.array_namespace = _get_namespace(self.components)
        self._nn_namespace_ = NN_NAMESPACE[self.array_namespace]

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
        cbs: dict[str, BoundsT] = {}
        for m in self.components.values():
            cbs.update(m.coord_bounds)
        self._coord_bounds = FrozenDict(cbs)

        # Add the param_names  # TODO: make sure no duplicates
        self._param_names: ParamNames = ParamNames(
            (f"{c}.{p[0]}", p[1]) if isinstance(p, tuple) else f"{c}.{p}"
            for c, m in self.components.items()
            for p in m.param_names
        )

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
    def coord_bounds(self) -> FrozenDict[str, BoundsT]:
        """Coordinate names."""
        return self._coord_bounds

    @coord_bounds.setter  # hack to match the Protocol
    def coord_bounds(self, value: Any) -> None:
        """Set the coordinate bounds."""
        msg = "cannot set coord_bounds"
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

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str) -> Model[Array, NNModel]:
        return self.components[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __hash__(self) -> int:
        return hash(tuple(self.keys()))

    def keys(self) -> KeysView[str]:
        """Return the components' keys."""
        return self.components.keys()

    def values(self) -> ValuesView[Model[Array, NNModel]]:
        """Return the components' values."""
        return self.components.values()

    def items(self) -> ItemsView[str, Model[Array, NNModel]]:
        """Return the components' items."""
        return self.components.items()

    # ===============================================================
    # Dataclass

    def __str__(self) -> str:
        """Return the string representation."""
        return (
            f"{type(self).__name__}(\n\t"
            + ",\n\t".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))
            + "\n)"
        )

    # ===============================================================

    def unpack_params(self, packed_pars: Mapping[str, Array], /) -> Params[Array]:
        """Unpack parameters into a dictionary.

        Unpack a flat dictionary of parameters -- where keys have coordinate
        name, parameter name, and model component name -- into a nested
        dictionary with parameters grouped by coordinate name.

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

    # ===============================================================
    # Statistics

    def ln_prior(
        self, mpars: Params[Array], data: Data[Array], current_lnp: Array | None = None
    ) -> Array:
        """Log prior.

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
        # Loop over the components
        lnp: Array = self.xp.zeros(()) if current_lnp is None else current_lnp
        for name, m in self.components.items():
            lnp = lnp + m.ln_prior(mpars.get_prefixed(name + "."), data)
        # Parameter Bounds
        for bounds in self.param_bounds.flatvalues():
            lnp = lnp + bounds.logpdf(mpars, data, self, lnp, xp=self.xp)
        # Plugin for priors
        for prior in self.priors:
            lnp = lnp + prior.logpdf(mpars, data, self, lnp, xp=self.xp)
        return lnp
