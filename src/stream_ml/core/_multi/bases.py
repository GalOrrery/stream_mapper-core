"""Base for multi-component models."""

from __future__ import annotations

__all__: list[str] = []

from abc import ABCMeta, abstractmethod
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from dataclasses import KW_ONLY, dataclass, fields
from typing import TYPE_CHECKING, Any, Protocol

from stream_ml.core._connect.nn_namespace import NN_NAMESPACE
from stream_ml.core._core.model_api import Model
from stream_ml.core.setup_package import CompiledShim
from stream_ml.core.typing import (
    Array,
    ArrayNamespace,
    BoundsT,
    NNModel,
    ParamsLikeDict,
)
from stream_ml.core.utils.cached_property import cached_property
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    from stream_ml.core._data import Data
    from stream_ml.core.params import ModelParameters, Params
    from stream_ml.core.prior import PriorBase


def _get_array_namespace(
    components: FrozenDict[str, Model[Array, NNModel]]
) -> ArrayNamespace[Array]:
    """Get the array namespace."""
    ns = {v.array_namespace for v in components.values()}
    if len(ns) != 1:
        msg = "all components must use the same array namespace."
        raise ValueError(msg)
    return ns.pop()


class UnpackParamsCallable(Protocol[Array]):
    """Protocol for unpacking parameters."""

    def __call__(self, *args: Any, **kwds: Any) -> ParamsLikeDict[Array]:
        """Callable."""
        ...


class SupportsComponentGetItem(Protocol[Array, NNModel]):
    def __getitem__(self, key: str) -> Model[Array, NNModel]:
        ...


# ========================================================================


@dataclass
class ModelsBase(
    Model[Array, NNModel],
    Mapping[str, Model[Array, NNModel]],
    SupportsComponentGetItem[Array, NNModel],
    CompiledShim,
    metaclass=ABCMeta,
):
    """Multi-model base class.

    Parameters
    ----------
    components : Mapping[str, Model], optional
        A mapping of the components of the model. The keys are the names of the
        component models, and the values are the models themselves. The names do
        not have to match the names on the model.

    indep_coord_names : tuple[str, ...], optional keyword-only
        The names of the independent coordinates, e.g. "phi1". Default is
        ("phi1",).

    priors: tuple[PriorBase[Array], ...], optional keyword-only
        Priors on the parameters. Default is empty.

    data_scaler : DataScaler[Array], keyword-only
        The data scaler.

    unpack_params_hooks : tuple[UnpackParamsCallable[Array], ...], optional keyword-only
        A tuple of callables that unpack the parameters. Default is empty.
        This is useful for unpacking parameters that are not part of a model,
        e.g. allowing one model to pass parameters to another model.

    array_namespace : ArrayNamespace[Array], keyword-only
        The array namespace.
    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModel`).

    Notes
    -----
    The following fields on :class:`~stream_ml.core.ModelAPI` are properties here:

    - :attr:`~stream_ml.core.ModelBase.coord_names`
    - :attr:`~stream_ml.core.ModelBase.coord_err_names`
    - :attr:`~stream_ml.core.ModelBase.coord_bounds`
    """

    components: FrozenDictField[str, Model[Array, NNModel]] = FrozenDictField()

    _: KW_ONLY
    name: str | None = None  # the name of the model
    priors: tuple[PriorBase[Array], ...] = ()
    unpack_params_hooks: tuple[UnpackParamsCallable[Array], ...] = ()

    def __post_init__(self) -> None:
        """Post-init validation."""
        self._mypyc_init_descriptor()  # TODO: Remove this when mypyc is fixed.

        self.array_namespace = _get_array_namespace(self.components)
        self._nn_namespace_ = NN_NAMESPACE[self.array_namespace]

        # Check that there is at least one component
        if not self.components:
            msg = "must have at least one component."
            raise ValueError(msg)

        super().__post_init__()

    @cached_property
    def coord_names(self) -> tuple[str, ...]:  # type: ignore[override]
        """Coordinate names, unsorted."""
        return tuple({n for m in self.components.values() for n in m.coord_names})

    @cached_property
    def coord_err_names(self) -> tuple[str, ...] | None:  # type: ignore[override]
        """Coordinate error names, unsorted."""
        return tuple(
            {n for m in self.components.values() for n in (m.coord_err_names or ())}
        )

    @cached_property  # type: ignore[override]
    def coord_bounds(self) -> FrozenDict[str, BoundsT]:
        """Coordinate names."""
        # Add the coord_bounds
        # TODO: make sure duplicates have the same bounds
        cbs: dict[str, BoundsT] = {
            k: v for m in self.components.values() for k, v in m.coord_bounds.items()
        }
        return FrozenDict(cbs)

    @property
    @abstractmethod
    def composite_params(self) -> ModelParameters[Array]:
        """Composite parameters."""
        return self.params

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
    # Statistics

    def ln_prior(
        self, mpars: Params[Array], data: Data[Array], current_lnp: Array | None = None
    ) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array[(N,)]], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array[(N, F)]]
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
            lnp = lnp + m.ln_prior(mpars.get_prefixed(name), data)
        # Parameter Bounds
        for param in self.params.flatvalues():
            lnp = lnp + param.bounds.logpdf(mpars, data, self, lnp)
        # Plugin for priors
        for prior in self.priors:
            lnp = lnp + prior.logpdf(mpars, data, self, lnp)
        return lnp
