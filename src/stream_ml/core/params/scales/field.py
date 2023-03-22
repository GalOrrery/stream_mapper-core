"""Core feature."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, Protocol

from stream_ml.core.params.scales.core import (
    IncompleteParamScalerMapping,
    ParamScalerMapping,
    is_completable,
)
from stream_ml.core.typing import Array
from stream_ml.core.utils.sentinel import MISSING, Sentinel

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import EllipsisType

    from stream_ml.core.params.names import ParamNamesField
    from stream_ml.core.params.scales.builtin import ParamScaler

__all__: list[str] = []


class SupportsCoordandParamNames(Protocol):
    """Protocol for coordinate names."""

    coord_names: tuple[str, ...]
    param_names: ParamNamesField


class ParamScalerField(Generic[Array]):
    """Dataclass descriptor for parameter bounds.

    Parameters
    ----------
    default : ParamScalerMapping or Mapping or None, optional
        The default parameter bounds, by default `None`. If `None`, there are no
        default bounds and the parameter bounds must be specified in the Model
        constructor. If not a `ParamScalerMapping` instance, it will be converted to
        one.

    Notes
    -----
    See https://docs.python.org/3/library/dataclasses.html for more information
    on descriptor-typed fields for dataclasses.
    """

    def __init__(
        self,
        default: ParamScalerMapping[Array]
        | Mapping[str | EllipsisType, Mapping[str, ParamScaler[Array]]]
        | Literal[Sentinel.MISSING] = MISSING,
    ) -> None:
        dft: ParamScalerMapping[Array] | IncompleteParamScalerMapping[Array] | Literal[
            Sentinel.MISSING
        ]
        if default is MISSING:
            dft = MISSING
        elif isinstance(default, ParamScalerMapping | IncompleteParamScalerMapping):
            dft = default
        elif is_completable(default):
            dft = ParamScalerMapping(default)  # e.g. fills in None -> NoBounds
        else:
            dft = IncompleteParamScalerMapping(default)

        self._default: ParamScalerMapping[Array] | IncompleteParamScalerMapping[
            Array
        ] | Literal[Sentinel.MISSING]
        self._default = dft

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(
        self, obj: object | None, _: type | None
    ) -> ParamScalerMapping[Array] | IncompleteParamScalerMapping[Array]:
        if obj is not None:
            val: ParamScalerMapping[Array] = getattr(obj, self._name)
            return val

        default = self._default
        if default is MISSING:
            msg = f"no default value for {self._name}"
            raise AttributeError(msg)
        return default

    def __set__(
        self,
        model: SupportsCoordandParamNames,
        value: ParamScalerMapping[Array] | IncompleteParamScalerMapping[Array],
    ) -> None:
        if isinstance(value, IncompleteParamScalerMapping):
            value = value.complete(
                c for c in model.coord_names if c in model.param_names.top_level
            )
        else:
            # TODO! make sure minimal copying is done
            value = ParamScalerMapping(value)

        if self._default is not MISSING:
            default = self._default
            if isinstance(default, IncompleteParamScalerMapping):
                default = default.complete(
                    c for c in model.coord_names if c in model.param_names.top_level
                )

            # Don't need to check if the default is completable, since it
            # merged with a complete value made by ``from_names``.
            # Also, it is validated against the model's parameter names.
            # Both these are done in `stream_ml.core.base.Model`.
            value = default | value

        # Validate param bounds.
        value.validate(model.param_names)

        object.__setattr__(model, self._name, value)
