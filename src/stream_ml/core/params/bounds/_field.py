"""Core feature."""

from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar

from stream_ml.core.params.bounds._core import (
    IncompleteParamBounds,
    ParamBounds,
    is_completable,
)
from stream_ml.core.typing import Array
from stream_ml.core.utils.sentinel import MISSING, MissingT

__all__: list[str] = []

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stream_ml.core.params.names import ParamNamesField
    from stream_ml.core.prior.bounds._core import PriorBounds

    Self = TypeVar("Self", bound="ParamBounds[Array]")  # type: ignore[valid-type]
    Object = TypeVar("Object")

T = TypeVar("T", bound=str | EllipsisType)


class SupportsCoordandParamNames(Protocol):
    """Protocol for coordinate names."""

    coord_names: tuple[str, ...]
    param_names: ParamNamesField
    DEFAULT_PARAM_BOUNDS: ClassVar


class ParamBoundsField(Generic[Array]):
    """Dataclass descriptor for parameter bounds.

    Parameters
    ----------
    default : ParamBounds or Mapping or None, optional
        The default parameter bounds, by default `None`. If `None`, there are no
        default bounds and the parameter bounds must be specified in the Model
        constructor. If not a `ParamBounds` instance, it will be converted to
        one.

    Notes
    -----
    See https://docs.python.org/3/library/dataclasses.html for more information
    on descriptor-typed fields for dataclasses.
    """

    def __init__(
        self,
        default: ParamBounds[Array]
        | Mapping[
            str | EllipsisType,
            PriorBounds[Array] | None | Mapping[str, PriorBounds[Array] | None],
        ]
        | MissingT = MISSING,
    ) -> None:
        dft: ParamBounds[Array] | IncompleteParamBounds[Array] | MissingT
        if default is MISSING:
            dft = MISSING
        elif isinstance(default, ParamBounds | IncompleteParamBounds):
            dft = default
        elif is_completable(default):
            dft = ParamBounds(default)  # e.g. fills in None -> NoBounds
        else:
            dft = IncompleteParamBounds(default)

        self._default: ParamBounds[Array] | IncompleteParamBounds[Array] | MissingT
        self._default = dft

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(
        self, obj: object | None, _: type | None
    ) -> ParamBounds[Array] | IncompleteParamBounds[Array]:
        # Accessing the descriptor on the class returns the default value.
        if obj is not None:
            val: ParamBounds[Array] = getattr(obj, self._name)
            return val

        # Accessing the descriptor from the instance returns the default value
        # unless the value is missing.
        default = self._default
        if default is MISSING:
            msg = f"no default value for {self._name}"
            raise AttributeError(msg)
        return default

    def __set__(
        self,
        model: SupportsCoordandParamNames,
        value: ParamBounds[Array] | IncompleteParamBounds[Array],
    ) -> None:
        if isinstance(value, IncompleteParamBounds):
            value = value.complete(
                c for c in model.coord_names if c in model.param_names.top_level
            )
        else:
            # TODO! make sure minimal copying is done
            value = ParamBounds(value)

        if self._default is not MISSING:
            default = self._default
            if isinstance(default, IncompleteParamBounds):
                default = default.complete(
                    c for c in model.coord_names if c in model.param_names.top_level
                )

            # Don't need to check if the default is completable, since it
            # merged with a complete value made by ``from_names``.
            # Also, it is validated against the model's parameter names.
            # Both these are done in `stream_ml.core.Model`.
            value = default | value

        # TODO: can this be done in the param_bounds field?
        # Make parameter bounds
        # 1) Make the default bounds for all parameters.
        # 2) Update from the user-specified bounds.
        # 3) Fix up the names so each bound references its parameter.
        value = (
            ParamBounds.from_names(
                model.param_names, default=model.DEFAULT_PARAM_BOUNDS
            )
            | value
        )
        value._fixup_param_names()
        # Validate param bounds.
        value.validate(model.param_names)

        object.__setattr__(model, self._name, value)


##############################################################################


class SupportsCoordandMixtureParamNames(SupportsCoordandParamNames, Protocol):
    """Protocol for coordinate names."""

    mixture_param_names: ParamNamesField


class MixtureParamBoundsField(ParamBoundsField[Array]):
    """Dataclass descriptor for parameter bounds.

    Parameters
    ----------
    default : ParamBounds or Mapping or None, optional
        The default parameter bounds, by default `None`. If `None`, there are no
        default bounds and the parameter bounds must be specified in the Model
        constructor. If not a `ParamBounds` instance, it will be converted to
        one.

    Notes
    -----
    See https://docs.python.org/3/library/dataclasses.html for more information
    on descriptor-typed fields for dataclasses.
    """

    def __init__(
        self,
        default: ParamBounds[Array]
        | Mapping[
            str | EllipsisType,
            PriorBounds[Array] | None | Mapping[str, PriorBounds[Array] | None],
        ]
        | MissingT = MISSING,
    ) -> None:
        dft: ParamBounds[Array] | IncompleteParamBounds[Array] | MissingT
        if default is MISSING:
            dft = MISSING
        elif isinstance(default, ParamBounds | IncompleteParamBounds):
            dft = default
        elif is_completable(default):
            dft = ParamBounds(default)  # e.g. fills in None -> NoBounds
        else:
            dft = IncompleteParamBounds(default)

        self._default: ParamBounds[Array] | IncompleteParamBounds[Array] | MissingT
        self._default = dft

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(
        self, obj: object | None, _: type | None
    ) -> ParamBounds[Array] | IncompleteParamBounds[Array]:
        # Accessing the descriptor on the class returns the default value.
        if obj is not None:
            val: ParamBounds[Array] = getattr(obj, self._name)
            return val

        # Accessing the descriptor from the instance returns the default value
        # unless the value is missing.
        default = self._default
        if default is MISSING:
            msg = f"no default value for {self._name}"
            raise AttributeError(msg)
        return default

    def __set__(
        self,
        model: SupportsCoordandMixtureParamNames,  # type: ignore[override]
        value: ParamBounds[Array] | IncompleteParamBounds[Array],
    ) -> None:
        if isinstance(value, IncompleteParamBounds):
            value = value.complete(
                c for c in model.coord_names if c in model.mixture_param_names.top_level
            )
        else:
            # TODO! make sure minimal copying is done
            value = ParamBounds(value)

        if self._default is not MISSING:
            default = self._default
            if isinstance(default, IncompleteParamBounds):
                default = default.complete(
                    c
                    for c in model.coord_names
                    if c in model.mixture_param_names.top_level
                )

            # Don't need to check if the default is completable, since it
            # merged with a complete value made by ``from_names``.
            # Also, it is validated against the model's parameter names.
            # Both these are done in `stream_ml.core.Model`.
            value = default | value

        # TODO: can this be done in the param_bounds field?
        # Make parameter bounds
        # 1) Make the default bounds for all parameters.
        # 2) Update from the user-specified bounds.
        # 3) Fix up the names so each bound references its parameter.
        value = (
            ParamBounds.from_names(
                model.mixture_param_names, default=model.DEFAULT_PARAM_BOUNDS
            )
            | value
        )
        value._fixup_param_names()
        # Validate param bounds.
        value.validate(model.mixture_param_names)

        object.__setattr__(model, self._name, value)
