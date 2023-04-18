"""Core feature."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, final, overload

from stream_ml.core.params.names._core import (
    IncompleteParamNames,
    ParamNames,
    is_complete,
)
from stream_ml.core.utils.sentinel import MISSING, MissingT

__all__: list[str] = []


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType

    CoordParamNamesT: TypeAlias = tuple[str, ...]
    CoordParamNameGroupT: TypeAlias = tuple[str, CoordParamNamesT]

    ParamNameGroupT: TypeAlias = str | CoordParamNameGroupT

    IncompleteCoordParamNameGroupT: TypeAlias = tuple[EllipsisType, CoordParamNamesT]
    IncompleteParamNameGroupT: TypeAlias = (
        str | tuple[str, tuple[str, ...]] | tuple[EllipsisType, tuple[str, ...]]
    )

    # TypeVar
    T = TypeVar("T")


class SupportsCoordNames(Protocol):
    """Protocol for coordinate names."""

    coord_names: tuple[str, ...]


@final
class ParamNamesField:
    """Dataclass descriptor for parameter names.

    Parameters
    ----------
    default : ParamNames or MISSING, optional
        Default value, by default MISSING.
        Coordinate-specific parameters can contain 'sub'-names.
    requires_all_coordinates : bool, optional
        Whether all coordinates are required, by default True.
        If False, coordinates can be a subset of the coordinate names.
    """

    def __init__(
        self,
        default: ParamNames
        | IncompleteParamNames
        | Sequence[IncompleteParamNameGroupT]
        | MissingT = MISSING,
        *,
        requires_all_coordinates: bool = True,
    ) -> None:
        dft: ParamNames | IncompleteParamNames | MissingT
        if default is MISSING:
            dft = MISSING
        elif isinstance(default, ParamNames | IncompleteParamNames):
            dft = default
        elif is_complete(default):
            dft = ParamNames(default)
        else:
            dft = IncompleteParamNames(default)

        self._default: ParamNames | IncompleteParamNames | MissingT
        self._default = dft

        self.requires_all_coordinates = requires_all_coordinates

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    @overload
    def __get__(self, obj: None, obj_cls: Any) -> ParamNames | IncompleteParamNames:
        ...

    @overload
    def __get__(self, obj: object, obj_cls: Any) -> ParamNames:
        ...

    def __get__(
        self, obj: object | None, obj_cls: type | None
    ) -> ParamNames | IncompleteParamNames:
        if obj is not None:
            val: ParamNames = getattr(obj, self._name)
            return val

        if self._default is MISSING:
            msg = f"no default value for {self._name}"
            raise AttributeError(msg)

        return self._default

    def __set__(
        self, model: SupportsCoordNames, value: ParamNames | IncompleteParamNames
    ) -> None:
        if isinstance(value, IncompleteParamNames):
            value = value.complete(model.coord_names)

        # Validate against the default value, if it exists
        if self._default is not MISSING:
            default = self._default
            if isinstance(default, IncompleteParamNames):
                default = default.complete(model.coord_names)

            # Some(/most) param_names must be over the full set of coordinate
            # names, but not all, so we can't just check for equality.
            if self.requires_all_coordinates:
                if value != default:
                    msg = (
                        f"invalid value for {type(model).__name__}.{self._name[1:]}:"
                        f" expected {default}, got {value}"
                    )
                    raise ValueError(msg)
            elif not set(value).issubset(default):
                msg = (
                    f"invalid value for {type(model).__name__}.{self._name[1:]}:"
                    f" expected a subset of {default}, got {value}"
                )
                raise ValueError(msg)

        object.__setattr__(model, self._name, ParamNames(value))
