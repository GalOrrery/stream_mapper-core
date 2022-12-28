"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Sequence
from types import EllipsisType
from typing import (
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    final,
    overload,
)

# LOCAL
from stream_ml.core.utils.sentinel import MISSING, Sentinel

__all__: list[str] = []


T = TypeVar("T")
Self = TypeVar("Self", bound="ParamNames")


CoordParamNamesT: TypeAlias = tuple[str, ...]

CoordParamNameGroupT: TypeAlias = tuple[str, CoordParamNamesT]
IncompleteCoordParamNameGroupT: TypeAlias = tuple[EllipsisType, CoordParamNamesT]

ParamNameGroupT: TypeAlias = str | CoordParamNameGroupT
IncompleteParamNameGroupT: TypeAlias = (
    str | tuple[str, tuple[str, ...]] | tuple[EllipsisType, tuple[str, ...]]  # type: ignore[misc]  # noqa: E501
)

FlatParamName: TypeAlias = tuple[str] | tuple[str, str]  # type: ignore[misc]
FlatParamNames: TypeAlias = tuple[FlatParamName, ...]


class ParamNamesBase(
    tuple[str | tuple[str, CoordParamNamesT] | tuple[T, CoordParamNamesT], ...]
):
    """Base class for parameter names."""

    @property
    def flat(self) -> tuple[str | T, ...]:
        """Flattened parameter names."""
        names: list[str | T] = []

        for pn in self:
            if isinstance(pn, str):
                names.append(pn)
            else:
                names.extend(f"{pn[0]}_{k}" for k in pn[1])

        return tuple(names)

    @property
    def flats(self) -> tuple[tuple[str] | tuple[str | T, str], ...]:
        """Flattened parameter names as tuples."""
        names: list[tuple[str] | tuple[str | T, str]] = []

        for pn in self:
            if isinstance(pn, str):
                names.append((pn,))
            else:
                names.extend((pn[0], k) for k in pn[1])

        return tuple(names)


@final
class ParamNames(ParamNamesBase[str]):
    """Parameter names."""

    def __self__(self, iterable: Any = (), /) -> None:
        """Create a new ParamNames instance."""
        super().__init__()

        # Validate structure
        for elt in self:
            if isinstance(elt, str):
                continue
            elif not isinstance(elt, tuple):
                raise TypeError(f"Invalid element type: {type(elt)}")
            elif len(elt) != 2:
                raise ValueError(f"Invalid element length: {len(elt)}")
            elif not isinstance(elt[0], str):
                raise TypeError(f"Invalid element type: {type(elt[0])}")
            elif not isinstance(elt[1], tuple):
                raise TypeError(f"Invalid element type: {type(elt[1])}")
            elif not all(isinstance(e, str) for e in elt[1]):
                raise TypeError(f"Invalid element type: {type(elt[1])}")


@final
class IncompleteParamNames(ParamNamesBase[EllipsisType]):
    """Incomplete parameter names."""

    def __self__(self, iterable: Any = (), /) -> None:
        """Create a new ParamNames instance."""
        super().__init__()

        # Validate structure
        for elt in self:
            if isinstance(elt, str):
                continue
            elif not isinstance(elt, tuple):
                raise TypeError(f"Invalid element type: {type(elt)}")
            elif len(elt) != 2:
                raise ValueError(f"Invalid element length: {len(elt)}")
            elif not isinstance(elt[0], (str, EllipsisType)):
                raise TypeError(f"Invalid element type: {type(elt[0])}")
            elif not isinstance(elt[1], tuple):
                raise TypeError(f"Invalid element type: {type(elt[1])}")
            elif not all(isinstance(e, str) for e in elt[1]):
                raise TypeError(f"Invalid element type: {type(elt[1])}")

    @property
    def is_complete(self) -> TypeGuard[tuple[ParamNameGroupT, ...]]:
        """Check if parameter names are complete."""
        return is_complete(self)

    def complete(self, coord_names: tuple[str, ...]) -> ParamNames:
        """Complete parameter names."""
        names: list[str | tuple[str, tuple[str, ...]]] = []

        cni = 0
        for pn in self:
            if isinstance(pn, str):
                names.append(pn)
                continue

            pn0 = pn[0]
            if isinstance(pn0, EllipsisType):
                names.append((coord_names[cni], pn[1]))
            else:
                names.append((pn0, pn[1]))

            cni += 1

        return ParamNames(names)


def is_complete(
    pns: Sequence[IncompleteParamNameGroupT], /
) -> TypeGuard[Sequence[ParamNameGroupT]]:
    """Check if parameter names are complete."""
    for pn in pns:
        if isinstance(pn, tuple) and isinstance(pn[0], EllipsisType):
            return False
    return True


#####################################################################


class SupportsCoordNames(Protocol):
    """Protocol for coordinate names."""

    coord_names: tuple[str, ...]


class ParamNamesField:
    """Dataclass descriptor for parameter names.

    Parameters
    ----------
    default : ParamNames or MISSING, optional
        Default value, by default MISSING.
        Coordinate-specific parameters can contain 'sub'-names.
    """

    def __init__(
        self,
        default: ParamNames
        | IncompleteParamNames
        | Sequence[IncompleteParamNameGroupT]
        | Literal[Sentinel.MISSING] = MISSING,
    ) -> None:
        dft: ParamNames | IncompleteParamNames | Literal[Sentinel.MISSING]
        if default is MISSING:
            dft = MISSING
        elif isinstance(default, (ParamNames, IncompleteParamNames)):
            dft = default
        elif is_complete(default):
            dft = ParamNames(default)
        else:
            dft = IncompleteParamNames(default)
        self._default: ParamNames | IncompleteParamNames | Literal[Sentinel.MISSING]
        self._default = dft

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

        default = self._default
        if default is not MISSING:
            return default
        else:
            raise AttributeError(f"no default value for {self._name}")

    def __set__(
        self, obj: SupportsCoordNames, value: ParamNames | IncompleteParamNames
    ) -> None:
        if isinstance(value, IncompleteParamNames):
            value = value.complete(obj.coord_names)

        object.__setattr__(obj, self._name, ParamNames(value))
