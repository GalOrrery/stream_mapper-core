"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Iterator, Sequence
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


LEN_NAME_TUPLE = 2


class ParamNamesBase(
    Sequence[str | tuple[str, CoordParamNamesT] | tuple[T, CoordParamNamesT]]
):
    """Base class for parameter names."""

    def __init__(self, iterable: Any = (), /) -> None:
        """Create a new ParamNames instance."""
        super().__init__()

        self._data = tuple(iterable)

        # hint property types
        self._top_level: tuple[str | T, ...]
        self._flat: tuple[str | T, ...]
        self._flats: tuple[tuple[str] | tuple[str | T, str], ...]

    @property
    def top_level(self) -> tuple[str | T, ...]:
        """Top-level parameter names."""
        if not hasattr(self, "_top_level"):
            object.__setattr__(
                self,
                "_top_level",
                tuple(k[0] if isinstance(k, tuple) else k for k in self),
            )
        return self._top_level

    # ========================================================================
    # Concretizing abstract methods

    def __contains__(self, value: object) -> bool:
        return value in self._data

    def __iter__(
        self,
    ) -> Iterator[str | tuple[str, CoordParamNamesT] | tuple[T, CoordParamNamesT]]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> str | tuple[str, CoordParamNamesT]:
        ...

    @overload
    def __getitem__(
        self, index: slice
    ) -> tuple[str | tuple[str, CoordParamNamesT], ...]:
        ...

    def __getitem__(
        self, index: int | slice
    ) -> str | tuple[str, CoordParamNamesT] | tuple[
        str | tuple[str, CoordParamNamesT], ...
    ]:
        return self._data[index]

    # ========================================================================
    # Comparison

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParamNamesBase):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        return hash(self._data)

    # ========================================================================
    # Flat

    @property
    def flat(self) -> tuple[str | T, ...]:
        """Flattened parameter names."""
        if not hasattr(self, "_flat"):
            names: list[str | T] = []

            for pn in self:
                if isinstance(pn, str):
                    names.append(pn)
                else:
                    names.extend(f"{pn[0]}_{k}" for k in pn[1])

            object.__setattr__(self, "_flat", tuple(names))

        return self._flat

    @property
    def flats(self) -> tuple[tuple[str] | tuple[str | T, str], ...]:
        """Flattened parameter names as tuples."""
        # ParamNamesBase is immutable, so we can cache this safely.
        if not hasattr(self, "_flats"):

            names: list[tuple[str] | tuple[str | T, str]] = []

            for pn in self:
                if isinstance(pn, str):
                    names.append((pn,))
                else:
                    names.extend((pn[0], k) for k in pn[1])

            object.__setattr__(self, "_flats", tuple(names))

        return self._flats

    # ========================================================================

    def __repr__(self) -> str:
        """Get representation."""
        return f"{type(self).__name__}({self._data!r})"


@final
class ParamNames(ParamNamesBase[str]):
    """Parameter names."""

    def __init__(self, iterable: Any = (), /) -> None:
        """Create a new ParamNames instance."""
        super().__init__(iterable)

        # Validate structure
        for elt in self:
            if isinstance(elt, str):
                continue
            elif not isinstance(elt, tuple):
                msg = f"Invalid element type: {type(elt)}"  # type: ignore[unreachable] # noqa: E501
                raise TypeError(msg)
            elif len(elt) != LEN_NAME_TUPLE:
                msg = f"Invalid element length: {len(elt)}"
                raise ValueError(msg)
            elif not isinstance(elt[0], str):
                msg = f"Invalid element type: {type(elt[0])}"  # type: ignore[unreachable] # noqa: E501
                raise TypeError(msg)
            elif not isinstance(elt[1], tuple):
                msg = f"Invalid element type: {type(elt[1])}"  # type: ignore[unreachable] # noqa: E501
                raise TypeError(msg)
            elif not all(isinstance(e, str) for e in elt[1]):
                msg = f"Invalid element type: {type(elt[1])}"
                raise TypeError(msg)


@final
class IncompleteParamNames(ParamNamesBase[EllipsisType]):
    """Incomplete parameter names."""

    def __init__(self, iterable: Any = (), /) -> None:
        """Create a new ParamNames instance."""
        super().__init__(iterable)

        # Validate structure
        for elt in self:
            if isinstance(elt, str):
                continue
            elif not isinstance(elt, tuple):
                msg = f"Invalid element type: {type(elt)}"  # type: ignore[unreachable]
                raise TypeError(msg)
            elif len(elt) != LEN_NAME_TUPLE:
                msg = f"Invalid element length: {len(elt)}"
                raise ValueError(msg)
            elif not isinstance(elt[0], (str, EllipsisType)):
                msg = f"Invalid element type: {type(elt[0])}"  # type: ignore[unreachable] # noqa: E501
                raise TypeError(msg)
            elif not isinstance(elt[1], tuple):
                msg = f"Invalid element type: {type(elt[1])}"  # type: ignore[unreachable] # noqa: E501
                raise TypeError(msg)
            elif not all(isinstance(e, str) for e in elt[1]):
                msg = f"Invalid element type: {type(elt[1])}"
                raise TypeError(msg)

    @property
    def is_complete(self) -> TypeGuard[tuple[ParamNameGroupT, ...]]:
        """Check if parameter names are complete."""
        return is_complete(self)

    def complete(self, coord_names: tuple[str, ...]) -> ParamNames:
        """Complete parameter names."""
        names: list[str | tuple[str, tuple[str, ...]]] = []

        for pn in self:
            if isinstance(pn, str):
                names.append(pn)
                continue

            pn0 = pn[0]
            if isinstance(pn0, EllipsisType):
                names.extend((cn, pn[1]) for cn in coord_names)
            else:
                names.append((pn0, pn[1]))

        return ParamNames(names)


def is_complete(
    pns: Sequence[IncompleteParamNameGroupT], /
) -> TypeGuard[Sequence[ParamNameGroupT]]:
    """Check if parameter names are complete."""
    return all(
        not (isinstance(pn, tuple) and isinstance(pn[0], EllipsisType)) for pn in pns
    )


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
    requires_all_coordinates : bool, optional
        Whether all coordinates are required, by default True.
        If False, coordinates can be a subset of the coordinate names.
    """

    def __init__(
        self,
        default: ParamNames
        | IncompleteParamNames
        | Sequence[IncompleteParamNameGroupT]
        | Literal[Sentinel.MISSING] = MISSING,
        *,
        requires_all_coordinates: bool = True,
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

        default = self._default
        if default is not MISSING:
            return default
        else:
            msg = f"no default value for {self._name}"
            raise AttributeError(msg)

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
