"""Core feature."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import EllipsisType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, TypeVar, overload

from stream_ml.core.utils.cached_property import cached_property

__all__: list[str] = []


if TYPE_CHECKING:
    CoordParamNamesT: TypeAlias = tuple[str, ...]
    CoordParamNameGroupT: TypeAlias = tuple[str, CoordParamNamesT]

    ParamNameGroupT: TypeAlias = str | CoordParamNameGroupT

    IncompleteCoordParamNameGroupT: TypeAlias = tuple[EllipsisType, CoordParamNamesT]
    IncompleteParamNameGroupT: TypeAlias = (
        str | tuple[str, tuple[str, ...]] | tuple[EllipsisType, tuple[str, ...]]
    )

    # TypeVar
    T = TypeVar("T")

FlatParamName: TypeAlias = tuple[str] | tuple[str, str]


LEN_NAME_TUPLE = 2


#####################################################################


class ParamNamesBase(
    Sequence[str | tuple[str, "CoordParamNamesT"] | tuple["T", "CoordParamNamesT"]]
):
    """Base class for parameter names."""

    def __init__(self, iterable: Any = (), /) -> None:
        """Create a new ParamNames instance."""
        super().__init__()

        self._data = tuple(iterable)

        # hint cached values
        self._top_level: tuple[str | T, ...]
        self._flat: tuple[str | T, ...]
        self._flats: tuple[tuple[str] | tuple[str | T, str], ...]

    @cached_property
    def top_level(self) -> tuple[str | T, ...]:
        """Top-level parameter names."""
        return tuple(k[0] if isinstance(k, tuple) else k for k in self)

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
    ) -> (
        str
        | tuple[str, CoordParamNamesT]
        | tuple[str | tuple[str, CoordParamNamesT], ...]
    ):
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

    @cached_property
    def flat(self) -> tuple[str | T, ...]:
        """Flattened parameter names."""
        names: list[str | T] = []
        for pn in self:
            if isinstance(pn, str):
                names.append(pn)
            else:
                names.extend(f"{pn[0]}_{k}" for k in pn[1])
        return tuple(names)

    @cached_property
    def flats(self) -> tuple[tuple[str] | tuple[str | T, str], ...]:
        """Flattened parameter names as tuples."""
        names: list[tuple[str] | tuple[str | T, str]] = []
        for pn in self:
            if isinstance(pn, str):
                names.append((pn,))
            else:
                names.extend((pn[0], k) for k in pn[1])
        return tuple(names)

    # ========================================================================

    def __repr__(self) -> str:
        """Get representation."""
        return f"{type(self).__name__}({self._data!r})"


class ParamNames(ParamNamesBase[str]):
    """Parameter names."""


class IncompleteParamNames(ParamNamesBase[EllipsisType]):
    """Incomplete parameter names."""

    @property
    def is_complete(self) -> bool:
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
