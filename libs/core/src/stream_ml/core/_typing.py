"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Mapping, MutableMapping
from typing import Any, Protocol, TypeVar

__all__ = [
    "Array",
    # Parameters
    "FlatParsT",
    "MutableFlatParsT",
    # Data
    "DataT",
    "MutableDataT",
]


Self = TypeVar("Self", bound="ArrayLike")


class ArrayLike(Protocol):
    """Protocol for array addition."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape."""
        ...

    # ========================================================================
    # Dunder methods

    def __getitem__(self: Self, key: Any) -> Self:
        """Indexing."""
        ...

    def __add__(self: Self, other: ArrayLike | int | float) -> Self:
        """Addition."""
        ...

    def __radd__(self: Self, other: ArrayLike | int | float) -> Self:
        """Right addition."""
        ...

    def __mul__(self: Self, other: ArrayLike | int | float) -> Self:
        """Multiplication."""
        ...

    def __sub__(self: Self, other: ArrayLike | int | float) -> Self:
        """Subtraction."""
        ...

    def __rsub__(self: Self, other: ArrayLike | int | float) -> Self:
        """Right subtraction."""
        ...

    def __truediv__(self: Self, other: ArrayLike) -> Self:
        """True division."""
        ...

    def __div__(self: Self, other: ArrayLike) -> Self:
        """Division."""
        ...

    # ========================================================================
    # Methods

    def max(self: Self) -> Self:  # noqa: A003
        ...

    def min(self: Self) -> Self:  # noqa: A003
        ...

    def sum(self: Self, axis: int | None = None) -> Self:  # noqa: A003
        """Sum."""
        ...


Array = TypeVar("Array", bound="ArrayLike")
Array_co = TypeVar("Array_co", bound="ArrayLike", covariant=True)


FlatParsT = Mapping[str, Array]
MutableFlatParsT = MutableMapping[str, Array]

DataT = Mapping[str, Array]
MutableDataT = MutableMapping[str, Array]
