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
    "ParsT",
    "MutableParsT",
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

    def __add__(self: Self, other: ArrayLike) -> Self:
        """Addition."""
        ...

    def __radd__(self: Self, other: ArrayLike | int) -> Self:
        """Right addition."""
        ...

    def __getitem__(self: Self, key: Any) -> Self:
        """Getitem."""
        ...

    # ========================================================================
    # Methods

    def sum(self: Self, axis: int | None = None) -> Self:  # noqa: A003
        """Sum."""
        ...


Array = TypeVar("Array", bound="ArrayLike")
Array_co = TypeVar("Array_co", bound="ArrayLike", covariant=True)


FlatParsT = Mapping[str, Array]
MutableFlatParsT = MutableMapping[str, Array]


ParsT = Mapping[str, Array | Mapping[str, Array]]
MutableParsT = MutableMapping[str, Array | MutableMapping[str, Array]]

DataT = Mapping[str, Array]
MutableDataT = MutableMapping[str, Array]
