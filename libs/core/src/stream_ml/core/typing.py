"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, TypeVar

__all__ = ["Array", "FlatParsT"]


Self = TypeVar("Self", bound="ArrayLike")


class ArrayLike(Protocol):
    """Protocol for array addition."""

    @property
    def dtype(self) -> Any:
        """Data type."""
        ...

    # ========================================================================
    # Properties

    def __len__(self) -> int:
        """Length."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape."""
        ...

    # ========================================================================
    # Dunder methods

    def __getitem__(self: Self, key: Any, /) -> Self:
        """Indexing."""
        ...

    # ========================================================================
    # Math

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

    # ========================================================================
    # Math Methods

    def max(self: Self) -> Self:  # noqa: A003
        """Maximum."""
        ...

    def min(self: Self) -> Self:  # noqa: A003
        """Minimum."""
        ...

    def sum(self: Self, axis: int | None = None) -> Self:  # noqa: A003
        """Sum."""
        ...


Array = TypeVar("Array", bound="ArrayLike")


FlatParsT = Mapping[str, Array]


BoundsT: TypeAlias = tuple[float, float]
