"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, TypeVar

__all__ = ["Array", "FlatParsT", "ArrayNamespace", "BoundsT"]


#####################################################################


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

    def __invert__(self: Self) -> Self:
        """Inversion."""
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

    def __neg__(self: Self) -> Self:
        """Negation."""
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


#####################################################################

FlatParsT: TypeAlias = Mapping[str, Array]


BoundsT: TypeAlias = tuple[float, float]


#####################################################################


class ArrayNamespace(Protocol[Array]):
    """Protocol for array API namespace."""

    @staticmethod
    def concatenate(arrays: tuple[Array, ...], axis: int = 0, **kwargs: Any) -> Array:
        """Concatenate."""
        ...

    @staticmethod
    def atleast_1d(array: Array) -> Array:
        """At least 1D."""
        ...

    @staticmethod
    def logsumexp(array: Array, *args: Any, **kwargs: Any) -> Array:
        """Log-sum-exp.

        First argument must be the axis ("dim" in pytorch, "axis" in jax).
        """
        ...

    @staticmethod
    def zeros(*args: Any, dtype: Any = ..., **kwargs: Any) -> Array:
        """Zeros.

        First argument must be the shape.
        """
        ...

    @staticmethod
    def ones(*args: Any, dtype: Any = ..., **kwargs: Any) -> Array:
        """Ones.

        First argument must be the shape.
        """
        ...

    @staticmethod
    def hstack(arrays: tuple[Array, ...]) -> Array:
        """Horizontal stack."""
        ...

    @staticmethod
    def zeros_like(array: Array) -> Array:
        """Zeros like."""
        ...

    @staticmethod
    def logical_or(array1: Array, array2: Array) -> Array:
        """Logical or."""
        ...

    @property
    def inf(self) -> Array:
        """Infinity."""
        ...
