"""Core feature."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

__all__ = ["Array", "ArrayNamespace", "BoundsT"]


BoundsT: TypeAlias = tuple[float, float]


#####################################################################


Self = TypeVar("Self", bound="ArrayLike")


@runtime_checkable
class ArrayLike(Protocol):
    """Protocol for array addition."""

    # ========================================================================
    # Properties

    @property
    def dtype(self) -> Any:
        """Data type."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape."""
        ...

    # ========================================================================
    # Methods

    def flatten(self: Self) -> Self:
        """Flatten."""
        ...

    # ========================================================================
    # Dunder methods

    def __len__(self) -> int:
        """Length."""
        ...

    def __getitem__(self: Self, key: Any, /) -> Self:
        """Indexing."""
        ...

    def __invert__(self: Self) -> Self:
        """Inversion."""
        ...

    def __and__(self: Self, other: ArrayLike) -> Self:
        """Bitwise and."""
        ...

    def __gt__(self: Self, other: ArrayLike | float) -> Self:
        """Greater than."""
        ...

    def __ge__(self: Self, other: ArrayLike | float) -> Self:
        """Greater than or equal."""
        ...

    def __lt__(self: Self, other: ArrayLike | float) -> Self:
        """Less than."""
        ...

    def __le__(self: Self, other: ArrayLike | float) -> Self:
        """Less than or equal."""
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

    def __truediv__(self: Self, other: ArrayLike | int) -> Self:
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


class FInfo(Protocol):
    """Protocol for floating point info."""

    eps: float


class ArrayNamespace(Protocol[Array]):
    """Protocol for array API namespace."""

    @staticmethod
    def asarray(array: Any, dtype: Any = ...) -> Array:
        """As array."""
        ...

    @staticmethod
    def atleast_1d(array: Array) -> Array:
        """At least 1D."""
        ...

    @staticmethod
    def clip(array: Array, *args: Any) -> Array:
        """Clip."""
        ...

    @staticmethod
    def concatenate(arrays: tuple[Array, ...], axis: int = 0, **kwargs: Any) -> Array:
        """Concatenate."""
        ...

    @staticmethod
    def exp(array: Array) -> Array:
        """Exponential."""
        ...

    @staticmethod
    def finfo(dtype: Any) -> FInfo:
        """Floating point info."""
        ...

    @staticmethod
    def hstack(arrays: tuple[Array, ...]) -> Array:
        """Horizontal stack."""
        ...

    @property
    def inf(self) -> Array:
        """Infinity."""
        ...

    @staticmethod
    def log(array: Array) -> Array:
        """Logarithm."""
        ...

    @staticmethod
    def logical_or(array1: Array, array2: Array) -> Array:
        """Logical or."""
        ...

    @staticmethod
    def logsumexp(array: Array, *args: Any, **kwargs: Any) -> Array:
        """Log-sum-exp.

        First argument must be the axis ("dim" in pytorch, "axis" in jax).
        """
        ...

    @staticmethod
    def ones(*args: Any, dtype: Any = ..., **kwargs: Any) -> Array:
        """Ones.

        First argument must be the shape.
        """
        ...

    @staticmethod
    def ones_like(array: Array, dtype: Any) -> Array:
        """Ones like."""
        ...

    @staticmethod
    def zeros(*args: Any, dtype: Any = ..., **kwargs: Any) -> Array:
        """Zeros.

        First argument must be the shape.
        """
        ...

    @staticmethod
    def zeros_like(array: Array) -> Array:
        """Zeros like."""
        ...


#####################################################################


class NNIdentity(Protocol[Array]):
    """Protocol for identity."""

    @staticmethod
    def __call__(x: Array) -> Array:
        """Call."""
        ...


class NNNamespace(Protocol[Array]):
    """Protocol for neural network API namespace."""

    @staticmethod
    def Identity(*args: Any, **kwargs: Any) -> NNIdentity[Array]:  # noqa: N802
        """Identity."""
        ...
