"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from typing import Any, Protocol

from stream_ml.core.typing._array import Array


class FInfo(Protocol):
    """Protocol for floating point info."""

    eps: float


class ArrayNamespace(Protocol[Array]):
    """Protocol for array API namespace."""

    @property
    def __name__(self) -> str:  # noqa: A003
        """Name of the namespace."""
        ...

    @property
    def special(self) -> ArraySpecialNamespace[Array]:
        """Special namespace."""
        ...

    # ==============================================================

    @property
    def pi(self) -> float:
        """Pi."""
        ...

    # ==============================================================

    @staticmethod
    def abs(array: Array) -> Array:  # noqa: A003
        """Absolute value."""
        ...

    @staticmethod
    def any(array: Array) -> Array:  # noqa: A003
        """Any."""
        ...

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
    def concatenate(
        arrays: tuple[Array, ...] | list[Array], axis: int = 0, **kwargs: Any
    ) -> Array:
        """Concatenate."""
        ...

    @staticmethod
    def empty_like(array: Array, dtype: Any = None) -> Array:
        """Empty-like."""
        ...

    @staticmethod
    def exp(array: Array) -> Array:
        """Exponential."""
        ...

    @staticmethod
    def expm1(array: Array) -> Array:
        """Exponential minus 1."""
        ...

    @staticmethod
    def finfo(dtype: Any) -> FInfo:
        """Floating point info."""
        ...

    @staticmethod
    def full(shape: tuple[int, ...], fill_value: Any) -> Array:
        """Full."""
        ...

    @staticmethod
    def full_like(array: Array, fill_value: Any) -> Array:
        """Full like."""
        ...

    @staticmethod
    def hstack(arrays: tuple[Array, ...] | list[Array]) -> Array:
        """Horizontal stack."""
        ...

    @property
    def inf(self) -> Array:
        """Infinity."""
        ...

    @property
    def isfinite(self) -> Any:
        """Is finite."""
        ...

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> Array:
        """Linearly spaced array."""
        ...

    @staticmethod
    def log(array: Array) -> Array:
        """Logarithm, base e."""
        ...

    @staticmethod
    def log10(array: Array) -> Array:
        """Logarithm, base 10."""
        ...

    @staticmethod
    def logaddexp(array1: Array, array2: Array, /) -> Array:
        """Logarithm of the sum of exponentials."""
        ...

    @staticmethod
    def logical_or(array1: Array, array2: Array) -> Array:
        """Logical or."""
        ...

    @staticmethod
    def mean(array: Array, /, axis: int | None = None) -> Array:
        """Mean."""
        ...

    @staticmethod
    def ones(*args: Any, dtype: Any = ..., **kwargs: Any) -> Array:
        """Ones.

        First argument must be the shape.
        """
        ...

    @staticmethod
    def ones_like(array: Array, *, dtype: Any = None) -> Array:
        """Ones like."""
        ...

    @staticmethod
    def sum(array: Array, /, axis: int | None = None) -> Array:  # noqa: A003
        """Sum."""
        ...

    @staticmethod
    def swapaxes(array: Array, axis1: int, axis2: int) -> Array:
        """Swap axes."""
        ...

    @staticmethod
    def where(condition: Array, x: Array, y: Array) -> Array:
        """Where."""
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

    @staticmethod
    def isneginf(array: Array) -> Array:
        """Is negative infinity."""
        ...

    @staticmethod
    def isposinf(array: Array) -> Array:
        """Is positive infinity."""
        ...

    @staticmethod
    def isinf(array: Array) -> Array:
        """Is infinity."""
        ...

    @staticmethod
    def sigmoid(array: Array) -> Array:
        """Sigmoid."""
        ...

    @staticmethod
    def stack(arrays: tuple[Array, ...] | list[Array], axis: int = ...) -> Array:
        """Vertical stack."""
        ...

    @staticmethod
    def std(array: Array, /, axis: int | None = ...) -> Array:
        """Standard deviation."""
        ...

    @staticmethod
    def square(array: Array) -> Array:
        """Square."""
        ...

    @staticmethod
    def sqrt(array: Array) -> Array:
        """Square root."""
        ...

    @staticmethod
    def squeeze(array: Array, axis: None | int | tuple[int, ...] = ...) -> Array:
        """Squeeze."""
        ...

    @staticmethod
    def vstack(arrays: tuple[Array, ...]) -> Array:
        """Vertical stack."""
        ...


# ========================================================================


class ArraySpecialNamespace(Protocol[Array]):
    """Protocol for array API namespace."""

    @staticmethod
    def erf(array: Array) -> Array:
        """Error function."""
        ...

    @staticmethod
    def erfc(array: Array) -> Array:
        """Complementary error function."""
        ...

    @staticmethod
    def logsumexp(array: Array, *args: Any, **kwargs: Any) -> Array:
        """Log-sum-exp.

        First argument must be the axis ("dim" in pytorch, "axis" in jax).
        """
        ...
