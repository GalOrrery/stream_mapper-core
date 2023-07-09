"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator


#####################################################################


Self = TypeVar("Self", bound="ArrayLike")


# @runtime_checkable  # TODO: when mypyc supports runtime_checkable protocols
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

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        ...

    @property
    def T(self: Self) -> Self:  # noqa: N802
        """Transpose."""
        ...

    # ========================================================================
    # Methods

    def flatten(self: Self) -> Self:
        """Flatten."""
        ...

    def reshape(self: Self, *shape: Any) -> Self:
        """Reshape."""
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

    def __iter__(self: Self) -> Iterator[Self]:
        """Iteration."""
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

    def __rmul__(self: Self, other: ArrayLike | int | float) -> Self:
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

    def __truediv__(self: Self, other: ArrayLike | int | float) -> Self:
        """True division."""
        ...

    def __rtruediv__(self: Self, other: ArrayLike | int | float) -> Self:
        """Right true division."""
        ...

    def __pow__(self: Self, other: ArrayLike | int | float) -> Self:
        """Power."""
        ...

    # ========================================================================
    # Math Methods

    def max(self: Self) -> Self:  # noqa: A003
        """Maximum."""
        ...

    def min(self: Self) -> Self:  # noqa: A003
        """Minimum."""
        ...

    def sum(self: Self, axis: int | None | tuple[int, ...] = ...) -> Self:  # noqa: A003
        """Sum."""
        ...


Array = TypeVar("Array", bound="ArrayLike")
Array_co = TypeVar("Array_co", bound="ArrayLike", covariant=True)
