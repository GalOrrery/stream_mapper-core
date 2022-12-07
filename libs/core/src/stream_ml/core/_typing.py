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


# class ParsT(Protocol[Array_co]):
#     """Parameters."""

#     @overload
#     def __getitem__(self, key: Literal["mixparam"]) -> Array_co:
#         ...

#     @overload
#     def __getitem__(self, key: str) -> Mapping[str, Array_co]:
#         ...

#     def __getitem__(
#         self, key: Literal["mixparam"] | str
#     ) -> Array_co | Mapping[str, Array_co]:
#         ...

#     def __iter__(self) -> Iterator[str]:
#         ...

#     def __len__(self) -> int:
#         ...


# class MutableParsT(ParsT[Array], Protocol):
#     """Mutable parameters."""

#     @overload
#     def __setitem__(self, key: Literal["mixparam"], value: Array) -> None:
#         ...

#     @overload
#     def __setitem__(self, key: str, value: Mapping[str, Array]) -> None:
#         ...

#     def __setitem__(
#         self, key: Literal["mixparam"] | str, value: Array | Mapping[str, Array]  # noqa: E501
#     ) -> None:
#         ...

#     def __delitem__(self, key: str) -> None:
#         ...


ParsT = Mapping[str, Array | Mapping[str, Array]]
MutableParsT = MutableMapping[str, Array | MutableMapping[str, Array]]

DataT = Mapping[str, Array]
MutableDataT = MutableMapping[str, Array]
