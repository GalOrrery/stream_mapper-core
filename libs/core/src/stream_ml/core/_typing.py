"""Core feature."""

# STDLIB
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

__all__: list[str] = ["ArrayLike", "DataT", "ParsT", "ArrayT"]


ArrayT = TypeVar("ArrayT", bound="ArrayLike[Any]")


class ArrayLike(Protocol[ArrayT]):
    """Protocol for array addition."""

    def __add__(self, other: ArrayT) -> ArrayT:
        """Addition."""
        ...

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> ArrayT:
        """Getitem."""
        ...


ParsT = Mapping[str, ArrayT]
DataT = Mapping[str, ArrayT]
