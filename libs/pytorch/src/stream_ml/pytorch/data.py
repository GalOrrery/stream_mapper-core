"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TypeVar, overload

# LOCAL
from stream_ml.core.data import Data as CoreData
from stream_ml.pytorch._typing import Array

Self = TypeVar("Self", bound="Data[Array]")  # type: ignore[type-arg]


@dataclass(frozen=True)
class Data(CoreData[Array]):
    """Data."""

    @overload
    def __getitem__(self, key: str, /) -> Array:  # get a column
        ...

    @overload
    def __getitem__(self: Self, key: int, /) -> Self:  # get a row
        ...

    @overload
    def __getitem__(self: Self, key: slice, /) -> Self:  # get a slice of rows
        ...

    @overload
    def __getitem__(self: Self, key: tuple[str, ...], /) -> Self:  # get columns
        ...

    @overload
    def __getitem__(self, key: tuple[int, ...], /) -> Array:  # get element
        ...

    @overload
    def __getitem__(self, key: tuple[slice, ...], /) -> Array:  # get elements
        ...

    @overload
    def __getitem__(
        self, key: tuple[int | slice | str | tuple[int | str, ...], ...], /
    ) -> Array:
        ...

    def __getitem__(
        self: Self,
        key: str
        | int
        | slice
        | tuple[int, ...]
        | tuple[str, ...]
        | tuple[slice, ...]
        | tuple[int | slice | str | tuple[int | str, ...], ...],
        /,
    ) -> Array | Self:
        out = super().__getitem__(key)

        if isinstance(out, type(self)) and self.array.ndim == 1:
            out.array.shape = (-1, 1)

        return out
