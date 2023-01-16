"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeGuard, TypeVar, cast, overload

# THIRD-PARTY
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

# LOCAL
from stream_ml.core.typing import Array

if TYPE_CHECKING:
    # THIRD-PARTY
    from numpy.typing import NDArray

    Self = TypeVar("Self", bound="Data[Array]")  # type: ignore[valid-type]


LEN_INDEXING_TUPLE = 2


def _all_strs(seq: tuple[Any, ...]) -> TypeGuard[tuple[str, ...]]:
    """Check if all elements of a tuple are strings."""
    return all(isinstance(x, str) for x in seq)


@dataclass(frozen=True)
class Data(Generic[Array]):
    """Labelled data.

    Parameters
    ----------
    array : Array
        The data. This should be a 2D array, where rows are observations and
        columns are features.
    names : tuple[str, ...]
        The names of the features. This should be the same length as the number
        of columns in `data`.

    Raises
    ------
    ValueError
        If the number of names does not match the number of columns in `data`.
    """

    array: Array
    _: KW_ONLY
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        # Check that the number of names matches the number of columns.
        if len(self.names) != self.array.shape[1]:
            msg = (
                f"Number of names ({len(self.names)}) does not match number of columns "
                f"({self.array.shape[1]}) in data."
            )
            raise ValueError(msg)

        # Map names to column indices.
        self._name_to_index: dict[str, int]
        object.__setattr__(
            self, "_name_to_index", {name: i for i, name in enumerate(self.names)}
        )

    def __getattr__(self, key: str) -> Any:
        """Get an attribute of the underlying array."""
        return getattr(self.array, key)

    def __len__(self) -> int:
        return len(self.array)

    # -----------------------------------------------------------------------

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
    def __getitem__(
        self: Self, key: list[int] | NDArray[np.integer[Any]], /
    ) -> Self:  # get rows
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
        | list[int]
        | NDArray[np.integer[Any]]
        | tuple[int, ...]
        | tuple[str, ...]
        | tuple[slice, ...]
        | tuple[int | slice | str | tuple[int | str, ...], ...],
        /,
    ) -> Array | Self:
        if isinstance(key, str):
            return cast("Array", self.array[:, self._name_to_index[key]])  # type: ignore[index] # noqa: E501

        elif isinstance(key, int):
            return type(self)(self.array[None, key], names=self.names)  # type: ignore[index] # noqa: E501

        elif isinstance(key, slice):
            return type(self)(self.array[key], names=self.names)  # type: ignore[index]

        elif isinstance(key, (list, np.ndarray)):
            return type(self)(self.array[key], names=self.names)  # type: ignore[index]

        elif isinstance(key, tuple) and len(key) >= LEN_INDEXING_TUPLE:
            if _all_strs(key):
                names = key
                key = (slice(None), tuple(self._name_to_index[k] for k in key))
                return type(self)(self.array[key], names=names)  # type: ignore[index]
            elif isinstance(key[1], str):
                key = (key[0], self._name_to_index[key[1]], *key[2:])
            elif isinstance(key[1], tuple):
                key = (
                    key[0],
                    tuple(
                        self._name_to_index[k] if isinstance(k, str) else k
                        for k in key[1]
                    ),
                    *key[2:],
                )

        return cast("Array", self.array[key])  # type: ignore[index]

    # =========================================================================
    # Mapping methods

    def keys(self) -> tuple[str, ...]:
        """Get the keys (the names)."""
        return self.names

    def values(self) -> tuple[Array, ...]:
        """Get the values as an iterator of the columns."""
        return tuple(self[k] for k in self.names)

    def items(self) -> tuple[tuple[str, Array], ...]:
        """Get the items as an iterator over the names and columns."""
        return tuple((k, self[k]) for k in self.names)

    # =========================================================================
    # Alternate constructors

    @classmethod
    def from_structured_array(cls: type[Self], array: Array, /) -> Self:
        """Create a `Data` instance from a structured array.

        Parameters
        ----------
        array : Array
            The structured array.

        Returns
        -------
        Data
            The data instance.
        """
        return cls(structured_to_unstructured(array), names=array.dtype.names)
