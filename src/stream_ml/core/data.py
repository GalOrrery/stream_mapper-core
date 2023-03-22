"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from textwrap import indent
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

from stream_ml.core.typing import Array

if TYPE_CHECKING:
    from collections.abc import Callable

    from stream_ml.core.typing import ArrayLike

    Self = TypeVar("Self", bound="Data[Array]")  # type: ignore[valid-type]

    ArrayT = TypeVar("ArrayT", bound=ArrayLike)
    T = TypeVar("T")


#####################################################################
# PARAMETERS


_LEN_INDEXING_TUPLE: Final = 1


#####################################################################


def _all_strs(seq: tuple[Any, ...]) -> TypeGuard[tuple[str, ...]]:
    """Check if all elements of a tuple are strings."""
    return all(isinstance(x, str) for x in seq)


def _is_arraylike(obj: Any) -> TypeGuard[ArrayLike]:
    """Check if an object is array-like.

    This only exists b/c mypyc does not yet support runtime_checkable protocols,
    so `isinstance(obj, ArrayLike)` does not work.
    """
    return hasattr(obj, "dtype") and hasattr(obj, "shape")


# @dataclass(frozen=True)  # TODO: when mypyc supports generic dataclasses
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

    def __init__(self, array: Array, /, *, names: tuple[str, ...]) -> None:
        super().__init__()

        self._array = array
        self._names = names

        self.__post_init__()

    @property
    def array(self) -> Array:
        """The underlying array."""
        return self._array

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the features."""
        return self._names

    # =========================================================================

    def __post_init__(self) -> None:
        # Check that the number of names matches the number of columns.
        if len(self.names) != self.array.shape[1]:
            msg = (
                f"Number of names ({len(self.names)}) does not match number of columns "
                f"({self.array.shape[1]}) in data."
            )
            raise ValueError(msg)

        # Map names to column indices. This could be a ``@cached_property``, but
        # it's not worth the overhead.
        self._n2k: dict[str, int]
        object.__setattr__(self, "_n2k", {name: i for i, name in enumerate(self.names)})

    def __getattr__(self, key: str) -> Any:
        """Get an attribute of the underlying array."""
        return getattr(self.array, key)

    def __len__(self) -> int:
        return len(self.array)

    def __repr__(self) -> str:
        """Get the representation."""
        return f"{type(self).__name__}({self.array!r}, names={self.names!r})"

    def __str__(self) -> str:
        """Get the string representation."""
        array = indent(repr(self.array), prefix="\t")[1:]
        return (
            "\n\t".join(
                (
                    f"{type(self).__name__}(",
                    f"names: {self.names!r}",
                    f"array: {array!s}",
                )
            )
            + "\n)"
        )

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
        self: Self, key: list[int] | Array, /  # Array[np.integer[Any]]
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
        | ArrayLike  # [np.integer[Any]]
        | tuple[int, ...]
        | tuple[str, ...]
        | tuple[slice, ...]
        | tuple[int | slice | str | tuple[int | str, ...], ...],
        /,
    ) -> Array | Self:
        out: Array | Self
        if isinstance(key, str):  # get a column
            out = cast("Array", self.array[:, self._n2k[key]])  # type: ignore[index] # noqa: E501

        elif isinstance(key, int):  # get a row
            out = type(self)(self.array[None, key], names=self.names)  # type: ignore[index] # noqa: E501

        elif isinstance(key, slice | list) or _is_arraylike(key):  # get rows
            out = type(self)(self.array[key], names=self.names)  # type: ignore[index]

        elif isinstance(key, tuple) and len(key) >= _LEN_INDEXING_TUPLE:
            if _all_strs(key):  # multiple columns
                names = key
                key = (slice(None), tuple(self._n2k[k] for k in key))
                out = type(self)(self.array[key], names=names)  # type: ignore[index]
            elif isinstance(key[1], str):
                key = (key[0], self._n2k[key[1]], *key[2:])
                out = cast("Array", self.array[key])  # type: ignore[index]
            elif isinstance(key[1], tuple):
                key = (
                    key[0],
                    tuple(self._n2k[k] if isinstance(k, str) else k for k in key[1]),
                    *key[2:],
                )
                out = cast("Array", self.array[key])  # type: ignore[index]
            else:
                out = cast("Array", self.array[key])  # type: ignore[index]

        else:
            out = cast("Array", self.array[key])  # type: ignore[index]

        return out

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

    def apply(self: Self, func: Callable[[Array], Array], /) -> Self:
        """Apply a function to the data.

        Parameters
        ----------
        func : Callable
            The function to apply. Must not change the names.

        Returns
        -------
        Data
            The transformed data.
        """
        return type(self)(func(self.array), names=self.names)

    # =========================================================================
    # I/O

    # TODO: instead interact with jax as a dictionary
    def __jax_array__(self) -> Array:
        """Convert to a JAX array."""
        return self.array

    def astype(self, fmt: type[ArrayT], /) -> Data[ArrayT]:
        """Convert the data to a different format.

        Parameters
        ----------
        fmt : type
            The format to convert to.

        Returns
        -------
        Data
            The converted data.
        """
        return cast("Data[ArrayT]", ASTYPE_REGISTRY[(type(self.array), fmt)](self))

    def to_format(self, fmt: type[ArrayT], /) -> ArrayT:
        """Convert the data to a different format.

        Parameters
        ----------
        fmt : type
            The format to convert to.

        Returns
        -------
        Data
            The converted data.
        """
        return cast("ArrayT", TO_FORMAT_REGISTRY[(type(self.array), fmt)](self))

    @classmethod
    def from_format(  # noqa: D417
        cls, data: Any, /, fmt: str, **kwargs: Any
    ) -> Data[Any]:
        """Convert the data from a different format.

        Parameters
        ----------
        data : Any, positional-only
            The data to convert.
        fmt : str
            The format to convert from.
        **kwargs : Any
            Additional keyword arguments to pass to the converter.

        Returns
        -------
        Data
            The converted data.
        """
        return FROM_FORMAT_REGISTRY[fmt](data, **kwargs)


###############################################################################
# HOOKS

ASTYPE_REGISTRY: dict[tuple[type, type], Callable[[Data[Any]], Data[ArrayLike]]] = {}
TO_FORMAT_REGISTRY: dict[tuple[type, type], Callable[[Data[Any]], ArrayLike]] = {}


class FromFormatCallable(Protocol):
    """Callable to convert data from a different format."""

    def __call__(self, obj: Any, /, **kwargs: Any) -> Data[Any]:  # noqa: D102
        ...


FROM_FORMAT_REGISTRY: dict[str, FromFormatCallable] = {}
