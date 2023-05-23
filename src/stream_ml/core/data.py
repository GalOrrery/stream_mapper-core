"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from textwrap import indent
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Protocol,
    TypeAlias,
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

    ArrayLikeT = TypeVar("ArrayLikeT", bound=ArrayLike)

    KeyT: TypeAlias = int | list[Any] | slice | tuple[int | list[Any] | slice, ...]


#####################################################################
# PARAMETERS


_LEN_IDX_TUPLE: Final = 1


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
        self,
        key: tuple[
            int
            | str
            | slice
            | tuple[int, ...]
            | tuple[str, ...]
            | tuple[slice, ...]
            | tuple[int | str | slice, ...],
            ...,
        ],
        /,
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
        | tuple[
            int
            | str
            | slice
            | tuple[int, ...]
            | tuple[str, ...]
            | tuple[slice, ...]
            | tuple[int | str | slice, ...],
            ...,
        ],
        /,
    ) -> Array | Self:
        """Get an item or items from the data.

        Parameters
        ----------
        key : Any
            The key.

        Returns
        -------
        Array | Self

        Examples
        --------
        >>> data = Data(np.arange(12).reshape(3, 4), names=("a", "b", "c", "d"))

        Normal indexing:

        >>> data[0]  # get a row
        Data(array([[0, 1, 2, 3]]), names=('a', 'b', 'c', 'd'))

        >>> data[0:2]  # get a slice of rows
        Data(array([[0, 1, 2, 3],
                [4, 5, 6, 7]]), names=('a', 'b', 'c', 'd'))

        >>> data[[0, 2]]  # get rows
        Data(array([[0, 1, 2, 3],
                [8, 9, 10, 11]]), names=('a', 'b', 'c', 'd'))

        >>> data[np.array([0, 1], dtype=int)]  # get rows
        Data(array([[0, 1, 2, 3],
                [4, 5, 6, 7]]), names=('a', 'b', 'c', 'd'))

        >>> data[:, 0]  # get a column
        array([0, 4, 8])

        >>> data[:, 0:2]  # get a slice of columns
        Data(array([[0, 1],
                [4, 5],
                [8, 9]]), names=('a', 'b'))

        >>> data[:, [0, 2]]  # get columns
        Data(array([[0, 1],
                [4, 5],
                [8, 9]]), names=('a', 'b'))

        >>> data[:, np.array([0, 1], dtype=int)]  # get columns
        Data(array([[0, 1],
                [4, 5],
                [8, 9]]), names=('a', 'b'))

        >>> data[0, 0]  # get an element
        0

        >>> data[0, 0:2]  # get columns of a row
        Data(array([0, 1]), names=('a', 'b'))

        >>> data[0, [0, 2]]  # get columns of a row
        Data(array([0, 2]), names=('a', 'c'))

        >>> data[0, np.array([0, 1], dtype=int)]  # get columns of a row
        Data(array([0, 1]), names=('a', 'b'))

        >>> data[[0, 2], :]  # get rows and columns
        Data(array([[0, 1, 2, 3],
                [8, 9, 10, 11]]), names=('a', 'b', 'c', 'd'))

        Key indexing:

        >>> data["a"]  # get a column
        array([0, 4, 8])

        >>> data[("a",)]  # get columns
        Data(array([[0],
                [4],
                [8]]), names=('a',))

        >>> data["a", "b"]  # get columns
        Data(array([[0, 1],
                [4, 5],
                [8, 9]]), names=('a', 'b'))

        >>> data[:, ("a", "b")]  # get columns
        Data(array([[0, 1],
                [4, 5],
                [8, 9]]), names=('a', 'b'))

        >>> data[:, ("a", "b"), None]  # get columns
        Data(array([[[0],
                [1]],

                [[4],
                [5]],

                [[8],
                [9]]]), names=('a', 'b'))
        """
        if isinstance(key, tuple):
            if _all_strs(key):  # multiple columns
                return type(self)(
                    self._array[:, [self._n2k[k] for k in key]],  # type: ignore[index]
                    names=key,
                )
            elif len(key) > _LEN_IDX_TUPLE and isinstance(key[1], int):  # get column
                return cast("Array", self._array[key])  # type: ignore[index]

            array: Array = self._array[(key[0], _parse_key_elt(key[1], self._n2k), *key[2:])]  # type: ignore[index]  # noqa: E501
            if array.ndim == 1:
                array = array[None, :]

            if isinstance(key[1], slice):
                names = self.names[key[1]]
            elif isinstance(key[1], int):
                names = (self.names[key[1]],)
            elif isinstance(key[1], str):
                names = (key[1],)
            else:
                names = tuple(
                    (i if isinstance(i, str) else str(self._names[i])) for i in key[1]
                )

            return type(self)(array, names=names)

        elif isinstance(key, int):  # get a row
            return type(self)(self._array[None, key, :], names=self._names)  # type: ignore[index]  # noqa: E501
        elif isinstance(key, str):  # get a column
            return cast("Array", self._array[:, self._n2k[key]])  # type: ignore[index]
        return type(self)(self._array[key], names=self._names)  # type: ignore[index]

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
    # I/O

    # TODO: instead interact with jax as a dictionary
    def __jax_array__(self) -> Array:
        """Convert to a JAX array."""
        return self.array

    def astype(self, fmt: type[ArrayLikeT], /) -> Data[ArrayLikeT]:
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
        return cast("Data[ArrayLikeT]", ASTYPE_REGISTRY[(type(self.array), fmt)](self))

    def to_format(self, fmt: type[ArrayLikeT], /) -> ArrayLikeT:
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
        return cast("ArrayLikeT", TO_FORMAT_REGISTRY[(type(self.array), fmt)](self))

    @classmethod
    def from_format(cls, data: Any, /, fmt: str, **kwargs: Any) -> Data[Any]:
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


def _parse_key_elt(key: Any, n2k: dict[str, int]) -> KeyT:
    """Parse a key.

    Parameters
    ----------
    key : Any
        The key to parse.
    n2k : dict[str, int]
        The name to index mapping.

    Returns
    -------
    int | slice | tuple[int | slice, ...]
    """
    if isinstance(key, int):
        return key
    elif isinstance(key, str):
        return n2k[key]
    elif isinstance(key, list) or _is_arraylike(key):
        return [n2k[k] if isinstance(k, str) else k for k in key]
    elif isinstance(key, slice):
        return slice(
            n2k[key.start] if isinstance(key.start, str) else key.start,
            n2k[key.stop] if isinstance(key.stop, str) else key.stop,
            key.step,
        )
    elif isinstance(key, tuple):
        return tuple(n2k[k] for k in key) if _all_strs(key) else key
    else:
        msg = f"Invalid key type: {type(key)}"
        raise TypeError(msg)


###############################################################################
# HOOKS

ASTYPE_REGISTRY: dict[tuple[type, type], Callable[[Data[Any]], Data[ArrayLike]]] = {}
TO_FORMAT_REGISTRY: dict[tuple[type, type], Callable[[Data[Any]], ArrayLike]] = {}


class FromFormatCallable(Protocol):
    """Callable to convert data from a different format."""

    def __call__(self, obj: Any, /, **kwargs: Any) -> Data[Any]:  # noqa: D102
        ...


FROM_FORMAT_REGISTRY: dict[str, FromFormatCallable] = {}
