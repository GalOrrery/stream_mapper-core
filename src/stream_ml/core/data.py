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

from stream_ml.core.typing import Array  # noqa: TCH001

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from stream_ml.core.typing import ArrayLike

    Self = TypeVar("Self", bound="Data[Array]")  # type: ignore[valid-type]

    ArrayT = TypeVar("ArrayT", bound=ArrayLike)
    T = TypeVar("T")


#####################################################################
# PARAMETERS


LEN_INDEXING_TUPLE: Final = 1


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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

        elif isinstance(key, (slice, list)) or _is_arraylike(key):  # get rows
            out = type(self)(self.array[key], names=self.names)  # type: ignore[index]

        elif isinstance(key, tuple) and len(key) >= LEN_INDEXING_TUPLE:
            if _all_strs(key):
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

        if isinstance(out, Data) and type(out.array) in DATA_HOOK:
            return cast("Self", DATA_HOOK[type(out.array)](out))
        elif type(out) in ARRAY_HOOK:
            return ARRAY_HOOK[type(out)](out, key=key)

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
    # I/O

    def __jax_array__(self) -> Array:
        """Convert to a JAX array."""
        return self.array

    def to_format(self, fmt: type[ArrayT], /) -> Data[ArrayT]:
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
        return cast("Data[ArrayT]", TO_FORMAT_REGISTRY[(type(self.array), fmt)](self))

    # TODO: a from_format method with registry.


###############################################################################
# HOOKS


class _ArrayHookCallable(Protocol):
    def __call__(self, array: Array, /, key: Any) -> Array:  # noqa: ANN001, N805
        ...


DATA_HOOK: dict[type, Callable[[Data[Array]], Data[Array]]] = {}
ARRAY_HOOK: dict[type, _ArrayHookCallable] = {}
TO_FORMAT_REGISTRY: dict[tuple[type, type], Callable[[Data[Any]], Data[ArrayLike]]] = {}


###############################################################################
# Alternate constructors


# TODO: this should be moved to a separate module, interfacing via ``from_format``.
def from_structured_array(array: NDArray[Any], /) -> Data[NDArray[Any]]:
    """Create a `Data` instance from a structured numpy array.

    Requires :mod:`numpy` to be installed.

    Parameters
    ----------
    array : ndarray
        The structured array.

    Returns
    -------
    Data
        The data instance.
    """
    from numpy.lib.recfunctions import structured_to_unstructured

    if not isinstance(array.dtype.names, tuple):
        msg = "The array must be structured."
        raise TypeError(msg)

    return Data(structured_to_unstructured(array), names=array.dtype.names)
