"""Stream Memberships Likelihood, with ML."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from abc import abstractmethod
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Protocol

from stream_ml.core.typing import Array

if TYPE_CHECKING:
    from stream_ml.core.typing import ArrayNamespace


class ArrayAt(Protocol[Array]):
    """Array at index."""

    @abstractmethod
    def set(self, value: Array | float) -> Array:  # noqa: A003
        """Set the value at the index."""
        ...


@singledispatch
def array_at(array: Array, idx: Any, /, *, inplace: bool = True) -> ArrayAt[Array]:
    """Get the array at the index.

    This is to emulate the `jax.numpy.ndarray.at` method.

    Parameters
    ----------
    array : Array
        Array to get the value at the index.
    idx : Any
        Index to get the value at.

    inplace : bool, optional
        Whether to set the value in-place, by default `False`.

    Returns
    -------
    ArrayAt[Array]
        Setter.
    """
    raise NotImplementedError


@singledispatch
def get_namespace(array: Array, /) -> ArrayNamespace[Array]:
    """Get the namespace of the array.

    Parameters
    ----------
    array : Array
        Array to get the namespace of.

    Returns
    -------
    ArrayNamespace[Array]
        Namespace.
    """
    msg = f"unknown array type {type(array)}."
    raise NotImplementedError(msg)


@singledispatch
def copy(array: Array, /) -> Array:
    """Copy the array.

    Parameters
    ----------
    array : Array
        Array to copy.

    Returns
    -------
    Array
        Copied array.
    """
    msg = f"unknown array type {type(array)}."
    raise NotImplementedError(msg)
