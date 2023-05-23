"""Stream Memberships Likelihood, with ML."""

from __future__ import annotations

__all__: list[str] = []

from abc import abstractmethod
from functools import singledispatch
from typing import Any, Protocol

from stream_ml.core.typing import Array


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
