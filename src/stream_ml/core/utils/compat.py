"""Stream Memberships Likelihood, with ML."""

from __future__ import annotations

from abc import abstractmethod
from functools import singledispatch
from typing import Any, Protocol

from stream_ml.core.typing import Array

__all__: list[str] = []


@singledispatch
def array_at(array: Array, idx: Any) -> ArrayAt[Array]:
    """Get the array at the index.

    This is to emulate the `jax.numpy.ndarray.at` method.

    Parameters
    ----------
    array : Array
        Array to get the value at the index.
    idx : Any
        Index to get the value at.

    Returns
    -------
    ArrayAt[Array]
        Setter.
    """
    raise NotImplementedError


class ArrayAt(Protocol[Array]):
    """Array at index."""

    @abstractmethod
    def set(self, value: Array) -> Array:  # noqa: A003
        """Set the value at the index."""
        ...
