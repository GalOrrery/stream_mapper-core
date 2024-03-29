"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from functools import singledispatch
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from stream_mapper.core.typing import Array, ArrayNamespace

    V = TypeVar("V")


def get_prefixed_kwargs(prefix: str, kwargs: dict[str, V]) -> dict[str, V]:
    """Get the kwargs with a given prefix.

    Parameters
    ----------
    prefix : str
        Prefix.
    kwargs : dict[str, V]
        Keyword arguments.

    Returns
    -------
    dict[str, V]
    """
    p = prefix + "_" if not prefix.endswith("_") else prefix
    return {k.removeprefix(p): v for k, v in kwargs.items() if k.startswith(p)}


@singledispatch
def within_bounds(
    value: Array,
    /,
    lower_bound: Array | float | None,
    upper_bound: Array | float | None,
    *,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> Array:
    """Check if a value is within the given bounds.

    Parameters
    ----------
    value : ndarray
        Value to check.
    lower_bound, upper_bound : float | None
        Bounds to check against.
    lower_inclusive, upper_inclusive : bool, optional
        Whether to include the bounds in the check, by default `True`.

    Returns
    -------
    ndarray
        Boolean array indicating whether the value is within the bounds.
    """
    raise NotImplementedError


###############################################################################


def _broadcast_index(
    index: int | slice, axis: int, ndim: int
) -> tuple[int | slice, ...]:
    """Return multidimensional index of a single-dimensional index at specified axis.

    Parameters
    ----------
    index : int | slice
        The indexing object to be applied on axis 'axis'.
    axis : int
        The axis on which to apply the index
    ndim : int
        The number of axes.

    Returns
    -------
    tuple[int | slice, ...]
        'index' at 'axis' and ``slice(None)`` on all other axes.
    """
    return tuple(index if i == axis else slice(None) for i in range(ndim))


def _perform_on_axes(ndim: int, skip_axes: tuple[int, ...] = ()) -> tuple[int, ...]:
    """Return tuple of axis indices excluding some axes.

    Parameters
    ----------
    ndim : int
        The number of axes
    skip_axes : tuple[int, ...]
        The axes to skip.

    Returns
    -------
    tuple[int, ...]
        The axes to include
    """
    return tuple(i for i in range(ndim) if i not in skip_axes)


def pairwise_distance(
    x: Array, /, axis: int = 0, *, xp: ArrayNamespace[Array]
) -> Array:
    """Return the pairwise distance along some axis.

    Parameters
    ----------
    x : array, positional-only
        Data for which to compute the pairwise distance.
    axis : int
        Axis along which to compute the pairwise distance.
    xp : ArrayNamespace, optional keyword-only
        Array-api namespace.
    """
    _axis1 = _broadcast_index(slice(1, None), 0, x.ndim)
    _axis2 = _broadcast_index(slice(None, -1), 0, x.ndim)
    sumaxis = _perform_on_axes(x.ndim, skip_axes=(axis,))
    if not sumaxis:
        return xp.sqrt(xp.square(x[_axis1] - x[_axis2]))
    return xp.sqrt(xp.square(x[_axis1] - x[_axis2]).sum(sumaxis))
