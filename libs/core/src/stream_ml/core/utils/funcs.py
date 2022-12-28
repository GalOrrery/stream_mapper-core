"""Stream Memberships Likelihood, with ML."""

from __future__ import annotations

# STDLIB
from typing import overload

# LOCAL
from stream_ml.core._typing import Array

__all__: list[str] = []


@overload
def map_to_range(x: Array, /, lower: float, upper: float) -> Array:
    ...


@overload
def map_to_range(x: Array, /, lower: Array, upper: Array) -> Array:
    ...


def map_to_range(x: Array, /, lower: float | Array, upper: float | Array) -> Array:
    """Map values from ``[min(x), max(x)]`` to range ``[lower, upper]``.

    Parameters
    ----------
    x : Array
        Values to map.
    lower : float | Array
        Lower bound.
    upper : float | Array
        Upper bound.

    Returns
    -------
    Array
        Mapped values from ``[min(x), max(x)]`` to range ``[lower, upper]``.

    Notes
    -----
    This will not work if lower or upper are not finite. In that case, a sigmoid
    function should be used beforehand. There are subtle differences between a
    scaled-sigmoid and ``map_to_range``(sigmoid). The former maps (-inf, inf) to
    (lower, upper), while the latter maps (min, max) to (lower, upper). As min,
    max approach infinities the two functions become more similar.
    """
    # Steps: (1) map to [0, 1], (2) map to [lower, upper]
    out: Array = ((x - x.min()) / (x.max() - x.min())) * (upper - lower) + lower
    return out
