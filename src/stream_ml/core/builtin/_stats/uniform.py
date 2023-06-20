from __future__ import annotations

__all__: list[str] = []

from math import inf, log
from typing import TYPE_CHECKING

from stream_ml.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_ml.core.typing import Array, ArrayNamespace


log2 = log(2)


def logpdf(
    x: Array,
    /,
    a: Array,
    b: Array,
    *,
    nil: Array | float = -inf,
    xp: ArrayNamespace[Array],
) -> Array:
    """Log-pdf of a uniform distribution.

    Parameters
    ----------
    x : (N, ...) Array, positional-only
        The data.
    a, b : Array
        The lower and upper bounds of the uniform distribution.

    nil : Array | float, keyword-only
        The value to return when the data is outside the bounds. Default is
        negative infinity.
    xp : ArrayNamespace[Array], keyword-only
        The array namespace.

    Returns
    -------
    Array
    """
    out = xp.full_like(x, nil)
    mask = (a <= x) & (x <= b)
    return array_at(out, mask).set(-xp.log(xp.zeros_like(mask) + b - a)[mask])
