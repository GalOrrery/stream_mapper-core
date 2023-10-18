from __future__ import annotations

__all__ = ("logpdf",)

from math import inf
from typing import TYPE_CHECKING

from stream_ml.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_ml.core.typing import Array, ArrayNamespace


def logpdf(
    x: Array,
    /,
    a: Array,
    b: Array,
    *,
    nil: Array | float = -inf,
    xp: ArrayNamespace[Array],
) -> Array:
    """Log-pdf of a truncated uniform distribution.

    Parameters
    ----------
    x : (N,) | (N,F) Array, positional-only
        The data.
    a, b : (N,) | (N,F) Array
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
    sel = (a <= x) & (x <= b)
    # the log-pdf is -log(b - a) for x in [a, b], and -inf otherwise
    return array_at(out, sel).set(-xp.log(xp.zeros_like(out) + b - a)[sel])
