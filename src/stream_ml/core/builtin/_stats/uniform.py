from __future__ import annotations

__all__: list[str] = []

from math import inf, log
from typing import TYPE_CHECKING

from stream_ml.core.builtin._stats.norm import cdf as norm_cdf
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
    # the log-pdf is -log(b - a) for x in [a, b], and -inf otherwise
    return array_at(out, mask).set(-xp.log(xp.zeros_like(out) + b - a)[mask])


# ============================================================================


def logpdf_gaussian_errors(
    x: Array,
    /,
    a: Array,
    b: Array,
    sigma_o: Array,
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    """Log-pdf of a uniform distribution convolved with a Gaussian."""
    # TODO: ensure broadcasting works
    return xp.log(  # yes, a - b
        norm_cdf(x, a, sigma_o, xp=xp) - norm_cdf(x, b, sigma_o, xp=xp)
    ) + logpdf(x, a=a, b=b, xp=xp)
