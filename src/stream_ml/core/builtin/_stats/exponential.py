from __future__ import annotations

__all__: list[str] = []

from math import inf
from typing import TYPE_CHECKING

from stream_ml.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_ml.core.typing import Array, ArrayNamespace


def logpdf(
    x: Array,
    m: Array,
    a: Array | float,
    b: Array | float,
    *,
    xp: ArrayNamespace[Array],
    nil: Array | float = -inf,
) -> Array:
    out = xp.full_like(x, nil)
    mask = (a <= x) & (x <= b)
    lpdf = xp.log(m / xp.expm1(m * (b - a))) + (m * (b - x))
    return array_at(out, mask).set(lpdf[mask])


# ============================================================================


def logpdf_gaussian_errors(  # noqa: PLR0913
    x: Array,
    /,
    m: Array,
    a: Array,
    b: Array,
    sigma_o: Array,
    *,
    xp: ArrayNamespace[Array],
    nil: float = -inf,
) -> Array:
    """Log-pdf of an exponential distribution convolved with a Gaussian."""
    return logpdf(x, m, a, b, xp=xp, nil=nil) + (m**2 * sigma_o**2 / 2)
