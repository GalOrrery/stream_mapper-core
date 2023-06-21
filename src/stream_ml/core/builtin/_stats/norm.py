from __future__ import annotations

__all__: list[str] = []

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream_ml.core.typing import Array, ArrayNamespace


sqrt2 = math.sqrt(2)
log2 = math.log(2)
log2pi = math.log(2 * math.pi)


def logpdf(x: Array, loc: Array, sigma: Array, *, xp: ArrayNamespace[Array]) -> Array:
    return -(((x - loc) / sigma) ** 2) / 2 - xp.log(sigma) - 0.5 * log2pi


def cdf(
    x: Array,
    loc: Array | float,
    sigma: Array | float,
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    return xp.special.erfc((loc - x) / sigma / sqrt2) / 2


def logcdf(
    x: Array | float, loc: Array, sigma: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    return xp.log(xp.special.erfc((loc - x) / sigma / sqrt2)) - log2


# ============================================================================


def logpdf_gaussian_errors(
    x: Array,
    /,
    loc: Array,
    sigma: Array,
    sigma_o: Array,
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    """Univariate log-pdf of a convolution of a uniform and a Gaussian."""
    return logpdf(x, loc, xp.sqrt(sigma**2 + sigma_o**2), xp=xp)
