from __future__ import annotations

__all__ = ("logpdf", "cdf", "logcdf")

import math
from typing import TYPE_CHECKING

from stream_ml.core.builtin._stats.norm import (
    cdf as norm_cdf,
    pdf as norm_pdf,
)

if TYPE_CHECKING:
    from stream_ml.core.typing import Array, ArrayNamespace


sqrt2 = math.sqrt(2)


def _owens_t_integrand(x: Array, t: Array, *, xp: ArrayNamespace[Array]) -> Array:
    return xp.exp(-0.5 * x**2 * (1 + t**2)) / (1 + t**2)


def owens_t_approx(x: Array, a: Array, *, xp: ArrayNamespace[Array]) -> Array:
    # https://en.wikipedia.org/wiki/Owen%27s_T_function
    # TODO! faster approximation
    ts: Array = xp.linspace(0, 1, 1_000)[None, :] * a[:, None]  # (N, 1000)
    return (
        xp.sum(_owens_t_integrand(x[:, None], ts, xp=xp), axis=-1)
        * (ts[:, 1] - ts[:, 0])
        / (2 * xp.pi)
    )


# ============================================================================


def _norm_cdf(
    x: Array,
    loc: Array | float,
    ln_sigma: Array,
    skew: Array,
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    """CDF of a Gaussian distribution.

    Parameters
    ----------
    x : (N,) | (N, F) array
        The input array.
    loc : (N,) | (N, F) array
        The location parameter.
    ln_sigma : (N,) | (N, F) array
        The log-scale parameter.
    skew : (N,) | (N, F) array
        The skew parameter.

    xp : array namespace, keyword-only
        The array namespace to use.

    Returns
    -------
    array
    """
    return xp.special.erfc(skew * (loc - x) / xp.exp(ln_sigma) / sqrt2) / 2


def pdf(
    x: Array,
    /,
    loc: Array,
    ln_sigma: Array,
    skew: Array,
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    """PDF of a skew-normal distribution.

    Parameters
    ----------
    x : (N,) | (N, F) array, positional-only
        The input array.
    loc : (N,) | (N, F) array
        The location parameter.
    ln_sigma : (N,) | (N, F) array
        The log-scale parameter.
    skew : (N,) | (N, F) array
        The skew parameter.

    xp : array namespace, keyword-only
        The array namespace to use.

    Returns
    -------
    array
    """
    return (
        2
        * norm_pdf(x, loc=loc, ln_sigma=ln_sigma, xp=xp)
        * _norm_cdf(x, loc=loc, ln_sigma=ln_sigma, skew=skew, xp=xp)
    )


def logpdf(
    x: Array,
    /,
    loc: Array,
    ln_sigma: Array,
    skew: Array,
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    """Log-PDF of a skew-normal distribution.

    Parameters
    ----------
    x : (N,) | (N, F) array, positional-only
        The input array.
    loc : (N,) | (N, F) array
        The location parameter.
    ln_sigma : (N,) | (N, F) array
        The log-scale parameter.
    skew : (N,) | (N, F) array
        The skew parameter.

    xp : array namespace, keyword-only
        The array namespace to use.
    nil : array or float, optional keyword-only
        The value to use for out-of-bounds values.

    Returns
    -------
    array
    """
    # # https://en.wikipedia.org/wiki/Skew_normal_distribution
    # Normally, we would use the following:
    # (
    #     log2
    #     + norm_logpdf(x, loc=loc, ln_sigma=ln_sigma, xp=xp)
    #     + norm_logcdf(skew * x, loc=loc, ln_sigma=ln_sigma, xp=xp)
    # )
    # However, norm_logcdf is numerically unstable since it takes the logarithm
    # of `erfc`. Instead we take the logarithm of the product of the PDF and
    # CDF, which is equivalent to the above, but is more numerically stable.
    return xp.log(pdf(x, loc=loc, ln_sigma=ln_sigma, skew=skew, xp=xp))


def cdf(
    x: Array, /, loc: Array, ln_sigma: Array, skew: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    """CDF of a skew-normal distribution.

    Parameters
    ----------
    x : (N,) | (N, F) array, positional-only
        The input array.
    loc : (N,) | (N, F) array
        The location parameter.
    ln_sigma : (N,) | (N, F) array
        The log-scale parameter.
    skew : (N,) | (N, F) array
        The skew parameter.

    xp : array namespace, keyword-only
        The array namespace to use.

    Returns
    -------
    array
    """
    return norm_cdf(x, loc=loc, ln_sigma=ln_sigma, xp=xp) - 2 * owens_t_approx(
        (x - loc) / xp.exp(ln_sigma), skew, xp=xp
    )


def logcdf(
    x: Array, /, loc: Array, ln_sigma: Array, skew: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    """log-CDF of a skew-normal distribution.

    Parameters
    ----------
    x : (N,) | (N, F) array, positional-only
        The input array.
    loc : (N,) | (N, F) array
        The location parameter.
    ln_sigma : (N,) | (N, F) array
        The log-scale parameter.
    skew : (N,) | (N, F) array
        The skew parameter.

    xp : array namespace, keyword-only
        The array namespace to use.

    Returns
    -------
    array
    """
    return xp.log(cdf(x, loc=loc, ln_sigma=ln_sigma, skew=skew, xp=xp))
