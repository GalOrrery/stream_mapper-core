"""Gaussian distribution functions."""

from __future__ import annotations

__all__ = ("logpdf", "cdf", "logcdf")

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream_ml.core.typing import Array, ArrayNamespace


sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2 * math.pi)
log2 = math.log(2)
log2pi = math.log(2 * math.pi)
logsqrt2pi = log2pi / 2


def pdf(x: Array, loc: Array, ln_sigma: Array, *, xp: ArrayNamespace[Array]) -> Array:
    """PDF of a Gaussian distribution.

    Parameters
    ----------
    x : (N,) | (N, F) array
        The input array.
    loc : (N,) | (N, F) array
        The location parameter.
    ln_sigma : (N,) | (N, F) array
        The log-scale parameter.

    xp : array namespace, keyword-only
        The array namespace to use.

    Returns
    -------
    array
    """
    sigma = xp.exp(ln_sigma)
    return xp.exp(-(((x - loc) / sigma) ** 2) / 2) / (sqrt2pi * sigma)


def logpdf(
    x: Array, loc: Array, ln_sigma: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    """Log-PDF of a Gaussian distribution.

    Parameters
    ----------
    x : (N,) | (N, F) array
        The input array.
    loc : (N,) | (N, F) array
        The location parameter.
    ln_sigma : (N,) | (N, F) array
        The log-scale parameter.

    xp : array namespace, keyword-only
        The array namespace to use.

    Returns
    -------
    array
    """
    return -0.5 * ((x - loc) / xp.exp(ln_sigma)) ** 2 - ln_sigma - logsqrt2pi


def cdf(
    x: Array,
    loc: Array | float,
    ln_sigma: Array,
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

    xp : array namespace, keyword-only
        The array namespace to use.

    Returns
    -------
    array
    """
    return xp.special.erfc((loc - x) / xp.exp(ln_sigma) / sqrt2) / 2


def logcdf(
    x: Array | float, loc: Array, ln_sigma: Array, *, xp: ArrayNamespace[Array]
) -> Array:
    """Log-CDF of a Gaussian distribution.

    Parameters
    ----------
    x : (N,) | (N, F) array
        The input array.
    loc : (N,) | (N, F) array
        The location parameter.
    ln_sigma : (N,) | (N, F) array
        The log-scale parameter.

    xp : array namespace, keyword-only
        The array namespace to use.

    Returns
    -------
    array
    """
    return xp.log(xp.special.erfc((loc - x) / xp.exp(ln_sigma) / sqrt2)) - log2
