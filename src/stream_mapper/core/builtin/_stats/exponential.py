"""Truncated exponential distribution."""

from __future__ import annotations

__all__ = ("logpdf",)

from math import inf, log
from typing import TYPE_CHECKING

from stream_mapper.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_mapper.core.typing import Array, ArrayNamespace


log2 = log(2.0)


def logpdf(  # noqa: PLR0913
    x: Array,
    m: Array,
    a: Array,
    b: Array,
    *,
    xp: ArrayNamespace[Array],
    nil: Array | float = -inf,
    m_eps: float = 1e-6,
) -> Array:
    """Log-pdf of a truncated exponential distribution.

    Parameters
    ----------
    x : (N,) array
        The input array.
    m : ([N],) array
        The scale parameter.
    a, b : (N,) array
        The lower and upper bounds of the distribution.
        The distribution is centered at ``b-a``.

    xp : array namespace, keyword-only
        The array namespace to use.
    nil : array or float, optional keyword-only
        The value to use for out-of-bounds values.
    m_eps : float, keyword-only
        The tolerance for ``m == 0``.

    Returns
    -------
    array
    """
    # Start with 0 probability
    out = xp.full_like(x, nil)
    sel = (a <= x) & (x <= b)

    # m == 0 => uniform distribution
    m0 = xp.abs(m) <= m_eps
    out = array_at(out, sel & m0).set(-xp.log(b - a)[sel & m0])

    # m != 0 => exponential
    # This avoids calculating anything that has log(~0) in it,
    # to avoid NaNs in the gradient.
    n0 = sel & ~m0
    lpdf = (
        xp.log(xp.abs(m[n0]))
        + (m * (b - x))[n0]
        - xp.log(xp.abs(xp.expm1((m * (b - a))[n0])))
    )
    return array_at(out, n0).set(lpdf)
