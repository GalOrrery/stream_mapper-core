from __future__ import annotations

__all__: tuple[str, ...] = ()

from math import inf
from typing import TYPE_CHECKING

from stream_ml.core.builtin._stats.skewnorm import (
    cdf as skewnorm_cdf,
    logpdf as skewnorm_logpdf,
)
from stream_ml.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_ml.core.typing import Array, ArrayNamespace


def _logpdf(  # noqa: PLR0913
    x: Array,
    /,
    loc: Array,
    ln_sigma: Array,
    skew: Array,
    a: Array,
    b: Array,
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    log_trunc = xp.log(
        skewnorm_cdf(b, loc=loc, ln_sigma=ln_sigma, skew=skew, xp=xp)
        - skewnorm_cdf(a, loc=loc, ln_sigma=ln_sigma, skew=skew, xp=xp)
    )
    return skewnorm_logpdf(x, loc=loc, ln_sigma=ln_sigma, skew=skew, xp=xp) - log_trunc


def logpdf(  # noqa: PLR0913
    x: Array,
    /,
    loc: Array,
    ln_sigma: Array,
    skew: Array,
    a: Array,
    b: Array,
    *,
    xp: ArrayNamespace[Array],
    nil: float = -inf,
) -> Array:
    out = xp.full_like(x, nil)
    sel = (a <= x) & (x <= b)
    ln_pdf = _logpdf(
        x[sel],
        loc=loc[sel],
        ln_sigma=ln_sigma[sel],
        skew=skew[sel],
        a=a[sel],
        b=b[sel],
        xp=xp,
    )
    return array_at(out, sel).set(ln_pdf)
