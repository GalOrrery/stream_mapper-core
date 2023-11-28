from __future__ import annotations

__all__ = ("logpdf",)

from math import inf
from typing import TYPE_CHECKING

from stream_mapper.core.builtin._stats.norm import (
    cdf as norm_cdf,
    logpdf as norm_logpdf,
)
from stream_mapper.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_mapper.core.typing import Array, ArrayNamespace


def _pdf_normalization(
    x: Array,
    /,
    loc: Array,
    ln_sigma: Array,
    a: Array,
    b: Array,
    *,
    xp: ArrayNamespace[Array],
) -> Array:
    return norm_cdf(b, loc=loc, ln_sigma=ln_sigma, xp=xp) - norm_cdf(
        a, loc=loc, ln_sigma=ln_sigma, xp=xp
    )


def logpdf(  # noqa: PLR0913
    x: Array,
    /,
    loc: Array,
    ln_sigma: Array,
    a: Array,
    b: Array,
    *,
    xp: ArrayNamespace[Array],
    nil: float = -inf,
) -> Array:
    out = xp.full_like(x, nil)
    sel = (a <= x) & (x <= b)

    ln_pdf = norm_logpdf(x[sel], loc=loc[sel], ln_sigma=ln_sigma[sel], xp=xp)
    ln_trunc = xp.log(
        _pdf_normalization(
            x[sel], loc=loc[sel], ln_sigma=ln_sigma[sel], a=a[sel], b=b[sel], xp=xp
        )
    )
    return array_at(out, sel).set(ln_pdf - ln_trunc)
