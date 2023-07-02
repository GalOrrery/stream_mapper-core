"""Gaussian stream model."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from stream_ml.core.builtin._skewnorm import SkewNormal
from stream_ml.core.builtin._stats.trunc_skewnorm import logpdf
from stream_ml.core.builtin._utils import WhereRequiredError
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_ml.core._data import Data
    from stream_ml.core.params import Params


@dataclass
class TruncatedSkewNormal(SkewNormal[Array, NNModel]):
    r"""Truncated Skew-Normal."""

    _: KW_ONLY
    require_where: bool = False

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of the distribution.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # 'where' is used to indicate which data points are available. If
        # 'where' is not provided, then all data points are assumed to be
        # available.
        if where is not None:
            idx = where[tuple(self.coord_bounds.keys())].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            idx = self.xp.ones((len(data)), dtype=bool)
            # This has shape (N,) so will broadcast correctly.

        cns, cens = self.coord_names, self.coord_err_names
        x = data[cns].array

        a, b = self.xp.asarray([self.coord_bounds[k] for k in cns]).T[:, None, :]
        mu = self.xp.stack(tuple(mpars[(k, "mu")] for k in cns), 1)[idx]
        ln_s = self.xp.stack(tuple(mpars[(k, "ln-sigma")] for k in cns), 1)[idx]
        skew = self.xp.stack(tuple(mpars[(k, "skew")] for k in cns), 1)[idx]
        if cens is not None:
            # it's fine if sigma_o is 0
            sigma_o = data[cens].array[idx]
            ln_s = self.xp.logaddexp(2 * ln_s, 2 * self.xp.log(sigma_o)) / 2
            skew = self.xp.sqrt(
                skew**2 / (1 + (sigma_o / self.xp.exp(ln_s)) ** 2 * (1 + skew**2))
            )

        _0 = self.xp.zeros_like(x)
        value = logpdf(
            x[idx], loc=mu, ln_sigma=ln_s, skew=skew, a=_0 + a, b=_0 + b, xp=self.xp
        )

        lnliks = self.xp.full_like(x, 0)  # missing data is ignored
        lnliks = array_at(lnliks, idx).set(value)
        return lnliks.sum(1)  # (N,)
