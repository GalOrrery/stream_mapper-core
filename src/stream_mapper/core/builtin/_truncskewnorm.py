"""Gaussian stream model."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream_mapper.core.builtin._skewnorm import SkewNormal
from stream_mapper.core.builtin._stats.trunc_skewnorm import logpdf
from stream_mapper.core.builtin._utils import WhereRequiredError
from stream_mapper.core.typing import Array, NNModel
from stream_mapper.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_mapper.core import Data, Params


@dataclass
class TruncatedSkewNormal(SkewNormal[Array, NNModel]):
    """Truncated Skew-Normal."""

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
            idx = where[self.coord_names].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            idx = self.xp.ones((len(data), self.ndim), dtype=bool)
            # This has shape (N,F) so will broadcast correctly.

        cns, cens = self.coord_names, self.coord_err_names
        x = data[cns].array

        a, b = self.xp.asarray([self.coord_bounds[k] for k in cns]).T[:, None, :]
        mu = self._stack_param(mpars, "mu", cns)[idx]
        ln_s = self._stack_param(mpars, "ln-sigma", cns)[idx]
        skew = self._stack_param(mpars, "skew", cns)[idx]
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
