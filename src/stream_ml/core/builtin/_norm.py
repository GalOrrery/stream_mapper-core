"""Gaussian model."""

from __future__ import annotations

from stream_ml.core.utils.compat import array_at

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from stream_ml.core._core.base import ModelBase
from stream_ml.core.builtin._stats.norm import logpdf as norm_logpdf
from stream_ml.core.builtin._utils import WhereRequiredError
from stream_ml.core.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params


@dataclass
class Normal(ModelBase[Array, NNModel]):
    r"""1D Gaussian.

    :math:`\mathcal{N}(weight, \mu, \ln\sigma)(\phi1)`

    Notes
    -----
    The model parameters are:
    - "mu" : mean
    - "ln-sigma" : log-standard deviation
    """

    _: KW_ONLY
    require_where: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        # Check that for every `coord_name` there is a parameter.
        for k in self.coord_names:
            if k not in self.params:
                msg = f"Missing parameter for coordinate {k}"
                raise ValueError(msg)

        # Check that `coord_err_name` <-> coord_name.
        if self.coord_err_names is not None and len(self.coord_names) != len(
            self.coord_err_names
        ):
            msg = "Number of coordinates and coordinate errors must match"
            raise ValueError(msg)

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

        mu = self.xp.stack(tuple(mpars[(k, "mu")] for k in cns), 1)[idx]
        ln_s = self.xp.stack(tuple(mpars[(k, "ln-sigma")] for k in cns), 1)[idx]
        if cens is not None:
            # it's fine if sigma_o is 0
            sigma_o = data[cens].array[idx]
            ln_s = self.xp.logaddexp(2 * ln_s, 2 * self.xp.log(sigma_o)) / 2

        value = norm_logpdf(x[idx], loc=mu, ln_sigma=ln_s, xp=self.xp)

        lnliks = self.xp.full_like(x, 0)  # missing data is ignored
        lnliks = array_at(lnliks, idx).set(value)
        return lnliks.sum(1)  # (N,)
