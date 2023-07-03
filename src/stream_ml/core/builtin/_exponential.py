"""Built-in background models."""

from __future__ import annotations

from stream_ml.core.utils.compat import array_at

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from stream_ml.core._core.base import ModelBase
from stream_ml.core.builtin._stats.exponential import logpdf as exponential_logpdf
from stream_ml.core.builtin._utils import WhereRequiredError
from stream_ml.core.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core._data import Data
    from stream_ml.core.params import Params


@dataclass
class Exponential(ModelBase[Array, NNModel]):
    r"""1D Exponential.

    In each dimension the background is an exponential distribution between
    points ``a`` and ``b``. The rate parameter is ``m``.

    The non-zero portion of the PDF, where :math:`a < x < b` is

    .. math::

        f(x) = \frac{m * e^{-m * (x -a)}}{1 - e^{-m * (b - a)}}

    However, we use the order-3 Taylor expansion of the exponential function
    around m=0, to avoid the m=0 indeterminancy.

    .. math::

        f(x) =   \frac{1}{b-a}
               + m * (0.5 - \frac{x-a}{b-a})
               + \frac{m^2}{2} * (\frac{b-a}{6} - (x-a) + \frac{(x-a)^2}{b-a})
               + \frac{m^3}{12(b-a)} (2(x-a)-(b-a))(x-a)(b-x)
    """

    _: KW_ONLY
    require_where: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        # Pre-compute the associated constant factors
        _a = [a for k, (a, _) in self.coord_bounds.items() if k in self.params]
        _b = [b for k, (_, b) in self.coord_bounds.items() if k in self.params]

        self._a = self.xp.asarray(_a)[None, :]  # ([N], F)
        self._b = self.xp.asarray(_b)[None, :]  # ([N], F)

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : (N, F) Data[Array]
            Labelled data.

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        (N,) Array
        """
        # 'where' is used to indicate which data points are available. If
        # 'where' is not provided, then all data points are assumed to be
        # available.
        if where is not None:
            idx = where[tuple(self.coord_bounds.keys())].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            idx = self.xp.ones((len(data), len(self.coord_names)), dtype=bool)
            # This has shape (N,F) so will broadcast correctly.

        x = data[self.coord_names].array  # (N, F)
        # Get the slope from `mpars` we check param names to see if the
        # slope is a parameter. If it is not, then we assume it is 0.
        # When the slope is 0, the log-likelihood reduces to a Uniform.
        ms = self.xp.stack(tuple(mpars[(k, "slope")] for k in self.coord_names), 1)[idx]

        _0 = self.xp.zeros_like(x)
        # the distribution is not affected by the errors!
        # if self.coord_err_names is not None: pass

        value = exponential_logpdf(
            x[idx], ms, (_0 + self._a)[idx], (_0 + self._b)[idx], xp=self.xp
        )

        lnliks = self.xp.full_like(x, 0)  # missing data is ignored
        lnliks = array_at(lnliks, idx).set(value)

        return lnliks.sum(-1)
