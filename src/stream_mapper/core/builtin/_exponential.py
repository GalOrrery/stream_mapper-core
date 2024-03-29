"""Exponential model."""

from __future__ import annotations

from stream_mapper.core.utils.compat import array_at

__all__: tuple[str, ...] = ()

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

from stream_mapper.core._core.base import ModelBase
from stream_mapper.core.builtin._stats.exponential import logpdf as exponential_logpdf
from stream_mapper.core.builtin._utils import WhereRequiredError
from stream_mapper.core.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_mapper.core import Data, Params


@dataclass(repr=False)
class Exponential(ModelBase[Array, NNModel]):
    r"""(Truncated) Univariate Exponential model.

    In each dimension the background is an exponential distribution between
    points ``a`` and ``b``. The rate parameter is ``m``.

    The non-zero portion of the PDF, where :math:`a < x < b` is

    .. math::

        f(x) = \frac{m * e^{-m * (x - a)}}{1 - e^{-m * (b - a)}}

    This form is numerically unstable as :math:`m \to 0`. The
    distribution mathematically reduces to a Uniform distribution.
    This is handled numerically.

    The model parameters are:

    - "slope" : rate parameter

    Notes
    -----
    This model supports many independent coordinates, allowing it to support
    a multivariate background. However, the background is still univariate
    in each dimension.

    Examples
    --------
    .. code-block:: python

        model = Exponential(
            ...,
            coord_names=("x", "y"),  # independent coordinates
            coord_bounds={"x": (0, 1), "y": (1, 2)},  # bounds for each coordinate
            params=ModelParameters(
                {
                    "x": {"slope": ModelParameter(...)},
                    "y": {"slope": ModelParameter(...)},
                }
            ),
        )
    """

    _: KW_ONLY
    m_eps: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()

        # Check that for every `coord_name` there is a parameter.
        if missing := set(self.coord_names) - set(self.params):
            msg = f"Missing parameter for coordinate(s) {missing}"
            raise ValueError(msg)

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
            idx = where[self.coord_names].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            idx = self.xp.ones((len(data), self.ndim), dtype=bool)
            # This has shape (N,F) so will broadcast correctly.

        x = data[self.coord_names].array  # (N, F)
        # Get the slope from `mpars` we check param names to see if the
        # slope is a parameter. If it is not, then we assume it is 0.
        # When the slope is 0, the log-likelihood reduces to a Uniform.
        ms = self._stack_param(mpars, "slope", self.coord_names)[idx]

        # the distribution is not affected by the errors!
        # if self.coord_err_names is not None: pass
        _0 = self.xp.zeros_like(x)
        value = exponential_logpdf(
            x[idx],
            m=ms,
            a=(_0 + self._a)[idx],
            b=(_0 + self._b)[idx],
            xp=self.xp,
            nil=-self.xp.inf,
            m_eps=self.m_eps,
        )
        # missing data has a log-likelihood of 0
        lnliks = self.xp.full_like(x, 0)  # missing data is ignored
        lnliks = array_at(lnliks, idx).set(value)

        return lnliks.sum(-1)  # sum over features (N,F) -> (N,)
