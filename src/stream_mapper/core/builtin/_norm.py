"""Gaussian model."""

from __future__ import annotations

from stream_mapper.core.utils.compat import array_at

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream_mapper.core._core.base import ModelBase
from stream_mapper.core.builtin._stats.norm import logpdf as norm_logpdf
from stream_mapper.core.builtin._utils import WhereRequiredError
from stream_mapper.core.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_mapper.core import Data, Params


@dataclass(repr=False)
class Normal(ModelBase[Array, NNModel]):
    r"""Univariate Gaussian.

    You probably want to use the :class:`stream_mapper.core.builtin.TruncatedNormal`
    model instead.

    In each dimension the background is a normal distribution:

    .. math::

        f(x) = \frac{\exp{-((x - \mu)/\sigma)^2 / 2}}{\sqrt{2\pi} \sigma}

    The model parameters are:

    - "mu" : mean
    - "ln-sigma" : log-standard deviation

    Examples
    --------
    .. code-block:: python

        model = Normal(
            ...,
            coord_names=("x", "y"),  # independent coordinates
            coord_bounds={"x": (0, 1), "y": (1, 2)},
            params=ModelParameters(
                {
                    "x": {"slope": ModelParameter(...)},
                    "y": {"slope": ModelParameter(...)},
                }
            ),
        )
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Check that for every `coord_name` there is a parameter.
        if missing := set(self.coord_names) - set(self.params):
            msg = f"Missing parameter for coordinate(s) {missing}"
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
            idx = where[self.coord_names].array
        elif self.require_where:
            raise WhereRequiredError
        else:
            idx = self.xp.ones((len(data), self.ndim), dtype=bool)
            # This has shape (N,F) so will broadcast correctly.

        cns, cens = self.coord_names, self.coord_err_names
        x = data[cns].array

        mu = self._stack_param(mpars, "mu", cns)[idx]
        ln_s = self._stack_param(mpars, "ln-sigma", cns)[idx]
        if cens is not None:
            # it's fine if sigma_o is 0
            sigma_o = data[cens].array[idx]
            ln_s = self.xp.logaddexp(2 * ln_s, 2 * self.xp.log(sigma_o)) / 2

        value = norm_logpdf(x[idx], loc=mu, ln_sigma=ln_s, xp=self.xp)

        lnliks = self.xp.full_like(x, 0)  # missing data is ignored
        lnliks = array_at(lnliks, idx).set(value)
        return lnliks.sum(1)  # (N,F) -> (N,)
