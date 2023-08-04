"""Built-in background models."""

from __future__ import annotations

from stream_ml.core.utils.compat import array_at

__all__: list[str] = []

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stream_ml.core._core.base import ModelBase
from stream_ml.core._core.field import NNField
from stream_ml.core.builtin._stats.uniform import logpdf as uniform_logpdf
from stream_ml.core.builtin._utils import WhereRequiredError
from stream_ml.core.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core._data import Data
    from stream_ml.core.params import Params


@dataclass
class Uniform(ModelBase[Array, NNModel]):
    """Uniform background model."""

    net: NNField[NNModel, None] = NNField(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        super().__post_init__()

        # net must be None:
        if self.net is not None:
            msg = "net must be None"
            raise ValueError(msg)

        # Pre-compute the log-difference, shape (1, F)
        # First need to check that the bound are finite.
        ab_ = self.xp.asarray(tuple(self.coord_bounds.values()))
        if not self.xp.isfinite(ab_).all():
            msg = "a bound of a coordinate is not finite"
            raise ValueError(msg)
        self._a: Array
        self._b: Array
        object.__setattr__(self, "_a", ab_[None, :, 0])  # ([N], F)
        object.__setattr__(self, "_b", ab_[None, :, 1])  # ([N], F)

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Any,
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Labelled data.

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

        x = data[self.coord_names].array  # (N, F)
        # Get the slope from `mpars` we check param names to see if the
        # slope is a parameter. If it is not, then we assume it is 0.
        # When the slope is 0, the log-likelihood reduces to a Uniform.

        _0 = self.xp.zeros_like(x)
        # the distribution is not affected by the errors!
        # if self.coord_err_names is not None: pass

        value = uniform_logpdf(
            x[idx], a=(_0 + self._a)[idx], b=(_0 + self._b)[idx], xp=self.xp
        )

        lnliks = self.xp.full_like(x, 0)  # missing data will be ignored
        lnliks = array_at(lnliks, idx).set(value)

        return lnliks.sum(1)
