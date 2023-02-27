"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from math import inf
from typing import TYPE_CHECKING

from stream_ml.core.prior.base import PriorBase
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.typing import Array, ArrayNamespace
from stream_ml.core.utils.compat import array_at
from stream_ml.core.utils.funcs import within_bounds

if TYPE_CHECKING:
    from stream_ml.core.api import Model
    from stream_ml.core.data import Data
    from stream_ml.core.params.core import Params
    from stream_ml.core.params.names import FlatParamName
    from stream_ml.core.typing import NNModel

__all__: list[str] = []


@dataclass(frozen=True)
class HardThreshold(PriorBase[Array]):
    """Threshold prior.

    Parameters
    ----------
    threshold : float, optional
        The threshold, by default 0.005

    lower : float, optional keyword-only
        The lower bound in the domain of the prior, by default `-inf`.
    upper : float, optional keyword-only
        The upper bound in the domain of the prior, by default `inf`.

    param_name : FlatParamName, optional keyword-only
        The name of the parameter to apply the prior to, by default
        `WEIGHT_NAME`.
    coord_name : str, optional keyword-only
        The name of the coordinate over which the parameter varies, by default
        `"phi1"`.
    """

    threshold: float = 5e-3
    _: KW_ONLY
    param_name: FlatParamName = (WEIGHT_NAME,)
    coord_name: str = "phi1"
    lower: float = -inf
    upper: float = inf

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array, NNModel],
        current_lnpdf: Array | None = None,
        /,
        *,
        xp: ArrayNamespace[Array],
    ) -> Array | float:
        """Evaluate the logpdf.

        This log-pdf is added to the current logpdf. So if you want to set the
        logpdf to a specific value, you can uses the `current_lnpdf` to set the
        output value such that ``current_lnpdf + logpdf = <want>``.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], position-only
            The data for which evaluate the prior.
        model : Model, position-only
            The model for which evaluate the prior.
        current_lnpdf : Array | None, optional position-only
            The current logpdf, by default `None`. This is useful for setting
            the additive log-pdf to a specific value.

        xp : ArrayNamespace[Array], keyword-only
            The array namespace.

        Returns
        -------
        Array
            The logpdf.
        """
        lnp = xp.zeros_like(mpars[self.param_name])
        where = within_bounds(data[self.coord_name], self.lower, self.upper) & (
            mpars[self.param_name] < self.threshold
        )
        return array_at(lnp, where).set(-xp.inf)

    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel]
    ) -> Array:
        """Evaluate the forward step in the prior.

        Parameters
        ----------
        pred : Array, position-only
            The input to evaluate the prior at.
        data : Data[Array], position-only
            The data to evaluate the prior at.
        model : `~stream_ml.core.Model`, position-only
            The model to evaluate the prior at.

        Returns
        -------
        Array
        """
        i = model.param_names.flats.index(self.param_name)
        where = within_bounds(
            data[self.coord_name].flatten(), self.lower, self.upper
        ) & (pred[:, i] <= self.threshold)
        return array_at(pred, (where, i), inplace=False).set(0)
