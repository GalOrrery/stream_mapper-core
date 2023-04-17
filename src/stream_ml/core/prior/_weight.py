"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from math import inf
from typing import TYPE_CHECKING

from stream_ml.core.params.scales.builtin import ParamScaler  # noqa: TCH001
from stream_ml.core.prior._base import PriorBase
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.typing import Array, ArrayNamespace
from stream_ml.core.utils.compat import array_at
from stream_ml.core.utils.funcs import within_bounds

if TYPE_CHECKING:
    from stream_ml.core._api import Model
    from stream_ml.core.data import Data
    from stream_ml.core.params._core import Params
    from stream_ml.core.typing import NNModel

__all__: list[str] = []


@dataclass(frozen=True)
class HardThreshold(PriorBase[Array]):
    """Threshold prior.

    Parameters
    ----------
    threshold : float
        The threshold. Everything below this (not inclusive) is set to
        ``set_to``.
    set_to : float
        What to set

    lower : float, optional keyword-only
        The lower bound in the domain of the prior, by default `-inf`.
    upper : float, optional keyword-only
        The upper bound in the domain of the prior, by default `inf`.

    coord_name : str, optional keyword-only
        The name of the coordinate over which the parameter varies, by default
        `"phi1"`.
    """

    threshold: float = 5e-3
    set_to: float = 1e-10
    _: KW_ONLY
    coord_name: str = "phi1"
    lower: float = -inf
    upper: float = inf

    scaler: ParamScaler[Array]

    def __post_init__(self) -> None:
        """Post-init."""
        super().__post_init__()
        if self.lower > self.upper:
            msg = f"lower > upper: {self.lower} > {self.upper}"
            raise ValueError(msg)

        self.scaled_bounds: tuple[Array | float, Array | float]
        object.__setattr__(
            self,
            "scaled_bounds",
            (self.scaler.transform(self.lower), self.scaler.transform(self.upper)),
        )

    @property
    def bounds(self) -> tuple[float, float]:
        """Return the (lower, upper) bounds."""
        return (self.lower, self.upper)

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array, NNModel],
        current_lnpdf: Array | None = None,
        /,
        *,
        xp: ArrayNamespace[Array],
    ) -> Array:
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
        lnp = xp.zeros_like(mpars[(WEIGHT_NAME,)])
        where = within_bounds(data[self.coord_name], self.lower, self.upper) & (
            mpars[(WEIGHT_NAME,)] > self.threshold
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
        i = model.param_names.flats.index((WEIGHT_NAME,))
        where = within_bounds(data[self.coord_name].flatten(), *self.scaled_bounds) & (
            pred[:, i] < self.threshold
        )
        return array_at(pred, (where, i), inplace=False).set(self.set_to)
