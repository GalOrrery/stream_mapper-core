"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import KW_ONLY, InitVar, dataclass
from math import inf
from typing import TYPE_CHECKING

from stream_ml.core.prior._base import Prior
from stream_ml.core.typing import Array
from stream_ml.core.utils import array_at, within_bounds
from stream_ml.core.utils.scale import DataScaler  # noqa: TCH001

if TYPE_CHECKING:
    from stream_ml.core import Data, ModelAPI as Model, ModelsBase, Params
    from stream_ml.core.typing import NNModel


@dataclass(frozen=True, repr=False)
class HardThreshold(Prior[Array]):
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

    threshold: float = -2.5
    set_to: float = -10
    _: KW_ONLY
    param_name: str
    coord_name: str = "phi1"
    lower: float = -inf
    upper: float = inf

    data_scaler: InitVar[DataScaler[Array]]

    def __post_init__(self, data_scaler: DataScaler[Array]) -> None:
        """Post-init."""
        super().__post_init__()
        if self.lower > self.upper:
            msg = f"lower > upper: {self.lower} > {self.upper}"
            raise ValueError(msg)

        self.scaled_bounds: tuple[Array | float, Array | float]
        object.__setattr__(
            self,
            "scaled_bounds",
            (
                data_scaler.transform(self.lower, names=(self.coord_name,), xp=self.xp),
                data_scaler.transform(self.upper, names=(self.coord_name,), xp=self.xp),
            ),
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

        Returns
        -------
        Array
            The logpdf.
        """
        lnp = self.xp.zeros_like(mpars[(self.param_name,)])
        where = within_bounds(data[self.coord_name], self.lower, self.upper) & (
            mpars[(self.param_name,)] > self.threshold
        )
        return array_at(lnp, where).set(-self.xp.inf)

    def __call__(
        self, pred: Array, data: Data[Array], model: ModelsBase[Array, NNModel]  # type: ignore[override]
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
        i = model.composite_params.flatskeys().index((self.param_name,))
        where = within_bounds(data[self.coord_name], *self.scaled_bounds) & (
            pred[:, i] < self.threshold
        )
        return array_at(pred, (where, i), inplace=False).set(self.set_to)
