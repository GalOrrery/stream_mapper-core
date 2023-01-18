"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.prior.base import PriorBase
from stream_ml.pytorch.typing import Array

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.api import Model
    from stream_ml.core.params.core import Params

__all__: list[str] = []


@dataclass(frozen=True)
class ControlPoints(PriorBase[Array]):
    """Control points prior.

    Parameters
    ----------
    control_points : Data[Array]
        The control points.
    lamda : float, optional
        Importance hyperparameter.
    """

    control_points: Data[Array]
    lamda: float = 0.05  # TODO? as a trainable Parameter.
    _: KW_ONLY
    coord_name: str = "phi1"
    component_param_name: str = "mu"

    def __post_init__(self) -> None:
        """Post-init."""
        # Pre-store the control points, seprated by indep & dep parameters.
        self._cpts_indep: Data[Array]
        object.__setattr__(self, "_cpts_indep", self.control_points[(self.coord_name,)])

        dep_names = tuple(n for n in self.control_points.names if n != self.coord_name)
        self._control_point_deps: Data[Array]
        object.__setattr__(self, "_control_point_deps", self.control_points[dep_names])

        super().__post_init__()

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array],
        current_lnpdf: Array | None = None,
        /,
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

        Returns
        -------
        Array
            The logpdf.
        """
        # Get the model parameters evaluated at the control points. shape (C, 1).
        cmpars = model.unpack_params_from_arr(model(self._cpts_indep))
        cmp_arr = xp.hstack(  # (C, F)
            tuple(
                cmpars[(n, self.component_param_name)]
                for n in self._control_point_deps.names
            )
        )

        # For each control point, add the squared distance to the logpdf.
        return -self.lamda * (  # (C, F) -> 1
            (cmp_arr - self._control_point_deps.array) ** 2
        ).sum(dim=0, keepdim=True)

    def __call__(self, pred: Array, data: Data[Array], model: Model[Array], /) -> Array:
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
        return pred
