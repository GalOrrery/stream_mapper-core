"""Built-in background models."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.params import ParamBoundsField, ParamNamesField, Params
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.background.base import BackgroundModel
from stream_ml.pytorch.prior.bounds import SigmoidBounds

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import DataT

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class UniformBackgroundModel(BackgroundModel):
    """Uniform background model."""

    n_features: int = 0
    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(("mixparam",))
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {"mixparam": SigmoidBounds(0.0, 1.0, param_name=("mixparam",))}
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the n_features
        if self.n_features != 0:
            raise ValueError("n_features must be 0 for the uniform background.")

        # Validate the param_names
        if self.param_names != ("mixparam",):
            raise ValueError(
                "param_names must be ('mixparam',) for the uniform background."
            )

        # Pre-compute the log-difference
        self._logdiff = xp.asarray(
            [xp.log(xp.asarray([b - a])) for a, b in self.coord_bounds.values()]
        ).sum()

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT, *args: Array
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data (phi1).
        *args : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # Need to protect the fraction if < 0
        return xp.log(xp.clip(pars[("mixparam",)], 0)) - self._logdiff

    def ln_prior_arr(self, pars: Params[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params
            Parameters.

        Returns
        -------
        Array
        """
        lp = xp.zeros_like(pars[("mixparam",)])

        for bounds in self.param_bounds.flatvalues():
            lp += bounds.logpdf(pars, lp)

        return lp

    # ========================================================================
    # ML

    def forward(self, *args: Array) -> Array:
        """Forward pass.

        Parameters
        ----------
        args : Array
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        # TODO: how to handle the forward pass of the Prior? This model has no
        #  parameters.
        return xp.asarray([])
