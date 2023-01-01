"""Built-in background models."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass

# THIRD-PARTY
import jax.numpy as xp

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBoundsField, ParamNames, ParamNamesField, Params
from stream_ml.jax._typing import Array
from stream_ml.jax.background.base import BackgroundModel
from stream_ml.jax.prior.bounds import SigmoidBounds

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class Uniform(BackgroundModel):
    """Uniform background model."""

    n_features: int = 0
    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(ParamNames(("weight",)))
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {"weight": SigmoidBounds(0.0, 1.0, param_name=("weight",))}
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the n_features
        if self.n_features != 0:
            msg = "n_features must be 0 for the uniform background"
            raise ValueError(msg)

        # Validate the param_names
        if self.param_names != ("weight",):
            msg = "param_names must be ('weight',) for the uniform background"
            raise ValueError(msg)

        # Pre-compute the log-difference
        self._logdiffs = xp.asarray(
            [xp.log(xp.asarray(b - a)) for a, b in self.coord_bounds.values()]
        )

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data (phi1).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # Need to protect the fraction if < 0
        eps = xp.finfo(pars[("weight",)].dtype).eps  # TOOD: or tiny?
        return xp.log(xp.clip(pars[("weight",)], eps)) - self._logdiffs.sum()

    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data: Data[Array]
            Data (phi1).

        Returns
        -------
        Array
        """
        return xp.zeros_like(pars[("weight",)])

    # ========================================================================
    # ML

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        nn = xp.asarray([])

        # Call the prior to limit the range of the parameters
        # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            nn = prior(nn, data, self)

        return nn
