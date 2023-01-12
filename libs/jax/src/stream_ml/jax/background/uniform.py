"""Built-in background models."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass
from typing import Any

# THIRD-PARTY
import jax.numpy as xp

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBoundsField, ParamNames, ParamNamesField, Params
from stream_ml.jax.background.base import BackgroundModel
from stream_ml.jax.prior.bounds import SigmoidBounds
from stream_ml.jax.typing import Array

__all__: list[str] = []

_eps = float(xp.finfo(xp.float32).eps)


@dataclass()
class Uniform(BackgroundModel):
    """Uniform background model.

    Raises
    ------
    ValueError
        If there are not 0 features.
    """

    n_features: int = 0
    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(ParamNames(("weight",)))
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {"weight": SigmoidBounds(_eps, 1.0, param_name=("weight",))}
    )

    def setup(self) -> None:
        """JSetup the module's NN.

        Raises
        ------
        ValueError
            If there are not 0 features.
        """
        # Validate the n_features
        if self.n_features != 0:
            msg = "n_features must be 0 for the uniform background"
            raise ValueError(msg)

        # Pre-compute the log-difference
        self._logdiffs = xp.asarray(
            [xp.log(xp.asarray(b - a)) for a, b in self.coord_bounds.values()]
        )

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # Need to protect the fraction if < 0
        eps = xp.finfo(mpars[("weight",)].dtype).eps  # TOOD: or tiny?
        return xp.log(xp.clip(mpars[("weight",)], eps)) - self._logdiffs.sum()

    def ln_prior_arr(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data: Data[Array]
            Data.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(mpars[("weight",)])
        return lnp + self._ln_prior_coord_bnds(mpars, data)

    # ========================================================================
    # ML

    def __call__(self, *args: Array, **kwargs: Any) -> Array:
        """Forward pass.

        Parameters
        ----------
        *args : Array
            Input.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        nn = xp.asarray([])

        # Call the prior to limit the range of the parameters
        for prior in self.priors:
            nn = prior(nn, args[0], self)

        return nn
