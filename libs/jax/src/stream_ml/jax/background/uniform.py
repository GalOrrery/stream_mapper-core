"""Built-in background models."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import jax.numpy as xp

# LOCAL
from stream_ml.core.utils.params import ParamBoundsField, ParamNamesField, Params
from stream_ml.jax.background.base import BackgroundModel

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.jax._typing import Array, DataT

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class UniformBackgroundModel(BackgroundModel):
    """Uniform background model."""

    n_features: int = 0
    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(("mixparam",))
    param_bounds: ParamBoundsField = ParamBoundsField({"mixparam": (0.0, 1.0)})

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
            [xp.log(xp.asarray(b - a)) for a, b in self.coord_bounds.values()]
        ).sum()

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT, *args: Array
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : Params[Array]
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
        pars : Params[Array]
            Parameters.

        Returns
        -------
        Array
        """
        return xp.zeros_like(pars[("mixparam",)])

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
        return xp.asarray([])
