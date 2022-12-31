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
class Uniform(BackgroundModel):
    """Uniform background model."""

    n_features: int = 0
    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(("weight",))
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
        self._ln_diffs = xp.asarray(
            [xp.log(xp.asarray([b - a])) for a, b in self.coord_bounds.values()]
        )[None, :]

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self,
        pars: Params[Array],
        data: DataT,
        *,
        mask: Array | None = None,
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data (phi1).

        mask : (N, 1) Array[bool], keyword-only
            Data availability. True if data is available, False if not.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        f = pars[("weight",)]
        eps = xp.finfo(f.dtype).eps  # TOOD: or tiny?

        indicator = xp.ones_like(self._ln_diffs) if mask is None else mask.int()
        return xp.log(xp.clip(f, eps)) - (indicator * self._ln_diffs).sum(
            dim=1, keepdim=True
        )

    def ln_prior_arr(self, pars: Params[Array], data: DataT) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(pars[("weight",)])
        for bounds in self.param_bounds.flatvalues():
            lnp += bounds.logpdf(pars, data, self, lnp)
        return lnp

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
