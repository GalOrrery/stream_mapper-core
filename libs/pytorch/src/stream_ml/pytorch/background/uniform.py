"""Built-in background models."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBoundsField, ParamNamesField, Params
from stream_ml.pytorch.background.base import BackgroundModel
from stream_ml.pytorch.prior.bounds import SigmoidBounds
from stream_ml.pytorch.typing import Array

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
    require_mask: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the n_features
        if self.n_features != 0:
            msg = "n_features must be 0 for the uniform background"
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
        data: Data[Array],
        *,
        mask: Array | None = None,
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : Data[Array]
            Data.

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

        if mask is None:
            if self.require_mask:
                msg = "mask is required"
                raise ValueError(msg)
            indicator = xp.ones_like(self._ln_diffs)
        else:
            indicator = mask.int()

        return xp.log(xp.clip(f, eps)) - (indicator * self._ln_diffs).sum(
            dim=1, keepdim=True
        )

    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
        """
        lnp = xp.zeros((len(data), 1))
        # Bounds
        lnp += self._ln_prior_coord_bnds(pars, data)
        for bounds in self.param_bounds.flatvalues():
            lnp += bounds.logpdf(pars, data, self, lnp)
        return lnp

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
