"""Built-in background models."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, ClassVar

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.pytorch.background.base import BackgroundModel

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array, DataT, ParsT

__all__: list[str] = []


class UniformBackgroundModel(BackgroundModel):
    """Stream Model."""

    _param_names: ClassVar[dict[str, int]] = {"fraction": 1}

    def __init__(self, bkg_min: Array, bkg_max: Array) -> None:
        super().__init__()

        self.bkg_min = bkg_min
        self.bkg_max = bkg_max

        self._logdiff = xp.log(self.bkg_max - self.bkg_min)

    @property
    def param_names(self) -> dict[str, int]:
        """Parameter names."""
        return self._param_names

    # ========================================================================
    # Statistics

    def ln_likelihood(self, pars: ParsT, data: DataT) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : DataT
            Data (phi1).

        Returns
        -------
        Array
        """
        # Need to protect the fraction if < 0
        return xp.log(xp.clamp(pars["fraction"], min=0)) - self._logdiff

    def ln_prior(self, pars: ParsT) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Array
        """
        return xp.zeros_like(pars["fraction"])

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
