"""Built-in background models."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.background.base import BackgroundModel

if TYPE_CHECKING:
    # THIRD-PARTY
    from torch import Tensor

    # LOCAL
    from stream_ml._typing import DataT, ParsT

__all__: list[str] = []


class UniformBackgroundModel(BackgroundModel):
    """Stream Model."""

    def __init__(self, bkg_min: Tensor, bkg_max: Tensor) -> None:
        super().__init__()

        self.bkg_min = bkg_min
        self.bkg_max = bkg_max

        self._logdiff = xp.log(self.bkg_max - self.bkg_min)

    def ln_likelihood(self, pars: ParsT, data: DataT) -> Tensor:
        """Log-likelihood of the background.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : Tensor
            Data (phi1).

        Returns
        -------
        Tensor
        """
        return xp.log(1 - pars["fraction"]) - self._logdiff

    def ln_prior(self, pars: ParsT) -> Tensor:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Tensor
        """
        return xp.tensor(0.0)  # TODO: Implement this
