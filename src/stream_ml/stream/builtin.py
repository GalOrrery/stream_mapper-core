"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
from torch import tensor

# LOCAL
from stream_ml.funcs import log_of_normal
from stream_ml.stream.base import StreamModel

if TYPE_CHECKING:
    # THIRD-PARTY
    from torch import Tensor

    # LOCAL
    from stream_ml._typing import DataT, ParsT

__all__: list[str] = []


class SingleGaussianStreamModel(StreamModel):
    """Stream Model."""

    def ln_likelihood(self, pars: ParsT, data: DataT) -> Tensor:
        """Log-likelihood of the stream.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : Tensor
            Data (phi1).
        """
        return xp.log(pars["fraction"]) + log_of_normal(data["phi2"], pars["phi2_mu"], pars["phi2_sigma"])

    def ln_prior(self, pars: ParsT) -> Tensor:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        """
        return tensor(0.0)  # TODO: Implement this
